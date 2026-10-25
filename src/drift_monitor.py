"""
Module E — Sliding-Window Concept Drift Monitor (`DriftMonitor`) & Retraining Workflow.

Implements `DriftMonitor` (matching the BCSE497J report Class Diagram):
- Maintains reference probability histograms derived from the baseline training split.
- Evaluates incoming flow windows using Kullback-Leibler (KL) divergence
  D_KL(P_ref || Q_live) with Laplace epsilon smoothing (eps = 1e-6) so zero-count bins
  never cause division-by-zero or infinite divergence.
- Configurable drift threshold (default tau = 0.15) and consecutive-window trigger count
  (default = 3 windows), as specified in report requirements FR-06 and FR-07.
- Explicitly separates covariate feature-distribution drift from verified accuracy
  degradation: classification metrics are reported ONLY when ground-truth labels
  accompany the window.
- Provides a safe retraining workflow on labelled baseline + drifted data and records
  every retraining event to `logs/retraining_history.json`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import rel_entr

from config.settings import (
    DEFAULT_DRIFT_CONSECUTIVE_WINDOWS,
    DEFAULT_DRIFT_TAU,
    DEFAULT_DRIFT_WINDOW_SIZE,
    DRIFT_HISTORY_PATH,
    DRIFT_REFERENCE_PATH,
    RETRAIN_LOG_PATH,
)


@dataclass
class DriftWindowResult:
    """Diagnostic result for a single monitored sliding window."""

    window_index: int
    timestamp_utc: str
    window_label: str
    sample_count: int
    overall_kl_divergence: float
    max_feature_kl: float
    max_drifted_feature: str
    tau_threshold: float
    window_exceeds_tau: bool
    consecutive_exceeded_count: int
    consecutive_trigger_limit: int
    retraining_triggered: bool
    per_feature_kl: Dict[str, float] = field(default_factory=dict)
    labels_available: bool = False
    measured_window_accuracy: Optional[float] = None
    measured_window_macro_f1: Optional[float] = None
    diagnostic_note: str = ""


def compute_smoothed_kl_divergence(
    ref_counts: np.ndarray,
    live_counts: np.ndarray,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute KL divergence D_KL(P_live || P_ref) with numerical smoothing for zero probabilities.
    Uses `scipy.special.rel_entr` over epsilon-smoothed normalized probability distributions.
    """
    p = np.asarray(live_counts, dtype=np.float64) + epsilon
    q = np.asarray(ref_counts, dtype=np.float64) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    kl = float(np.sum(rel_entr(p, q)))
    return max(0.0, kl)


class DriftMonitor:
    """
    Sliding-Window KL-Divergence Concept Drift Monitor.
    Matches `DriftMonitor` in the BCSE497J report Class Diagram.
    """

    def __init__(
        self,
        tau: float = DEFAULT_DRIFT_TAU,
        window_size: int = DEFAULT_DRIFT_WINDOW_SIZE,
        consecutive_trigger: int = DEFAULT_DRIFT_CONSECUTIVE_WINDOWS,
        n_bins: int = 12,
        epsilon: float = 1e-6,
    ) -> None:
        self.tau = float(tau)
        self.window_size = max(10, int(window_size))
        self.consecutive_trigger = max(1, int(consecutive_trigger))
        self.n_bins = max(4, int(n_bins))
        self.epsilon = float(epsilon)
        self.consecutive_over: int = 0
        self.feature_bins: Dict[str, List[float]] = {}
        self.reference_histograms: Dict[str, List[float]] = {}
        self.reference_means: Dict[str, float] = {}
        self.reference_stds: Dict[str, float] = {}
        self.window_history: List[DriftWindowResult] = []
        self.is_initialized: bool = False

    @classmethod
    def from_reference_dataframe(
        cls,
        X_ref: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        tau: float = DEFAULT_DRIFT_TAU,
        window_size: int = DEFAULT_DRIFT_WINDOW_SIZE,
        consecutive_trigger: int = DEFAULT_DRIFT_CONSECUTIVE_WINDOWS,
        n_bins: int = 12,
    ) -> "DriftMonitor":
        """Build reference histograms from the baseline training feature matrix."""
        monitor = cls(
            tau=tau,
            window_size=window_size,
            consecutive_trigger=consecutive_trigger,
            n_bins=n_bins,
        )
        cols = feature_names if feature_names is not None else list(X_ref.columns)
        for col in cols:
            vals = pd.to_numeric(X_ref[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                edges = np.linspace(0.0, 1.0, n_bins + 1)
                counts = np.ones(n_bins, dtype=np.float64) / n_bins
                mean_v, std_v = 0.0, 1.0
            else:
                vmin, vmax = float(np.min(vals)), float(np.max(vals))
                if np.isclose(vmin, vmax):
                    vmin -= 0.5
                    vmax += 0.5
                else:
                    pad = 0.05 * (vmax - vmin)
                    vmin -= pad
                    vmax += pad
                edges = np.linspace(vmin, vmax, n_bins + 1)
                raw_counts, _ = np.histogram(vals, bins=edges)
                counts = (raw_counts.astype(np.float64) + monitor.epsilon)
                counts /= counts.sum()
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals)) if np.std(vals) > 0 else 1.0

            monitor.feature_bins[col] = [float(e) for e in edges]
            monitor.reference_histograms[col] = [float(c) for c in counts]
            monitor.reference_means[col] = mean_v
            monitor.reference_stds[col] = std_v

        monitor.is_initialized = True
        return monitor

    def compute_kl_for_window(self, X_window: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """
        Compute per-feature and overall mean KL divergence between `X_window` and the
        reference baseline distribution.
        Includes out-of-range tail bins so shifts beyond the training min/max are accurately captured.
        """
        if not self.is_initialized:
            raise RuntimeError("DriftMonitor reference distribution is not initialized.")

        per_feature_kl: Dict[str, float] = {}
        for col, edges_list in self.feature_bins.items():
            ref_probs = np.asarray(self.reference_histograms[col], dtype=np.float64)
            if col not in X_window.columns:
                per_feature_kl[col] = 0.0
                continue

            vals = pd.to_numeric(X_window[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                per_feature_kl[col] = 0.0
                continue

            edges = np.asarray(edges_list, dtype=np.float64)
            # Clip values to [edges[0], edges[-1]] so out-of-range shifted values accumulate in boundary bins
            clipped_vals = np.clip(vals, edges[0], edges[-1])
            live_counts, _ = np.histogram(clipped_vals, bins=edges)

            kl_val = compute_smoothed_kl_divergence(ref_probs, live_counts, epsilon=self.epsilon)
            per_feature_kl[col] = round(kl_val, 5)

        overall_kl = float(np.mean(list(per_feature_kl.values()))) if per_feature_kl else 0.0
        return round(overall_kl, 5), per_feature_kl

    def evaluate_window(
        self,
        X_window: pd.DataFrame,
        window_label: str = "Incoming Traffic Window",
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        persist_history: bool = True,
    ) -> DriftWindowResult:
        """
        Evaluate a single traffic window, update consecutive drift counters, and check trigger condition.
        """
        from sklearn.metrics import accuracy_score, f1_score

        overall_kl, per_feat_kl = self.compute_kl_for_window(X_window)
        exceeds = overall_kl > self.tau
        if exceeds:
            self.consecutive_over += 1
        else:
            self.consecutive_over = 0

        triggered = self.consecutive_over >= self.consecutive_trigger

        max_feat = max(per_feat_kl, key=per_feat_kl.get) if per_feat_kl else "N/A"
        max_kl = per_feat_kl.get(max_feat, 0.0)

        labels_avail = y_true is not None and y_pred is not None and len(y_true) == len(y_pred) and len(y_true) > 0
        win_acc: Optional[float] = None
        win_f1: Optional[float] = None
        if labels_avail:
            win_acc = round(float(accuracy_score(y_true, y_pred)), 4)
            win_f1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
            note = (
                f"Feature KL={overall_kl:.4f} (threshold tau={self.tau}). "
                f"Verified window accuracy={win_acc*100:.2f}%, Macro-F1={win_f1:.4f} (ground-truth labels provided)."
            )
        else:
            note = (
                f"Feature-distribution KL={overall_kl:.4f} (threshold tau={self.tau}). "
                "Ground-truth labels not present in window; reporting covariate drift only without claiming accuracy decay."
            )

        res = DriftWindowResult(
            window_index=len(self.window_history) + 1,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            window_label=window_label,
            sample_count=int(len(X_window)),
            overall_kl_divergence=overall_kl,
            max_feature_kl=max_kl,
            max_drifted_feature=max_feat,
            tau_threshold=self.tau,
            window_exceeds_tau=exceeds,
            consecutive_exceeded_count=self.consecutive_over,
            consecutive_trigger_limit=self.consecutive_trigger,
            retraining_triggered=triggered,
            per_feature_kl=per_feat_kl,
            labels_available=labels_avail,
            measured_window_accuracy=win_acc,
            measured_window_macro_f1=win_f1,
            diagnostic_note=note,
        )
        self.window_history.append(res)
        if persist_history:
            self.save_history(DRIFT_HISTORY_PATH)
        return res

    def save_reference(self, path: Path = DRIFT_REFERENCE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tau": self.tau,
            "window_size": self.window_size,
            "consecutive_trigger": self.consecutive_trigger,
            "n_bins": self.n_bins,
            "epsilon": self.epsilon,
            "feature_bins": self.feature_bins,
            "reference_histograms": self.reference_histograms,
            "reference_means": self.reference_means,
            "reference_stds": self.reference_stds,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_reference(cls, path: Path = DRIFT_REFERENCE_PATH) -> "DriftMonitor":
        if not path.exists():
            raise FileNotFoundError(f"Drift reference file not found at {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        monitor = cls(
            tau=payload.get("tau", DEFAULT_DRIFT_TAU),
            window_size=payload.get("window_size", DEFAULT_DRIFT_WINDOW_SIZE),
            consecutive_trigger=payload.get("consecutive_trigger", DEFAULT_DRIFT_CONSECUTIVE_WINDOWS),
            n_bins=payload.get("n_bins", 12),
            epsilon=payload.get("epsilon", 1e-6),
        )
        monitor.feature_bins = payload["feature_bins"]
        monitor.reference_histograms = payload["reference_histograms"]
        monitor.reference_means = payload.get("reference_means", {})
        monitor.reference_stds = payload.get("reference_stds", {})
        monitor.is_initialized = True
        if DRIFT_HISTORY_PATH.exists():
            try:
                hist_raw = json.loads(DRIFT_HISTORY_PATH.read_text(encoding="utf-8"))
                monitor.window_history = [DriftWindowResult(**w) for w in hist_raw]
                if monitor.window_history:
                    monitor.consecutive_over = monitor.window_history[-1].consecutive_exceeded_count
            except Exception:
                pass
        return monitor

    def save_history(self, path: Path = DRIFT_HISTORY_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(w) for w in self.window_history], indent=2),
            encoding="utf-8",
        )

    def reset_history(self, path: Path = DRIFT_HISTORY_PATH) -> None:
        self.window_history = []
        self.consecutive_over = 0
        if path.exists():
            path.unlink()


def execute_drift_retraining(
    baseline_df: pd.DataFrame,
    drifted_labelled_df: pd.DataFrame,
    target_col: str = "Attack_type",
    k_features: int = 14,
) -> Dict[str, Any]:
    """
    Execute a safe retraining workflow combining baseline data with newly labelled
    drifted samples, update the saved model and drift reference, and log the outcome.
    """
    from src.train_model import train_and_evaluate_pipeline

    t_start_iso = datetime.now(timezone.utc).isoformat()
    try:
        combined_df = pd.concat([baseline_df, drifted_labelled_df], ignore_index=True)
        _, eval_res = train_and_evaluate_pipeline(
            df=combined_df,
            target_col=target_col,
            dataset_name=f"Drift-Adapted Combined Dataset ({len(combined_df)} rows)",
            is_synthetic_dataset=True,
            k_features=k_features,
            compare_with_full_features=False,
        )
        record = {
            "timestamp_utc": t_start_iso,
            "status": "Success",
            "baseline_rows": int(len(baseline_df)),
            "drifted_rows_added": int(len(drifted_labelled_df)),
            "total_retrain_rows": int(len(combined_df)),
            "updated_accuracy": round(eval_res.accuracy, 4),
            "updated_macro_f1": round(eval_res.macro_f1, 4),
            "training_time_sec": eval_res.training_time_sec,
            "notes": "Model and baseline reference histograms updated with labelled drift window traffic.",
        }
    except Exception as exc:
        record = {
            "timestamp_utc": t_start_iso,
            "status": f"Failed: {exc}",
            "baseline_rows": int(len(baseline_df)),
            "drifted_rows_added": int(len(drifted_labelled_df)),
            "total_retrain_rows": 0,
            "updated_accuracy": None,
            "updated_macro_f1": None,
            "training_time_sec": 0.0,
            "notes": str(exc),
        }

    RETRAIN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    if RETRAIN_LOG_PATH.exists():
        try:
            history = json.loads(RETRAIN_LOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            history = []
    history.append(record)
    RETRAIN_LOG_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return record
