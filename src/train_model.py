"""
Module B (Part 1) — Reproducible LightGBM Intrusion Detection Model Training & Evaluation.

Features:
- Supports binary and multiclass IoT traffic classification.
- Trains on the APAFC-selected feature subset and optionally trains a full-feature
  baseline model to empirically measure accuracy/Macro-F1 retention vs. model size & latency.
- Records actual measured training duration, per-record inference latency (mean, p50, p95, p99),
  model artifact size on disk, confusion matrix, and per-class precision/recall/F1.
- Saves a complete, self-contained artifact bundle (`models/lgbm_ids_bundle.joblib`)
  containing the trained LightGBM model, `DatasetPreprocessor`, `APAFCExtractor`,
  label mappings, and evaluation metadata.
- Also generates and saves the baseline reference distribution for the Concept Drift Monitor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config.settings import (
    BASELINE_FULL_MODEL_PATH,
    DEFAULT_APAFC_K,
    DEFAULT_RANDOM_SEED,
    DRIFT_REFERENCE_PATH,
    EVALUATION_RESULTS_PATH,
    FEATURE_CONFIG_PATH,
    MODEL_ARTIFACT_PATH,
)
from src.feature_selection import APAFCExtractor
from src.preprocessing import DatasetPreprocessor, SplitDataBundle, prepare_train_test_data


@dataclass
class ModelEvaluationResult:
    """Actual measured evaluation metrics from a training run (never hardcoded)."""

    timestamp_utc: str
    dataset_name: str
    is_synthetic_dataset: bool
    num_train_samples: int
    num_test_samples: int
    num_classes: int
    total_candidate_features: int
    selected_feature_count: int
    selected_features: List[str]
    resampling_applied: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    training_time_sec: float
    inference_latency_mean_ms: float
    inference_latency_p50_ms: float
    inference_latency_p95_ms: float
    inference_latency_p99_ms: float
    artifact_size_kb: float
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    class_labels: List[str] = field(default_factory=list)
    baseline_comparison: Optional[Dict[str, Any]] = None


def _measure_inference_latency(
    model: lgb.LGBMClassifier,
    X_sample: pd.DataFrame,
    n_Single_trials: int = 120,
) -> Dict[str, float]:
    """Measure actual single-record inference latency in milliseconds without feature-name warnings."""
    n_rows = min(len(X_sample), max(10, n_Single_trials))
    latencies_ms: List[float] = []

    # Warm-up call
    _ = model.predict_proba(X_sample.iloc[:1])

    for i in range(n_rows):
        single_row = X_sample.iloc[i : i + 1]
        t0 = time.perf_counter()
        _ = model.predict_proba(single_row)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(lat_arr)),
        "p50_ms": float(np.percentile(lat_arr, 50)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "p99_ms": float(np.percentile(lat_arr, 99)),
    }


def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_mapping: Dict[int, str],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], List[List[int]], List[str]]:
    """Compute overall and per-class classification metrics."""
    present_ids = sorted(label_mapping.keys())
    class_names = [label_mapping[i] for i in present_ids]

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        labels=present_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class: Dict[str, Dict[str, float]] = {}
    for cname in class_names:
        if cname in report:
            per_class[cname] = {
                "precision": float(report[cname]["precision"]),
                "recall": float(report[cname]["recall"]),
                "f1_score": float(report[cname]["f1-score"]),
                "support": int(report[cname]["support"]),
            }

    cm = confusion_matrix(y_true, y_pred, labels=present_ids).tolist()
    overall = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
    }
    return overall, per_class, cm, class_names


def train_and_evaluate_pipeline(
    df: pd.DataFrame,
    target_col: str = "Attack_type",
    dataset_name: str = "Synthetic Edge-IIoTset Demo",
    is_synthetic_dataset: bool = True,
    k_features: int = DEFAULT_APAFC_K,
    test_size: float = 0.20,
    resampling_method: str = "smote_enn",
    n_estimators: int = 120,
    learning_rate: float = 0.08,
    num_leaves: int = 31,
    max_depth: int = -1,
    compare_with_full_features: bool = True,
    random_state: int = DEFAULT_RANDOM_SEED,
    artifact_path: Path = MODEL_ARTIFACT_PATH,
) -> Tuple[Dict[str, Any], ModelEvaluationResult]:
    """
    Execute the complete leak-free training pipeline:
    1. Split & preprocess data (`prepare_train_test_data`).
    2. Fit APAFC feature selector on `X_train` to select top `k_features`.
    3. Train LightGBM classifier on the reduced `k_features` training matrix.
    4. Optionally train a full-feature baseline model for empirical comparison.
    5. Measure test accuracy, Macro-F1, latency percentiles, and saved artifact size.
    6. Save model bundle, APAFC config, evaluation summary, and drift reference stats.
    """
    split_bundle, preprocessor = prepare_train_test_data(
        df=df,
        target_col=target_col,
        test_size=test_size,
        resampling_method=resampling_method,
        random_state=random_state,
    )

    # Fit APAFC strictly on training partition
    apafc = APAFCExtractor(
        k_features=k_features,
        protocol_diversity_weight=0.15,
        random_state=random_state,
    )
    apafc.fit(split_bundle.X_train, split_bundle.y_train)
    apafc.save_config(FEATURE_CONFIG_PATH)

    X_train_reduced = apafc.extract(split_bundle.X_train)
    X_test_reduced = apafc.extract(split_bundle.X_test)

    num_classes = len(split_bundle.label_mapping)
    objective = "binary" if num_classes == 2 else "multiclass"
    extra_params = {"num_class": num_classes} if num_classes > 2 else {}

    lgbm_params = {
        "objective": objective,
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "class_weight": "balanced",
        "random_state": int(random_state),
        "n_jobs": -1,
        "importance_type": "gain",
        "verbose": -1,
        **extra_params,
    }

    model = lgb.LGBMClassifier(**lgbm_params)
    t_start = time.perf_counter()
    model.fit(X_train_reduced, split_bundle.y_train)
    train_time_sec = time.perf_counter() - t_start

    y_pred = model.predict(X_test_reduced)
    overall_metrics, per_class_metrics, cm, class_names = _evaluate_predictions(
        split_bundle.y_test,
        y_pred,
        split_bundle.label_mapping,
    )
    latency_stats = _measure_inference_latency(model, X_test_reduced)

    # Optional comparison against full-feature baseline model
    baseline_comparison: Optional[Dict[str, Any]] = None
    if compare_with_full_features and len(split_bundle.feature_names) > len(apafc.selected_subset):
        full_model = lgb.LGBMClassifier(**lgbm_params)
        t_full_start = time.perf_counter()
        full_model.fit(split_bundle.X_train, split_bundle.y_train)
        full_train_time = time.perf_counter() - t_full_start
        y_pred_full = full_model.predict(split_bundle.X_test)
        full_overall, _, _, _ = _evaluate_predictions(
            split_bundle.y_test,
            y_pred_full,
            split_bundle.label_mapping,
        )
        full_latency = _measure_inference_latency(full_model, split_bundle.X_test)
        joblib.dump(full_model, BASELINE_FULL_MODEL_PATH)
        full_size_kb = BASELINE_FULL_MODEL_PATH.stat().st_size / 1024.0

        baseline_comparison = {
            "full_feature_count": len(split_bundle.feature_names),
            "reduced_feature_count": len(apafc.selected_subset),
            "feature_reduction_pct": round(
                (1.0 - len(apafc.selected_subset) / max(len(split_bundle.feature_names), 1)) * 100.0, 2
            ),
            "full_accuracy": full_overall["accuracy"],
            "reduced_accuracy": overall_metrics["accuracy"],
            "full_macro_f1": full_overall["macro_f1"],
            "reduced_macro_f1": overall_metrics["macro_f1"],
            "macro_f1_retention_pct": round(
                (overall_metrics["macro_f1"] / max(full_overall["macro_f1"], 1e-9)) * 100.0, 2
            ),
            "full_training_time_sec": round(full_train_time, 4),
            "reduced_training_time_sec": round(train_time_sec, 4),
            "full_p95_latency_ms": round(full_latency["p95_ms"], 4),
            "reduced_p95_latency_ms": round(latency_stats["p95_ms"], 4),
            "full_model_size_kb": round(full_size_kb, 2),
        }

    timestamp_utc = datetime.now(timezone.utc).isoformat()

    # Build and save artifact bundle first so we can measure exact artifact size on disk
    bundle_dict: Dict[str, Any] = {
        "model": model,
        "preprocessor": preprocessor,
        "apafc": apafc,
        "selected_features": apafc.selected_subset,
        "all_candidate_features": split_bundle.feature_names,
        "label_mapping": split_bundle.label_mapping,
        "inverse_label_mapping": split_bundle.inverse_label_mapping,
        "target_col": target_col,
        "dataset_name": dataset_name,
        "is_synthetic_dataset": is_synthetic_dataset,
        "trained_at_utc": timestamp_utc,
        "lgbm_params": lgbm_params,
        "preprocessing_notes": split_bundle.preprocessing_notes,
        "train_class_counts_before": split_bundle.train_class_counts_before,
        "train_class_counts_after": split_bundle.train_class_counts_after,
        "test_class_counts": split_bundle.test_class_counts,
        # Store small background sample for fast TreeSHAP & drift reference initialization
        "background_sample": X_train_reduced.iloc[: min(200, len(X_train_reduced))].copy(),
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle_dict, artifact_path)
    artifact_size_kb = artifact_path.stat().st_size / 1024.0

    if baseline_comparison is not None:
        baseline_comparison["reduced_bundle_size_kb"] = round(artifact_size_kb, 2)

    eval_result = ModelEvaluationResult(
        timestamp_utc=timestamp_utc,
        dataset_name=dataset_name,
        is_synthetic_dataset=is_synthetic_dataset,
        num_train_samples=int(len(split_bundle.X_train)),
        num_test_samples=int(len(split_bundle.X_test)),
        num_classes=num_classes,
        total_candidate_features=len(split_bundle.feature_names),
        selected_feature_count=len(apafc.selected_subset),
        selected_features=apafc.selected_subset,
        resampling_applied=split_bundle.resampling_method_applied,
        accuracy=overall_metrics["accuracy"],
        macro_f1=overall_metrics["macro_f1"],
        weighted_f1=overall_metrics["weighted_f1"],
        macro_precision=overall_metrics["macro_precision"],
        macro_recall=overall_metrics["macro_recall"],
        training_time_sec=round(train_time_sec, 4),
        inference_latency_mean_ms=round(latency_stats["mean_ms"], 4),
        inference_latency_p50_ms=round(latency_stats["p50_ms"], 4),
        inference_latency_p95_ms=round(latency_stats["p95_ms"], 4),
        inference_latency_p99_ms=round(latency_stats["p99_ms"], 4),
        artifact_size_kb=round(artifact_size_kb, 2),
        per_class_metrics=per_class_metrics,
        confusion_matrix=cm,
        class_labels=class_names,
        baseline_comparison=baseline_comparison,
    )

    bundle_dict["evaluation"] = asdict(eval_result)
    joblib.dump(bundle_dict, artifact_path)

    EVALUATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVALUATION_RESULTS_PATH.write_text(json.dumps(asdict(eval_result), indent=2), encoding="utf-8")

    # Initialize and save reference feature histograms for Module E (DriftMonitor)
    try:
        from src.drift_monitor import DriftMonitor

        monitor = DriftMonitor.from_reference_dataframe(
            X_ref=X_train_reduced,
            feature_names=apafc.selected_subset,
        )
        monitor.save_reference(DRIFT_REFERENCE_PATH)
    except Exception:
        pass

    return bundle_dict, eval_result
