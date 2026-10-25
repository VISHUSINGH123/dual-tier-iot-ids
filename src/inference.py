"""
Module B (Part 2) — EdgeClassifier Inference Interface.

Implements `EdgeClassifier` (matching the report's Class Diagram):
- Loads the saved LightGBM model and preprocessing/APAFC artifacts.
- Supports single-record (`dict` / `Series` / `FlowRecord`) and batch (`DataFrame`) predictions.
- Returns predicted label, confidence score, per-class probability distribution,
  suspicious/benign flag, and measured inference time in milliseconds.
- Gracefully handles missing required features and unknown extra columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from config.settings import MODEL_ARTIFACT_PATH
from src.data_loader import FlowRecord


@dataclass
class PredictionRecord:
    """Structured prediction output for a single flow record."""

    record_index: int
    timestamp_utc: str
    predicted_class: str
    predicted_class_id: int
    confidence: float
    is_suspicious: bool
    severity: str
    class_probabilities: Dict[str, float]
    inference_latency_ms: float
    reduced_features: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class BatchPredictionOutput:
    """Structured output for a batch of predicted flow records."""

    predictions: List[PredictionRecord]
    results_df: pd.DataFrame
    total_records: int
    benign_count: int
    suspicious_count: int
    class_counts: Dict[str, int]
    total_latency_ms: float
    mean_latency_per_record_ms: float
    warnings: List[str] = field(default_factory=list)


def _compute_severity(predicted_class: str, confidence: float) -> Tuple[bool, str]:
    """Determine whether a predicted label is benign or suspicious and assign alert severity."""
    norm_label = predicted_class.strip().lower()
    if norm_label in ("normal", "benign", "0"):
        return False, "Benign"

    high_crit_keywords = ("ransomware", "backdoor", "mitm", "sql_injection", "ddos")
    if any(k in norm_label for k in high_crit_keywords):
        return True, "Critical" if confidence >= 0.80 else "High"
    if confidence >= 0.85:
        return True, "High"
    if confidence >= 0.60:
        return True, "Medium"
    return True, "Low"


class EdgeClassifier:
    """
    Lightweight Edge Intrusion Classifier wrapping the trained LightGBM model,
    `DatasetPreprocessor`, and `APAFCExtractor`.
    """

    def __init__(self, artifact_path: Path = MODEL_ARTIFACT_PATH) -> None:
        self.artifact_path = Path(artifact_path)
        self.bundle: Optional[Dict[str, Any]] = None
        self.model = None
        self.preprocessor = None
        self.apafc = None
        self.selected_features: List[str] = []
        self.label_mapping: Dict[int, str] = {}
        self.is_loaded: bool = False

    def load(self, artifact_path: Optional[Path] = None) -> "EdgeClassifier":
        """Load and validate the saved model artifact bundle."""
        target_path = Path(artifact_path) if artifact_path is not None else self.artifact_path
        if not target_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at '{target_path}'. Please train the model first."
            )
        try:
            bundle = joblib.load(target_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to deserialize model artifact at '{target_path}': {exc}") from exc

        required_keys = {"model", "preprocessor", "apafc", "selected_features", "label_mapping"}
        if not isinstance(bundle, dict) or not required_keys.issubset(bundle.keys()):
            raise ValueError(
                f"Incompatible model artifact at '{target_path}'. Missing required keys: "
                f"{required_keys - set(bundle.keys() if isinstance(bundle, dict) else [])}"
            )

        self.bundle = bundle
        self.model = bundle["model"]
        self.preprocessor = bundle["preprocessor"]
        self.apafc = bundle["apafc"]
        self.selected_features = list(bundle["selected_features"])
        self.label_mapping = {int(k): str(v) for k, v in bundle["label_mapping"].items()}
        self.artifact_path = target_path
        self.is_loaded = True
        return self

    def preprocess_to_reduced(
        self,
        records: Union[pd.DataFrame, pd.Series, Dict[str, Any], FlowRecord],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Transform raw input record(s) into the exact APAFC-reduced DataFrame."""
        if not self.is_loaded:
            self.load()

        if isinstance(records, FlowRecord):
            df_in = pd.DataFrame([records.to_feature_vector()])
        elif isinstance(records, dict):
            df_in = pd.DataFrame([records])
        elif isinstance(records, pd.Series):
            df_in = pd.DataFrame([records.to_dict()])
        elif isinstance(records, pd.DataFrame):
            df_in = records.copy()
        else:
            raise TypeError(f"Unsupported input type for prediction: {type(records)}")

        if df_in.empty:
            raise ValueError("Cannot run prediction on an empty input (0 rows).")

        X_full, prep_warnings = self.preprocessor.transform_features(df_in, return_warnings=True)
        X_reduced = self.apafc.extract(X_full)
        return X_reduced, prep_warnings

    def predict_single(
        self,
        record: Union[Dict[str, Any], pd.Series, FlowRecord],
        record_index: int = 0,
    ) -> PredictionRecord:
        """Predict a single record and return its structured PredictionRecord."""
        batch_out = self.predict_batch(record)
        pred = batch_out.predictions[0]
        pred.record_index = record_index
        return pred

    def predict_batch(
        self,
        records: Union[pd.DataFrame, pd.Series, Dict[str, Any], FlowRecord],
    ) -> BatchPredictionOutput:
        """
        Run batch inference on one or more flow records.
        Measures actual inference time and builds both structured PredictionRecords
        and an enriched DataFrame ready for UI inspection or CSV export.
        """
        if not self.is_loaded:
            self.load()

        if isinstance(records, FlowRecord):
            raw_df = pd.DataFrame([records.to_feature_vector()])
        elif isinstance(records, dict):
            raw_df = pd.DataFrame([records])
        elif isinstance(records, pd.Series):
            raw_df = pd.DataFrame([records.to_dict()])
        elif isinstance(records, pd.DataFrame):
            raw_df = records.copy().reset_index(drop=True)
        else:
            raise TypeError(f"Unsupported input type: {type(records)}")

        X_reduced, warnings_list = self.preprocess_to_reduced(raw_df)

        t0 = time.perf_counter()
        proba_matrix = self.model.predict_proba(X_reduced)
        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        n_rows = len(X_reduced)
        per_row_latency_ms = total_latency_ms / max(n_rows, 1)

        pred_indices = np.argmax(proba_matrix, axis=1)
        confidences = np.max(proba_matrix, axis=1)

        now_iso = datetime.now(timezone.utc).isoformat()
        predictions: List[PredictionRecord] = []
        class_counts: Dict[str, int] = {}
        benign_count = 0
        suspicious_count = 0

        class_ids_sorted = sorted(self.label_mapping.keys())

        for idx in range(n_rows):
            cls_id = int(pred_indices[idx])
            cls_name = self.label_mapping.get(cls_id, f"Class_{cls_id}")
            conf = float(confidences[idx])
            is_susp, severity = _compute_severity(cls_name, conf)

            if is_susp:
                suspicious_count += 1
            else:
                benign_count += 1

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

            prob_dict = {
                self.label_mapping[cid]: round(float(proba_matrix[idx, cid]), 5)
                for cid in class_ids_sorted
                if cid < proba_matrix.shape[1]
            }
            feat_dict = {col: float(X_reduced.iloc[idx][col]) for col in self.selected_features}

            predictions.append(
                PredictionRecord(
                    record_index=idx,
                    timestamp_utc=now_iso,
                    predicted_class=cls_name,
                    predicted_class_id=cls_id,
                    confidence=round(conf, 5),
                    is_suspicious=is_susp,
                    severity=severity,
                    class_probabilities=prob_dict,
                    inference_latency_ms=round(per_row_latency_ms, 4),
                    reduced_features=feat_dict,
                    warnings=warnings_list if idx == 0 else [],
                )
            )

        results_df = raw_df.copy()
        results_df["Predicted_Class"] = [p.predicted_class for p in predictions]
        results_df["Confidence"] = [round(p.confidence, 4) for p in predictions]
        results_df["Status"] = ["Suspicious" if p.is_suspicious else "Benign" for p in predictions]
        results_df["Severity"] = [p.severity for p in predictions]
        results_df["Inference_Latency_ms"] = [p.inference_latency_ms for p in predictions]

        return BatchPredictionOutput(
            predictions=predictions,
            results_df=results_df,
            total_records=n_rows,
            benign_count=benign_count,
            suspicious_count=suspicious_count,
            class_counts=class_counts,
            total_latency_ms=round(total_latency_ms, 4),
            mean_latency_per_record_ms=round(per_row_latency_ms, 4),
            warnings=warnings_list,
        )
