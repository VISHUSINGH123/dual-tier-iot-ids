"""
Module D (Part 2) — Audit & Explainable Alert Logger (`AlertLogger`).

Implements `AlertLogger` (matching the report's Class Diagram):
- Logs predictions and their top-k TreeSHAP feature attributions to a persistent
  JSON-Lines audit store (`logs/alerts_log.jsonl`).
- Supports filtering by predicted class, severity, benign/suspicious status,
  minimum confidence, and timestamp window.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import ALERT_LOG_PATH
from src.explainability import LocalExplanation
from src.inference import PredictionRecord


@dataclass
class AlertLogEntry:
    """Single auditable security alert entry persisted in the alert log."""

    alert_id: str
    timestamp_utc: str
    src_ip: str
    dst_ip: str
    predicted_class: str
    confidence: float
    is_suspicious: bool
    severity: str
    inference_latency_ms: float
    explanation_method: str
    is_local_shap: bool
    top_contributing_features: List[Dict[str, Any]] = field(default_factory=list)


class AlertLogger:
    """
    Persistent alert logger and query engine.
    Matches `AlertLogger` in the BCSE497J report Class Diagram.
    """

    def __init__(self, log_path: Path = ALERT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_alert(
        self,
        prediction: PredictionRecord,
        explanation: Optional[LocalExplanation] = None,
        src_ip: str = "192.168.1.100",
        dst_ip: str = "10.0.0.1",
    ) -> AlertLogEntry:
        """Persist a single prediction and its explanation to the JSONL audit log."""
        top_feats = []
        exp_method = "Not computed"
        is_local = False
        if explanation is not None:
            exp_method = explanation.explanation_method
            is_local = explanation.is_local_shap
            for contrib in explanation.top_contributions[:5]:
                top_feats.append(
                    {
                        "feature": contrib.feature_name,
                        "protocol": contrib.protocol_family,
                        "value": contrib.feature_value,
                        "shap_value": contrib.shap_value,
                        "direction": contrib.direction,
                    }
                )

        alert_id = f"ALT-{prediction.timestamp_utc[-12:].replace(':', '').replace('.', '')}-{prediction.record_index:04d}"
        entry = AlertLogEntry(
            alert_id=alert_id,
            timestamp_utc=prediction.timestamp_utc,
            src_ip=str(src_ip),
            dst_ip=str(dst_ip),
            predicted_class=prediction.predicted_class,
            confidence=prediction.confidence,
            is_suspicious=prediction.is_suspicious,
            severity=prediction.severity,
            inference_latency_ms=prediction.inference_latency_ms,
            explanation_method=exp_method,
            is_local_shap=is_local,
            top_contributing_features=top_feats,
        )
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
        return entry

    def log_batch(
        self,
        predictions: List[PredictionRecord],
        explanations: Optional[Dict[int, LocalExplanation]] = None,
        raw_df: Optional[pd.DataFrame] = None,
        max_records_to_log: int = 250,
    ) -> List[AlertLogEntry]:
        """Log up to `max_records_to_log` records from a batch prediction run."""
        logged: List[AlertLogEntry] = []
        explanations = explanations or {}
        for pred in predictions[:max_records_to_log]:
            idx = pred.record_index
            src_ip = "192.168.1.100"
            dst_ip = "10.0.0.1"
            if raw_df is not None and idx < len(raw_df):
                row = raw_df.iloc[idx]
                src_ip = str(row.get("ip.src_host", row.get("src_ip", src_ip)))
                dst_ip = str(row.get("ip.dst_host", row.get("dst_ip", dst_ip)))
            entry = self.log_alert(
                prediction=pred,
                explanation=explanations.get(idx),
                src_ip=src_ip,
                dst_ip=dst_ip,
            )
            logged.append(entry)
        return logged

    def read_alerts(
        self,
        predicted_classes: Optional[List[str]] = None,
        severities: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        suspicious_only: bool = False,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Read and filter persisted alerts (most recent first)."""
        if not self.log_path.exists():
            return []

        records: List[Dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    records.append(item)
                except json.JSONDecodeError:
                    continue

        # Most recent first
        records.reverse()

        filtered: List[Dict[str, Any]] = []
        for r in records:
            if suspicious_only and not r.get("is_suspicious", False):
                continue
            if r.get("confidence", 0.0) < min_confidence:
                continue
            if predicted_classes and r.get("predicted_class") not in predicted_classes:
                continue
            if severities and r.get("severity") not in severities:
                continue
            filtered.append(r)
            if len(filtered) >= limit:
                break
        return filtered

    def to_dataframe(self, alerts: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
        """Convert alert entries into a clean tabular DataFrame for dashboard display & export."""
        if alerts is None:
            alerts = self.read_alerts(limit=500)
        if not alerts:
            return pd.DataFrame(
                columns=[
                    "Alert_ID",
                    "Timestamp_UTC",
                    "Src_IP",
                    "Dst_IP",
                    "Predicted_Class",
                    "Confidence",
                    "Status",
                    "Severity",
                    "Top_Feature_1",
                    "Top_Feature_2",
                    "Explanation_Method",
                ]
            )
        rows = []
        for a in alerts:
            top_feats = a.get("top_contributing_features", [])
            f1 = (
                f"{top_feats[0]['feature']} ({top_feats[0]['shap_value']:+.3f})"
                if len(top_feats) >= 1
                else "N/A"
            )
            f2 = (
                f"{top_feats[1]['feature']} ({top_feats[1]['shap_value']:+.3f})"
                if len(top_feats) >= 2
                else "N/A"
            )
            rows.append(
                {
                    "Alert_ID": a.get("alert_id", ""),
                    "Timestamp_UTC": a.get("timestamp_utc", ""),
                    "Src_IP": a.get("src_ip", ""),
                    "Dst_IP": a.get("dst_ip", ""),
                    "Predicted_Class": a.get("predicted_class", ""),
                    "Confidence": round(float(a.get("confidence", 0.0)), 4),
                    "Status": "Suspicious" if a.get("is_suspicious") else "Benign",
                    "Severity": a.get("severity", ""),
                    "Latency_ms": round(float(a.get("inference_latency_ms", 0.0)), 4),
                    "Top_Feature_1": f1,
                    "Top_Feature_2": f2,
                    "Explanation_Method": a.get("explanation_method", ""),
                }
            )
        return pd.DataFrame(rows)

    def clear_logs(self) -> None:
        if self.log_path.exists():
            self.log_path.unlink()
