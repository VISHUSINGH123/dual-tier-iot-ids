"""
Module for Unified Canonical Network Event Pipeline & Evaluation Bus.

Connects:
  Network Input (Live Socket / PCAP Replay / Real Datasets)
  -> Flow Extraction
  -> Feature Validation & Schema Routing
  -> ML Inference
  -> Alert Engine & Policy Enforcement
  -> Thread-Safe Storage & Metric Computation

Enforces absolute data integrity:
- Unlabelled live observations explicitly record `is_ground_truth_available=False`.
- False Positive Rate (FPR), Precision, Recall, and F1 are strictly marked as
  'Unavailable (Unlabelled Live Traffic)' when ground truth is absent.
- Full verification metrics (TP, FP, TN, FN, FPR, Macro-F1) are only computed
  over explicitly labelled evaluation batches.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from config.settings import CANONICAL_EVENTS_PATH

logger = logging.getLogger(__name__)

SEVERITY_HIERARCHY = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "INFO": 0,
}


@dataclass
class CanonicalNetworkEvent:
    """
    Unified canonical event representation across all input adapters, models,
    and storage layers.
    """

    event_id: str
    timestamp_utc: str
    input_mode: str  # 'live_socket' | 'pcap_replay' | 'dataset_batch'
    source_identifier: str  # e.g., '127.0.0.1:50555', 'Backdoor_attack.pcap', 'conn.log.labeled'
    flow_id: str  # e.g., '192.168.1.50:443 -> 10.0.0.1:54321 (TCP)'
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: str
    byte_count: int
    duration_ms: float
    feature_schema_id: str  # 'APAFC-14' | 'ZEEK-17' | 'EDGEIIOT-49' | 'RAW-FLOW'
    features: Dict[str, Any]
    model_id: str  # 'edge_ids_onnx' | 'experimental_zeek17_lgbm' | 'lgbm_ids_bundle' | 'none'
    model_version: str
    prediction: str  # predicted class or label
    confidence_score: float
    decision_threshold: float
    alert_status: str  # 'ALERT' | 'NOMINAL' | 'REJECTED' | 'ERROR'
    alert_severity: str  # 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'INFO'
    inference_latency_ms: float
    packet_count: Optional[int] = None
    packet_count_status: str = "observed"  # 'observed' | 'estimated' | 'unavailable_stream_socket'
    ground_truth_label: Optional[str] = None
    ground_truth_category: Optional[str] = None
    is_ground_truth_available: bool = False
    evaluation_match: Optional[str] = None  # 'TP' | 'FP' | 'TN' | 'FN' | None
    explanation: Optional[str] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CanonicalNetworkEvent:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def evaluate_event_match(
    prediction: str,
    alert_status: str,
    ground_truth_label: Optional[str],
) -> Optional[str]:
    """
    Compute binary TP / FP / TN / FN classification result.
    Returns None if ground truth is not provided.
    """
    if ground_truth_label is None or ground_truth_label == "":
        return None

    gt = str(ground_truth_label).strip().lower()
    is_gt_malicious = gt not in ("normal", "benign", "0", "false", "clean", "-")
    is_pred_malicious = (alert_status == "ALERT") or (
        str(prediction).strip().lower() not in ("normal", "benign", "0", "false", "clean")
    )

    if is_gt_malicious and is_pred_malicious:
        return "TP"
    elif not is_gt_malicious and is_pred_malicious:
        return "FP"
    elif not is_gt_malicious and not is_pred_malicious:
        return "TN"
    else:
        return "FN"


def determine_alert_severity(predicted_class: str, confidence: float) -> str:
    """Assign alert severity based on class risk profile and model confidence."""
    p = str(predicted_class).lower()
    if any(k in p for k in ("ddos", "ransomware", "backdoor", "c2", "botnet")):
        return "CRITICAL" if confidence >= 0.70 else "HIGH"
    elif any(k in p for k in ("injection", "sql", "xss", "uploading")):
        return "HIGH" if confidence >= 0.65 else "MEDIUM"
    elif any(k in p for k in ("scan", "fingerprint", "mitm", "password")):
        return "MEDIUM" if confidence >= 0.60 else "LOW"
    elif any(k in p for k in ("malicious", "attack", "suspicious")):
        return "HIGH" if confidence >= 0.75 else "MEDIUM"
    elif p in ("normal", "benign"):
        return "INFO"
    return "LOW"


class CanonicalEventBus:
    """
    Thread-safe event bus and circular storage buffer for network events.
    Supports in-memory ring buffer, persistent JSONL audit trail,
    and rigorous dual-mode metric aggregation (Operational vs Verified Evaluation).
    """

    def __init__(self, log_path: Path = CANONICAL_EVENTS_PATH, maxlen: int = 1500) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.events: deque = deque(maxlen=maxlen)
        self._load_recent_persisted(limit=250)

    def _load_recent_persisted(self, limit: int = 250) -> None:
        """Seed the in-memory ring buffer with recent persisted entries if present."""
        if not self.log_path.exists():
            return
        try:
            lines = []
            with self.log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(line)
            for line in lines[-limit:]:
                try:
                    data = json.loads(line)
                    self.events.append(CanonicalNetworkEvent.from_dict(data))
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Could not pre-load canonical events: %s", exc)

    def publish(self, event: CanonicalNetworkEvent) -> None:
        """Publish a single canonical event to in-memory deque and JSONL log."""
        with self.lock:
            self.events.append(event)
            try:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")
            except Exception as exc:
                logger.error("Failed to append event to JSONL log: %s", exc)

    def publish_batch(self, events: List[CanonicalNetworkEvent]) -> None:
        """Publish multiple events in an atomic lock block."""
        with self.lock:
            self.events.extend(events)
            try:
                with self.log_path.open("a", encoding="utf-8") as f:
                    for ev in events:
                        f.write(json.dumps(ev.to_dict()) + "\n")
            except Exception as exc:
                logger.error("Failed to append batch events to JSONL log: %s", exc)

    def get_events(
        self,
        limit: int = 200,
        input_mode: Optional[str] = None,
        alert_only: bool = False,
        min_severity: Optional[str] = None,
    ) -> List[CanonicalNetworkEvent]:
        """Query recent events (newest first) with optional filters."""
        with self.lock:
            filtered = list(self.events)

        if input_mode:
            filtered = [e for e in filtered if e.input_mode == input_mode]
        if alert_only:
            filtered = [e for e in filtered if e.alert_status == "ALERT"]
        if min_severity and min_severity in SEVERITY_HIERARCHY:
            req_lvl = SEVERITY_HIERARCHY[min_severity]
            filtered = [e for e in filtered if SEVERITY_HIERARCHY.get(e.alert_severity, 0) >= req_lvl]

        filtered.reverse()
        return filtered[:limit]

    def get_historical_alerts(self, limit: int = 500) -> List[CanonicalNetworkEvent]:
        """
        Return unified alert history, deduplicating between canonical events
        and historical ONNX/LightGBM logs without double-counting.
        """
        with self.lock:
            canonical_alerts = [e for e in self.events if e.alert_status == "ALERT"]

        seen_keys = {
            (e.timestamp_utc[:19], e.src_ip, e.src_port, e.dst_ip, e.dst_port, e.prediction)
            for e in canonical_alerts
        }
        all_alerts: List[CanonicalNetworkEvent] = list(canonical_alerts)

        # Check core framework ONNX alerts
        core_alerts_path = Path("iot_ids_framework/artifacts/alerts.jsonl")
        if core_alerts_path.exists() and len(all_alerts) < limit:
            try:
                lines = [l.strip() for l in core_alerts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                for line in reversed(lines):
                    if len(all_alerts) >= limit:
                        break
                    try:
                        item = json.loads(line)
                        ft = item.get("flow_5_tuple", {})
                        ts = str(item.get("timestamp", ""))
                        key = (
                            ts[:19],
                            str(ft.get("src_ip", "")),
                            int(ft.get("src_port", 0)),
                            str(ft.get("dst_ip", "")),
                            int(ft.get("dst_port", 0)),
                            str(item.get("predicted_class", "")),
                        )
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        top_attr = ""
                        attrs = item.get("attribution", [])
                        if attrs:
                            top_attr = f"{attrs[0]['feature']} ({attrs[0]['contribution']:+.4f})"

                        adapted = CanonicalNetworkEvent(
                            event_id=f"EVT-HIST-ONNX-{len(all_alerts):04d}",
                            timestamp_utc=ts,
                            input_mode="pcap_replay",
                            source_identifier="iot_ids_framework/alerts.jsonl",
                            flow_id=f"{ft.get('src_ip')}:{ft.get('src_port')} -> {ft.get('dst_ip')}:{ft.get('dst_port')} ({str(ft.get('protocol')).upper()})",
                            src_ip=str(ft.get("src_ip", "")),
                            src_port=int(ft.get("src_port", 0)),
                            dst_ip=str(ft.get("dst_ip", "")),
                            dst_port=int(ft.get("dst_port", 0)),
                            protocol=str(ft.get("protocol", "TCP")).upper(),
                            byte_count=0,
                            duration_ms=0.0,
                            feature_schema_id="APAFC-14",
                            features={},
                            model_id="edge_ids_onnx",
                            model_version="1.0.0",
                            prediction=str(item.get("predicted_class", "unknown")),
                            confidence_score=float(item.get("confidence", 0.9)),
                            decision_threshold=0.5,
                            alert_status="ALERT",
                            alert_severity=determine_alert_severity(
                                str(item.get("predicted_class")), float(item.get("confidence", 0.9))
                            ),
                            inference_latency_ms=1.0,
                            packet_count=None,
                            packet_count_status="unavailable_stream_socket",
                            ground_truth_label=None,
                            ground_truth_category=None,
                            is_ground_truth_available=False,
                            evaluation_match=None,
                            explanation=top_attr,
                            rejection_reason=None,
                        )
                        all_alerts.append(adapted)
                    except Exception:
                        continue
            except Exception as exc:
                logger.warning("Could not read historical core ONNX alerts: %s", exc)

        all_alerts.sort(key=lambda x: x.timestamp_utc, reverse=True)
        return all_alerts[:limit]

    def clear(self) -> None:
        """Clear the in-memory ring buffer."""
        with self.lock:
            self.events.clear()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Compute dual-mode metrics across all buffered events:
        1. Operational metrics (always valid for live and captured traffic)
        2. Ground-truth evaluation metrics (strictly separated, never fabricated)
        """
        with self.lock:
            events_snapshot = list(self.events)

        total_events = len(events_snapshot)
        if total_events == 0:
            return {
                "total_events": 0,
                "total_flows": 0,
                "total_packets": 0,
                "total_bytes": 0,
                "total_alerts": 0,
                "mean_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "alert_rate_pct": 0.0,
                "modes": {},
                "model_distribution": {},
                "has_ground_truth": False,
                "evaluation_status": "No events recorded",
                "verified_metrics": None,
            }

        total_flows = total_events
        observed_packets = sum(
            e.packet_count for e in events_snapshot
            if e.packet_count is not None and getattr(e, "packet_count_status", "observed") == "observed"
        )
        estimated_packets = sum(
            e.packet_count for e in events_snapshot
            if e.packet_count is not None and getattr(e, "packet_count_status", "observed") == "estimated"
        )
        has_unavailable_packets = any(
            e.packet_count is None or getattr(e, "packet_count_status", "observed") == "unavailable_stream_socket"
            for e in events_snapshot
        )
        total_packets = observed_packets + estimated_packets
        total_bytes = sum(e.byte_count for e in events_snapshot)
        total_alerts = sum(1 for e in events_snapshot if e.alert_status == "ALERT")
        latencies = [e.inference_latency_ms for e in events_snapshot if e.inference_latency_ms > 0]
        mean_latency = float(sum(latencies) / len(latencies)) if latencies else 0.0
        p95_latency = float(sorted(latencies)[int(len(latencies) * 0.95)]) if latencies else 0.0

        modes: Dict[str, int] = {}
        models: Dict[str, int] = {}
        for e in events_snapshot:
            modes[e.input_mode] = modes.get(e.input_mode, 0) + 1
            models[e.model_id] = models.get(e.model_id, 0) + 1

        # Check for verified evaluation ground truth
        labelled = [e for e in events_snapshot if e.is_ground_truth_available and e.evaluation_match is not None]
        has_gt = len(labelled) > 0

        verified_metrics: Optional[Dict[str, Any]] = None
        eval_status = "Unavailable (Unlabelled Live Traffic)"

        if has_gt:
            tp = sum(1 for e in labelled if e.evaluation_match == "TP")
            fp = sum(1 for e in labelled if e.evaluation_match == "FP")
            tn = sum(1 for e in labelled if e.evaluation_match == "TN")
            fn = sum(1 for e in labelled if e.evaluation_match == "FN")

            total_eval = tp + fp + tn + fn
            actual_benign = fp + tn
            actual_malicious = tp + fn
            predicted_alerts = tp + fp

            # Strict denominator-aware metrics: never fabricate 0% or assume missing classes exist
            accuracy = ((tp + tn) / total_eval) if total_eval > 0 else None
            precision = (tp / predicted_alerts) if predicted_alerts > 0 else None
            recall = (tp / actual_malicious) if actual_malicious > 0 else None
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = (2 * precision * recall) / (precision + recall)
            else:
                f1 = None
            fpr = (fp / actual_benign) if actual_benign > 0 else None

            verified_metrics = {
                "labelled_flows": total_eval,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "actual_benign": actual_benign,
                "actual_malicious": actual_malicious,
                "predicted_alerts": predicted_alerts,
                "accuracy": round(accuracy, 4) if accuracy is not None else None,
                "accuracy_percent": f"{accuracy * 100:.2f}%" if accuracy is not None else "Unavailable (No Labelled Samples)",
                "precision": round(precision, 4) if precision is not None else None,
                "precision_percent": f"{precision * 100:.2f}%" if precision is not None else "Unavailable (No Alerts Emitted)",
                "recall": round(recall, 4) if recall is not None else None,
                "recall_percent": f"{recall * 100:.2f}%" if recall is not None else "Unavailable (No Malicious Samples)",
                "f1_score": round(f1, 4) if f1 is not None else None,
                "f1_display": f"{f1:.4f}" if f1 is not None else "Unavailable",
                "fpr": round(fpr, 4) if fpr is not None else None,
                "fpr_percent": f"{fpr * 100:.2f}%" if fpr is not None else "Unavailable (No Benign Samples)",
            }
            eval_status = f"Verified across {total_eval} labelled flows"

        packet_semantics = "All Observed" if (not has_unavailable_packets and estimated_packets == 0) else (
            f"Observed: {observed_packets:,}, Estimated: {estimated_packets:,}" if not has_unavailable_packets else (
                f"Observed: {observed_packets:,} (Stream Socket Frames Excluded)"
            )
        )

        return {
            "total_events": total_events,
            "total_flows": total_flows,
            "total_packets": total_packets,
            "packet_semantics": packet_semantics,
            "total_bytes": total_bytes,
            "total_alerts": total_alerts,
            "mean_latency_ms": round(mean_latency, 3),
            "p95_latency_ms": round(p95_latency, 3),
            "alert_rate_pct": round((total_alerts / total_events) * 100.0, 2),
            "modes": modes,
            "model_distribution": models,
            "has_ground_truth": has_gt,
            "evaluation_status": eval_status,
            "verified_metrics": verified_metrics,
        }


# Global process-wide singleton
_GLOBAL_EVENT_BUS: Optional[CanonicalEventBus] = None
_BUS_LOCK = threading.Lock()


def get_event_bus() -> CanonicalEventBus:
    """Return thread-safe process-wide CanonicalEventBus singleton."""
    global _GLOBAL_EVENT_BUS
    if _GLOBAL_EVENT_BUS is None:
        with _BUS_LOCK:
            if _GLOBAL_EVENT_BUS is None:
                _GLOBAL_EVENT_BUS = CanonicalEventBus()
    return _GLOBAL_EVENT_BUS
