"""Autonomous edge inference, explainable alerts, drift requests, and atomic model replacement."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import threading
from collections.abc import Callable, Mapping
import numpy as np

from .drift import KLDriftMonitor
from .features import APAFCFeatureExtractor


@dataclass(frozen=True)
class Alert:
    timestamp: str
    flow_5_tuple: dict[str, object]
    predicted_class: str
    confidence: float
    attribution_method: str
    attribution: list[dict[str, float]]


class EdgeIDS:
    def __init__(self, session, class_names: list[str], monitor: KLDriftMonitor,
                 extractor: APAFCFeatureExtractor | None = None, top_k: int = 5,
                 on_retrain: Callable[[dict[str, object]], None] | None = None,
                 tree_explainer=None, audit_path: str | Path | None = None) -> None:
        if not class_names or class_names[0] != "benign":
            raise ValueError("class_names must put benign at index 0")
        self._session, self._lock = session, threading.RLock()
        self.class_names, self.monitor = class_names, monitor
        self.extractor, self.top_k, self.on_retrain = extractor or APAFCFeatureExtractor(), top_k, on_retrain
        self.tree_explainer = tree_explainer
        self.audit_path = Path(audit_path) if audit_path else None

    def classify(self, flow: Mapping[str, object]) -> dict[str, object]:
        vector = self.extractor.extract(flow)
        with self._lock:
            input_name = self._session.get_inputs()[0].name
            raw_probs = self._session.run(None, {input_name: np.asarray([vector], dtype=np.float32)})[1][0]
        if isinstance(raw_probs, dict):
            probabilities = np.array([raw_probs.get(i, 0.0) for i in range(len(self.class_names))], dtype=np.float32)
        else:
            probabilities = np.asarray(raw_probs, dtype=np.float32)
        index = int(np.argmax(probabilities))
        result: dict[str, object] = {"class": self.class_names[index], "confidence": float(probabilities[index])}
        drift = self.monitor.observe(vector)
        if drift:
            result["drift"] = {"kl": drift.score, "consecutive_windows": drift.consecutive, "retrain_requested": drift.retrain_requested}
            if drift.retrain_requested and self.on_retrain:
                self.on_retrain(result["drift"])
        if index != 0:
            method = "TreeSHAP" if self.tree_explainer is not None else "feature-magnitude fallback"
            alert = Alert(datetime.now(timezone.utc).isoformat(), self._five_tuple(flow), self.class_names[index], float(probabilities[index]), method, self._explain(vector, index))
            result["alert"] = asdict(alert)
            self._write_alert(result["alert"])
        return result

    def hot_swap(self, replacement_session) -> None:
        """Install a validated new session atomically; in-flight requests retain the old one."""
        if not replacement_session.get_inputs():
            raise ValueError("replacement ONNX session has no input")
        with self._lock:
            self._session = replacement_session

    def _explain(self, vector: list[float], predicted_index: int) -> list[dict[str, float]]:
        if self.tree_explainer is not None:
            values = self.tree_explainer.shap_values(np.asarray([vector], dtype=float))
            # SHAP returns either [class][row, feature] or [row, feature, class].
            if isinstance(values, list):
                contributions = enumerate(values[predicted_index][0])
            else:
                contributions = enumerate(values[0, :, predicted_index] if values.ndim == 3 else values[0])
            ranked = sorted(contributions, key=lambda item: abs(item[1]), reverse=True)[:self.top_k]
            return [{"feature": self.extractor.feature_names[i], "contribution": float(value)} for i, value in ranked]
        # A model artifact may be unavailable during recovery. Explicitly label the fallback
        # so operators do not confuse it with a TreeSHAP explanation.
        contributions = sorted(enumerate(np.abs(vector)), key=lambda item: item[1], reverse=True)[:self.top_k]
        return [{"feature": self.extractor.feature_names[i], "contribution": float(value)} for i, value in contributions]

    @staticmethod
    def _five_tuple(flow: Mapping[str, object]) -> dict[str, object]:
        return {key: flow.get(key) for key in ("src_ip", "src_port", "dst_ip", "dst_port", "protocol")}

    def _write_alert(self, alert: dict[str, object]) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.audit_path.open("a", encoding="utf-8") as audit:
            audit.write(json.dumps(alert, separators=(",", ":")) + "\n")


def load_monitor(reference_path: str | Path, config: Mapping[str, object]) -> KLDriftMonitor:
    payload = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    drift = config["drift"]
    return KLDriftMonitor(np.asarray(payload["reference_features"], dtype=float), **drift)


def load_tree_explainer(model_path: str | Path):
    """Load the matching LightGBM model used to produce actual TreeSHAP attributions."""
    from lightgbm import Booster
    import shap
    return shap.TreeExplainer(Booster(model_file=str(model_path)))
