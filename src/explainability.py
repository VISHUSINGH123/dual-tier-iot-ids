"""
Module D (Part 1) — Explainable Intrusion Alerts via TreeSHAP (`TreeSHAPExplainer`).

Implements `TreeSHAPExplainer` (matching the BCSE497J report Class Diagram):
- Computes local per-record feature attributions using `shap.TreeExplainer` on the
  trained LightGBM model (or LightGBM's native exact TreeSHAP `pred_contrib=True`
  if `shap.TreeExplainer` encounters version-specific array shape quirks).
- If a non-tree model configuration is ever passed where TreeSHAP is unavailable,
  provides an explicitly-labelled global feature importance fallback so global
  importance is NEVER misrepresented as a local per-record SHAP explanation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_loader import classify_feature_origin


@dataclass
class FeatureContribution:
    """Single feature attribution for a predicted record."""

    feature_name: str
    protocol_family: str
    feature_value: float
    shap_value: float
    abs_shap_value: float
    direction: str  # "Increases class log-odds" or "Decreases class log-odds"


@dataclass
class LocalExplanation:
    """Complete per-record explanation for an IDS decision."""

    record_index: int
    predicted_class: str
    predicted_class_id: int
    confidence: float
    is_suspicious: bool
    explanation_method: str  # e.g., "TreeSHAP (Exact Local Attribution)" or "Global Gain Fallback"
    is_local_shap: bool
    base_value: float
    top_contributions: List[FeatureContribution] = field(default_factory=list)
    all_contributions: List[FeatureContribution] = field(default_factory=list)


class TreeSHAPExplainer:
    """
    TreeSHAP explanation engine for the LightGBM Edge Classifier.
    Matches `TreeSHAPExplainer` in the report's Class Diagram.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        label_mapping: Dict[int, str],
        top_k: int = 8,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.label_mapping = {int(k): str(v) for k, v in label_mapping.items()}
        self.top_k = max(1, int(top_k))
        self._shap_explainer = None
        self._shap_init_error: Optional[str] = None
        self._init_explainer()

    def _init_explainer(self) -> None:
        try:
            import shap

            self._shap_explainer = shap.TreeExplainer(self.model)
        except Exception as exc:
            self._shap_explainer = None
            self._shap_init_error = str(exc)

    def _extract_class_shap_vector(
        self,
        X_row: pd.DataFrame,
        target_class_id: int,
    ) -> Tuple[np.ndarray, float, str, bool]:
        """
        Return (shap_vector_for_predicted_class, base_value, method_label, is_local_shap).
        Tries `shap.TreeExplainer` first, then LightGBM's native exact TreeSHAP (`pred_contrib=True`),
        and only falls back to clearly-labelled global gain importance if both fail.
        """
        n_feats = len(self.feature_names)
        n_classes = len(self.label_mapping)

        # 1. Try shap.TreeExplainer
        if self._shap_explainer is not None:
            try:
                raw_shap = self._shap_explainer.shap_values(X_row)
                expected_val = self._shap_explainer.expected_value

                if isinstance(raw_shap, list):
                    # List of arrays, one per class: each has shape (1, n_feats)
                    cls_idx = min(target_class_id, len(raw_shap) - 1)
                    vec = np.asarray(raw_shap[cls_idx])[0]
                    base = (
                        float(expected_val[cls_idx])
                        if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > cls_idx
                        else float(np.asarray(expected_val).flat[0])
                    )
                    return vec, base, "TreeSHAP (shap.TreeExplainer Local Attribution)", True

                arr = np.asarray(raw_shap)
                if arr.ndim == 3:
                    # Shape: (1, n_feats, n_classes)
                    cls_idx = min(target_class_id, arr.shape[2] - 1)
                    vec = arr[0, :, cls_idx]
                    base = (
                        float(expected_val[cls_idx])
                        if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > cls_idx
                        else float(np.asarray(expected_val).flat[0])
                    )
                    return vec, base, "TreeSHAP (shap.TreeExplainer Local Attribution)", True
                elif arr.ndim == 2 and arr.shape[1] == n_feats:
                    # Binary classification shape: (1, n_feats)
                    vec = arr[0]
                    if n_classes == 2 and target_class_id == 0:
                        vec = -vec
                    base = float(np.asarray(expected_val).flat[-1])
                    return vec, base, "TreeSHAP (shap.TreeExplainer Local Attribution)", True
            except Exception:
                pass

        # 2. Try LightGBM native C++ exact TreeSHAP via pred_contrib=True
        try:
            contrib = self.model.predict(X_row, pred_contrib=True)
            contrib_arr = np.asarray(contrib)
            if contrib_arr.ndim == 2 and contrib_arr.shape[1] == (n_feats + 1) * n_classes:
                reshaped = contrib_arr[0].reshape(n_classes, n_feats + 1)
                cls_idx = min(target_class_id, n_classes - 1)
                vec = reshaped[cls_idx, :n_feats]
                base = float(reshaped[cls_idx, n_feats])
                return vec, base, "TreeSHAP (LightGBM Native Exact TreeSHAP)", True
            elif contrib_arr.ndim == 2 and contrib_arr.shape[1] == n_feats + 1:
                vec = contrib_arr[0, :n_feats]
                if n_classes == 2 and target_class_id == 0:
                    vec = -vec
                base = float(contrib_arr[0, n_feats])
                return vec, base, "TreeSHAP (LightGBM Native Exact TreeSHAP)", True
        except Exception:
            pass

        # 3. Explicitly labelled Global Feature Importance Fallback (never presented as local SHAP)
        raw_imp = getattr(self.model, "feature_importances_", np.ones(n_feats))
        imp_arr = np.asarray(raw_imp, dtype=np.float64)
        total = float(imp_arr.sum())
        norm_imp = imp_arr / total if total > 0 else imp_arr
        return (
            norm_imp,
            0.0,
            "FALLBACK: Global Model Split-Gain Importance (Not a per-record local SHAP attribution)",
            False,
        )

    def explain_instance(
        self,
        X_reduced_row: pd.DataFrame,
        predicted_class_id: int,
        confidence: float,
        record_index: int = 0,
    ) -> LocalExplanation:
        """
        Compute feature contributions for a single preprocessed & APAFC-reduced row.
        """
        if len(X_reduced_row) != 1:
            X_reduced_row = X_reduced_row.iloc[:1]

        X_aligned = X_reduced_row[self.feature_names]
        shap_vec, base_val, method_label, is_local = self._extract_class_shap_vector(
            X_aligned, predicted_class_id
        )

        pred_label = self.label_mapping.get(int(predicted_class_id), f"Class_{predicted_class_id}")
        is_suspicious = pred_label.strip().lower() not in ("normal", "benign", "0")

        row_vals = X_aligned.iloc[0]
        contributions: List[FeatureContribution] = []
        for idx, fname in enumerate(self.feature_names):
            fval = float(row_vals[fname])
            sval = float(shap_vec[idx])
            direction = (
                f"Pushes toward '{pred_label}' (+)"
                if sval >= 0
                else f"Pushes away from '{pred_label}' (-)"
            )
            if not is_local:
                direction = "Global split-gain importance (non-directional)"
            contributions.append(
                FeatureContribution(
                    feature_name=fname,
                    protocol_family=classify_feature_origin(fname),
                    feature_value=round(fval, 4),
                    shap_value=round(sval, 5),
                    abs_shap_value=round(abs(sval), 5),
                    direction=direction,
                )
            )

        contributions.sort(key=lambda c: c.abs_shap_value, reverse=True)
        top_k_list = contributions[: self.top_k]

        return LocalExplanation(
            record_index=record_index,
            predicted_class=pred_label,
            predicted_class_id=int(predicted_class_id),
            confidence=round(float(confidence), 5),
            is_suspicious=is_suspicious,
            explanation_method=method_label,
            is_local_shap=is_local,
            base_value=round(base_val, 5),
            top_contributions=top_k_list,
            all_contributions=contributions,
        )

    def get_global_feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of global LightGBM gain importances with protocol family tags."""
        raw_imp = getattr(self.model, "feature_importances_", np.zeros(len(self.feature_names)))
        imp_arr = np.asarray(raw_imp, dtype=np.float64)
        total = float(imp_arr.sum())
        norm_imp = (imp_arr / total) if total > 0 else imp_arr

        rows = []
        for fname, gain, norm in zip(self.feature_names, imp_arr, norm_imp):
            proto = classify_feature_origin(fname)
            rows.append(
                {
                    "Feature": fname,
                    "Protocol_Family": proto,
                    "Feature_Category": "Protocol-Derived" if proto != "Generic_Flow" else "Generic Flow Statistic",
                    "Gain_Importance": round(float(gain), 4),
                    "Normalized_Importance": round(float(norm), 4),
                }
            )
        df_imp = pd.DataFrame(rows).sort_values("Gain_Importance", ascending=False).reset_index(drop=True)
        return df_imp


def explain_core_event_shap(
    features_dict: Dict[str, Any],
    predicted_class: str = "benign",
    top_k: int = 14,
) -> Optional[Dict[str, Any]]:
    """
    Compute exact local TreeSHAP attributions for a given 14-feature flow event
    using the core LightGBM booster (`iot_ids_framework/artifacts/edge_ids.txt`).
    """
    from pathlib import Path
    from lightgbm import Booster
    from src.framework_bridge import FEATURE_NAMES, LABELS_JSON_PATH, TREE_MODEL_PATH

    if not TREE_MODEL_PATH.exists() or not LABELS_JSON_PATH.exists():
        return None

    try:
        labels = json.loads(LABELS_JSON_PATH.read_text(encoding="utf-8"))
        booster = Booster(model_file=str(TREE_MODEL_PATH))

        # Map protocol string to protocol_id if necessary
        proto_val = features_dict.get("protocol", "unknown")
        proto_map = {"unknown": 0.0, "mqtt": 1.0, "coap": 2.0, "http": 3.0, "6lowpan": 4.0}
        protocol_id = float(features_dict.get("protocol_id", proto_map.get(str(proto_val).lower(), 0.0)))

        vec: List[float] = []
        for f in FEATURE_NAMES:
            if f == "protocol_id":
                vec.append(protocol_id)
            elif f == "mean_packet_bytes":
                bc = float(features_dict.get("byte_count", 0.0))
                pc = float(features_dict.get("packet_count", 1.0) or 1.0)
                vec.append(bc / max(pc, 1.0))
            else:
                val = features_dict.get(f, 0.0)
                try:
                    vec.append(float(val) if val is not None else 0.0)
                except (ValueError, TypeError):
                    vec.append(0.0)

        x = np.asarray([vec], dtype=np.float32)
        contrib = booster.predict(x, pred_contrib=True)

        target_cls = str(predicted_class).lower()
        c_idx = labels.index(target_cls) if target_cls in labels else 0

        # LightGBM pred_contrib layout: n_classes * (n_features + 1)
        stride = len(FEATURE_NAMES) + 1
        shap_vals = contrib[0, c_idx * stride : c_idx * stride + len(FEATURE_NAMES)]
        base_bias = float(contrib[0, c_idx * stride + len(FEATURE_NAMES)])

        contributions = []
        for fname, fval, sval in zip(FEATURE_NAMES, vec, shap_vals):
            proto_fam = classify_feature_origin(fname)
            contributions.append(
                {
                    "feature": fname,
                    "protocol_family": proto_fam,
                    "feature_value": round(float(fval), 4),
                    "shap_value": round(float(sval), 5),
                    "abs_shap": round(abs(float(sval)), 5),
                    "direction": "Pushes toward Alert (Malicious)" if sval > 0 else "Pushes toward Nominal (Benign)",
                }
            )

        contributions.sort(key=lambda item: item["abs_shap"], reverse=True)

        return {
            "predicted_class": target_cls,
            "class_index": c_idx,
            "base_value": round(base_bias, 4),
            "feature_vector": {fn: round(fv, 4) for fn, fv in zip(FEATURE_NAMES, vec)},
            "top_contributions": contributions[:top_k],
            "all_contributions": contributions,
            "attribution_method": "Exact TreeSHAP (LightGBM Booster)",
        }
    except Exception as exc:
        import sys
        print(f"explain_core_event_shap error: {exc}", file=sys.stderr)
        return None


