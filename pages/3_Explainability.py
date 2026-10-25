"""
Page 3 — Explainable AI (`TreeSHAP`) & APAFC Feature Analysis.

Provides:
1. Primary Core TreeSHAP Explanations (`iot_ids.edge.load_tree_explainer` on `edge_ids.txt`)
   for any flow record or alert in `iot_ids_framework/artifacts/alerts.jsonl`.
2. Exploratory 61-to-k APAFC Protocol vs. Generic Flow Feature Ranking & Local TreeSHAP Waterfall.
"""
from __future__ import annotations

import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lightgbm import Booster

from config.settings import FEATURE_CONFIG_PATH, MODEL_ARTIFACT_PATH
from src.explainability import TreeSHAPExplainer
from src.framework_bridge import (
    ALERTS_JSONL_PATH,
    FEATURE_NAMES,
    LABELS_JSON_PATH,
    SYNTHETIC_APAFC14_CSV,
    TREE_MODEL_PATH,
    load_core_edge_ids,
    load_tree_explainer,
)
from src.inference import EdgeClassifier
from src.ui_theme import render_header

st.set_page_config(page_title="Explainability | IoT IDS", layout="wide")

render_header(
    title=" Module C & D — TreeSHAP Local Explainability & APAFC Feature Attribution",
    subtitle="Inspect per-decision TreeSHAP attributions (`iot_ids.edge`) and global protocol-aware feature rankings.",
    badge_text="TreeSHAP Active",
)

tab_core_shap, tab_apafc_rank = st.tabs(
    [
        "1. Primary 14-Feature TreeSHAP Attribution (`iot_ids.edge`)",
        "2. Exploratory 61-to-k Protocol-Aware Feature Importance (`APAFCExtractor`)",
    ]
)

with tab_core_shap:
    st.subheader(" Per-Record TreeSHAP Explanation (`iot_ids_framework/artifacts/edge_ids.txt`)")
    if not TREE_MODEL_PATH.exists() or not LABELS_JSON_PATH.exists():
        st.warning("Primary LightGBM Booster (`edge_ids.txt`) not found yet. Train the Core ONNX model on Page 1 first.")
    else:
        booster = Booster(model_file=str(TREE_MODEL_PATH))
        labels = json.loads(LABELS_JSON_PATH.read_text(encoding="utf-8"))
        gains = booster.feature_importance(importance_type="gain")
        df_global = pd.DataFrame(
            {
                "Feature": list(FEATURE_NAMES),
                "Split_Gain": [round(float(g), 3) for g in gains],
            }
        ).sort_values("Split_Gain", ascending=False)

        gcol1, gcol2 = st.columns([1.0, 1.3])
        with gcol1:
            st.markdown("##### Global LightGBM Split-Gain (14 APAFC Features)")
            fig_g = px.bar(
                df_global,
                x="Split_Gain",
                y="Feature",
                orientation="h",
                color="Split_Gain",
                color_continuous_scale="Tealgrn",
                template="plotly_dark",
            )
            fig_g.update_layout(height=420, yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(fig_g, width="stretch")

        with gcol2:
            st.markdown("##### Local TreeSHAP Attribution for Logged Incident / Flow")
            from src.event_pipeline import get_event_bus
            from src.explainability import explain_core_event_shap
            from src.live_listener import SIMULATION_SCENARIOS

            alerts = get_event_bus().get_historical_alerts(limit=50)

            source_choice = st.radio(
                "Source Record for TreeSHAP Explanation",
                options=["Recent Logged Security Alerts", "Standard IoT Scenario Presets"],
                horizontal=True,
            )

            if source_choice == "Recent Logged Security Alerts" and alerts:
                alert_map = {
                    f"{a.event_id} | {a.prediction.upper()} ({a.confidence_score*100:.1f}%) | {a.protocol} {a.src_ip}:{a.src_port}->{a.dst_port}": a
                    for a in alerts
                }
                sel_label = st.selectbox("Select Logged Alert Incident", options=list(alert_map.keys()))
                chosen_alert = alert_map[sel_label]

                shap_res = explain_core_event_shap(chosen_alert.features, predicted_class=chosen_alert.prediction)
                pred_cls = chosen_alert.prediction
                conf_val = chosen_alert.confidence_score
                method_lbl = "Exact TreeSHAP (LightGBM Booster `edge_ids.txt`)"
            else:
                if source_choice == "Recent Logged Security Alerts":
                    st.info("No logged alerts yet. Displaying standard IoT threat scenario presets.")

                scen_keys = list(SIMULATION_SCENARIOS.keys())
                sel_scen = st.selectbox(
                    "Select Flow Scenario Preset",
                    options=scen_keys,
                    format_func=lambda k: SIMULATION_SCENARIOS[k]["name"],
                )
                scen_data = SIMULATION_SCENARIOS[sel_scen]
                edge_ids = load_core_edge_ids(top_k=14)
                pred_out = edge_ids.classify(scen_data)
                pred_cls = pred_out["class"]
                conf_val = pred_out["confidence"]
                shap_res = explain_core_event_shap(scen_data, predicted_class=pred_cls)
                method_lbl = "Exact TreeSHAP on Scenario Features"

            if shap_res and shap_res.get("top_contributions"):
                tc = shap_res["top_contributions"]
                df_local = pd.DataFrame(tc)

                st.markdown(
                    f"**Predicted Class:** `{pred_cls.upper()}` (`{conf_val*100:.2f}%`) | "
                    f"**Base Log-Odds:** `{shap_res['base_value']}` | "
                    f"**Method:** `{method_lbl}`"
                )

                colors = ["#ef4444" if s > 0 else "#10b981" for s in df_local["shap_value"]]
                fig_l = go.Figure(
                    go.Bar(
                        x=df_local["shap_value"],
                        y=df_local["feature"],
                        orientation="h",
                        marker=dict(color=colors),
                        text=[f"{v:+.4f}" for v in df_local["shap_value"]],
                        textposition="auto",
                    )
                )
                fig_l.update_layout(
                    title=f"TreeSHAP Feature Contributions for '{pred_cls.upper()}'",
                    xaxis_title="SHAP Value (Contribution to Log-Odds)",
                    yaxis_title="Feature",
                    yaxis=dict(autorange="reversed"),
                    height=360,
                    margin=dict(l=10, r=10, t=35, b=10),
                    template="plotly_dark",
                )
                st.plotly_chart(fig_l, width="stretch")
                st.dataframe(
                    df_local[["feature", "feature_value", "shap_value", "direction"]],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.warning("Could not compute TreeSHAP attribution for the selected record.")

with tab_apafc_rank:
    st.subheader(" Exploratory 61-to-k Feature Ranking (`models/apafc_feature_config.json`)")
    if FEATURE_CONFIG_PATH.exists():
        cfg_data = json.loads(FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))
        summary = cfg_data.get("summary", {})
        st.info(summary.get("limitation_note", ""))
        rankings = pd.DataFrame(summary.get("feature_rankings", []))
        if not rankings.empty:
            fig_r = px.bar(
                rankings.head(20),
                x="normalized_importance",
                y="feature_name",
                color="protocol_family",
                orientation="h",
                title="Top-20 Ranked Candidate Features by Protocol Origin (Exploratory 61-Feature Study)",
                template="plotly_dark",
            )
            fig_r.update_layout(height=480, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_r, width="stretch")
            st.dataframe(rankings, width="stretch", hide_index=True)
    else:
        st.info("Run the Exploratory 61-to-k study on Page 1 (Tab 2) to populate 61-feature rankings.")
