"""
Page 4 — Concept Drift Monitoring (`iot_ids.drift.KLDriftMonitor` & `DriftBenchmark`) and Retraining Workflow.
"""
from __future__ import annotations

import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.settings import (
    DRIFT_DEMO_CSV,
    DRIFT_HISTORY_PATH,
    DRIFT_REFERENCE_PATH,
    RETRAIN_LOG_PATH,
    SYNTHETIC_DEMO_CSV,
)
from src.drift_monitor import DriftMonitor, execute_drift_retraining
from src.framework_bridge import (
    FEATURE_NAMES,
    FRAMEWORK_ARTIFACTS,
    REFERENCE_JSON_PATH,
    SYNTHETIC_APAFC14_CSV,
    DriftBenchmark,
    KLDriftMonitor,
    generate_synthetic_apafc14_dataset,
    run_clean_drift_evaluation,
    train_core_onnx_pipeline,
)
from src.inference import EdgeClassifier
from src.ui_theme import render_header

st.set_page_config(page_title="Concept Drift Monitor | IoT IDS", layout="wide")

render_header(
    title=" Module E — Sliding-Window KL-Divergence Drift Monitor & Retraining",
    subtitle=(
        "Monitor feature-distribution shifts using `iot_ids.drift.KLDriftMonitor` & `DriftBenchmark` "
        "(Default τ = 0.15, 3 Consecutive Windows) and execute safe retraining."
    ),
    badge_text="KL Divergence (τ = 0.15)",
)

st.markdown(
    """
    <div class="sim-banner">
        <strong>Covariate Drift vs. Accuracy Decay Notice:</strong>
        KL divergence measures shifts in input feature distributions relative to the training reference.
        When ground-truth labels are absent in live traffic, the monitor reports covariate drift without
        claiming verified classification accuracy decay.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_core_drift, tab_window_sim, tab_retrain = st.tabs(
    [
        "1. Core `KLDriftMonitor` & `DriftBenchmark` (`iot_ids_framework`)",
        "2. Per-Feature Window History & Reference vs. Live Histograms",
        "3. Drift-Triggered Retraining Workflow & Audit History",
    ]
)

# ==============================================================================
# TAB 1: CORE KLDriftMonitor & DriftBenchmark (iot_ids.drift / drift_benchmark)
# ==============================================================================
with tab_core_drift:
    st.subheader(" `iot_ids.drift.KLDriftMonitor` & `DriftBenchmark` Harness")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tau_val = st.number_input("KL Threshold (τ)", min_value=0.01, max_value=2.0, value=0.15, step=0.01)
    with c2:
        consec_val = st.number_input("Consecutive Windows Trigger", min_value=1, max_value=10, value=3, step=1)
    with c3:
        win_size = st.number_input("Window Size (flows)", min_value=20, max_value=1000, value=100, step=20)
    with c4:
        shift_mag = st.slider("Injected Drift Magnitude (σ multiplier)", min_value=0.0, max_value=5.0, value=2.2, step=0.2)

    if st.button("▶ Run `KLDriftMonitor` Regime Evaluation (Baseline vs. Injected Concept Drift)", type="primary"):
        if not REFERENCE_JSON_PATH.exists():
            st.error("`iot_ids_framework/artifacts/reference.json` not found. Train the Core ONNX model first.")
        else:
            ref_payload = json.loads(REFERENCE_JSON_PATH.read_text(encoding="utf-8"))
            ref_arr = np.asarray(ref_payload["reference_features"], dtype=float)
            clean_res = run_clean_drift_evaluation(
                reference_data=ref_arr,
                window_size=int(win_size),
                bins=16,
                threshold=float(tau_val),
                consecutive_windows=int(consec_val),
                shift_magnitude=float(shift_mag),
                num_baseline_windows=4,
                num_drift_windows=6,
                random_state=42,
            )
            st.session_state["core_drift_bench"] = (clean_res, clean_res.window_curve_df, float(tau_val))

    if "core_drift_bench" in st.session_state:
        res, curve_df, used_tau = st.session_state["core_drift_bench"]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(
            "1. Mean KL Scores",
            f"Base: {res.mean_kl_baseline:.4f} | Drift: {res.mean_kl_drift:.4f}",
            delta=f"Threshold τ = {used_tau}",
        )
        m2.metric(
            "2. Exceeded Windows (KL > τ)",
            f"Drift: {res.drift_exceeded_windows}/{res.total_drift_windows} ({res.window_detection_rate * 100:.1f}%)",
            delta=f"Baseline Alarms: {res.baseline_exceeded_windows}/{res.total_baseline_windows} ({res.false_alarm_window_rate * 100:.1f}%)",
        )
        m3.metric(
            "3. Retraining Requests Fired",
            f"{res.drift_retrain_requests} request(s)",
            delta=f"Target C = {res.consecutive_windows_target} consecutive windows",
        )
        m4.metric(
            "4. Episode Detection Rate",
            f"{res.episode_detection_rate * 100:.1f}% ({res.detected_drift_episodes}/{res.total_drift_episodes})",
            delta=f"Window Recall: {res.window_detection_rate * 100:.1f}%",
        )
        delay_label = (
            f"{res.detection_delay_windows} win ({res.detection_delay_flows} flows)"
            if res.detection_delay_windows is not None
            else "Not Triggered"
        )
        m5.metric(
            "5. Detection Delay",
            delay_label,
            delta=f"Window size = {res.window_size} flows",
        )

        st.caption(
            "**Metric Denominators**: "
            f"**False Alarm Window Rate** = baseline windows with $\\text{{KL}} > \\tau$ / `total_baseline_windows` ({res.total_baseline_windows}); "
            f"**Window Detection Rate** = shifted windows with $\\text{{KL}} > \\tau$ / `total_drift_windows` ({res.total_drift_windows}); "
            f"**Episode Detection Rate** = drift episodes where `consecutive >= C` triggered retraining / `total_drift_episodes` ({res.total_drift_episodes}); "
            "**Detection Delay** = windows (and flows) elapsed from drift regime onset until `retrain_requested == True`."
        )

        fig_kl = px.line(
            curve_df,
            x="Window",
            y="KL_Divergence",
            markers=True,
            color="Regime",
            title=f"Sliding-Window KL Divergence Progression (Threshold τ = {used_tau})",
            template="plotly_dark",
        )
        fig_kl.add_hline(y=used_tau, line_dash="dash", line_color="#ef4444", annotation_text=f"Threshold τ = {used_tau}")
        st.plotly_chart(fig_kl, width="stretch")
        st.dataframe(curve_df, width="stretch", hide_index=True)


# ==============================================================================
# TAB 2: PER-FEATURE WINDOW HISTORY & HISTOGRAM COMPARISON
# ==============================================================================
with tab_window_sim:
    st.subheader(" Per-Feature KL Divergence & Reference vs. Incoming Distribution")
    if DRIFT_REFERENCE_PATH.exists():
        from config.settings import REAL_EDGE_IIOTSET_CSV

        monitor_ext = DriftMonitor.load_reference(DRIFT_REFERENCE_PATH)
        wcol1, wcol2 = st.columns(2)
        with wcol1:
            if st.button(" Append Real Baseline Traffic Window (200 Normal Flows)"):
                if not REAL_EDGE_IIOTSET_CSV.exists():
                    st.error(f"Real dataset missing at {REAL_EDGE_IIOTSET_CSV}")
                else:
                    clf = EdgeClassifier().load()
                    df_all = pd.read_csv(REAL_EDGE_IIOTSET_CSV, low_memory=False)
                    df_b = df_all[df_all["Attack_type"] == "Normal"].sample(n=200, random_state=42).reset_index(drop=True)
                    X_w, _ = clf.preprocess_to_reduced(df_b)
                    y_t = clf.preprocessor.encode_labels(df_b["Attack_type"])
                    y_p = clf.model.predict(X_w)
                    monitor_ext.evaluate_window(X_w, window_label="Real Baseline Normal Window", y_true=y_t, y_pred=y_p)
                    st.rerun()
        with wcol2:
            if st.button(" Append Real Shifted Attack Window (200 DDoS/Ransomware Flows)"):
                if not REAL_EDGE_IIOTSET_CSV.exists():
                    st.error(f"Real dataset missing at {REAL_EDGE_IIOTSET_CSV}")
                else:
                    clf = EdgeClassifier().load()
                    df_all = pd.read_csv(REAL_EDGE_IIOTSET_CSV, low_memory=False)
                    df_d = df_all[df_all["Attack_type"].isin(["DDoS_UDP", "Ransomware"])].sample(n=200, random_state=99).reset_index(drop=True)
                    X_w, _ = clf.preprocess_to_reduced(df_d)
                    y_t = clf.preprocessor.encode_labels(df_d["Attack_type"])
                    y_p = clf.model.predict(X_w)
                    monitor_ext.evaluate_window(X_w, window_label="Real Shifted Attack Window", y_true=y_t, y_pred=y_p)
                    st.rerun()

        if monitor_ext.window_history:
            hist_rows = [
                {
                    "Window_#": w.window_index,
                    "Label": w.window_label,
                    "Overall_KL": w.overall_kl_divergence,
                    "Max_Feature_KL": w.max_feature_kl,
                    "Most_Drifted_Feature": w.max_drifted_feature,
                    "Consecutive": f"{w.consecutive_exceeded_count}/{w.consecutive_trigger_limit}",
                    "Retrain_Triggered": w.retraining_triggered,
                    "Verified_Macro_F1": w.measured_window_macro_f1 if w.labels_available else "Unlabelled",
                }
                for w in monitor_ext.window_history
            ]
            st.dataframe(pd.DataFrame(hist_rows), width="stretch", hide_index=True)

            last_win = monitor_ext.window_history[-1]
            pf_df = pd.DataFrame(
                [{"Feature": k, "KL_Divergence": v} for k, v in last_win.per_feature_kl.items()]
            ).sort_values("KL_Divergence", ascending=False)
            fig_pf = px.bar(
                pf_df,
                x="Feature",
                y="KL_Divergence",
                color="KL_Divergence",
                color_continuous_scale="OrRd",
                title=f"Per-Feature KL Divergence in Window #{last_win.window_index} ({last_win.window_label})",
                template="plotly_dark",
            )
            fig_pf.add_hline(y=monitor_ext.tau, line_dash="dash", line_color="#38bdf8")
            st.plotly_chart(fig_pf, width="stretch")


# ==============================================================================
# TAB 3: DRIFT-TRIGGERED RETRAINING WORKFLOW
# ==============================================================================
with tab_retrain:
    st.subheader(" Safe Drift-Adaptive Retraining Workflow")
    st.write(
        "When sustained concept drift crosses $\\tau$ for $C$ consecutive windows, you can trigger "
        "either **Primary Core ONNX Retraining + Atomic Hot-Swap** or **Exploratory Bundle Retraining**."
    )
    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.markdown("##### Primary Core ONNX Retraining")
        st.info(
            "Primary Core ONNX model retraining requires a fully compatible 14-feature APAFC real dataset. "
            "In accordance with the project data policy, synthetic smoke data cannot be used to overwrite production models."
        )
        st.button(" Retrain Core ONNX Model (Disabled — Real APAFC-14 dataset required)", disabled=True)

    with rcol2:
        st.markdown("##### Exploratory Model Retraining")
        if st.button(" Retrain Exploratory Model on Real Baseline + Shifted Data", type="primary"):
            from config.settings import REAL_EDGE_IIOTSET_CSV

            if not REAL_EDGE_IIOTSET_CSV.exists():
                st.error(f"Real dataset missing at {REAL_EDGE_IIOTSET_CSV}")
            else:
                with st.spinner("Extracting real baseline normal and shifted attack partitions and retraining..."):
                    df_all = pd.read_csv(REAL_EDGE_IIOTSET_CSV, low_memory=False)
                    df_b = df_all[df_all["Attack_type"] == "Normal"].sample(n=2500, random_state=42)
                    df_d = df_all[df_all["Attack_type"].isin(["DDoS_UDP", "Ransomware", "Backdoor"])].sample(n=2500, random_state=99)
                    rec = execute_drift_retraining(df_b, df_d)
                    st.success(f"Retraining Status: {rec['status']} | Updated Macro-F1: {rec['updated_macro_f1']}")

    if RETRAIN_LOG_PATH.exists():
        st.markdown("##### Retraining Audit History (`logs/retraining_history.json`)")
        r_hist = json.loads(RETRAIN_LOG_PATH.read_text(encoding="utf-8"))
        st.dataframe(pd.DataFrame(r_hist), width="stretch", hide_index=True)
