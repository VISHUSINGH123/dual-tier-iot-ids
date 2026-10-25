"""
Page 1 — Dataset Preparation, Core ONNX Training (`iot_ids_framework`) & 61->k APAFC Ablation Study.

Explicitly separates:
1. Primary Core Pipeline (`iot_ids_framework`): Fixed 14-feature APAFC flow schema
   (`iot_ids.features.FEATURE_NAMES`), `iot_ids.prepare.map_csv`, Optuna + in-fold SMOTE-ENN
   LightGBM training (`iot_ids.training.train`), ONNX export (`edge_ids.onnx`), and
   classical ML baseline comparison (`iot_ids.baselines.benchmark_comparison`).
2. Exploratory Dataset-Driven Workflow (`src/`): Configurable 61-to-k feature selection
   experiment (`APAFCExtractor` + `DatasetPreprocessor`) for arbitrary CSV datasets.
"""
from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

from config.settings import (
    DEFAULT_APAFC_K,
    EVALUATION_RESULTS_PATH,
    MODEL_ARTIFACT_PATH,
    SYNTHETIC_DEMO_CSV,
)
from src.data_loader import (
    generate_synthetic_edge_iiotset,
    inspect_dataset,
    load_csv_dataset,
    load_or_create_demo_dataset,
)
from src.framework_bridge import (
    FEATURE_NAMES,
    FRAMEWORK_ARTIFACTS,
    FRAMEWORK_CONFIGS,
    SYNTHETIC_APAFC14_CSV,
    benchmark_baseline_models,
    check_core_artifacts_status,
    generate_sample_iot_pcap,
    generate_synthetic_apafc14_dataset,
    map_csv,
    train_core_onnx_pipeline,
)
from src.train_model import train_and_evaluate_pipeline
from src.ui_theme import render_header

st.set_page_config(page_title="Dataset & Training | IoT IDS", layout="wide")

render_header(
    title=" Module A, B & C — Core ONNX Training (`iot_ids_framework`) & 61→k APAFC Study",
    subtitle=(
        "Train the Primary Fixed-14 APAFC ONNX Edge Model (`iot_ids.training`) or run the "
        "Exploratory 61-to-k Dataset-Driven Feature Selection Experiment."
    ),
    badge_text="Dual-Workflow Training",
)

tab_core, tab_ablation, tab_baselines, tab_pcap_bench = st.tabs(
    [
        "1. Primary Core Pipeline (Fixed 14-Feature APAFC → Optuna LightGBM → ONNX)",
        "2. Exploratory 61-to-k Dataset-Driven Feature Selection & CSV Upload",
        "3. Classical ML Baseline Comparison (`iot_ids.baselines`)",
        "4. Genuine PCAP Evaluation Benchmark (Edge-IIoTset 24-Capture Audit)",
    ]
)

# ==============================================================================
# TAB 1: PRIMARY CORE PIPELINE (iot_ids_framework)
# ==============================================================================
with tab_core:
    st.markdown(
        """
        <div class="sim-banner">
            <strong>Primary Edge Inference Schema (`iot_ids.features.FEATURE_NAMES`):</strong>
            Uses the fixed 14-feature vector required by <code>EdgeIDS</code>, <code>PCAPReader</code>,
            and <code>cpp/edge_runtime.cpp</code>:
            <code>protocol_id, duration_ms, packet_count, byte_count, mean_packet_bytes, src_port, dst_port,
            tcp_flags, mqtt_qos, mqtt_payload_bytes, coap_code, http_status, sixlowpan_dispatch, payload_entropy</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    core_status = check_core_artifacts_status()
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric(
        "ONNX Edge Artifact",
        "Present" if core_status["files"]["edge_ids.onnx"]["exists"] else "Missing",
        delta=f"{core_status['files']['edge_ids.onnx']['size_kb']} KB",
    )
    sc2.metric(
        "LightGBM Tree Booster",
        "Present" if core_status["files"]["edge_ids.txt"]["exists"] else "Missing",
        delta=f"{core_status['files']['edge_ids.txt']['size_kb']} KB",
    )
    sc3.metric(
        "Drift Reference Matrix",
        "Present" if core_status["files"]["reference.json"]["exists"] else "Missing",
        delta=f"{core_status['files']['reference.json']['size_kb']} KB",
    )
    sc4.metric(
        "Sample PCAP Capture",
        "Present" if core_status["files"]["sample_traffic.pcap"]["exists"] else "Missing",
        delta=f"{core_status['files']['sample_traffic.pcap']['size_kb']} KB",
    )

    st.markdown("####  Model Training Workflow & Dataset Mode Selector")
    training_mode = st.radio(
        "Select Model Training Workflow",
        options=[
            "1. Real IoT-23 LightGBM Training (17 Zeek-Native Features - Recommended for IoT-23)",
            "2. Core 14-Feature ONNX Training (Genuinely Compatible Real Dataset Only)",
        ],
        index=0,
        help="Select between real-data training on IoT-23 or real compatible ONNX training.",
    )

    if "1. Real IoT-23" in training_mode:
        st.info(
            " **Active Mode: Real IoT-23 Scenario-Based LightGBM Training**\n\n"
            "- **Dataset Source**: Streams directly from your real IoT-23 `conn.log.labeled` files across 14 training scenarios.\n"
            "- **Feature Schema**: Uses strictly the verified 17 native Zeek connection features without hallucinating L7 payload fields.\n"
            "- **Output Isolation**: Saves self-contained artifacts to `models/experimental_iot23_lgbm_bundle.joblib` and "
            "`models/experimental_iot23_evaluation.json`. Does **not** modify or overwrite the production `edge_ids.onnx` artifact.\n"
            "- **Cross-Capture Evaluation**: Evaluates strictly against 9 held-out test scenarios."
        )
        col_tr1, col_tr2, col_tr3 = st.columns(3)
        with col_tr1:
            tr_rows = st.number_input("Max Training Rows per Scenario", min_value=100, max_value=5000, value=2000, step=500)
        with col_tr2:
            te_rows = st.number_input("Max Evaluation Rows per Scenario", min_value=100, max_value=5000, value=2000, step=500)
        with col_tr3:
            tr_seed = st.number_input("Random Seed", min_value=1, max_value=99999, value=42)

        if st.button(" Train & Evaluate Real IoT-23 Model (`src.iot23_loader`)", type="primary"):
            from src.iot23_loader import train_and_evaluate_experimental_iot23_model

            with st.spinner("Streaming real IoT-23 scenarios, fitting LightGBM, and evaluating held-out scenarios..."):
                try:
                    res = train_and_evaluate_experimental_iot23_model(
                        train_rows_per_scenario=int(tr_rows),
                        test_rows_per_scenario=int(te_rows),
                        random_seed=int(tr_seed),
                    )
                    st.success(
                        f"Real IoT-23 training complete! Evaluated {res['test_sample_size']} held-out flows. "
                        f"Accuracy: {res['binary_metrics']['accuracy']*100:.2f}% | "
                        f"Macro-F1: {res['binary_metrics']['macro_f1']:.4f} | "
                        f"FPR: {res['binary_metrics']['false_positive_rate']*100:.2f}%."
                    )
                    st.json(res["binary_metrics"])
                except Exception as exc:
                    st.error(f"IoT-23 training failed: {exc}")

    elif "2. Core 14-Feature" in training_mode:
        from src.framework_bridge import (
            EXPERIMENTAL_ONNX_OUTPUT_DIR,
            validate_apafc14_semantic_compatibility,
        )

        st.info(
            " **Active Mode: Core 14-Feature ONNX Training via Compatible Real Dataset**\n\n"
            "- **Requirement**: The dataset must legitimately provide all 14 APAFC features (including L7 protocol fields: "
            "`mqtt_qos`, `mqtt_payload_bytes`, `coap_code`, `http_status`, `sixlowpan_dispatch`, `payload_entropy`).\n"
            "- **Output Isolation**: Any newly trained experimental ONNX models will be written to `models/experimental_onnx_runs/` "
            "and will never overwrite the production `edge_ids.onnx` artifact.\n"
            "- **Strict Guard**: Column-name matching alone is insufficient. Zero-filling or guessing missing L7 fields is rejected."
        )

        edge_csv_candidate = Path("archive/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv")
        edge_mapper_candidate = Path("iot_ids_framework/configs/edge_iiotset_feature_mapping.json")

        sem_check = validate_apafc14_semantic_compatibility(edge_csv_candidate, edge_mapper_candidate)
        if not sem_check["compatible"]:
            st.error(
                f" **Real-Data 14-Feature ONNX Training Disabled**\n\n"
                f"**Reason**: {sem_check['reason']}\n\n"
                f"- **Missing / Unmapped Features**: `{sem_check['missing_features']}`\n"
                f"- **Unsupported L7 Deep-Packet Fields**: `{sem_check['unsupported_l7_features']}`\n\n"
                "**Integrity Safeguard**: To prevent invalid models, training cannot proceed on this dataset. "
                "The 14-feature ONNX schema strictly requires legitimate L7 protocol features that are absent in this mapper."
            )
            st.button(" Train Core ONNX from Real Dataset (Disabled due to schema mismatch)", disabled=True)
        else:
            if st.button(" Train Experimental Core ONNX from Real Dataset", type="primary"):
                st.warning("Training would run here if a fully compatible dataset is provided.")

    # Real IoT-23 Dataset Explorer
    st.markdown("####  Real Dataset Source: IoT-23 Zeek Connection Logs (`conn.log.labeled`)")
    from src.iot23_loader import (
        IOT23_RAW_ROOT,
        get_available_iot23_scenarios,
        preview_iot23_scenario_records,
    )

    available_scenarios = get_available_iot23_scenarios()
    if available_scenarios:
        scen_names = list(available_scenarios.keys())
        default_idx = scen_names.index("CTU-IoT-Malware-Capture-8-1") if "CTU-IoT-Malware-Capture-8-1" in scen_names else 0

        c_scen1, c_scen2 = st.columns([1.5, 1.0])
        with c_scen1:
            selected_scen = st.selectbox(
                "Select Real IoT-23 Scenario to Explore",
                options=scen_names,
                index=default_idx,
                help="Select one of the 23 real-world CTU IoT malware and honeypot capture scenarios.",
            )
        with c_scen2:
            preview_limit = st.slider("Preview Record Limit", min_value=10, max_value=200, value=50, step=10)

        try:
            prev_data = preview_iot23_scenario_records(selected_scen, max_rows=int(preview_limit))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Scenario", prev_data["scenario"])
            m2.metric("Log File Size", f"{prev_data['file_size_mb']} MB")
            m3.metric("Records Previewed", prev_data["rows_sampled"])
            m4.metric("Unique Labels in Sample", len(prev_data["labels"]))

            st.caption(f"**Source File**: `{prev_data['file_path']}`")

            l_col1, l_col2 = st.columns(2)
            with l_col1:
                st.markdown("**Binary Ground-Truth Distribution in Sample:**")
                st.json(prev_data["labels"])
            with l_col2:
                st.markdown("**Detailed Taxonomic Ground-Truth Distribution:**")
                st.json(prev_data["detailed_labels"])

            with st.expander(f" Preview Real Zeek Connection Records ({len(prev_data['preview_df'])} rows)", expanded=True):
                st.dataframe(prev_data["preview_df"], width="stretch")

            st.info(
                " **Dataset Integrity Notice**: IoT-23 consists of genuine Zeek/Bro network connection summaries (`conn.log.labeled`). "
                "Each record contains transport-level connection telemetry, flags, and ground-truth annotations from the CTU research lab. "
                "Zeek summaries do not contain raw packet payloads (L7 deep packet inspection fields)."
            )
        except Exception as exc:
            st.error(f"Error reading IoT-23 scenario: {exc}")
    else:
        st.warning(f"No extracted IoT-23 scenarios discovered under `{IOT23_RAW_ROOT}`.")

    st.markdown("---")
    st.markdown("####  Synthetic Smoke Test Fixture (Core ONNX 14-Feature Baseline)")
    if SYNTHETIC_APAFC14_CSV.exists():
        df_14_preview = pd.read_csv(SYNTHETIC_APAFC14_CSV)
        with st.expander(f" View Synthetic Smoke-Test Fixture (`{SYNTHETIC_APAFC14_CSV.name}`, {len(df_14_preview)} rows)"):
            st.caption(
                "**Notice**: This synthetic 14-feature CSV is an artificial smoke-test fixture generated for validating "
                "the `iot_ids.training` Optuna pipeline when raw packet captures or compatible 14-feature datasets are absent. "
                "It is not real network traffic."
            )
            st.dataframe(df_14_preview.head(20), width="stretch")

    st.markdown("#### Optional: Map Raw Dataset Export via `iot_ids.prepare.map_csv`")
    with st.expander(" Map Raw Edge-IIoTset / ToN-IoT CSV to Fixed 14-Feature Schema"):
        raw_upload = st.file_uploader("Upload Raw CSV to Map", type=["csv"], key="map_csv_uploader")
        mapping_choice = st.selectbox(
            "Mapping Configuration (`iot_ids_framework/configs/`)",
            options=[
                "edge_iiotset_feature_mapping.json",
                "ton_iot_feature_mapping.json",
                "feature_mapping.example.json",
            ],
        )
        raw_label_col = st.text_input("Source Label Column Name", value="Attack_type")
        if raw_upload is not None and st.button("Convert via `map_csv`"):
            try:
                tmp_raw = FRAMEWORK_ARTIFACTS.parent / "datasets" / f"raw_{raw_upload.name}"
                tmp_raw.parent.mkdir(parents=True, exist_ok=True)
                tmp_raw.write_bytes(raw_upload.getvalue())
                out_mapped = tmp_raw.parent / f"mapped_{raw_upload.name}"
                map_csv(
                    source=tmp_raw,
                    mapping_file=FRAMEWORK_CONFIGS / mapping_choice,
                    label_column=raw_label_col,
                    output=out_mapped,
                )
                st.success(f"Mapped dataset saved to `{out_mapped}`.")
            except Exception as exc:
                st.error(f"`map_csv` error: {exc}")


# ==============================================================================
# TAB 2: EXPLORATORY 61-TO-K FEATURE SELECTION EXPERIMENT
# ==============================================================================
with tab_ablation:
    from config.settings import REAL_EDGE_IIOTSET_CSV

    st.markdown(
        """
        <div class="sim-banner">
            <strong>Real-Data Exploratory Workflow (61-to-k Feature Selection & Model Evaluation):</strong>
            Connects directly to the genuine <strong>Edge-IIoTset</strong> dataset (157,800 rows across 15 attack classes)
            or an uploaded custom CSV. Performs leak-free preprocessing, ranks features via LightGBM split gain,
            applies training-only class balancing, and evaluates on held-out test records.
        </div>
        """,
        unsafe_allow_html=True,
    )

    src_col1, src_col2 = st.columns([1.2, 1.0])
    with src_col1:
        data_source = st.radio(
            "Exploratory Dataset Source",
            options=[
                f"Real Edge-IIoTset Dataset ({REAL_EDGE_IIOTSET_CSV.name}, 157,800 rows)",
                "Upload Custom CSV Dataset",
            ],
            index=0,
        )
    with src_col2:
        if "Real Edge-IIoTset" in data_source:
            sample_mode = st.selectbox(
                "Training Partition Strategy",
                options=[
                    "15,000 Rows (Stratified 1,000 flows/class - Fast Evaluation)",
                    "30,000 Rows (Stratified 2,000 flows/class)",
                    "Full Dataset (157,800 flows - Intensive)",
                ],
                index=0,
            )

    df_loaded = None
    dataset_label = ""
    is_synthetic_flag = False

    if "Real Edge-IIoTset" in data_source:
        if not REAL_EDGE_IIOTSET_CSV.exists():
            st.error(f"Real Edge-IIoTset dataset not found at expected path: `{REAL_EDGE_IIOTSET_CSV}`.")
        else:
            with st.spinner("Loading real Edge-IIoTset dataset from disk..."):
                if "15,000" in sample_mode:
                    df_full = pd.read_csv(REAL_EDGE_IIOTSET_CSV, low_memory=False)
                    df_loaded = pd.concat(
                        [group.sample(n=min(len(group), 1000), random_state=42) for _, group in df_full.groupby("Attack_type")],
                        ignore_index=True,
                    )
                elif "30,000" in sample_mode:
                    df_full = pd.read_csv(REAL_EDGE_IIOTSET_CSV, low_memory=False)
                    df_loaded = pd.concat(
                        [group.sample(n=min(len(group), 2000), random_state=42) for _, group in df_full.groupby("Attack_type")],
                        ignore_index=True,
                    )
                else:
                    df_loaded = pd.read_csv(REAL_EDGE_IIOTSET_CSV, low_memory=False)

                dataset_label = f"ML-EdgeIIoT-dataset.csv (Real Edge-IIoTset - {len(df_loaded):,} rows)"
                is_synthetic_flag = False
    else:
        uploaded_file = st.file_uploader("Upload Custom IoT CSV", type=["csv"], key="ablation_uploader")
        if uploaded_file is not None:
            try:
                df_loaded = load_csv_dataset(uploaded_file, max_rows=50000)
                dataset_label = uploaded_file.name
                is_synthetic_flag = False
            except Exception as exc:
                st.error(f"Failed to load CSV: {exc}")

    if df_loaded is not None:
        initial_summary = inspect_dataset(df_loaded, dataset_name=dataset_label, is_synthetic_demo=is_synthetic_flag)
        all_cols = list(df_loaded.columns)
        default_target_idx = (
            all_cols.index(initial_summary.detected_label_col)
            if initial_summary.detected_label_col in all_cols
            else len(all_cols) - 1
        )
        selected_target_col = st.selectbox("Target Label Column", options=all_cols, index=default_target_idx)
        summary = inspect_dataset(
            df_loaded,
            dataset_name=dataset_label,
            target_col=selected_target_col,
            is_synthetic_demo=is_synthetic_flag,
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows / Columns", f"{summary.num_rows:,} / {summary.num_columns}")
        k2.metric("Classes", f"{summary.num_classes}")
        k3.metric("Protocol / Generic Feats", f"{len(summary.protocol_feature_columns)} / {len(summary.generic_feature_columns)}")
        k4.metric("NaN / Inf / Duplicates", f"{summary.missing_value_count} / {summary.infinite_value_count} / {summary.duplicate_row_count}")

        cfg1, cfg2, cfg3 = st.columns(3)
        with cfg1:
            k_features = st.slider(
                "Number of Selected Features (k)",
                min_value=1,
                max_value=max(1, len(summary.feature_columns)),
                value=min(DEFAULT_APAFC_K, max(1, len(summary.feature_columns))),
            )
        with cfg2:
            resampling_choice = st.selectbox("Training-Only Resampling", options=["smote_enn", "smote", "none"], index=0)
        with cfg3:
            compare_baseline = st.checkbox("Compare against Full-Feature Baseline", value=True)

        if st.button(" Run 61-to-k Exploratory Feature Selection & Training", disabled=bool(summary.validation_errors)):
            with st.spinner("Executing leak-free preprocessing, feature selection, and LightGBM evaluation..."):
                _, eval_res = train_and_evaluate_pipeline(
                    df=df_loaded,
                    target_col=selected_target_col,
                    dataset_name=dataset_label,
                    is_synthetic_dataset=is_synthetic_flag,
                    k_features=int(k_features),
                    resampling_method=resampling_choice,
                    compare_with_full_features=bool(compare_baseline),
                    artifact_path=MODEL_ARTIFACT_PATH,
                )
                st.success(
                    f"Exploratory run complete! Accuracy: {eval_res.accuracy * 100:.2f}% | "
                    f"Macro-F1: {eval_res.macro_f1:.4f} (Synthetic Demo: {eval_res.is_synthetic_dataset})"
                )

    if EVALUATION_RESULTS_PATH.exists():
        eval_dict = json.loads(EVALUATION_RESULTS_PATH.read_text(encoding="utf-8"))
        if eval_dict.get("baseline_comparison"):
            bc = eval_dict["baseline_comparison"]
            st.markdown("##### Exploratory Full-Feature vs. Top-k Comparison (Synthetic Demo Data)")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Model Variant": f"Full Candidate Set ({bc['full_feature_count']} features)",
                            "Accuracy (%)": round(bc["full_accuracy"] * 100, 2),
                            "Macro-F1": round(bc["full_macro_f1"], 4),
                            "p95 Latency (ms)": bc["full_p95_latency_ms"],
                            "Training Time (s)": bc["full_training_time_sec"],
                        },
                        {
                            "Model Variant": f"Top-{bc['reduced_feature_count']} APAFC Subset (-{bc['feature_reduction_pct']}%)",
                            "Accuracy (%)": round(bc["reduced_accuracy"] * 100, 2),
                            "Macro-F1": round(bc["reduced_macro_f1"], 4),
                            "p95 Latency (ms)": bc["reduced_p95_latency_ms"],
                            "Training Time (s)": bc["reduced_training_time_sec"],
                        },
                    ]
                ),
                width="stretch",
                hide_index=True,
            )


# ==============================================================================
# TAB 3: CLASSICAL ML BASELINE BENCHMARK (iot_ids.baselines)
# ==============================================================================
with tab_baselines:
    st.subheader(" Classical ML Baseline Benchmark (`iot_ids.baselines.benchmark_comparison`)")
    st.caption(
        "Executes `benchmark_baseline_models` from `iot_ids_framework/src/iot_ids/baselines/benchmark_comparison.py` "
        "(Decision Tree, Random Forest, KNN, Logistic Regression) measuring Accuracy, Macro-F1, p95 single-sample latency, and RSS memory."
    )
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        bench_task = st.selectbox("Classification Task", options=["multi", "binary"], index=0)
    with b_col2:
        bench_samples = st.number_input("Sample Size", min_value=200, max_value=10000, value=1200, step=200)

    if st.button("▶ Run Classical ML Baselines Benchmark"):
        if not SYNTHETIC_APAFC14_CSV.exists():
            generate_synthetic_apafc14_dataset(output_csv=SYNTHETIC_APAFC14_CSV, rows_per_class=80)
        with st.spinner("Training and profiling Decision Tree, Random Forest, KNN, and Logistic Regression..."):
            try:
                bench_df = benchmark_baseline_models(
                    dataset_csv=SYNTHETIC_APAFC14_CSV,
                    task=bench_task,
                    sample_size=int(bench_samples),
                )
                st.session_state["latest_baseline_bench"] = bench_df
            except Exception as exc:
                st.error(f"Baseline benchmark failed: {exc}")

    if "latest_baseline_bench" in st.session_state:
        bdf = st.session_state["latest_baseline_bench"]
        st.dataframe(bdf, width="stretch", hide_index=True)
        fig_b = px.bar(
            bdf,
            x="model",
            y=["accuracy", "p95_latency_ms"],
            barmode="group",
            title="Classical Baselines: Accuracy (%) vs. Single-Sample p95 Latency (ms) [Synthetic Smoke Data]",
            template="plotly_dark",
        )
        st.plotly_chart(fig_b, width="stretch")


# ==============================================================================
# TAB 4: GENUINE PCAP EVALUATION BENCHMARK (Edge-IIoTset 24-Capture Audit)
# ==============================================================================
with tab_pcap_bench:
    st.subheader(" Genuine PCAP Evaluation Benchmark (Edge-IIoTset 24-Capture Audit)")
    st.caption(
        "Complete ground-truth audit of `iot_ids_framework/artifacts/edge_ids.onnx` evaluated across all 24 genuine "
        "Edge-IIoTset PCAP files (14 attack, 10 normal). Preserves flow provenance, packet counting integrity, and denominator-aware metrics."
    )

    report_path = Path("results/genuine_pcap_evaluation_report.json")
    if report_path.exists():
        try:
            pcap_rep = json.loads(report_path.read_text(encoding="utf-8"))
            be = pcap_rep["binary_evaluation"]
            me = pcap_rep["multiclass_evaluation"]

            st.markdown(
                f"""
                <div style="background-color:#090d16; border:1px solid #1e293b; border-radius:4px; padding:12px 16px; margin-bottom:1rem;">
                    <strong>Audit Timestamp:</strong> <code>{pcap_rep.get('evaluation_timestamp', 'N/A')}</code> |
                    <strong>Evaluated Model:</strong> <code>{pcap_rep.get('model_evaluated', 'EdgeIDS ONNX')}</code> |
                    <strong>Captures:</strong> <code>{pcap_rep['captures_evaluated']} files ({pcap_rep['attack_captures_count']} Attack, {pcap_rep['normal_captures_count']} Normal)</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

            p1, p2, p3, p4, p5, p6 = st.columns(6)
            p1.metric("Packets Processed", f"{pcap_rep['total_packets_read']:,}")
            p2.metric("Flows Assembled", f"{pcap_rep['total_flows_assembled']:,}")
            p3.metric("Binary Precision", be["precision_percent"], delta=f"TP={be['TP']}, FP={be['FP']}")
            p4.metric("Binary Recall", be["recall_percent"], delta=f"FN={be['FN']}, TN={be['TN']}")
            p5.metric("Binary Accuracy", be["accuracy_percent"], delta=f"F1 = {be['f1_score']:.4f}")
            p6.metric("False Positive Rate", be["fpr_percent"], delta=f"Macro-F1 = {me['macro_f1']:.4f}")

            # Confusion Matrix Card
            st.markdown("#####  Binary Detection Matrix (Malicious Alerts vs. Benign Flows)")
            cm_col1, cm_col2 = st.columns([1.0, 1.4])
            with cm_col1:
                df_cm = pd.DataFrame(
                    [
                        {"Actual": "Malicious Flow", "Predicted Alert (Malicious)": be["TP"], "Predicted Benign": be["FN"]},
                        {"Actual": "Benign Flow", "Predicted Alert (Malicious)": be["FP"], "Predicted Benign": be["TN"]},
                    ]
                )
                st.dataframe(df_cm, width="stretch", hide_index=True)

            with cm_col2:
                fig_cm = px.imshow(
                    [[be["TP"], be["FN"]], [be["FP"], be["TN"]]],
                    labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Flows"),
                    x=["Malicious (Alert)", "Benign"],
                    y=["Malicious", "Benign"],
                    text_auto=True,
                    color_continuous_scale="Tealgrn",
                    template="plotly_dark",
                )
                fig_cm.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_cm, width="stretch")

            # Multiclass Breakdown
            st.markdown("#####  Multiclass Attack Breakdown (11 Edge-IIoTset Attack Categories)")
            if "per_class" in me:
                rows_mc = []
                for cls_name, c_data in me["per_class"].items():
                    rows_mc.append(
                        {
                            "Attack_Category": cls_name.upper(),
                            "Ground_Truth_Support": c_data.get("support", 0),
                            "Model_Predicted_Count": c_data.get("predicted", 0),
                            "Precision": f"{c_data['precision'] * 100:.1f}%" if isinstance(c_data["precision"], (int, float)) else str(c_data["precision"]),
                            "Recall": f"{c_data['recall'] * 100:.1f}%" if isinstance(c_data["recall"], (int, float)) else str(c_data["recall"]),
                            "F1_Score": f"{c_data['f1']:.4f}" if isinstance(c_data["f1"], (int, float)) else str(c_data["f1"]),
                        }
                    )
                st.dataframe(pd.DataFrame(rows_mc), width="stretch", hide_index=True)

            # Per-Capture Details
            with st.expander(" View Breakdown by Individual PCAP File (24 Captures)", expanded=False):
                if "per_capture_summary" in pcap_rep:
                    df_caps = pd.DataFrame(pcap_rep["per_capture_summary"])
                    st.dataframe(df_caps, width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Could not load PCAP benchmark report: {exc}")
    else:
        st.info("No genuine PCAP benchmark report found at `results/genuine_pcap_evaluation_report.json`.")

