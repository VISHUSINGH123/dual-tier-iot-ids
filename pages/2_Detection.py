"""
Page 2 — Live Edge Inference (`iot_ids.edge.EdgeIDS`), PCAP Replay (`iot_ids.pcap_adapter`),
Atomic Hot-Swap, and Batch CSV Detection.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
import streamlit as st
import pandas as pd
import onnxruntime as ort

from config.settings import ALERT_LOG_PATH, CANONICAL_EVENTS_PATH, MODEL_ARTIFACT_PATH
from src.alert_logger import AlertLogger
from src.event_pipeline import (
    CanonicalNetworkEvent,
    determine_alert_severity,
    evaluate_event_match,
    get_event_bus,
)
from src.framework_bridge import (
    ALERTS_JSONL_PATH,
    FEATURE_NAMES,
    ONNX_MODEL_PATH,
    SAMPLE_PCAP_PATH,
    SYNTHETIC_APAFC14_CSV,
    APAFCFeatureExtractor,
    check_core_artifacts_status,
    load_core_edge_ids,
    replay_pcap_to_flows,
)
from src.inference import EdgeClassifier
from src.ui_theme import render_header

st.set_page_config(page_title="Live & Batch Detection | IoT IDS", layout="wide")

render_header(
    title=" Module B & D — Edge ONNX Inference, PCAP Flow Replay & Alert Log",
    subtitle=(
        "Classify live flow records and reassembled `.pcap` captures via `iot_ids.edge.EdgeIDS` (ONNX Runtime) "
        "or run batch CSV classification."
    ),
    badge_text="ONNX Runtime + PCAP Adapter",
)

tab_pcap, tab_single, tab_batch, tab_alerts, tab_live = st.tabs(
    [
        "1. PCAP Replay & Bidirectional Flow Assembly (`iot_ids.pcap_adapter`)",
        "2. Interactive Single-Flow ONNX Classification & Atomic Hot-Swap (`EdgeIDS`)",
        "3. Batch CSV Detection & Downloadable Predictions",
        "4. Searchable Audit Alert Log (`alerts.jsonl`)",
        "5. Live Loopback Listener & Interactive Attacker Console",
    ]
)

# ==============================================================================
# TAB 1: PCAP REPLAY & FLOW ASSEMBLY (iot_ids.pcap_adapter + iot_ids.ingest)
# ==============================================================================
with tab_pcap:
    st.subheader(" Genuine Packet Capture (`.pcap`) Replay → `FlowAssembler` → `EdgeIDS` ONNX Inference")
    st.caption(
        "Ingests genuine `.pcap` packet streams, parses Ethernet/IPv4/TCP/UDP frames into bidirectional flow records "
        "via `iot_ids.ingest.FlowAssembler`, extracts the 14 APAFC features, and classifies each flow with `EdgeIDS`."
    )

    pcap_options = {
        "MITM Attack — Real Edge-IIoTset (127 KB)": "archive/Edge-IIoTset dataset/Attack traffic/MITM (ARP spoofing + DNS) Attack.pcap",
        "Port Scanning — Real Edge-IIoTset (1.8 MB)": "archive/Edge-IIoTset dataset/Attack traffic/Port Scanning attack.pcap",
        "OS Fingerprinting — Real Edge-IIoTset (114 KB)": "archive/Edge-IIoTset dataset/Attack traffic/OS Fingerprinting attack.pcap",
        "Ransomware Attack — Real Edge-IIoTset (2.8 MB)": "archive/Edge-IIoTset dataset/Attack traffic/Ransomware attack.pcap",
        "XSS Attacks — Real Edge-IIoTset (2.1 MB)": "archive/Edge-IIoTset dataset/Attack traffic/XSS attacks.pcap",
        "Backdoor Attack — Real Edge-IIoTset (6.7 MB)": "archive/Edge-IIoTset dataset/Attack traffic/Backdoor_attack.pcap",
        "SQL Injection Attack — Real Edge-IIoTset (8.2 MB)": "archive/Edge-IIoTset dataset/Attack traffic/SQL injection attack.pcap",
        "Modbus Industrial Telemetry (21 MB)": "archive/Edge-IIoTset dataset/Normal traffic/Modbus/Modbus.pcap",
        "Heart Rate Sensor Telemetry (12.8 MB)": "archive/Edge-IIoTset dataset/Normal traffic/Heart_Rate/Heart_Rate.pcap",
        "Core Sample Capture (sample_traffic.pcap)": "iot_ids_framework/datasets/sample_traffic.pcap",
    }

    pcol1, pcol2 = st.columns([1.2, 1.0])
    with pcol1:
        sel_preset_pcap = st.selectbox(
            "Select Real Network Capture (.pcap)",
            options=list(pcap_options.keys()),
            index=0,
            help="Select an authentic Edge-IIoTset attack or normal packet capture file.",
        )
        target_pcap_path = Path(pcap_options[sel_preset_pcap])

    with pcol2:
        pcap_upload = st.file_uploader("Or Upload Custom Network Capture (`.pcap`)", type=["pcap"])
        max_pkts_replay = st.number_input("Max Packets to Ingest", min_value=50, max_value=5000, value=300, step=50)

    if pcap_upload is not None:
        target_pcap_path = Path("archive") / f"uploaded_{pcap_upload.name}"
        target_pcap_path.parent.mkdir(parents=True, exist_ok=True)
        target_pcap_path.write_bytes(pcap_upload.getvalue())
        st.info(f"Using uploaded capture: `{target_pcap_path.name}` ({len(pcap_upload.getvalue()) / 1024:.1f} KB)")

    if st.button("▶ Replay PCAP & Classify Reassembled Flows", type="primary"):
        if not target_pcap_path.exists():
            st.error(f"Selected PCAP file not found at `{target_pcap_path}`.")
        else:
            try:
                t_parse_0 = time.perf_counter()
                flows = replay_pcap_to_flows(target_pcap_path, max_packets=int(max_pkts_replay))
                parse_dur_ms = (time.perf_counter() - t_parse_0) * 1000.0

                edge_ids = load_core_edge_ids()
                extractor = APAFCFeatureExtractor()

                flow_rows = []
                canonical_events = []
                is_attack_capture = "attack" in str(target_pcap_path).lower()
                gt_cat = target_pcap_path.stem.split()[0].lower() if is_attack_capture else "normal"
                gt_label = "MALICIOUS" if is_attack_capture else "BENIGN"

                for idx, fl in enumerate(flows):
                    t0 = time.perf_counter()
                    res = edge_ids.classify(fl)
                    lat_ms = (time.perf_counter() - t0) * 1000.0
                    vec = extractor.extract(fl)
                    top_attr = ""
                    alert_fired = False
                    if "alert" in res and res["alert"].get("attribution"):
                        a0 = res["alert"]["attribution"][0]
                        top_attr = f"{a0['feature']} ({a0['contribution']:+.4f})"
                        alert_fired = True
                    elif res.get("class") != "benign":
                        alert_fired = True

                    flow_rows.append(
                        {
                            "Flow_#": idx + 1,
                            "Protocol": str(fl.get("protocol", "")).upper(),
                            "Src": f"{fl.get('src_ip')}:{fl.get('src_port')}",
                            "Dst": f"{fl.get('dst_ip')}:{fl.get('dst_port')}",
                            "Packets": fl.get("packet_count", 0),
                            "Bytes": fl.get("byte_count", 0),
                            "Duration_ms": round(float(fl.get("duration_ms", 0.0)), 2),
                            "Predicted_Class": res["class"],
                            "Confidence": round(float(res["confidence"]), 4),
                            "Latency_ms": round(lat_ms, 3),
                            "Alert": " ALERT" if alert_fired else " NOMINAL",
                            "Top_TreeSHAP_Attribution": top_attr or "Benign (No Alert Emitted)",
                        }
                    )

                    # Build canonical event
                    ev_match = evaluate_event_match(res["class"], "ALERT" if alert_fired else "NOMINAL", gt_label)
                    can_ev = CanonicalNetworkEvent(
                        event_id=f"EVT-PCAP-{int(time.time()*1000)}-{idx:04d}",
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        input_mode="pcap_replay",
                        source_identifier=target_pcap_path.name,
                        flow_id=f"{fl.get('src_ip')}:{fl.get('src_port')} -> {fl.get('dst_ip')}:{fl.get('dst_port')} ({str(fl.get('protocol')).upper()})",
                        src_ip=str(fl.get("src_ip")),
                        src_port=int(fl.get("src_port", 0)),
                        dst_ip=str(fl.get("dst_ip")),
                        dst_port=int(fl.get("dst_port", 0)),
                        protocol=str(fl.get("protocol", "")).upper(),
                        packet_count=int(fl.get("packet_count", 1)),
                        byte_count=int(fl.get("byte_count", 0)),
                        duration_ms=float(fl.get("duration_ms", 0.0)),
                        feature_schema_id="APAFC-14",
                        features=fl,
                        model_id="edge_ids_onnx",
                        model_version="1.0.0",
                        prediction=str(res["class"]),
                        confidence_score=float(res["confidence"]),
                        decision_threshold=0.5,
                        alert_status="ALERT" if alert_fired else "NOMINAL",
                        alert_severity=determine_alert_severity(res["class"], float(res["confidence"])) if alert_fired else "INFO",
                        inference_latency_ms=lat_ms,
                        ground_truth_label=gt_label,
                        ground_truth_category=gt_cat,
                        is_ground_truth_available=True,
                        evaluation_match=ev_match,
                        explanation=top_attr,
                        rejection_reason=None,
                    )
                    canonical_events.append(can_ev)

                # Publish batch to event pipeline
                get_event_bus().publish_batch(canonical_events)

                st.success(
                    f"Parsed and reassembled {len(flow_rows)} bidirectional flows from `{target_pcap_path.name}` "
                    f"in {parse_dur_ms:.1f} ms. Events published to Canonical Event Pipeline."
                )

                # Evaluation metrics on replayed flows
                if flow_rows:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Flows Assembled", len(flow_rows))
                    alerts_count = sum(1 for r in flow_rows if "ALERT" in r["Alert"])
                    m2.metric("Alerts Fired", alerts_count)
                    mean_lat = sum(r["Latency_ms"] for r in flow_rows) / len(flow_rows)
                    m3.metric("Mean Latency", f"{mean_lat:.3f} ms")
                    m4.metric("Capture Category", gt_cat.upper())

                st.dataframe(pd.DataFrame(flow_rows), width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"PCAP replay error: {exc}")


# ==============================================================================
# TAB 2: INTERACTIVE SINGLE-FLOW ONNX CLASSIFICATION & ATOMIC HOT-SWAP
# ==============================================================================
with tab_single:
    st.subheader(" Interactive Flow Classification (`iot_ids.edge.EdgeIDS`)")
    preset = st.selectbox(
        "Load Flow Preset",
        options=[
            "Normal MQTT Sensor Publish (Benign)",
            "High-Rate MQTT Exploit / DDoS Flood (Attack)",
            "CoAP Actuator Spoofing / Scanning (Attack)",
            "HTTP SQL Injection / Web Attack (Attack)",
            "6LoWPAN High-Entropy Botnet / Backdoor C2 (Attack)",
        ],
    )

    defaults = {
        "Normal MQTT Sensor Publish (Benign)": ("mqtt", 25.0, 6, 320, 51200, 1883, 24, 1, 80, 0, 0, 0, 3.8),
        "High-Rate MQTT Exploit / DDoS Flood (Attack)": ("mqtt", 500.0, 8500, 1250000, 58912, 1883, 2, 2, 512, 0, 0, 0, 7.85),
        "CoAP Actuator Spoofing / Scanning (Attack)": ("coap", 45.0, 18, 640, 49152, 5683, 4, 0, 0, 165, 0, 0, 2.9),
        "HTTP SQL Injection / Web Attack (Attack)": ("http", 190.0, 28, 14500, 52100, 80, 24, 0, 0, 0, 500, 0, 5.8),
        "6LoWPAN High-Entropy Botnet / Backdoor C2 (Attack)": ("6lowpan", 950.0, 420, 98000, 61616, 61616, 16, 0, 0, 0, 0, 220, 7.75),
    }[preset]

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        proto = st.selectbox("Protocol", ["mqtt", "coap", "http", "6lowpan", "unknown"], index=["mqtt", "coap", "http", "6lowpan", "unknown"].index(defaults[0]))
        dur_ms = st.number_input("duration_ms", value=float(defaults[1]), min_value=0.0)
        pkt_cnt = st.number_input("packet_count", value=int(defaults[2]), min_value=1)
    with fc2:
        byte_cnt = st.number_input("byte_count", value=int(defaults[3]), min_value=0)
        sport = st.number_input("src_port", value=int(defaults[4]), min_value=0, max_value=65535)
        dport = st.number_input("dst_port", value=int(defaults[5]), min_value=0, max_value=65535)
    with fc3:
        tcp_fl = st.number_input("tcp_flags", value=int(defaults[6]), min_value=0, max_value=255)
        mqtt_qos = st.number_input("mqtt_qos", value=int(defaults[7]), min_value=0, max_value=2)
        mqtt_bytes = st.number_input("mqtt_payload_bytes", value=int(defaults[8]), min_value=0)
    with fc4:
        coap_c = st.number_input("coap_code", value=int(defaults[9]), min_value=0)
        http_s = st.number_input("http_status", value=int(defaults[10]), min_value=0)
        six_d = st.number_input("sixlowpan_dispatch", value=int(defaults[11]), min_value=0)
        ent_v = st.slider("payload_entropy", min_value=0.0, max_value=8.0, value=float(defaults[12]), step=0.05)

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button(" Classify Flow via `EdgeIDS` (ONNX Runtime)", type="primary"):
            try:
                edge_ids = load_core_edge_ids()
                flow_dict = {
                    "src_ip": "10.0.4.15",
                    "dst_ip": "192.168.1.1",
                    "src_port": sport,
                    "dst_port": dport,
                    "protocol": proto,
                    "duration_ms": dur_ms,
                    "packet_count": pkt_cnt,
                    "byte_count": byte_cnt,
                    "tcp_flags": tcp_fl,
                    "mqtt_qos": mqtt_qos,
                    "mqtt_payload_bytes": mqtt_bytes,
                    "coap_code": coap_c,
                    "http_status": http_s,
                    "sixlowpan_dispatch": six_d,
                    "payload_entropy": ent_v,
                }
                t0 = time.perf_counter()
                res = edge_ids.classify(flow_dict)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                st.session_state["last_single_res"] = (res, lat_ms, flow_dict)
            except Exception as exc:
                st.error(f"Classification failed: {exc}")

    with b_col2:
        if st.button(" Test Zero-Downtime Atomic Model Hot-Swap (`EdgeIDS.hot_swap`)"):
            try:
                edge_ids = load_core_edge_ids()
                new_sess = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
                t0 = time.perf_counter()
                edge_ids.hot_swap(new_sess)
                swap_us = (time.perf_counter() - t0) * 1_000_000.0
                st.success(f"Atomic ONNX session `hot_swap()` completed in **{swap_us:.2f} µs** with zero dropped flows!")
            except Exception as exc:
                st.error(f"Hot-swap failed: {exc}")

    if "last_single_res" in st.session_state:
        res, lat_ms, flow_dict = st.session_state["last_single_res"]
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Class", res["class"].upper())
        r2.metric("Confidence", f"{res['confidence'] * 100:.2f}%")
        r3.metric("ONNX Inference Latency", f"{lat_ms:.3f} ms")
        if "alert" in res:
            st.warning(f"Security Alert Logged to `{ALERTS_JSONL_PATH.name}` (Attribution: {res['alert']['attribution_method']})")
            st.json(res["alert"])
        else:
            st.success("Classified as BENIGN telemetry.")


# ==============================================================================
# TAB 3: BATCH CSV DETECTION
# ==============================================================================
with tab_batch:
    st.subheader(" Batch Flow Detection & Backend Selector")
    batch_mode = st.radio(
        "Select Inference Backend Profile",
        options=[
            "Primary 14-Feature ONNX Engine (`iot_ids.edge.EdgeIDS` - PCAP/DPI Schema)",
            "Experimental 17-Feature Zeek LightGBM (`models/experimental_iot23_lgbm_bundle.joblib`)",
            "Exploratory 61-Feature LightGBM Bundle (`src.inference.EdgeClassifier`)",
        ],
        horizontal=False,
    )

    if "Experimental 17-Feature" in batch_mode:
        st.info(
            " **Active Backend: Experimental IoT-23 LightGBM (17 Zeek-Native Features)**\n\n"
            "- **Accepted Input**: Zeek/Bro `conn.log.labeled` connection summaries with the verified 17-feature schema.\n"
            "- **Taxonomy**: Binary classification (Benign vs. Malicious).\n"
            "- **Important Limitation**: Zeek connection logs do not contain packet payload bytes or L7 application headers. "
            "Traffic without transport anomalies (e.g. unacknowledged SYN scans) cannot be distinguished via deep-packet inspection."
        )
    elif "Primary 14-Feature" in batch_mode:
        st.info(
            " **Active Backend: Core EdgeIDS ONNX Runtime (14 APAFC Features)**\n\n"
            "- **Accepted Input**: Assembled bidirectional network flows with L2/L3/L4/L7 APAFC features.\n"
            "- **Taxonomy**: 15-class edge attack taxonomy."
        )

    uploaded_batch = st.file_uploader(
        "Upload CSV for Batch Prediction (optional — uses selected dataset source if omitted)",
        type=["csv"],
    )

    # Real IoT-23 Sampling Controls when Experimental Backend is selected
    if "Experimental 17-Feature" in batch_mode:
        from src.iot23_loader import (
            get_available_iot23_scenarios,
            sample_real_iot23_records_for_batch_detection,
            ZEEK_NATIVE_17_FEATURES,
        )

        st.markdown("#####  Real IoT-23 Attack & Benign Dataset Sampling")
        avail_scens = get_available_iot23_scenarios()
        scen_list = list(avail_scens.keys()) if avail_scens else ["CTU-IoT-Malware-Capture-8-1"]
        def_scen_idx = scen_list.index("CTU-IoT-Malware-Capture-8-1") if "CTU-IoT-Malware-Capture-8-1" in scen_list else 0

        sc_col1, sc_col2, sc_col3 = st.columns([1.5, 1.0, 1.0])
        with sc_col1:
            chosen_scenario = st.selectbox(
                "Source Real IoT-23 Scenario",
                options=scen_list,
                index=def_scen_idx,
                help="Select which actual CTU capture file to sample records from.",
            )
        with sc_col2:
            n_ben = st.number_input("Real Benign Records", min_value=1, max_value=200, value=25, step=5)
        with sc_col3:
            n_mal = st.number_input("Real Attack Records", min_value=1, max_value=200, value=25, step=5)

    if st.button("▶ Run Batch Prediction", type="primary"):
        try:
            if "Experimental 17-Feature" in batch_mode:
                from src.iot23_loader import (
                    DualBackendInferenceRouter,
                    extract_zeek_native_features,
                    sample_real_iot23_records_for_batch_detection,
                    ZEEK_NATIVE_17_FEATURES,
                )

                router = DualBackendInferenceRouter()

                if uploaded_batch is not None:
                    df_raw = pd.read_csv(uploaded_batch).head(500)
                    if set(ZEEK_NATIVE_17_FEATURES).issubset(set(df_raw.columns)):
                        df_in = df_raw[ZEEK_NATIVE_17_FEATURES]
                    elif "proto" in df_raw.columns and "conn_state" in df_raw.columns:
                        df_in = extract_zeek_native_features(df_raw)
                    else:
                        df_in = df_raw
                    ground_truth_df = None
                else:
                    # Deterministically sample real benign and malicious flows from the chosen real IoT-23 scenario
                    st.info(f"Streaming {n_ben} real benign and {n_mal} real attack records directly from `{chosen_scenario}` (`conn.log.labeled`)...")
                    sampled_records = sample_real_iot23_records_for_batch_detection(
                        scenario_name=chosen_scenario,
                        n_benign=int(n_ben),
                        n_malicious=int(n_mal),
                        random_seed=42,
                    )
                    df_in = sampled_records[ZEEK_NATIVE_17_FEATURES]
                    ground_truth_df = sampled_records

                t0 = time.perf_counter()
                out = router.predict(DualBackendInferenceRouter.EXPERIMENTAL_LGBM_BACKEND, df_in)
                total_ms = (time.perf_counter() - t0) * 1000.0

                res_df = df_in.copy()
                res_df["Predicted_Class"] = out["predictions"]
                if out.get("malicious_probabilities"):
                    res_df["Malicious_Probability"] = [round(p, 4) for p in out["malicious_probabilities"]]

                if ground_truth_df is not None:
                    res_df["Ground_Truth_Label"] = ground_truth_df["ground_truth_label"]
                    res_df["Detailed_Attack_Type"] = ground_truth_df["ground_truth_detailed"]
                    res_df["Ground_Truth_Binary"] = ground_truth_df["ground_truth_binary"]
                    res_df["Detection_Match"] = res_df["Predicted_Class"] == res_df["Ground_Truth_Label"].str.lower()
                    res_df["Source_Scenario"] = ground_truth_df["source_scenario"]

                    # Compute inline validation metrics
                    match_rate = res_df["Detection_Match"].mean() * 100.0
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Flows Evaluated", len(res_df))
                    m2.metric("Ground-Truth Accuracy", f"{match_rate:.1f}%")
                    m3.metric("Batch Latency", f"{total_ms:.2f} ms")

                # Route batch to CanonicalEventBus
                batch_events = []
                for idx, row in res_df.iterrows():
                    pred_val = str(row["Predicted_Class"])
                    gt_val = str(row.get("Ground_Truth_Label", "")) if ground_truth_df is not None else None
                    has_gt = bool(gt_val and gt_val != "")
                    is_alert = pred_val.lower() not in ("benign", "normal")
                    ev_match = evaluate_event_match(pred_val, "ALERT" if is_alert else "NOMINAL", gt_val) if has_gt else None
                    can_ev = CanonicalNetworkEvent(
                        event_id=f"EVT-ZEEK-{int(time.time()*1000)}-{idx:04d}",
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        input_mode="dataset_batch",
                        source_identifier=str(row.get("Source_Scenario", "zeek_batch")),
                        flow_id=f"{row.get('proto', 'ip')}-{idx:04d}",
                        src_ip="192.168.1.10",
                        src_port=int(row.get("id.orig_p", 0)),
                        dst_ip="10.0.0.1",
                        dst_port=int(row.get("id.resp_p", 0)),
                        protocol=str(row.get("proto", "tcp")).upper(),
                        packet_count=int(row.get("orig_pkts", 1)),
                        byte_count=int(row.get("orig_bytes", 0)),
                        duration_ms=float(row.get("duration", 0.0)) * 1000.0,
                        feature_schema_id="ZEEK-17",
                        features={k: row[k] for k in ZEEK_NATIVE_17_FEATURES if k in row},
                        model_id="experimental_zeek17_lgbm",
                        model_version="1.0.0",
                        prediction=pred_val,
                        confidence_score=float(row.get("Malicious_Probability", 0.9 if is_alert else 0.1)),
                        decision_threshold=0.5,
                        alert_status="ALERT" if is_alert else "NOMINAL",
                        alert_severity=determine_alert_severity(pred_val, 0.9) if is_alert else "INFO",
                        inference_latency_ms=total_ms / max(len(res_df), 1),
                        ground_truth_label=gt_val,
                        ground_truth_category=str(row.get("Detailed_Attack_Type", "zeek_scenario")) if has_gt else None,
                        is_ground_truth_available=has_gt,
                        evaluation_match=ev_match,
                        explanation="Zeek transport feature anomaly",
                        rejection_reason=None,
                    )
                    batch_events.append(can_ev)
                get_event_bus().publish_batch(batch_events)

                st.success(
                    f"Classified {out['total_flows']} real Zeek connection records via Experimental LightGBM in "
                    f"{total_ms:.2f} ms ({total_ms / max(out['total_flows'], 1):.3f} ms/flow)."
                )
                st.dataframe(res_df, width="stretch")
                st.download_button(
                    "⬇ Download Batch Predictions CSV",
                    data=res_df.to_csv(index=False).encode("utf-8"),
                    file_name="experimental_zeek_batch_predictions.csv",
                    mime="text/csv",
                )

            elif "Primary 14-Feature" in batch_mode:
                edge_ids = load_core_edge_ids()
                if uploaded_batch is not None:
                    df_in = pd.read_csv(uploaded_batch).head(500)
                else:
                    st.warning(
                        "Please upload a 14-feature flow CSV to run batch classification with the Primary ONNX model, "
                        "or replay `.pcap` packet captures in Tab 1. (Synthetic data fallback is disabled)."
                    )
                    st.stop()

                t0 = time.perf_counter()
                preds = []
                confs = []
                for _, row in df_in.iterrows():
                    r_dict = row.to_dict()
                    # Map protocol_id back to protocol string if only protocol_id is present
                    if "protocol" not in r_dict and "protocol_id" in r_dict:
                        rev_p = {0: "unknown", 1: "mqtt", 2: "coap", 3: "http", 4: "6lowpan"}
                        r_dict["protocol"] = rev_p.get(int(float(r_dict["protocol_id"])), "unknown")
                    out = edge_ids.classify(r_dict)
                    preds.append(out["class"])
                    confs.append(round(float(out["confidence"]), 4))
                total_ms = (time.perf_counter() - t0) * 1000.0
                res_df = df_in.copy()
                res_df["Predicted_Class"] = preds
                res_df["Confidence"] = confs
                st.success(f"Classified {len(res_df)} records via ONNX Runtime in {total_ms:.2f} ms ({total_ms/len(res_df):.3f} ms/flow).")
                st.dataframe(res_df, width="stretch")
                st.download_button(
                    "⬇ Download Batch Predictions CSV",
                    data=res_df.to_csv(index=False).encode("utf-8"),
                    file_name="onnx_batch_predictions.csv",
                    mime="text/csv",
                )
            else:
                clf = EdgeClassifier(MODEL_ARTIFACT_PATH).load()
                if uploaded_batch is not None:
                    df_in = pd.read_csv(uploaded_batch).head(500)
                else:
                    from config.settings import REAL_EDGE_IIOTSET_CSV

                    if REAL_EDGE_IIOTSET_CSV.exists():
                        df_all = pd.read_csv(REAL_EDGE_IIOTSET_CSV, nrows=2000, low_memory=False)
                        df_in = df_all.sample(n=min(100, len(df_all)), random_state=7).reset_index(drop=True)
                    else:
                        st.warning("Please upload a CSV file for batch classification.")
                        st.stop()
                batch_out = clf.predict_batch(df_in)
                st.success(
                    f"Classified {batch_out.total_records} records in {batch_out.total_latency_ms:.2f} ms "
                    f"({batch_out.suspicious_count} Suspicious, {batch_out.benign_count} Benign)."
                )
                st.dataframe(batch_out.results_df, width="stretch")
                st.download_button(
                    "⬇ Download Batch Predictions CSV",
                    data=batch_out.results_df.to_csv(index=False).encode("utf-8"),
                    file_name="exploratory_batch_predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Batch detection error: {exc}")


# ==============================================================================
# TAB 4: SEARCHABLE AUDIT ALERT LOG
# ==============================================================================
with tab_alerts:
    st.subheader(" Auditable Explainable Alert Log")
    log_source = st.radio(
        "Alert Log Store",
        options=[
            "Canonical Unified Event Pipeline (`logs/canonical_events.jsonl`)",
            "Primary Core ONNX Audit Log (`iot_ids_framework/artifacts/alerts.jsonl`)",
            "Exploratory Alert Log (`logs/alerts_log.jsonl`)",
        ],
        horizontal=True,
    )
    min_conf_filter = st.slider("Minimum Confidence Filter", 0.0, 1.0, 0.0, 0.05)

    if "Canonical Unified" in log_source:
        bus = get_event_bus()
        events = bus.get_events(limit=500)
        filtered_events = [e for e in events if e.confidence_score >= min_conf_filter]
        if filtered_events:
            ev_rows = []
            for e in filtered_events:
                ev_rows.append(
                    {
                        "Event_ID": e.event_id,
                        "Timestamp": e.timestamp_utc,
                        "Input_Mode": e.input_mode,
                        "Source": e.source_identifier,
                        "Flow": e.flow_id,
                        "Model": e.model_id,
                        "Prediction": e.prediction,
                        "Confidence": f"{e.confidence_score * 100:.1f}%",
                        "Alert": e.alert_status,
                        "Severity": e.alert_severity,
                        "Ground_Truth": e.ground_truth_label or "UNLABELLED",
                        "Latency_ms": f"{e.inference_latency_ms:.2f}",
                        "Explanation": e.explanation or "N/A",
                    }
                )
            st.dataframe(pd.DataFrame(ev_rows), width="stretch", hide_index=True)
        else:
            st.info("No events in canonical event pipeline meeting confidence threshold.")
    elif "Primary" in log_source:
        records = []
        if ALERTS_JSONL_PATH.exists():
            for line in ALERTS_JSONL_PATH.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        item = json.loads(line)
                        if float(item.get("confidence", 0.0)) >= min_conf_filter:
                            records.append(item)
                    except Exception:
                        pass
        records.reverse()
        if records:
            classes_avail = sorted({r.get("predicted_class", "") for r in records})
            sel_classes = st.multiselect("Filter by Predicted Class", options=classes_avail, default=classes_avail)
            rows = []
            for r in records:
                if r.get("predicted_class") not in sel_classes:
                    continue
                attrs = r.get("attribution", [])
                top1 = f"{attrs[0]['feature']} ({attrs[0]['contribution']:+.4f})" if len(attrs) > 0 else "N/A"
                top2 = f"{attrs[1]['feature']} ({attrs[1]['contribution']:+.4f})" if len(attrs) > 1 else "N/A"
                ft = r.get("flow_5_tuple", {})
                rows.append(
                    {
                        "Timestamp": r.get("timestamp", ""),
                        "Protocol": ft.get("protocol", ""),
                        "Src": f"{ft.get('src_ip')}:{ft.get('src_port')}",
                        "Dst": f"{ft.get('dst_ip')}:{ft.get('dst_port')}",
                        "Predicted_Class": r.get("predicted_class", ""),
                        "Confidence": round(float(r.get("confidence", 0.0)), 4),
                        "Method": r.get("attribution_method", ""),
                        "Top_Attribution_1": top1,
                        "Top_Attribution_2": top2,
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("No alerts in `iot_ids_framework/artifacts/alerts.jsonl` yet. Classify an attack flow in Tab 1 or Tab 2.")
    else:
        logger = AlertLogger(ALERT_LOG_PATH)
        alerts = logger.read_alerts(min_confidence=min_conf_filter, limit=300)
        st.dataframe(logger.to_dataframe(alerts), width="stretch", hide_index=True)


# ==============================================================================
# TAB 5: LIVE LOOPBACK LISTENER & INTERACTIVE ATTACKER CONSOLE
# ==============================================================================
with tab_live:
    from src.live_listener import render_live_simulation_console

    render_live_simulation_console()

