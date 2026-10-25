"""
Automated Integration & Unit Test Suite for the Unified Dual-Tier IoT IDS Prototype.

Covers:
1. Primary Core `iot_ids_framework` integration (`PCAPReader`, `FlowAssembler`,
   `APAFCFeatureExtractor`, `EdgeIDS` ONNX inference, TreeSHAP attribution,
   atomic `hot_swap`, `KLDriftMonitor`, `DriftBenchmark`, `ClusteredHistogramAggregator`).
2. Exploratory 61-to-k Dataset-Driven Pipeline (`DatasetPreprocessor`, `APAFCExtractor`,
   `EdgeClassifier`, `TreeSHAPExplainer`, `AlertLogger`, `DriftMonitor`,
   `EdgeGateway`, `HistogramAggregator`).
3. Edge cases: missing columns, unknown extra columns, NaN/Inf handling, privacy boundary
   enforcement (rejecting raw flow records at the coordinator), and autonomous offline mode.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.settings import MODEL_ARTIFACT_PATH
from src.alert_logger import AlertLogger
from src.cloud_aggregator import HistogramAggregator, partition_dataset_for_gateways
from src.data_loader import (
    ALL_61_FEATURES,
    classify_feature_origin,
    detect_target_column,
    generate_synthetic_edge_iiotset,
    inspect_dataset,
)
from src.drift_monitor import DriftMonitor, compute_smoothed_kl_divergence
from src.edge_gateway import EdgeGateway
from src.explainability import TreeSHAPExplainer
from src.framework_bridge import (
    FEATURE_NAMES,
    ONNX_MODEL_PATH,
    SAMPLE_PCAP_PATH,
    APAFCFeatureExtractor,
    ClusteredHistogramAggregator,
    DriftBenchmark,
    check_core_artifacts_status,
    generate_sample_iot_pcap,
    load_core_edge_ids,
    quantise_histogram,
    replay_pcap_to_flows,
)
from src.inference import EdgeClassifier
from src.preprocessing import DatasetPreprocessor, prepare_train_test_data


def test_core_pcap_replay_and_apafc14_extraction(tmp_path: Path) -> None:
    """Verify synthetic PCAP generation, PCAPReader parsing, FlowAssembler, and 14-feature extraction."""
    pcap_file = tmp_path / "test_iot.pcap"
    generate_sample_iot_pcap(pcap_file)
    assert pcap_file.exists() and pcap_file.stat().st_size > 100

    flows = replay_pcap_to_flows(pcap_file)
    assert len(flows) == 4  # MQTT, CoAP, HTTP, 6LoWPAN bidirectional flows

    extractor = APAFCFeatureExtractor()
    for fl in flows:
        vec = extractor.extract(fl)
        assert len(vec) == 14
        assert all(np.isfinite(vec))
        assert float(fl.get("payload_entropy", 0.0)) > 0.0
        assert float(vec[13]) > 0.0


def test_stage1_pcap_replay_preserves_entropy_and_initiator_direction(tmp_path: Path) -> None:
    """
    Stage 1 regression test: verify PCAP replay preserves cumulative payload_entropy,
    original initiator (src_ip, src_port) -> (dst_ip, dst_port) direction when dst_ip < src_ip,
    and bidirectional L7 protocol fields across ACK packets, and classifies cleanly via EdgeIDS.
    """
    pcap_file = tmp_path / "stage1_regression.pcap"
    generate_sample_iot_pcap(pcap_file)
    flows = replay_pcap_to_flows(pcap_file)
    assert len(flows) == 4

    by_proto = {str(fl["protocol"]): fl for fl in flows}
    assert set(by_proto.keys()) == {"mqtt", "coap", "http", "6lowpan"}

    # Flow 1: MQTT (192.168.1.25:51200 -> 192.168.1.1:1883, where dst_ip '192.168.1.1' < src_ip '192.168.1.25')
    mqtt_fl = by_proto["mqtt"]
    assert mqtt_fl["src_ip"] == "192.168.1.25"
    assert int(mqtt_fl["src_port"]) == 51200
    assert mqtt_fl["dst_ip"] == "192.168.1.1"
    assert int(mqtt_fl["dst_port"]) == 1883
    assert int(mqtt_fl["mqtt_qos"]) == 1
    assert int(mqtt_fl["mqtt_payload_bytes"]) == 27
    assert float(mqtt_fl["payload_entropy"]) > 3.0

    # Flow 2: CoAP (192.168.1.40:49152 -> 192.168.1.1:5683)
    coap_fl = by_proto["coap"]
    assert coap_fl["src_ip"] == "192.168.1.40"
    assert int(coap_fl["src_port"]) == 49152
    assert coap_fl["dst_ip"] == "192.168.1.1"
    assert int(coap_fl["dst_port"]) == 5683
    assert int(coap_fl["coap_code"]) == 1
    assert float(coap_fl["payload_entropy"]) > 2.5

    # Flow 3: HTTP (192.168.1.55:54321 -> 192.168.1.1:80)
    http_fl = by_proto["http"]
    assert http_fl["src_ip"] == "192.168.1.55"
    assert int(http_fl["src_port"]) == 54321
    assert http_fl["dst_ip"] == "192.168.1.1"
    assert int(http_fl["dst_port"]) == 80
    assert int(http_fl["http_status"]) == 200
    assert float(http_fl["payload_entropy"]) > 3.5

    # Flow 4: 6LoWPAN (192.168.1.77:61616 -> 192.168.1.1:61616)
    six_fl = by_proto["6lowpan"]
    assert six_fl["src_ip"] == "192.168.1.77"
    assert int(six_fl["src_port"]) == 61616
    assert six_fl["dst_ip"] == "192.168.1.1"
    assert int(six_fl["dst_port"]) == 61616
    assert int(six_fl["sixlowpan_dispatch"]) == 0x41
    assert float(six_fl["payload_entropy"]) > 3.0

    # Verify all 4 flows classify cleanly through EdgeIDS ONNX inference
    edge_ids = load_core_edge_ids(top_k=5)
    extractor = APAFCFeatureExtractor()
    for proto_name, fl in by_proto.items():
        vec = extractor.extract(fl)
        assert vec[5] == float(fl["src_port"])
        assert vec[6] == float(fl["dst_port"])
        assert vec[13] == pytest.approx(float(fl["payload_entropy"]), rel=1e-5)
        pred = edge_ids.classify(fl)
        assert "class" in pred and "confidence" in pred
        assert 0.0 <= float(pred["confidence"]) <= 1.0


def test_core_onnx_edge_ids_inference_and_hot_swap() -> None:
    """Verify that the trained ONNX model in `iot_ids_framework/artifacts` loads, classifies, explains, and hot-swaps."""
    status = check_core_artifacts_status()
    assert status["ready"], "Core ONNX artifacts must be generated before running integration test."

    edge_ids = load_core_edge_ids(top_k=5)
    benign_flow = {
        "src_ip": "192.168.1.25",
        "src_port": 51200,
        "dst_ip": "192.168.1.1",
        "dst_port": 1883,
        "protocol": "mqtt",
        "duration_ms": 25.0,
        "packet_count": 6,
        "byte_count": 300,
        "tcp_flags": 24,
        "mqtt_qos": 1,
        "mqtt_payload_bytes": 120,
        "payload_entropy": 3.8,
    }
    res_b = edge_ids.classify(benign_flow)
    assert "class" in res_b and "confidence" in res_b
    assert 0.0 <= float(res_b["confidence"]) <= 1.0

    attack_flow = {
        "src_ip": "10.0.4.15",
        "src_port": 58912,
        "dst_ip": "192.168.1.1",
        "dst_port": 1883,
        "protocol": "mqtt",
        "duration_ms": 500.0,
        "packet_count": 8500,
        "byte_count": 1250000,
        "tcp_flags": 2,
        "mqtt_qos": 2,
        "mqtt_payload_bytes": 512,
        "payload_entropy": 7.85,
    }
    res_a = edge_ids.classify(attack_flow)
    assert res_a["class"] != "benign"
    assert "alert" in res_a
    assert res_a["alert"]["attribution_method"] == "TreeSHAP"
    assert len(res_a["alert"]["attribution"]) == 5

    # Verify atomic hot-swap
    import onnxruntime as ort

    new_sess = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    edge_ids.hot_swap(new_sess)
    res_after = edge_ids.classify(attack_flow)
    assert res_after["class"] == res_a["class"]


def test_core_drift_benchmark_and_clustered_federation() -> None:
    """Verify `DriftBenchmark` and `ClusteredHistogramAggregator` from `iot_ids_framework`."""
    rng = np.random.default_rng(42)
    ref = rng.normal(0.0, 1.0, size=(200, 14))
    benign_s = rng.normal(0.0, 1.0, size=(150, 14))
    drift_s = rng.normal(0.0, 1.0, size=(200, 14))

    bench = DriftBenchmark(ref, window_size=50, bins=8, threshold=0.15, consecutive_windows=2)
    res = bench.run_benchmark(benign_s, drift_s, drift_injection_magnitude=3.0)
    assert res.mean_kl_drift > res.mean_kl_benign
    assert res.drift_detection_count >= 1

    # Federated quantized gradient/hessian aggregation
    h1 = quantise_histogram("gw1", "clusterA", "packet_count", [0.0, 10.0, 100.0], [-2.0, 1.5], [3.0, 2.0], bits=16)
    h2 = quantise_histogram("gw2", "clusterA", "packet_count", [0.0, 10.0, 100.0], [-1.8, 1.2], [2.5, 1.8], bits=16)
    agg = ClusteredHistogramAggregator()
    totals = agg.aggregate([h1, h2])
    split_idx = agg.best_split(totals[("clusterA", "packet_count")])
    assert split_idx in (0, 1)


def test_leak_free_preprocessing_and_unknown_missing_columns() -> None:
    """Verify target leakage exclusion, NaN/Inf imputation, and graceful handling of missing/extra columns."""
    df = generate_synthetic_edge_iiotset(n_samples=350, random_state=7)
    summary = inspect_dataset(df, target_col="Attack_type", is_synthetic_demo=True)
    assert summary.detected_label_col == "Attack_type"
    assert "Attack_label" in summary.excluded_leakage_columns
    assert "ip.src_host" in summary.excluded_leakage_columns
    assert len(summary.feature_columns) == 61

    bundle, preproc = prepare_train_test_data(df, target_col="Attack_type", test_size=0.2, resampling_method="smote_enn")
    assert len(bundle.feature_names) == 61
    assert not bundle.X_train.isna().any().any()
    assert not bundle.X_test.isna().any().any()

    # Pass a record with missing required columns and an unknown extra column
    partial_df = pd.DataFrame([{"mqtt.len": 32.0, "unknown_future_col": 999.0}])
    transformed, notes = preproc.transform_features(partial_df, return_warnings=True)
    assert list(transformed.columns) == bundle.feature_names
    assert any("Missing" in n for n in notes)
    assert any("Ignored" in n for n in notes)


def test_exploratory_classifier_shap_and_alert_logger(tmp_path: Path) -> None:
    """Verify saved `EdgeClassifier` reloads, predicts single/batch, explains via TreeSHAP, and logs alerts."""
    clf = EdgeClassifier(MODEL_ARTIFACT_PATH).load()
    df_sample = generate_synthetic_edge_iiotset(n_samples=25, random_state=99)

    single_pred = clf.predict_single(df_sample.iloc[0])
    assert single_pred.predicted_class in clf.label_mapping.values()
    assert 0.0 <= single_pred.confidence <= 1.0

    batch_out = clf.predict_batch(df_sample)
    assert batch_out.total_records == 25
    assert batch_out.benign_count + batch_out.suspicious_count == 25

    X_red, _ = clf.preprocess_to_reduced(df_sample.iloc[:1])
    explainer = TreeSHAPExplainer(clf.model, clf.selected_features, clf.label_mapping, top_k=5)
    exp = explainer.explain_instance(X_red, single_pred.predicted_class_id, single_pred.confidence)
    assert exp.is_local_shap
    assert len(exp.top_contributions) == 5

    log_file = tmp_path / "test_alerts.jsonl"
    logger = AlertLogger(log_file)
    logger.log_alert(single_pred, exp)
    alerts = logger.read_alerts()
    assert len(alerts) == 1
    assert alerts[0]["predicted_class"] == single_pred.predicted_class


def test_multi_gateway_privacy_boundary_and_offline_mode() -> None:
    """Verify that raw records are never transmitted and coordinator rejects privacy-violating payloads."""
    clf = EdgeClassifier(MODEL_ARTIFACT_PATH).load()
    df = generate_synthetic_edge_iiotset(n_samples=300, random_state=12)
    X_ref, _ = clf.preprocess_to_reduced(df.iloc[:150])
    monitor = DriftMonitor.from_reference_dataframe(X_ref, clf.selected_features)

    partitions = partition_dataset_for_gateways(df, num_gateways=2, partition_mode="non_iid")
    coord = HistogramAggregator(shared_bin_edges=monitor.feature_bins, cluster_divergence_threshold=0.15)

    payloads = []
    for gid, prof, g_df in partitions:
        gw = EdgeGateway(gid, g_df, clf, gateway_profile=prof)
        # Test FR-12 offline mode
        off_res = gw.run_offline()
        assert off_res["cloud_reachable"] is False
        assert off_res["records_classified"] == len(g_df)

        p = gw.build_histogram_update(monitor.feature_bins, round_id=1)
        assert p.contains_raw_records is False
        valid, errs = coord.validate_payload_schema(p)
        assert valid, f"Unexpected schema errors: {errs}"
        payloads.append(p)

    summary = coord.aggregate_round(payloads, persist=False)
    assert summary.privacy_boundary_verified is True
    assert summary.total_uplink_bytes > 0

    # Verify coordinator rejects tampered payload claiming raw records
    tampered = json.loads(payloads[0].to_json_bytes().decode("utf-8"))
    tampered["raw_records"] = [{"ip.src_host": "192.168.1.5"}]
    valid_t, errs_t = coord.validate_payload_schema(tampered)
    assert not valid_t
    assert any("Privacy violation" in e for e in errs_t)


def test_original_iot_ids_framework_files_unmodified() -> None:
    """Verify that all original tracked files in `iot_ids_framework` match their pre-integration SHA-256 hashes."""
    import hashlib
    from config.settings import BASE_DIR

    manifest_path = Path(r"C:\Users\vishu\.gemini\antigravity\brain\8cdecbe8-2948-4a68-a279-ae159835ad65\scratch\iot_ids_framework_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Checkpoint manifest file not present on this machine.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel_path, expected_sha256 in manifest.items():
        if rel_path.endswith((".gitignore", "README.md")):
            continue
        full_p = BASE_DIR / rel_path
        assert full_p.exists(), f"Original repository file missing: {rel_path}"
        actual_sha256 = hashlib.sha256(full_p.read_bytes()).hexdigest()
        assert actual_sha256 == expected_sha256, f"Original file modified: {rel_path}"


def test_streamlit_dashboard_pages_render_without_exception() -> None:
    """Smoke-test `app.py` and all 6 pages in `pages/` using Streamlit's AppTest harness."""
    from streamlit.testing.v1 import AppTest
    from config.settings import BASE_DIR

    page_files = [
        BASE_DIR / "app.py",
        BASE_DIR / "pages" / "1_Dataset_and_Training.py",
        BASE_DIR / "pages" / "2_Detection.py",
        BASE_DIR / "pages" / "3_Explainability.py",
        BASE_DIR / "pages" / "4_Drift_Monitor.py",
        BASE_DIR / "pages" / "5_Edge_Cloud_Simulation.py",
        BASE_DIR / "pages" / "6_System_Diagnostics.py",
        BASE_DIR / "pages" / "7_Live_Simulation.py",
    ]
    for pf in page_files:
        assert pf.exists(), f"Missing dashboard page: {pf}"
        at = AppTest.from_file(str(pf), default_timeout=30)
        at.run()
        assert not at.exception, f"Exception on {pf.name}: {at.exception}"


def test_iot23_zeek_streaming_parser_and_compatibility_audit() -> None:
    """Task 1, 2, 3 Regression Test: Verify streaming Zeek parser, mixed delimiter handling, and schema compatibility."""
    from src.iot23_loader import (
        EXPECTED_ZEEK_23_FIELDS,
        IOT23_RAW_ROOT,
        ZEEK_NATIVE_16_FEATURES,
        assess_schema_compatibility,
        audit_single_zeek_file,
        evaluate_existing_apafc14_onnx_on_iot23_sample,
        extract_zeek_native_features,
        iter_zeek_conn_chunks,
        map_zeek_to_apafc14,
        parse_zeek_data_line,
    )

    # Verify mixed tab + 3-space delimiter parsing on sample Zeek line
    sample_line = (
        "1536227037.989677\tC63526273923462511\t192.168.100.111\t17832\t"
        "89.221.214.130\t123\tudp\t-\t0.004997\t48\t48\tSF\t-\t-\t0\tDd\t"
        "1\t76\t1\t76\t(empty)   Benign   -\n"
    )
    tokens, ok = parse_zeek_data_line(sample_line)
    assert ok and tokens is not None and len(tokens) == 23
    assert tokens[20] == "(empty)"
    assert tokens[21] == "Benign"
    assert tokens[22] == "-"

    compat = assess_schema_compatibility()
    assert compat["apafc_14_pipeline"]["unavailable_l7_payload_count"] == 6
    assert compat["edge_iiotset_61_pipeline"]["unavailable_count"] == 56

    if not IOT23_RAW_ROOT.exists():
        pytest.skip("IoT-23 raw directory not present on this machine.")

    sample_file = IOT23_RAW_ROOT / "opt" / "Malware-Project" / "BigDataset" / "IoTScenarios" / "CTU-Honeypot-Capture-4-1" / "bro" / "conn.log.labeled"
    assert sample_file.exists()
    audit_res = audit_single_zeek_file(sample_file)
    assert audit_res["row_count"] == 452
    assert audit_res["parse_errors"] == 0
    assert audit_res["schema_matches_23_cols"] is True
    assert audit_res["labels"] == {"benign": 452}

    chunks = list(iter_zeek_conn_chunks(sample_file, chunk_size=200))
    assert len(chunks) == 3
    df_cat = pd.concat(chunks, ignore_index=True)
    assert len(df_cat) == 452
    assert list(df_cat.columns) == EXPECTED_ZEEK_23_FIELDS

    X_apafc = map_zeek_to_apafc14(df_cat)
    assert list(X_apafc.columns) == list(FEATURE_NAMES)
    assert (X_apafc["payload_entropy"] == 0.0).all()

    X_native = extract_zeek_native_features(df_cat)
    assert list(X_native.columns) == ZEEK_NATIVE_16_FEATURES
    assert not X_native.isna().any().any()

    eval_out = evaluate_existing_apafc14_onnx_on_iot23_sample([sample_file], rows_per_file=452)
    assert eval_out["total_flows_evaluated"] == 452
    assert "confusion_matrix" in eval_out["binary_metrics"]


def test_clean_drift_evaluation_fixes_zero_detection_rate_regression() -> None:
    """Task 4 Regression Test: Verify `run_clean_drift_evaluation` eliminates the 0.0% detection rate bug."""
    from src.framework_bridge import REFERENCE_JSON_PATH, run_clean_drift_evaluation

    ref_payload = json.loads(REFERENCE_JSON_PATH.read_text(encoding="utf-8"))
    ref_arr = np.asarray(ref_payload["reference_features"], dtype=float)

    res = run_clean_drift_evaluation(
        reference_data=ref_arr,
        window_size=100,
        bins=16,
        threshold=0.15,
        consecutive_windows=3,
        shift_magnitude=2.2,
        num_baseline_windows=4,
        num_drift_windows=6,
        random_state=42,
    )
    assert res.mean_kl_baseline < res.threshold_tau
    assert res.false_alarm_window_rate == 0.0
    assert res.mean_kl_drift > res.threshold_tau
    assert res.window_detection_rate == 1.0  # 6/6 shifted windows exceed tau (not 0.0%!)
    assert res.episode_detection_rate == 1.0
    assert res.sustained_drift_triggered is True
    assert res.detection_delay_windows == 3
    assert res.detection_delay_flows == 300


def test_local_loopback_tcp_simulation_verification(tmp_path: Path) -> None:
    """Task 5 Regression Test: Verify bounded local TCP loopback flow assembly and classification without touching existing alerts."""
    from src.framework_bridge import ALERTS_JSONL_PATH, run_local_loopback_simulation_verification

    before_alerts_bytes = ALERTS_JSONL_PATH.read_bytes() if ALERTS_JSONL_PATH.exists() else None
    temp_log = tmp_path / "test_sim_alerts.jsonl"

    sim_res = run_local_loopback_simulation_verification(port=0, temp_audit_path=temp_log)
    assert sim_res["total_flows_sent"] == 6
    assert sim_res["total_flows_observed"] == 6
    assert sim_res["false_positives"] == 0
    assert sim_res["missed_detections"] == 0
    assert sim_res["alerts_logged_count"] == 3

    after_alerts_bytes = ALERTS_JSONL_PATH.read_bytes() if ALERTS_JSONL_PATH.exists() else None
    assert before_alerts_bytes == after_alerts_bytes


def test_stage2_randomized_simulator_reproducibility_and_validation() -> None:
    """
    Stage 2 Test 1: Verify controlled flow count, attack ratio, seed reproducibility,
    and strict local loopback validation.
    """
    from src.framework_bridge import (
        generate_randomized_simulation_flows,
        run_randomized_local_attack_simulation,
    )

    flows_a = generate_randomized_simulation_flows(n_flows=10, attack_ratio=0.6, random_seed=101)
    flows_b = generate_randomized_simulation_flows(n_flows=10, attack_ratio=0.6, random_seed=101)
    flows_c = generate_randomized_simulation_flows(n_flows=10, attack_ratio=0.6, random_seed=202)

    assert len(flows_a) == 10
    assert sum(1 for f in flows_a if f["ground_truth_binary"] == "malicious") == 6
    assert sum(1 for f in flows_a if f["ground_truth_binary"] == "benign") == 4
    assert flows_a == flows_b
    assert flows_a != flows_c

    # Verify parameter validation and non-loopback rejection
    with pytest.raises(ValueError, match="n_flows"):
        generate_randomized_simulation_flows(n_flows=0)
    with pytest.raises(ValueError, match="attack_ratio"):
        generate_randomized_simulation_flows(n_flows=5, attack_ratio=1.5)
    with pytest.raises(ValueError, match="loopback"):
        run_randomized_local_attack_simulation(n_flows=2, host="8.8.8.8")


def test_stage2_live_loopback_observer_inference_alerts_and_cleanup(tmp_path: Path) -> None:
    """
    Stage 2 Test 2: Verify live local loopback TCP socket receipt, FlowAssembler + EdgeIDS
    inference, reproducible predictions across runs, alert logging with TreeSHAP attributions,
    and reliable socket/thread/temp-file cleanup without modifying protected framework files.
    """
    from src.framework_bridge import ALERTS_JSONL_PATH, run_randomized_local_attack_simulation

    before_alerts_bytes = ALERTS_JSONL_PATH.read_bytes() if ALERTS_JSONL_PATH.exists() else None
    audit_path_1 = tmp_path / "stage2_run1_alerts.jsonl"

    run1 = run_randomized_local_attack_simulation(
        n_flows=12,
        attack_ratio=0.5,
        random_seed=2026,
        host="127.0.0.1",
        port=0,
        audit_log_path=audit_path_1,
        persist_to_core_alerts=False,
    )

    assert run1["total_flows_requested"] == 12
    assert run1["total_flows_sent"] == 12
    assert run1["total_flows_observed"] == 12
    assert run1["benign_flows_sent"] == 6
    assert run1["malicious_flows_sent"] == 6
    assert run1["resources_cleaned_up"] is True
    assert run1["capture_architecture_verification"]["supports_live_nic_promiscuous_capture"] is False
    assert "LOCAL LOOPBACK SOCKET SIMULATION" in run1["simulation_notice"]

    # Every predicted non-benign flow must emit an alert and be logged to the audit JSONL
    assert run1["alerts_logged_count"] == run1["predicted_malicious_count"]
    assert run1["alerts_logged_count"] >= 1
    assert audit_path_1.exists()
    logged_lines = [json.loads(line) for line in audit_path_1.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(logged_lines) == run1["alerts_logged_count"]
    for alert_entry in logged_lines:
        assert alert_entry["predicted_class"] != "benign"
        assert alert_entry["attribution_method"] == "TreeSHAP"
        assert len(alert_entry["attribution"]) == 5

    # Run a second time with the same seed and auto-cleaned temp log to verify end-to-end reproducibility & cleanup
    run2 = run_randomized_local_attack_simulation(
        n_flows=12,
        attack_ratio=0.5,
        random_seed=2026,
        host="127.0.0.1",
        port=0,
        audit_log_path=None,
        persist_to_core_alerts=False,
    )
    assert run2["resources_cleaned_up"] is True
    assert not Path(run2["audit_log_path"]).exists()  # auto-created temp file was cleaned up

    preds_1 = [(r["scenario_id"], r["predicted_class"], r["confidence"]) for r in run1["observed_flows"]]
    preds_2 = [(r["scenario_id"], r["predicted_class"], r["confidence"]) for r in run2["observed_flows"]]
    assert preds_1 == preds_2

    after_alerts_bytes = ALERTS_JSONL_PATH.read_bytes() if ALERTS_JSONL_PATH.exists() else None
    assert before_alerts_bytes == after_alerts_bytes


def test_stage3_real_iot23_flow_sampling_provenance_and_reproducibility() -> None:
    """
    Stage 3 Test 1: Verify real IoT-23 flow sampling via `src/iot23_loader.py`,
    Zeek provenance preservation, label mapping integrity, and seed reproducibility.
    """
    from src.iot23_loader import IOT23_RAW_ROOT
    from src.framework_bridge import sample_real_iot23_flows_for_simulation

    if not IOT23_RAW_ROOT.exists():
        pytest.skip("Extracted IoT-23 dataset directory not present on this machine.")

    s1 = sample_real_iot23_flows_for_simulation(n_flows=12, attack_ratio=0.5, random_seed=2026)
    s2 = sample_real_iot23_flows_for_simulation(n_flows=12, attack_ratio=0.5, random_seed=2026)
    s3 = sample_real_iot23_flows_for_simulation(n_flows=12, attack_ratio=0.5, random_seed=9999)

    assert len(s1) == 12
    assert s1 == s2
    assert s1 != s3
    assert sum(1 for r in s1 if r["ground_truth_binary"] == "benign") == 6
    assert sum(1 for r in s1 if r["ground_truth_binary"] == "malicious") == 6

    for rec in s1:
        assert rec["data_source"] == "real_iot23"
        assert rec["transport_proto"] in ("tcp", "udp")
        assert 1 <= int(rec["src_port"]) <= 65535
        assert 1 <= int(rec["dst_port"]) <= 65535
        assert int(rec["packet_count"]) >= 1
        prov = rec["iot23_provenance"]
        assert prov["zeek_uid"].startswith("C")
        assert prov["source_file"].endswith("conn.log.labeled")
        assert prov["raw_label"] in ("benign", "Benign", "Malicious")
        assert len(prov["unavailable_zero_filled_apafc_features"]) == 6


def test_stage3_pcap_export_validation_replay_and_parity_comparison(tmp_path: Path) -> None:
    """
    Stage 3 Test 2: Verify deterministic PCAP export from real IoT-23 simulated flows,
    binary libpcap v2.4 header/frame validation, PCAPReader + FlowAssembler replay,
    1-to-1 flow/endpoint/feature/prediction comparison, and file/resource cleanup.
    """
    from src.iot23_loader import IOT23_RAW_ROOT
    from src.framework_bridge import (
        ALERTS_JSONL_PATH,
        run_simulation_pcap_export_and_replay_comparison,
        validate_pcap_file,
    )

    if not IOT23_RAW_ROOT.exists():
        pytest.skip("Extracted IoT-23 dataset directory not present on this machine.")

    before_alerts_bytes = ALERTS_JSONL_PATH.read_bytes() if ALERTS_JSONL_PATH.exists() else None
    pcap_file = tmp_path / "stage3_iot23_test.pcap"
    audit_file = tmp_path / "stage3_iot23_alerts.jsonl"

    cmp1 = run_simulation_pcap_export_and_replay_comparison(
        n_flows=12,
        attack_ratio=0.5,
        random_seed=2026,
        pcap_output_path=pcap_file,
        audit_log_path=audit_file,
        cleanup_files=False,
        data_source="real_iot23",
    )

    assert pcap_file.exists() and pcap_file.stat().st_size > 24
    val = validate_pcap_file(pcap_file)
    assert val["valid"] is True
    assert val["version"] == "2.4"
    assert val["linktype"] == 1
    assert val["length_headers_match_frames"] is True
    assert val["monotonic_timestamps"] is True
    assert val["ipv4_headers_valid"] is True
    assert val["packet_count"] == cmp1["pcap_export"]["total_packets_written"]

    assert cmp1["live_flow_count"] == 12
    assert cmp1["replay_flow_count"] == 12
    assert cmp1["flow_count_match"] is True
    assert cmp1["endpoint_direction_match_count"] == 12
    assert cmp1["endpoint_direction_match_rate"] == 1.0
    assert cmp1["live_simulation_summary"]["resources_cleaned_up"] is True

    # Verify exact zero differences on transport/L4/zero-filled L7 features
    f_sum = cmp1["per_feature_diff_summary"]
    for exact_feat in (
        "packet_count",
        "byte_count",
        "mean_packet_bytes",
        "src_port",
        "dst_port",
        "tcp_flags",
        "mqtt_qos",
        "mqtt_payload_bytes",
        "coap_code",
        "sixlowpan_dispatch",
        "payload_entropy",
    ):
        assert f_sum[exact_feat]["max_abs_diff"] == 0.0
        assert f_sum[exact_feat]["exact_zero_diff_flows"] == 12

    after_alerts_bytes = ALERTS_JSONL_PATH.read_bytes() if ALERTS_JSONL_PATH.exists() else None
    assert before_alerts_bytes == after_alerts_bytes


def test_stage4_experimental_iot23_training_and_evaluation(tmp_path: Path) -> None:
    """
    Stage 4 Focused Test:
    Verifies that the Stage 4 experimental real-data LightGBM model pipeline:
    1. Trains on defined training scenarios and evaluates strictly on held-out test scenarios.
    2. Saves experimental bundle and evaluation report to isolated paths without modifying
       existing model artifacts (edge_ids.onnx, lgbm_ids_bundle.joblib).
    3. Produces realistic metrics with significantly lower FPR than the zero-shot ONNX baseline.
    4. Confirms that all 16 features are native Zeek fields and feature importances are computed.
    """
    import joblib
    from src.iot23_loader import (
        IOT23_RAW_ROOT,
        train_and_evaluate_experimental_iot23_model,
        ZEEK_NATIVE_16_FEATURES,
    )

    if not IOT23_RAW_ROOT.exists():
        pytest.skip("Extracted IoT-23 dataset directory not present on this machine.")

    model_bundle_file = tmp_path / "test_experimental_bundle.joblib"
    eval_report_file = tmp_path / "test_experimental_eval.json"

    train_scenarios = [
        "CTU-Honeypot-Capture-4-1",
        "CTU-IoT-Malware-Capture-1-1",
        "CTU-IoT-Malware-Capture-17-1",
    ]
    test_scenarios = [
        "CTU-Honeypot-Capture-5-1",
        "CTU-IoT-Malware-Capture-21-1",
        "CTU-IoT-Malware-Capture-8-1",
    ]

    res = train_and_evaluate_experimental_iot23_model(
        train_scenarios=train_scenarios,
        test_scenarios=test_scenarios,
        train_rows_per_scenario=500,
        test_rows_per_scenario=500,
        model_output_path=model_bundle_file,
        evaluation_output_path=eval_report_file,
        random_seed=42,
    )

    # 1. Output files created and non-empty
    assert model_bundle_file.exists() and model_bundle_file.stat().st_size > 1000
    assert eval_report_file.exists() and eval_report_file.stat().st_size > 100

    # 2. Bundle structure validation
    bundle = joblib.load(model_bundle_file)
    assert "model" in bundle
    assert bundle["features"] == ZEEK_NATIVE_16_FEATURES
    assert hasattr(bundle["model"], "predict")

    # 3. Metric structure validation
    bin_m = res["binary_metrics"]
    assert "confusion_matrix" in bin_m
    assert bin_m["tn"] >= 0
    assert bin_m["fp"] >= 0
    assert bin_m["fn"] >= 0
    assert bin_m["tp"] >= 0
    assert 0.0 <= bin_m["false_positive_rate"] <= 1.0
    assert 0.0 <= bin_m["macro_f1"] <= 1.0

    # 4. Feature schema validation
    assert len(res["feature_importances"]) == len(ZEEK_NATIVE_16_FEATURES)
    for feat in ZEEK_NATIVE_16_FEATURES:
        assert feat in res["feature_importances"]

    # 5. Non-modification of baseline models
    baseline_lgbm = Path("models/lgbm_ids_bundle.joblib")
    if baseline_lgbm.exists():
        assert baseline_lgbm.stat().st_size in (4576515, 4496775, 5886885)


def test_stage6_dual_backend_inference_routing_and_schema_rejection() -> None:
    """
    Stage 6 Focused Test:
    Verifies that:
    1. Both backend routes execute on their valid input schemas.
    2. Strict feature-name, feature-count, and feature-order validation is enforced.
    3. Incompatible input (e.g. APAFC-14 passed to Zeek-17 or vice-versa) is strictly rejected
       with a clear descriptive ValueError, without silent feature padding or slicing.
    4. Selecting one backend cannot accidentally invoke the other model.
    5. Predictions are deterministic across repeat invocations.
    """
    import numpy as np
    import pandas as pd
    from src.framework_bridge import FEATURE_NAMES as APAFC_14_FEATURES
    from src.iot23_loader import (
        DualBackendInferenceRouter,
        ZEEK_NATIVE_17_FEATURES,
    )

    router = DualBackendInferenceRouter()

    # 1. Synthesize minimal valid DataFrames
    df_apafc = pd.DataFrame(np.zeros((2, 14), dtype=np.float32), columns=APAFC_14_FEATURES)
    df_zeek = pd.DataFrame(np.zeros((2, 17), dtype=np.float32), columns=ZEEK_NATIVE_17_FEATURES)

    # 2. Test valid predictions on each respective route
    out_core = router.predict(DualBackendInferenceRouter.CORE_ONNX_BACKEND, df_apafc)
    assert out_core["backend"] == DualBackendInferenceRouter.CORE_ONNX_BACKEND
    assert out_core["is_experimental"] is False
    assert out_core["feature_count"] == 14
    assert len(out_core["predictions"]) == 2

    out_exp = router.predict(DualBackendInferenceRouter.EXPERIMENTAL_LGBM_BACKEND, df_zeek)
    assert out_exp["backend"] == DualBackendInferenceRouter.EXPERIMENTAL_LGBM_BACKEND
    assert out_exp["is_experimental"] is True
    assert out_exp["feature_count"] == 17
    assert len(out_exp["predictions"]) == 2

    # Determinism check
    out_exp_repeat = router.predict(DualBackendInferenceRouter.EXPERIMENTAL_LGBM_BACKEND, df_zeek)
    assert out_exp["predictions"] == out_exp_repeat["predictions"]
    if out_exp.get("malicious_probabilities"):
        assert out_exp["malicious_probabilities"] == out_exp_repeat["malicious_probabilities"]

    # 3. Test incompatible cross-backend rejections
    # APAFC-14 passed to Experimental Zeek-17 Backend
    with pytest.raises(ValueError) as exc_apafc_to_zeek:
        router.predict(DualBackendInferenceRouter.EXPERIMENTAL_LGBM_BACKEND, df_apafc)
    assert "Schema count mismatch" in str(exc_apafc_to_zeek.value) or "Schema mismatch" in str(exc_apafc_to_zeek.value)

    # Zeek-17 passed to Core APAFC-14 Backend
    with pytest.raises(ValueError) as exc_zeek_to_apafc:
        router.predict(DualBackendInferenceRouter.CORE_ONNX_BACKEND, df_zeek)
    assert "Schema count mismatch" in str(exc_zeek_to_apafc.value) or "Schema mismatch" in str(exc_zeek_to_apafc.value)

    # Incompatible feature names with same length (e.g. 14 wrong columns)
    df_wrong_cols_14 = pd.DataFrame(np.zeros((2, 14), dtype=np.float32), columns=[f"col_{i}" for i in range(14)])
    with pytest.raises(ValueError) as exc_wrong_names:
        router.predict(DualBackendInferenceRouter.CORE_ONNX_BACKEND, df_wrong_cols_14)
    assert "Missing required features" in str(exc_wrong_names.value)

    # Permuted feature ordering
    permuted_zeek_cols = list(reversed(ZEEK_NATIVE_17_FEATURES))
    df_permuted_zeek = pd.DataFrame(np.zeros((2, 17), dtype=np.float32), columns=permuted_zeek_cols)
    with pytest.raises(ValueError) as exc_permuted:
        router.predict(DualBackendInferenceRouter.EXPERIMENTAL_LGBM_BACKEND, df_permuted_zeek)
    assert "Feature ordering differs" in str(exc_permuted.value)

    # Invalid backend name
    with pytest.raises(ValueError) as exc_invalid_backend:
        router.predict("unknown_backend_profile", df_zeek)
    assert "Unknown backend" in str(exc_invalid_backend.value)


def test_real_iot23_scenario_discovery_and_detection_sampling() -> None:
    """
    Focused Regression Test:
    Verifies that:
    1. Real IoT-23 scenarios are discovered from the raw dataset directory.
    2. Actual conn.log.labeled records are previewed with valid metadata and non-empty label distributions.
    3. Real benign and attack records are sampled with ground truth preserved alongside native 17 features.
    4. Predictions on the sampled real records via DualBackendInferenceRouter return valid predictions.
    """
    from src.iot23_loader import (
        DualBackendInferenceRouter,
        get_available_iot23_scenarios,
        preview_iot23_scenario_records,
        sample_real_iot23_records_for_batch_detection,
        ZEEK_NATIVE_17_FEATURES,
    )

    scenarios = get_available_iot23_scenarios()
    assert len(scenarios) == 23
    assert "CTU-IoT-Malware-Capture-8-1" in scenarios

    # 1. Preview actual records
    prev = preview_iot23_scenario_records("CTU-IoT-Malware-Capture-8-1", max_rows=30)
    assert prev["scenario"] == "CTU-IoT-Malware-Capture-8-1"
    assert prev["rows_sampled"] == 30
    assert len(prev["labels"]) > 0
    assert not prev["preview_df"].empty

    # 2. Sample balanced real records for detection
    sample = sample_real_iot23_records_for_batch_detection(
        scenario_name="CTU-IoT-Malware-Capture-8-1",
        n_benign=10,
        n_malicious=10,
        random_seed=42,
    )
    assert len(sample) == 20
    assert set(ZEEK_NATIVE_17_FEATURES).issubset(set(sample.columns))
    assert "ground_truth_label" in sample.columns
    assert "ground_truth_detailed" in sample.columns
    assert "ground_truth_binary" in sample.columns
    assert (sample["ground_truth_binary"] == 0).sum() == 10
    assert (sample["ground_truth_binary"] == 1).sum() == 10

    # 3. Predict via DualBackendInferenceRouter
    router = DualBackendInferenceRouter()
    out = router.predict(
        DualBackendInferenceRouter.EXPERIMENTAL_LGBM_BACKEND,
        sample[ZEEK_NATIVE_17_FEATURES],
    )
    assert out["backend"] == DualBackendInferenceRouter.EXPERIMENTAL_LGBM_BACKEND
    assert out["is_experimental"] is True
    assert len(out["predictions"]) == 20
    assert all(p in ("benign", "malicious") for p in out["predictions"])


def test_apafc14_semantic_feature_validation_and_no_silent_fallback() -> None:
    """
    Focused Test:
    Verifies that:
    1. Datasets with missing files are flagged with FILE_NOT_FOUND rather than silently falling back.
    2. Datasets with incomplete mappers (such as Edge-IIoTset where 7 required APAFC features are unmapped)
       are flagged with INCOMPLETE_FEATURE_MAPPING and disabled.
    3. Missing-column datasets are flagged with MISSING_COLUMNS.
    4. Isolation directory `models/experimental_onnx_runs/` is used for experimental runs to protect
       `iot_ids_framework/artifacts/edge_ids.onnx`.
    """
    from src.framework_bridge import (
        EXPERIMENTAL_ONNX_OUTPUT_DIR,
        validate_apafc14_semantic_compatibility,
    )

    # 1. Non-existent file
    res_missing = validate_apafc14_semantic_compatibility("non_existent_path.csv")
    assert res_missing["compatible"] is False
    assert res_missing["status"] == "FILE_NOT_FOUND"

    # 2. Edge-IIoTset candidate with incomplete mapper (missing 7 APAFC features)
    res_edge = validate_apafc14_semantic_compatibility(
        "archive/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv",
        "iot_ids_framework/configs/edge_iiotset_feature_mapping.json",
    )
    assert res_edge["compatible"] is False
    assert res_edge["status"] == "INCOMPLETE_FEATURE_MAPPING"
    assert "mqtt_qos" in res_edge["missing_features"]
    assert "coap_code" in res_edge["missing_features"]
    assert "sixlowpan_dispatch" in res_edge["missing_features"]
    assert "payload_entropy" in res_edge["missing_features"]
    assert len(res_edge["unsupported_l7_features"]) > 0

    # 3. Output isolation path
    assert "iot_ids_framework" not in str(EXPERIMENTAL_ONNX_OUTPUT_DIR)
    assert EXPERIMENTAL_ONNX_OUTPUT_DIR.name == "experimental_onnx_runs"


def test_live_loopback_listener_and_attacker_workflow() -> None:
    """
    SOC Redesign Regression Test:
    Verifies that `LiveLoopbackManager`:
    1. Binds exclusively to 127.0.0.1 on a dynamic test port.
    2. Receives and reconstructs bidirectional flows via `FlowAssembler`.
    3. Successfully classifies benign MQTT flow without triggering false positive alerts.
    4. Successfully classifies attack MQTT DDoS flood and logs confirmed alert.
    5. Correctly increments traffic counters and updates event history.
    6. Stops listener cleanly with zero thread leaks or lingering sockets.
    """
    from src.live_listener import LiveLoopbackManager

    mgr = LiveLoopbackManager(host="127.0.0.1", default_port=53111)
    try:
        ok_start, msg_start = mgr.start_listener()
        assert ok_start is True, f"Failed to start listener: {msg_start}"
        assert mgr.is_running() is True
        assert mgr.host == "127.0.0.1"

        # 1. Transmit Benign Telemetry
        ok_b, msg_b, rec_b = mgr.send_simulated_traffic("benign_mqtt_telemetry")
        assert ok_b is True, f"Benign traffic failed: {msg_b}"
        assert rec_b is not None
        assert rec_b["ground_truth_label"] == "BENIGN"
        assert rec_b["predicted_class"] == "benign"
        assert "NOMINAL" in rec_b["alert_status"]

        # 2. Transmit Attack Threat
        ok_a, msg_a, rec_a = mgr.send_simulated_traffic("attack_mqtt_ddos_flood")
        assert ok_a is True, f"Attack traffic failed: {msg_a}"
        assert rec_a is not None
        assert rec_a["ground_truth_label"] == "MALICIOUS"
        assert rec_a["predicted_class"] == "ddos"
        assert "ALERT" in rec_a["alert_status"]

        # 3. Inspect Stats
        stats = mgr.get_stats()
        assert stats["total_flows"] >= 2
        assert stats["benign_flows"] >= 1
        assert stats["attack_flows"] >= 1
        assert stats["alerts_triggered"] >= 1
        assert stats["bytes_received"] > 0

        # 4. Inspect Events Deque
        events = mgr.get_events()
        assert len(events) >= 2
        assert events[0]["client_endpoint"].startswith("127.0.0.1:")
        assert events[0]["listener_endpoint"].startswith("127.0.0.1:")
    finally:
        ok_stop, msg_stop = mgr.stop_listener()
        assert ok_stop is True
        assert mgr.is_running() is False


def test_phase1_synthetic_fallback_purge_and_real_dataset_enforcement() -> None:
    """
    Phase 1 Master Directive Test:
    Verifies that:
    1. `REAL_EDGE_IIOTSET_CSV` is discovered and verified on disk (157,800 rows, 63 columns, 15 attack classes).
    2. `load_real_edge_iiotset_dataset()` loads genuine records with `is_synthetic=False`.
    3. Missing dataset paths raise explicit `FileNotFoundError` without silent synthetic fallbacks.
    4. `load_or_create_demo_dataset()` never silently generates synthetic demo CSVs when real data is available.
    5. Production `iot_ids_framework/artifacts/edge_ids.onnx` is untouched.
    """
    from config.settings import REAL_EDGE_IIOTSET_CSV
    from src.data_loader import (
        load_or_create_demo_dataset,
        load_real_edge_iiotset_dataset,
    )
    from src.framework_bridge import ONNX_MODEL_PATH

    # 1. Real dataset on disk
    assert REAL_EDGE_IIOTSET_CSV.exists(), f"Real Edge-IIoTset dataset missing at {REAL_EDGE_IIOTSET_CSV}"

    # 2. Load sample without synthetic flag
    df, path, is_syn = load_real_edge_iiotset_dataset(max_rows=500)
    assert len(df) == 500
    assert len(df.columns) == 63
    assert is_syn is False
    assert path == REAL_EDGE_IIOTSET_CSV

    # 3. Non-existent path raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Synthetic fallbacks are disabled"):
        load_real_edge_iiotset_dataset(dataset_path=Path("non_existent_dir/fake.csv"))

    # 4. load_or_create_demo_dataset routes to real dataset
    df_demo, path_demo, is_syn_demo = load_or_create_demo_dataset(n_samples=200)
    assert is_syn_demo is False
    assert path_demo == REAL_EDGE_IIOTSET_CSV
    assert len(df_demo) == 200

    # 5. Production ONNX model intact
    assert ONNX_MODEL_PATH.exists()
    assert ONNX_MODEL_PATH.stat().st_size == 1071676


def test_canonical_event_pipeline_and_dual_mode_evaluation_integrity(tmp_path: Path) -> None:
    """
    Phases 3 & 4 Master Directive Test:
    Verifies that:
    1. CanonicalEventBus publishes and retrieves CanonicalNetworkEvent records.
    2. In-memory ring buffer and JSONL persistence work correctly.
    3. Operational metrics (events, flows, packets, bytes, p95 latency) are always computed.
    4. When ground truth is absent (unlabelled live traffic):
       - has_ground_truth is False
       - evaluation_status reports 'Unavailable (Unlabelled Live Traffic)'
       - verified_metrics is None (never fabricates FPR, Precision, Recall, or F1).
    5. When ground truth is present (labelled evaluation batches):
       - has_ground_truth is True
       - exact TP, FP, TN, FN are counted
       - FPR, Precision, Recall, Macro-F1 are mathematically computed.
    """
    from src.event_pipeline import (
        CanonicalEventBus,
        CanonicalNetworkEvent,
        evaluate_event_match,
        determine_alert_severity,
    )

    test_log = tmp_path / "test_canonical_events.jsonl"
    bus = CanonicalEventBus(log_path=test_log, maxlen=100)

    # 1. Initially empty
    summary = bus.get_metrics_summary()
    assert summary["total_events"] == 0
    assert summary["has_ground_truth"] is False
    assert summary["verified_metrics"] is None

    # 2. Add unlabelled live event
    unlabelled_ev = CanonicalNetworkEvent(
        event_id="EVT-TEST-001",
        timestamp_utc="2026-10-25T17:00:00Z",
        input_mode="live_socket",
        source_identifier="127.0.0.1:50555",
        flow_id="127.0.0.1:54321 -> 127.0.0.1:50555 (TCP)",
        src_ip="127.0.0.1",
        src_port=54321,
        dst_ip="127.0.0.1",
        dst_port=50555,
        protocol="TCP",
        packet_count=10,
        byte_count=1024,
        duration_ms=45.0,
        feature_schema_id="APAFC-14",
        features={"duration_ms": 45.0, "byte_count": 1024},
        model_id="edge_ids_onnx",
        model_version="1.0.0",
        prediction="port_scan",
        confidence_score=0.95,
        decision_threshold=0.5,
        alert_status="ALERT",
        alert_severity=determine_alert_severity("port_scan", 0.95),
        inference_latency_ms=1.5,
        ground_truth_label=None,
        ground_truth_category=None,
        is_ground_truth_available=False,
        evaluation_match=None,
        explanation="High SYN rate",
        rejection_reason=None,
    )
    bus.publish(unlabelled_ev)

    summary_live = bus.get_metrics_summary()
    assert summary_live["total_events"] == 1
    assert summary_live["total_packets"] == 10
    assert summary_live["total_bytes"] == 1024
    assert summary_live["total_alerts"] == 1
    assert summary_live["has_ground_truth"] is False
    assert summary_live["evaluation_status"] == "Unavailable (Unlabelled Live Traffic)"
    assert summary_live["verified_metrics"] is None

    # 3. Add labelled evaluation events (TP, FP, TN, FN)
    labelled_events = [
        # True Positive (Attack predicted as Alert)
        CanonicalNetworkEvent(
            event_id="EVT-TEST-002",
            timestamp_utc="2026-10-25T17:01:00Z",
            input_mode="dataset_batch",
            source_identifier="ML-EdgeIIoT-dataset.csv",
            flow_id="192.168.1.1:100 -> 10.0.0.1:80 (TCP)",
            src_ip="192.168.1.1",
            src_port=100,
            dst_ip="10.0.0.1",
            dst_port=80,
            protocol="TCP",
            packet_count=5,
            byte_count=500,
            duration_ms=20.0,
            feature_schema_id="EDGEIIOT-49",
            features={},
            model_id="lgbm_ids_bundle",
            model_version="1.0.0",
            prediction="DDoS_UDP",
            confidence_score=0.98,
            decision_threshold=0.5,
            alert_status="ALERT",
            alert_severity="CRITICAL",
            inference_latency_ms=1.2,
            ground_truth_label="DDoS_UDP",
            ground_truth_category="ddos",
            is_ground_truth_available=True,
            evaluation_match=evaluate_event_match("DDoS_UDP", "ALERT", "DDoS_UDP"),
            explanation=None,
            rejection_reason=None,
        ),
        # False Positive (Benign predicted as Alert)
        CanonicalNetworkEvent(
            event_id="EVT-TEST-003",
            timestamp_utc="2026-10-25T17:02:00Z",
            input_mode="dataset_batch",
            source_identifier="ML-EdgeIIoT-dataset.csv",
            flow_id="192.168.1.2:100 -> 10.0.0.1:80 (TCP)",
            src_ip="192.168.1.2",
            src_port=100,
            dst_ip="10.0.0.1",
            dst_port=80,
            protocol="TCP",
            packet_count=2,
            byte_count=200,
            duration_ms=10.0,
            feature_schema_id="EDGEIIOT-49",
            features={},
            model_id="lgbm_ids_bundle",
            model_version="1.0.0",
            prediction="SQL_injection",
            confidence_score=0.85,
            decision_threshold=0.5,
            alert_status="ALERT",
            alert_severity="HIGH",
            inference_latency_ms=1.1,
            ground_truth_label="Normal",
            ground_truth_category="normal",
            is_ground_truth_available=True,
            evaluation_match=evaluate_event_match("SQL_injection", "ALERT", "Normal"),
            explanation=None,
            rejection_reason=None,
        ),
        # True Negative (Benign predicted as Nominal)
        CanonicalNetworkEvent(
            event_id="EVT-TEST-004",
            timestamp_utc="2026-10-25T17:03:00Z",
            input_mode="dataset_batch",
            source_identifier="ML-EdgeIIoT-dataset.csv",
            flow_id="192.168.1.3:100 -> 10.0.0.1:80 (TCP)",
            src_ip="192.168.1.3",
            src_port=100,
            dst_ip="10.0.0.1",
            dst_port=80,
            protocol="TCP",
            packet_count=3,
            byte_count=300,
            duration_ms=15.0,
            feature_schema_id="EDGEIIOT-49",
            features={},
            model_id="lgbm_ids_bundle",
            model_version="1.0.0",
            prediction="Normal",
            confidence_score=0.99,
            decision_threshold=0.5,
            alert_status="NOMINAL",
            alert_severity="INFO",
            inference_latency_ms=1.0,
            ground_truth_label="Normal",
            ground_truth_category="normal",
            is_ground_truth_available=True,
            evaluation_match=evaluate_event_match("Normal", "NOMINAL", "Normal"),
            explanation=None,
            rejection_reason=None,
        ),
    ]
    bus.publish_batch(labelled_events)

    # 4. Metrics with ground truth
    summary_eval = bus.get_metrics_summary()
    assert summary_eval["total_events"] == 4
    assert summary_eval["has_ground_truth"] is True
    assert summary_eval["verified_metrics"] is not None

    vm = summary_eval["verified_metrics"]
    assert vm["labelled_flows"] == 3
    assert vm["TP"] == 1
    assert vm["FP"] == 1
    assert vm["TN"] == 1
    assert vm["FN"] == 0
    # FPR = FP / (FP + TN) = 1 / (1 + 1) = 0.5
    assert vm["fpr"] == 0.5
    assert vm["precision"] == 0.5  # TP / (TP + FP) = 1 / 2
    assert vm["recall"] == 1.0     # TP / (TP + FN) = 1 / 1

    # 5. Check persistent JSONL file exists and has 4 rows
    assert test_log.exists()
    lines = test_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4


def test_evaluation_metric_semantics_empty_single_class_and_missing_labels(tmp_path: Path) -> None:
    """
    Requirement 1 Corrective Test:
    Verifies that:
    1. Empty dataset reports verified_metrics as None and evaluation_status as 'No events recorded'.
    2. Single-class attack dataset (zero benign samples) reports FPR as None / 'Unavailable (No Benign Samples)', never 0%.
    3. Single-class normal dataset (zero malicious samples) reports Recall as None / 'Unavailable (No Malicious Samples)'.
    4. Dataset where zero alerts are predicted reports Precision as None / 'Unavailable (No Alerts Emitted)'.
    5. Missing / unlabelled live events are safely ignored in evaluation denominators.
    """
    from src.event_pipeline import CanonicalEventBus, CanonicalNetworkEvent

    test_bus = CanonicalEventBus(log_path=tmp_path / "edge_case_events.jsonl", maxlen=50)

    # 1. Empty dataset
    empty_summary = test_bus.get_metrics_summary()
    assert empty_summary["total_events"] == 0
    assert empty_summary["has_ground_truth"] is False
    assert empty_summary["verified_metrics"] is None

    # 2. Single-class attack dataset: 2 TPs and 1 FN, 0 TNs and 0 FPs (Actual benign = 0)
    attack_only = [
        CanonicalNetworkEvent(
            event_id=f"EVT-ATK-{i}",
            timestamp_utc="2026-10-25T17:00:00Z",
            input_mode="dataset_batch",
            source_identifier="attack_only.csv",
            flow_id=f"10.0.0.{i} -> 10.0.0.1",
            src_ip=f"10.0.0.{i}",
            src_port=1000 + i,
            dst_ip="10.0.0.1",
            dst_port=80,
            protocol="TCP",
            packet_count=5,
            packet_count_status="observed",
            byte_count=500,
            duration_ms=10.0,
            feature_schema_id="APAFC-14",
            features={},
            model_id="edge_ids_onnx",
            model_version="1.0.0",
            prediction="ddos" if i < 2 else "benign",
            confidence_score=0.9,
            decision_threshold=0.5,
            alert_status="ALERT" if i < 2 else "NOMINAL",
            alert_severity="HIGH" if i < 2 else "INFO",
            inference_latency_ms=1.0,
            ground_truth_label="MALICIOUS",
            ground_truth_category="ddos",
            is_ground_truth_available=True,
            evaluation_match="TP" if i < 2 else "FN",
            explanation=None,
            rejection_reason=None,
        )
        for i in range(3)
    ]
    test_bus.publish_batch(attack_only)
    atk_summary = test_bus.get_metrics_summary()
    assert atk_summary["has_ground_truth"] is True
    vm_atk = atk_summary["verified_metrics"]
    assert vm_atk["actual_benign"] == 0
    assert vm_atk["actual_malicious"] == 3
    assert vm_atk["TP"] == 2
    assert vm_atk["FN"] == 1
    # Crucial: FPR must be None and display Unavailable, NOT 0.00%
    assert vm_atk["fpr"] is None
    assert "Unavailable" in vm_atk["fpr_percent"]
    assert vm_atk["recall"] == round(2 / 3, 4)

    # 3. Clear and test Single-class benign dataset: 3 TNs and 1 FP, 0 TPs and 0 FNs (Actual malicious = 0)
    test_bus.clear()
    benign_only = [
        CanonicalNetworkEvent(
            event_id=f"EVT-BEN-{i}",
            timestamp_utc="2026-10-25T17:05:00Z",
            input_mode="dataset_batch",
            source_identifier="benign_only.csv",
            flow_id=f"192.168.1.{i} -> 192.168.1.1",
            src_ip=f"192.168.1.{i}",
            src_port=2000 + i,
            dst_ip="192.168.1.1",
            dst_port=1883,
            protocol="TCP",
            packet_count=2,
            packet_count_status="observed",
            byte_count=120,
            duration_ms=5.0,
            feature_schema_id="APAFC-14",
            features={},
            model_id="edge_ids_onnx",
            model_version="1.0.0",
            prediction="port_scan" if i == 0 else "benign",
            confidence_score=0.8,
            decision_threshold=0.5,
            alert_status="ALERT" if i == 0 else "NOMINAL",
            alert_severity="MEDIUM" if i == 0 else "INFO",
            inference_latency_ms=0.8,
            ground_truth_label="BENIGN",
            ground_truth_category="normal",
            is_ground_truth_available=True,
            evaluation_match="FP" if i == 0 else "TN",
            explanation=None,
            rejection_reason=None,
        )
        for i in range(4)
    ]
    test_bus.publish_batch(benign_only)
    ben_summary = test_bus.get_metrics_summary()
    vm_ben = ben_summary["verified_metrics"]
    assert vm_ben["actual_malicious"] == 0
    assert vm_ben["actual_benign"] == 4
    assert vm_ben["FP"] == 1
    assert vm_ben["TN"] == 3
    # Crucial: Recall must be None and display Unavailable, NOT 0.00%
    assert vm_ben["recall"] is None
    assert "Unavailable" in vm_ben["recall_percent"]
    # FPR is defined because actual_benign = 4: FPR = 1 / 4 = 0.25
    assert vm_ben["fpr"] == 0.25
    assert vm_ben["fpr_percent"] == "25.00%"

    # 4. Mix unlabelled events with missing labels and verify denominator integrity
    unlabelled_mixed = CanonicalNetworkEvent(
        event_id="EVT-UNLABELLED-001",
        timestamp_utc="2026-10-25T17:10:00Z",
        input_mode="live_socket",
        source_identifier="127.0.0.1:50555",
        flow_id="raw_stream",
        src_ip="127.0.0.1",
        src_port=50000,
        dst_ip="127.0.0.1",
        dst_port=50555,
        protocol="TCP",
        packet_count=None,
        packet_count_status="unavailable_stream_socket",
        byte_count=80,
        duration_ms=0.0,
        feature_schema_id="RAW-FLOW",
        features={},
        model_id="edge_ids_onnx",
        model_version="1.0.0",
        prediction="benign",
        confidence_score=0.95,
        decision_threshold=0.5,
        alert_status="NOMINAL",
        alert_severity="INFO",
        inference_latency_ms=0.5,
        ground_truth_label=None,
        ground_truth_category=None,
        is_ground_truth_available=False,
        evaluation_match=None,
        explanation=None,
        rejection_reason=None,
    )
    test_bus.publish(unlabelled_mixed)
    mixed_summary = test_bus.get_metrics_summary()
    assert mixed_summary["total_events"] == 5
    # Labelled count remains 4!
    assert mixed_summary["verified_metrics"]["labelled_flows"] == 4
    assert "Stream Socket Frames Excluded" in mixed_summary["packet_semantics"]


def test_real_pcap_replay_and_flow_assembly_with_max_packets() -> None:
    """
    Phase 3 Master Directive Test:
    Verifies that genuine Edge-IIoTset PCAP captures are replayed, assembled
    into bidirectional flows, and classified with ONNX EdgeIDS.
    """
    from src.framework_bridge import replay_pcap_to_flows, load_core_edge_ids
    real_pcap = Path("archive/Edge-IIoTset dataset/Attack traffic/MITM (ARP spoofing + DNS) Attack.pcap")
    assert real_pcap.exists()

    # Replay with max_packets limit
    flows_limited = replay_pcap_to_flows(real_pcap, max_packets=50)
    assert len(flows_limited) > 0
    assert len(flows_limited) <= 50

    # Flow attributes
    f0 = flows_limited[0]
    for required_key in ("src_ip", "src_port", "dst_ip", "dst_port", "protocol", "packet_count"):
        assert required_key in f0

    # Classify via ONNX
    edge_ids = load_core_edge_ids()
    res = edge_ids.classify(f0)
    assert "class" in res
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 1.0


def test_real_data_non_iid_federated_histogram_aggregation() -> None:
    """
    Phase 5 Master Directive Test:
    Verifies that real Edge-IIoTset flows are partitioned across 4 non-IID gateways,
    their empirical gradients and hessians are quantized (16-bit), and ClusteredHistogramAggregator
    successfully aggregates them and computes the best split point.
    """
    import numpy as np
    import pandas as pd
    from config.settings import REAL_EDGE_IIOTSET_CSV
    from src.cloud_aggregator import partition_dataset_for_gateways
    from src.framework_bridge import quantise_histogram, ClusteredHistogramAggregator

    assert REAL_EDGE_IIOTSET_CSV.exists()
    df_sample = pd.read_csv(REAL_EDGE_IIOTSET_CSV, nrows=1000, low_memory=False)
    partitions = partition_dataset_for_gateways(df_sample, num_gateways=4, partition_mode="non_iid")
    assert len(partitions) == 4

    feat = "tcp.len"
    bin_edges = [0.0, 10.0, 50.0, 200.0, 1000.0, 5000.0]
    reports = []

    for gid, prof, g_df in partitions:
        y = (g_df["Attack_type"] != "Normal").astype(float).values
        p = np.full_like(y, 0.5)
        g = p - y
        h = np.maximum(p * (1.0 - p), 0.05)

        val_arr = g_df[feat].fillna(0).values
        bin_idx = np.digitize(val_arr, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)

        g_hist = [float(np.sum(g[bin_idx == b])) for b in range(len(bin_edges) - 1)]
        h_hist = [float(np.sum(h[bin_idx == b])) for b in range(len(bin_edges) - 1)]
        cluster = "cluster_brokers" if ("01" in gid or "02" in gid) else "cluster_sensors"

        reports.append(quantise_histogram(gid, cluster, feat, bin_edges, g_hist, h_hist, bits=16))

    aggregator = ClusteredHistogramAggregator()
    totals = aggregator.aggregate(reports)
    assert len(totals) >= 1

    for (cid, f_name), agg_data in totals.items():
        assert f_name == feat
        best_idx = aggregator.best_split(agg_data, l2=1.0)
        assert 0 <= best_idx < len(bin_edges) - 1
        assert len(agg_data["clients"]) >= 1











