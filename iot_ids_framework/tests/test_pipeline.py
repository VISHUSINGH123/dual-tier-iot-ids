"""End-to-end unit and integration tests across the dual-tier IoT IDS modules."""
from __future__ import annotations

import tempfile
import json
from pathlib import Path
import numpy as np
import pandas as pd

from iot_ids.features import APAFCFeatureExtractor, FEATURE_NAMES
from iot_ids.ingest import FlowAssembler
from iot_ids.drift import KLDriftMonitor, DriftEvent
from iot_ids.federation import ClusteredHistogramAggregator, quantise_histogram, QuantisedHistogram
from iot_ids.prepare import map_csv
from iot_ids.edge import EdgeIDS


class MockONNXSession:
    """Mock ONNX Runtime InferenceSession for unit testing without onnx file."""
    def __init__(self, num_classes: int = 15):
        self.num_classes = num_classes

    def get_inputs(self):
        class MockInput:
            name = "features"
        return [MockInput()]

    def run(self, output_names, input_feed):
        # Return mock class probabilities: index 1 (attack) has highest probability
        batch_size = len(next(iter(input_feed.values())))
        probs = np.zeros((batch_size, self.num_classes), dtype=np.float32)
        probs[:, 1] = 0.95  # attack class
        probs[:, 0] = 0.05  # benign
        return [None, probs]


def test_flow_assembler_bidirectional_and_timeout():
    assembler = FlowAssembler(idle_timeout_ms=50, max_flows=5)
    
    # Forward packet
    assembler.submit({
        "timestamp_ms": 100, "src_ip": "10.0.0.1", "src_port": 1883,
        "dst_ip": "10.0.0.2", "dst_port": 50000, "protocol": "mqtt",
        "length": 64, "mqtt_qos": 1, "mqtt_payload_bytes": 32
    })
    
    # Reverse packet
    assembler.submit({
        "timestamp_ms": 120, "src_ip": "10.0.0.2", "src_port": 50000,
        "dst_ip": "10.0.0.1", "dst_port": 1883, "protocol": "mqtt",
        "length": 128, "mqtt_qos": 1, "mqtt_payload_bytes": 64
    })
    
    # No expiration yet at t=140 (only 20ms idle)
    assert len(assembler.expire(140)) == 0
    
    # Expire at t=180 (60ms idle > 50ms timeout)
    expired = assembler.expire(180)
    assert len(expired) == 1
    flow = expired[0]
    assert flow["packet_count"] == 2
    assert flow["byte_count"] == 192
    assert flow["duration_ms"] == 20
    assert flow["mqtt_qos"] == 1


def test_flow_assembler_max_flows_bound():
    assembler = FlowAssembler(idle_timeout_ms=1000, max_flows=3)
    for i in range(5):
        assembler.submit({
            "timestamp_ms": i * 10, "src_ip": f"10.0.0.{i}", "src_port": 1000 + i,
            "dst_ip": "10.0.0.200", "dst_port": 80, "protocol": "http", "length": 50
        })
    # Cannot exceed max_flows (3)
    assert len(assembler._flows) <= 3


def test_apafc_feature_extractor_protocols_and_entropy():
    extractor = APAFCFeatureExtractor()
    assert len(extractor.feature_names) == 14
    
    # Test MQTT
    mqtt_flow = {"protocol": "mqtt", "duration_ms": 10, "packet_count": 2, "byte_count": 100, "payload": b"hello MQTT"}
    vec_mqtt = extractor.extract(mqtt_flow)
    assert len(vec_mqtt) == 14
    assert vec_mqtt[0] == 1.0  # mqtt protocol_id
    assert vec_mqtt[4] == 50.0  # mean_packet_bytes (100 / 2)
    assert vec_mqtt[13] > 0.0   # payload entropy
    
    # Test CoAP
    vec_coap = extractor.extract({"protocol": "coap", "coap_code": 69})
    assert vec_coap[0] == 2.0  # coap protocol_id
    assert vec_coap[10] == 69.0
    
    # Test 6LoWPAN
    vec_6lowpan = extractor.extract({"protocol": "6lowpan/ipv6", "sixlowpan_dispatch": 65})
    assert vec_6lowpan[0] == 4.0
    assert vec_6lowpan[12] == 65.0


def test_kl_drift_monitor_sliding_window():
    # 100 reference rows, 14 features
    reference = np.zeros((100, 14))
    monitor = KLDriftMonitor(reference, window_size=20, bins=4, threshold=0.15, consecutive_windows=3)
    
    # In-distribution samples: KL should be ~0, consecutive count stays 0
    last_event = None
    for _ in range(25):
        event = monitor.observe(np.zeros(14))
        if event:
            last_event = event
    assert last_event is not None
    assert last_event.consecutive == 0
    assert not last_event.retrain_requested
    
    # Out-of-distribution shift (massive mean shift)
    shifted = np.ones((60, 14)) * 50.0
    retrain_triggered = False
    for row in shifted:
        event = monitor.observe(row)
        if event and event.retrain_requested:
            retrain_triggered = True
            break
    assert retrain_triggered


def test_federated_histogram_aggregation():
    # Client 1
    h1 = quantise_histogram(
        client_id="gw_1", cluster_id="c_smart_meter", feature="packet_count",
        bin_edges=[0.0, 10.0, 50.0, 100.0],
        gradient=[1.5, -2.0, 0.5], hessian=[2.0, 3.0, 1.0], bits=16
    )
    # Client 2
    h2 = quantise_histogram(
        client_id="gw_2", cluster_id="c_smart_meter", feature="packet_count",
        bin_edges=[0.0, 10.0, 50.0, 100.0],
        gradient=[0.5, -1.0, 0.5], hessian=[1.0, 2.0, 1.0], bits=16
    )
    
    aggregator = ClusteredHistogramAggregator()
    aggregates = aggregator.aggregate([h1, h2])
    
    key = ("c_smart_meter", "packet_count")
    assert key in aggregates
    assert aggregates[key]["clients"] == {"gw_1", "gw_2"}
    
    best_split = aggregator.best_split(aggregates[key])
    assert isinstance(best_split, int)


def test_prepare_csv_mapping(tmp_path):
    # Create mock raw dataset
    raw_df = pd.DataFrame({
        "src_id": [1, 2],
        "flow_duration": [12.0, 34.0],
        "total_packets": [4, 10],
        "total_bytes": [200, 1000],
        "proto_name": ["MQTT", "HTTP"],
        "class_label": ["benign", "ddos"]
    })
    raw_csv = tmp_path / "raw.csv"
    raw_df.to_csv(raw_csv, index=False)
    
    mapping = {
        "protocol": "proto_name",
        "duration_ms": "flow_duration",
        "packet_count": "total_packets",
        "byte_count": "total_bytes"
    }
    mapping_json = tmp_path / "mapping.json"
    mapping_json.write_text(json.dumps(mapping), encoding="utf-8")
    
    output_csv = tmp_path / "prepared.csv"
    map_csv(raw_csv, mapping_json, label_column="class_label", output=output_csv)
    
    res = pd.read_csv(output_csv)
    assert len(res) == 2
    assert "protocol_id" in res.columns
    assert "mean_packet_bytes" in res.columns
    assert "label" in res.columns
    assert res.loc[0, "protocol_id"] == 1.0  # MQTT
    assert res.loc[0, "mean_packet_bytes"] == 50.0


def test_edge_ids_inference_and_alerts(tmp_path):
    class_names = ["benign", "ddos", "dos", "injection", "mitm", "password", "port_scan", "ransomware", "scanning", "spoofing", "xss", "backdoor", "botnet", "brute_force", "web_attack"]
    reference = np.zeros((50, 14))
    monitor = KLDriftMonitor(reference, window_size=10, bins=4, threshold=0.15, consecutive_windows=2)
    session = MockONNXSession(num_classes=15)
    audit_file = tmp_path / "alerts.jsonl"
    
    edge_ids = EdgeIDS(
        session=session,
        class_names=class_names,
        monitor=monitor,
        audit_path=audit_file
    )
    
    flow = {
        "src_ip": "192.168.1.5", "src_port": 1883, "dst_ip": "192.168.1.100", "dst_port": 50000,
        "protocol": "mqtt", "packet_count": 100, "byte_count": 5000, "duration_ms": 50
    }
    
    classification = edge_ids.classify(flow)
    assert classification["class"] == "ddos"
    assert round(classification["confidence"], 2) == 0.95
    assert "alert" in classification
    assert audit_file.exists()
    
    # Test atomic hot swap
    new_session = MockONNXSession(num_classes=15)
    edge_ids.hot_swap(new_session)
    assert edge_ids._session is new_session
