import numpy as np

from iot_ids.drift import KLDriftMonitor
from iot_ids.features import APAFCFeatureExtractor
from iot_ids.federation import ClusteredHistogramAggregator, quantise_histogram
from iot_ids.ingest import FlowAssembler


def test_feature_vector_has_apa_fc_width():
    vector = APAFCFeatureExtractor().extract({"protocol": "mqtt", "packet_count": 2, "byte_count": 100})
    assert len(vector) == 14
    assert vector[0] == 1.0


def test_kl_monitor_requests_retrain_after_sustained_shift():
    reference = np.zeros((10, 14))
    monitor = KLDriftMonitor(reference, window_size=2, bins=2, threshold=0.01, consecutive_windows=2)
    monitor.observe(np.ones(14) * 100)
    event = monitor.observe(np.ones(14) * 100)
    assert event and not event.retrain_requested
    event = monitor.observe(np.ones(14) * 100)
    assert event and event.retrain_requested


def test_histograms_aggregate_by_cluster_only():
    first = quantise_histogram("a", "cluster-1", "packet_count", [0, 1, 2], [2, -1], [3, 4])
    second = quantise_histogram("b", "cluster-1", "packet_count", [0, 1, 2], [1, -2], [2, 5])
    aggregate = ClusteredHistogramAggregator().aggregate([first, second])
    assert aggregate[("cluster-1", "packet_count")]["clients"] == {"a", "b"}


def test_bidirectional_packets_form_one_completed_flow():
    assembler = FlowAssembler(idle_timeout_ms=10)
    assembler.submit({"timestamp_ms": 0, "src_ip": "a", "src_port": 100, "dst_ip": "b", "dst_port": 1883, "protocol": "mqtt", "length": 20})
    assembler.submit({"timestamp_ms": 5, "src_ip": "b", "src_port": 1883, "dst_ip": "a", "dst_port": 100, "protocol": "mqtt", "length": 30})
    flows = assembler.expire(16)
    assert len(flows) == 1 and flows[0]["packet_count"] == 2 and flows[0]["byte_count"] == 50
