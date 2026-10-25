"""Protocol-aware APAFC feature extraction (61 candidate columns reduced to 14)."""
from __future__ import annotations

import math
from collections.abc import Mapping

FEATURE_NAMES = (
    "protocol_id", "duration_ms", "packet_count", "byte_count", "mean_packet_bytes",
    "src_port", "dst_port", "tcp_flags", "mqtt_qos", "mqtt_payload_bytes",
    "coap_code", "http_status", "sixlowpan_dispatch", "payload_entropy",
)
PROTOCOL_IDS = {"unknown": 0.0, "mqtt": 1.0, "coap": 2.0, "http": 3.0, "6lowpan": 4.0}


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


class APAFCFeatureExtractor:
    """Converts a completed flow record into the fixed 14-value edge vector."""

    feature_names = FEATURE_NAMES

    def extract(self, flow: Mapping[str, object]) -> list[float]:
        protocol = str(flow.get("protocol", "unknown")).lower()
        protocol = "6lowpan" if protocol in {"6lowpan", "6lowpan/ipv6"} else protocol
        packets = max(_number(flow.get("packet_count")), 1.0)
        bytes_ = max(_number(flow.get("byte_count")), 0.0)
        payload = flow.get("payload", b"")
        if isinstance(payload, str):
            payload = payload.encode("utf-8", errors="ignore")
        entropy = _number(flow.get("payload_entropy"), self._entropy(payload) if isinstance(payload, bytes) else 0.0)
        return [
            PROTOCOL_IDS.get(protocol, 0.0), _number(flow.get("duration_ms")), packets, bytes_,
            bytes_ / packets, _number(flow.get("src_port")), _number(flow.get("dst_port")),
            _number(flow.get("tcp_flags")), _number(flow.get("mqtt_qos")),
            _number(flow.get("mqtt_payload_bytes")), _number(flow.get("coap_code")),
            _number(flow.get("http_status")), _number(flow.get("sixlowpan_dispatch")), entropy,
        ]

    @staticmethod
    def _entropy(payload: bytes) -> float:
        if not payload:
            return 0.0
        counts = [payload.count(bytes([item])) for item in range(256)]
        length = len(payload)
        return -sum((count / length) * math.log2(count / length) for count in counts if count)
