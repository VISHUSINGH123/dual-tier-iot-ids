"""Bounded bidirectional flow assembly for packet-capture adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping


@dataclass
class _FlowState:
    first_seen_ms: float
    last_seen_ms: float
    packet_count: int = 0
    byte_count: int = 0
    protocol_fields: dict[str, object] = field(default_factory=dict)


class FlowAssembler:
    """Turns packet metadata into completed, direction-independent flow records.

    Capture adapters must pass only decoded packet metadata, never packet payloads to
    the cloud-facing components. `expire` returns flows idle past the configured
    timeout and bounds the amount of state held at the edge.
    """
    def __init__(self, idle_timeout_ms: int = 1_000, max_flows: int = 20_000) -> None:
        self.idle_timeout_ms, self.max_flows = idle_timeout_ms, max_flows
        self._flows: dict[tuple[object, ...], _FlowState] = {}

    def submit(self, packet: Mapping[str, object]) -> None:
        key = self._key(packet)
        now = float(packet["timestamp_ms"])
        state = self._flows.get(key)
        if state is None:
            if len(self._flows) >= self.max_flows:
                self._flows.pop(next(iter(self._flows)))
            state = self._flows[key] = _FlowState(now, now)
        state.last_seen_ms = now
        state.packet_count += 1
        state.byte_count += int(packet.get("length", 0))
        state.protocol_fields.update({key: value for key, value in packet.items() if key in {"mqtt_qos", "mqtt_payload_bytes", "coap_code", "http_status", "sixlowpan_dispatch", "tcp_flags", "payload_entropy"}})

    def expire(self, now_ms: float) -> list[dict[str, object]]:
        completed = []
        for key, state in list(self._flows.items()):
            if now_ms - state.last_seen_ms >= self.idle_timeout_ms:
                src_ip, src_port, dst_ip, dst_port, protocol = key
                completed.append({"src_ip": src_ip, "src_port": src_port, "dst_ip": dst_ip, "dst_port": dst_port, "protocol": protocol, "duration_ms": state.last_seen_ms - state.first_seen_ms, "packet_count": state.packet_count, "byte_count": state.byte_count, **state.protocol_fields})
                del self._flows[key]
        return completed

    @staticmethod
    def _key(packet: Mapping[str, object]) -> tuple[object, ...]:
        forward = (packet.get("src_ip"), packet.get("src_port"), packet.get("dst_ip"), packet.get("dst_port"))
        reverse = (packet.get("dst_ip"), packet.get("dst_port"), packet.get("src_ip"), packet.get("src_port"))
        first, _ = sorted((forward, reverse), key=repr)
        return (*first, str(packet.get("protocol", "unknown")).lower())
