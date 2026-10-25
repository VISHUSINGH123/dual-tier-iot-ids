"""Packet Capture (PCAP) Adapter for M1 Ingestion into FlowAssembler.

Supports pure-Python streaming replay of standard .pcap captures without external C dependencies,
extracting 5-tuples and protocol metadata (MQTT, CoAP, HTTP, TCP flags).
"""
from __future__ import annotations

import socket
import struct
from pathlib import Path
from typing import Iterator, BinaryIO
from collections.abc import Mapping

from .ingest import FlowAssembler


class PCAPReader:
    """Reads standard PCAP capture files and streams parsed packet metadata."""

    PCAP_MAGIC_SAME_ENDIAN = 0xA1B2C3D4
    PCAP_MAGIC_SWAPPED_ENDIAN = 0xD4C3B2A1

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def read_packets(self) -> Iterator[dict[str, object]]:
        """Yields packet metadata dictionaries compatible with FlowAssembler.submit()."""
        with open(self.file_path, "rb") as f:
            header = f.read(24)
            if len(header) < 24:
                return

            magic, v_maj, v_min, thiszone, sigfigs, snaplen, linktype = struct.unpack("<IHHiIII", header)
            endian = "<"
            if magic == self.PCAP_MAGIC_SWAPPED_ENDIAN:
                endian = ">"
                magic, v_maj, v_min, thiszone, sigfigs, snaplen, linktype = struct.unpack(">IHHiIII", header)
            elif magic != self.PCAP_MAGIC_SAME_ENDIAN:
                raise ValueError(f"Invalid PCAP magic number: {hex(magic)}")

            while True:
                pkt_hdr = f.read(16)
                if len(pkt_hdr) < 16:
                    break
                ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f"{endian}IIII", pkt_hdr)
                pkt_data = f.read(incl_len)
                if len(pkt_data) < incl_len:
                    break

                timestamp_ms = (ts_sec * 1000.0) + (ts_usec / 1000.0)
                parsed = self._parse_packet(pkt_data, timestamp_ms, orig_len, linktype)
                if parsed:
                    yield parsed

    def _parse_packet(self, data: bytes, ts_ms: float, length: int, linktype: int) -> dict[str, object] | None:
        # Linktype 1: Ethernet
        if linktype != 1 or len(data) < 14:
            return None

        eth_type = struct.unpack("!H", data[12:14])[0]
        offset = 14

        # Handle 802.1Q VLAN tag
        if eth_type == 0x8100:
            eth_type = struct.unpack("!H", data[16:18])[0]
            offset = 18

        # IPv4 (0x0800)
        if eth_type != 0x0800 or len(data) < offset + 20:
            return None

        ip_hdr = data[offset:offset + 20]
        ihl = (ip_hdr[0] & 0x0F) * 4
        protocol_num = ip_hdr[9]
        src_ip = socket.inet_ntoa(ip_hdr[12:16])
        dst_ip = socket.inet_ntoa(ip_hdr[16:20])
        l4_offset = offset + ihl

        src_port, dst_port, tcp_flags = 0, 0, 0
        protocol = "unknown"
        payload_bytes = b""

        # TCP (6)
        if protocol_num == 6 and len(data) >= l4_offset + 20:
            tcp_hdr = data[l4_offset:l4_offset + 20]
            src_port, dst_port = struct.unpack("!HH", tcp_hdr[:4])
            tcp_data_offset = ((tcp_hdr[12] >> 4) & 0x0F) * 4
            tcp_flags = tcp_hdr[13]
            payload_bytes = data[l4_offset + tcp_data_offset:]
            protocol = self._identify_tcp_proto(src_port, dst_port, payload_bytes)

        # UDP (17)
        elif protocol_num == 17 and len(data) >= l4_offset + 8:
            udp_hdr = data[l4_offset:l4_offset + 8]
            src_port, dst_port = struct.unpack("!HH", udp_hdr[:4])
            payload_bytes = data[l4_offset + 8:]
            protocol = self._identify_udp_proto(src_port, dst_port, payload_bytes)

        res: dict[str, object] = {
            "timestamp_ms": ts_ms,
            "src_ip": src_ip,
            "src_port": src_port,
            "dst_ip": dst_ip,
            "dst_port": dst_port,
            "protocol": protocol,
            "length": length,
            "tcp_flags": tcp_flags,
            "payload": payload_bytes
        }

        # Protocol specific telemetry
        if protocol == "mqtt":
            res["mqtt_qos"] = self._extract_mqtt_qos(payload_bytes)
            res["mqtt_payload_bytes"] = len(payload_bytes)
        elif protocol == "coap":
            res["coap_code"] = self._extract_coap_code(payload_bytes)
        elif protocol == "http":
            res["http_status"] = self._extract_http_status(payload_bytes)

        return res

    @staticmethod
    def _identify_tcp_proto(src: int, dst: int, payload: bytes) -> str:
        if 1883 in (src, dst) or 8883 in (src, dst):
            return "mqtt"
        if 80 in (src, dst) or 8080 in (src, dst) or 443 in (src, dst):
            return "http"
        return "tcp"

    @staticmethod
    def _identify_udp_proto(src: int, dst: int, payload: bytes) -> str:
        if 5683 in (src, dst) or 5684 in (src, dst):
            return "coap"
        if 61616 in (src, dst):
            return "6lowpan"
        return "udp"

    @staticmethod
    def _extract_mqtt_qos(payload: bytes) -> int:
        if len(payload) >= 1:
            return (payload[0] >> 1) & 0x03
        return 0

    @staticmethod
    def _extract_coap_code(payload: bytes) -> int:
        if len(payload) >= 2:
            return payload[1]
        return 0

    @staticmethod
    def _extract_http_status(payload: bytes) -> int:
        if payload.startswith(b"HTTP/"):
            parts = payload.split(b" ", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
        return 200 if payload else 0


def replay_pcap_to_flows(pcap_path: str | Path, assembler: FlowAssembler | None = None) -> list[dict[str, object]]:
    """Replay a PCAP capture file into a FlowAssembler and return all reassembled flows."""
    reader = PCAPReader(pcap_path)
    asm = assembler or FlowAssembler(idle_timeout_ms=1000)
    last_ts = 0.0

    for pkt in reader.read_packets():
        last_ts = float(pkt["timestamp_ms"])
        asm.submit(pkt)

    # Flush all remaining active flows past the last packet timestamp
    return asm.expire(last_ts + asm.idle_timeout_ms + 100)
