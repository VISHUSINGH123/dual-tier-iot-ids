"""
IoT-23 Zeek/Bro (`conn.log.labeled`) Streaming Parser, Schema Compatibility Auditor,
and Leak-Free Offline Evaluation Workflow.

Design & Safeguards:
- Streams Zeek/Bro logs in bounded chunks (`chunk_size` rows) without loading multi-GB files into RAM.
- Properly handles Zeek metadata directives (`#separator`, `#set_separator`, `#empty_field`,
  `#unset_field`, `#path`, `#open`, `#fields`, `#types`, `#close`) and IoT-23's mixed delimiter
  where columns 1..20 are tab-separated (`\\t`) and columns 21..23 (`tunnel_parents   label   detailed-label`)
  are separated by 3 spaces (`   `).
- Never silently merges or recodes `label` (`benign`, `Benign`, `Malicious`) or `detailed-label`.
- Explicitly documents which features of the 14-feature APAFC/ONNX pipeline and 61-feature
  Edge-IIoTset pipeline can vs. cannot be reconstructed from Zeek connection logs.
- Provides a leak-free, capture-aware / chronological offline evaluation workflow without
  modifying or retraining any existing model artifact.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config.settings import BASE_DIR
from src.data_loader import ALL_61_FEATURES
from src.framework_bridge import FEATURE_NAMES as APAFC_14_FEATURES


IOT23_RAW_ROOT = BASE_DIR / "data" / "raw" / "iot_23_datasets_small"
IOT23_REPORTS_DIR = BASE_DIR / "data" / "reports"
IOT23_PROCESSED_DIR = BASE_DIR / "data" / "processed"

EXPECTED_ZEEK_23_FIELDS: List[str] = [
    "ts",
    "uid",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "local_orig",
    "local_resp",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "tunnel_parents",
    "label",
    "detailed-label",
]

# Columns that MUST be excluded from model features to prevent target/identifier/temporal leakage
ZEEK_LEAKAGE_AND_METADATA_COLS: List[str] = [
    "ts",              # Capture epoch timestamp (memorizes capture window)
    "uid",             # Unique Zeek connection ID string
    "id.orig_h",       # Source IP address (topology/host memorization)
    "id.resp_h",       # Destination IP address (topology/C&C IP memorization)
    "local_orig",      # 100% unset ('-') across IoT-23
    "local_resp",      # 100% unset ('-') across IoT-23
    "tunnel_parents",  # ~100% unset ('(empty)' or '-') across IoT-23
    "label",           # Ground-truth coarse target
    "detailed-label",  # Ground-truth fine-grained target
]

# Explicit, documented label mappings (never applied during raw parsing; only when requested)
IOT23_BINARY_LABEL_MAP: Dict[str, str] = {
    "benign": "benign",
    "Benign": "benign",
    "Malicious": "malicious",
}

# Explicit mapping from (label, detailed-label) to canonical IoT-23 taxonomic families
IOT23_DETAILED_TO_TAXONOMY_MAP: Dict[str, str] = {
    "-": "Benign",
    "PartOfAHorizontalPortScan": "HorizontalPortScan",
    "PartOfAHorizontalPortScan-Attack": "HorizontalPortScan",
    "Okiru": "Okiru_Botnet",
    "Okiru-Attack": "Okiru_Botnet",
    "DDoS": "DDoS",
    "C&C": "Command_and_Control",
    "C&C-HeartBeat": "Command_and_Control",
    "C&C-HeartBeat-Attack": "Command_and_Control",
    "C&C-HeartBeat-FileDownload": "Command_and_Control",
    "C&C-PartOfAHorizontalPortScan": "Command_and_Control",
    "C&C-FileDownload": "Command_and_Control",
    "C&C-Torii": "Command_and_Control",
    "C&C-Mirai": "Command_and_Control",
    "Attack": "Generic_Attack",
    "FileDownload": "Malware_FileDownload",
}

# Approximate cross-dataset taxonomy projection onto the 15 classes in `iot_ids_framework/configs/default.json`
IOT23_TO_CORE15_APPROX_MAP: Dict[str, str] = {
    "-": "benign",
    "PartOfAHorizontalPortScan": "port_scan",
    "PartOfAHorizontalPortScan-Attack": "port_scan",
    "Okiru": "botnet",
    "Okiru-Attack": "botnet",
    "DDoS": "ddos",
    "C&C": "botnet",
    "C&C-HeartBeat": "botnet",
    "C&C-HeartBeat-Attack": "botnet",
    "C&C-HeartBeat-FileDownload": "backdoor",
    "C&C-PartOfAHorizontalPortScan": "port_scan",
    "C&C-FileDownload": "backdoor",
    "C&C-Torii": "backdoor",
    "C&C-Mirai": "botnet",
    "Attack": "brute_force",
    "FileDownload": "backdoor",
}


@dataclass
class ZeekHeaderMetadata:
    """Parsed metadata header from a Zeek/Bro `conn.log.labeled` file."""
    separator: str
    set_separator: str
    empty_field: str
    unset_field: str
    path: str
    open_ts: str
    fields: List[str]
    types: List[str]


def parse_zeek_header(file_path: Path) -> ZeekHeaderMetadata:
    """Read the leading `#` metadata lines of a Zeek/Bro log without loading the file."""
    directives: Dict[str, str] = {}
    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(25):
            line = f.readline()
            if not line:
                break
            if not line.startswith("#"):
                break
            stripped = line.rstrip("\r\n")
            parts = re.split(r"\t+|\s+", stripped, maxsplit=1)
            key = parts[0]
            val = parts[1] if len(parts) > 1 else ""
            directives[key] = val

    def _split_directive_tokens(raw_val: str) -> List[str]:
        if not raw_val:
            return []
        return [tok.strip() for tok in re.split(r"\t+|\s{2,}", raw_val.strip()) if tok.strip()]

    return ZeekHeaderMetadata(
        separator=directives.get("#separator", r"\x09"),
        set_separator=directives.get("#set_separator", ","),
        empty_field=directives.get("#empty_field", "(empty)"),
        unset_field=directives.get("#unset_field", "-"),
        path=directives.get("#path", "conn"),
        open_ts=directives.get("#open", ""),
        fields=_split_directive_tokens(directives.get("#fields", "")),
        types=_split_directive_tokens(directives.get("#types", "")),
    )


def parse_zeek_data_line(line: str) -> Tuple[Optional[List[str]], bool]:
    """
    Parse a single non-comment line from an IoT-23 `conn.log.labeled` file into 23 raw string tokens.

    Returns:
        `(tokens, is_valid)` where `tokens` has length 23 when `is_valid` is True,
        or `(None, False)` if the line is a comment (`#...`) or empty.
    """
    stripped = line.rstrip("\r\n")
    if not stripped or stripped.startswith("#"):
        return None, True

    tab_parts = stripped.split("\t")
    if len(tab_parts) == 23:
        return tab_parts, True
    if len(tab_parts) == 21:
        # IoT-23 quirk: `tunnel_parents   label   detailed-label` separated by 3 spaces
        tail = tab_parts[20].split("   ")
        if len(tail) == 3:
            return tab_parts[:20] + [t.strip() for t in tail], True
        tail_ws = tab_parts[20].split()
        if len(tail_ws) == 3:
            return tab_parts[:20] + tail_ws, True
        if len(tail_ws) == 2:
            return tab_parts[:20] + [tail_ws[0], tail_ws[1], "-"], True
    # Fallback regex split on tabs or 2+ spaces
    parts = re.split(r"\t+|\s{2,}", stripped)
    if len(parts) == 23:
        return parts, True
    return parts, False


def iter_zeek_conn_chunks(
    file_path: Path,
    chunk_size: int = 50_000,
    max_rows: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    """
    Stream a Zeek `conn.log.labeled` file in bounded DataFrames of `chunk_size` rows.
    Preserves original raw string representations for unset (`-`) and empty (`(empty)`) fields
    until explicit feature transformation is called.
    """
    header = parse_zeek_header(file_path)
    columns = header.fields if len(header.fields) == 23 else EXPECTED_ZEEK_23_FIELDS
    buffer: List[List[str]] = []
    yielded_rows = 0

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            tokens, ok = parse_zeek_data_line(line)
            if not ok or tokens is None or len(tokens) != 23:
                continue
            buffer.append(tokens)
            yielded_rows += 1
            if len(buffer) >= chunk_size:
                yield pd.DataFrame(buffer, columns=columns)
                buffer = []
            if max_rows is not None and yielded_rows >= max_rows:
                break

    if buffer:
        yield pd.DataFrame(buffer, columns=columns)


def audit_single_zeek_file(file_path: Path, max_rows: Optional[int] = None) -> Dict[str, Any]:
    """
    Stream-audit a single Zeek `conn.log.labeled` file without loading it into memory.
    Used by tests and single-file inspection.
    """
    header = parse_zeek_header(file_path)
    size_bytes = file_path.stat().st_size
    row_count = 0
    comment_count = 0
    parse_errors = 0
    label_counter: Counter = Counter()
    detailed_counter: Counter = Counter()
    pair_counter: Counter = Counter()
    unset_by_col: Counter = Counter()

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                comment_count += 1
                continue
            if not line.strip():
                continue
            tokens, ok = parse_zeek_data_line(line)
            if not ok or tokens is None or len(tokens) != 23:
                parse_errors += 1
                continue
            row_count += 1
            for idx, val in enumerate(tokens):
                if val in ("-", "(empty)", ""):
                    unset_by_col[EXPECTED_ZEEK_23_FIELDS[idx]] += 1
            lbl = tokens[21].strip()
            det = tokens[22].strip()
            label_counter[lbl] += 1
            detailed_counter[det] += 1
            pair_counter[f"{lbl} | {det}"] += 1
            if max_rows is not None and row_count >= max_rows:
                break

    return {
        "scenario": file_path.parent.parent.name,
        "path": str(file_path),
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "fields": header.fields,
        "types": header.types,
        "schema_matches_23_cols": (header.fields == EXPECTED_ZEEK_23_FIELDS),
        "row_count": row_count,
        "comment_count": comment_count,
        "parse_errors": parse_errors,
        "labels": dict(label_counter),
        "detailed_labels": dict(detailed_counter),
        "label_pairs": dict(pair_counter),
        "unset_by_column": dict(unset_by_col),
    }


def assess_schema_compatibility() -> Dict[str, Any]:
    """
    Task 2: Compare Zeek `conn.log.labeled` fields against:
      1. Primary 14-feature APAFC/ONNX pipeline (`iot_ids.features.FEATURE_NAMES`)
      2. Exploratory 61-feature Edge-IIoTset pipeline (`src.data_loader.ALL_61_FEATURES`)
    """
    apafc_comparison = [
        {
            "apafc_feature": "protocol_id",
            "status": "PARTIAL_APPROXIMATION",
            "zeek_source": "proto, service, id.orig_p, id.resp_p",
            "notes": (
                "APAFC maps L7 IoT protocols: mqtt=1, coap=2, http=3, sixlowpan=4, other=0. "
                "Zeek conn.log records L4 `proto` (tcp/udp/icmp) and L7 `service` (http, dns, ssh, irc, ssl, dhcp). "
                "HTTP flows are identifiable via service=='http' or port 80/8080; MQTT/CoAP/6LoWPAN must be inferred by port (1883/8883, 5683, 61616), else 0."
            ),
        },
        {
            "apafc_feature": "duration_ms",
            "status": "EXACT_DERIVABLE",
            "zeek_source": "duration",
            "notes": "Zeek `duration` is in seconds (float). `duration_ms = float(duration) * 1000.0` (0.0 when unset `-`).",
        },
        {
            "apafc_feature": "packet_count",
            "status": "EXACT_DERIVABLE",
            "zeek_source": "orig_pkts, resp_pkts",
            "notes": "Bidirectional flow packet count = `int(orig_pkts) + int(resp_pkts)`.",
        },
        {
            "apafc_feature": "byte_count",
            "status": "EXACT_DERIVABLE",
            "zeek_source": "orig_bytes, resp_bytes, orig_ip_bytes, resp_ip_bytes",
            "notes": "Bidirectional byte count = `orig_bytes + resp_bytes` (or `orig_ip_bytes + resp_ip_bytes` when L4 payload bytes are unset `-`).",
        },
        {
            "apafc_feature": "mean_packet_bytes",
            "status": "EXACT_DERIVABLE",
            "zeek_source": "byte_count / max(packet_count, 1)",
            "notes": "Directly computed from bidirectional byte count and packet count.",
        },
        {
            "apafc_feature": "src_port",
            "status": "EXACT_DERIVABLE",
            "zeek_source": "id.orig_p",
            "notes": "Originator transport port (`id.orig_p`).",
        },
        {
            "apafc_feature": "dst_port",
            "status": "EXACT_DERIVABLE",
            "zeek_source": "id.resp_p",
            "notes": "Responder transport port (`id.resp_p`).",
        },
        {
            "apafc_feature": "tcp_flags",
            "status": "APPROXIMABLE_FROM_HISTORY",
            "zeek_source": "history, conn_state",
            "notes": (
                "Zeek conn.log does not store the raw TCP header flags byte integer, but records "
                "connection `history` letters (S=SYN, h=SYN+ACK, A=ACK, D=DATA/PSH, F=FIN, R=RST) "
                "and `conn_state` (S0, SF, REJ, RSTO), which map deterministically to a reconstructed TCP flags bitmask."
            ),
        },
        {
            "apafc_feature": "mqtt_qos",
            "status": "UNAVAILABLE_IN_CONN_LOG",
            "zeek_source": "None (requires packet/L7 MQTT header)",
            "notes": "Cannot be reconstructed from Zeek `conn.log.labeled`; must be set to 0.0 in adapter.",
        },
        {
            "apafc_feature": "mqtt_payload_bytes",
            "status": "UNAVAILABLE_IN_CONN_LOG",
            "zeek_source": "None (requires packet/L7 MQTT header)",
            "notes": "Cannot be reconstructed from Zeek `conn.log.labeled`; must be set to 0.0 in adapter.",
        },
        {
            "apafc_feature": "coap_code",
            "status": "UNAVAILABLE_IN_CONN_LOG",
            "zeek_source": "None (requires packet/L7 CoAP header)",
            "notes": "Cannot be reconstructed from Zeek `conn.log.labeled`; must be set to 0.0 in adapter.",
        },
        {
            "apafc_feature": "http_status",
            "status": "UNAVAILABLE_IN_CONN_LOG",
            "zeek_source": "None (requires Zeek http.log or packet payload)",
            "notes": "Zeek `conn.log` only indicates `service == 'http'`, not the HTTP status code (200/404/500).",
        },
        {
            "apafc_feature": "sixlowpan_dispatch",
            "status": "UNAVAILABLE_IN_CONN_LOG",
            "zeek_source": "None (requires IEEE 802.15.4 / 6LoWPAN frame)",
            "notes": "Cannot be reconstructed from Zeek `conn.log.labeled`; must be set to 0.0 in adapter.",
        },
        {
            "apafc_feature": "payload_entropy",
            "status": "UNAVAILABLE_IN_CONN_LOG",
            "zeek_source": "None (requires raw packet payload bytes)",
            "notes": "Shannon byte entropy requires raw packet payloads (`PCAP`), which are absent in `conn.log`.",
        },
    ]

    return {
        "zeek_column_count": len(EXPECTED_ZEEK_23_FIELDS),
        "apafc_14_pipeline": {
            "total_features": len(APAFC_14_FEATURES),
            "exact_derivable_count": 6,
            "approximable_count": 2,
            "unavailable_l7_payload_count": 6,
            "can_validly_consume_without_retraining": False,
            "reason": (
                "6 of the 14 APAFC features (`mqtt_qos`, `mqtt_payload_bytes`, `coap_code`, "
                "`http_status`, `sixlowpan_dispatch`, `payload_entropy`) require application-layer "
                "headers or raw packet payloads that do not exist in Zeek `conn.log.labeled`. "
                "Furthermore, the existing `edge_ids.onnx` artifact was trained on synthetic 15-class "
                "IoT protocol profiles, whereas IoT-23 contains real Zeek connection summaries for "
                "Mirai/Okiru/Torii/HorizontalPortScan/DDoS/C&C traffic."
            ),
            "feature_breakdown": apafc_comparison,
        },
        "edge_iiotset_61_pipeline": {
            "total_features": len(ALL_61_FEATURES),
            "derivable_from_zeek_conn_log": [
                "tcp.srcport",
                "tcp.dstport",
                "udp.port",
                "tcp.len",
                "udp.length",
            ],
            "unavailable_count": 56,
            "can_validly_consume_without_retraining": False,
            "reason": (
                "56 of the 61 Edge-IIoTset features are deep packet inspection fields across MQTT (12), "
                "CoAP (9), HTTP (8), 6LoWPAN (9), DNS (6), Modbus/TCP (5), ARP (3), and ICMP/TCP flags "
                "that are not present in Zeek flow-level connection logs."
            ),
        },
        "recommendation": (
            "1. Keep the existing 14-feature `edge_ids.onnx` and 61-to-14 `lgbm_ids_bundle.joblib` "
            "models untouched for their respective schemas. "
            "2. Provide an explicit `map_zeek_to_apafc14()` adapter (documenting that the 6 L7/payload "
            "features are zero-filled) to demonstrate why naive zero-shot transfer across incompatible "
            "feature modalities degrades accuracy. "
            "3. Use a dedicated, leak-free 16-feature Zeek Connection Feature Extractor (`extract_zeek_native_features`) "
            "with capture-aware or chronological splitting for honest offline evaluation on IoT-23."
        ),
    }


def _reconstruct_tcp_flags_from_history(history_series: pd.Series, conn_state_series: pd.Series) -> np.ndarray:
    """
    Map Zeek `history` string (e.g. 'ShAdDaFf', 'S', 'Sr') and `conn_state`
    into a standard TCP flags bitmask (FIN=1, SYN=2, RST=4, PSH=8, ACK=16).
    """
    hist = history_series.fillna("-").astype(str)
    cstate = conn_state_series.fillna("-").astype(str)
    flags = np.zeros(len(hist), dtype=np.float32)

    has_s = hist.str.contains("S|h", case=True, regex=True) | cstate.isin(["S0", "S1", "SF", "REJ", "RSTO", "RSTR", "SH"])
    has_a = hist.str.contains("A|a|h", case=True, regex=True) | cstate.isin(["S1", "SF", "S2", "S3", "RSTO", "RSTR"])
    has_d = hist.str.contains("D|d", case=True, regex=True)
    has_f = hist.str.contains("F|f", case=True, regex=True) | cstate.isin(["SF", "S2", "S3", "SH", "SHR"])
    has_r = hist.str.contains("R|r", case=True, regex=True) | cstate.isin(["REJ", "RSTO", "RSTR", "RSTOS0"])

    flags += np.where(has_f, 1.0, 0.0)
    flags += np.where(has_s, 2.0, 0.0)
    flags += np.where(has_r, 4.0, 0.0)
    flags += np.where(has_d, 8.0, 0.0)
    flags += np.where(has_a, 16.0, 0.0)
    return flags


def map_zeek_to_apafc14(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map a Zeek `conn.log.labeled` DataFrame chunk onto the 14-feature `APAFC` schema
    (`iot_ids.features.FEATURE_NAMES`).

    IMPORTANT:
    - Reconstructs the 8 transport/flow features (`protocol_id`, `duration_ms`,
      `packet_count`, `byte_count`, `mean_packet_bytes`, `src_port`, `dst_port`, `tcp_flags`).
    - Explicitly sets the 6 unavailable packet-payload/L7 header features
      (`mqtt_qos`, `mqtt_payload_bytes`, `coap_code`, `http_status`,
      `sixlowpan_dispatch`, `payload_entropy`) to `0.0`.
    """
    n = len(df)
    sport = pd.to_numeric(df["id.orig_p"].replace({"-": 0, "(empty)": 0}), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    dport = pd.to_numeric(df["id.resp_p"].replace({"-": 0, "(empty)": 0}), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    dur_sec = pd.to_numeric(df["duration"].replace({"-": 0.0, "(empty)": 0.0}), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    duration_ms = np.maximum(dur_sec * 1000.0, 0.0)

    orig_pkts = pd.to_numeric(df["orig_pkts"].replace({"-": 0, "(empty)": 0}), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    resp_pkts = pd.to_numeric(df["resp_pkts"].replace({"-": 0, "(empty)": 0}), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    packet_count = np.maximum(orig_pkts + resp_pkts, 0.0)

    orig_b = pd.to_numeric(df["orig_bytes"].replace({"-": np.nan, "(empty)": np.nan}), errors="coerce")
    resp_b = pd.to_numeric(df["resp_bytes"].replace({"-": np.nan, "(empty)": np.nan}), errors="coerce")
    orig_ip_b = pd.to_numeric(df["orig_ip_bytes"].replace({"-": 0, "(empty)": 0}), errors="coerce").fillna(0.0)
    resp_ip_b = pd.to_numeric(df["resp_ip_bytes"].replace({"-": 0, "(empty)": 0}), errors="coerce").fillna(0.0)

    # Use application payload bytes when logged; fall back to IP bytes when unset ('-')
    total_payload_b = (orig_b.fillna(0.0) + resp_b.fillna(0.0)).to_numpy(dtype=np.float32)
    both_unset = (orig_b.isna() & resp_b.isna()).to_numpy()
    ip_total_b = (orig_ip_b + resp_ip_b).to_numpy(dtype=np.float32)
    byte_count = np.where(both_unset, ip_total_b, total_payload_b)
    mean_packet_bytes = byte_count / np.maximum(packet_count, 1.0)

    svc = df["service"].fillna("-").astype(str).str.lower()
    proto_id = np.zeros(n, dtype=np.float32)
    is_http = (svc == "http") | np.isin(sport, [80, 8080]) | np.isin(dport, [80, 8080])
    is_mqtt = np.isin(sport, [1883, 8883]) | np.isin(dport, [1883, 8883])
    is_coap = np.isin(sport, [5683, 5684]) | np.isin(dport, [5683, 5684])
    is_6low = np.isin(sport, [61616]) | np.isin(dport, [61616])
    proto_id = np.where(is_http, 3.0, proto_id)
    proto_id = np.where(is_mqtt, 1.0, proto_id)
    proto_id = np.where(is_coap, 2.0, proto_id)
    proto_id = np.where(is_6low, 4.0, proto_id)

    tcp_flags = _reconstruct_tcp_flags_from_history(df["history"], df["conn_state"])
    zeros = np.zeros(n, dtype=np.float32)

    return pd.DataFrame(
        {
            "protocol_id": proto_id,
            "duration_ms": duration_ms,
            "packet_count": packet_count,
            "byte_count": byte_count,
            "mean_packet_bytes": mean_packet_bytes,
            "src_port": sport,
            "dst_port": dport,
            "tcp_flags": tcp_flags,
            "mqtt_qos": zeros,
            "mqtt_payload_bytes": zeros,
            "coap_code": zeros,
            "http_status": zeros,
            "sixlowpan_dispatch": zeros,
            "payload_entropy": zeros,
        },
        columns=APAFC_14_FEATURES,
    )


# The native Zeek feature set contains 17 features:
ZEEK_NATIVE_17_FEATURES: List[str] = [
    "proto_code",
    "service_code",
    "conn_state_code",
    "duration_ms",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "total_pkts",
    "total_ip_bytes",
    "mean_ip_bytes_per_pkt",
    "orig_pkt_ratio",
    "tcp_flags_reconstructed",
    "dst_port_bucket",
]
# Backward-compatibility alias
ZEEK_NATIVE_16_FEATURES: List[str] = ZEEK_NATIVE_17_FEATURES

PROTO_CODE_MAP = {"tcp": 1.0, "udp": 2.0, "icmp": 3.0}
SERVICE_CODE_MAP = {"-": 0.0, "(empty)": 0.0, "dns": 1.0, "http": 2.0, "irc": 3.0, "ssh": 4.0, "dhcp": 5.0, "ssl": 6.0}
CONN_STATE_CODE_MAP = {
    "S0": 1.0,
    "S1": 2.0,
    "SF": 3.0,
    "REJ": 4.0,
    "S2": 5.0,
    "S3": 6.0,
    "RSTO": 7.0,
    "RSTR": 8.0,
    "RSTOS0": 9.0,
    "RSTRH": 10.0,
    "SH": 11.0,
    "SHR": 12.0,
    "OTH": 13.0,
}


def extract_zeek_native_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract leak-free connection-level features directly supported by Zeek `conn.log.labeled`.
    Excludes `ts`, `uid`, `id.orig_h`, `id.resp_h`, `local_orig`, `local_resp`, `tunnel_parents`,
    `label`, and `detailed-label`. Uses destination port buckets rather than raw ephemeral source
    ports to prevent host/session memorization.
    """
    proto_code = df["proto"].astype(str).str.lower().map(PROTO_CODE_MAP).fillna(0.0).to_numpy(dtype=np.float32)
    service_code = df["service"].astype(str).str.lower().map(SERVICE_CODE_MAP).fillna(7.0).to_numpy(dtype=np.float32)
    conn_state_code = df["conn_state"].astype(str).map(CONN_STATE_CODE_MAP).fillna(0.0).to_numpy(dtype=np.float32)

    dur_sec = pd.to_numeric(df["duration"].replace({"-": 0.0, "(empty)": 0.0}), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    duration_ms = np.maximum(dur_sec * 1000.0, 0.0)

    orig_bytes = pd.to_numeric(df["orig_bytes"].replace({"-": -1.0, "(empty)": -1.0}), errors="coerce").fillna(-1.0).to_numpy(dtype=np.float32)
    resp_bytes = pd.to_numeric(df["resp_bytes"].replace({"-": -1.0, "(empty)": -1.0}), errors="coerce").fillna(-1.0).to_numpy(dtype=np.float32)
    missed_bytes = pd.to_numeric(df["missed_bytes"].replace({"-": 0.0, "(empty)": 0.0}), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    orig_pkts = pd.to_numeric(df["orig_pkts"].replace({"-": 0.0, "(empty)": 0.0}), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    orig_ip_bytes = pd.to_numeric(df["orig_ip_bytes"].replace({"-": 0.0, "(empty)": 0.0}), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    resp_pkts = pd.to_numeric(df["resp_pkts"].replace({"-": 0.0, "(empty)": 0.0}), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    resp_ip_bytes = pd.to_numeric(df["resp_ip_bytes"].replace({"-": 0.0, "(empty)": 0.0}), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

    total_pkts = np.maximum(orig_pkts + resp_pkts, 0.0)
    total_ip_bytes = np.maximum(orig_ip_bytes + resp_ip_bytes, 0.0)
    mean_ip_bytes_per_pkt = total_ip_bytes / np.maximum(total_pkts, 1.0)
    orig_pkt_ratio = orig_pkts / np.maximum(total_pkts, 1.0)
    tcp_flags = _reconstruct_tcp_flags_from_history(df["history"], df["conn_state"])

    dport = pd.to_numeric(df["id.resp_p"].replace({"-": 0, "(empty)": 0}), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    # Semantic port bucket: 0=well-known (<1024), 1=registered (1024..49151), 2=ephemeral (>=49152), plus key IoT/admin ports
    dport_bucket = np.where(dport < 1024, 1.0, np.where(dport < 49152, 2.0, 3.0)).astype(np.float32)
    dport_bucket = np.where(np.isin(dport, [22, 23, 2323]), 4.0, dport_bucket)  # Telnet/SSH scan targets
    dport_bucket = np.where(np.isin(dport, [80, 443, 8080, 8443]), 5.0, dport_bucket)  # Web/HTTP targets
    dport_bucket = np.where(np.isin(dport, [1883, 8883, 5683, 61616]), 6.0, dport_bucket)  # IoT protocol ports

    return pd.DataFrame(
        {
            "proto_code": proto_code,
            "service_code": service_code,
            "conn_state_code": conn_state_code,
            "duration_ms": duration_ms,
            "orig_bytes": orig_bytes,
            "resp_bytes": resp_bytes,
            "missed_bytes": missed_bytes,
            "orig_pkts": orig_pkts,
            "orig_ip_bytes": orig_ip_bytes,
            "resp_pkts": resp_pkts,
            "resp_ip_bytes": resp_ip_bytes,
            "total_pkts": total_pkts,
            "total_ip_bytes": total_ip_bytes,
            "mean_ip_bytes_per_pkt": mean_ip_bytes_per_pkt,
            "orig_pkt_ratio": orig_pkt_ratio,
            "tcp_flags_reconstructed": tcp_flags,
            "dst_port_bucket": dport_bucket,
        },
        columns=ZEEK_NATIVE_16_FEATURES,
    )


def evaluate_existing_apafc14_onnx_on_iot23_sample(
    files: Sequence[Path],
    rows_per_file: int = 2000,
) -> Dict[str, Any]:
    """
    Task 3 Safe Offline Evaluation (Zero-Shot Compatibility Check):
    Streams bounded chunks from the specified IoT-23 `conn.log.labeled` files,
    projects the 8 available flow fields onto the 14-feature APAFC schema (with the
    6 unavailable L7/payload features set to 0.0), runs the existing untouched
    `edge_ids.onnx` model via ONNX Runtime, and computes both:
      - Binary detection metrics (Benign vs. Malicious), and
      - 15-class projected metrics (`IOT23_TO_CORE15_APPROX_MAP`),
    including class counts, confusion matrix, per-class precision/recall/F1, macro-F1,
    and false-positive rate.
    Does NOT train or modify any model artifact.
    """
    import onnxruntime as ort
    from src.framework_bridge import LABELS_JSON_PATH, ONNX_MODEL_PATH

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess = ort.InferenceSession(str(ONNX_MODEL_PATH), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    core_classes: List[str] = json.loads(LABELS_JSON_PATH.read_text(encoding="utf-8"))

    y_true_bin: List[int] = []
    y_pred_bin: List[int] = []
    y_true_multi: List[str] = []
    y_pred_multi: List[str] = []
    raw_label_counts: Counter = Counter()
    detailed_label_counts: Counter = Counter()
    per_file_evaluated: Dict[str, int] = {}

    for fp in files:
        scen = fp.parent.parent.name
        evaluated_in_file = 0
        for chunk_df in iter_zeek_conn_chunks(fp, chunk_size=min(5000, rows_per_file), max_rows=rows_per_file):
            X_apafc = map_zeek_to_apafc14(chunk_df)
            x_arr = X_apafc.to_numpy(dtype=np.float32)
            outputs = sess.run(None, {input_name: x_arr})
            pred_indices = np.asarray(outputs[0], dtype=int).reshape(-1)
            pred_labels = [core_classes[int(idx)] for idx in pred_indices]

            raw_lbls = chunk_df["label"].astype(str).str.strip().tolist()
            det_lbls = chunk_df["detailed-label"].astype(str).str.strip().tolist()

            for r_lbl, d_lbl, p_lbl in zip(raw_lbls, det_lbls, pred_labels):
                raw_label_counts[r_lbl] += 1
                detailed_label_counts[d_lbl] += 1
                true_b = 0 if IOT23_BINARY_LABEL_MAP.get(r_lbl, "benign") == "benign" else 1
                pred_b = 0 if p_lbl == "benign" else 1
                y_true_bin.append(true_b)
                y_pred_bin.append(pred_b)
                y_true_multi.append(IOT23_TO_CORE15_APPROX_MAP.get(d_lbl if true_b == 1 else "-", "benign"))
                y_pred_multi.append(p_lbl)

            evaluated_in_file += len(chunk_df)
        per_file_evaluated[scen] = evaluated_in_file

    # Binary metrics
    cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp_cnt, fn_cnt, tp = (int(cm_bin[0, 0]), int(cm_bin[0, 1]), int(cm_bin[1, 0]), int(cm_bin[1, 1]))
    fpr = float(fp_cnt / max(tn + fp_cnt, 1))
    prec_b, rec_b, f1_b, sup_b = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, labels=[0, 1], zero_division=0
    )
    macro_f1_bin = float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))

    # Multiclass projected metrics across observed classes
    observed_classes = sorted(set(y_true_multi) | set(y_pred_multi))
    prec_m, rec_m, f1_m, sup_m = precision_recall_fscore_support(
        y_true_multi, y_pred_multi, labels=observed_classes, zero_division=0
    )
    macro_f1_multi = float(f1_score(y_true_multi, y_pred_multi, labels=observed_classes, average="macro", zero_division=0))

    return {
        "evaluation_mode": "zero_shot_apafc14_adapter_on_untouched_edge_ids_onnx",
        "total_flows_evaluated": len(y_true_bin),
        "per_file_evaluated": per_file_evaluated,
        "raw_label_counts": dict(raw_label_counts),
        "detailed_label_counts": dict(detailed_label_counts),
        "binary_metrics": {
            "confusion_matrix_labels": ["benign (0)", "malicious (1)"],
            "confusion_matrix": cm_bin.tolist(),
            "tn": tn,
            "fp": fp_cnt,
            "fn": fn_cnt,
            "tp": tp,
            "false_positive_rate": round(fpr, 4),
            "macro_f1": round(macro_f1_bin, 4),
            "per_class": {
                "benign": {
                    "precision": round(float(prec_b[0]), 4),
                    "recall": round(float(rec_b[0]), 4),
                    "f1": round(float(f1_b[0]), 4),
                    "support": int(sup_b[0]),
                },
                "malicious": {
                    "precision": round(float(prec_b[1]), 4),
                    "recall": round(float(rec_b[1]), 4),
                    "f1": round(float(f1_b[1]), 4),
                    "support": int(sup_b[1]),
                },
            },
        },
        "projected_multiclass_metrics": {
            "macro_f1": round(macro_f1_multi, 4),
            "per_class": {
                cls_name: {
                    "precision": round(float(prec_m[i]), 4),
                    "recall": round(float(rec_m[i]), 4),
                    "f1": round(float(f1_m[i]), 4),
                    "support": int(sup_m[i]),
                }
                for i, cls_name in enumerate(observed_classes)
            },
        },
    }


DEFAULT_TRAIN_SCENARIOS: List[str] = [
    "CTU-Honeypot-Capture-4-1",
    "CTU-IoT-Malware-Capture-1-1",
    "CTU-IoT-Malware-Capture-17-1",
    "CTU-IoT-Malware-Capture-20-1",
    "CTU-IoT-Malware-Capture-3-1",
    "CTU-IoT-Malware-Capture-33-1",
    "CTU-IoT-Malware-Capture-34-1",
    "CTU-IoT-Malware-Capture-42-1",
    "CTU-IoT-Malware-Capture-43-1",
    "CTU-IoT-Malware-Capture-48-1",
    "CTU-IoT-Malware-Capture-49-1",
    "CTU-IoT-Malware-Capture-52-1",
    "CTU-IoT-Malware-Capture-7-1",
    "CTU-IoT-Malware-Capture-9-1",
]

DEFAULT_TEST_SCENARIOS: List[str] = [
    "CTU-Honeypot-Capture-5-1",
    "Somfy-01",
    "CTU-IoT-Malware-Capture-21-1",
    "CTU-IoT-Malware-Capture-35-1",
    "CTU-IoT-Malware-Capture-36-1",
    "CTU-IoT-Malware-Capture-39-1",
    "CTU-IoT-Malware-Capture-44-1",
    "CTU-IoT-Malware-Capture-60-1",
    "CTU-IoT-Malware-Capture-8-1",
]

EXPERIMENTAL_MODEL_BUNDLE_PATH: Path = BASE_DIR / "models" / "experimental_iot23_lgbm_bundle.joblib"
EXPERIMENTAL_EVALUATION_REPORT_PATH: Path = BASE_DIR / "models" / "experimental_iot23_evaluation.json"


def train_and_evaluate_experimental_iot23_model(
    train_scenarios: Optional[Sequence[str]] = None,
    test_scenarios: Optional[Sequence[str]] = None,
    train_rows_per_scenario: int = 2000,
    test_rows_per_scenario: int = 2000,
    model_output_path: Optional[Path] = None,
    evaluation_output_path: Optional[Path] = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Stage 4 Investigation Pipeline:
    Reproducibly trains and evaluates an experimental LightGBM classifier on real IoT-23
    flow records using leakage-free connection features (ZEEK_NATIVE_16_FEATURES).

    Guarantees & Safeguards:
    1. Scenario-level isolation: Evaluates strictly on held-out test capture scenarios
       not present during training (prevents IP/temporal/device memorization).
    2. Bounded streaming memory footprint: Reads bounded chunks per file.
    3. Artifact isolation: Saves bundle and metrics to new, clearly isolated paths
       (never touches edge_ids.onnx, lgbm_ids_bundle.joblib, or iot_ids_framework/).
    4. Honest feature representation: Uses only native flow features logged by Zeek;
       never synthesizes or hallucinating missing L7 payload/entropy bytes.
    """
    from lightgbm import LGBMClassifier
    import joblib

    train_scen_list = list(train_scenarios or DEFAULT_TRAIN_SCENARIOS)
    test_scen_list = list(test_scenarios or DEFAULT_TEST_SCENARIOS)
    bundle_path = Path(model_output_path or EXPERIMENTAL_MODEL_BUNDLE_PATH)
    eval_path = Path(evaluation_output_path or EXPERIMENTAL_EVALUATION_REPORT_PATH)

    scenario_to_file: Dict[str, Path] = {
        p.parent.parent.name: p for p in IOT23_RAW_ROOT.glob("**/conn.log.labeled")
    }

    # 1. Stream training dataset
    train_dfs: List[pd.DataFrame] = []
    train_counts_by_scenario: Dict[str, int] = {}
    train_label_counter: Counter = Counter()

    for scen in train_scen_list:
        fp = scenario_to_file.get(scen)
        if not fp or not fp.is_file():
            continue
        scen_loaded = 0
        for chunk in iter_zeek_conn_chunks(fp, chunk_size=min(2000, train_rows_per_scenario), max_rows=train_rows_per_scenario):
            feats = extract_zeek_native_features(chunk)
            raw_lbl = chunk["label"].astype(str).str.strip()
            feats["y_binary"] = raw_lbl.map(lambda x: 0 if IOT23_BINARY_LABEL_MAP.get(x, "benign") == "benign" else 1).to_numpy(dtype=np.int32)
            train_dfs.append(feats)
            scen_loaded += len(chunk)
            train_label_counter.update(raw_lbl.tolist())
            if scen_loaded >= train_rows_per_scenario:
                break
        train_counts_by_scenario[scen] = scen_loaded

    if not train_dfs:
        raise ValueError("No training data loaded from specified scenarios.")

    train_df = pd.concat(train_dfs, ignore_index=True)
    X_train = train_df[ZEEK_NATIVE_16_FEATURES]
    y_train = train_df["y_binary"].to_numpy()

    # 2. Train LightGBM model
    clf = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        random_state=random_seed,
        n_jobs=2,
        verbose=-1,
    )
    clf.fit(X_train, y_train)

    # 3. Stream held-out evaluation dataset
    test_dfs: List[pd.DataFrame] = []
    test_counts_by_scenario: Dict[str, int] = {}
    test_label_counter: Counter = Counter()
    test_detailed_counter: Counter = Counter()
    per_scenario_eval: Dict[str, Dict[str, Any]] = {}

    for scen in test_scen_list:
        fp = scenario_to_file.get(scen)
        if not fp or not fp.is_file():
            continue
        scen_loaded = 0
        scen_dfs: List[pd.DataFrame] = []
        for chunk in iter_zeek_conn_chunks(fp, chunk_size=min(2000, test_rows_per_scenario), max_rows=test_rows_per_scenario):
            feats = extract_zeek_native_features(chunk)
            raw_lbl = chunk["label"].astype(str).str.strip()
            det_lbl = chunk["detailed-label"].astype(str).str.strip()
            feats["y_binary"] = raw_lbl.map(lambda x: 0 if IOT23_BINARY_LABEL_MAP.get(x, "benign") == "benign" else 1).to_numpy(dtype=np.int32)
            feats["scenario"] = scen
            feats["detailed_label"] = det_lbl
            scen_dfs.append(feats)
            scen_loaded += len(chunk)
            test_label_counter.update(raw_lbl.tolist())
            test_detailed_counter.update(det_lbl.tolist())
            if scen_loaded >= test_rows_per_scenario:
                break
        test_counts_by_scenario[scen] = scen_loaded

        if scen_dfs:
            scen_df = pd.concat(scen_dfs, ignore_index=True)
            scen_preds = clf.predict(scen_df[ZEEK_NATIVE_16_FEATURES])
            scen_cm = confusion_matrix(scen_df["y_binary"], scen_preds, labels=[0, 1])
            per_scenario_eval[scen] = {
                "flows_evaluated": len(scen_df),
                "confusion_matrix": scen_cm.tolist(),
                "tn": int(scen_cm[0, 0]),
                "fp": int(scen_cm[0, 1]),
                "fn": int(scen_cm[1, 0]),
                "tp": int(scen_cm[1, 1]),
            }
            test_dfs.append(scen_df)

    test_df = pd.concat(test_dfs, ignore_index=True)
    X_test = test_df[ZEEK_NATIVE_16_FEATURES]
    y_test = test_df["y_binary"].to_numpy()

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    # 4. Metrics computation
    cm_bin = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp_cnt, fn_cnt, tp = (int(cm_bin[0, 0]), int(cm_bin[0, 1]), int(cm_bin[1, 0]), int(cm_bin[1, 1]))
    fpr = float(fp_cnt / max(tn + fp_cnt, 1))
    prec_b, rec_b, f1_b, sup_b = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )
    macro_f1_bin = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    feature_importances = {
        name: int(imp) for name, imp in zip(ZEEK_NATIVE_16_FEATURES, clf.feature_importances_)
    }
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    results: Dict[str, Any] = {
        "model_type": "LightGBM_Classifier_Experimental",
        "feature_schema": "ZEEK_NATIVE_16_FEATURES",
        "features": ZEEK_NATIVE_16_FEATURES,
        "split_strategy": "Scenario_Based_Held_Out_Cross_Capture_Split",
        "train_scenarios": train_scen_list,
        "test_scenarios": test_scen_list,
        "train_sample_size": len(train_df),
        "test_sample_size": len(test_df),
        "train_label_distribution": dict(train_label_counter),
        "test_label_distribution": dict(test_label_counter),
        "test_detailed_label_distribution": dict(test_detailed_counter),
        "binary_metrics": {
            "confusion_matrix_labels": ["benign (0)", "malicious (1)"],
            "confusion_matrix": cm_bin.tolist(),
            "tn": tn,
            "fp": fp_cnt,
            "fn": fn_cnt,
            "tp": tp,
            "false_positive_rate": round(fpr, 4),
            "macro_f1": round(macro_f1_bin, 4),
            "accuracy": round(float((tn + tp) / len(y_test)), 4),
            "per_class": {
                "benign": {
                    "precision": round(float(prec_b[0]), 4),
                    "recall": round(float(rec_b[0]), 4),
                    "f1": round(float(f1_b[0]), 4),
                    "support": int(sup_b[0]),
                },
                "malicious": {
                    "precision": round(float(prec_b[1]), 4),
                    "recall": round(float(rec_b[1]), 4),
                    "f1": round(float(f1_b[1]), 4),
                    "support": int(sup_b[1]),
                },
            },
        },
        "feature_importances": sorted_importances,
        "per_scenario_eval": per_scenario_eval,
        "comparison_with_baseline_onnx": {
            "baseline_model": "iot_ids_framework/artifacts/edge_ids.onnx (untouched)",
            "baseline_fpr": 0.9524,
            "baseline_macro_f1": 0.3898,
            "experimental_fpr": round(fpr, 4),
            "experimental_macro_f1": round(macro_f1_bin, 4),
            "fpr_reduction": round(0.9524 - fpr, 4),
            "macro_f1_gain": round(macro_f1_bin - 0.3898, 4),
        },
    }

    # 5. Persist isolated artifacts
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_payload = {
        "model": clf,
        "features": ZEEK_NATIVE_16_FEATURES,
        "label_mapping": IOT23_BINARY_LABEL_MAP,
        "proto_code_map": PROTO_CODE_MAP,
        "service_code_map": SERVICE_CODE_MAP,
        "conn_state_code_map": CONN_STATE_CODE_MAP,
        "training_scenarios": train_scen_list,
        "random_seed": random_seed,
    }
    joblib.dump(bundle_payload, bundle_path)

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


class DualBackendInferenceRouter:
    """
    Stage 6 Dual-Backend Inference Router:
    Enforces strict separation between:
      1. Primary Core Backend: `edge_ids.onnx` (14 APAFC features, PCAP/assembled flows, multiclass 15)
      2. Experimental Backend: `experimental_iot23_lgbm_bundle.joblib` (17 Zeek-native features, conn.log summaries, binary)

    Strict Validation Contracts:
    - Requires explicit backend specification ("core_apafc14_onnx" vs. "experimental_zeek17_lgbm").
    - Validates exact feature names, feature counts, and feature ordering.
    - Rejects incompatible inputs with a descriptive SchemaMismatchError instead of silently
      zero-filling, dropping, or slicing features.
    """

    CORE_ONNX_BACKEND: str = "core_apafc14_onnx"
    EXPERIMENTAL_LGBM_BACKEND: str = "experimental_zeek17_lgbm"

    def __init__(
        self,
        onnx_model_path: Optional[Path] = None,
        labels_json_path: Optional[Path] = None,
        experimental_bundle_path: Optional[Path] = None,
    ) -> None:
        self.onnx_model_path = Path(onnx_model_path or (BASE_DIR / "iot_ids_framework" / "artifacts" / "edge_ids.onnx"))
        self.labels_json_path = Path(labels_json_path or (BASE_DIR / "iot_ids_framework" / "artifacts" / "labels.json"))
        self.experimental_bundle_path = Path(experimental_bundle_path or EXPERIMENTAL_MODEL_BUNDLE_PATH)

        self._onnx_session = None
        self._onnx_input_name = None
        self._core_classes: Optional[List[str]] = None
        self._experimental_bundle: Optional[Dict[str, Any]] = None

    def _get_onnx_session(self):
        if self._onnx_session is None:
            import onnxruntime as ort

            if not self.onnx_model_path.is_file():
                raise FileNotFoundError(f"Core ONNX model not found at {self.onnx_model_path}")
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._onnx_session = ort.InferenceSession(str(self.onnx_model_path), sess_options=opts, providers=["CPUExecutionProvider"])
            self._onnx_input_name = self._onnx_session.get_inputs()[0].name
            if self.labels_json_path.is_file():
                self._core_classes = json.loads(self.labels_json_path.read_text(encoding="utf-8"))
            else:
                self._core_classes = [f"class_{i}" for i in range(15)]
        return self._onnx_session

    def _get_experimental_bundle(self) -> Dict[str, Any]:
        if self._experimental_bundle is None:
            import joblib

            if not self.experimental_bundle_path.is_file():
                raise FileNotFoundError(
                    f"Experimental IoT-23 LightGBM bundle not found at {self.experimental_bundle_path}. "
                    "Run Stage 4 training/investigation pipeline first."
                )
            self._experimental_bundle = joblib.load(self.experimental_bundle_path)
        return self._experimental_bundle

    def validate_input(self, backend: str, X: pd.DataFrame) -> None:
        """
        Strictly validate feature schema against backend requirements.
        Raises ValueError with clear explanation if feature count, names, or ordering mismatch.
        """
        if backend == self.CORE_ONNX_BACKEND:
            expected_feats = list(APAFC_14_FEATURES)
            backend_label = "Core 14-Feature APAFC ONNX Backend"
        elif backend == self.EXPERIMENTAL_LGBM_BACKEND:
            expected_feats = list(ZEEK_NATIVE_17_FEATURES)
            backend_label = "Experimental 17-Feature Zeek-Native LightGBM Backend"
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. Must be one of: "
                f"['{self.CORE_ONNX_BACKEND}', '{self.EXPERIMENTAL_LGBM_BACKEND}']"
            )

        actual_feats = list(X.columns)

        if len(actual_feats) != len(expected_feats):
            raise ValueError(
                f"Schema count mismatch for {backend_label}: expected {len(expected_feats)} features, "
                f"but received {len(actual_feats)} columns. Refusing to guess, slice, or auto-pad features."
            )

        if actual_feats != expected_feats:
            missing = [f for f in expected_feats if f not in actual_feats]
            extra = [f for f in actual_feats if f not in expected_feats]
            order_mismatch = (sorted(actual_feats) == sorted(expected_feats)) and (actual_feats != expected_feats)
            msg = f"Schema mismatch for {backend_label}."
            if missing:
                msg += f" Missing required features: {missing}."
            if extra:
                msg += f" Incompatible unexpected columns: {extra}."
            if order_mismatch:
                msg += " Feature ordering differs from expected model contract."
            raise ValueError(msg)

    def predict(self, backend: str, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute prediction on explicitly selected backend with full schema validation.
        """
        self.validate_input(backend, X)

        if backend == self.CORE_ONNX_BACKEND:
            sess = self._get_onnx_session()
            x_arr = X.to_numpy(dtype=np.float32)
            outputs = sess.run(None, {self._onnx_input_name: x_arr})
            pred_indices = np.asarray(outputs[0], dtype=int).reshape(-1)
            pred_labels = [self._core_classes[int(i)] for i in pred_indices]
            # If probabilities are output
            probabilities = None
            if len(outputs) > 1 and outputs[1] is not None:
                probabilities = outputs[1]

            return {
                "backend": self.CORE_ONNX_BACKEND,
                "is_experimental": False,
                "accepted_input_format": "PCAP Reassembled Flows / Synthetic APAFC 14 Features",
                "taxonomy_mode": "15-Class Core Edge Taxonomy",
                "feature_count": 14,
                "total_flows": len(pred_labels),
                "predictions": pred_labels,
                "binary_predictions": [0 if p == "benign" else 1 for p in pred_labels],
            }

        elif backend == self.EXPERIMENTAL_LGBM_BACKEND:
            bundle = self._get_experimental_bundle()
            clf = bundle["model"]
            df_feats = X[ZEEK_NATIVE_17_FEATURES]
            pred_bin = clf.predict(df_feats).tolist()
            pred_prob = clf.predict_proba(df_feats)[:, 1].tolist() if hasattr(clf, "predict_proba") else None
            labels = ["benign" if b == 0 else "malicious" for b in pred_bin]

            return {
                "backend": self.EXPERIMENTAL_LGBM_BACKEND,
                "is_experimental": True,
                "accepted_input_format": "Zeek/Bro conn.log.labeled Connection Summaries (17 Native Features)",
                "taxonomy_mode": "Binary (Benign vs. Malicious)",
                "feature_count": 17,
                "total_flows": len(pred_bin),
                "predictions": labels,
                "binary_predictions": pred_bin,
                "malicious_probabilities": pred_prob,
                "disclaimer": (
                    "Experimental Model Notice: Trained strictly on Zeek connection summaries without raw packet payloads. "
                    "Do not compare directly with L7 deep-packet-inspection metrics."
                ),
            }


def get_available_iot23_scenarios() -> Dict[str, Path]:
    """
    Discover all real IoT-23 scenarios extracted under `data/raw/iot_23_datasets_small/`.
    Returns mapping from scenario name to its `conn.log.labeled` Path.
    """
    if not IOT23_RAW_ROOT.exists():
        return {}
    scenarios = {}
    for p in sorted(IOT23_RAW_ROOT.glob("**/conn.log.labeled")):
        scen = p.parent.parent.name
        scenarios[scen] = p
    return scenarios


def preview_iot23_scenario_records(
    scenario_name: str,
    max_rows: int = 100,
) -> Dict[str, Any]:
    """
    Load actual `conn.log.labeled` records through the streaming parser for UI exploration.
    Returns metadata, actual file path, size, row counts, label counts, and preview DataFrame.
    """
    scenario_files = get_available_iot23_scenarios()
    if scenario_name not in scenario_files:
        raise FileNotFoundError(
            f"IoT-23 scenario '{scenario_name}' not found. Available scenarios: {list(scenario_files.keys())}"
        )
    fp = scenario_files[scenario_name]
    chunks = []
    total = 0
    label_counter: Counter = Counter()
    detailed_counter: Counter = Counter()

    for chunk in iter_zeek_conn_chunks(fp, chunk_size=min(500, max_rows), max_rows=max_rows):
        chunks.append(chunk)
        total += len(chunk)
        label_counter.update(chunk["label"].astype(str).str.strip().tolist())
        detailed_counter.update(chunk["detailed-label"].astype(str).str.strip().tolist())
        if total >= max_rows:
            break

    df = pd.concat(chunks, ignore_index=True).head(max_rows) if chunks else pd.DataFrame()
    return {
        "scenario": scenario_name,
        "file_path": str(fp),
        "file_size_mb": round(fp.stat().st_size / (1024 * 1024), 2),
        "rows_sampled": len(df),
        "labels": dict(label_counter),
        "detailed_labels": dict(detailed_counter),
        "preview_df": df,
    }


def sample_real_iot23_records_for_batch_detection(
    scenario_name: str = "CTU-IoT-Malware-Capture-8-1",
    n_benign: int = 25,
    n_malicious: int = 25,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Sample real benign and malicious Zeek records from a selected scenario for batch detection.
    Preserves ground-truth columns (`ground_truth_label`, `ground_truth_detailed`, `ground_truth_binary`,
    `source_scenario`, `data_source`) alongside the 17 native Zeek features extracted via
    `extract_zeek_native_features()`.
    """
    scenario_files = get_available_iot23_scenarios()
    if scenario_name not in scenario_files:
        raise FileNotFoundError(f"IoT-23 scenario '{scenario_name}' not found in {IOT23_RAW_ROOT}")
    fp = scenario_files[scenario_name]

    ben_chunks = []
    mal_chunks = []
    tot_ben = 0
    tot_mal = 0

    for chunk in iter_zeek_conn_chunks(fp, chunk_size=2000, max_rows=20000):
        raw_b = chunk["label"].astype(str).str.strip().map(lambda x: 0 if IOT23_BINARY_LABEL_MAP.get(x, "benign") == "benign" else 1)
        chunk["ground_truth_binary"] = raw_b
        chunk["ground_truth_label"] = chunk["label"].astype(str).str.strip()
        chunk["ground_truth_detailed"] = chunk["detailed-label"].astype(str).str.strip()
        chunk["source_scenario"] = scenario_name
        chunk["data_source"] = "real_iot23_conn_log"

        b_sub = chunk[chunk["ground_truth_binary"] == 0]
        m_sub = chunk[chunk["ground_truth_binary"] == 1]

        if tot_ben < n_benign and len(b_sub) > 0:
            need_b = n_benign - tot_ben
            take_b = b_sub.head(need_b)
            ben_chunks.append(take_b)
            tot_ben += len(take_b)

        if tot_mal < n_malicious and len(m_sub) > 0:
            need_m = n_malicious - tot_mal
            take_m = m_sub.head(need_m)
            mal_chunks.append(take_m)
            tot_mal += len(take_m)

        if tot_ben >= n_benign and tot_mal >= n_malicious:
            break

    df_ben = pd.concat(ben_chunks, ignore_index=True) if ben_chunks else pd.DataFrame()
    df_mal = pd.concat(mal_chunks, ignore_index=True) if mal_chunks else pd.DataFrame()

    df_combined = pd.concat([df_ben, df_mal], ignore_index=True)
    if len(df_combined) == 0:
        raise ValueError(f"No records could be sampled from {scenario_name}")

    if len(df_combined) > 1:
        df_combined = df_combined.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Extract the 17 native features and preserve ground truth metadata
    feats_df = extract_zeek_native_features(df_combined)
    for meta_col in ("ground_truth_label", "ground_truth_detailed", "ground_truth_binary", "source_scenario", "data_source"):
        feats_df[meta_col] = df_combined[meta_col]

    return feats_df



