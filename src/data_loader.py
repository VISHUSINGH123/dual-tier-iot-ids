"""
Module A (Part 1) — Dataset Loading, Schema Inspection, and Synthetic Demo Data Generator.

Provides:
- Automatic target column detection and ambiguity flagging.
- Dataset health summary (missing values, infinities, duplicates, class counts).
- Distinction between protocol-aware features (MQTT, CoAP, HTTP, 6LoWPAN, DNS/Modbus)
  and generic flow/packet statistics.
- Clearly labelled synthetic demonstration dataset generator matching the 61-feature,
  15-class Edge-IIoTset structure described in the BCSE497J report when the full
  20M+ row Edge-IIoTset corpus is not locally present.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import io

import numpy as np
import pandas as pd

from config.settings import (
    CANDIDATE_LABEL_COLUMNS,
    DEFAULT_RANDOM_SEED,
    DRIFT_DEMO_CSV,
    EDGE_IIOTSET_15_CLASSES,
    LEAKAGE_AND_ID_COLUMNS,
    PROTOCOL_FEATURE_PREFIXES,
    REAL_EDGE_IIOTSET_CSV,
    SYNTHETIC_DEMO_CSV,
)


# 61 realistic Edge-IIoTset-aligned features grouped into protocol-specific and generic flow features
PROTOCOL_FEATURES_61: List[str] = [
    # MQTT protocol features (10)
    "mqtt.conack.flags",
    "mqtt.conflag.cleansess",
    "mqtt.conflags",
    "mqtt.hdrflags",
    "mqtt.len",
    "mqtt.msg_decoded_as",
    "mqtt.msgtype",
    "mqtt.proto_len",
    "mqtt.protoname",
    "mqtt.ver",
    # CoAP protocol features (6)
    "coap.code",
    "coap.opt.delta",
    "coap.opt.length",
    "coap.payload_length",
    "coap.type",
    "coap.token_len",
    # HTTP protocol features (8)
    "http.content_length",
    "http.request.method",
    "http.referer",
    "http.request.version",
    "http.response.code",
    "http.tls_port",
    "http.uri_len",
    "http.header_count",
    # 6LoWPAN / IPv6 header compression & sensor mesh features (7)
    "sixlowpan.iphc.tf",
    "sixlowpan.iphc.nh",
    "sixlowpan.iphc.hlim",
    "sixlowpan.frag.size",
    "sixlowpan.mesh.hops",
    "ipv6.plen",
    "icmpv6.type",
    # DNS & Modbus industrial protocol features (6)
    "dns.qry.name.len",
    "dns.qry.qu",
    "dns.qry.type",
    "dns.retransmission",
    "mbtcp.len",
    "mbtcp.unit_id",
]

GENERIC_FLOW_FEATURES_61: List[str] = [
    # TCP / UDP / ICMP / ARP / IP generic flow & packet statistics (24)
    "tcp.ack",
    "tcp.ack_raw",
    "tcp.checksum",
    "tcp.connection.fin",
    "tcp.connection.rst",
    "tcp.connection.syn",
    "tcp.connection.synack",
    "tcp.dstport",
    "tcp.flags",
    "tcp.flags.ack",
    "tcp.len",
    "tcp.seq",
    "tcp.srcport",
    "tcp.window_size",
    "udp.port",
    "udp.length",
    "udp.time_delta",
    "icmp.checksum",
    "icmp.seq_le",
    "icmp.transmit_timestamp",
    "arp.opcode",
    "arp.hw.size",
    "flow.duration_ms",
    "flow.pkt_rate",
]

ALL_61_FEATURES: List[str] = PROTOCOL_FEATURES_61 + GENERIC_FLOW_FEATURES_61
assert len(ALL_61_FEATURES) == 61, f"Expected 61 features, got {len(ALL_61_FEATURES)}"


@dataclass
class FlowRecord:
    """Represents a single network flow record (matches report Class Diagram)."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    byte_count: int
    pkt_count: int
    duration_ms: float
    features: Dict[str, Union[float, str]] = field(default_factory=dict)

    def to_feature_vector(self, feature_Order: Optional[List[str]] = None) -> Dict[str, Union[float, str]]:
        """Convert FlowRecord into a feature dictionary ready for APAFC extraction."""
        cols = feature_Order if feature_Order is not None else list(self.features.keys())
        return {c: self.features.get(c, 0.0) for c in cols}


@dataclass
class DatasetSummary:
    """Structured diagnostic summary of a loaded dataset."""

    dataset_name: str
    is_synthetic_demo: bool
    num_rows: int
    num_columns: int
    detected_label_col: Optional[str]
    candidate_label_cols: List[str]
    label_is_ambiguous: bool
    class_distribution: Dict[str, int]
    num_classes: int
    feature_columns: List[str]
    protocol_feature_columns: List[str]
    generic_feature_columns: List[str]
    excluded_leakage_columns: List[str]
    categorical_columns: List[str]
    numeric_columns: List[str]
    missing_value_count: int
    infinite_value_count: int
    duplicate_row_count: int
    validation_warnings: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)


def classify_feature_origin(column_name: str) -> str:
    """
    Distinguish genuine protocol-derived attributes (MQTT, CoAP, HTTP, 6LoWPAN, DNS/Modbus)
    from generic flow/transport statistics.
    """
    col_lower = column_name.lower()
    for proto_name, prefixes in PROTOCOL_FEATURE_PREFIXES.items():
        if any(col_lower.startswith(p.lower()) for p in prefixes):
            return proto_name
    return "Generic_Flow"


def detect_target_column(df: pd.DataFrame) -> Tuple[Optional[str], List[str], bool]:
    """
    Inspect DataFrame columns to identify the target label column.
    Returns (selected_label_col, candidate_cols, is_ambiguous).
    """
    present_candidates = [c for c in CANDIDATE_LABEL_COLUMNS if c in df.columns]

    # Prefer Attack_type / attack_type (multiclass) over Attack_label (binary) if both exist
    for preferred in ("Attack_type", "attack_type", "label", "Label", "class", "Class"):
        if preferred in present_candidates:
            return preferred, present_candidates, False

    if len(present_candidates) == 1:
        return present_candidates[0], present_candidates, False

    if len(present_candidates) > 1:
        return present_candidates[0], present_candidates, True

    # Fallback: check low-cardinality string/categorical columns near the end of the table
    fallback_candidates = []
    for col in reversed(df.columns):
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= 50 and (df[col].dtype == "object" or str(df[col].dtype).startswith("category")):
            if col not in LEAKAGE_AND_ID_COLUMNS:
                fallback_candidates.append(col)

    if len(fallback_candidates) == 1:
        return fallback_candidates[0], fallback_candidates, True
    elif len(fallback_candidates) > 1:
        return None, fallback_candidates, True

    return None, [], True


def inspect_dataset(
    df: pd.DataFrame,
    dataset_name: str = "Uploaded Dataset",
    target_col: Optional[str] = None,
    is_synthetic_demo: bool = False,
) -> DatasetSummary:
    """
    Perform comprehensive validation and inspection of a DataFrame.
    """
    warnings_list: List[str] = []
    errors_list: List[str] = []

    if df is None or df.empty:
        errors_list.append("Dataset is empty (0 rows).")
        return DatasetSummary(
            dataset_name=dataset_name,
            is_synthetic_demo=is_synthetic_demo,
            num_rows=0,
            num_columns=0,
            detected_label_col=None,
            candidate_label_cols=[],
            label_is_ambiguous=True,
            class_distribution={},
            num_classes=0,
            feature_columns=[],
            protocol_feature_columns=[],
            generic_feature_columns=[],
            excluded_leakage_columns=[],
            categorical_columns=[],
            numeric_columns=[],
            missing_value_count=0,
            infinite_value_count=0,
            duplicate_row_count=0,
            validation_warnings=warnings_list,
            validation_errors=errors_list,
        )

    detected_col, candidates, is_ambiguous = detect_target_column(df)
    selected_target = target_col if (target_col and target_col in df.columns) else detected_col

    if selected_target is None:
        warnings_list.append(
            "Target label column could not be automatically confirmed. Please select the target column explicitly."
        )

    # Identify leakage/ID columns and feature columns
    excluded_cols = []
    feature_cols = []
    for col in df.columns:
        if col == selected_target:
            continue
        if col in LEAKAGE_AND_ID_COLUMNS or col.lower() in {c.lower() for c in LEAKAGE_AND_ID_COLUMNS}:
            excluded_cols.append(col)
        else:
            feature_cols.append(col)

    if not feature_cols:
        errors_list.append("No valid input feature columns remain after excluding target and identifier columns.")

    protocol_cols = [c for c in feature_cols if classify_feature_origin(c) != "Generic_Flow"]
    generic_cols = [c for c in feature_cols if classify_feature_origin(c) == "Generic_Flow"]

    numeric_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # Count missing, infinite, and duplicate values
    missing_count = int(df[feature_cols].isna().sum().sum()) if feature_cols else 0
    inf_count = 0
    if numeric_cols:
        num_vals = df[numeric_cols].to_numpy(dtype=np.float64, na_value=np.nan)
        inf_count = int(np.isinf(num_vals).sum())

    dup_count = int(df.duplicated().sum())

    if missing_count > 0:
        warnings_list.append(f"Found {missing_count} missing (NaN) values across feature columns; will be imputed during preprocessing.")
    if inf_count > 0:
        warnings_list.append(f"Found {inf_count} infinite (+/-inf) values; will be replaced and imputed during preprocessing.")
    if dup_count > 0:
        warnings_list.append(f"Found {dup_count} duplicate rows; duplicates will be removed before train/test partitioning.")

    class_dist: Dict[str, int] = {}
    num_classes = 0
    if selected_target and selected_target in df.columns:
        vc = df[selected_target].dropna().astype(str).value_counts()
        class_dist = {str(k): int(v) for k, v in vc.items()}
        num_classes = len(class_dist)
        if num_classes < 2:
            errors_list.append(f"Target column '{selected_target}' has fewer than 2 classes ({num_classes} found).")
        else:
            min_class_count = min(class_dist.values())
            max_class_count = max(class_dist.values())
            if min_class_count < 5:
                warnings_list.append(
                    f"Minority class has only {min_class_count} records. Stratified splitting may fall back to non-stratified if < 2 per split."
                )
            if max_class_count / max(min_class_count, 1) > 5.0:
                warnings_list.append(
                    f"Class imbalance detected (majority/minority ratio = {max_class_count / max(min_class_count, 1):.1f}x). "
                    "Training-only SMOTE-ENN or class weighting is recommended."
                )

    return DatasetSummary(
        dataset_name=dataset_name,
        is_synthetic_demo=is_synthetic_demo,
        num_rows=int(len(df)),
        num_columns=int(df.shape[1]),
        detected_label_col=selected_target,
        candidate_label_cols=candidates,
        label_is_ambiguous=is_ambiguous and (target_col is None),
        class_distribution=class_dist,
        num_classes=num_classes,
        feature_columns=feature_cols,
        protocol_feature_columns=protocol_cols,
        generic_feature_columns=generic_cols,
        excluded_leakage_columns=excluded_cols,
        categorical_columns=categorical_cols,
        numeric_columns=numeric_cols,
        missing_value_count=missing_count,
        infinite_value_count=inf_count,
        duplicate_row_count=dup_count,
        validation_warnings=warnings_list,
        validation_errors=errors_list,
    )


def generate_synthetic_edge_iiotset(
    n_samples: int = 3000,
    random_state: int = DEFAULT_RANDOM_SEED,
    apply_concept_drift: bool = False,
    drift_magnitude: float = 0.0,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate a clearly-labelled synthetic demonstration dataset modelled on the
    61-feature, 15-class Edge-IIoTset structure described in the BCSE497J report.

    NOTE: This dataset is explicitly marked as synthetic demonstration data and
    must never be claimed as benchmark measurements on the full 20M-flow Edge-IIoTset.
    """
    rng = np.random.default_rng(random_state)

    # Realistic imbalanced class proportions across 15 Edge-IIoTset categories
    base_weights = np.array([
        0.28,  # Normal
        0.09,  # DDoS_UDP
        0.08,  # DDoS_ICMP
        0.08,  # DDoS_TCP
        0.07,  # DDoS_HTTP
        0.05,  # SQL_injection
        0.04,  # Uploading
        0.04,  # XSS
        0.03,  # Backdoor
        0.05,  # Password
        0.06,  # Port_Scanning
        0.05,  # Vulnerability_scanner
        0.03,  # Ransomware
        0.025, # MITM
        0.025, # Fingerprinting
    ], dtype=np.float64)
    base_weights /= base_weights.sum()

    if apply_concept_drift and drift_magnitude > 0:
        # Under concept drift, shift traffic mixture toward DDoS_HTTP, MQTT Password, 6LoWPAN/CoAP floods
        drift_weights = np.array([
            0.15, 0.14, 0.05, 0.10, 0.14, 0.06, 0.05, 0.05,
            0.04, 0.07, 0.04, 0.04, 0.03, 0.02, 0.02
        ], dtype=np.float64)
        drift_weights /= drift_weights.sum()
        alpha = min(max(drift_magnitude, 0.0), 1.0)
        weights = (1.0 - alpha) * base_weights + alpha * drift_weights
        weights /= weights.sum()
    else:
        weights = base_weights

    labels = rng.choice(EDGE_IIOTSET_15_CLASSES, size=n_samples, p=weights)

    # Initialize numeric matrix for 61 features
    data: Dict[str, np.ndarray] = {}
    for col in ALL_61_FEATURES:
        if col in ("mqtt.protoname", "http.request.method", "http.request.version"):
            continue
        data[col] = rng.exponential(scale=1.0, size=n_samples) * 0.1

    # Categorical columns
    mqtt_proto = np.full(n_samples, "None", dtype=object)
    http_method = np.full(n_samples, "NONE", dtype=object)
    http_ver = np.full(n_samples, "NONE", dtype=object)

    # Class-conditional realistic protocol & flow signatures with moderate overlap noise
    for idx, cls in enumerate(labels):
        # Baseline noise across flows
        pkt_rate = rng.gamma(shape=2.0, scale=15.0)
        duration = rng.gamma(shape=2.0, scale=120.0)

        if cls == "Normal":
            proto_choice = rng.choice(["MQTT", "CoAP", "HTTP", "6LoWPAN"], p=[0.35, 0.25, 0.25, 0.15])
            if proto_choice == "MQTT":
                mqtt_proto[idx] = "MQTT"
                data["mqtt.ver"][idx] = 4.0
                data["mqtt.msgtype"][idx] = rng.choice([1, 3, 8, 12])
                data["mqtt.len"][idx] = rng.normal(24, 6)
                data["mqtt.conflag.cleansess"][idx] = 1.0
                data["tcp.dstport"][idx] = 1883
                data["tcp.window_size"][idx] = rng.normal(4096, 512)
            elif proto_choice == "CoAP":
                data["coap.type"][idx] = rng.choice([0, 1, 2])
                data["coap.code"][idx] = rng.choice([1, 69])
                data["coap.payload_length"][idx] = rng.normal(18, 5)
                data["udp.port"][idx] = 5683
                data["udp.length"][idx] = rng.normal(32, 6)
            elif proto_choice == "HTTP":
                http_method[idx] = "GET"
                http_ver[idx] = "HTTP/1.1"
                data["http.response.code"][idx] = 200
                data["http.content_length"][idx] = rng.normal(240, 60)
                data["http.uri_len"][idx] = rng.normal(16, 4)
                data["tcp.dstport"][idx] = 80
            else:
                data["sixlowpan.iphc.tf"][idx] = 1.0
                data["sixlowpan.iphc.hlim"][idx] = 64.0
                data["sixlowpan.mesh.hops"][idx] = rng.integers(1, 4)
                data["ipv6.plen"][idx] = rng.normal(40, 8)
            data["flow.pkt_rate"][idx] = max(1.0, rng.normal(22, 8))
            data["flow.duration_ms"][idx] = max(5.0, rng.normal(350, 90))

        elif cls == "DDoS_UDP":
            data["udp.port"][idx] = rng.choice([5683, 53, 123, 1900])
            data["udp.length"][idx] = rng.normal(512, 95)
            data["udp.time_delta"][idx] = max(0.0001, rng.normal(0.002, 0.001))
            data["coap.payload_length"][idx] = rng.normal(380, 90)
            data["coap.type"][idx] = 1.0
            data["flow.pkt_rate"][idx] = rng.normal(1450, 280)
            data["flow.duration_ms"][idx] = rng.normal(45, 18)

        elif cls == "DDoS_ICMP":
            data["icmp.checksum"][idx] = rng.normal(32000, 5000)
            data["icmp.seq_le"][idx] = rng.normal(850, 220)
            data["icmp.transmit_timestamp"][idx] = rng.normal(5000, 900)
            data["sixlowpan.frag.size"][idx] = rng.normal(640, 120)
            data["icmpv6.type"][idx] = 128.0
            data["flow.pkt_rate"][idx] = rng.normal(1600, 320)
            data["flow.duration_ms"][idx] = rng.normal(35, 12)

        elif cls == "DDoS_TCP":
            data["tcp.connection.syn"][idx] = 1.0
            data["tcp.connection.ack"] = data.get("tcp.connection.ack", np.zeros(n_samples))
            data["tcp.flags"][idx] = 2.0
            data["tcp.window_size"][idx] = rng.normal(512, 120)
            data["tcp.dstport"][idx] = rng.choice([80, 443, 1883, 502])
            data["tcp.seq"][idx] = rng.normal(9000, 2000)
            data["flow.pkt_rate"][idx] = rng.normal(1250, 260)
            data["flow.duration_ms"][idx] = rng.normal(55, 20)

        elif cls == "DDoS_HTTP":
            http_method[idx] = rng.choice(["GET", "POST"], p=[0.7, 0.3])
            http_ver[idx] = "HTTP/1.1"
            data["http.uri_len"][idx] = rng.normal(95, 22)
            data["http.header_count"][idx] = rng.normal(14, 3)
            data["http.content_length"][idx] = rng.normal(620, 150)
            data["http.response.code"][idx] = rng.choice([200, 503, 500])
            data["tcp.dstport"][idx] = 80
            data["flow.pkt_rate"][idx] = rng.normal(820, 180)
            data["flow.duration_ms"][idx] = rng.normal(110, 35)

        elif cls == "SQL_injection":
            http_method[idx] = rng.choice(["POST", "GET"], p=[0.65, 0.35])
            http_ver[idx] = "HTTP/1.1"
            data["http.uri_len"][idx] = rng.normal(240, 45)
            data["http.content_length"][idx] = rng.normal(480, 110)
            data["http.response.code"][idx] = rng.choice([200, 500, 403])
            data["http.header_count"][idx] = rng.normal(9, 2)
            data["tcp.len"][idx] = rng.normal(540, 110)
            data["tcp.dstport"][idx] = 80
            data["flow.pkt_rate"][idx] = rng.normal(85, 25)
            data["flow.duration_ms"][idx] = rng.normal(290, 75)

        elif cls == "Uploading":
            http_method[idx] = "POST"
            http_ver[idx] = "HTTP/1.1"
            data["http.content_length"][idx] = rng.normal(4200, 850)
            data["http.uri_len"][idx] = rng.normal(68, 15)
            data["http.response.code"][idx] = rng.choice([200, 201, 413])
            data["tcp.len"][idx] = rng.normal(1380, 180)
            data["tcp.window_size"][idx] = rng.normal(16384, 2400)
            data["flow.pkt_rate"][idx] = rng.normal(210, 50)
            data["flow.duration_ms"][idx] = rng.normal(950, 240)

        elif cls == "XSS":
            http_method[idx] = rng.choice(["GET", "POST"], p=[0.55, 0.45])
            http_ver[idx] = "HTTP/1.1"
            data["http.uri_len"][idx] = rng.normal(175, 35)
            data["http.referer"][idx] = rng.normal(85, 20)
            data["http.content_length"][idx] = rng.normal(310, 80)
            data["http.response.code"][idx] = rng.choice([200, 400])
            data["tcp.len"][idx] = rng.normal(390, 90)
            data["flow.pkt_rate"][idx] = rng.normal(65, 20)
            data["flow.duration_ms"][idx] = rng.normal(240, 60)

        elif cls == "Backdoor":
            mqtt_proto[idx] = rng.choice(["MQTT", "None"], p=[0.6, 0.4])
            data["tcp.dstport"][idx] = rng.choice([4444, 1883, 8080, 502])
            data["tcp.flags.ack"][idx] = 1.0
            data["tcp.len"][idx] = rng.normal(280, 75)
            data["mqtt.hdrflags"][idx] = rng.normal(52, 12)
            data["mqtt.len"][idx] = rng.normal(190, 45)
            data["mbtcp.len"][idx] = rng.normal(64, 20)
            data["flow.pkt_rate"][idx] = rng.normal(48, 16)
            data["flow.duration_ms"][idx] = rng.normal(1400, 380)

        elif cls == "Password":
            mqtt_proto[idx] = "MQTT"
            data["mqtt.msgtype"][idx] = 1.0  # CONNECT flood / brute force
            data["mqtt.conack.flags"][idx] = rng.choice([4, 5])  # Bad username/password or not authorized
            data["mqtt.conflags"][idx] = rng.normal(194, 15)
            data["mqtt.len"][idx] = rng.normal(64, 14)
            data["mqtt.proto_len"][idx] = 4.0
            data["tcp.dstport"][idx] = 1883
            data["tcp.connection.rst"][idx] = rng.choice([0.0, 1.0], p=[0.3, 0.7])
            data["flow.pkt_rate"][idx] = rng.normal(320, 75)
            data["flow.duration_ms"][idx] = rng.normal(180, 45)

        elif cls == "Port_Scanning":
            data["tcp.connection.syn"][idx] = 1.0
            data["tcp.connection.rst"][idx] = 1.0
            data["tcp.dstport"][idx] = rng.integers(1, 65535)
            data["tcp.window_size"][idx] = rng.normal(1024, 180)
            data["tcp.len"][idx] = rng.normal(4, 2)
            data["flow.pkt_rate"][idx] = rng.normal(540, 130)
            data["flow.duration_ms"][idx] = rng.normal(18, 7)

        elif cls == "Vulnerability_scanner":
            http_method[idx] = rng.choice(["GET", "HEAD", "OPTIONS"])
            http_ver[idx] = "HTTP/1.1"
            data["http.uri_len"][idx] = rng.normal(130, 30)
            data["http.response.code"][idx] = rng.choice([404, 403, 500, 200])
            data["coap.code"][idx] = rng.normal(120, 30)
            data["coap.opt.delta"][idx] = rng.normal(11, 3)
            data["tcp.dstport"][idx] = rng.choice([80, 443, 8080, 1883])
            data["flow.pkt_rate"][idx] = rng.normal(410, 95)
            data["flow.duration_ms"][idx] = rng.normal(95, 28)

        elif cls == "Ransomware":
            data["mbtcp.len"][idx] = rng.normal(220, 50)
            data["mbtcp.unit_id"][idx] = rng.integers(1, 16)
            data["dns.qry.name.len"][idx] = rng.normal(48, 12)
            data["dns.qry.type"][idx] = 16.0
            data["tcp.len"][idx] = rng.normal(1150, 240)
            data["tcp.window_size"][idx] = rng.normal(8192, 1500)
            data["flow.pkt_rate"][idx] = rng.normal(175, 45)
            data["flow.duration_ms"][idx] = rng.normal(820, 190)

        elif cls == "MITM":
            data["arp.opcode"][idx] = 2.0  # ARP reply spoofing
            data["arp.hw.size"][idx] = 6.0
            data["dns.retransmission"][idx] = 1.0
            data["dns.qry.name.len"][idx] = rng.normal(34, 8)
            data["sixlowpan.mesh.hops"][idx] = rng.normal(7, 2)
            data["flow.pkt_rate"][idx] = rng.normal(140, 38)
            data["flow.duration_ms"][idx] = rng.normal(460, 110)

        elif cls == "Fingerprinting":
            data["tcp.connection.syn"][idx] = 1.0
            data["tcp.connection.fin"][idx] = 1.0
            data["tcp.window_size"][idx] = rng.normal(29200, 4200)
            data["sixlowpan.iphc.nh"][idx] = rng.normal(58, 10)
            data["icmp.checksum"][idx] = rng.normal(14000, 3200)
            data["flow.pkt_rate"][idx] = rng.normal(95, 28)
            data["flow.duration_ms"][idx] = rng.normal(75, 22)

    # Add realistic background cross-traffic overlap (~3.5% of flows exhibit noisy/ambiguous telemetry)
    noisy_indices = rng.choice(n_samples, size=max(10, int(0.035 * n_samples)), replace=False)
    for n_idx in noisy_indices:
        swap_partner = int(rng.integers(0, n_samples))
        for col in ALL_61_FEATURES:
            if col not in ("mqtt.protoname", "http.request.method", "http.request.version"):
                if rng.random() < 0.55:
                    data[col][n_idx] = data[col][swap_partner]

    # Apply concept drift feature distribution shift if requested
    if apply_concept_drift and drift_magnitude > 0:
        shift = 1.0 + 1.8 * drift_magnitude
        for col in (
            "flow.pkt_rate",
            "mqtt.len",
            "http.uri_len",
            "coap.payload_length",
            "udp.length",
            "sixlowpan.frag.size",
            "tcp.window_size",
        ):
            data[col] = data[col] * shift + rng.normal(25.0 * drift_magnitude, 8.0, size=n_samples)

    # Assemble DataFrame with metadata/ID columns + 61 features + target labels
    df_dict: Dict[str, Union[np.ndarray, List]] = {
        "frame.time": [f"2026-10-24T12:{i // 60:02d}:{i % 60:02d}.000Z" for i in range(n_samples)],
        "ip.src_host": [f"192.168.1.{rng.integers(2, 250)}" for _ in range(n_samples)],
        "ip.dst_host": [f"10.0.0.{rng.integers(2, 50)}" for _ in range(n_samples)],
    }

    for col in ALL_61_FEATURES:
        if col == "mqtt.protoname":
            df_dict[col] = mqtt_proto
        elif col == "http.request.method":
            df_dict[col] = http_method
        elif col == "http.request.version":
            df_dict[col] = http_ver
        else:
            df_dict[col] = np.round(data[col], 4)

    df_dict["Attack_label"] = np.where(labels == "Normal", 0, 1)
    df_dict["Attack_type"] = labels

    df = pd.DataFrame(df_dict)

    # Inject a small number of realistic missing values (NaN) and infinities (~0.3%)
    # so the preprocessing pipeline's imputation and sanitization are genuinely exercised
    n_missing = max(5, int(0.003 * n_samples))
    missing_rows = rng.choice(n_samples, size=n_missing, replace=False)
    df.loc[missing_rows[: n_missing // 2], "mqtt.len"] = np.nan
    df.loc[missing_rows[n_missing // 2 :], "http.content_length"] = np.inf

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df


def load_real_edge_iiotset_dataset(
    max_rows: Optional[int] = None,
    dataset_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Path, bool]:
    """
    Load genuine Edge-IIoTset CSV records directly from disk without synthetic generation or fallback.
    Returns (df, path, is_synthetic=False).
    Raises FileNotFoundError if the file does not exist.
    """
    target = Path(dataset_path) if dataset_path else REAL_EDGE_IIOTSET_CSV
    if not target.exists():
        raise FileNotFoundError(
            f"Genuine Edge-IIoTset dataset not found at '{target}'. "
            "Synthetic fallbacks are disabled in accordance with the project data policy."
        )
    df = pd.read_csv(target, nrows=max_rows, low_memory=False)
    return df, target, False


def load_or_create_demo_dataset(
    force_regenerate: bool = False,
    n_samples: int = 3000,
) -> Tuple[pd.DataFrame, Path, bool]:
    """
    Load genuine Edge-IIoTset dataset if available.
    Never silently generates synthetic data in the active application.
    """
    if REAL_EDGE_IIOTSET_CSV.exists():
        df = pd.read_csv(REAL_EDGE_IIOTSET_CSV, nrows=n_samples if n_samples else None, low_memory=False)
        return df, REAL_EDGE_IIOTSET_CSV, False

    # Check if user placed any real CSV in data/ (excluding synthetic/drifted files)
    data_dir = SYNTHETIC_DEMO_CSV.parent
    for csv_file in sorted(data_dir.glob("*.csv")):
        if csv_file.name not in (SYNTHETIC_DEMO_CSV.name, DRIFT_DEMO_CSV.name):
            df = pd.read_csv(csv_file, low_memory=False)
            return df, csv_file, False

    raise FileNotFoundError(
        f"Real Edge-IIoTset dataset missing at {REAL_EDGE_IIOTSET_CSV}. "
        "Automatic synthetic data generation in active application paths is strictly disabled."
    )


def load_csv_dataset(
    source: Union[str, Path, io.BytesIO],
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a CSV dataset from path or uploaded stream with safe type handling."""
    df = pd.read_csv(source, nrows=max_rows, low_memory=False)
    if df.empty:
        raise ValueError("The provided CSV file contains 0 rows.")
    return df
