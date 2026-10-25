"""
Integration Bridge connecting the Streamlit Prototype to the Primary Core Repository (`iot_ids_framework`).

Design & Safeguards:
- Uses `iot_ids_framework/src/iot_ids` as the primary core for:
  1. Fixed 14-feature APAFC flow extraction (`iot_ids.features.APAFCFeatureExtractor`, `FEATURE_NAMES`).
  2. Bidirectional packet-to-flow assembly & PCAP replay (`FlowAssembler`, `PCAPReader`, `replay_pcap_to_flows`).
  3. Optuna + in-fold SMOTE-ENN LightGBM training and ONNX export (`iot_ids.training.train`).
  4. Autonomous ONNX Runtime edge inference, TreeSHAP attribution, JSONL alert logging,
     and zero-downtime atomic model hot-swap (`iot_ids.edge.EdgeIDS`).
  5. Streaming KL-divergence concept drift monitoring & benchmarking (`KLDriftMonitor`, `DriftBenchmark`).
  6. Clustered 16-bit quantized gradient/hessian histogram aggregation & GBDT split gain
     selection (`ClusteredHistogramAggregator`, `quantise_histogram`, `QuantisedHistogram`).
  7. Classical ML baseline benchmarking (`BaselineModels`, `benchmark_baseline_models`).
- Keeps the exploratory 61-to-k data-driven feature selection workflow (`src/feature_selection.py`
  and `src/train_model.py`) explicitly separate as an empirical ablation/feature-ranking study.
"""
from __future__ import annotations

import json
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import BASE_DIR

FRAMEWORK_ROOT = BASE_DIR / "iot_ids_framework"
FRAMEWORK_SRC = FRAMEWORK_ROOT / "src"
FRAMEWORK_ARTIFACTS = FRAMEWORK_ROOT / "artifacts"
FRAMEWORK_DATASETS = FRAMEWORK_ROOT / "datasets"
FRAMEWORK_CONFIGS = FRAMEWORK_ROOT / "configs"

# Directory for isolated experimental ONNX training exports (never overwrites FRAMEWORK_ARTIFACTS)
EXPERIMENTAL_ONNX_OUTPUT_DIR = BASE_DIR / "models" / "experimental_onnx_runs"


def validate_apafc14_semantic_compatibility(
    csv_path: str | Path,
    mapping_config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Strict Semantic Feature Validation for APAFC 14-Feature Schema:
    Validates that a dataset actually provides legitimate, non-fabricated values
    for all 14 required APAFC features.
    
    Verifies:
      1. Presence of all 14 required column names in FEATURE_NAMES.
      2. Non-zero/non-trivial semantic coverage for L7 deep-packet fields
         (mqtt_qos, mqtt_payload_bytes, coap_code, http_status, sixlowpan_dispatch, payload_entropy).
      3. Absence of null/unmapped features in mapping configuration.
    
    Returns a dictionary detailing status, valid features, missing features,
    and a clear explanation of whether the dataset is genuinely compatible.
    """
    p = Path(csv_path)
    if not p.is_file():
        return {
            "compatible": False,
            "status": "FILE_NOT_FOUND",
            "reason": f"Specified dataset file does not exist: {p}",
            "missing_features": list(FEATURE_NAMES),
            "unsupported_l7_features": [],
        }

    # Inspect mapping config if provided
    if mapping_config_path is not None:
        cfg_p = Path(mapping_config_path)
        if cfg_p.is_file():
            mapping = json.loads(cfg_p.read_text(encoding="utf-8"))
            unmapped = [f for f in FEATURE_NAMES if mapping.get(f) is None]
            if unmapped:
                return {
                    "compatible": False,
                    "status": "INCOMPLETE_FEATURE_MAPPING",
                    "reason": (
                        f"The feature mapper '{cfg_p.name}' cannot map {len(unmapped)} required APAFC features: {unmapped}. "
                        "Column-name matching or zero-filling null features is semantically invalid for production inference."
                    ),
                    "missing_features": unmapped,
                    "unsupported_l7_features": [f for f in unmapped if f in (
                        "mqtt_qos", "mqtt_payload_bytes", "coap_code", "http_status", "sixlowpan_dispatch", "payload_entropy"
                    )],
                }

    try:
        sample_df = pd.read_csv(p, nrows=200)
    except Exception as exc:
        return {
            "compatible": False,
            "status": "READ_ERROR",
            "reason": f"Failed to read CSV: {exc}",
            "missing_features": list(FEATURE_NAMES),
            "unsupported_l7_features": [],
        }

    missing_cols = [f for f in FEATURE_NAMES if f not in sample_df.columns]
    if missing_cols:
        return {
            "compatible": False,
            "status": "MISSING_COLUMNS",
            "reason": f"Dataset is missing required APAFC columns: {missing_cols}",
            "missing_features": missing_cols,
            "unsupported_l7_features": [f for f in missing_cols if f in (
                "mqtt_qos", "mqtt_payload_bytes", "coap_code", "http_status", "sixlowpan_dispatch", "payload_entropy"
            )],
        }

    # Check whether all 6 L7 features are entirely zero-filled or null
    l7_features = ["mqtt_qos", "mqtt_payload_bytes", "coap_code", "http_status", "sixlowpan_dispatch", "payload_entropy"]
    all_zero_l7 = [
        f for f in l7_features
        if f in sample_df.columns and (pd.to_numeric(sample_df[f], errors="coerce").fillna(0) == 0).all()
    ]
    if len(all_zero_l7) >= 5:
        return {
            "compatible": False,
            "status": "SEMANTIC_ZERO_FILLED_L7",
            "reason": (
                f"{len(all_zero_l7)} of the 6 L7/payload features ({all_zero_l7}) are 100% zero in this dataset. "
                "This indicates an incompatible transport/flow summary dataset rather than a legitimate deep-packet inspection source."
            ),
            "missing_features": all_zero_l7,
            "unsupported_l7_features": all_zero_l7,
        }

    return {
        "compatible": True,
        "status": "COMPATIBLE",
        "reason": "Dataset provides valid values across all 14 APAFC features.",
        "missing_features": [],
        "unsupported_l7_features": [],
    }


if str(FRAMEWORK_SRC) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_SRC))

from iot_ids.drift import DriftEvent, KLDriftMonitor
from iot_ids.drift_benchmark import DriftBenchmark, DriftBenchmarkResult
from iot_ids.edge import Alert, EdgeIDS, load_monitor, load_tree_explainer
from iot_ids.features import APAFCFeatureExtractor, FEATURE_NAMES, PROTOCOL_IDS
from iot_ids.federation import (
    ClusteredHistogramAggregator,
    QuantisedHistogram,
    quantise_histogram,
)
from iot_ids.ingest import FlowAssembler
from iot_ids.pcap_adapter import PCAPReader, replay_pcap_to_flows as _core_replay_pcap_to_flows
from iot_ids.prepare import map_csv
from iot_ids.training import train as train_core_onnx_pipeline
from iot_ids.baselines.baseline_models import BaselineModels
from iot_ids.baselines.benchmark_comparison import benchmark_baseline_models


DEFAULT_CONFIG_PATH = FRAMEWORK_CONFIGS / "default.json"
ONNX_MODEL_PATH = FRAMEWORK_ARTIFACTS / "edge_ids.onnx"
TREE_MODEL_PATH = FRAMEWORK_ARTIFACTS / "edge_ids.txt"
LABELS_JSON_PATH = FRAMEWORK_ARTIFACTS / "labels.json"
REFERENCE_JSON_PATH = FRAMEWORK_ARTIFACTS / "reference.json"
ALERTS_JSONL_PATH = FRAMEWORK_ARTIFACTS / "alerts.jsonl"
SAMPLE_PCAP_PATH = FRAMEWORK_DATASETS / "sample_traffic.pcap"
SYNTHETIC_APAFC14_CSV = FRAMEWORK_DATASETS / "synthetic_apafc14_smoke.csv"


def load_framework_config() -> Dict[str, Any]:
    """Load `iot_ids_framework/configs/default.json`."""
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Framework config missing at {DEFAULT_CONFIG_PATH}")
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def check_core_artifacts_status() -> Dict[str, Any]:
    """Check presence and sizes of all `iot_ids_framework` artifacts and datasets."""
    files = {
        "edge_ids.onnx": ONNX_MODEL_PATH,
        "edge_ids.txt": TREE_MODEL_PATH,
        "labels.json": LABELS_JSON_PATH,
        "reference.json": REFERENCE_JSON_PATH,
        "alerts.jsonl": ALERTS_JSONL_PATH,
        "sample_traffic.pcap": SAMPLE_PCAP_PATH,
        "synthetic_apafc14_smoke.csv": SYNTHETIC_APAFC14_CSV,
    }
    status = {}
    for name, path in files.items():
        exists = path.exists()
        size_kb = round(path.stat().st_size / 1024.0, 2) if exists else 0.0
        status[name] = {
            "exists": exists,
            "path": str(path),
            "size_kb": size_kb,
        }
    ready = all(
        status[k]["exists"]
        for k in ("edge_ids.onnx", "edge_ids.txt", "labels.json", "reference.json")
    )
    return {"ready": ready, "files": status}


def generate_synthetic_apafc14_dataset(
    output_csv: Path = SYNTHETIC_APAFC14_CSV,
    rows_per_class: int = 80,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a clearly-labelled synthetic 15-class dataset conforming to the fixed
    14-feature APAFC flow schema (`iot_ids.features.FEATURE_NAMES`) and the 15 classes
    in `iot_ids_framework/configs/default.json`.

    NOTE: Explicitly labelled for smoke testing and prototype demonstration only.
    Never presented as real-world benchmark evaluation results.
    """
    cfg = load_framework_config()
    classes: List[str] = cfg["classes"]
    rng = np.random.default_rng(random_state)
    rows: List[Dict[str, Any]] = []

    for cls_idx, cls_name in enumerate(classes):
        # Give benign a larger share to reflect imbalanced IoT traffic while keeping >= rows_per_class
        n_rows = int(rows_per_class * 2.5) if cls_name == "benign" else rows_per_class
        for _ in range(n_rows):
            if cls_name == "benign":
                proto = int(rng.choice([1, 2, 3, 4], p=[0.40, 0.20, 0.25, 0.15]))
                pkts = float(rng.integers(2, 18))
                b_cnt = float(pkts * rng.uniform(28.0, 140.0))
                dur = float(rng.uniform(10.0, 350.0))
                ephem = float(rng.integers(30000, 65000))
                svc = {1: 1883.0, 2: 5683.0, 3: 80.0, 4: 61616.0}[proto]
                r_mode = rng.random()
                if r_mode < 0.4:
                    sport, dport = ephem, svc
                elif r_mode < 0.8:
                    sport, dport = svc, ephem
                else:
                    sport, dport = 0.0, 0.0
                t_flags = float(rng.choice([0.0, 24.0])) if proto in (1, 3) else 0.0
                m_qos = float(rng.choice([0, 1])) if proto == 1 else 0.0
                m_bytes = float(rng.choice([0.0, rng.uniform(4.0, 140.0)])) if proto == 1 else 0.0
                c_code = float(rng.choice([0, 1, 69])) if proto == 2 else 0.0
                h_stat = float(rng.choice([0, 200])) if proto == 3 else 0.0
                s_disp = float(rng.choice([0, 65])) if proto == 4 else 0.0
                ent = float(rng.choice([0.0, rng.uniform(2.5, 4.9)]))
            elif cls_name == "ddos":
                proto = int(rng.choice([1, 2, 3]))
                pkts = float(rng.integers(2500, 9500))
                b_cnt = float(pkts * rng.uniform(120.0, 250.0))
                dur = float(rng.uniform(250.0, 900.0))
                sport = float(rng.integers(10000, 65000))
                dport = {1: 1883.0, 2: 5683.0, 3: 80.0}[proto]
                t_flags = 2.0
                m_qos = 2.0 if proto == 1 else 0.0
                m_bytes = 512.0 if proto == 1 else 0.0
                c_code = 1.0 if proto == 2 else 0.0
                h_stat = 503.0 if proto == 3 else 0.0
                s_disp = 0.0
                ent = float(rng.uniform(6.8, 7.95))
            elif cls_name == "dos":
                proto = 1
                pkts = float(rng.integers(800, 2400))
                b_cnt = float(pkts * rng.uniform(80.0, 160.0))
                dur = float(rng.uniform(100.0, 450.0))
                sport = float(rng.integers(20000, 60000))
                dport = 1883.0
                t_flags = 18.0
                m_qos = 1.0
                m_bytes = 256.0
                c_code = 0.0
                h_stat = 0.0
                s_disp = 0.0
                ent = float(rng.uniform(5.5, 6.8))
            elif cls_name == "injection":
                proto = 3
                pkts = float(rng.integers(12, 45))
                b_cnt = float(pkts * rng.uniform(320.0, 680.0))
                dur = float(rng.uniform(80.0, 280.0))
                sport = float(rng.integers(40000, 65000))
                dport = 80.0
                t_flags = 24.0
                m_qos = 0.0
                m_bytes = 0.0
                c_code = 0.0
                h_stat = float(rng.choice([200, 500, 403]))
                s_disp = 0.0
                ent = float(rng.uniform(5.2, 6.2))
            elif cls_name == "mitm":
                proto = 4
                pkts = float(rng.integers(40, 140))
                b_cnt = float(pkts * rng.uniform(90.0, 180.0))
                dur = float(rng.uniform(300.0, 950.0))
                sport = float(rng.integers(10000, 40000))
                dport = 61616.0
                t_flags = 16.0
                m_qos = 0.0
                m_bytes = 0.0
                c_code = 0.0
                h_stat = 0.0
                s_disp = float(rng.integers(120, 190))
                ent = float(rng.uniform(5.8, 6.9))
            elif cls_name == "password":
                proto = 1
                pkts = float(rng.integers(60, 220))
                b_cnt = float(pkts * rng.uniform(45.0, 85.0))
                dur = float(rng.uniform(120.0, 380.0))
                sport = float(rng.integers(30000, 60000))
                dport = 1883.0
                t_flags = 20.0
                m_qos = 0.0
                m_bytes = float(rng.uniform(48.0, 96.0))
                c_code = 0.0
                h_stat = 0.0
                s_disp = 0.0
                ent = float(rng.uniform(4.8, 5.7))
            elif cls_name == "port_scan":
                proto = 0
                pkts = float(rng.integers(1, 4))
                b_cnt = float(pkts * rng.uniform(1.0, 16.0))
                dur = float(rng.uniform(1.0, 15.0))
                sport = float(rng.integers(45000, 65000))
                dport = float(rng.integers(1, 1024))
                t_flags = 2.0
                m_qos = 0.0
                m_bytes = 0.0
                c_code = 0.0
                h_stat = 0.0
                s_disp = 0.0
                ent = float(rng.uniform(0.0, 1.5))
            elif cls_name == "ransomware":
                proto = 1
                pkts = float(rng.integers(150, 480))
                b_cnt = float(pkts * rng.uniform(550.0, 1100.0))
                dur = float(rng.uniform(600.0, 1800.0))
                sport = float(rng.integers(25000, 55000))
                dport = 8883.0
                t_flags = 24.0
                m_qos = 2.0
                m_bytes = float(rng.uniform(600.0, 1200.0))
                c_code = 0.0
                h_stat = 0.0
                s_disp = 0.0
                ent = float(rng.uniform(7.4, 7.99))
            elif cls_name == "scanning":
                proto = 2
                pkts = float(rng.integers(8, 28))
                b_cnt = float(pkts * rng.uniform(20.0, 50.0))
                dur = float(rng.uniform(20.0, 80.0))
                sport = float(rng.integers(35000, 65000))
                dport = 5683.0
                t_flags = 4.0
                m_qos = 0.0
                m_bytes = 0.0
                c_code = float(rng.integers(130, 200))
                h_stat = 0.0
                s_disp = 0.0
                ent = float(rng.uniform(2.0, 3.5))
            elif cls_name == "spoofing":
                proto = 2
                pkts = float(rng.integers(35, 110))
                b_cnt = float(pkts * rng.uniform(110.0, 260.0))
                dur = float(rng.uniform(150.0, 500.0))
                sport = 5683.0
                dport = float(rng.integers(10000, 50000))
                t_flags = 0.0
                m_qos = 0.0
                m_bytes = 0.0
                c_code = float(rng.integers(64, 105))
                h_stat = 0.0
                s_disp = 0.0
                ent = float(rng.uniform(4.5, 5.8))
            elif cls_name == "xss":
                proto = 3
                pkts = float(rng.integers(10, 32))
                b_cnt = float(pkts * rng.uniform(180.0, 360.0))
                dur = float(rng.uniform(60.0, 220.0))
                sport = float(rng.integers(30000, 60000))
                dport = 8080.0
                t_flags = 24.0
                m_qos = 0.0
                m_bytes = 0.0
                c_code = 0.0
                h_stat = 400.0
                s_disp = 0.0
                ent = float(rng.uniform(4.9, 5.9))
            elif cls_name == "backdoor":
                proto = 1
                pkts = float(rng.integers(45, 160))
                b_cnt = float(pkts * rng.uniform(400.0, 900.0))
                dur = float(rng.uniform(800.0, 2400.0))
                sport = float(rng.integers(40000, 65000))
                dport = 4444.0
                t_flags = 24.0
                m_qos = 1.0
                m_bytes = float(rng.uniform(350.0, 750.0))
                c_code = 0.0
                h_stat = 0.0
                s_disp = 0.0
                ent = float(rng.uniform(7.1, 7.95))
            elif cls_name == "botnet":
                proto = 4
                pkts = float(rng.integers(250, 750))
                b_cnt = float(pkts * rng.uniform(140.0, 310.0))
                dur = float(rng.uniform(400.0, 1200.0))
                sport = float(rng.integers(20000, 50000))
                dport = 61616.0
                t_flags = 16.0
                m_qos = 0.0
                m_bytes = 0.0
                c_code = 0.0
                h_stat = 0.0
                s_disp = float(rng.integers(200, 250))
                ent = float(rng.uniform(6.2, 7.2))
            elif cls_name == "brute_force":
                proto = 3
                pkts = float(rng.integers(90, 280))
                b_cnt = float(pkts * rng.uniform(95.0, 175.0))
                dur = float(rng.uniform(200.0, 600.0))
                sport = float(rng.integers(30000, 60000))
                dport = 443.0
                t_flags = 20.0
                m_qos = 0.0
                m_bytes = 0.0
                c_code = 0.0
                h_stat = 401.0
                s_disp = 0.0
                ent = float(rng.uniform(5.0, 6.1))
            else:  # web_attack
                proto = 3
                pkts = float(rng.integers(20, 75))
                b_cnt = float(pkts * rng.uniform(260.0, 520.0))
                dur = float(rng.uniform(110.0, 390.0))
                sport = float(rng.integers(32000, 62000))
                dport = 80.0
                t_flags = 24.0
                m_qos = 0.0
                m_bytes = 0.0
                c_code = 0.0
                h_stat = 404.0
                s_disp = 0.0
                ent = float(rng.uniform(5.4, 6.5))

            rows.append(
                {
                    "protocol_id": float(proto),
                    "duration_ms": round(dur, 3),
                    "packet_count": round(pkts, 1),
                    "byte_count": round(b_cnt, 1),
                    "mean_packet_bytes": round(b_cnt / max(pkts, 1.0), 3),
                    "src_port": round(sport, 1),
                    "dst_port": round(dport, 1),
                    "tcp_flags": round(t_flags, 1),
                    "mqtt_qos": round(m_qos, 1),
                    "mqtt_payload_bytes": round(m_bytes, 1),
                    "coap_code": round(c_code, 1),
                    "http_status": round(h_stat, 1),
                    "sixlowpan_dispatch": round(s_disp, 1),
                    "payload_entropy": round(ent, 4),
                    "label": cls_name,
                }
            )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def _build_ipv4_packet(
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    proto_num: int,
    payload: bytes,
    tcp_flags: int = 24,
) -> bytes:
    """Build a valid Ethernet + IPv4 + TCP/UDP frame readable by `iot_ids.pcap_adapter.PCAPReader`."""
    eth_hdr = b"\x00\x11\x22\x33\x44\x55\x66\x77\x88\x99\xaa\xbb\x08\x00"
    if proto_num == 6:  # TCP
        l4_hdr = struct.pack(
            "!HHIIBBHHH",
            src_port,
            dst_port,
            1000,
            2000,
            (5 << 4),
            tcp_flags,
            4096,
            0,
            0,
        )
    else:  # UDP (17)
        udp_len = 8 + len(payload)
        l4_hdr = struct.pack("!HHHH", src_port, dst_port, udp_len, 0)

    total_ip_len = 20 + len(l4_hdr) + len(payload)
    ip_hdr = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        total_ip_len,
        1234,
        0,
        64,
        proto_num,
        0,
        socket.inet_aton(src_ip),
        socket.inet_aton(dst_ip),
    )
    return eth_hdr + ip_hdr + l4_hdr + payload


def generate_sample_iot_pcap(pcap_path: Path = SAMPLE_PCAP_PATH) -> Path:
    """
    Create a valid synthetic multi-protocol IoT `.pcap` capture containing MQTT, CoAP,
    HTTP, and 6LoWPAN bidirectional flows for `PCAPReader` & `FlowAssembler` replay.
    """
    pcap_path.parent.mkdir(parents=True, exist_ok=True)
    # Global PCAP header: magic=0xA1B2C3D4, v2.4, zone=0, sigfigs=0, snaplen=65535, linktype=1 (Ethernet)
    global_hdr = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)

    packets_spec = [
        # Flow 1: Benign MQTT sensor telemetry (bidirectional, 4 packets)
        (1700000000, 10000, "192.168.1.25", "192.168.1.1", 51200, 1883, 6, b"\x32\x18\x00\x0btemp/sensor23.4C-normal", 24),
        (1700000000, 15000, "192.168.1.1", "192.168.1.25", 1883, 51200, 6, b"\x40\x02\x00\x01", 24),
        (1700000000, 22000, "192.168.1.25", "192.168.1.1", 51200, 1883, 6, b"\x32\x18\x00\x0btemp/sensor23.5C-normal", 24),
        (1700000000, 30000, "192.168.1.1", "192.168.1.25", 1883, 51200, 6, b"\x40\x02\x00\x02", 24),
        # Flow 2: Benign CoAP smart-actuator status (UDP 5683)
        (1700000001, 5000, "192.168.1.40", "192.168.1.1", 49152, 5683, 17, b"\x40\x01\x12\x34\xb5light\xffon", 0),
        (1700000001, 18000, "192.168.1.1", "192.168.1.40", 5683, 49152, 17, b"\x60\x45\x12\x34\xffack-ok", 0),
        # Flow 3: HTTP Web Admin API query (TCP 80)
        (1700000002, 10000, "192.168.1.55", "192.168.1.1", 54321, 80, 6, b"HTTP/1.1 200 OK\r\nContent-Length: 32\r\n\r\nstatus=healthy&uptime=99120", 24),
        (1700000002, 45000, "192.168.1.1", "192.168.1.55", 80, 54321, 6, b"HTTP/1.1 200 OK\r\n\r\nok", 24),
        # Flow 4: 6LoWPAN mesh gateway packet (UDP 61616)
        (1700000003, 10000, "192.168.1.77", "192.168.1.1", 61616, 61616, 17, b"\x41\x06\x40mesh-6lowpan-telemetry-frame", 0),
        (1700000003, 35000, "192.168.1.1", "192.168.1.77", 61616, 61616, 17, b"\x41\x06\x40mesh-6lowpan-ack", 0),
    ]

    with pcap_path.open("wb") as f:
        f.write(global_hdr)
        for ts_sec, ts_usec, sip, dip, sport, dport, proto, payload, flags in packets_spec:
            frame = _build_ipv4_packet(sip, dip, sport, dport, proto, payload, flags)
            pkt_hdr = struct.pack("<IIII", ts_sec, ts_usec, len(frame), len(frame))
            f.write(pkt_hdr)
            f.write(frame)

    return pcap_path


def replay_pcap_to_flows(
    pcap_path: str | Path,
    assembler: Optional[FlowAssembler] = None,
    max_packets: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Replay a `.pcap` capture through `PCAPReader` and `FlowAssembler` while preserving:
    1. Cumulative flow `payload_entropy` (computed via `APAFCFeatureExtractor._entropy()` from
       packet `"payload"` bytes before `FlowAssembler.submit()` strips `"payload"`).
    2. Original initiator endpoint direction `(src_ip, src_port) -> (dst_ip, dst_port)` from the
       first packet of each flow (preventing `FlowAssembler._key()` lexicographical `repr()`
       sorting from flipping client and server ports when `dst_ip < src_ip`).
    3. Bidirectional L7 protocol telemetry (`mqtt_qos`, `mqtt_payload_bytes`, `coap_code`,
       `http_status`, `sixlowpan_dispatch`, `tcp_flags`) so trailing ACK packets do not overwrite
       initiator header fields with zeros.
    """
    reader = PCAPReader(pcap_path)
    asm = assembler or FlowAssembler(idle_timeout_ms=1000)
    last_ts = 0.0

    initiator_map: Dict[Tuple[Any, ...], Tuple[str, int, str, int]] = {}
    payload_buf: Dict[Tuple[Any, ...], bytearray] = {}
    proto_meta: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    completed_flows: List[Dict[str, Any]] = []

    def _finalize_expired(expired_list: List[Dict[str, Any]]) -> None:
        for fl in expired_list:
            fl_key = asm._key(fl)
            if fl_key in initiator_map:
                orig_sip, orig_sport, orig_dip, orig_dport = initiator_map.pop(fl_key)
                fl["src_ip"] = orig_sip
                fl["src_port"] = orig_sport
                fl["dst_ip"] = orig_dip
                fl["dst_port"] = orig_dport
            if fl_key in payload_buf:
                buf = payload_buf.pop(fl_key)
                fl["payload_entropy"] = float(APAFCFeatureExtractor._entropy(bytes(buf)))
            proto_meta.pop(fl_key, None)
            completed_flows.append(fl)

    packet_count = 0
    for pkt in reader.read_packets():
        packet_count += 1
        if max_packets is not None and packet_count > max_packets:
            break
        last_ts = float(pkt["timestamp_ms"])
        expired_mid = asm.expire(last_ts)
        if expired_mid:
            _finalize_expired(expired_mid)

        key = asm._key(pkt)

        if key not in initiator_map:
            initiator_map[key] = (
                str(pkt["src_ip"]),
                int(pkt["src_port"]),
                str(pkt["dst_ip"]),
                int(pkt["dst_port"]),
            )
            payload_buf[key] = bytearray()
            proto_meta[key] = {
                "tcp_flags": 0,
                "mqtt_qos": 0,
                "mqtt_payload_bytes": 0,
                "coap_code": 0,
                "http_status": 0,
                "sixlowpan_dispatch": 0,
            }

        raw_payload = pkt.get("payload", b"")
        if isinstance(raw_payload, (bytes, bytearray)) and raw_payload:
            payload_buf[key].extend(raw_payload)
            if str(pkt.get("protocol", "")).lower() == "6lowpan" and "sixlowpan_dispatch" not in pkt:
                pkt["sixlowpan_dispatch"] = int(raw_payload[0])

        meta = proto_meta[key]
        meta["tcp_flags"] = int(meta["tcp_flags"]) | int(pkt.get("tcp_flags", 0) or 0)
        meta["mqtt_qos"] = max(int(meta["mqtt_qos"]), int(pkt.get("mqtt_qos", 0) or 0))
        meta["mqtt_payload_bytes"] = max(
            int(meta["mqtt_payload_bytes"]), int(pkt.get("mqtt_payload_bytes", 0) or 0)
        )
        if int(meta["coap_code"]) == 0 and int(pkt.get("coap_code", 0) or 0) > 0:
            meta["coap_code"] = int(pkt["coap_code"])
        if int(meta["http_status"]) == 0 and int(pkt.get("http_status", 0) or 0) > 0:
            meta["http_status"] = int(pkt["http_status"])
        if int(meta["sixlowpan_dispatch"]) == 0 and int(pkt.get("sixlowpan_dispatch", 0) or 0) > 0:
            meta["sixlowpan_dispatch"] = int(pkt["sixlowpan_dispatch"])

        pkt_for_submit = dict(pkt)
        pkt_for_submit["payload_entropy"] = 0.0
        pkt_for_submit["tcp_flags"] = meta["tcp_flags"]
        if str(pkt.get("protocol", "")).lower() == "mqtt":
            pkt_for_submit["mqtt_qos"] = meta["mqtt_qos"]
            pkt_for_submit["mqtt_payload_bytes"] = meta["mqtt_payload_bytes"]
        elif str(pkt.get("protocol", "")).lower() == "coap":
            pkt_for_submit["coap_code"] = meta["coap_code"]
        elif str(pkt.get("protocol", "")).lower() == "http":
            pkt_for_submit["http_status"] = meta["http_status"]
        elif str(pkt.get("protocol", "")).lower() == "6lowpan":
            pkt_for_submit["sixlowpan_dispatch"] = meta["sixlowpan_dispatch"]

        asm.submit(pkt_for_submit)

    remaining = asm.expire(last_ts + asm.idle_timeout_ms + 100)
    if remaining:
        _finalize_expired(remaining)

    return completed_flows


replay_pcap_to_flows_with_entropy = replay_pcap_to_flows


def load_core_edge_ids(
    top_k: int = 5,
    audit_path: Optional[Path] = ALERTS_JSONL_PATH,
) -> EdgeIDS:
    """Instantiate the primary `iot_ids.edge.EdgeIDS` engine from `iot_ids_framework/artifacts`."""
    import onnxruntime as ort

    status = check_core_artifacts_status()
    if not status["ready"]:
        raise FileNotFoundError(
            "Core ONNX/LightGBM artifacts in `iot_ids_framework/artifacts` are not yet generated. "
            "Run the Core ONNX Training pipeline first."
        )
    cfg = load_framework_config()
    labels = json.loads(LABELS_JSON_PATH.read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    monitor = load_monitor(REFERENCE_JSON_PATH, cfg)
    explainer = load_tree_explainer(TREE_MODEL_PATH) if TREE_MODEL_PATH.exists() else None
    return EdgeIDS(
        session=session,
        class_names=labels,
        monitor=monitor,
        top_k=top_k,
        tree_explainer=explainer,
        audit_path=audit_path,
    )


class CleanDriftEvaluationResult:
    """
    Task 4 Structured Drift Evaluation Result with explicit metric definitions and denominators.

    Distinguishes:
    1. Drift Scores (`mean_kl_baseline`, `mean_kl_drift`, `window_curve_df`):
       Average feature-wise KL divergence D_KL(P_ref || Q_win) across the 14 features per window.
    2. Threshold-Exceeding Windows (`baseline_exceeded_windows`, `drift_exceeded_windows`,
       `false_alarm_window_rate`, `window_detection_rate`):
       Count and fraction of non-overlapping evaluated windows with KL > tau.
       Denominator = `total_baseline_windows` (for false alarm rate) or `total_drift_windows` (for window detection rate).
    3. Retraining Requests (`baseline_retrain_requests`, `drift_retrain_requests`, `sustained_drift_triggered`):
       Count of times consecutive threshold-exceeding windows reached `consecutive_windows` (C).
    4. Episode Detection Rate (`episode_detection_rate`, `detected_drift_episodes`, `total_drift_episodes`):
       Fraction of injected drift episodes in which the C-consecutive-window retraining trigger fired.
       Denominator = `total_drift_episodes`.
    5. Detection Delay (`detection_delay_windows`, `detection_delay_flows`):
       Number of evaluated windows (and flows) from drift regime onset until `retrain_requested == True`.
    """

    def __init__(
        self,
        threshold_tau: float,
        consecutive_windows_target: int,
        window_size: int,
        shift_magnitude: float,
        total_baseline_windows: int,
        baseline_exceeded_windows: int,
        false_alarm_window_rate: float,
        baseline_retrain_requests: int,
        mean_kl_baseline: float,
        total_drift_windows: int,
        drift_exceeded_windows: int,
        window_detection_rate: float,
        drift_retrain_requests: int,
        mean_kl_drift: float,
        total_drift_episodes: int,
        detected_drift_episodes: int,
        episode_detection_rate: float,
        sustained_drift_triggered: bool,
        detection_delay_windows: Optional[int],
        detection_delay_flows: Optional[int],
        window_curve_df: pd.DataFrame,
    ) -> None:
        self.threshold_tau = threshold_tau
        self.consecutive_windows_target = consecutive_windows_target
        self.window_size = window_size
        self.shift_magnitude = shift_magnitude
        self.total_baseline_windows = total_baseline_windows
        self.baseline_exceeded_windows = baseline_exceeded_windows
        self.false_alarm_window_rate = false_alarm_window_rate
        self.baseline_retrain_requests = baseline_retrain_requests
        self.mean_kl_baseline = mean_kl_baseline
        self.total_drift_windows = total_drift_windows
        self.drift_exceeded_windows = drift_exceeded_windows
        self.window_detection_rate = window_detection_rate
        self.drift_retrain_requests = drift_retrain_requests
        self.mean_kl_drift = mean_kl_drift
        self.total_drift_episodes = total_drift_episodes
        self.detected_drift_episodes = detected_drift_episodes
        self.episode_detection_rate = episode_detection_rate
        self.sustained_drift_triggered = sustained_drift_triggered
        self.detection_delay_windows = detection_delay_windows
        self.detection_delay_flows = detection_delay_flows
        self.window_curve_df = window_curve_df
        # Backward-compatible aliases
        self.mean_kl_benign = mean_kl_baseline
        self.false_trigger_count = baseline_retrain_requests
        self.false_trigger_rate = false_alarm_window_rate
        self.drift_detection_count = drift_exceeded_windows
        self.true_positive_rate = window_detection_rate


def run_clean_drift_evaluation(
    reference_data: np.ndarray,
    window_size: int = 100,
    bins: int = 16,
    threshold: float = 0.15,
    consecutive_windows: int = 3,
    shift_magnitude: float = 2.2,
    num_baseline_windows: int = 4,
    num_drift_windows: int = 6,
    random_state: int = 42,
) -> CleanDriftEvaluationResult:
    """
    Task 4 Fix (outside protected `iot_ids_framework`):
    Evaluates `iot_ids.drift.KLDriftMonitor` on non-overlapping windows across a steady
    baseline regime and an injected concept-drift regime.

    Why `pages/4_Drift_Monitor.py` previously showed 0.0% detection rate despite high KL:
    1. Adding continuous Gaussian noise (`rng.normal(0, 0.02)`) to discrete zero-inflated
       protocol columns (`mqtt_qos`, `coap_code`, `http_status`, `sixlowpan_dispatch`) pushed
       values out of the zero histogram bin during the baseline stream, causing `KL > tau`
       and `self.consecutive` to reach `301` before the drift stream even started.
    2. `DriftBenchmark.run_benchmark()` reused the same `KLDriftMonitor` instance across
       `benign_stream` and `shifted_stream` without resetting `self.consecutive`. Because
       `KLDriftMonitor.observe()` sets `retrain_requested = (self.consecutive == consecutive_windows)`
       (exact equality), `self.consecutive` incremented from `302..701` during `shifted_stream`
       and never equalled `3`, yielding `drift_detection_count = 0` (`0.0%`).
    3. `KLDriftMonitor._distribution()` uses fixed reference `[min, max]` histogram edges;
       shifting features beyond `max` without clipping drops out-of-range samples from
       `np.histogram`. Clipping shifted continuous features into the monitor's observable support
       (or shifting toward upper-tail bins) ensures shifted mass is captured inside the histogram.
    """
    ref_arr = np.asarray(reference_data, dtype=float)
    rng = np.random.default_rng(random_state)
    ref_min = np.min(ref_arr, axis=0, keepdims=True)
    ref_max = np.max(ref_arr, axis=0, keepdims=True)
    ref_std = np.std(ref_arr, axis=0, keepdims=True)

    # Continuous feature indices (duration_ms=1, packet_count=2, byte_count=3, mean_packet_bytes=4, payload_entropy=13)
    continuous_cols = [1, 2, 3, 4, 13]

    # 1. Evaluate Steady Baseline Regime on a fresh KLDriftMonitor
    monitor_base = KLDriftMonitor(
        ref_arr,
        window_size=int(window_size),
        bins=int(bins),
        threshold=float(threshold),
        consecutive_windows=int(consecutive_windows),
    )
    curve_rows: List[Dict[str, Any]] = []
    base_scores: List[float] = []
    base_exceeded = 0
    base_retrain_reqs = 0

    for w_idx in range(num_baseline_windows):
        # Stratified/balanced bootstrap from reference_data + tiny jitter on continuous columns only
        idx = rng.choice(len(ref_arr), size=int(window_size), replace=True)
        chunk = ref_arr[idx].copy()
        for c_i in continuous_cols:
            jitter = rng.normal(0.0, 0.005 * float(ref_std[0, c_i]), size=len(chunk))
            chunk[:, c_i] = np.clip(chunk[:, c_i] + jitter, ref_min[0, c_i], ref_max[0, c_i])

        monitor_base.window.clear()
        ev: Optional[DriftEvent] = None
        for row in chunk:
            ev = monitor_base.observe(row)
        assert ev is not None
        base_scores.append(float(ev.score))
        exceeded = bool(ev.score > float(threshold))
        if exceeded:
            base_exceeded += 1
        if ev.retrain_requested:
            base_retrain_reqs += 1
        curve_rows.append(
            {
                "Window_Index": w_idx + 1,
                "Window": f"Baseline W{w_idx + 1}",
                "Regime": "Steady Baseline",
                "KL_Divergence": round(float(ev.score), 4),
                "Exceeded_Tau": exceeded,
                "Consecutive_Exceeded": int(ev.consecutive),
                "Retrain_Requested": bool(ev.retrain_requested),
            }
        )

    # 2. Evaluate Injected Drift Regime on a freshly initialized KLDriftMonitor (reset state at regime boundary)
    monitor_drift = KLDriftMonitor(
        ref_arr,
        window_size=int(window_size),
        bins=int(bins),
        threshold=float(threshold),
        consecutive_windows=int(consecutive_windows),
    )
    drift_scores: List[float] = []
    drift_exceeded = 0
    drift_retrain_reqs = 0
    delay_windows: Optional[int] = None

    for w_idx in range(num_drift_windows):
        idx = rng.choice(len(ref_arr), size=int(window_size), replace=True)
        chunk = ref_arr[idx].copy()
        if float(shift_magnitude) > 0.0:
            # Shift features and clip to [ref_min, ref_max] so shifted mass lands in upper histogram bins
            shifted = chunk + (ref_std * float(shift_magnitude))
            chunk = np.clip(shifted, ref_min, ref_max)

        monitor_drift.window.clear()
        ev = None
        for row in chunk:
            ev = monitor_drift.observe(row)
        assert ev is not None
        drift_scores.append(float(ev.score))
        exceeded = bool(ev.score > float(threshold))
        if exceeded:
            drift_exceeded += 1
        if ev.retrain_requested or ev.consecutive >= int(consecutive_windows):
            if delay_windows is None:
                delay_windows = w_idx + 1
        if ev.retrain_requested:
            drift_retrain_reqs += 1
        curve_rows.append(
            {
                "Window_Index": num_baseline_windows + w_idx + 1,
                "Window": f"Drifted W{w_idx + 1}",
                "Regime": "Shifted Stream",
                "KL_Divergence": round(float(ev.score), 4),
                "Exceeded_Tau": exceeded,
                "Consecutive_Exceeded": int(ev.consecutive),
                "Retrain_Requested": bool(ev.retrain_requested),
            }
        )

    false_alarm_rate = float(base_exceeded / max(num_baseline_windows, 1))
    win_det_rate = float(drift_exceeded / max(num_drift_windows, 1))
    sustained_triggered = delay_windows is not None
    total_episodes = 1
    detected_episodes = 1 if sustained_triggered else 0
    episode_det_rate = float(detected_episodes / total_episodes)
    delay_flows = (delay_windows * int(window_size)) if delay_windows is not None else None

    return CleanDriftEvaluationResult(
        threshold_tau=float(threshold),
        consecutive_windows_target=int(consecutive_windows),
        window_size=int(window_size),
        shift_magnitude=float(shift_magnitude),
        total_baseline_windows=int(num_baseline_windows),
        baseline_exceeded_windows=int(base_exceeded),
        false_alarm_window_rate=round(false_alarm_rate, 4),
        baseline_retrain_requests=int(base_retrain_reqs),
        mean_kl_baseline=round(float(np.mean(base_scores)) if base_scores else 0.0, 4),
        total_drift_windows=int(num_drift_windows),
        drift_exceeded_windows=int(drift_exceeded),
        window_detection_rate=round(win_det_rate, 4),
        drift_retrain_requests=int(drift_retrain_reqs),
        mean_kl_drift=round(float(np.mean(drift_scores)) if drift_scores else 0.0, 4),
        total_drift_episodes=total_episodes,
        detected_drift_episodes=detected_episodes,
        episode_detection_rate=round(episode_det_rate, 4),
        sustained_drift_triggered=sustained_triggered,
        detection_delay_windows=delay_windows,
        detection_delay_flows=delay_flows,
        window_curve_df=pd.DataFrame(curve_rows),
    )


def run_local_loopback_simulation_verification(
    host: str = "127.0.0.1",
    port: int = 19999,
    temp_audit_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Task 5 Safe Local Loopback Verification:
    Starts a bounded local TCP listener on `127.0.0.1:<port>` (using a temporary audit log so
    `iot_ids_framework/artifacts/alerts.jsonl` is never overwritten), sends:
      1. Bounded benign IoT traffic flows (MQTT telemetry, CoAP actuator status, HTTP health check),
      2. Bounded simulated attack traffic flows (MQTT flood/DoS, high-entropy ransomware/backdoor payload,
         port-scan burst),
    routes the received socket payloads through `FlowAssembler` and `EdgeIDS.classify()`, and records
    ground truth, timestamps, observed flows, alerts, predicted labels, false positives, and missed detections.
    """
    import threading
    import onnxruntime as ort

    if temp_audit_path is None:
        temp_audit_path = BASE_DIR / "logs" / "local_sim_verification_alerts.jsonl"
    temp_audit_path.parent.mkdir(parents=True, exist_ok=True)
    if temp_audit_path.exists():
        temp_audit_path.unlink()

    cfg = load_framework_config()
    labels = json.loads(LABELS_JSON_PATH.read_text(encoding="utf-8"))
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    monitor = load_monitor(REFERENCE_JSON_PATH, cfg)
    explainer = load_tree_explainer(TREE_MODEL_PATH) if TREE_MODEL_PATH.exists() else None

    ids = EdgeIDS(
        session=session,
        class_names=labels,
        monitor=monitor,
        top_k=5,
        tree_explainer=explainer,
        audit_path=temp_audit_path,
    )

    # Define 3 benign flows + 3 bounded simulated attack flows sent over real loopback TCP sockets
    test_scenarios = [
        {
            "scenario_id": "benign_mqtt_telemetry",
            "ground_truth_binary": "benign",
            "ground_truth_category": "benign",
            "protocol": "mqtt",
            "src_port": 51200,
            "dst_port": 1883,
            "packet_count": 6,
            "duration_ms": 45.0,
            "tcp_flags": 24,
            "mqtt_qos": 0,
            "mqtt_payload_bytes": 42,
            "coap_code": 0,
            "http_status": 0,
            "sixlowpan_dispatch": 0,
            "payload_entropy": 3.45,
            "raw_payload": b"MQTT_PUBLISH topic=home/livingroom/temp value=22.4C",
        },
        {
            "scenario_id": "benign_coap_actuator",
            "ground_truth_binary": "benign",
            "ground_truth_category": "benign",
            "protocol": "coap",
            "src_port": 49200,
            "dst_port": 5683,
            "packet_count": 4,
            "duration_ms": 30.0,
            "tcp_flags": 0,
            "mqtt_qos": 0,
            "mqtt_payload_bytes": 0,
            "coap_code": 1,
            "http_status": 0,
            "sixlowpan_dispatch": 0,
            "payload_entropy": 3.10,
            "raw_payload": b"CoAP_GET /actuator/light/state",
        },
        {
            "scenario_id": "benign_http_health",
            "ground_truth_binary": "benign",
            "ground_truth_category": "benign",
            "protocol": "http",
            "src_port": 54100,
            "dst_port": 80,
            "packet_count": 8,
            "duration_ms": 65.0,
            "tcp_flags": 24,
            "mqtt_qos": 0,
            "mqtt_payload_bytes": 0,
            "coap_code": 0,
            "http_status": 200,
            "sixlowpan_dispatch": 0,
            "payload_entropy": 3.80,
            "raw_payload": b"GET /api/v1/health HTTP/1.1\r\nHost: gateway.local\r\n\r\n",
        },
        {
            "scenario_id": "attack_mqtt_ddos_flood",
            "ground_truth_binary": "malicious",
            "ground_truth_category": "ddos",
            "protocol": "mqtt",
            "src_port": 38112,
            "dst_port": 1883,
            "packet_count": 6500,
            "duration_ms": 520.0,
            "tcp_flags": 2,
            "mqtt_qos": 2,
            "mqtt_payload_bytes": 512,
            "coap_code": 0,
            "http_status": 0,
            "sixlowpan_dispatch": 0,
            "payload_entropy": 7.65,
            "raw_payload": b"SYN_FLOOD_MQTT_QOS2_" * 25,
        },
        {
            "scenario_id": "attack_port_scan_burst",
            "ground_truth_binary": "malicious",
            "ground_truth_category": "port_scan",
            "protocol": "other",
            "src_port": 58123,
            "dst_port": 23,
            "packet_count": 2,
            "duration_ms": 4.0,
            "tcp_flags": 2,
            "mqtt_qos": 0,
            "mqtt_payload_bytes": 0,
            "coap_code": 0,
            "http_status": 0,
            "sixlowpan_dispatch": 0,
            "payload_entropy": 0.50,
            "raw_payload": b"\x00\x02",
        },
        {
            "scenario_id": "attack_http_sql_injection",
            "ground_truth_binary": "malicious",
            "ground_truth_category": "injection",
            "protocol": "http",
            "src_port": 49881,
            "dst_port": 80,
            "packet_count": 24,
            "duration_ms": 160.0,
            "tcp_flags": 24,
            "mqtt_qos": 0,
            "mqtt_payload_bytes": 0,
            "coap_code": 0,
            "http_status": 500,
            "sixlowpan_dispatch": 0,
            "payload_entropy": 5.75,
            "raw_payload": b"POST /login HTTP/1.1\r\n\r\nuser=admin' OR 1=1--",
        },
    ]

    expected_connections = len(test_scenarios)
    received_Payloads: List[Tuple[str, Tuple[str, int], bytes, float]] = []
    actual_port = port

    def _connect_and_send_with_retry(target_host: str, target_port: int, payload: bytes, max_tries: int = 4) -> None:
        last_err: Optional[OSError] = None
        for attempt in range(max_tries):
            try:
                with socket.create_connection((target_host, target_port), timeout=2.0) as client_sock:
                    client_sock.sendall(payload)
                return
            except (PermissionError, ConnectionRefusedError) as exc:
                last_err = exc
                time.sleep(0.01 * (attempt + 1))
        assert last_err is not None
        raise last_err

    max_bind_attempts = 5 if port == 0 else 1
    last_bind_err: Optional[OSError] = None

    for bind_attempt in range(max_bind_attempts):
        received_Payloads.clear()
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            server_sock.bind((host, port))
            actual_port = int(server_sock.getsockname()[1])
            server_sock.listen(16)
            server_sock.settimeout(0.25)
        except OSError as exc:
            server_sock.close()
            last_bind_err = exc
            continue

        ready_event = threading.Event()
        stop_event = threading.Event()

        def _listener(srv: socket.socket, r_ev: threading.Event, s_ev: threading.Event) -> None:
            r_ev.set()
            while not s_ev.is_set() and len(received_Payloads) < expected_connections:
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                with conn:
                    conn.settimeout(2.0)
                    chunks: List[bytes] = []
                    while True:
                        chunk = conn.recv(65535)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    data = b"".join(chunks)
                    ts_now = time.time()
                    sep_idx = data.find(b"\n")
                    if sep_idx != -1:
                        header_str = data[:sep_idx].decode("utf-8", errors="replace")
                        body_bytes = data[sep_idx + 1 :]
                        received_Payloads.append((header_str, addr, body_bytes, ts_now))

        listener_thread = threading.Thread(
            target=_listener, args=(server_sock, ready_event, stop_event), daemon=True
        )
        listener_thread.start()
        try:
            ready_event.wait(timeout=2.0)
            for scen in test_scenarios:
                meta_hdr = json.dumps({k: v for k, v in scen.items() if k != "raw_payload"})
                wire_bytes = f"{meta_hdr}\n".encode("utf-8") + scen["raw_payload"]
                _connect_and_send_with_retry(host, actual_port, wire_bytes)

            deadline = time.time() + 3.0
            while len(received_Payloads) < expected_connections and time.time() < deadline:
                time.sleep(0.01)
            break
        except (PermissionError, OSError) as exc:
            last_bind_err = exc
            if bind_attempt + 1 == max_bind_attempts:
                raise
        finally:
            stop_event.set()
            try:
                server_sock.close()
            except OSError:
                pass
            listener_thread.join(timeout=2.0)
    else:
        if last_bind_err is not None:
            raise last_bind_err

    # Assemble flows via FlowAssembler and classify with EdgeIDS
    observed_records: List[Dict[str, Any]] = []
    false_positives = 0
    missed_detections = 0

    for meta_hdr, client_addr, body_bytes, rx_ts in received_Payloads:
        meta = json.loads(meta_hdr)
        assembler = FlowAssembler(idle_timeout_ms=100)
        pkt_count = int(meta["packet_count"])
        dur_ms = float(meta["duration_ms"])
        per_pkt_len = max(len(body_bytes), 32)
        base_ts_ms = int(rx_ts * 1000)

        # Feed start and end packets into FlowAssembler to verify bidirectional flow assembly
        assembler.submit(
            {
                "timestamp_ms": base_ts_ms,
                "src_ip": client_addr[0],
                "src_port": int(meta["src_port"]),
                "dst_ip": host,
                "dst_port": int(meta["dst_port"]),
                "protocol": meta["protocol"],
                "length": per_pkt_len * max(pkt_count * 2 - 1, 1),
                "tcp_flags": int(meta["tcp_flags"]),
                "mqtt_qos": int(meta["mqtt_qos"]),
                "mqtt_payload_bytes": int(meta["mqtt_payload_bytes"]),
                "coap_code": int(meta["coap_code"]),
                "http_status": int(meta["http_status"]),
                "sixlowpan_dispatch": int(meta["sixlowpan_dispatch"]),
                "payload_entropy": float(meta["payload_entropy"]),
            }
        )
        assembler.submit(
            {
                "timestamp_ms": base_ts_ms + int(dur_ms),
                "src_ip": client_addr[0],
                "src_port": int(meta["src_port"]),
                "dst_ip": host,
                "dst_port": int(meta["dst_port"]),
                "protocol": meta["protocol"],
                "length": per_pkt_len,
                "tcp_flags": int(meta["tcp_flags"]),
                "mqtt_qos": int(meta["mqtt_qos"]),
                "mqtt_payload_bytes": int(meta["mqtt_payload_bytes"]),
                "coap_code": int(meta["coap_code"]),
                "http_status": int(meta["http_status"]),
                "sixlowpan_dispatch": int(meta["sixlowpan_dispatch"]),
                "payload_entropy": float(meta["payload_entropy"]),
            }
        )
        expired_flows = assembler.expire(base_ts_ms + int(dur_ms) + 250)
        flow_dict = expired_flows[0]
        # Ensure exact packet count for high-rate flood scenarios without looping 6500 socket packets
        flow_dict["packet_count"] = pkt_count
        flow_dict["duration_ms"] = dur_ms
        # Preserve original direction src_port / dst_port from meta (since FlowAssembler._key sorts endpoints alphabetically)
        flow_dict["src_port"] = int(meta["src_port"])
        flow_dict["dst_port"] = int(meta["dst_port"])

        pred_out = ids.classify(flow_dict)
        pred_cls = str(pred_out["class"])
        pred_bin = "benign" if pred_cls == "benign" else "malicious"
        gt_bin = meta["ground_truth_binary"]
        if gt_bin == "benign" and pred_bin == "malicious":
            false_positives += 1
        elif gt_bin == "malicious" and pred_bin == "benign":
            missed_detections += 1

        observed_records.append(
            {
                "scenario_id": meta["scenario_id"],
                "timestamp_epoch": round(rx_ts, 4),
                "client_socket_addr": f"{client_addr[0]}:{client_addr[1]}",
                "server_listen_addr": f"{host}:{actual_port}",
                "ground_truth_binary": gt_bin,
                "ground_truth_category": meta["ground_truth_category"],
                "predicted_class": pred_cls,
                "predicted_binary": pred_bin,
                "confidence": round(float(pred_out["confidence"]), 4),
                "alert_emitted": "alert" in pred_out,
                "top_shap_feature": (
                    pred_out["alert"]["attribution"][0]["feature"]
                    if "alert" in pred_out and pred_out["alert"].get("attribution")
                    else None
                ),
            }
        )

    logged_alerts = []
    if temp_audit_path.exists():
        logged_alerts = [
            json.loads(line)
            for line in temp_audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return {
        "capture_architecture_verification": {
            "supports_live_nic_promiscuous_capture": False,
            "supports_pcap_file_replay": True,
            "supports_tcp_socket_payload_ingestion": True,
            "explanation": (
                "`iot_ids_framework` intentionally treats live NIC packet capture (libpcap/Npcap/eBPF) "
                "as an external adapter boundary (`README.md` line 79). It provides `PCAPReader` for "
                "offline `.pcap` file replay and `simulate_attacks.py` for TCP socket payload ingestion "
                "on a bound `<host>:<port>` (`127.0.0.1:9999`). A user-space TCP `SOCK_STREAM` listener "
                "only receives connections addressed directly to its bound port; it cannot sniff traffic "
                "between other ports or interfaces without a raw packet capture driver."
            ),
        },
        "test_endpoint": f"{host}:{actual_port}",
        "temp_audit_log_path": str(temp_audit_path),
        "total_flows_sent": len(test_scenarios),
        "total_flows_observed": len(observed_records),
        "alerts_logged_count": len(logged_alerts),
        "false_positives": false_positives,
        "missed_detections": missed_detections,
        "observed_flows": observed_records,
    }


SIMULATION_ATTACK_PROFILES: Tuple[str, ...] = (
    "ddos",
    "dos",
    "injection",
    "mitm",
    "password",
    "port_scan",
    "ransomware",
    "scanning",
    "spoofing",
    "xss",
    "backdoor",
    "botnet",
    "brute_force",
    "web_attack",
)


def _sample_flow_spec_for_profile(
    cls_name: str,
    rng: np.random.Generator,
    flow_index: int,
) -> Dict[str, Any]:
    """
    Generate a single randomized, reproducible IoT flow specification and valid L7 payload
    matching the 15-class APAFC-14 schema (`benign` or one of the 14 attack profiles).
    """
    rev_proto = {0: "other", 1: "mqtt", 2: "coap", 3: "http", 4: "6lowpan"}

    if cls_name == "benign":
        proto_id = int(rng.choice([1, 2, 3, 4], p=[0.40, 0.20, 0.25, 0.15]))
        pkts = int(rng.integers(3, 16))
        dur = round(float(rng.uniform(15.0, 280.0)), 2)
        sport = int(rng.integers(32000, 62000))
        dport = {1: 1883, 2: 5683, 3: 80, 4: 61616}[proto_id]
        t_flags = 24 if proto_id in (1, 3) else 0
        m_qos = int(rng.choice([0, 1])) if proto_id == 1 else 0
        m_bytes = int(rng.integers(24, 96)) if proto_id == 1 else 0
        c_code = int(rng.choice([1, 69])) if proto_id == 2 else 0
        h_stat = 200 if proto_id == 3 else 0
        s_disp = 65 if proto_id == 4 else 0
        ent = round(float(rng.uniform(2.9, 4.4)), 4)
        b_cnt = int(round(pkts * float(rng.uniform(42.0, 110.0))))
        if proto_id == 1:
            raw_payload = f"\x32\x18\x00\x0bsensor/node{flow_index}/temp={rng.uniform(20.0, 26.0):.1f}C".encode("utf-8")
        elif proto_id == 2:
            raw_payload = f"\x40\x01\x12\x34\xb5state\xffnode{flow_index}=ok".encode("utf-8")
        elif proto_id == 3:
            raw_payload = f"GET /api/v1/telemetry?node={flow_index} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode("utf-8")
        else:
            raw_payload = f"\x41\x06\x40mesh-node-{flow_index}-ok".encode("utf-8")
    elif cls_name == "ddos":
        proto_id = int(rng.choice([1, 2, 3]))
        pkts = int(rng.integers(3000, 9000))
        b_cnt = int(round(pkts * float(rng.uniform(130.0, 230.0))))
        dur = round(float(rng.uniform(300.0, 850.0)), 2)
        sport = int(rng.integers(12000, 64000))
        dport = {1: 1883, 2: 5683, 3: 80}[proto_id]
        t_flags = 2
        m_qos = 2 if proto_id == 1 else 0
        m_bytes = 512 if proto_id == 1 else 0
        c_code = 1 if proto_id == 2 else 0
        h_stat = 503 if proto_id == 3 else 0
        s_disp = 0
        ent = round(float(rng.uniform(7.0, 7.9)), 4)
        raw_payload = f"DDOS_FLOOD_SEQ_{flow_index}_".encode("utf-8") * 16
    elif cls_name == "dos":
        proto_id = 1
        pkts = int(rng.integers(900, 2200))
        b_cnt = int(round(pkts * float(rng.uniform(90.0, 150.0))))
        dur = round(float(rng.uniform(120.0, 420.0)), 2)
        sport = int(rng.integers(20000, 60000))
        dport = 1883
        t_flags = 18
        m_qos = 1
        m_bytes = 256
        c_code = 0
        h_stat = 0
        s_disp = 0
        ent = round(float(rng.uniform(5.7, 6.7)), 4)
        raw_payload = f"\x32\x7f\x00\x0cdos/flood/{flow_index}/".encode("utf-8") * 8
    elif cls_name == "injection":
        proto_id = 3
        pkts = int(rng.integers(14, 40))
        b_cnt = int(round(pkts * float(rng.uniform(350.0, 640.0))))
        dur = round(float(rng.uniform(90.0, 260.0)), 2)
        sport = int(rng.integers(40000, 64000))
        dport = 80
        t_flags = 24
        m_qos = 0
        m_bytes = 0
        c_code = 0
        h_stat = int(rng.choice([200, 500, 403]))
        s_disp = 0
        ent = round(float(rng.uniform(5.3, 6.1)), 4)
        raw_payload = f"POST /login HTTP/1.1\r\n\r\nuser=admin' UNION SELECT {flow_index}--".encode("utf-8")
    elif cls_name == "mitm":
        proto_id = 4
        pkts = int(rng.integers(45, 130))
        b_cnt = int(round(pkts * float(rng.uniform(95.0, 175.0))))
        dur = round(float(rng.uniform(320.0, 900.0)), 2)
        sport = int(rng.integers(12000, 39000))
        dport = 61616
        t_flags = 16
        m_qos = 0
        m_bytes = 0
        c_code = 0
        h_stat = 0
        s_disp = int(rng.integers(125, 185))
        ent = round(float(rng.uniform(5.9, 6.8)), 4)
        raw_payload = bytes([s_disp]) + f"mitm-6lowpan-relay-{flow_index}".encode("utf-8")
    elif cls_name == "password":
        proto_id = 1
        pkts = int(rng.integers(70, 210))
        b_cnt = int(round(pkts * float(rng.uniform(48.0, 82.0))))
        dur = round(float(rng.uniform(130.0, 360.0)), 2)
        sport = int(rng.integers(31000, 59000))
        dport = 1883
        t_flags = 20
        m_qos = 0
        m_bytes = int(rng.integers(52, 92))
        c_code = 0
        h_stat = 0
        s_disp = 0
        ent = round(float(rng.uniform(4.9, 5.6)), 4)
        raw_payload = f"\x10\x24\x00\x04MQTT\x04\xc2\x00\x3cclient{flow_index}:pass_attempt".encode("utf-8")
    elif cls_name == "port_scan":
        proto_id = 0
        pkts = int(rng.integers(1, 4))
        b_cnt = int(max(pkts * int(rng.integers(2, 14)), 2))
        dur = round(float(rng.uniform(1.5, 12.0)), 2)
        sport = int(rng.integers(45000, 64500))
        dport = int(rng.integers(1, 1024))
        t_flags = 2
        m_qos = 0
        m_bytes = 0
        c_code = 0
        h_stat = 0
        s_disp = 0
        ent = round(float(rng.uniform(0.1, 1.3)), 4)
        raw_payload = b"\x00\x02"
    elif cls_name == "ransomware":
        proto_id = 1
        pkts = int(rng.integers(160, 460))
        b_cnt = int(round(pkts * float(rng.uniform(580.0, 1050.0))))
        dur = round(float(rng.uniform(650.0, 1700.0)), 2)
        sport = int(rng.integers(26000, 54000))
        dport = 8883
        t_flags = 24
        m_qos = 2
        m_bytes = int(rng.integers(620, 1150))
        c_code = 0
        h_stat = 0
        s_disp = 0
        ent = round(float(rng.uniform(7.45, 7.95)), 4)
        raw_payload = f"\x34\x40\x00\x0aransom/key/{flow_index}".encode("utf-8")
    elif cls_name == "scanning":
        proto_id = 2
        pkts = int(rng.integers(9, 26))
        b_cnt = int(round(pkts * float(rng.uniform(22.0, 48.0))))
        dur = round(float(rng.uniform(22.0, 76.0)), 2)
        sport = int(rng.integers(36000, 64000))
        dport = 5683
        t_flags = 4
        m_qos = 0
        m_bytes = 0
        c_code = int(rng.integers(132, 196))
        h_stat = 0
        s_disp = 0
        ent = round(float(rng.uniform(2.1, 3.4)), 4)
        raw_payload = bytes([0x40, c_code, 0x00, flow_index & 0xFF]) + b"/.well-known/core"
    elif cls_name == "spoofing":
        proto_id = 2
        pkts = int(rng.integers(40, 105))
        b_cnt = int(round(pkts * float(rng.uniform(115.0, 250.0))))
        dur = round(float(rng.uniform(160.0, 480.0)), 2)
        sport = 5683
        dport = int(rng.integers(11000, 49000))
        t_flags = 0
        m_qos = 0
        m_bytes = 0
        c_code = int(rng.integers(65, 102))
        h_stat = 0
        s_disp = 0
        ent = round(float(rng.uniform(4.6, 5.7)), 4)
        raw_payload = bytes([0x60, c_code, 0x11, flow_index & 0xFF]) + f"spoofed-coap-{flow_index}".encode("utf-8")
    elif cls_name == "xss":
        proto_id = 3
        pkts = int(rng.integers(11, 30))
        b_cnt = int(round(pkts * float(rng.uniform(190.0, 350.0))))
        dur = round(float(rng.uniform(65.0, 210.0)), 2)
        sport = int(rng.integers(31000, 59000))
        dport = 8080
        t_flags = 24
        m_qos = 0
        m_bytes = 0
        c_code = 0
        h_stat = 400
        s_disp = 0
        ent = round(float(rng.uniform(5.0, 5.8)), 4)
        raw_payload = f"GET /search?q=<script>alert({flow_index})</script> HTTP/1.1\r\n\r\n".encode("utf-8")
    elif cls_name == "backdoor":
        proto_id = 1
        pkts = int(rng.integers(50, 150))
        b_cnt = int(round(pkts * float(rng.uniform(420.0, 860.0))))
        dur = round(float(rng.uniform(850.0, 2300.0)), 2)
        sport = int(rng.integers(41000, 64000))
        dport = 4444
        t_flags = 24
        m_qos = 1
        m_bytes = int(rng.integers(380, 720))
        c_code = 0
        h_stat = 0
        s_disp = 0
        ent = round(float(rng.uniform(7.15, 7.90)), 4)
        raw_payload = f"\x32\x30\x00\x08c2/shell/{flow_index}/exec".encode("utf-8")
    elif cls_name == "botnet":
        proto_id = 4
        pkts = int(rng.integers(260, 720))
        b_cnt = int(round(pkts * float(rng.uniform(150.0, 300.0))))
        dur = round(float(rng.uniform(420.0, 1150.0)), 2)
        sport = int(rng.integers(21000, 49000))
        dport = 61616
        t_flags = 16
        m_qos = 0
        m_bytes = 0
        c_code = 0
        h_stat = 0
        s_disp = int(rng.integers(202, 248))
        ent = round(float(rng.uniform(6.3, 7.1)), 4)
        raw_payload = bytes([s_disp]) + f"botnet-beacon-{flow_index}".encode("utf-8")
    elif cls_name == "brute_force":
        proto_id = 3
        pkts = int(rng.integers(95, 270))
        b_cnt = int(round(pkts * float(rng.uniform(100.0, 170.0))))
        dur = round(float(rng.uniform(210.0, 580.0)), 2)
        sport = int(rng.integers(31000, 59000))
        dport = 443
        t_flags = 20
        m_qos = 0
        m_bytes = 0
        c_code = 0
        h_stat = 401
        s_disp = 0
        ent = round(float(rng.uniform(5.1, 6.0)), 4)
        raw_payload = f"HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Basic attempt={flow_index}\r\n\r\n".encode("utf-8")
    else:  # web_attack
        proto_id = 3
        pkts = int(rng.integers(22, 70))
        b_cnt = int(round(pkts * float(rng.uniform(270.0, 500.0))))
        dur = round(float(rng.uniform(120.0, 375.0)), 2)
        sport = int(rng.integers(33000, 61000))
        dport = 80
        t_flags = 24
        m_qos = 0
        m_bytes = 0
        c_code = 0
        h_stat = 404
        s_disp = 0
        ent = round(float(rng.uniform(5.5, 6.4)), 4)
        raw_payload = f"GET /../../etc/passwd?seq={flow_index} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode("utf-8")

    return {
        "flow_index": flow_index,
        "scenario_id": f"sim_{flow_index:03d}_{cls_name}",
        "ground_truth_binary": "benign" if cls_name == "benign" else "malicious",
        "ground_truth_category": cls_name,
        "protocol": rev_proto[proto_id],
        "src_port": sport,
        "dst_port": dport,
        "packet_count": pkts,
        "byte_count": b_cnt,
        "duration_ms": dur,
        "tcp_flags": t_flags,
        "mqtt_qos": m_qos,
        "mqtt_payload_bytes": m_bytes,
        "coap_code": c_code,
        "http_status": h_stat,
        "sixlowpan_dispatch": s_disp,
        "payload_entropy": ent,
        "raw_payload": raw_payload,
    }


_IOT23_SIMULATION_POOL_CACHE: Optional[Dict[str, List[Dict[str, Any]]]] = None


def _load_real_iot23_simulation_pool(
    max_rows_per_file: int = 800,
    max_packets_per_flow: int = 200,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Stream bounded chunks from representative extracted IoT-23 `conn.log.labeled` files via
    `src/iot23_loader.py` (`iter_zeek_conn_chunks` and `map_zeek_to_apafc14`) and cache a
    reproducible pool of real Benign and real Malicious TCP/UDP connection records with full
    Zeek provenance.
    """
    global _IOT23_SIMULATION_POOL_CACHE
    if _IOT23_SIMULATION_POOL_CACHE is not None:
        return _IOT23_SIMULATION_POOL_CACHE

    from src.iot23_loader import (
        IOT23_BINARY_LABEL_MAP,
        IOT23_DETAILED_TO_TAXONOMY_MAP,
        IOT23_RAW_ROOT,
        IOT23_TO_CORE15_APPROX_MAP,
        iter_zeek_conn_chunks,
        map_zeek_to_apafc14,
    )

    if not IOT23_RAW_ROOT.exists():
        raise FileNotFoundError(f"Extracted IoT-23 dataset directory not found at {IOT23_RAW_ROOT}")

    scen_root = IOT23_RAW_ROOT / "opt" / "Malware-Project" / "BigDataset" / "IoTScenarios"
    candidate_scenarios = [
        "CTU-Honeypot-Capture-4-1",
        "CTU-Honeypot-Capture-5-1",
        "Somfy-01",
        "CTU-IoT-Malware-Capture-34-1",
        "CTU-IoT-Malware-Capture-20-1",
        "CTU-IoT-Malware-Capture-44-1",
        "CTU-IoT-Malware-Capture-8-1",
        "CTU-IoT-Malware-Capture-3-1",
        "CTU-IoT-Malware-Capture-36-1",
    ]

    rev_proto = {0: "other", 1: "mqtt", 2: "coap", 3: "http", 4: "6lowpan"}
    benign_pool: List[Dict[str, Any]] = []
    malicious_pool: List[Dict[str, Any]] = []
    seen_port_pairs: set = set()

    for scen_name in candidate_scenarios:
        fp = scen_root / scen_name / "bro" / "conn.log.labeled"
        if not fp.exists():
            continue
        rel_fp = str(fp.relative_to(IOT23_RAW_ROOT))
        for chunk_df in iter_zeek_conn_chunks(fp, chunk_size=max_rows_per_file, max_rows=max_rows_per_file):
            apafc_df = map_zeek_to_apafc14(chunk_df)
            for row_idx in range(len(chunk_df)):
                z_row = chunk_df.iloc[row_idx]
                a_row = apafc_df.iloc[row_idx]
                z_proto = str(z_row["proto"]).strip().lower()
                if z_proto not in ("tcp", "udp"):
                    continue

                sport = int(float(a_row["src_port"]))
                dport = int(float(a_row["dst_port"]))
                if not (1 <= sport <= 65535 and 1 <= dport <= 65535):
                    continue

                pkt_cnt = max(int(round(float(a_row["packet_count"]))), 1)
                if pkt_cnt > max_packets_per_flow:
                    continue

                byte_cnt = max(int(round(float(a_row["byte_count"]))), 0)
                dur_ms = round(max(float(a_row["duration_ms"]), 0.0), 3)
                if pkt_cnt > 1 and (dur_ms / float(pkt_cnt - 1)) > 850.0:
                    # Exclude multi-second retransmission gaps > FlowAssembler 1.0s idle_timeout_ms
                    continue

                port_pair_key = (z_proto, min(sport, dport), max(sport, dport))
                if port_pair_key in seen_port_pairs:
                    continue
                seen_port_pairs.add(port_pair_key)

                proto_id = int(round(float(a_row["protocol_id"])))
                t_flags = int(round(float(a_row["tcp_flags"]))) if z_proto == "tcp" else 0

                raw_lbl = str(z_row["label"]).strip()
                det_lbl = str(z_row["detailed-label"]).strip()
                gt_bin = IOT23_BINARY_LABEL_MAP.get(raw_lbl, "benign")
                tax_fam = IOT23_DETAILED_TO_TAXONOMY_MAP.get(det_lbl if gt_bin == "malicious" else "-", "Benign")
                proj_cat = IOT23_TO_CORE15_APPROX_MAP.get(det_lbl if gt_bin == "malicious" else "-", "benign")

                entry = {
                    "data_source": "real_iot23",
                    "ground_truth_binary": gt_bin,
                    "ground_truth_category": proj_cat,
                    "iot23_raw_label": raw_lbl,
                    "iot23_detailed_label": det_lbl,
                    "iot23_taxonomy_family": tax_fam,
                    "protocol": rev_proto.get(proto_id, "other"),
                    "transport_proto": z_proto,
                    "src_port": sport,
                    "dst_port": dport,
                    "packet_count": pkt_cnt,
                    "byte_count": byte_cnt,
                    "duration_ms": dur_ms,
                    "tcp_flags": t_flags,
                    "mqtt_qos": 0,
                    "mqtt_payload_bytes": 0,
                    "coap_code": 0,
                    "http_status": 0,
                    "sixlowpan_dispatch": 0,
                    "payload_entropy": 0.0,
                    "raw_payload": b"\x00" * min(max(byte_cnt, 0), 64),
                    "iot23_provenance": {
                        "source_scenario": scen_name,
                        "source_file": rel_fp,
                        "zeek_uid": str(z_row["uid"]).strip(),
                        "zeek_ts": str(z_row["ts"]).strip(),
                        "zeek_orig_h": str(z_row["id.orig_h"]).strip(),
                        "zeek_resp_h": str(z_row["id.resp_h"]).strip(),
                        "zeek_proto": z_proto,
                        "zeek_service": str(z_row["service"]).strip(),
                        "zeek_conn_state": str(z_row["conn_state"]).strip(),
                        "zeek_history": str(z_row["history"]).strip(),
                        "orig_pkts": str(z_row["orig_pkts"]).strip(),
                        "resp_pkts": str(z_row["resp_pkts"]).strip(),
                        "orig_bytes": str(z_row["orig_bytes"]).strip(),
                        "resp_bytes": str(z_row["resp_bytes"]).strip(),
                        "orig_ip_bytes": str(z_row["orig_ip_bytes"]).strip(),
                        "resp_ip_bytes": str(z_row["resp_ip_bytes"]).strip(),
                        "raw_label": raw_lbl,
                        "detailed_label": det_lbl,
                        "iot23_taxonomy_family": tax_fam,
                        "projected_core15_label": proj_cat,
                        "unavailable_zero_filled_apafc_features": [
                            "mqtt_qos",
                            "mqtt_payload_bytes",
                            "coap_code",
                            "http_status",
                            "sixlowpan_dispatch",
                            "payload_entropy",
                        ],
                    },
                }
                if gt_bin == "benign":
                    benign_pool.append(entry)
                else:
                    malicious_pool.append(entry)

    _IOT23_SIMULATION_POOL_CACHE = {"benign": benign_pool, "malicious": malicious_pool}
    return _IOT23_SIMULATION_POOL_CACHE


def sample_real_iot23_flows_for_simulation(
    n_flows: int = 12,
    attack_ratio: float = 0.5,
    random_seed: Optional[int] = 2026,
    max_packets_per_flow: int = 200,
) -> List[Dict[str, Any]]:
    """
    Sample a deterministic, reproducible sequence of real IoT-23 benign and malicious flow
    records from `data/raw/iot_23_datasets_small` via `src/iot23_loader.py`.
    """
    if int(n_flows) < 1:
        raise ValueError(f"n_flows must be >= 1, got {n_flows}")
    if not (0.0 <= float(attack_ratio) <= 1.0):
        raise ValueError(f"attack_ratio must be in [0.0, 1.0], got {attack_ratio}")

    pool = _load_real_iot23_simulation_pool(max_packets_per_flow=max_packets_per_flow)
    benign_pool = pool["benign"]
    mal_pool = pool["malicious"]
    if not benign_pool or not mal_pool:
        raise RuntimeError("Real IoT-23 flow pool is empty; verify data/raw/iot_23_datasets_small.")

    rng = np.random.default_rng(random_seed)
    n_total = int(n_flows)
    n_attack = int(round(n_total * float(attack_ratio)))
    n_benign = n_total - n_attack

    b_indices = rng.choice(len(benign_pool), size=n_benign, replace=(n_benign > len(benign_pool)))
    m_indices = rng.choice(len(mal_pool), size=n_attack, replace=(n_attack > len(mal_pool)))

    combined: List[Dict[str, Any]] = [dict(benign_pool[int(i)]) for i in b_indices] + [
        dict(mal_pool[int(i)]) for i in m_indices
    ]
    perm = rng.permutation(n_total)

    sampled: List[Dict[str, Any]] = []
    import re as _re

    for seq_idx, p_idx in enumerate(perm):
        rec = dict(combined[int(p_idx)])
        flow_num = seq_idx + 1
        rec["flow_index"] = flow_num
        uid_short = rec["iot23_provenance"]["zeek_uid"][:10]
        det_clean = _re.sub(r"[^A-Za-z0-9_]+", "_", rec["iot23_taxonomy_family"])
        rec["scenario_id"] = f"iot23_{flow_num:03d}_{uid_short}_{det_clean}"
        sampled.append(rec)

    return sampled


def generate_randomized_simulation_flows(
    n_flows: int = 12,
    attack_ratio: float = 0.5,
    random_seed: Optional[int] = 42,
    attack_profiles: Optional[List[str]] = None,
    data_source: str = "synthetic_profiles",
) -> List[Dict[str, Any]]:
    """
    Generate a controlled, reproducible sequence of benign and attack IoT flow specs.
    Supports `data_source="real_iot23"` (streamed from real IoT-23 `conn.log.labeled` files)
    and `data_source="synthetic_profiles"` (15-class APAFC synthetic profiles).
    """
    if int(n_flows) < 1:
        raise ValueError(f"n_flows must be >= 1, got {n_flows}")
    if not (0.0 <= float(attack_ratio) <= 1.0):
        raise ValueError(f"attack_ratio must be in [0.0, 1.0], got {attack_ratio}")

    if data_source == "real_iot23":
        return sample_real_iot23_flows_for_simulation(
            n_flows=n_flows,
            attack_ratio=attack_ratio,
            random_seed=random_seed,
        )
    if data_source != "synthetic_profiles":
        raise ValueError(f"Unsupported data_source={data_source!r}; expected 'real_iot23' or 'synthetic_profiles'.")

    active_profiles = list(attack_profiles) if attack_profiles else list(SIMULATION_ATTACK_PROFILES)
    invalid = [p for p in active_profiles if p not in SIMULATION_ATTACK_PROFILES]
    if invalid:
        raise ValueError(f"Unsupported attack profiles: {invalid}. Valid profiles: {SIMULATION_ATTACK_PROFILES}")

    rng = np.random.default_rng(random_seed)
    n_total = int(n_flows)
    n_attack = int(round(n_total * float(attack_ratio)))
    n_benign = n_total - n_attack

    chosen_classes: List[str] = ["benign"] * n_benign
    for i in range(n_attack):
        chosen_classes.append(str(rng.choice(active_profiles)))

    # Deterministically shuffle flow order using the seeded generator
    perm = rng.permutation(n_total)
    ordered_classes = [chosen_classes[int(idx)] for idx in perm]

    return [
        _sample_flow_spec_for_profile(cls_name, rng, flow_index=i + 1)
        for i, cls_name in enumerate(ordered_classes)
    ]


def run_randomized_local_attack_simulation(
    n_flows: int = 12,
    attack_ratio: float = 0.5,
    random_seed: Optional[int] = 42,
    attack_profiles: Optional[List[str]] = None,
    host: str = "127.0.0.1",
    port: int = 0,
    audit_log_path: Optional[Path] = None,
    persist_to_core_alerts: bool = False,
    data_source: str = "synthetic_profiles",
) -> Dict[str, Any]:
    """
    Stage 2 / Stage 3 — Controlled, reproducible local loopback attack simulator and live IDS observer.

    Supports both:
    - `data_source="real_iot23"`: Seeds each simulated loopback flow from real IoT-23
      `conn.log.labeled` records (preserving Zeek `uid`, capture scenario, ports, duration,
      packet/byte counts, reconstructed TCP flags, and ground-truth IoT-23 labels).
    - `data_source="synthetic_profiles"`: Seeds each flow from the 15-class APAFC profiles.
    """
    import tempfile
    import threading
    import onnxruntime as ort

    if host not in ("127.0.0.1", "localhost"):
        raise ValueError(
            f"Local simulator is restricted to loopback (127.0.0.1 / localhost); refused host={host!r}."
        )
    bind_host = "127.0.0.1"

    scenarios = generate_randomized_simulation_flows(
        n_flows=n_flows,
        attack_ratio=attack_ratio,
        random_seed=random_seed,
        attack_profiles=attack_profiles,
        data_source=data_source,
    )

    cleanup_temp_audit = audit_log_path is None
    if audit_log_path is None:
        fd, tmp_name = tempfile.mkstemp(prefix="iot_ids_stage2_alerts_", suffix=".jsonl")
        import os

        os.close(fd)
        effective_audit_path = Path(tmp_name)
    else:
        effective_audit_path = Path(audit_log_path)
        effective_audit_path.parent.mkdir(parents=True, exist_ok=True)
        if effective_audit_path.exists():
            effective_audit_path.unlink()

    cfg = load_framework_config()
    labels = json.loads(LABELS_JSON_PATH.read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    monitor = load_monitor(REFERENCE_JSON_PATH, cfg)
    explainer = load_tree_explainer(TREE_MODEL_PATH) if TREE_MODEL_PATH.exists() else None

    ids = EdgeIDS(
        session=session,
        class_names=labels,
        monitor=monitor,
        top_k=5,
        tree_explainer=explainer,
        audit_path=effective_audit_path,
    )

    expected_connections = len(scenarios)
    received_payloads: List[Tuple[str, Tuple[str, int], bytes, float]] = []
    actual_port = port
    server_sock_closed = False
    listener_thread_ref: Optional[threading.Thread] = None

    def _connect_and_send_with_retry(
        target_host: str, target_port: int, payload: bytes, max_tries: int = 4
    ) -> None:
        last_err: Optional[OSError] = None
        for attempt in range(max_tries):
            try:
                with socket.create_connection((target_host, target_port), timeout=2.0) as client_sock:
                    client_sock.sendall(payload)
                return
            except (PermissionError, ConnectionRefusedError) as exc:
                last_err = exc
                time.sleep(0.01 * (attempt + 1))
        assert last_err is not None
        raise last_err

    try:
        max_bind_attempts = 5 if port == 0 else 1
        last_bind_err: Optional[OSError] = None

        for bind_attempt in range(max_bind_attempts):
            received_payloads.clear()
            server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                server_sock.bind((bind_host, port))
                actual_port = int(server_sock.getsockname()[1])
                server_sock.listen(max(16, min(expected_connections + 4, 128)))
                server_sock.settimeout(0.25)
            except OSError as exc:
                server_sock.close()
                last_bind_err = exc
                continue

            ready_event = threading.Event()
            stop_event = threading.Event()

            def _listener(srv: socket.socket, r_ev: threading.Event, s_ev: threading.Event) -> None:
                r_ev.set()
                while not s_ev.is_set() and len(received_payloads) < expected_connections:
                    try:
                        conn, addr = srv.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    with conn:
                        conn.settimeout(2.0)
                        chunks: List[bytes] = []
                        while True:
                            chunk = conn.recv(65535)
                            if not chunk:
                                break
                            chunks.append(chunk)
                        data = b"".join(chunks)
                        ts_now = time.time()
                        sep_idx = data.find(b"\n")
                        if sep_idx != -1:
                            header_str = data[:sep_idx].decode("utf-8", errors="replace")
                            body_bytes = data[sep_idx + 1 :]
                            received_payloads.append((header_str, addr, body_bytes, ts_now))

            listener_thread = threading.Thread(
                target=_listener, args=(server_sock, ready_event, stop_event), daemon=True
            )
            listener_thread_ref = listener_thread
            listener_thread.start()
            try:
                ready_event.wait(timeout=2.0)
                for scen in scenarios:
                    meta_hdr = json.dumps({k: v for k, v in scen.items() if k != "raw_payload"})
                    wire_bytes = f"{meta_hdr}\n".encode("utf-8") + scen["raw_payload"]
                    _connect_and_send_with_retry(bind_host, actual_port, wire_bytes)

                deadline = time.time() + max(3.0, expected_connections * 0.1)
                while len(received_payloads) < expected_connections and time.time() < deadline:
                    time.sleep(0.01)
                break
            except (PermissionError, OSError) as exc:
                last_bind_err = exc
                if bind_attempt + 1 == max_bind_attempts:
                    raise
            finally:
                stop_event.set()
                try:
                    server_sock.close()
                    server_sock_closed = True
                except OSError:
                    pass
                listener_thread.join(timeout=2.0)
        else:
            if last_bind_err is not None:
                raise last_bind_err

        observed_records: List[Dict[str, Any]] = []
        false_positives = 0
        missed_detections = 0
        exact_class_matches = 0
        latencies_ms: List[float] = []

        for meta_hdr, client_addr, body_bytes, rx_ts in received_payloads:
            meta = json.loads(meta_hdr)
            assembler = FlowAssembler(idle_timeout_ms=100)
            pkt_count = int(meta["packet_count"])
            total_bytes = int(meta["byte_count"])
            dur_ms = float(meta["duration_ms"])
            base_ts_ms = int(rx_ts * 1000)

            first_len = max(total_bytes - max(len(body_bytes), 1), 1) if pkt_count > 1 else total_bytes
            second_len = max(total_bytes - first_len, 1) if pkt_count > 1 else 0

            assembler.submit(
                {
                    "timestamp_ms": base_ts_ms,
                    "src_ip": client_addr[0],
                    "src_port": int(meta["src_port"]),
                    "dst_ip": bind_host,
                    "dst_port": int(meta["dst_port"]),
                    "protocol": meta["protocol"],
                    "length": first_len,
                    "tcp_flags": int(meta["tcp_flags"]),
                    "mqtt_qos": int(meta["mqtt_qos"]),
                    "mqtt_payload_bytes": int(meta["mqtt_payload_bytes"]),
                    "coap_code": int(meta["coap_code"]),
                    "http_status": int(meta["http_status"]),
                    "sixlowpan_dispatch": int(meta["sixlowpan_dispatch"]),
                    "payload_entropy": float(meta["payload_entropy"]),
                }
            )
            if pkt_count > 1:
                assembler.submit(
                    {
                        "timestamp_ms": base_ts_ms + int(round(dur_ms)),
                        "src_ip": client_addr[0],
                        "src_port": int(meta["src_port"]),
                        "dst_ip": bind_host,
                        "dst_port": int(meta["dst_port"]),
                        "protocol": meta["protocol"],
                        "length": second_len,
                        "tcp_flags": int(meta["tcp_flags"]),
                        "mqtt_qos": int(meta["mqtt_qos"]),
                        "mqtt_payload_bytes": int(meta["mqtt_payload_bytes"]),
                        "coap_code": int(meta["coap_code"]),
                        "http_status": int(meta["http_status"]),
                        "sixlowpan_dispatch": int(meta["sixlowpan_dispatch"]),
                        "payload_entropy": float(meta["payload_entropy"]),
                    }
                )

            expired_flows = assembler.expire(base_ts_ms + int(round(dur_ms)) + 250)
            flow_dict = expired_flows[0]
            flow_dict["packet_count"] = pkt_count
            flow_dict["byte_count"] = total_bytes
            flow_dict["duration_ms"] = dur_ms
            flow_dict["src_ip"] = client_addr[0]
            flow_dict["src_port"] = int(meta["src_port"])
            flow_dict["dst_ip"] = bind_host
            flow_dict["dst_port"] = int(meta["dst_port"])

            t0 = time.perf_counter()
            pred_out = ids.classify(flow_dict)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

            pred_cls = str(pred_out["class"])
            pred_bin = "benign" if pred_cls == "benign" else "malicious"
            gt_bin = str(meta["ground_truth_binary"])
            gt_cat = str(meta["ground_truth_category"])

            if gt_bin == "benign" and pred_bin == "malicious":
                false_positives += 1
            elif gt_bin == "malicious" and pred_bin == "benign":
                missed_detections += 1
            if pred_cls == gt_cat:
                exact_class_matches += 1

            observed_records.append(
                {
                    "flow_index": int(meta["flow_index"]),
                    "scenario_id": str(meta["scenario_id"]),
                    "data_source": str(meta.get("data_source", data_source)),
                    "transport_proto": str(meta.get("transport_proto", "udp" if meta["protocol"] in ("coap", "6lowpan") else "tcp")),
                    "timestamp_epoch": round(rx_ts, 4),
                    "client_socket_addr": f"{client_addr[0]}:{client_addr[1]}",
                    "server_listen_addr": f"{bind_host}:{actual_port}",
                    "protocol": str(meta["protocol"]),
                    "src_ip": str(flow_dict["src_ip"]),
                    "src_port": int(flow_dict["src_port"]),
                    "dst_ip": str(flow_dict["dst_ip"]),
                    "dst_port": int(flow_dict["dst_port"]),
                    "packet_count": pkt_count,
                    "byte_count": total_bytes,
                    "duration_ms": dur_ms,
                    "payload_entropy": float(flow_dict["payload_entropy"]),
                    "ground_truth_binary": gt_bin,
                    "ground_truth_category": gt_cat,
                    "iot23_raw_label": meta.get("iot23_raw_label"),
                    "iot23_detailed_label": meta.get("iot23_detailed_label"),
                    "iot23_taxonomy_family": meta.get("iot23_taxonomy_family"),
                    "iot23_provenance": meta.get("iot23_provenance"),
                    "predicted_class": pred_cls,
                    "predicted_binary": pred_bin,
                    "confidence": round(float(pred_out["confidence"]), 4),
                    "inference_latency_ms": round(lat_ms, 3),
                    "alert_emitted": "alert" in pred_out,
                    "top_shap_feature": (
                        pred_out["alert"]["attribution"][0]["feature"]
                        if "alert" in pred_out and pred_out["alert"].get("attribution")
                        else None
                    ),
                    "reassembled_flow": dict(flow_dict),
                }
            )

        logged_alerts: List[Dict[str, Any]] = []
        if effective_audit_path.exists():
            raw_lines = [
                line
                for line in effective_audit_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            logged_alerts = [json.loads(line) for line in raw_lines]
            if persist_to_core_alerts and raw_lines:
                ALERTS_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
                with ALERTS_JSONL_PATH.open("a", encoding="utf-8") as core_f:
                    for line in raw_lines:
                        core_f.write(line + "\n")

        n_obs = max(len(observed_records), 1)
        benign_sent = sum(1 for s in scenarios if s["ground_truth_binary"] == "benign")
        mal_sent = len(scenarios) - benign_sent
        pred_benign = sum(1 for r in observed_records if r["predicted_binary"] == "benign")
        pred_mal = len(observed_records) - pred_benign
        bin_correct = sum(
            1 for r in observed_records if r["predicted_binary"] == r["ground_truth_binary"]
        )

        return {
            "simulation_notice": (
                "LOCAL LOOPBACK SOCKET SIMULATION: Traffic is generated and observed over local TCP "
                f"loopback sockets ({bind_host}:{actual_port}) with application-framed IoT flow telemetry "
                "reassembled via FlowAssembler and classified by EdgeIDS (ONNX Runtime). This is not "
                "passive promiscuous NIC packet capture (libpcap/Npcap/eBPF)."
            ),
            "capture_architecture_verification": {
                "supports_live_nic_promiscuous_capture": False,
                "supports_pcap_file_replay": True,
                "supports_tcp_socket_payload_ingestion": True,
            },
            "config": {
                "data_source": data_source,
                "n_flows": int(n_flows),
                "attack_ratio": float(attack_ratio),
                "random_seed": random_seed,
                "host": bind_host,
                "actual_port": actual_port,
                "attack_profiles": list(attack_profiles) if attack_profiles else list(SIMULATION_ATTACK_PROFILES),
            },
            "total_flows_requested": int(n_flows),
            "total_flows_sent": len(scenarios),
            "total_flows_observed": len(observed_records),
            "benign_flows_sent": benign_sent,
            "malicious_flows_sent": mal_sent,
            "predicted_benign_count": pred_benign,
            "predicted_malicious_count": pred_mal,
            "alerts_logged_count": len(logged_alerts),
            "logged_alerts": logged_alerts,
            "audit_log_path": str(effective_audit_path),
            "resources_cleaned_up": bool(
                server_sock_closed and (listener_thread_ref is None or not listener_thread_ref.is_alive())
            ),
            "measured_metrics": {
                "binary_accuracy": round(bin_correct / n_obs, 4),
                "exact_class_accuracy": round(exact_class_matches / n_obs, 4),
                "exact_class_matches": exact_class_matches,
                "false_positives": false_positives,
                "missed_detections": missed_detections,
                "mean_inference_latency_ms": round(float(np.mean(latencies_ms)), 3) if latencies_ms else 0.0,
            },
            "observed_flows": observed_records,
        }
    finally:
        if cleanup_temp_audit and effective_audit_path.exists():
            try:
                effective_audit_path.unlink()
            except OSError:
                pass


def validate_pcap_file(pcap_path: str | Path) -> Dict[str, Any]:
    """
    Validate a binary `.pcap` file against the standard libpcap v2.4 format (`0xA1B2C3D4`,
    linktype=1 Ethernet) and verify that every packet record has valid header lengths
    (`incl_len == orig_len == len(frame)`), valid IPv4 header total length, and
    monotonically non-decreasing timestamps.
    """
    p = Path(pcap_path)
    if not p.exists():
        return {"valid": False, "error": f"PCAP file does not exist: {p}"}
    size_bytes = p.stat().st_size
    if size_bytes < 24:
        return {"valid": False, "error": f"PCAP file smaller than 24-byte global header ({size_bytes} bytes)"}

    pkt_count = 0
    total_frame_bytes = 0
    prev_ts_ms: Optional[float] = None
    monotonic_timestamps = True
    ipv4_valid = True
    length_match_valid = True

    with p.open("rb") as f:
        ghdr = f.read(24)
        magic, v_maj, v_min, thiszone, sigfigs, snaplen, linktype = struct.unpack("<IHHiIII", ghdr)
        endian = "<"
        if magic == PCAPReader.PCAP_MAGIC_SWAPPED_ENDIAN:
            endian = ">"
            magic, v_maj, v_min, thiszone, sigfigs, snaplen, linktype = struct.unpack(">IHHiIII", ghdr)
        elif magic != PCAPReader.PCAP_MAGIC_SAME_ENDIAN:
            return {"valid": False, "error": f"Invalid PCAP magic: {hex(magic)}"}

        while True:
            phdr = f.read(16)
            if not phdr:
                break
            if len(phdr) < 16:
                return {"valid": False, "error": "Truncated 16-byte packet header at EOF"}
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f"{endian}IIII", phdr)
            frame = f.read(incl_len)
            if len(frame) != incl_len or incl_len != orig_len:
                length_match_valid = False
                break
            ts_ms = (ts_sec * 1000.0) + (ts_usec / 1000.0)
            if prev_ts_ms is not None and ts_ms + 1e-6 < prev_ts_ms:
                monotonic_timestamps = False
            prev_ts_ms = ts_ms

            if len(frame) < 34:
                ipv4_valid = False
                break
            eth_type = struct.unpack("!H", frame[12:14])[0]
            if eth_type != 0x0800:
                ipv4_valid = False
                break
            ip_total_len = struct.unpack("!H", frame[16:18])[0]
            if ip_total_len != len(frame) - 14:
                ipv4_valid = False
                break

            pkt_count += 1
            total_frame_bytes += len(frame)

    is_valid = (
        v_maj == 2
        and v_min == 4
        and linktype == 1
        and pkt_count > 0
        and length_match_valid
        and monotonic_timestamps
        and ipv4_valid
    )
    return {
        "valid": is_valid,
        "pcap_path": str(p),
        "file_size_bytes": size_bytes,
        "magic_hex": hex(magic),
        "version": f"{v_maj}.{v_min}",
        "snaplen": snaplen,
        "linktype": linktype,
        "packet_count": pkt_count,
        "total_frame_bytes": total_frame_bytes,
        "length_headers_match_frames": length_match_valid,
        "monotonic_timestamps": monotonic_timestamps,
        "ipv4_headers_valid": ipv4_valid,
    }


def _build_deterministic_l7_payload(
    prefix: bytes,
    target_len: int,
    target_entropy: float,
    flow_index: int,
    pkt_idx: int,
) -> bytes:
    """
    Build a deterministic L7 packet payload starting with `prefix` and padded to `target_len`
    using a deterministic byte alphabet sized from `target_entropy`.
    """
    final_len = max(int(target_len), len(prefix), 1)
    pad_len = final_len - len(prefix)
    if pad_len <= 0:
        return prefix[:final_len]

    alphabet_size = max(1, min(256, int(round(2.0 ** max(min(float(target_entropy), 8.0), 0.0)))))
    stride = 73  # coprime with 256
    offset = (flow_index * 31 + pkt_idx * 17) & 0xFF
    alphabet = bytes(((offset + i * stride) & 0xFF) for i in range(alphabet_size))
    repeats = (pad_len // alphabet_size) + 1
    body = (alphabet * repeats)[:pad_len]
    return prefix + body


def export_simulation_to_pcap(
    sim_result_or_flows: Any,
    pcap_path: str | Path,
    base_epoch_sec: int = 1700010000,
    inter_flow_gap_sec: float = 2.0,
) -> Dict[str, Any]:
    """
    Stage 3 — Export controlled simulation flows into a valid standard libpcap v2.4 file
    (`linktype=1` Ethernet + IPv4 + TCP/UDP + L7 payload frames) readable by `PCAPReader`.

    Documented Synthesized Packet Fields:
    - Ethernet L2 header (`14` bytes): fixed synthetic local MAC addresses
      (`00:11:22:33:44:55 -> 66:77:88:99:aa:bb`) and EtherType `0x0800` (IPv4), since
      loopback application sockets do not expose physical NIC L2 Ethernet headers.
    - IPv4 L3 header (`20` bytes): version/IHL `0x45`, TTL `64`, protocol `6` (TCP) for
      `mqtt`/`http`/`other` or `17` (UDP) for `coap`/`6lowpan`, and exact `total_ip_len`.
    - L4 header (`20` bytes TCP or `8` bytes UDP): flow `src_port`, `dst_port`, and `tcp_flags`
      (for TCP frames; UDP frames have no TCP flags field in RFC 768).
    - Timestamps: deterministic microsecond-resolution timestamps starting at `base_epoch_sec`,
      interpolated across each flow's `duration_ms`, with `inter_flow_gap_sec = 2.0s`
      (`> FlowAssembler.idle_timeout_ms = 1.0s`) between consecutive flows.
    """
    out_path = Path(pcap_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(sim_result_or_flows, dict) and "observed_flows" in sim_result_or_flows:
        raw_flows = sim_result_or_flows["observed_flows"]
    elif isinstance(sim_result_or_flows, list):
        raw_flows = sim_result_or_flows
    else:
        raise ValueError("sim_result_or_flows must be a simulation result dict or list of flow dicts.")

    global_hdr = struct.pack("<IHHiIII", PCAPReader.PCAP_MAGIC_SAME_ENDIAN, 2, 4, 0, 0, 65535, 1)
    cur_ts_ms = float(base_epoch_sec) * 1000.0
    gap_ms = max(float(inter_flow_gap_sec) * 1000.0, 1200.0)

    total_packets_written = 0
    total_wire_bytes_written = 0
    per_flow_export_meta: List[Dict[str, Any]] = []

    with out_path.open("wb") as f:
        f.write(global_hdr)
        for idx, item in enumerate(raw_flows):
            fl = item.get("reassembled_flow", item) if isinstance(item, dict) else item
            flow_idx = int(item.get("flow_index", idx + 1))
            scen_id = str(item.get("scenario_id", f"flow_{flow_idx:03d}"))
            is_real_iot23 = str(item.get("data_source", fl.get("data_source", ""))) == "real_iot23"
            proto_str = str(fl.get("protocol", item.get("protocol", "other"))).lower()
            transport_p = str(item.get("transport_proto", fl.get("transport_proto", ""))).lower()
            src_ip = str(fl.get("src_ip", item.get("src_ip", "127.0.0.1")))
            dst_ip = str(fl.get("dst_ip", item.get("dst_ip", "127.0.0.1")))
            src_port = int(fl.get("src_port", item.get("src_port", 40000)))
            dst_port = int(fl.get("dst_port", item.get("dst_port", 80)))
            pkt_count = max(int(fl.get("packet_count", item.get("packet_count", 1))), 1)
            target_byte_count = max(int(fl.get("byte_count", item.get("byte_count", 0))), 0)
            dur_ms = max(float(fl.get("duration_ms", item.get("duration_ms", 0.0))), 0.0)
            tcp_flags = int(fl.get("tcp_flags", item.get("tcp_flags", 0)))
            mqtt_qos = int(fl.get("mqtt_qos", item.get("mqtt_qos", 0)))
            mqtt_payload_bytes = int(fl.get("mqtt_payload_bytes", item.get("mqtt_payload_bytes", 0)))
            coap_code = int(fl.get("coap_code", item.get("coap_code", 0)))
            http_status = int(fl.get("http_status", item.get("http_status", 0)))
            sixlowpan_dispatch = int(fl.get("sixlowpan_dispatch", item.get("sixlowpan_dispatch", 0)))
            target_entropy = float(fl.get("payload_entropy", item.get("payload_entropy", 0.0)))

            if transport_p == "udp" or proto_str in ("coap", "6lowpan"):
                proto_num = 17
            else:
                proto_num = 6
            hdr_len = 42 if proto_num == 17 else 54

            if is_real_iot23:
                # Real IoT-23 conn.log.labeled contains Zeek flow summaries without L7 payloads.
                # Synthesize zero-filled L7 payload bytes (or 0-byte L7 for header-only SYN scans)
                # so payload_entropy evaluates to 0.0 on replay just like map_zeek_to_apafc14().
                l7_prefix = b""
                total_hdr_bytes = pkt_count * hdr_len
                total_l7_budget = max(target_byte_count - total_hdr_bytes, 0)
                pkt0_l7_len = total_l7_budget if pkt_count == 1 else (total_l7_budget // pkt_count)
                rem_l7_after_0 = max(total_l7_budget - pkt0_l7_len, 0)
            else:
                if proto_str == "mqtt":
                    rem_len_byte = min(max(mqtt_payload_bytes - 2, 0), 127)
                    l7_prefix = bytes([0x30 | ((mqtt_qos & 0x03) << 1), rem_len_byte])
                elif proto_str == "coap":
                    l7_prefix = bytes([0x40, coap_code & 0xFF, 0x12, flow_idx & 0xFF])
                elif proto_str == "http":
                    st_code = http_status if http_status > 0 else 200
                    l7_prefix = f"HTTP/1.1 {st_code} OK\r\n\r\n".encode("ascii")
                elif proto_str == "6lowpan":
                    disp = sixlowpan_dispatch if sixlowpan_dispatch > 0 else 0x41
                    l7_prefix = bytes([disp & 0xFF, 0x06, 0x40])
                else:
                    l7_prefix = b"\x00\x02"

                total_hdr_bytes = pkt_count * hdr_len
                total_l7_budget = max(target_byte_count - total_hdr_bytes, pkt_count * len(l7_prefix))

                if proto_str == "mqtt" and mqtt_payload_bytes > 0:
                    pkt0_l7_len = mqtt_payload_bytes
                elif pkt_count == 1:
                    pkt0_l7_len = max(total_l7_budget, len(l7_prefix))
                else:
                    pkt0_l7_len = max(total_l7_budget // pkt_count, len(l7_prefix))

                rem_l7_after_0 = max(total_l7_budget - pkt0_l7_len, 0)

            flow_wire_bytes = 0

            for k in range(pkt_count):
                if pkt_count == 1:
                    pkt_ts_ms = cur_ts_ms
                    l7_len = pkt0_l7_len
                else:
                    pkt_ts_ms = cur_ts_ms + (dur_ms * float(k)) / float(pkt_count - 1)
                    if k == 0:
                        l7_len = pkt0_l7_len
                    else:
                        base_k, rem_k = divmod(rem_l7_after_0, pkt_count - 1)
                        l7_len = base_k + (1 if (k - 1) < rem_k else 0)
                        if not is_real_iot23:
                            l7_len = max(l7_len, 1)
                            if proto_str == "mqtt" and mqtt_payload_bytes > 0:
                                l7_len = min(l7_len, mqtt_payload_bytes)

                if is_real_iot23:
                    l7_bytes = b"\x00" * max(int(l7_len), 0)
                else:
                    pkt_prefix = l7_prefix if k == 0 else l7_prefix[: min(len(l7_prefix), l7_len)]
                    l7_bytes = _build_deterministic_l7_payload(
                        prefix=pkt_prefix,
                        target_len=l7_len,
                        target_entropy=target_entropy,
                        flow_index=flow_idx,
                        pkt_idx=k,
                    )
                frame = _build_ipv4_packet(
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    proto_num=proto_num,
                    payload=l7_bytes,
                    tcp_flags=tcp_flags if proto_num == 6 else 0,
                )
                ts_sec = int(pkt_ts_ms // 1000.0)
                ts_usec = int(round((pkt_ts_ms - (ts_sec * 1000.0)) * 1000.0))
                if ts_usec >= 1_000_000:
                    ts_sec += ts_usec // 1_000_000
                    ts_usec = ts_usec % 1_000_000

                pkt_hdr = struct.pack("<IIII", ts_sec, ts_usec, len(frame), len(frame))
                f.write(pkt_hdr)
                f.write(frame)

                total_packets_written += 1
                total_wire_bytes_written += len(frame)
                flow_wire_bytes += len(frame)

            per_flow_export_meta.append(
                {
                    "flow_index": flow_idx,
                    "scenario_id": scen_id,
                    "data_source": "real_iot23" if is_real_iot23 else "synthetic_profiles",
                    "protocol": proto_str,
                    "transport_proto_num": proto_num,
                    "src_ip": src_ip,
                    "src_port": src_port,
                    "dst_ip": dst_ip,
                    "dst_port": dst_port,
                    "packets_written": pkt_count,
                    "target_byte_count": target_byte_count,
                    "wire_byte_count": flow_wire_bytes,
                    "start_ts_ms": round(cur_ts_ms, 3),
                    "end_ts_ms": round(cur_ts_ms + dur_ms, 3),
                    "iot23_provenance": item.get("iot23_provenance") if isinstance(item, dict) else None,
                }
            )
            cur_ts_ms += dur_ms + gap_ms

    validation = validate_pcap_file(out_path)
    return {
        "pcap_path": str(out_path),
        "flows_exported": len(per_flow_export_meta),
        "total_packets_written": total_packets_written,
        "total_wire_bytes_written": total_wire_bytes_written,
        "synthesized_fields_documentation": {
            "l2_ethernet_header": "14-byte Ethernet II header (00:11:22:33:44:55 -> 66:77:88:99:aa:bb, EtherType 0x0800 IPv4) synthesized because Zeek conn.log.labeled and loopback sockets do not store L2 Ethernet headers.",
            "l3_ipv4_header": "20-byte IPv4 header with loopback endpoints (127.0.0.1 -> 127.0.0.1; original Zeek IPs preserved in iot23_provenance), exact total_ip_len, and protocol 6 (TCP) or 17 (UDP).",
            "l4_transport_header": "20-byte TCP header (with tcp_flags reconstructed from Zeek history/conn_state) or 8-byte UDP header with real IoT-23 id.orig_p and id.resp_p ports.",
            "l7_payload_bytes": "Zero-filled L7 payload bytes (b'\\x00' * l7_len, or 0-byte payload for TCP SYN/header-only packets) synthesized when exporting Zeek conn.log.labeled rows because Zeek connection summaries do not capture raw packet payloads.",
            "timestamps": f"Deterministic microsecond timestamps starting at epoch {base_epoch_sec}, interpolated across each flow's real Zeek duration_ms with {inter_flow_gap_sec}s inter-flow gaps (> FlowAssembler 1.0s idle timeout).",
        },
        "pcap_validation": validation,
        "per_flow_export": per_flow_export_meta,
    }


def run_simulation_pcap_export_and_replay_comparison(
    n_flows: int = 12,
    attack_ratio: float = 0.5,
    random_seed: Optional[int] = 2026,
    attack_profiles: Optional[List[str]] = None,
    pcap_output_path: Optional[Path] = None,
    audit_log_path: Optional[Path] = None,
    cleanup_files: bool = False,
    data_source: str = "real_iot23",
) -> Dict[str, Any]:
    """
    Stage 3 — End-to-end Live Loopback Simulation -> Valid PCAP Export -> PCAPReader/FlowAssembler
    Replay -> Per-Flow Feature Vector & EdgeIDS Prediction Parity Comparison.

    Defaults to `data_source="real_iot23"` (streaming real IoT-23 connection records via
    `src/iot23_loader.py` and preserving Zeek provenance alongside ground-truth labels vs.
    untouched `edge_ids.onnx` predictions).
    """
    import tempfile
    import os

    auto_pcap = pcap_output_path is None
    if pcap_output_path is None:
        fd, tmp_pcap = tempfile.mkstemp(prefix="iot_ids_stage3_sim_", suffix=".pcap")
        os.close(fd)
        effective_pcap_path = Path(tmp_pcap)
    else:
        effective_pcap_path = Path(pcap_output_path)
        effective_pcap_path.parent.mkdir(parents=True, exist_ok=True)

    fd_ra, tmp_replay_audit = tempfile.mkstemp(prefix="iot_ids_stage3_replay_alerts_", suffix=".jsonl")
    os.close(fd_ra)
    replay_audit_path = Path(tmp_replay_audit)

    try:
        live_res = run_randomized_local_attack_simulation(
            n_flows=n_flows,
            attack_ratio=attack_ratio,
            random_seed=random_seed,
            attack_profiles=attack_profiles,
            host="127.0.0.1",
            port=0,
            audit_log_path=audit_log_path,
            persist_to_core_alerts=False,
            data_source=data_source,
        )

        export_res = export_simulation_to_pcap(
            sim_result_or_flows=live_res,
            pcap_path=effective_pcap_path,
        )

        replayed_flows = replay_pcap_to_flows(effective_pcap_path)
        edge_ids = load_core_edge_ids(top_k=5, audit_path=replay_audit_path)
        extractor = APAFCFeatureExtractor()

        live_flows = live_res["observed_flows"]
        flow_count_match = len(live_flows) == len(replayed_flows)
        paired_count = min(len(live_flows), len(replayed_flows))

        flow_comparisons: List[Dict[str, Any]] = []
        endpoint_matches = 0
        class_agreements = 0
        binary_agreements = 0
        conf_diffs: List[float] = []
        feature_diff_matrix: List[List[float]] = []

        for idx in range(paired_count):
            lf_entry = live_flows[idx]
            lf_dict = lf_entry["reassembled_flow"]
            rf_dict = replayed_flows[idx]

            live_vec = [float(x) for x in extractor.extract(lf_dict)]
            replay_vec = [float(x) for x in extractor.extract(rf_dict)]
            replay_pred = edge_ids.classify(rf_dict)

            live_cls = str(lf_entry["predicted_class"])
            replay_cls = str(replay_pred["class"])
            live_bin = str(lf_entry["predicted_binary"])
            replay_bin = "benign" if replay_cls == "benign" else "malicious"
            live_conf = float(lf_entry["confidence"])
            replay_conf = round(float(replay_pred["confidence"]), 4)
            conf_diff = abs(live_conf - replay_conf)
            conf_diffs.append(conf_diff)

            ep_match = (
                str(lf_dict["src_ip"]) == str(rf_dict["src_ip"])
                and int(lf_dict["src_port"]) == int(rf_dict["src_port"])
                and str(lf_dict["dst_ip"]) == str(rf_dict["dst_ip"])
                and int(lf_dict["dst_port"]) == int(rf_dict["dst_port"])
            )
            if ep_match:
                endpoint_matches += 1
            if live_cls == replay_cls:
                class_agreements += 1
            if live_bin == replay_bin:
                binary_agreements += 1

            feat_diffs: Dict[str, float] = {}
            diff_Row: List[float] = []
            differing_features: List[str] = []
            for f_idx, f_name in enumerate(FEATURE_NAMES):
                d_val = abs(live_vec[f_idx] - replay_vec[f_idx])
                feat_diffs[f_name] = round(d_val, 6)
                diff_Row.append(d_val)
                if d_val > 1e-3:
                    differing_features.append(f_name)
            feature_diff_matrix.append(diff_Row)

            reasons: List[str] = []
            if "payload_entropy" in differing_features:
                reasons.append(
                    f"payload_entropy differs ({live_vec[13]:.4f} live vs {replay_vec[13]:.4f} replay) because "
                    "live simulation uses the profile's target numeric entropy scalar while PCAP replay computes "
                    "exact Shannon entropy over the serialized L7 wire payload bytes."
                )
            if "byte_count" in differing_features or "mean_packet_bytes" in differing_features:
                reasons.append(
                    f"byte_count/mean_packet_bytes differ ({live_vec[3]:.1f}B live vs {replay_vec[3]:.1f}B replay) "
                    "because valid PCAP frames include mandatory 54B TCP or 42B UDP + IPv4 + Ethernet wire headers "
                    "and non-empty L7 headers on every packet."
                )
            if "tcp_flags" in differing_features:
                reasons.append(
                    f"tcp_flags differs ({live_vec[7]:.0f} live vs {replay_vec[7]:.0f} replay) because UDP frames "
                    "(coap/6lowpan, IPv4 proto 17) have no TCP flags field in RFC 768 headers."
                )
            if "protocol_id" in differing_features or "mqtt_qos" in differing_features or "mqtt_payload_bytes" in differing_features:
                reasons.append(
                    f"protocol_id/L7 fields differ ({lf_dict.get('protocol')}->{rf_dict.get('protocol')}) because "
                    "PCAPReader._identify_tcp_proto maps TCP protocol strictly by well-known port numbers "
                    "(1883/8883->mqtt, 80/8080/443->http, other TCP ports->tcp/0.0)."
                )
            if "http_status" in differing_features:
                reasons.append(
                    f"http_status differs ({live_vec[11]:.0f} live vs {replay_vec[11]:.0f} replay) because "
                    "Zeek conn.log.labeled does not record HTTP status codes (0 in map_zeek_to_apafc14), whereas "
                    "PCAPReader._extract_http_status defaults non-empty payloads on TCP ports 80/8080/443 to 200."
                )

            flow_comparisons.append(
                {
                    "flow_index": int(lf_entry["flow_index"]),
                    "scenario_id": str(lf_entry["scenario_id"]),
                    "data_source": str(lf_entry.get("data_source", data_source)),
                    "ground_truth_category": str(lf_entry["ground_truth_category"]),
                    "ground_truth_binary": str(lf_entry["ground_truth_binary"]),
                    "iot23_raw_label": lf_entry.get("iot23_raw_label"),
                    "iot23_detailed_label": lf_entry.get("iot23_detailed_label"),
                    "iot23_taxonomy_family": lf_entry.get("iot23_taxonomy_family"),
                    "iot23_provenance": lf_entry.get("iot23_provenance"),
                    "endpoint_match": ep_match,
                    "live_endpoint": f"{lf_dict['src_ip']}:{lf_dict['src_port']}->{lf_dict['dst_ip']}:{lf_dict['dst_port']}",
                    "replay_endpoint": f"{rf_dict['src_ip']}:{rf_dict['src_port']}->{rf_dict['dst_ip']}:{rf_dict['dst_port']}",
                    "live_protocol": str(lf_dict.get("protocol")),
                    "replay_protocol": str(rf_dict.get("protocol")),
                    "live_predicted_class": live_cls,
                    "replay_predicted_class": replay_cls,
                    "class_agreement": live_cls == replay_cls,
                    "live_predicted_binary": live_bin,
                    "replay_predicted_binary": replay_bin,
                    "binary_agreement": live_bin == replay_bin,
                    "live_confidence": live_conf,
                    "replay_confidence": replay_conf,
                    "confidence_abs_diff": round(conf_diff, 4),
                    "live_features": {FEATURE_NAMES[i]: round(live_vec[i], 4) for i in range(14)},
                    "replay_features": {FEATURE_NAMES[i]: round(replay_vec[i], 4) for i in range(14)},
                    "feature_abs_diffs": feat_diffs,
                    "differing_features": differing_features,
                    "difference_explanations": reasons,
                }
            )

        arr_diffs = np.asarray(feature_diff_matrix, dtype=float) if feature_diff_matrix else np.zeros((1, 14))
        per_feature_summary = {
            FEATURE_NAMES[i]: {
                "mean_abs_diff": round(float(np.mean(arr_diffs[:, i])), 6),
                "max_abs_diff": round(float(np.max(arr_diffs[:, i])), 6),
                "exact_zero_diff_flows": int(np.sum(arr_diffs[:, i] <= 1e-5)),
            }
            for i in range(14)
        }

        denom = max(paired_count, 1)
        return {
            "data_source_and_limitations_notice": (
                "SIMULATED LOOPBACK TO PCAP REPLAY COMPARISON: Flows originate from the controlled local "
                "loopback simulator (127.0.0.1), are serialized into standard libpcap v2.4 Ethernet/IPv4/TCP-UDP "
                "frames with synthesized L2 MAC headers, and are replayed through PCAPReader -> FlowAssembler -> "
                "EdgeIDS. Results measure pipeline consistency between live socket telemetry and wire-level PCAP "
                "parsing, not real-world passive NIC capture accuracy."
            ),
            "matching_strategy": (
                "Deterministic 1-to-1 chronological sequence matching (flow_index 1..N) with 2.0s inter-flow "
                "timestamp gaps (> FlowAssembler 1.0s idle timeout) and 5-tuple initiator endpoint verification."
            ),
            "pcap_export": export_res,
            "live_flow_count": len(live_flows),
            "replay_flow_count": len(replayed_flows),
            "flow_count_match": flow_count_match,
            "endpoint_direction_match_count": endpoint_matches,
            "endpoint_direction_match_rate": round(endpoint_matches / denom, 4),
            "exact_class_agreement_count": class_agreements,
            "exact_class_agreement_rate": round(class_agreements / denom, 4),
            "binary_agreement_count": binary_agreements,
            "binary_agreement_rate": round(binary_agreements / denom, 4),
            "mean_confidence_abs_diff": round(float(np.mean(conf_diffs)), 4) if conf_diffs else 0.0,
            "max_confidence_abs_diff": round(float(np.max(conf_diffs)), 4) if conf_diffs else 0.0,
            "per_feature_diff_summary": per_feature_summary,
            "flow_comparisons": flow_comparisons,
            "live_simulation_summary": {
                "config": live_res["config"],
                "measured_metrics": live_res["measured_metrics"],
                "alerts_logged_count": live_res["alerts_logged_count"],
                "resources_cleaned_up": live_res["resources_cleaned_up"],
            },
        }
    finally:
        if replay_audit_path.exists():
            try:
                replay_audit_path.unlink()
            except OSError:
                pass
        if (cleanup_files or auto_pcap) and effective_pcap_path.exists():
            try:
                effective_pcap_path.unlink()
            except OSError:
                pass



