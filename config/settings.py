"""
Central configuration and constants for the Dual-Tier Edge-Cloud IoT IDS Prototype.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

for _d in (DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Default file paths
_edge_raw_candidate = DATA_DIR / "raw" / "archive" / "Edge-IIoTset dataset" / "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv"
REAL_EDGE_IIOTSET_CSV = (
    _edge_raw_candidate
    if _edge_raw_candidate.exists()
    else (
        BASE_DIR
        / "archive"
        / "Edge-IIoTset dataset"
        / "Selected dataset for ML and DL"
        / "ML-EdgeIIoT-dataset.csv"
    )
)
SYNTHETIC_DEMO_CSV = DATA_DIR / "synthetic_edge_iiotset_demo.csv"
DRIFT_DEMO_CSV = DATA_DIR / "synthetic_edge_iiotset_drifted.csv"
MODEL_ARTIFACT_PATH = MODELS_DIR / "lgbm_ids_bundle.joblib"
BASELINE_FULL_MODEL_PATH = MODELS_DIR / "lgbm_full_features_bundle.joblib"
FEATURE_CONFIG_PATH = MODELS_DIR / "apafc_feature_config.json"
DRIFT_REFERENCE_PATH = MODELS_DIR / "drift_reference_stats.json"
DRIFT_HISTORY_PATH = LOGS_DIR / "drift_window_history.json"
RETRAIN_LOG_PATH = LOGS_DIR / "retraining_history.json"
ALERT_LOG_PATH = LOGS_DIR / "alerts_log.jsonl"
CANONICAL_EVENTS_PATH = LOGS_DIR / "canonical_events.jsonl"
FEDERATED_ROUNDS_PATH = RESULTS_DIR / "federated_rounds_history.json"
EVALUATION_RESULTS_PATH = RESULTS_DIR / "latest_evaluation.json"

# Standard 15-class Edge-IIoTset taxonomy referenced in the report
EDGE_IIOTSET_15_CLASSES = [
    "Normal",
    "DDoS_UDP",
    "DDoS_ICMP",
    "DDoS_TCP",
    "DDoS_HTTP",
    "SQL_injection",
    "Uploading",
    "XSS",
    "Backdoor",
    "Password",
    "Port_Scanning",
    "Vulnerability_scanner",
    "Ransomware",
    "MITM",
    "Fingerprinting",
]

# Known candidate label columns across Edge-IIoTset, CICIoT2023, ToN-IoT, etc.
CANDIDATE_LABEL_COLUMNS = [
    "Attack_type",
    "attack_type",
    "Attack_label",
    "attack_label",
    "label",
    "Label",
    "class",
    "Class",
    "category",
    "Category",
    "type",
    "target",
]

# Columns that represent identifiers, raw addresses, timestamps, or secondary labels
# that must be excluded from input features to prevent target leakage or overfitting to IPs
LEAKAGE_AND_ID_COLUMNS = {
    "frame.time",
    "timestamp",
    "ts",
    "flow_id",
    "ip.src_host",
    "ip.dst_host",
    "src_ip",
    "dst_ip",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "http.request.uri.query",
    "mqtt.msg",
    "mqtt.topic",
    "tcp.options",
    "tcp.payload",
    "Attack_label",
    "attack_label",
    "Attack_type",
    "attack_type",
    "label",
    "Label",
    "class",
    "Class",
    "is_attack",
    "binary_label",
}

# Protocol-specific feature prefixes for distinguishing protocol-derived attributes
# from generic flow/packet statistics in APAFC
PROTOCOL_FEATURE_PREFIXES = {
    "MQTT": ("mqtt.", "mqtt_"),
    "CoAP": ("coap.", "coap_"),
    "HTTP": ("http.", "http_"),
    "6LoWPAN": ("sixlowpan.", "6lowpan_", "ipv6.", "icmpv6."),
    "DNS_MMODBUS": ("dns.", "DNS_", "mbtcp.", "modbus_"),
}

# Default APAFC & Drift settings from the BCSE497J report
DEFAULT_APAFC_K = 14
DEFAULT_DRIFT_TAU = 0.15
DEFAULT_DRIFT_CONSECUTIVE_WINDOWS = 3
DEFAULT_DRIFT_WINDOW_SIZE = 200
DEFAULT_RANDOM_SEED = 42
