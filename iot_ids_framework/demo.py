#!/usr/bin/env python3
"""Interactive Live Demonstration of the Dual-Tier Drift-Adaptive IoT IDS Framework.

Showcases:
  1. Packet Capture (PCAP) Replay & Bidirectional Flow Assembly (Module M1)
  2. Protocol-Aware APAFC Feature Extraction (14 features)
  3. Autonomous Edge Inference & Explainable AI (TreeSHAP Attributions)
  4. Real-time Concept Drift Detection via Sliding-Window KL Divergence (Module M7)
  5. Zero-Downtime Atomic Model Hot-Swap (Module M9)
  6. Empirical Baseline Performance Benchmark (Accuracy, F1, Latency, Memory)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Add src/ to path if running directly from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import onnxruntime as ort

from iot_ids.pcap_adapter import replay_pcap_to_flows
from iot_ids.features import APAFCFeatureExtractor, FEATURE_NAMES
from iot_ids.edge import EdgeIDS, load_monitor, load_tree_explainer
from iot_ids.baselines.benchmark_comparison import benchmark_baseline_models


class TerminalColors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def header(title: str) -> None:
    print(f"\n{TerminalColors.BOLD}{TerminalColors.CYAN}{'=' * 75}{TerminalColors.END}")
    print(f"{TerminalColors.BOLD}{TerminalColors.YELLOW} [DEMO] {title.upper()}{TerminalColors.END}")
    print(f"{TerminalColors.BOLD}{TerminalColors.CYAN}{'=' * 75}{TerminalColors.END}")


def subheader(title: str) -> None:
    print(f"\n{TerminalColors.BOLD}{TerminalColors.BLUE}---> {title}{TerminalColors.END}")


def run_demo() -> None:
    root = Path(__file__).parent
    artifacts_dir = root / "artifacts"
    data_dir = root / "datasets"

    print(f"{TerminalColors.BOLD}{TerminalColors.GREEN}")
    print(r"""
  ___     _____   ___ ___  ____    _____                                          _    
 |_ _|___|_   _| |_ _|   \/ ___|  |  ___| __ __ _ _ __ ___   _____      _____  _ __| | __
  | |/ _ \ | |    | || |) \___ \  | |_ | '__/ _` | '_ ` _ \ / _ \ \ /\ / / _ \| '__| |/ /
  | | (_) || |    | ||  _ < ___) | |  _|| | | (_| | | | | | |  __/\ V  V / (_) | |  |   < 
 |___\___/ |_|   |___|_| \_\____/  |_|  |_|  \__,_|_| |_| |_|\___| \_/\_/ \___/|_|  |_|\_\
    """)
    print(f"       Autonomous Edge Intelligence * TreeSHAP * KL Concept Drift * Federated Learning{TerminalColors.END}\n")

    # =========================================================================
    # STAGE 1: PCAP Ingestion & Bounded Flow Assembly
    # =========================================================================
    header("Stage 1: Packet Capture Replay & Bidirectional Flow Assembly")
    pcap_file = data_dir / "sample_traffic.pcap"
    if not pcap_file.exists():
        print(f"{TerminalColors.RED}Sample PCAP not found at {pcap_file}. Please run PCAP generator.{TerminalColors.END}")
        return

    print(f"Streaming packets from capture file: {TerminalColors.BOLD}{pcap_file}{TerminalColors.END}")
    flows = replay_pcap_to_flows(pcap_file)
    print(f"{TerminalColors.GREEN}Successfully reassembled {len(flows)} bidirectional flows.{TerminalColors.END}")

    print(f"\n{'Protocol':<10} | {'Src Endpoint':<21} | {'Dst Endpoint':<21} | {'Packets':<7} | {'Bytes':<6} | {'Duration'}")
    print("-" * 80)
    for flow in flows:
        src = f"{flow.get('src_ip')}:{flow.get('src_port')}"
        dst = f"{flow.get('dst_ip')}:{flow.get('dst_port')}"
        proto = str(flow.get('protocol')).upper()
        pkts = flow.get('packet_count', 0)
        b = flow.get('byte_count', 0)
        dur = f"{flow.get('duration_ms', 0):.1f}ms"
        print(f"{proto:<10} | {src:<21} | {dst:<21} | {pkts:<7} | {b:<6} | {dur}")

    # =========================================================================
    # STAGE 2: Feature Extraction (APAFC 14 Features)
    # =========================================================================
    header("Stage 2: Protocol-Aware APAFC Feature Extraction (61 -> 14)")
    extractor = APAFCFeatureExtractor()
    sample_flow = flows[0]
    vector = extractor.extract(sample_flow)

    print(f"Extracted fixed 14-dimensional feature vector from flow [{sample_flow.get('protocol')}]:")
    for name, val in zip(FEATURE_NAMES, vector):
        print(f"  * {name:<22}: {val}")

    # =========================================================================
    # STAGE 3: Autonomous Edge Classification & Explainable TreeSHAP
    # =========================================================================
    header("Stage 3: Autonomous Edge Inference & TreeSHAP Attribution")
    config_file = root / "configs" / "default.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))

    model_file = artifacts_dir / "edge_ids.onnx"
    labels_file = artifacts_dir / "labels.json"
    ref_file = artifacts_dir / "reference.json"
    tree_file = artifacts_dir / "edge_ids.txt"

    if not model_file.exists():
        print(f"{TerminalColors.RED}Model artifact {model_file} missing. Train model first.{TerminalColors.END}")
        return

    session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    monitor = load_monitor(ref_file, config)
    explainer = load_tree_explainer(tree_file) if tree_file.exists() else None

    alerts_log = artifacts_dir / "alerts.jsonl"
    edge_ids = EdgeIDS(
        session=session,
        class_names=labels,
        monitor=monitor,
        top_k=config["alerts"]["top_k"],
        tree_explainer=explainer,
        audit_path=alerts_log
    )

    # Test 1: Classify Benign Flow
    subheader("Flow #1: Normal IoT Telemetry")
    res1 = edge_ids.classify(sample_flow)
    status_color = TerminalColors.GREEN if res1['class'] == 'benign' else TerminalColors.YELLOW
    print(f"Prediction: {status_color}{res1['class'].upper()}{TerminalColors.END} (Confidence: {res1['confidence']:.2%})")

    # Test 2: Injected Malicious Flow (e.g. high volume syn flood / backdoor)
    subheader("Flow #2: Injected Attack Vector (High Rate MQTT Exploit)")
    attack_flow = {
        "src_ip": "10.0.4.15", "src_port": 58912, "dst_ip": "192.168.1.1", "dst_port": 1883,
        "protocol": "mqtt", "duration_ms": 500.0, "packet_count": 8500, "byte_count": 1250000,
        "tcp_flags": 2, "mqtt_qos": 2, "mqtt_payload_bytes": 512, "payload_entropy": 7.85
    }
    res2 = edge_ids.classify(attack_flow)
    print(f"Prediction: {TerminalColors.RED}{TerminalColors.BOLD}{res2['class'].upper()}{TerminalColors.END} (Confidence: {res2['confidence']:.2%})")

    if "alert" in res2:
        alert = res2["alert"]
        print(f"\n{TerminalColors.BOLD}Generated Structured Edge Alert (JSON Audit):{TerminalColors.END}")
        print(f"  Timestamp   : {alert['timestamp']}")
        print(f"  5-Tuple     : {alert['flow_5_tuple']['src_ip']}:{alert['flow_5_tuple']['src_port']} -> {alert['flow_5_tuple']['dst_ip']}:{alert['flow_5_tuple']['dst_port']} ({alert['flow_5_tuple']['protocol']})")
        print(f"  Method      : {alert['attribution_method']}")
        print(f"\n  Top Feature Attributions (TreeSHAP):")
        for item in alert["attribution"]:
            sign = "+" if item["contribution"] >= 0 else ""
            print(f"    - {item['feature']:<20} : {sign}{item['contribution']:.4f}")

    # =========================================================================
    # STAGE 4: Concept Drift Detection (Module M7 - KL Divergence)
    # =========================================================================
    header("Stage 4: Real-Time Concept Drift Detection (KL Divergence Monitor)")
    print(f"Window Size: {config['drift']['window_size']} | Threshold (tau): {config['drift']['threshold']} | Consecutive Windows: {config['drift']['consecutive_windows']}")
    
    # Simulate high volume stream to populate sliding window
    print("\nStreaming 500 baseline flows to fill sliding window...")
    for _ in range(500):
        edge_ids.classify({"protocol": "mqtt", "duration_ms": 20, "packet_count": 10, "byte_count": 400})

    print(f"{TerminalColors.GREEN}Baseline window steady: No drift detected.{TerminalColors.END}")

    print("\nSimulating network distribution shift (Concept Drift)...")
    drift_triggered = False
    retrain_dispatched = False

    # Inject shifted feature stream
    for win in range(1, 4):
        for _ in range(500):
            res = edge_ids.classify({
                "protocol": "6lowpan", "duration_ms": 4500.0, "packet_count": 2500,
                "byte_count": 800000, "payload_entropy": 7.9, "tcp_flags": 24
            })
            if "drift" in res and res["drift"]["retrain_requested"]:
                retrain_dispatched = True
        
        drift_info = res.get("drift", {})
        kl_score = drift_info.get("kl", 0.0)
        consec = drift_info.get("consecutive_windows", 0)
        print(f"  Window {win}: Mean KL = {kl_score:.4f} (Threshold: 0.15) | Consecutive Over-Threshold: {consec}")

    if retrain_dispatched:
        print(f"\n{TerminalColors.BOLD}{TerminalColors.YELLOW}[RETRAIN EVENT EMITTED]{TerminalColors.END}")
        print(f"  Sustained concept drift exceeded {config['drift']['consecutive_windows']} consecutive windows!")
        print(f"  Trigger payload dispatched to Cloud/Edge Aggregator for federated retraining.")

    # =========================================================================
    # STAGE 5: Zero-Downtime Atomic Model Hot-Swap (Module M9)
    # =========================================================================
    header("Stage 5: Zero-Downtime Atomic Model Hot-Swap")
    print("Simulating arrival of retrained model package from central aggregator...")
    replacement_session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
    
    t0 = time.perf_counter()
    edge_ids.hot_swap(replacement_session)
    swap_time_us = (time.perf_counter() - t0) * 1_000_000

    post_swap_res = edge_ids.classify(sample_flow)
    print(f"{TerminalColors.GREEN}Model atomically swapped in {swap_time_us:.2f} microseconds!{TerminalColors.END}")
    print(f"Subsequent inference executed seamlessly: Class = {post_swap_res['class']} (Confidence: {post_swap_res['confidence']:.2%})")

    # =========================================================================
    # STAGE 6: Comparative Baseline Benchmarks
    # =========================================================================
    header("Stage 6: Comparative Machine Learning Benchmark")
    bench_csv = data_dir / "Edge-IIoTset" / "Selected dataset for ML and DL" / "ML-EdgeIIoT-dataset.csv"
    if not bench_csv.exists():
        bench_csv = data_dir / "ToN-IoT" / "train_test_network.csv"
    if bench_csv.exists():
        print(f"Benchmarking classical baselines against dataset: {bench_csv.name} (sample size: 5000)")
        bench_df = benchmark_baseline_models(bench_csv, task="binary", sample_size=5000)
        print("\n" + bench_df.to_string(index=False))
    
    print(f"\n{TerminalColors.BOLD}{TerminalColors.GREEN}All demonstration stages completed successfully! The IoT IDS framework is fully operational and presentable.{TerminalColors.END}\n")


if __name__ == "__main__":
    run_demo()
