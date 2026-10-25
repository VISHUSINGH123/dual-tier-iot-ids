#!/usr/bin/env python3
"""Realistic Network Attack Simulator & Local Port Server for IoT IDS Framework.

Features:
  1. Live Server Mode: Listens on a local port (default: 9999), ingests real socket-level packets,
     performs bidirectional flow assembly, calculates Shannon payload entropy and protocol fields,
     and executes real-time ONNX classification + TreeSHAP explainability.
  2. Attack Simulator Mode: Sends realistic network attack patterns over raw sockets:
     - Normal IoT Telemetry (MQTT publish, standard entropy)
     - DDoS SYN / Volume Flood (High-frequency bursts, large byte counts)
     - Port Scan / Probe Attack (Rapid transient single-packet connections)
     - Encrypted Backdoor Exfiltration (High-entropy pseudorandom encrypted payload)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import threading
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import onnxruntime as ort
from iot_ids.edge import EdgeIDS, load_monitor, load_tree_explainer
from iot_ids.ingest import FlowAssembler


class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def calc_shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [data.count(bytes([b])) for b in range(256)]
    length = len(data)
    return -sum((c / length) * math.log2(c / length) for c in counts if c > 0)


# =====================================================================
# 1. LIVE EDGE IDS SERVER (Listens on Local Port)
# =====================================================================
def run_ids_server(host: str = "127.0.0.1", port: int = 9999, ready_event: threading.Event | None = None) -> None:
    root = Path(__file__).parent
    artifacts_dir = root / "artifacts"
    config = json.loads((root / "configs" / "default.json").read_text(encoding="utf-8"))
    labels = json.loads((artifacts_dir / "labels.json").read_text(encoding="utf-8"))
    
    session = ort.InferenceSession(str(artifacts_dir / "edge_ids.onnx"), providers=["CPUExecutionProvider"])
    monitor = load_monitor(artifacts_dir / "reference.json", config)
    explainer = load_tree_explainer(artifacts_dir / "edge_ids.txt") if (artifacts_dir / "edge_ids.txt").exists() else None
    
    edge_ids = EdgeIDS(
        session=session,
        class_names=labels,
        monitor=monitor,
        top_k=5,
        tree_explainer=explainer,
        audit_path=artifacts_dir / "alerts.jsonl"
    )
    
    assembler = FlowAssembler(idle_timeout_ms=150)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(10)

    if ready_event:
        ready_event.set()

    print(f"{Colors.BOLD}{Colors.CYAN}{'='*75}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN} [*] IoT IDS Edge Engine running live on {host}:{port}{Colors.END}")
    print(f" [*] Ingesting raw network flows -> ONNX Runtime inference -> TreeSHAP alerts")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*75}{Colors.END}\n")

    def handle_client(conn: socket.socket, addr: tuple[str, int]) -> None:
        client_ip, client_port = addr

        try:
            total_payload = b""
            packet_count = 0
            
            while True:
                data = conn.recv(65536)
                if not data:
                    break
                now_ms = time.time() * 1000.0
                total_payload += data
                packet_count += 1
                
                # Protocol identification
                is_mqtt = data.startswith(b"\x10") or data.startswith(b"\x30") or client_port == 1883
                is_http = b"HTTP" in data or b"GET" in data or b"POST" in data
                protocol = "mqtt" if is_mqtt else ("http" if is_http else "tcp")
                
                # Extract telemetry fields
                mqtt_qos = float((data[0] >> 1) & 0x03) if (is_mqtt and len(data) >= 1) else 0.0
                mqtt_bytes = float(len(data)) if is_mqtt else 0.0
                entropy = calc_shannon_entropy(data)
                
                pkt_meta = {
                    "timestamp_ms": now_ms,
                    "src_ip": client_ip,
                    "src_port": client_port,
                    "dst_ip": host,
                    "dst_port": port,
                    "protocol": protocol,
                    "length": len(data),
                    "tcp_flags": 2 if len(data) == 0 else 24,
                    "mqtt_qos": mqtt_qos,
                    "mqtt_payload_bytes": mqtt_bytes,
                    "payload_entropy": entropy
                }
                assembler.submit(pkt_meta)

            # Client socket closed: evaluate flow
            time.sleep(0.05)
            completed_flows = assembler.expire(time.time() * 1000.0 + 300)
            for flow in completed_flows:
                # Ensure entropy is preserved
                if "payload_entropy" not in flow or flow["payload_entropy"] == 0:
                    flow["payload_entropy"] = calc_shannon_entropy(total_payload)
                    
                res = edge_ids.classify(flow)
                pred_class = res["class"]
                conf = res["confidence"]
                
                if pred_class == "benign":
                    print(f"{Colors.GREEN}[NORMAL TELEMETRY]{Colors.END} {client_ip}:{client_port} -> {flow['protocol'].upper()} | Class: {pred_class.upper()} ({conf:.1%})")
                else:
                    print(f"\n{Colors.RED}{Colors.BOLD}[SECURITY ALERT - ATTACK DETECTED]{Colors.END}")
                    print(f"  Source Endpoint : {client_ip}:{client_port} -> {flow['protocol'].upper()}")
                    print(f"  Classification  : {Colors.RED}{Colors.BOLD}{pred_class.upper()}{Colors.END} (Confidence: {conf:.1%})")
                    print(f"  Flow Telemetry  : {flow.get('packet_count', 1)} pkts | {flow.get('byte_count', 0)} bytes | Entropy: {flow.get('payload_entropy', 0):.2f}")
                    if "alert" in res:
                        print("  TreeSHAP Key Contributing Attributions:")
                        for attr in res["alert"]["attribution"][:4]:
                            sign = "+" if attr["contribution"] >= 0 else ""
                            print(f"    - {attr['feature']:<20}: {sign}{attr['contribution']:.4f}")
                    print("-" * 60)

        except Exception:
            pass
        finally:
            conn.close()

    try:
        while True:
            conn, addr = server.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[*] Shutting down IDS server.{Colors.END}")
        server.close()


# =====================================================================
# 2. REALISTIC MULTI-VECTOR ATTACK SIMULATOR
# =====================================================================
def run_attacks(host: str = "127.0.0.1", port: int = 9999) -> None:
    print(f"\n{Colors.BOLD}{Colors.YELLOW}Launching Realistic IoT Attack Simulation against {host}:{port}...{Colors.END}\n")
    time.sleep(1)

    # Sim 1: Benign Normal Traffic (Standard MQTT Publish)
    print(f"{Colors.CYAN}---> [Simulation 1/4] Normal IoT Sensor Telemetry (MQTT Publish)...{Colors.END}")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        # MQTT CONNECT
        s.sendall(b"\x10\x12\x00\x04MQTT\x04\x02\x00<\x00\x06sensor")
        time.sleep(0.05)
        # MQTT PUBLISH (temperature: 22.5C)
        s.sendall(b"\x30\x15\x00\x0btemp/living\x32\x32\x2e\x35\x43")
        s.close()
    except Exception as e:
        print(f"{Colors.RED}Connection failed: {e}{Colors.END}")
        return

    time.sleep(1.5)

    # Sim 2: DDoS / High-Volume SYN Flood simulation
    print(f"\n{Colors.CYAN}---> [Simulation 2/4] High-Rate DDoS Flood Attack (High Packet Burst)...{Colors.END}")
    for _ in range(6):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            # Send continuous bursts of heavy packet data over MQTT
            header = b"\x30\xfe\x07\x00\x09flood/ddos"
            payload = header + os.urandom(8192)
            for _ in range(10):
                s.sendall(payload)
                time.sleep(0.005)
            s.close()
        except Exception:
            pass
        time.sleep(0.02)

    time.sleep(1.5)

    # Sim 3: Rapid Port Scanning Behavior
    print(f"\n{Colors.CYAN}---> [Simulation 3/4] Rapid Network Port Scanning & Reconnaissance...{Colors.END}")
    for probe_port in (80, 8080, 1883, 5683, 22):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.1)
            s.connect((host, port))
            # Minimal probe
            s.send(b"\x00")
            s.close()
        except Exception:
            pass
        time.sleep(0.03)

    time.sleep(1.5)

    # Sim 4: Encrypted Backdoor Exfiltration
    print(f"\n{Colors.CYAN}---> [Simulation 4/4] Encrypted Backdoor C2 Exfiltration (High-Entropy Payload)...{Colors.END}")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        # High entropy pseudorandom data simulating encrypted command & control payload over MQTT
        c2_header = b"\x34\xff\x0f\x00\x0ac2/exfiltrate"
        encrypted_stream = c2_header + os.urandom(4096)
        s.sendall(encrypted_stream)
        time.sleep(0.05)
        s.close()
    except Exception:
        pass

    time.sleep(1.5)
    print(f"\n{Colors.GREEN}{Colors.BOLD}[*] Simulation completed.{Colors.END}\n")


# =====================================================================
# CLI ENTRYPOINT
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="IoT IDS Live Port Server & Realistic Attack Generator")
    parser.add_argument("mode", choices=["server", "attack", "both"], help="Run server, launch attacks, or run both concurrently")
    parser.add_argument("--host", default="127.0.0.1", help="Target host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9999, help="Target port (default 9999)")

    args = parser.parse_args()

    if args.mode == "server":
        run_ids_server(args.host, args.port)
    elif args.mode == "attack":
        run_attacks(args.host, args.port)
    elif args.mode == "both":
        ready = threading.Event()
        t = threading.Thread(target=run_ids_server, args=(args.host, args.port, ready), daemon=True)
        t.start()
        ready.wait(timeout=10.0)
        time.sleep(0.5)
        run_attacks(args.host, args.port)
        time.sleep(2.0)


if __name__ == "__main__":
    main()
