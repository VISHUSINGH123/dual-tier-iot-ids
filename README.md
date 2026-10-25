# Dual-Tier Edge–Cloud IoT Intrusion Detection System (BCSE497J Project-I Prototype)

**Title:** *Lightweight, Drift-Adaptive and Privacy-Preserving Intrusion Detection in Heterogeneous IoT Networks: A Dual-Tier Edge–Cloud Framework*  
**Authors:** Vishu Singh (`23BAI0091`) & Shivam Raj (`23BCI0162`)  
**Supervisor:** Prof. Jim Solomon Raja — School of Computer Science and Engineering (SCOPE), VIT  

---

## 1. Architecture & Repository Structure

This workspace integrates two complementary layers without modifying or overwriting any source file in the original `iot_ids_framework` repository:

1. **Primary Core Repository (`iot_ids_framework/`)**:
   - **Fixed 14-Feature APAFC Edge Schema (`iot_ids.features`)**: Extracts `(protocol_id, duration_ms, packet_count, byte_count, mean_packet_bytes, src_port, dst_port, tcp_flags, mqtt_qos, mqtt_payload_bytes, coap_code, http_status, sixlowpan_dispatch, payload_entropy)` across MQTT, CoAP, HTTP, and 6LoWPAN flows.
   - **Packet & Flow Ingestion (`iot_ids.ingest`, `iot_ids.pcap_adapter`)**: Pure-Python `.pcap` parser (`PCAPReader`) and bounded bidirectional 5-tuple `FlowAssembler`.
   - **Optuna + SMOTE-ENN LightGBM Training & ONNX Export (`iot_ids.training`)**: Stratified K-Fold CV with in-fold `SMOTEENN`, Bayesian hyperparameter search via Optuna, and export to `edge_ids.onnx` + `edge_ids.txt`.
   - **Autonomous Edge Inference & Atomic Hot-Swap (`iot_ids.edge.EdgeIDS`)**: Serves `edge_ids.onnx` via `onnxruntime`, computes per-alert TreeSHAP attributions, monitors streaming KL drift, writes JSONL audit alerts, and performs thread-safe zero-downtime `hot_swap()`.
   - **Streaming KL Drift Monitor (`iot_ids.drift`, `iot_ids.drift_benchmark`)**: Sliding-window KL divergence ($\tau = 0.15$, $3$ consecutive windows) and false-alarm/detection benchmark harness.
   - **Clustered Federated GBDT Split Aggregation (`iot_ids.federation`)**: 16-bit quantized gradient & hessian histogram aggregation (`ClusteredHistogramAggregator`) and L2-regularized GBDT split gain selector (`best_split`).
   - **Classical ML Baselines (`iot_ids.baselines`)**: Empirical latency, memory, and F1 comparison across Decision Tree, Random Forest, KNN, and Logistic Regression.
   - **Reference C++ Runtime (`iot_ids_framework/cpp/edge_runtime.cpp`)**: 29-line C++17 ONNX Runtime single-thread inference reference implementation.

2. **Interactive Streamlit Dashboard & Exploratory 61→k Study (`app.py`, `pages/`, `src/`)**:
   - Bridges directly to `iot_ids_framework` via `src/framework_bridge.py`.
   - Provides a **separate, explicitly documented 61-to-$k$ dataset-driven feature selection workflow** (`src/preprocessing.py`, `src/feature_selection.py`, `src/train_model.py`, `src/inference.py`) for arbitrary uploaded CSV datasets, plus multi-gateway non-IID Jensen-Shannon divergence clustering (`src/edge_gateway.py`, `src/cloud_aggregator.py`).

```text
M:\projects\proj2(3 credits)\
├── app.py                              # Streamlit Executive Overview & SOC Dashboard
├── train_pipeline.py                   # Unified CLI initializer (Primary ONNX + Exploratory 61->14)
├── requirements.txt                    # Unified pinned dependencies + editable ./iot_ids_framework
├── README.md                           # Setup, verification, and 28 Oct 2026 review guide
├── config/
│   └── settings.py                     # Paths, constants, and leakage exclusion rules
├── src/
│   ├── framework_bridge.py             # Bridge to iot_ids_framework core modules & artifacts
│   ├── data_loader.py                  # CSV inspection, target detection & synthetic 61-feat generator
│   ├── preprocessing.py                # Leak-free split, median imputation & training-only SMOTE-ENN
│   ├── feature_selection.py            # Exploratory 61->k APAFC feature ranking
│   ├── train_model.py                  # Exploratory LightGBM training & full-vs-reduced ablation
│   ├── inference.py                    # Batch/single inference wrapper (EdgeClassifier)
│   ├── explainability.py               # TreeSHAP local explanation engine (TreeSHAPExplainer)
│   ├── alert_logger.py                 # Searchable JSONL alert logger (AlertLogger)
│   ├── drift_monitor.py                # Windowed KL divergence monitor & retraining workflow
│   ├── edge_gateway.py                 # Simulated Tier-1 EdgeGateway + FR-12 offline mode
│   ├── cloud_aggregator.py             # Simulated Tier-2 HistogramAggregator & JS clustering
│   └── ui_theme.py                     # Dark cybersecurity theme styling for Streamlit
├── pages/
│   ├── 1_Dataset_and_Training.py       # Core ONNX training, 61->k ablation & classical baselines
│   ├── 2_Detection.py                  # PCAP replay, live ONNX classification, hot-swap & alert log
│   ├── 3_Explainability.py             # Local TreeSHAP attributions & protocol feature rankings
│   ├── 4_Drift_Monitor.py              # KLDriftMonitor, DriftBenchmark & adaptive retraining
│   ├── 5_Edge_Cloud_Simulation.py      # Clustered GBDT split histograms & non-IID gateway clustering
│   └── 6_System_Diagnostics.py         # Artifact status, package versions & Git integrity
├── tests/
│   └── test_unified_prototype.py       # Integration & unit tests across both layers
└── iot_ids_framework/                  # Preserved primary reference repository (unmodified source)
    ├── demo.py                         # 6-stage CLI live presentation showcase
    ├── simulate_attacks.py             # Local loopback TCP socket server & attack generator
    ├── src/iot_ids/                    # Primary core package
    ├── tests/                          # 14 core unit & pipeline tests
    ├── configs/                        # JSON feature mappings & default parameters
    ├── cpp/                            # C++17 ONNX Runtime edge inference source
    ├── artifacts/                      # Generated edge_ids.onnx, edge_ids.txt, labels.json, reference.json
    └── datasets/                       # Generated sample_traffic.pcap & synthetic smoke CSV
```

---

## 2. Exact Windows PowerShell Commands

Run all commands from `M:\projects\proj2(3 credits)`:

### Step 1: Create Virtual Environment & Install Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Step 2: Initialize Models, Sample PCAP & Artifacts (Both Core ONNX & Exploratory)
```powershell
.\.venv\Scripts\python.exe train_pipeline.py
```

### Step 3: Run Full Automated Test Suite (Core + Unified Integration Tests)
```powershell
.\.venv\Scripts\pytest.exe iot_ids_framework\tests tests -v
```

### Step 4: Run the 6-Stage CLI Showcase (`iot_ids_framework/demo.py`)
```powershell
.\.venv\Scripts\python.exe iot_ids_framework\demo.py
```

### Step 5: Launch the Interactive Streamlit Dashboard
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

---

## 3. Component Status Matrix (Implemented vs. Simulated vs. Pending Hardware)

| Component | Status | Verification Details |
| :--- | :--- | :--- |
| **Fixed 14-Feature APAFC Extraction (`iot_ids.features`)** | **Implemented & Verified** | Tested on MQTT, CoAP, HTTP, and 6LoWPAN flows (`test_core.py`, `test_pipeline.py`). |
| **Pure-Python PCAP Replay & Bidirectional `FlowAssembler`** | **Implemented & Verified** | Reassembles Ethernet/IPv4/TCP/UDP packets from `.pcap` files (`test_unified_prototype.py`). |
| **Optuna + In-Fold SMOTE-ENN LightGBM & ONNX Export** | **Implemented & Verified** | Exports valid `edge_ids.onnx` and `edge_ids.txt`; verified with `onnxruntime` (`test_training.py`). |
| **ONNX Runtime `EdgeIDS` + TreeSHAP + Atomic `hot_swap()`** | **Implemented & Verified** | Sub-millisecond swap time, exact TreeSHAP top-$k$ attributions, and JSONL alert persistence. |
| **Exploratory 61→k Dataset-Driven Feature Selection** | **Implemented & Verified** | Leak-free train/test split, median imputation, training-only SMOTE-ENN, and full-61 vs. top-14 ablation. |
| **Sliding-Window KL Drift Monitor & Retraining Workflow** | **Implemented & Verified** | Smoothed KL divergence ($\tau = 0.15$, $3$ consecutive windows), `DriftBenchmark`, and retraining audit log. |
| **Tier-2 Clustered Quantized Histogram Aggregation** | **Simulated Prototype** | Validates 8/16-bit quantized gradient/hessian and feature histograms, rejects raw records, computes `best_split` & JS clustering. Does **not** claim automatic multi-round global LightGBM tree reconstruction without iterative gradient exchange. |
| **Synthetic Smoke / Demo Datasets (`sample_traffic.pcap`, CSVs)** | **Synthetic Smoke Data** | Clearly labelled synthetic data used because the 20M-row Edge-IIoTset CSV is not stored locally. Never claim synthetic metrics as real-world benchmark results. |
| **C++ ONNX Runtime (`cpp/edge_runtime.cpp`) & Raspberry Pi 4** | **Reference Source Only** | Source code is provided in `iot_ids_framework/cpp/`; physical Raspberry Pi 4 (4 GB) compilation, eBPF live NIC capture, and hardware RSS/latency profiling are Project-II / hardware-dependent tasks. |

---

## 4. Measurable Metrics & How to Collect Real Benchmark Numbers

When you place the real `ML-EdgeIIoT-dataset.csv` (from the public Edge-IIoTset archive) on your machine:

1. **Map & Train the Primary Core ONNX Model**:
   ```powershell
   .\.venv\Scripts\python.exe -m iot_ids.cli prepare --csv "path\to\ML-EdgeIIoT-dataset.csv" --mapping iot_ids_framework\configs\edge_iiotset_feature_mapping.json --label-column Attack_type --output iot_ids_framework\datasets\prepared_edge_iiotset.csv
   .\.venv\Scripts\python.exe -m iot_ids.cli train --csv iot_ids_framework\datasets\prepared_edge_iiotset.csv --label-column label --output iot_ids_framework\artifacts --trials 20
   ```
2. **Or Upload `ML-EdgeIIoT-dataset.csv` in the Dashboard (`1_Dataset_and_Training`)**:
   - Select **Tab 2 (Exploratory 61-to-k Study)**, upload the CSV, choose `Attack_type` as the target column, and click **Run 61-to-k Exploratory Feature Selection & Training**.
   - Record the measured **Test Accuracy**, **Macro-F1**, **Weighted-F1**, **Per-Class Recall**, **Single-Flow p50/p95/p99 Inference Latency (ms)**, **Artifact Size (KB)**, and **61-vs-14 Macro-F1 Retention (%)** from `results/latest_evaluation.json`.

---

## 5. Short Final Review Demonstration Script (28 October 2026)

1. **Terminal Demo (1 min)**:
   - Run `.\.venv\Scripts\python.exe iot_ids_framework\demo.py` to show the 6 stages executing live in the console: `.pcap` packet replay $\to$ 14-feature APAFC vector $\to$ ONNX Runtime + TreeSHAP alert $\to$ 3-window KL drift trigger $\to$ sub-millisecond atomic ONNX `hot_swap()`.
2. **Dashboard Overview & Training (`app.py` & Page 1 — 2 min)**:
   - Open the Streamlit dashboard (`.\.venv\Scripts\streamlit.exe run app.py`).
   - On **Page 1**, show **Tab 1** (Fixed 14-feature ONNX pipeline) and **Tab 2** (Exploratory 61-to-14 feature compression ablation table and confusion matrix), plus **Tab 3** (Classical ML baselines comparison: Decision Tree, Random Forest, KNN, Logistic Regression).
3. **Live PCAP & Single-Flow Detection + Atomic Hot-Swap (Page 2 — 2 min)**:
   - On **Page 2 (Tab 1)**, click **Replay PCAP & Classify Reassembled Flows** to show bidirectional MQTT, CoAP, HTTP, and 6LoWPAN flows reassembled from `sample_traffic.pcap` and classified by `EdgeIDS`.
   - Switch to **Tab 2**, select an attack preset, classify it, and click **Test Zero-Downtime Atomic Model Hot-Swap**.
4. **TreeSHAP Explainability (Page 3 — 1 min)**:
   - On **Page 3**, select any flow record to display its exact per-record TreeSHAP directional bar chart alongside the global split-gain ranking.
5. **Concept Drift & Retraining (Page 4 — 1.5 min)**:
   - On **Page 4 (Tab 1)**, click **Run `DriftBenchmark`** to show steady baseline windows ($\text{KL} < 0.15$, 0 false triggers) vs. shifted traffic windows ($\text{KL} > 0.15$ for 3 consecutive windows $\to$ retraining request).
6. **Edge–Cloud Privacy-Preserving Aggregation (Page 5 — 1.5 min)**:
   - On **Page 5 (Tab 1)**, run `ClusteredHistogramAggregator` to show 16-bit quantized gradient/hessian aggregation and `best_split` selection.
   - On **Tab 2 & Tab 3**, run the 3-gateway non-IID simulation (~99% bandwidth reduction, zero raw flow records transmitted) and demonstrate **FR-12 Autonomous Offline Operation**.
