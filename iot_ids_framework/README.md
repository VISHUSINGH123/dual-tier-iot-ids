# Dual-tier IoT IDS reference implementation

This is a runnable prototype derived from the Review-2 specification. It keeps the
edge path autonomous and uses only feature vectors, quantised histograms, and model
artifacts outside the gateway.

## Design parameters encoded here

| Parameter | Value |
| --- | --- |
| Protocols | MQTT, CoAP, HTTP, 6LoWPAN |
| Input reduction | 61 source columns to 14 APAFC features |
| Classes | 15, including `benign` |
| Drift statistic | KL divergence, threshold 0.15 |
| Consecutive windows | 3 by default, configurable | 
| Alert explanations | top 5 TreeSHAP values |
| Federation | quantised gradient histograms grouped by client cluster |

The presentation does not define the number of consecutive windows for FR-07. The
configuration uses `3`; change `drift.consecutive_windows` after selecting it with
your guide.

## Quick start

Use Python 3.10+.

```powershell
pip install -r requirements.txt
pip install -e .
```

### 1. Run the One-Click Live Presentation Demo
Run the interactive showcase demonstrating all 6 stages (PCAP ingestion, APAFC extraction, autonomous edge inference, TreeSHAP attributions, real-time KL concept drift detection, atomic zero-downtime hot-swap, and classical baseline benchmarks):

```powershell
python demo.py
```

### 2. Live Edge Serving & Real-Time Flow Classification
The repository includes pre-trained production model artifacts in `artifacts/`. You can immediately start the edge inference server:

```powershell
# Start edge serving with ONNX Runtime, TreeSHAP explainer, and KL drift monitoring:
python -m iot_ids.cli serve --model artifacts\edge_ids.onnx --reference artifacts\reference.json --tree-model artifacts\edge_ids.txt
```

To submit a flow record directly:
```powershell
'{"protocol":"mqtt","packet_count":4,"byte_count":310,"duration_ms":20,"src_port":1883,"dst_port":60000,"mqtt_qos":1,"mqtt_retain":false,"mqtt_payload_bytes":120}' | python -m iot_ids.cli serve --model artifacts\edge_ids.onnx --reference artifacts\reference.json --tree-model artifacts\edge_ids.txt
```

### 3. Model Training Pipeline (Optional)
To prepare and retrain with Bayesian hyperparameter tuning and SMOTE-ENN on the extracted dataset:
```powershell
python -m iot_ids.cli prepare --csv "datasets\Edge-IIoTset\Selected dataset for ML and DL\ML-EdgeIIoT-dataset.csv" --mapping configs\edge_iiotset_feature_mapping.json --label-column Attack_type --output "datasets\prepared_edge_iiotset.csv"
python -m iot_ids.cli train --csv "datasets\prepared_edge_iiotset.csv" --label-column label --trials 20
```

To submit a completed flow record to the live prototype, pipe one JSON object per
line into `serve`. It prints a JSON audit record for each attack and a retraining
request after sustained drift.

```powershell
'{"protocol":"mqtt","packet_count":4,"byte_count":310,"duration_ms":20,"src_port":1883,"dst_port":60000,"mqtt_qos":1,"mqtt_retain":false,"mqtt_payload_bytes":120}' | python -m iot_ids.cli serve --model artifacts\edge_ids.onnx --reference artifacts\reference.json
```

## Modules

- `features.py`: M1/M3 protocol-aware flow feature extraction.
- `prepare.py`: maps a dataset export into the fixed, auditable APAFC training schema.
- `training.py`: M2/M4 in-fold SMOTE-ENN, Optuna-tuned LightGBM, ONNX export.
- `ingest.py`: M1 bidirectional flow assembly, ready to receive eBPF/libpcap packet metadata.
- `edge.py`: M5/M6/M7/M9 edge inference, SHAP explanations, KL monitoring, JSON alert logging, hot swap.
- `federation.py`: M8 clustered quantised split-histogram aggregation. It rejects raw flow payloads.
- `cpp/edge_runtime.cpp`: minimal ONNX Runtime C++ inference path for FR-08.

## Notes for deployment

This repository provides the application logic; packet capture is intentionally an
adapter boundary. Connect `FlowAssembler.submit()` to your approved eBPF/libpcap
collector, and run the supplied C++ runtime on the Raspberry Pi for the production
inference process. Run the performance harness on the Pi before claiming the NFR
latency, RSS, throughput, or drop-rate targets.
