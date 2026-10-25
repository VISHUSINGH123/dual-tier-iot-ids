r"""
CLI Training & Prototype Initialization Entry Point.

Usage (PowerShell):
    .\.venv\Scripts\python.exe train_pipeline.py

Executes and verifies both:
A. Primary Core Pipeline (`iot_ids_framework`):
   - Generates clearly-labelled synthetic 14-feature APAFC smoke dataset (`synthetic_apafc14_smoke.csv`).
   - Generates multi-protocol synthetic IoT packet capture (`sample_traffic.pcap`).
   - Runs `iot_ids.training.train()` (Optuna + in-fold SMOTE-ENN + LightGBM + ONNX export)
     producing `edge_ids.onnx`, `edge_ids.txt`, `labels.json`, and `reference.json`.
   - Verifies `iot_ids.pcap_adapter.replay_pcap_to_flows` and `iot_ids.edge.EdgeIDS` ONNX inference + TreeSHAP.
B. Exploratory 61-to-14 Dataset-Driven Pipeline (`src/`):
   - Runs leak-free preprocessing, 61->14 APAFC feature selection, LightGBM training,
     TreeSHAP alert logging, KL drift window evaluation, and multi-gateway histogram aggregation.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from config.settings import (
    DEFAULT_APAFC_K,
    DRIFT_DEMO_CSV,
    DRIFT_REFERENCE_PATH,
    MODEL_ARTIFACT_PATH,
)
from src.alert_logger import AlertLogger
from src.cloud_aggregator import HistogramAggregator, partition_dataset_for_gateways
from src.data_loader import inspect_dataset, load_or_create_demo_dataset
from src.drift_monitor import DriftMonitor
from src.edge_gateway import EdgeGateway
from src.explainability import TreeSHAPExplainer
from src.framework_bridge import (
    FRAMEWORK_ARTIFACTS,
    SAMPLE_PCAP_PATH,
    SYNTHETIC_APAFC14_CSV,
    generate_sample_iot_pcap,
    generate_synthetic_apafc14_dataset,
    load_core_edge_ids,
    replay_pcap_to_flows,
    train_core_onnx_pipeline,
)
from src.inference import EdgeClassifier
from src.train_model import train_and_evaluate_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and verify the Unified Dual-Tier IoT IDS Prototype.")
    parser.add_argument("--samples", type=int, default=3000, help="Number of samples for 61-feature synthetic demo dataset.")
    parser.add_argument("--k-features", type=int, default=DEFAULT_APAFC_K, help="Number of APAFC features in exploratory study.")
    parser.add_argument("--trials", type=int, default=3, help="Optuna trials for primary core ONNX training.")
    parser.add_argument("--resampling", type=str, default="smote_enn", choices=["smote_enn", "smote", "none"])
    args = parser.parse_args()

    print("=== [STAGE A: PRIMARY CORE (`iot_ids_framework`) ONNX & PCAP PIPELINE] ===")
    df_14 = generate_synthetic_apafc14_dataset(output_csv=SYNTHETIC_APAFC14_CSV, rows_per_class=80)
    pcap_path = generate_sample_iot_pcap(SAMPLE_PCAP_PATH)
    print(f"Generated synthetic 14-feature smoke dataset: {SYNTHETIC_APAFC14_CSV} ({len(df_14)} rows)")
    print(f"Generated synthetic multi-protocol PCAP: {pcap_path} ({pcap_path.stat().st_size} bytes)")

    core_metrics = train_core_onnx_pipeline(
        csv_path=SYNTHETIC_APAFC14_CSV,
        label_column="label",
        output_dir=FRAMEWORK_ARTIFACTS,
        trials=args.trials,
    )
    print(f"Primary Core ONNX Training Complete -> CV Macro-F1: {core_metrics['cv_macro_f1']:.4f} | "
          f"Test Macro-F1: {core_metrics['test_macro_f1']:.4f}")

    flows = replay_pcap_to_flows(pcap_path)
    edge_ids = load_core_edge_ids()
    pcap_preds = [edge_ids.classify(fl)["class"] for fl in flows]
    print(f"Replayed {len(flows)} bidirectional flows from {pcap_path.name} via ONNX EdgeIDS -> Predictions: {pcap_preds}")

    print("\n=== [STAGE B.1: EXPLORATORY 61-FEATURE DATASET & LEAK-FREE PREPROCESSING] ===")
    df, csv_path, is_synthetic = load_or_create_demo_dataset(n_samples=args.samples, force_regenerate=True)
    summary = inspect_dataset(
        df,
        dataset_name=csv_path.name,
        target_col="Attack_type",
        is_synthetic_demo=is_synthetic,
    )
    print(f"Dataset: {summary.dataset_name} (Synthetic Demo: {summary.is_synthetic_demo})")
    print(f"Rows: {summary.num_rows}, Columns: {summary.num_columns}, Classes: {summary.num_classes}")

    print("\n=== [STAGE B.2: EXPLORATORY 61->14 APAFC SELECTION & LIGHTGBM TRAINING] ===")
    bundle, eval_res = train_and_evaluate_pipeline(
        df=df,
        target_col=summary.detected_label_col or "Attack_type",
        dataset_name="Synthetic Edge-IIoTset Demo (61 Features, 15 Classes)" if is_synthetic else csv_path.name,
        is_synthetic_dataset=is_synthetic,
        k_features=args.k_features,
        resampling_method=args.resampling,
        compare_with_full_features=True,
        artifact_path=MODEL_ARTIFACT_PATH,
    )
    print(f"Saved exploratory bundle: {MODEL_ARTIFACT_PATH.name} ({eval_res.artifact_size_kb:.2f} KB)")
    print(f"Test Accuracy: {eval_res.accuracy * 100:.2f}% | Macro-F1: {eval_res.macro_f1:.4f} | "
          f"p95 Latency: {eval_res.inference_latency_p95_ms:.3f} ms")

    print("\n=== [STAGE B.3: TREESHAP EXPLANATIONS & ALERT LOGGING] ===")
    classifier = EdgeClassifier(MODEL_ARTIFACT_PATH).load()
    sample_df = df.sample(n=30, random_state=101).reset_index(drop=True)
    batch_out = classifier.predict_batch(sample_df)
    X_red, _ = classifier.preprocess_to_reduced(sample_df)

    explainer = TreeSHAPExplainer(
        model=classifier.model,
        feature_names=classifier.selected_features,
        label_mapping=classifier.label_mapping,
        top_k=6,
    )
    explanations = {
        idx: explainer.explain_instance(
            X_reduced_row=X_red.iloc[idx : idx + 1],
            predicted_class_id=batch_out.predictions[idx].predicted_class_id,
            confidence=batch_out.predictions[idx].confidence,
            record_index=idx,
        )
        for idx in range(min(15, len(batch_out.predictions)))
    }
    logger = AlertLogger()
    logger.clear_logs()
    logged = logger.log_batch(batch_out.predictions[:15], explanations=explanations, raw_df=sample_df)
    print(f"Logged {len(logged)} alerts to {logger.log_path.name}.")

    print("\n=== [STAGE B.4: CONCEPT DRIFT MONITOR & MULTI-GATEWAY AGGREGATION] ===")
    monitor = DriftMonitor.load_reference(DRIFT_REFERENCE_PATH)
    monitor.reset_history()
    for w_idx in range(2):
        sub_base = df.iloc[w_idx * 200 : (w_idx + 1) * 200].reset_index(drop=True)
        X_w, _ = classifier.preprocess_to_reduced(sub_base)
        y_true_w = classifier.preprocessor.encode_labels(sub_base["Attack_type"])
        y_pred_w = classifier.model.predict(X_w)
        monitor.evaluate_window(X_w, window_label=f"Baseline Window #{w_idx + 1}", y_true=y_true_w, y_pred=y_pred_w)

    drifted_df = pd.read_csv(DRIFT_DEMO_CSV, low_memory=False)
    for w_idx in range(3):
        sub_drift = drifted_df.iloc[w_idx * 200 : (w_idx + 1) * 200].reset_index(drop=True)
        X_w, _ = classifier.preprocess_to_reduced(sub_drift)
        y_true_w = classifier.preprocessor.encode_labels(sub_drift["Attack_type"])
        y_pred_w = classifier.model.predict(X_w)
        res = monitor.evaluate_window(X_w, window_label=f"Shifted Traffic Window #{w_idx + 1}", y_true=y_true_w, y_pred=y_pred_w)
        print(f"  {res.window_label}: KL={res.overall_kl_divergence:.4f} (retrain_triggered={res.retraining_triggered})")

    partitions = partition_dataset_for_gateways(df, num_gateways=3, partition_mode="non_iid")
    aggregator = HistogramAggregator(shared_bin_edges=monitor.feature_bins, cluster_divergence_threshold=0.16)
    payloads = []
    raw_bytes_map = {}
    for gid, profile, g_df in partitions:
        gw = EdgeGateway(gateway_id=gid, local_df=g_df, classifier=classifier, gateway_profile=profile, quantization_bits=8)
        gw.run_local_detection()
        payloads.append(gw.build_histogram_update(shared_bin_edges=monitor.feature_bins, round_id=1))
        raw_bytes_map[gid] = len(g_df.to_csv(index=False).encode("utf-8"))

    round_summary = aggregator.aggregate_round(payloads, raw_bytes_Map=raw_bytes_map)
    print(f"Aggregated {len(round_summary.participating_gateways)} gateways | "
          f"Bandwidth saving: {round_summary.bandwidth_reduction_pct:.2f}% | "
          f"Clusters: {json.dumps(round_summary.gateway_clusters)}")


if __name__ == "__main__":
    main()
