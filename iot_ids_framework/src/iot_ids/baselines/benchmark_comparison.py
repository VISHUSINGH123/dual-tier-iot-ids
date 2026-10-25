"""Comparative benchmarking harness: Classical ML baselines vs. Proposed LightGBM + APAFC.

Measures Macro-F1, inference latency, model artifact size, and RAM footprints.
"""
from __future__ import annotations

from pathlib import Path
import time
import os
try:
    import psutil
except ImportError:
    psutil = None
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .baseline_models import BaselineModels


def benchmark_baseline_models(
    dataset_csv: str | Path,
    task: str = "binary",
    test_size: float = 0.2,
    sample_size: int | None = 10000,
    models_dir: str | Path = "artifacts/baselines"
) -> pd.DataFrame:
    """Run empirical benchmark across baseline models and return comparative DataFrame."""
    df = pd.read_csv(dataset_csv, low_memory=False)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    target_col = "label" if "label" in df.columns else ("Attack_label" if "Attack_label" in df.columns else df.columns[-1])
    drop_cols = [target_col]
    for col in ("Attack_type", "type", "attack_cat"):
        if col in df.columns:
            drop_cols.append(col)
    feature_df = df.drop(columns=drop_cols).select_dtypes(include=[np.number]).fillna(0)
    x = feature_df.to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    model_names = ["decision_tree", "random_forest", "knn", "logistic_regression"]
    results = []

    for name in model_names:
        # Check if pre-trained exists, else fit
        model_file = Path(models_dir) / f"{name}_{task}.pkl"
        start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) if psutil else 0.0

        if model_file.exists():
            try:
                artifact_size_mb = model_file.stat().st_size / (1024 * 1024)
                model = BaselineModels.load_pretrained(name, task=task, models_dir=models_dir)
            except Exception:
                # Version mismatch or corrupted pickle; fall back to retraining
                artifact_size_mb = 0.0
                model = BaselineModels.get_model(name, task=task)
                model.fit(x_train, y_train)
        else:
            artifact_size_mb = 0.0
            model = BaselineModels.get_model(name, task=task)
            model.fit(x_train, y_train)

        # Measure p95 inference latency across single-sample calls
        latencies_ms = []
        subset_test = x_test[:min(len(x_test), 500)]
        for i in range(len(subset_test)):
            sample = subset_test[i:i + 1]
            t0 = time.perf_counter()
            _ = model.predict(sample)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        p95_latency = float(np.percentile(latencies_ms, 95))
        metrics = BaselineModels.evaluate(model, x_test, y_test, task=task)
        end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) if psutil else 0.0

        results.append({
            "model": f"{name} ({task})",
            "accuracy": round(metrics["accuracy"] * 100, 2),
            "macro_f1": round(metrics["macro_f1"], 4),
            "p95_latency_ms": round(p95_latency, 2),
            "artifact_size_mb": round(artifact_size_mb, 2),
            "rss_mb": round(end_mem - start_mem, 2)
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import sys
    default_path = "datasets/Edge-IIoTset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv"
    data_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    res = benchmark_baseline_models(data_path, task="binary", sample_size=5000)
    print(res.to_string(index=False))
