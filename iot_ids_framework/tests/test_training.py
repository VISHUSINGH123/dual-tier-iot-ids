"""Test suite for training pipeline, Optuna hyperparameter tuning, and ONNX export."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from iot_ids.features import FEATURE_NAMES


def create_synthetic_multiclass_dataset(file_path: Path, rows_per_class: int = 15):
    """Creates a minimal synthetic dataset with all 15 classes and 14 APAFC features."""
    classes = [
        "benign", "ddos", "dos", "injection", "mitm", "password", "port_scan",
        "ransomware", "scanning", "spoofing", "xss", "backdoor", "botnet",
        "brute_force", "web_attack"
    ]
    data = []
    rng = np.random.RandomState(42)
    for class_idx, class_name in enumerate(classes):
        for _ in range(rows_per_class):
            # Form distinct clusters with small variance so SMOTE-ENN maintains the distribution
            center = float(class_idx * 20.0)
            row = {feat: float(rng.normal(loc=center, scale=1.0)) for feat in FEATURE_NAMES}
            row["protocol_id"] = float(class_idx % 4 + 1)
            row["label"] = class_name
            data.append(row)
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return df


def test_train_pipeline_execution(tmp_path):
    # Try importing dependencies
    try:
        import lightgbm
        import optuna
        import imblearn
        import onnxmltools
        from iot_ids.training import train
    except ImportError as e:
        pytest.skip(f"Training dependencies not yet installed in active environment: {e}")

    dataset_path = tmp_path / "synthetic_15_classes.csv"
    create_synthetic_multiclass_dataset(dataset_path, rows_per_class=20)
    output_dir = tmp_path / "model_artifacts"

    # Train with 2 trials for fast unit test verification
    metrics = train(
        csv_path=dataset_path,
        label_column="label",
        output_dir=output_dir,
        trials=2
    )

    assert "cv_macro_f1" in metrics
    assert "test_macro_f1" in metrics
    assert (output_dir / "edge_ids.txt").exists()
    assert (output_dir / "labels.json").exists()
    assert (output_dir / "reference.json").exists()
    assert (output_dir / "edge_ids.onnx").exists()
