"""Reproducible LightGBM training with in-fold SMOTE-ENN and ONNX export."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import optuna

from .features import FEATURE_NAMES


def train(csv_path: str | Path, label_column: str, output_dir: str | Path, trials: int = 120) -> dict[str, float]:
    """Train against a pre-mapped CSV containing the 14 APAFC feature names."""
    frame = pd.read_csv(csv_path)
    missing = set(FEATURE_NAMES) - set(frame.columns)
    if missing:
        raise ValueError(f"mapped CSV is missing required columns: {sorted(missing)}")
    x = frame.loc[:, FEATURE_NAMES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
    source_labels = frame[label_column].astype(str).str.strip().str.lower()
    if "benign" not in set(source_labels) and "normal" in set(source_labels):
        source_labels = source_labels.replace("normal", "benign")
    if "benign" not in set(source_labels):
        raise ValueError("the mapped dataset must include the benign class (or 'normal')")
    class_names = ["benign", *sorted(set(source_labels) - {"benign"})]
    if len(class_names) < 2:
        raise ValueError(f"expected at least 2 classes, received {len(class_names)}: {class_names}")
    label_index = {name: index for index, name in enumerate(class_names)}
    y = source_labels.map(label_index).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, stratify=y, random_state=42)
    min_class_count = int(pd.Series(y_train).value_counts().min())
    n_splits = min(10, max(2, min_class_count))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    min_child_samples = max(2, min(20, min_class_count // 2))

    def objective(trial: optuna.Trial) -> float:
        model = LGBMClassifier(n_estimators=trial.suggest_int("n_estimators", 160, 320), max_depth=trial.suggest_int("max_depth", 4, 10), learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1), num_leaves=trial.suggest_int("num_leaves", 15, 127), min_child_samples=min_child_samples, objective="multiclass", verbosity=-1, random_state=42)
        # Resampling belongs inside each fold to avoid synthetic-sample leakage.
        from imblearn.pipeline import Pipeline
        pipeline = Pipeline([("resample", SMOTEENN(random_state=42)), ("model", model)])
        try:
            return float(cross_val_score(pipeline, x_train, y_train, scoring="f1_macro", cv=cv, n_jobs=-1).mean())
        except Exception:
            return float(cross_val_score(model, x_train, y_train, scoring="f1_macro", cv=cv, n_jobs=-1).mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    try:
        sampler = SMOTEENN(random_state=42)
        x_balanced, y_balanced = sampler.fit_resample(x_train, y_train)
        if len(set(y_balanced)) < len(class_names):
            x_balanced, y_balanced = x_train, y_train
    except Exception:
        x_balanced, y_balanced = x_train, y_train
    model = LGBMClassifier(**study.best_params, min_child_samples=min_child_samples, objective="multiclass", verbosity=-1, random_state=42)
    model.fit(x_balanced, y_balanced)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(out / "edge_ids.txt"))
    (out / "labels.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (out / "reference.json").write_text(json.dumps({"reference_features": x_train.tolist()}), encoding="utf-8")
    try:
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        onnx_model = convert_lightgbm(model.booster_, initial_types=[("features", FloatTensorType([None, len(FEATURE_NAMES)]))], target_opset=None)
        (out / "edge_ids.onnx").write_bytes(onnx_model.SerializeToString())
    except ImportError as exc:
        raise RuntimeError("ONNX export requires onnxmltools; install requirements.txt") from exc
    return {"cv_macro_f1": study.best_value, "test_macro_f1": float(__import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(y_test, model.predict(x_test), average="macro"))}
