"""Baseline machine learning algorithms from the reference project.

Encapsulates Decision Tree, Random Forest, KNN, Linear SVM, and MLP for binary & multiclass evaluation.
"""
from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error


class BaselineModels:
    """Provides unified training, loading, and evaluation for classical ML baselines."""

    AVAILABLE_MODELS = {
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "knn": KNeighborsClassifier,
        "lsvm": SVC,
        "mlp": MLPClassifier,
        "logistic_regression": LogisticRegression,
        "linear_regression": LinearRegression,
    }

    @staticmethod
    def get_model(name: str, task: str = "binary", **kwargs: Any) -> Any:
        """Instantiate a baseline model with configuration matching reference experiments."""
        if name == "decision_tree":
            return DecisionTreeClassifier(random_state=123, **kwargs)
        elif name == "random_forest":
            return RandomForestClassifier(random_state=123 if task == "binary" else 50, **kwargs)
        elif name == "knn":
            return KNeighborsClassifier(n_neighbors=5, **kwargs)
        elif name == "lsvm":
            return SVC(kernel="linear", gamma="auto", **kwargs)
        elif name == "mlp":
            return MLPClassifier(random_state=123, solver="adam", max_iter=5000 if task == "binary" else 8000, **kwargs)
        elif name == "logistic_regression":
            return LogisticRegression(random_state=123, max_iter=5000, solver="newton-cg" if task == "multi" else "lbfgs", **kwargs)
        elif name == "linear_regression":
            return LinearRegression(**kwargs)
        else:
            raise ValueError(f"Unknown baseline model: {name}")

    @staticmethod
    def load_pretrained(model_name: str, task: str = "binary", models_dir: str | Path = "models/baselines") -> Any:
        """Load pickled pre-trained baseline model."""
        target_path = Path(models_dir) / f"{model_name}_{task}.pkl"
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found: {target_path}")
        with open(target_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def evaluate(model: Any, x_test: np.ndarray, y_test: np.ndarray, task: str = "binary") -> dict[str, float]:
        """Evaluate baseline model accuracy, macro F1, and error metrics."""
        y_pred = model.predict(x_test)
        if isinstance(model, LinearRegression):
            y_pred = np.round(y_pred).astype(int)

        avg_type = "binary" if task == "binary" else "macro"
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average=avg_type, zero_division=0)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }
