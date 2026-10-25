import numpy as np
from iot_ids.baselines.baseline_models import BaselineModels
from iot_ids.baselines.unsw_preprocessor import UNSWPreprocessor


def test_baseline_models_instantiation():
    dt = BaselineModels.get_model("decision_tree", task="binary")
    assert dt is not None
    rf = BaselineModels.get_model("random_forest", task="multi")
    assert rf is not None


def test_baseline_evaluation_metrics():
    # Mock data
    x_test = np.array([[0.1, 0.2], [0.8, 0.9], [0.2, 0.3], [0.7, 0.6]])
    y_test = np.array([0, 1, 0, 1])

    dt = BaselineModels.get_model("decision_tree", task="binary")
    dt.fit(x_test, y_test)
    metrics = BaselineModels.evaluate(dt, x_test, y_test, task="binary")

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["accuracy"] == 1.0
