"""Drift Monitor Evaluation & False-Trigger Rate Benchmark (Module M7).

Quantifies the KL divergence drift monitor's false-trigger rate under benign steady state,
and evaluates detection response under injected concept drift scenarios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from .drift import KLDriftMonitor


@dataclass(frozen=True)
class DriftBenchmarkResult:
    benign_windows_evaluated: int
    false_trigger_count: int
    false_trigger_rate: float
    drift_windows_evaluated: int
    drift_detection_count: int
    true_positive_rate: float
    mean_kl_benign: float
    mean_kl_drift: float


class DriftBenchmark:
    """Benchmark harness for validating threshold tau and consecutive window sensitivity."""

    def __init__(
        self,
        reference_data: np.ndarray,
        window_size: int = 500,
        bins: int = 16,
        threshold: float = 0.15,
        consecutive_windows: int = 3,
    ) -> None:
        self.reference_data = reference_data
        self.window_size = window_size
        self.bins = bins
        self.threshold = threshold
        self.consecutive_windows = consecutive_windows

    def run_benchmark(
        self,
        benign_stream: np.ndarray,
        drift_stream: np.ndarray,
        drift_injection_magnitude: float = 2.0
    ) -> DriftBenchmarkResult:
        """Evaluates false alarm rate on benign stream, then injects drift and measures response."""
        monitor = KLDriftMonitor(
            self.reference_data,
            window_size=self.window_size,
            bins=self.bins,
            threshold=self.threshold,
            consecutive_windows=self.consecutive_windows,
        )

        # 1. Benign evaluation (Measure False Alarms)
        benign_kl_scores = []
        false_triggers = 0
        benign_windows = 0

        for row in benign_stream:
            event = monitor.observe(row)
            if event is not None:
                benign_kl_scores.append(event.score)
                if event.retrain_requested:
                    false_triggers += 1
        
        benign_windows = len(benign_kl_scores)
        false_rate = (false_triggers / benign_windows) if benign_windows > 0 else 0.0

        # 2. Injected Concept Drift evaluation
        # Inject variance/mean shifts mimicking new attack vectors or network updates
        shifted_stream = drift_stream + (np.std(self.reference_data, axis=0, keepdims=True) * drift_injection_magnitude)
        
        drift_kl_scores = []
        drift_detections = 0
        drift_windows = 0

        for row in shifted_stream:
            event = monitor.observe(row)
            if event is not None:
                drift_kl_scores.append(event.score)
                if event.retrain_requested:
                    drift_detections += 1

        drift_windows = len(drift_kl_scores)
        detection_rate = (drift_detections / max(1, (drift_windows // self.consecutive_windows)))
        detection_rate = min(1.0, detection_rate)

        return DriftBenchmarkResult(
            benign_windows_evaluated=benign_windows,
            false_trigger_count=false_triggers,
            false_trigger_rate=round(false_rate, 4),
            drift_windows_evaluated=drift_windows,
            drift_detection_count=drift_detections,
            true_positive_rate=round(detection_rate, 4),
            mean_kl_benign=round(float(np.mean(benign_kl_scores)), 4) if benign_kl_scores else 0.0,
            mean_kl_drift=round(float(np.mean(drift_kl_scores)), 4) if drift_kl_scores else 0.0,
        )
