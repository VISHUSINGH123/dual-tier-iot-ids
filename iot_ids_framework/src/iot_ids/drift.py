"""Smoothed KL-divergence monitoring for live feature windows."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass(frozen=True)
class DriftEvent:
    score: float
    consecutive: int
    retrain_requested: bool


class KLDriftMonitor:
    def __init__(self, reference: np.ndarray, window_size: int = 500, bins: int = 16,
                 threshold: float = 0.15, consecutive_windows: int = 3) -> None:
        if reference.ndim != 2 or len(reference) == 0:
            raise ValueError("reference must be a non-empty 2D feature matrix")
        self.window_size, self.bins = window_size, bins
        self.threshold, self.consecutive_windows = threshold, consecutive_windows
        self.edges = [np.histogram_bin_edges(reference[:, i], bins=bins) for i in range(reference.shape[1])]
        self.reference = [self._distribution(reference[:, i], edge) for i, edge in enumerate(self.edges)]
        self.window: deque[np.ndarray] = deque(maxlen=window_size)
        self.consecutive = 0

    def observe(self, vector: list[float] | np.ndarray) -> DriftEvent | None:
        self.window.append(np.asarray(vector, dtype=float))
        if len(self.window) < self.window_size:
            return None
        live = np.vstack(self.window)
        scores = [self._kl(self._distribution(live[:, i], edge), self.reference[i]) for i, edge in enumerate(self.edges)]
        score = float(np.mean(scores))
        self.consecutive = self.consecutive + 1 if score > self.threshold else 0
        # Emit one request at the threshold crossing. The caller should keep serving the
        # last-known-good model until a replacement has passed validation.
        return DriftEvent(score, self.consecutive, self.consecutive == self.consecutive_windows)

    @staticmethod
    def _distribution(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(values, bins=edges)
        distribution = counts.astype(float) + 1e-8
        return distribution / distribution.sum()

    @staticmethod
    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log(p / q)))
