"""Clustered, quantised histogram aggregation for federated GBDT split selection."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
import numpy as np


@dataclass(frozen=True)
class QuantisedHistogram:
    client_id: str
    cluster_id: str
    feature: str
    bin_edges: tuple[float, ...]
    gradient: tuple[int, ...]
    hessian: tuple[int, ...]
    scale: float

    def __post_init__(self) -> None:
        if len(self.gradient) != len(self.hessian) or len(self.bin_edges) != len(self.gradient) + 1:
            raise ValueError("incompatible split histogram dimensions")


def quantise_histogram(client_id: str, cluster_id: str, feature: str, bin_edges: Iterable[float],
                       gradient: Iterable[float], hessian: Iterable[float], bits: int = 16) -> QuantisedHistogram:
    gradient, hessian = np.asarray(list(gradient)), np.asarray(list(hessian))
    max_value = max(float(np.abs(gradient).max(initial=0)), float(np.abs(hessian).max(initial=0)), 1e-12)
    scale = ((2 ** (bits - 1)) - 1) / max_value
    return QuantisedHistogram(client_id, cluster_id, feature, tuple(bin_edges),
                              tuple(np.rint(gradient * scale).astype(int)), tuple(np.rint(hessian * scale).astype(int)), scale)


class ClusteredHistogramAggregator:
    """Aggregates only compatible summary statistics; raw flows are intentionally not accepted."""

    def aggregate(self, reports: Iterable[QuantisedHistogram]) -> dict[tuple[str, str], dict[str, object]]:
        totals: dict[tuple[str, str], dict[str, object]] = {}
        for report in reports:
            key = (report.cluster_id, report.feature)
            if key not in totals:
                totals[key] = {"bin_edges": report.bin_edges, "gradient": np.zeros(len(report.gradient)), "hessian": np.zeros(len(report.hessian)), "clients": set()}
            current = totals[key]
            if current["bin_edges"] != report.bin_edges:
                raise ValueError(f"incompatible bins for {key}; clients must use the shared bin plan")
            current["gradient"] += np.asarray(report.gradient) / report.scale
            current["hessian"] += np.asarray(report.hessian) / report.scale
            current["clients"].add(report.client_id)
        return totals

    @staticmethod
    def best_split(aggregate: Mapping[str, object], l2: float = 1.0) -> int:
        gradient, hessian = np.asarray(aggregate["gradient"]), np.asarray(aggregate["hessian"])
        total_g, total_h = gradient.sum(), hessian.sum()
        gains = []
        for point in range(1, len(gradient)):
            left_g, left_h = gradient[:point].sum(), hessian[:point].sum()
            right_g, right_h = total_g - left_g, total_h - left_h
            gains.append((left_g ** 2 / (left_h + l2)) + (right_g ** 2 / (right_h + l2)) - (total_g ** 2 / (total_h + l2)))
        return int(np.argmax(gains)) if gains else 0
