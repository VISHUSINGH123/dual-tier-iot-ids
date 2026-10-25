"""
Module F (Part 1) — Simulated Tier-1 Edge Gateway (`EdgeGateway`).

Implements `EdgeGateway` (matching the BCSE497J report Class Diagram):
- Holds a local partition of IoT flow records.
- Executes local intrusion detection via `EdgeClassifier` (autonomous operation even
  when the Tier-2 Cloud Aggregator is unreachable — FR-12 Graceful Degradation).
- Computes local quantized feature histograms over shared candidate split bins and
  predicted class-distribution statistics.
- Generates a schema-validated aggregate payload (`GatewayUpdatePayload`) that NEVER
  contains raw sample records, IP addresses, or reconstructable flow rows (FR-09 Privacy Boundary).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.inference import BatchPredictionOutput, EdgeClassifier


FORBIDDEN_RAW_KEYS = {
    "raw_records",
    "raw_flows",
    "rows",
    "samples",
    "ip.src_host",
    "ip.dst_host",
    "src_ip",
    "dst_ip",
    "dataframe",
}


@dataclass
class GatewayUpdatePayload:
    """
    Privacy-preserving upstream payload transmitted from a Tier-1 Edge Gateway
    to the Tier-2 Cloud Coordinator. Contains ONLY quantized histogram bin counts
    and aggregate prediction summaries — never raw flow records.
    """

    schema_version: str
    gateway_id: str
    gateway_profile: str
    round_id: int
    timestamp_utc: str
    local_sample_count: int
    quantization_bits: int
    feature_names: List[str]
    shared_bin_edges: Dict[str, List[float]]
    quantized_feature_histograms: Dict[str, List[int]]
    dequantization_scales: Dict[str, float]
    predicted_class_counts: Dict[str, int]
    suspicious_ratio: float
    mean_confidence: float
    contains_raw_records: bool = False

    def to_json_bytes(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode("utf-8")


class EdgeGateway:
    """
    Simulated Tier-1 Edge Gateway node.
    Matches `EdgeGateway` in the BCSE497J report Class Diagram.
    """

    def __init__(
        self,
        gateway_id: str,
        local_df: pd.DataFrame,
        classifier: EdgeClassifier,
        gateway_profile: str = "Heterogeneous IoT Subnet",
        quantization_bits: int = 8,
    ) -> None:
        self.gateway_id = str(gateway_id)
        self.local_df = local_df.copy().reset_index(drop=True)
        self.classifier = classifier
        self.gateway_profile = gateway_profile
        self.quantization_bits = 8 if quantization_bits not in (8, 16) else quantization_bits
        self.last_batch_output: Optional[BatchPredictionOutput] = None
        self.cloud_reachable: bool = True

    def run_local_detection(self) -> BatchPredictionOutput:
        """
        Execute local LightGBM classification on the gateway's local data partition.
        Continues to work autonomously on the last-known-good local model if the cloud
        coordinator is offline (FR-12 Graceful Degradation).
        """
        self.last_batch_output = self.classifier.predict_batch(self.local_df)
        return self.last_batch_output

    def run_offline(self) -> Dict[str, Any]:
        """Simulate autonomous offline classification when Tier-2 coordinator is unreachable."""
        self.cloud_reachable = False
        batch_out = self.run_local_detection()
        return {
            "gateway_id": self.gateway_id,
            "cloud_reachable": False,
            "mode": "Autonomous Local Inference (Last-Known-Good Model — FR-12)",
            "records_classified": batch_out.total_records,
            "suspicious_detected": batch_out.suspicious_count,
            "benign_detected": batch_out.benign_count,
            "mean_latency_ms": batch_out.mean_latency_per_record_ms,
        }

    def build_histogram_update(
        self,
        shared_bin_edges: Dict[str, List[float]],
        round_id: int = 1,
    ) -> GatewayUpdatePayload:
        """
        Compute local feature histograms over `shared_bin_edges`, quantize bin counts to
        `quantization_bits`-bit unsigned integers (uint8 or uint16) to minimize uplink
        bandwidth, and package into a `GatewayUpdatePayload` without any raw records.
        """
        if self.last_batch_output is None:
            self.run_local_detection()
        assert self.last_batch_output is not None

        X_reduced, _ = self.classifier.preprocess_to_reduced(self.local_df)
        max_int = (1 << self.quantization_bits) - 1

        quantized_hists: Dict[str, List[int]] = {}
        scales: Dict[str, float] = {}
        feat_names = list(shared_bin_edges.keys())

        for col in feat_names:
            edges = np.asarray(shared_bin_edges[col], dtype=np.float64)
            n_bins = len(edges) - 1
            if col not in X_reduced.columns or len(X_reduced) == 0:
                quantized_hists[col] = [0] * n_bins
                scales[col] = 1.0
                continue

            vals = pd.to_numeric(X_reduced[col], errors="coerce").dropna().to_numpy(dtype=np.float64)
            vals = np.clip(vals, edges[0], edges[-1])
            raw_counts, _ = np.histogram(vals, bins=edges)
            max_cnt = int(raw_counts.max()) if len(raw_counts) > 0 else 0
            if max_cnt == 0:
                quantized_hists[col] = [0] * n_bins
                scales[col] = 1.0
            else:
                scale = float(max_cnt) / float(max_int)
                q_counts = np.rint(raw_counts.astype(np.float64) / scale).astype(int)
                q_counts = np.clip(q_counts, 0, max_int)
                quantized_hists[col] = [int(x) for x in q_counts]
                scales[col] = round(scale, 6)

        total_recs = max(self.last_batch_output.total_records, 1)
        susp_ratio = round(self.last_batch_output.suspicious_count / total_recs, 4)
        mean_conf = round(
            float(np.mean([p.confidence for p in self.last_batch_output.predictions])),
            4,
        )

        return GatewayUpdatePayload(
            schema_version="1.0-quantized-histogram",
            gateway_id=self.gateway_id,
            gateway_profile=self.gateway_profile,
            round_id=int(round_id),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            local_sample_count=int(len(self.local_df)),
            quantization_bits=self.quantization_bits,
            feature_names=feat_names,
            shared_bin_edges={k: [round(float(e), 4) for e in v] for k, v in shared_bin_edges.items()},
            quantized_feature_histograms=quantized_hists,
            dequantization_scales=scales,
            predicted_class_counts=dict(self.last_batch_output.class_counts),
            suspicious_ratio=susp_ratio,
            mean_confidence=mean_conf,
            contains_raw_records=False,
        )
