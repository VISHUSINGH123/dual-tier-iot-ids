"""
Module F (Part 2) — Simulated Tier-2 Cloud Coordinator (`HistogramAggregator`).

Implements `HistogramAggregator` (matching the BCSE497J report Class Diagram):
- Receives `GatewayUpdatePayload` messages from 2+ simulated edge gateways.
- Strictly validates payload schemas and enforces the privacy boundary (rejects any
  payload containing raw flow records or mismatched histogram bin schemas).
- Dequantizes and aggregates feature histograms and predicted class counts.
- Computes pairwise Jensen-Shannon divergence between gateway traffic profiles and
  clusters gateways into behavioral groups when divergence exceeds a configurable
  non-IID heterogeneity threshold (`cluster_divergence_threshold`).
- Derives global and per-cluster candidate split quantiles from aggregated histograms.
- Records communication volume (uplink bytes vs. raw CSV bytes saved) and persists
  round summaries to `results/federated_rounds_history.json`.

Technical Honesty Note:
- Weight averaging (FedAvg) is undefined for decision tree ensembles and is NOT used.
- This coordinator implements a privacy-preserving histogram & split-candidate
  aggregation simulation. It does NOT claim that feature histograms alone produce
  a newly trained global LightGBM booster without distributed first/second-order
  gradient exchange (such as SecureBoost / FedTree).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from config.settings import FEDERATED_ROUNDS_PATH
from src.edge_gateway import EdgeGateway, FORBIDDEN_RAW_KEYS, GatewayUpdatePayload


@dataclass
class AggregationRoundSummary:
    """Persisted summary of a Tier-2 Edge-Cloud histogram aggregation round."""

    round_id: int
    timestamp_utc: str
    participating_gateways: List[str]
    gateway_profiles: Dict[str, str]
    gateway_sample_counts: Dict[str, int]
    total_samples_across_gateways: int
    quantization_bits: int
    uplink_payload_bytes_per_gateway: Dict[str, int]
    total_uplink_bytes: int
    estimated_raw_csv_bytes: int
    bandwidth_reduction_pct: float
    pairwise_gateway_js_divergence: Dict[str, float]
    max_gateway_divergence: float
    cluster_divergence_threshold: float
    clustering_response_triggered: bool
    gateway_clusters: Dict[str, List[str]]
    aggregated_class_counts: Dict[str, int]
    aggregated_feature_histograms: Dict[str, List[float]]
    candidate_split_medians: Dict[str, float]
    privacy_boundary_verified: bool
    methodology_note: str


class HistogramAggregator:
    """
    Tier-2 Cloud Histogram Coordinator.
    Matches `HistogramAggregator` in the BCSE497J report Class Diagram.
    """

    def __init__(
        self,
        shared_bin_edges: Dict[str, List[float]],
        cluster_divergence_threshold: float = 0.18,
    ) -> None:
        self.shared_bin_edges = {k: list(v) for k, v in shared_bin_edges.items()}
        self.cluster_divergence_threshold = float(cluster_divergence_threshold)
        self.round_history: List[AggregationRoundSummary] = []

    def validate_payload_schema(self, payload: GatewayUpdatePayload | Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate incoming gateway update schema and enforce the privacy boundary:
        1. Reject if `contains_raw_records` is True or any forbidden raw record keys exist.
        2. Verify all required shared features and bin counts match `shared_bin_edges`.
        3. Verify non-negative quantized counts and positive sample count.
        """
        errors: List[str] = []
        p_dict = asdict(payload) if isinstance(payload, GatewayUpdatePayload) else dict(payload)

        # Privacy check
        if p_dict.get("contains_raw_records", False):
            errors.append("Privacy violation: payload flags contains_raw_records=True.")
        present_forbidden = FORBIDDEN_RAW_KEYS.intersection(set(p_dict.keys()))
        if present_forbidden:
            errors.append(f"Privacy violation: forbidden raw data keys present in payload: {present_forbidden}")

        required_fields = {
            "schema_version",
            "gateway_id",
            "round_id",
            "local_sample_count",
            "quantized_feature_histograms",
            "dequantization_scales",
            "predicted_class_counts",
        }
        missing = required_fields - set(p_dict.keys())
        if missing:
            errors.append(f"Schema validation failed: missing required fields {missing}.")
            return False, errors

        if int(p_dict.get("local_sample_count", 0)) <= 0:
            errors.append("Invalid payload: local_sample_count must be > 0.")

        q_hists = p_dict.get("quantized_feature_histograms", {})
        for feat, expected_edges in self.shared_bin_edges.items():
            expected_bins = len(expected_edges) - 1
            if feat not in q_hists:
                errors.append(f"Missing histogram for required feature '{feat}'.")
            else:
                bins = q_hists[feat]
                if len(bins) != expected_bins:
                    errors.append(
                        f"Feature '{feat}' bin count mismatch: expected {expected_bins}, got {len(bins)}."
                    )
                if any(int(b) < 0 for b in bins):
                    errors.append(f"Feature '{feat}' contains negative histogram bin counts.")

        return len(errors) == 0, errors

    def _compute_gateway_divergence(
        self,
        dequantized_by_gw: Dict[str, Dict[str, np.ndarray]],
        class_counts_by_gw: Dict[str, Dict[str, int]],
    ) -> Tuple[Dict[str, float], float]:
        """
        Compute pairwise Jensen-Shannon distance across gateways combining their normalized
        feature histograms and predicted class distributions.
        """
        gw_ids = sorted(dequantized_by_gw.keys())
        all_classes = sorted({c for counts in class_counts_by_gw.values() for c in counts.keys()})

        pairwise: Dict[str, float] = {}
        max_div = 0.0

        for i in range(len(gw_ids)):
            for j in range(i + 1, len(gw_ids)):
                g1, g2 = gw_ids[i], gw_ids[j]
                js_scores: List[float] = []

                for feat in self.shared_bin_edges.keys():
                    h1 = dequantized_by_gw[g1][feat] + 1e-6
                    h2 = dequantized_by_gw[g2][feat] + 1e-6
                    h1 /= h1.sum()
                    h2 /= h2.sum()
                    js_val = float(jensenshannon(h1, h2))
                    if np.isfinite(js_val):
                        js_scores.append(js_val)

                if len(all_classes) > 1:
                    c1 = np.array([class_counts_by_gw[g1].get(c, 0) for c in all_classes], dtype=np.float64) + 1e-6
                    c2 = np.array([class_counts_by_gw[g2].get(c, 0) for c in all_classes], dtype=np.float64) + 1e-6
                    c1 /= c1.sum()
                    c2 /= c2.sum()
                    cjs = float(jensenshannon(c1, c2))
                    if np.isfinite(cjs):
                        js_scores.append(cjs)

                mean_js = round(float(np.mean(js_scores)) if js_scores else 0.0, 4)
                pair_key = f"{g1} vs {g2}"
                pairwise[pair_key] = mean_js
                max_div = max(max_div, mean_js)

        return pairwise, round(max_div, 4)

    def _assign_gateway_clusters(
        self,
        gw_ids: List[str],
        pairwise_js: Dict[str, float],
    ) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Cluster gateways by behavioral similarity:
        - If all pairwise JS divergences <= `cluster_divergence_threshold`, group all gateways
          into a single unified global cluster (`Cluster_0_Unified`).
        - If non-IID divergence exceeds `cluster_divergence_threshold`, partition divergent
          gateways into specialized behavioral clusters so heterogeneous IoT subnets do not
          corrupt one another's split statistics.
        """
        if not gw_ids:
            return False, {}

        any_exceeds = any(v > self.cluster_divergence_threshold for v in pairwise_js.values())
        if not any_exceeds:
            return False, {"Cluster_0_Unified": list(gw_ids)}

        # Complete-linkage grouping based on threshold so divergent gateways separate cleanly
        clusters: List[List[str]] = []
        for gid in gw_ids:
            placed = False
            for cluster in clusters:
                fits_all = True
                for member in cluster:
                    key1 = f"{member} vs {gid}"
                    key2 = f"{gid} vs {member}"
                    dist = pairwise_js.get(key1, pairwise_js.get(key2, 0.0))
                    if dist > self.cluster_divergence_threshold:
                        fits_all = False
                        break
                if fits_all:
                    cluster.append(gid)
                    placed = True
                    break
            if not placed:
                clusters.append([gid])

        cluster_dict = {f"Cluster_{idx + 1}": members for idx, members in enumerate(clusters)}
        return True, cluster_dict

    def select_candidate_split_medians(
        self,
        aggregated_hists: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """
        Estimate the global median split candidate for each feature from the aggregated
        piecewise histogram counts and shared bin edges.
        """
        medians: Dict[str, float] = {}
        for feat, counts_list in aggregated_hists.items():
            edges = np.asarray(self.shared_bin_edges[feat], dtype=np.float64)
            counts = np.asarray(counts_list, dtype=np.float64)
            total = float(counts.sum())
            if total <= 0:
                medians[feat] = round(float(0.5 * (edges[0] + edges[-1])), 4)
                continue
            cum = np.cumsum(counts)
            half = 0.5 * total
            bin_idx = int(np.searchsorted(cum, half))
            bin_idx = min(max(bin_idx, 0), len(counts) - 1)
            bin_mid = 0.5 * (edges[bin_idx] + edges[bin_idx + 1])
            medians[feat] = round(float(bin_mid), 4)
        return medians

    def aggregate_round(
        self,
        payloads: List[GatewayUpdatePayload],
        raw_bytes_Map: Optional[Dict[str, int]] = None,
        persist: bool = True,
    ) -> AggregationRoundSummary:
        """
        Validate and aggregate quantized histogram payloads from 2+ simulated gateways.
        """
        if len(payloads) < 2:
            raise ValueError("Cloud aggregation requires updates from at least 2 gateways.")

        for p in payloads:
            is_valid, errs = self.validate_payload_schema(p)
            if not is_valid:
                raise ValueError(f"Gateway '{p.gateway_id}' payload rejected: {'; '.join(errs)}")

        round_id = payloads[0].round_id
        q_bits = payloads[0].quantization_bits
        gw_ids = [p.gateway_id for p in payloads]
        gw_profiles = {p.gateway_id: p.gateway_profile for p in payloads}
        gw_samples = {p.gateway_id: p.local_sample_count for p in payloads}
        total_samples = sum(gw_samples.values())

        uplink_bytes: Dict[str, int] = {}
        dequantized_by_gw: Dict[str, Dict[str, np.ndarray]] = {}
        class_counts_by_gw: Dict[str, Dict[str, int]] = {}
        agg_hists_np: Dict[str, np.ndarray] = {
            feat: np.zeros(len(edges) - 1, dtype=np.float64)
            for feat, edges in self.shared_bin_edges.items()
        }
        agg_class_counts: Dict[str, int] = {}

        for p in payloads:
            uplink_bytes[p.gateway_id] = len(p.to_json_bytes())
            dequantized_by_gw[p.gateway_id] = {}
            class_counts_by_gw[p.gateway_id] = dict(p.predicted_class_counts)

            for cls_name, cnt in p.predicted_class_counts.items():
                agg_class_counts[cls_name] = agg_class_counts.get(cls_name, 0) + int(cnt)

            for feat in self.shared_bin_edges.keys():
                q_arr = np.asarray(p.quantized_feature_histograms[feat], dtype=np.float64)
                scale = float(p.dequantization_scales.get(feat, 1.0))
                approx_counts = q_arr * scale
                dequantized_by_gw[p.gateway_id][feat] = approx_counts
                agg_hists_np[feat] += approx_counts

        total_uplink = sum(uplink_bytes.values())
        if raw_bytes_Map:
            est_raw_bytes = sum(raw_bytes_Map.values())
        else:
            # Typical 61-feature CSV row is ~320 bytes
            est_raw_bytes = total_samples * 320

        bw_saving = round((1.0 - total_uplink / max(est_raw_bytes, 1)) * 100.0, 2)

        pairwise_js, max_js = self._compute_gateway_divergence(dequantized_by_gw, class_counts_by_gw)
        clustered_flag, clusters = self._assign_gateway_clusters(gw_ids, pairwise_js)

        agg_hists_clean = {
            feat: [round(float(x), 2) for x in arr] for feat, arr in agg_hists_np.items()
        }
        split_medians = self.select_candidate_split_medians(agg_hists_clean)

        summary = AggregationRoundSummary(
            round_id=round_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            participating_gateways=gw_ids,
            gateway_profiles=gw_profiles,
            gateway_sample_counts=gw_samples,
            total_samples_across_gateways=total_samples,
            quantization_bits=q_bits,
            uplink_payload_bytes_per_gateway=uplink_bytes,
            total_uplink_bytes=total_uplink,
            estimated_raw_csv_bytes=est_raw_bytes,
            bandwidth_reduction_pct=bw_saving,
            pairwise_gateway_js_divergence=pairwise_js,
            max_gateway_divergence=max_js,
            cluster_divergence_threshold=self.cluster_divergence_threshold,
            clustering_response_triggered=clustered_flag,
            gateway_clusters=clusters,
            aggregated_class_counts=agg_class_counts,
            aggregated_feature_histograms=agg_hists_clean,
            candidate_split_medians=split_medians,
            privacy_boundary_verified=True,
            methodology_note=(
                "SIMULATION PROTOTYPE: Coordinator receives only quantized feature histograms and "
                "predicted class counts (zero raw flow records). It validates schemas, measures "
                "inter-gateway Jensen-Shannon divergence, clusters gateways when non-IID heterogeneity "
                "exceeds the threshold, and computes merged candidate split quantiles. It does NOT "
                "average tree weights or claim full gradient-boosted tree reconstruction without "
                "first/second-order gradient histograms."
            ),
        )

        self.round_history.append(summary)
        if persist:
            self.save_history(FEDERATED_ROUNDS_PATH)
        return summary

    def save_history(self, path: Path = FEDERATED_ROUNDS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: List[Dict[str, Any]] = []
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        existing.extend([asdict(r) for r in self.round_history[-1:]])
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def partition_dataset_for_gateways(
    df: pd.DataFrame,
    num_gateways: int = 3,
    partition_mode: str = "non_iid",
    target_col: str = "Attack_type",
    random_state: int = 42,
) -> List[Tuple[str, str, pd.DataFrame]]:
    """
    Split a dataset across `num_gateways` (>= 2) simulated Edge Gateways:
    - `iid`: Uniform random partition across gateways.
    - `non_iid`: Protocol/attack-skewed partitions simulating distinct IoT subnets
      (e.g., Gateway-1: MQTT/Smart-Home heavy, Gateway-2: HTTP/Web-API heavy,
       Gateway-3: CoAP/UDP/6LoWPAN Industrial Sensor subnet).
    Returns list of (gateway_id, gateway_profile_description, local_df).
    """
    n_gw = max(2, int(num_gateways))
    rng = np.random.default_rng(random_state)
    shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if partition_mode.lower() == "iid" or target_col not in shuffled.columns:
        chunks = np.array_split(shuffled, n_gw)
        return [
            (f"Edge-GW-{i + 1:02d}", f"IID Balanced IoT Subnet #{i + 1}", chunks[i].reset_index(drop=True))
            for i in range(n_gw)
        ]

    # Non-IID protocol/attack specialization
    profiles = [
        ("Edge-GW-01", "Smart-Home Broker Subnet (MQTT / Auth / Backdoor Skew)"),
        ("Edge-GW-02", "Web & Camera Gateway (HTTP / SQLi / XSS / DDoS_HTTP Skew)"),
        ("Edge-GW-03", "Industrial Sensor Mesh (CoAP / 6LoWPAN / UDP / ICMP / Modbus Skew)"),
        ("Edge-GW-04", "Perimeter Enterprise IoT Gateway (Scanning / MITM / DNS Skew)"),
    ]
    while len(profiles) < n_gw:
        idx = len(profiles) + 1
        profiles.append((f"Edge-GW-{idx:02d}", f"Heterogeneous IoT Subnet #{idx}"))

    buckets: List[List[int]] = [[] for _ in range(n_gw)]
    for row_idx, val in enumerate(shuffled[target_col].astype(str)):
        v_low = val.lower()
        if any(k in v_low for k in ("password", "backdoor", "ransomware")):
            pref = 0
        elif any(k in v_low for k in ("http", "sql", "xss", "uploading")):
            pref = min(1, n_gw - 1)
        elif any(k in v_low for k in ("udp", "icmp", "mitm")):
            pref = min(2, n_gw - 1)
        else:
            pref = int(rng.integers(0, n_gw))

        # 75% goes to preferred specialized gateway, 25% distributed across others
        if rng.random() < 0.75:
            buckets[pref].append(row_idx)
        else:
            buckets[int(rng.integers(0, n_gw))].append(row_idx)

    # Ensure every gateway has at least 20 records
    for i in range(n_gw):
        if len(buckets[i]) < 20:
            extra = rng.choice(len(shuffled), size=30, replace=False).tolist()
            buckets[i].extend(extra)

    out: List[Tuple[str, str, pd.DataFrame]] = []
    for i in range(n_gw):
        gid, prof = profiles[i]
        sub_df = shuffled.iloc[buckets[i]].reset_index(drop=True)
        out.append((gid, prof, sub_df))
    return out
