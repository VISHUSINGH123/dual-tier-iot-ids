"""
Page 5 — Simulated Edge–Cloud Aggregation (`iot_ids.federation.ClusteredHistogramAggregator`
and `src.edge_gateway.EdgeGateway` / `src.cloud_aggregator.HistogramAggregator`).
"""
from __future__ import annotations

from dataclasses import asdict
import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from config.settings import DRIFT_REFERENCE_PATH, FEDERATED_ROUNDS_PATH, REAL_EDGE_IIOTSET_CSV
from src.cloud_aggregator import HistogramAggregator, partition_dataset_for_gateways
from src.drift_monitor import DriftMonitor
from src.edge_gateway import EdgeGateway
from src.framework_bridge import (
    FEATURE_NAMES,
    ClusteredHistogramAggregator,
    quantise_histogram,
)
from src.inference import EdgeClassifier
from src.ui_theme import render_header

st.set_page_config(page_title="Edge-Cloud Simulation | IoT IDS", layout="wide")

render_header(
    title=" Module F — Simulated Dual-Tier Edge–Cloud Histogram Aggregation",
    subtitle=(
        "Simulate privacy-preserving 16-bit quantized gradient/hessian split aggregation "
        "(`iot_ids.federation.ClusteredHistogramAggregator`) and multi-gateway non-IID clustering."
    ),
    badge_text="SIMULATION PROTOTYPE",
)

st.markdown(
    """
    <div class="sim-banner">
        <strong>SIMULATION & TECHNICAL HONESTY NOTICE:</strong>
        Decision tree ensembles cannot be updated via neural-network weight averaging (FedAvg).
        Tier-1 gateways transmit <strong>only quantized histogram statistics</strong> (never raw flow records or IPs).
        Tab 1 runs <code>iot_ids.federation.ClusteredHistogramAggregator</code> (16-bit quantized gradient &amp; hessian
        histograms with L2-regularized GBDT split gain <code>best_split()</code>). Tab 2 runs multi-gateway
        feature/class histogram aggregation, inter-gateway Jensen-Shannon divergence clustering, and
        autonomous offline fallback (FR-12). Neither claims physical multi-Raspberry-Pi deployment or
        automatic full global LightGBM booster reconstruction without iterative tree-building rounds.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_core_fed, tab_multi_gw, tab_offline = st.tabs(
    [
        "1. Core Federated GBDT Split-Gain Aggregation (`iot_ids.federation`)",
        "2. Multi-Gateway Non-IID Clustering & Bandwidth Simulation",
        "3. Autonomous Offline Gateway Operation (FR-12 Graceful Degradation)",
    ]
)

# ==============================================================================
# TAB 1: CORE iot_ids.federation.ClusteredHistogramAggregator
# ==============================================================================
with tab_core_fed:
    st.subheader(" `iot_ids.federation.ClusteredHistogramAggregator` (Quantized Gradient/Hessian Split Selection)")
    st.caption(
        "Derives sample-level binary cross-entropy gradients $g_i$ and hessians $h_i$ from genuine Edge-IIoTset traffic partitions, "
        "quantizes them with 16-bit or 8-bit precision across candidate split bins, and computes the L2-regularized split point via `best_split()`."
    )
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        feature_candidates = [
            "tcp.len",
            "frame.len",
            "tcp.flags",
            "http.content_length",
            "udp.port",
            "dns.qry.name.len",
            "icmp.seq_le",
        ]
        sel_feat = st.selectbox("Candidate Real Feature for Split Evaluation", options=feature_candidates, index=0)
    with fcol2:
        q_bits = st.selectbox("Histogram Quantization Precision (bits)", options=[16, 8], index=0)
    with fcol3:
        l2_reg = st.number_input("GBDT L2 Regularization (λ)", min_value=0.1, max_value=10.0, value=1.0, step=0.5)

    if st.button("▶ Run Clustered Gradient/Hessian Aggregation & Compute `best_split`", type="primary"):
        if not REAL_EDGE_IIOTSET_CSV.exists():
            st.error(f"Real dataset missing at {REAL_EDGE_IIOTSET_CSV}")
        else:
            df_all = pd.read_csv(REAL_EDGE_IIOTSET_CSV, nrows=3000, low_memory=False)
            if sel_feat not in df_all.columns:
                st.error(f"Feature `{sel_feat}` not found in real dataset.")
            else:
                parts = partition_dataset_for_gateways(df_all, num_gateways=4, partition_mode="non_iid")
                feat_vals = df_all[sel_feat].dropna().values
                p95 = float(np.percentile(feat_vals, 95))
                min_v = float(feat_vals.min())
                bin_edges = np.linspace(min_v, max(p95, min_v + 4.0), 6).tolist()

                reports = []
                for gid, prof, g_df in parts:
                    y = (g_df["Attack_type"] != "Normal").astype(float).values
                    p = np.full_like(y, 0.5)
                    g = p - y
                    h = np.maximum(p * (1.0 - p), 0.05)

                    val_arr = g_df[sel_feat].fillna(0).values
                    bin_idx = np.digitize(val_arr, bin_edges) - 1
                    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)

                    g_hist = [float(np.sum(g[bin_idx == b])) for b in range(len(bin_edges) - 1)]
                    h_hist = [float(np.sum(h[bin_idx == b])) for b in range(len(bin_edges) - 1)]
                    cluster = "cluster_brokers" if ("01" in gid or "02" in gid) else "cluster_industrial"

                    rep = quantise_histogram(
                        client_id=gid,
                        cluster_id=cluster,
                        feature=sel_feat,
                        bin_edges=bin_edges,
                        gradient=g_hist,
                        hessian=h_hist,
                        bits=int(q_bits),
                    )
                    reports.append(rep)

                aggregator = ClusteredHistogramAggregator()
                totals = aggregator.aggregate(reports)

                summary_rows = []
                for (cid, feat), agg_data in totals.items():
                    best_idx = aggregator.best_split(agg_data, l2=float(l2_reg))
                    edges = agg_data["bin_edges"]
                    split_threshold = edges[best_idx]
                    summary_rows.append(
                        {
                            "Cluster_ID": cid,
                            "Feature": feat,
                            "Participating_Gateways": ", ".join(sorted(agg_data["clients"])),
                            "Quantization_Bits": int(q_bits),
                            "Best_Split_Bin_Index": best_idx,
                            "Selected_Split_Threshold": split_threshold,
                            "Dequantized_Gradients": [round(float(x), 3) for x in agg_data["gradient"]],
                            "Dequantized_Hessians": [round(float(x), 3) for x in agg_data["hessian"]],
                        }
                    )
                st.success("Aggregated genuine quantized gradient & hessian histograms per cluster from real data!")
                st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

                with st.expander(" Inspect Transmitted `QuantisedHistogram` Payload (Zero Raw Traffic Proof)"):
                    st.json(asdict(reports[0]))


# ==============================================================================
# TAB 2: MULTI-GATEWAY NON-IID CLUSTERING & BANDWIDTH SIMULATION
# ==============================================================================
with tab_multi_gw:
    st.subheader(" Multi-Gateway Partitioning, Non-IID JS Divergence & Uplink Bandwidth Savings")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        n_gateways = st.slider("Simulated Edge Gateways", min_value=2, max_value=5, value=3)
    with g2:
        part_mode = st.selectbox("Traffic Partition Mode", options=["non_iid", "iid"], index=0)
    with g3:
        gw_q_bits = st.selectbox("Gateway Bin Quantization", options=[8, 16], index=0)
    with g4:
        js_thresh = st.slider("Non-IID Clustering JS Threshold", min_value=0.05, max_value=0.50, value=0.16, step=0.01)

    if st.button(" Execute Multi-Gateway Aggregation Round", type="primary"):
        from config.settings import REAL_EDGE_IIOTSET_CSV

        if not REAL_EDGE_IIOTSET_CSV.exists():
            st.error(f"Real Edge-IIoTset dataset missing at `{REAL_EDGE_IIOTSET_CSV}`. Synthetic fallbacks are disabled.")
        else:
            try:
                clf = EdgeClassifier().load()
                monitor = DriftMonitor.load_reference(DRIFT_REFERENCE_PATH)
                df_all = pd.read_csv(REAL_EDGE_IIOTSET_CSV, low_memory=False)
                df_full = pd.concat(
                    [group.sample(n=min(len(group), 500), random_state=42) for _, group in df_all.groupby("Attack_type")],
                    ignore_index=True,
                )
                partitions = partition_dataset_for_gateways(
                    df_full,
                    num_gateways=int(n_gateways),
                    partition_mode=part_mode,
                )
                coord = HistogramAggregator(
                    shared_bin_edges=monitor.feature_bins,
                    cluster_divergence_threshold=float(js_thresh),
                )
                payloads = []
                raw_bytes_map = {}
                for gid, prof, g_df in partitions:
                    gw = EdgeGateway(
                        gateway_id=gid,
                        local_df=g_df,
                        classifier=clf,
                        gateway_profile=prof,
                        quantization_bits=int(gw_q_bits),
                    )
                    gw.run_local_detection()
                    p = gw.build_histogram_update(shared_bin_edges=monitor.feature_bins, round_id=len(coord.round_history) + 1)
                    payloads.append(p)
                    raw_bytes_map[gid] = len(g_df.to_csv(index=False).encode("utf-8"))

                round_res = coord.aggregate_round(payloads, raw_bytes_map=raw_bytes_map)
                st.session_state["last_multi_gw_round"] = (round_res, payloads[0])
            except Exception as exc:
                st.error(f"Simulation error: {exc}")

    if "last_multi_gw_round" in st.session_state:
        r_sum, sample_payload = st.session_state["last_multi_gw_round"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Flows Across Gateways", f"{r_sum.total_samples_across_gateways:,}")
        k2.metric("Quantized Histogram Uplink", f"{r_sum.total_uplink_bytes:,} B", delta=f"-{r_sum.bandwidth_reduction_pct}% vs Raw CSV")
        k3.metric("Max Pairwise JS Divergence", f"{r_sum.max_gateway_divergence:.4f}", delta=f"Threshold: {r_sum.cluster_divergence_threshold}")
        k4.metric("Behavioral Clusters Formed", f"{len(r_sum.gateway_clusters)}")

        st.info(r_sum.methodology_note)

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("##### Gateway Cluster Assignments & Pairwise Divergence")
            st.json(
                {
                    "gateway_clusters": r_sum.gateway_clusters,
                    "pairwise_js_divergence": r_sum.pairwise_gateway_js_divergence,
                    "uplink_bytes_per_gateway": r_sum.uplink_payload_bytes_per_gateway,
                }
            )
        with c_right:
            st.markdown("##### Aggregated Class Distribution Across Gateways")
            cls_df = pd.DataFrame(
                [{"Class": k, "Count": v} for k, v in r_sum.aggregated_class_counts.items()]
            ).sort_values("Count", ascending=False)
            fig_c = px.bar(cls_df, x="Class", y="Count", color="Count", template="plotly_dark")
            fig_c.update_layout(height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_c, width="stretch")


# ==============================================================================
# TAB 3: AUTONOMOUS OFFLINE GATEWAY OPERATION (FR-12)
# ==============================================================================
with tab_offline:
    st.subheader(" FR-12 Graceful Degradation — Autonomous Local Detection When Cloud is Unreachable")
    if st.button("▶ Simulate Tier-2 Cloud Disconnection (`EdgeGateway.run_offline()`)"):
        from config.settings import REAL_EDGE_IIOTSET_CSV

        if not REAL_EDGE_IIOTSET_CSV.exists():
            st.error(f"Real dataset missing at {REAL_EDGE_IIOTSET_CSV}")
        else:
            clf = EdgeClassifier().load()
            df_all = pd.read_csv(REAL_EDGE_IIOTSET_CSV, nrows=1000, low_memory=False)
            df_sub = df_all.sample(n=min(150, len(df_all)), random_state=21).reset_index(drop=True)
            gw = EdgeGateway("Edge-GW-01", df_sub, clf)
            offline_report = gw.run_offline()
            st.success("Gateway continued autonomous local classification on last-known-good model without interruption!")
            st.json(offline_report)
