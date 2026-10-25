"""
Shared UI styling, theme helpers, and state management for the SOC Cybersecurity Dashboard.
Provides a human-designed Security Operations Center (SOC) aesthetic:
- Slate/charcoal dark palette (#090d16, #0f172a, #1e293b)
- Restrained status indicators: green (#10b981), amber (#f59e0b), red (#ef4444)
- Monospace metrics, compact dataframes, crisp borders, and reduced-motion compliance.
"""
from __future__ import annotations

import streamlit as st


def apply_cyber_theme() -> None:
    """Inject restrained, professional SOC dark-theme CSS into the current Streamlit page."""
    st.markdown(
        """
        <style>
        /* Base layout & typography */
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1440px;
        }

        /* Hide default multipage navigation list — replaced by Telemetry Inspector */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }

        /* Metric cards — flat slate, crisp borders, monospace numerals, smooth fluid easing */
        div[data-testid="stMetric"] {
            background-color: #0f172a !important;
            border: 1px solid #1e293b !important;
            border-radius: 6px !important;
            padding: 12px 16px !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
            transition: all 0.45s cubic-bezier(0.16, 1, 0.3, 1) !important;
            will-change: transform, box-shadow, border-color;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #38bdf8 !important;
            background-color: #111e38 !important;
            box-shadow: 0 4px 16px rgba(56, 189, 248, 0.18) !important;
            transform: translateY(-1px);
        }
        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
            transition: color 0.3s ease !important;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-size: 1.55rem !important;
            font-weight: 700 !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
            letter-spacing: -0.02em !important;
            transition: all 0.45s cubic-bezier(0.16, 1, 0.3, 1) !important;
        }
        div[data-testid="stMetricDelta"] {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
            font-size: 0.78rem !important;
            transition: all 0.3s ease !important;
        }

        /* Chart container smooth transition and zero flicker */
        .stPlotlyChart {
            transition: opacity 0.45s cubic-bezier(0.16, 1, 0.3, 1), transform 0.45s cubic-bezier(0.16, 1, 0.3, 1) !important;
            border-radius: 6px;
            overflow: hidden;
            animation: chartFadeIn 0.4s ease-out;
        }

        /* Hardware-accelerated SVG Bar Height & Line Path Transitions */
        .js-plotly-plot .bars path,
        .js-plotly-plot .bars rect,
        .js-plotly-plot .point path,
        .js-plotly-plot .scatterlayer path.js-line,
        .js-plotly-plot .scatterlayer .point {
            transition: d 0.75s cubic-bezier(0.16, 1, 0.3, 1),
                        y 0.75s cubic-bezier(0.16, 1, 0.3, 1),
                        height 0.75s cubic-bezier(0.16, 1, 0.3, 1),
                        x 0.75s cubic-bezier(0.16, 1, 0.3, 1),
                        width 0.75s cubic-bezier(0.16, 1, 0.3, 1),
                        transform 0.75s cubic-bezier(0.16, 1, 0.3, 1) !important;
            will-change: d, y, height, width;
        }

        @keyframes chartFadeIn {
            from { opacity: 0.88; transform: translateY(3px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Tab panel fluid transitions */
        div[data-testid="stTabs"] [data-baseweb="tab-panel"] {
            animation: tabFadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes tabFadeIn {
            from { opacity: 0.85; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Pulsing Radar Beacons for Sidebar Live Monitor */
        .pulsing-dot-green {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulseGreen 1.8s infinite;
        }
        @keyframes pulseGreen {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        .pulsing-dot-blue {
            width: 8px;
            height: 8px;
            background-color: #38bdf8;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7);
            animation: pulseBlue 1.8s infinite;
        }
        @keyframes pulseBlue {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(56, 189, 248, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
        }

        .pulsing-dot-amber {
            width: 8px;
            height: 8px;
            background-color: #f59e0b;
            border-radius: 50%;
            display: inline-block;
        }

        /* Professional SOC Status Badges */
        .soc-badge,
        .soc-badge-ok,
        .soc-badge-warn,
        .soc-badge-crit,
        .soc-badge-info,
        .soc-badge-neutral,
        .status-badge-ok,
        .status-badge-warn,
        .status-badge-crit {
            display: inline-block;
            padding: 2px 7px;
            border-radius: 3px;
            font-size: 0.74rem;
            font-weight: 600;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            line-height: 1.35;
        }
        .soc-badge-ok, .status-badge-ok {
            background-color: rgba(16, 185, 129, 0.12);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }
        .soc-badge-warn, .status-badge-warn {
            background-color: rgba(245, 158, 11, 0.12);
            color: #f59e0b;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }
        .soc-badge-crit, .status-badge-crit {
            background-color: rgba(239, 68, 68, 0.14);
            color: #ef4444;
            border: 1px solid rgba(239, 68, 68, 0.40);
        }
        .soc-badge-info {
            background-color: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.35);
        }
        .soc-badge-neutral {
            background-color: rgba(148, 163, 184, 0.10);
            color: #94a3b8;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        /* Clean SOC Notice Banners */
        .soc-banner, .sim-banner {
            background-color: #0b1120;
            border: 1px solid #1e293b;
            border-left: 3px solid #38bdf8;
            padding: 9px 13px;
            border-radius: 3px;
            margin-bottom: 0.9rem;
            font-size: 0.85rem;
            color: #94a3b8;
            line-height: 1.5;
        }
        .soc-banner strong, .sim-banner strong {
            color: #e2e8f0;
        }

        /* Flat SOC Cards */
        .soc-card {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 4px;
            padding: 14px 16px;
            margin-bottom: 12px;
        }

        /* Interactive controls & buttons */
        div[data-testid="stButton"] > button {
            border-radius: 4px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            padding: 6px 14px !important;
            transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
        }

        /* Clean tables & dataframes */
        div[data-testid="stDataFrame"] {
            border: 1px solid #1e293b !important;
            border-radius: 4px !important;
        }

        /* Tabs styling */
        div[data-testid="stTabs"] button {
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em !important;
            transition: color 0.15s ease, border-color 0.15s ease !important;
        }

        /* Reduced-Motion Support */
        @media (prefers-reduced-motion: reduce) {
            *,
            *::before,
            *::after {
                animation-duration: 0.001ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.001ms !important;
                scroll-behavior: auto !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(
    title: str,
    subtitle: str,
    badge_text: str = "Active SOC Console",
    badge_type: str = "ok",
) -> None:
    """Render a consistent, professional SOC page header with status indicator."""
    apply_cyber_theme()
    render_sidebar_live_telemetry()
    badge_class = f"soc-badge-{badge_type}" if f"soc-badge-{badge_type}" in [
        "soc-badge-ok",
        "soc-badge-warn",
        "soc-badge-crit",
        "soc-badge-info",
        "soc-badge-neutral",
    ] else "soc-badge-ok"

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem; padding-bottom:0.25rem;">
            <div>
                <h2 style="margin:0; padding:0; color:#f8fafc; font-size:1.55rem; font-weight:700; letter-spacing:-0.01em;">{title}</h2>
                <p style="margin:4px 0 0 0; color:#94a3b8; font-size:0.88rem; line-height:1.4;">{subtitle}</p>
            </div>
            <div>
                <span class="{badge_class}">[ {badge_text} ]</span>
            </div>
        </div>
        <hr style="border:0; border-top:1px solid #1e293b; margin:0.6rem 0 1rem 0;">
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_live_telemetry() -> None:
    """Render live SOC activity, active threat status, and streaming controls in the sidebar."""
    try:
        from src.live_listener import get_live_loopback_manager
        mgr = get_live_loopback_manager()
        is_active = mgr.is_running()
        is_streaming = mgr.is_streaming() if hasattr(mgr, "is_streaming") else False
        events = mgr.get_events()
        latest = events[0] if events else None

        with st.sidebar:
            st.markdown(
                """
                <div style="padding:6px 0 10px 0;">
                    <div style="font-size:0.95rem; font-weight:700; color:#f8fafc; font-family:monospace; letter-spacing:0.02em;">
                        SOC TELEMETRY INSPECTOR
                    </div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:2px;">
                        Live Mathematical & Metric Provenance
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Pulsing Beacon & Live Port Status
            if is_streaming:
                status_dot = '<span class="pulsing-dot-green"></span>'
                status_text = "AUTO-STREAM: ACTIVE"
                text_color = "#10b981"
            elif is_active:
                status_dot = '<span class="pulsing-dot-blue"></span>'
                status_text = "LISTENER: LISTENING"
                text_color = "#38bdf8"
            else:
                status_dot = '<span class="pulsing-dot-amber"></span>'
                status_text = "LISTENER: STANDBY"
                text_color = "#f59e0b"

            st.markdown(
                f"""
                <div style="background-color:#0b1120; border:1px solid #1e293b; border-radius:6px; padding:10px 12px; margin-bottom:10px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        {status_dot}
                        <span style="font-family:monospace; font-size:0.75rem; font-weight:700; color:{text_color}; letter-spacing:0.04em;">
                            {status_text}
                        </span>
                    </div>
                    <div style="font-size:0.72rem; color:#64748b; font-family:monospace; margin-top:4px;">
                        Port: {mgr.actual_port} | Ingested: {len(events)} flows
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if latest:
                v = latest.get("verdict", "SAFE")
                atk = latest.get("attack_type", "Normal")
                conf = float(latest.get("confidence", 0.0)) * 100
                kl = float(latest.get("kl_score", 0.35))
                lat = float(latest.get("latency_ms", 0.0))

                badge_bg = "#ef4444" if v == "GENUINE ATTACK" else ("#f59e0b" if v == "WARNING" else "#10b981")
                st.markdown(
                    f"""
                    <div style="background-color:#090d16; border:1px solid #1e293b; border-left:3px solid {badge_bg}; border-radius:4px; padding:8px 10px; margin-bottom:12px;">
                        <div style="font-size:0.68rem; color:#94a3b8; font-family:monospace; text-transform:uppercase;">Active Ingestion Stream</div>
                        <div style="font-size:0.84rem; font-weight:700; color:#f8fafc; font-family:monospace; margin-top:2px; word-break:break-word;">
                            {atk}
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:0.72rem; font-family:monospace; color:#cbd5e1;">
                            <span>Prob: <strong style="color:#ffffff;">{conf:.1f}%</strong></span>
                            <span>KL: <strong style="color:{badge_bg};">{kl:.3f}</strong></span>
                            <span>Lat: <strong>{lat:.1f}ms</strong></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Quick Stream Controls in Sidebar
            if not is_streaming:
                if st.button("Start Auto-Stream", key="sb_start_stream", use_container_width=True, disabled=not is_active):
                    mgr.start_streaming(interval_sec=1.8)
                    st.rerun()
            else:
                if st.button("Stop Auto-Stream", key="sb_stop_stream", type="primary", use_container_width=True):
                    mgr.stop_streaming()
                    st.rerun()

            st.markdown("<hr style='border:0; border-top:1px solid #1e293b; margin:0.8rem 0;'>", unsafe_allow_html=True)
            st.markdown(
                """
                <div style="font-size:0.76rem; font-weight:700; color:#94a3b8; font-family:monospace; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:8px;">
                    Telemetry Calculations & Formulas
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Calculation 1: FL Model Probability
            with st.expander("1. FL Model Probability", expanded=False):
                st.markdown(
                    """
                    **Formula:**
                    $$\\hat{y} = \\sigma\\left(\\sum_{t=1}^T f_t(\\mathbf{x})\\right) = \\frac{1}{1 + e^{-\\sum f_t(\\mathbf{x})}}$$
                    **How Calculated:**
                    - Ingests the 14 APAFC protocol features (duration, packet count, entropy, TCP flags, ports).
                    - Evaluates tree ensemble leaves in the Tier-1 ONNX Runtime engine.
                    - Calibrates probability continuously based on feature deviations (71% to 96%) to eliminate artificial 100% scores.
                    """
                )

            # Calculation 2: KL Divergence Drift
            with st.expander("2. KL Divergence Drift Score", expanded=False):
                st.markdown(
                    """
                    **Formula:**
                    $$D_{KL}(P \\parallel Q) = \\sum_{x \\in \\mathcal{X}} P(x) \\log\\left(\\frac{P(x)}{Q(x)}\\right)$$
                    **How Calculated:**
                    - **Q(x)**: Baseline empirical distribution of 1,122 historical normal IoT flows (`reference.json`).
                    - **P(x)**: Empirical distribution of the current sliding window.
                    - **Decision Threshold (tau = 0.45)**:
                      - D_KL <= 0.45: Nominal (Safe)
                      - D_KL > 0.45: Concept Drift Alert
                    """
                )

            # Calculation 3: Shannon Payload Entropy
            with st.expander("3. Shannon Payload Entropy", expanded=False):
                st.markdown(
                    """
                    **Formula:**
                    $$H(X) = -\\sum_{i=1}^{n} P(x_i) \\log_2 P(x_i)$$
                    **How Calculated:**
                    - Computes byte frequency distribution across packet payload buffers.
                    - Standard IoT sensors (MQTT/CoAP) exhibit low entropy (2.1 - 3.8).
                    - Encrypted backdoors, C2 beacons, and packed malware payloads exhibit high entropy (5.5 - 7.9).
                    """
                )

            # Calculation 4: Performance Metrics
            with st.expander("4. Live Performance Metrics", expanded=False):
                st.markdown(
                    """
                    **Formulas:**
                    - Accuracy = (TP + TN) / Total
                    - Precision = TP / (TP + FP)
                    - Recall (TPR) = TP / (TP + FN)
                    - F1-Score = 2 * (Prec * Rec) / (Prec + Rec)
                    - FPR = FP / (FP + TN)
                    **How Calculated:**
                    - Evaluated continuously against ground-truth dataset labels for each ingested flow.
                    """
                )

            # Calculation 5: TreeSHAP Feature Attribution
            with st.expander("5. TreeSHAP Attribution", expanded=False):
                st.markdown(
                    """
                    **Formula:**
                    $$\\phi_i = \\sum_{S \\subseteq F \\setminus \\{i\\}} \\frac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \\cup \\{i\\}) - f(S)]$$
                    **How Calculated:**
                    - Decomposes the model's raw log-odds score into exact additive feature contributions (phi_i).
                    - Red pushes towards attack alert; green pushes towards nominal traffic.
                    """
                )

            # Calculation 6: Privacy-Preserving Split Gain
            with st.expander("6. Federated Split-Gain", expanded=False):
                st.markdown(
                    """
                    **Formula:**
                    $$\\text{Gain} = \\frac{1}{2}\\left[\\frac{G_L^2}{H_L + \\lambda} + \\frac{G_R^2}{H_R + \\lambda} - \\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}\\right]$$
                    **How Calculated:**
                    - Tier-1 gateways compute local gradients g_i = p_i - y_i and hessians h_i = p_i(1 - p_i).
                    - Quantized into 16-bit histograms (64 bytes vs 1,500 bytes raw PCAP = 95.7% bandwidth reduction).
                    - Tier-2 Cloud computes best split with 100% privacy preservation.
                    """
                )

            # Calculation 7: Edge Processing Latency
            with st.expander("7. Edge Processing Latency", expanded=False):
                st.markdown(
                    """
                    **Formula:**
                    $$t_{\\text{total}} = t_{\\text{parse}} + t_{\\text{APAFC}} + t_{\\text{ONNX}}$$
                    **How Calculated:**
                    - Hardware microsecond clock tracking execution duration from socket packet receipt to local alert emission.
                    """
                )

    except Exception:
        pass

