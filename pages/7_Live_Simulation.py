"""
Page 7 — Live Simulation Console: Loopback IDS Listener & Simulated Attacker Client.

Provides a dedicated Security Operations Center (SOC) console with:
- Isolated local TCP socket listener bound strictly to loopback (`127.0.0.1`).
- Bidirectional flow reconstruction via `FlowAssembler`.
- Real-time flow classification via `iot_ids.edge.EdgeIDS` (ONNX Runtime).
- Attacker client dispatching benign IoT telemetry or realistic attack vectors.
- Live traffic counters, alert status, TreeSHAP attributions, and deep-packet payload inspection.
"""
from __future__ import annotations

import streamlit as st

from src.live_listener import render_live_simulation_console
from src.ui_theme import render_header

st.set_page_config(
    page_title="Live Simulation | IoT IDS",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_header(
    title=" Live Socket Simulation — IDS Listener & Interactive Attacker",
    subtitle=(
        "Run an isolated local loopback socket listener (`127.0.0.1`) and inject realistic IoT telemetry or "
        "bounded attack payloads to evaluate edge detection in real time."
    ),
    badge_text="LIVE SOCKET INGESTION",
)

st.markdown(
    """
    <div class="soc-banner">
        <strong>SECURITY & OPERATIONAL NOTICE:</strong>
        This console runs an actual TCP socket listener bound exclusively to <code>127.0.0.1</code>.
        Traffic is transmitted across your local loopback stack, assembled into bidirectional flow records via
        <code>iot_ids.ingest.FlowAssembler</code>, and evaluated through <code>iot_ids.edge.EdgeIDS</code> (ONNX Runtime).
        No public network ports are opened, and no external packets are captured or forwarded.
    </div>
    """,
    unsafe_allow_html=True,
)

render_live_simulation_console()
