"""
Page 6 — System Diagnostics, Artifact Integrity, Dependency Versions & Checkpoint Status.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import streamlit as st
import pandas as pd

from config.settings import (
    ALERT_LOG_PATH,
    DRIFT_REFERENCE_PATH,
    EVALUATION_RESULTS_PATH,
    FEATURE_CONFIG_PATH,
    MODEL_ARTIFACT_PATH,
    REAL_EDGE_IIOTSET_CSV,
    SYNTHETIC_DEMO_CSV,
)
from src.framework_bridge import (
    FRAMEWORK_ROOT,
    check_core_artifacts_status,
)
from src.ui_theme import render_header

st.set_page_config(page_title="System Diagnostics | IoT IDS", layout="wide")

render_header(
    title=" Module G — System Diagnostics, Dependencies & Repository Integrity",
    subtitle="Inspect runtime environment, installed package versions, model artifacts, and `iot_ids_framework` preservation status.",
    badge_text="Diagnostics Active",
)

# 1. Runtime & Environment Summary
st.subheader("1. Python Runtime & OS Environment")
e1, e2, e3, e4 = st.columns(4)
e1.metric("Python Version", sys.version.split()[0])
e2.metric("Platform / OS", f"{platform.system()} {platform.release()}")
e3.metric("Architecture", platform.machine())
git_head_file = FRAMEWORK_ROOT / ".git" / "refs" / "heads" / "main"
git_commit = git_head_file.read_text(encoding="utf-8").strip()[:10] if git_head_file.exists() else "N/A"
e4.metric("`iot_ids_framework` Git HEAD", f"main @ {git_commit}")

# 2. Primary Core Artifacts (`iot_ids_framework/artifacts` & `datasets`)
st.subheader("2. Primary Core Repository Artifacts (`iot_ids_framework`)")
core_status = check_core_artifacts_status()
core_rows = []
for name, info in core_status["files"].items():
    core_rows.append(
        {
            "Artifact": name,
            "Status": " Present" if info["exists"] else " Missing",
            "Size_KB": info["size_kb"],
            "Path": info["path"],
        }
    )
st.dataframe(pd.DataFrame(core_rows), width="stretch", hide_index=True)

# 3. Exploratory Workspace Artifacts (`models/`, `data/`, `logs/`, `results/`)
st.subheader("3. Exploratory & Dashboard Workspace Artifacts")
exp_files = [
    ("Genuine Edge-IIoTset Dataset (157,800 Rows)", REAL_EDGE_IIOTSET_CSV),
    ("Exploratory LightGBM Bundle", MODEL_ARTIFACT_PATH),
    ("APAFC 61->14 Config JSON", FEATURE_CONFIG_PATH),
    ("Drift Reference Histograms", DRIFT_REFERENCE_PATH),
    ("Latest Evaluation Metrics JSON", EVALUATION_RESULTS_PATH),
    ("Dashboard Alert Log JSONL", ALERT_LOG_PATH),
]
exp_rows = []
for label, path in exp_files:
    exists = path.exists()
    exp_rows.append(
        {
            "Component": label,
            "Status": " Present" if exists else " Missing",
            "Size_KB": round(path.stat().st_size / 1024.0, 2) if exists else 0.0,
            "Path": str(path),
        }
    )
st.dataframe(pd.DataFrame(exp_rows), width="stretch", hide_index=True)

# 4. Dependency Health Check
st.subheader("4. Installed Package Versions")
packages = [
    "numpy",
    "pandas",
    "scikit-learn",
    "lightgbm",
    "optuna",
    "onnx",
    "onnxmltools",
    "onnxruntime",
    "skl2onnx",
    "shap",
    "imbalanced-learn",
    "scipy",
    "streamlit",
    "plotly",
    "pytest",
    "psutil",
    "iot-ids-framework",
]
pkg_rows = []
for pkg in packages:
    try:
        ver = importlib.metadata.version(pkg)
        status = " Installed"
    except importlib.metadata.PackageNotFoundError:
        ver = "Not Installed"
        status = " Missing"
    pkg_rows.append({"Package": pkg, "Status": status, "Version": ver})
st.dataframe(pd.DataFrame(pkg_rows), width="stretch", hide_index=True)
