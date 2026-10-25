"""Dataset adapter that maps a raw Edge-IIoTset export to the APAFC training schema."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .features import FEATURE_NAMES, PROTOCOL_IDS


def map_csv(source: str | Path, mapping_file: str | Path, label_column: str, output: str | Path) -> None:
    raw = pd.read_csv(source)
    mapping = json.loads(Path(mapping_file).read_text(encoding="utf-8"))
    if label_column not in raw.columns:
        raise ValueError(f"label column '{label_column}' does not exist in source CSV")
    absent = {column for column in mapping.values() if column is not None and column not in raw.columns}
    if absent:
        raise ValueError(f"mapping refers to source columns that do not exist: {sorted(absent)}")
    result = pd.DataFrame(index=raw.index)
    for target in FEATURE_NAMES:
        if target == "protocol_id":
            result[target] = raw[mapping["protocol"]].astype(str).str.lower().map(PROTOCOL_IDS).fillna(0)
        elif target == "mean_packet_bytes":
            packets = pd.to_numeric(raw[mapping["packet_count"]], errors="coerce").clip(lower=1)
            result[target] = pd.to_numeric(raw[mapping["byte_count"]], errors="coerce").fillna(0) / packets
        else:
            source_column = mapping.get(target)
            result[target] = pd.to_numeric(raw[source_column], errors="coerce").fillna(0) if source_column else 0
    result["label"] = raw[label_column].astype(str).str.strip().str.lower()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
