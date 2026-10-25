from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnxruntime as ort

from .edge import EdgeIDS, load_monitor, load_tree_explainer
from .prepare import map_csv
from .training import train


def _config(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-tier IoT IDS")
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--csv", required=True); training.add_argument("--label-column", required=True)
    training.add_argument("--output", default="artifacts"); training.add_argument("--trials", type=int, default=120)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--csv", required=True); prepare.add_argument("--mapping", required=True)
    prepare.add_argument("--label-column", required=True); prepare.add_argument("--output", required=True)
    serving = sub.add_parser("serve")
    serving.add_argument("--model", required=True); serving.add_argument("--reference", required=True)
    serving.add_argument("--labels", default="artifacts/labels.json"); serving.add_argument("--config", default="configs/default.json")
    serving.add_argument("--tree-model", help="Matching LightGBM text model for TreeSHAP explanations")
    serving.add_argument("--audit-log", default="artifacts/alerts.jsonl")
    args = parser.parse_args()
    if args.command == "prepare":
        map_csv(args.csv, args.mapping, args.label_column, args.output); return
    if args.command == "train":
        print(json.dumps(train(args.csv, args.label_column, args.output, args.trials), indent=2)); return
    config = _config(args.config)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    def retrain(event: dict) -> None:
        print(json.dumps({"type": "retrain_request", **event}), file=sys.stderr, flush=True)
    explainer = load_tree_explainer(args.tree_model) if args.tree_model else None
    ids = EdgeIDS(ort.InferenceSession(args.model, providers=["CPUExecutionProvider"]), labels, load_monitor(args.reference, config), top_k=config["alerts"]["top_k"], on_retrain=retrain, tree_explainer=explainer, audit_path=args.audit_log)
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(ids.classify(json.loads(line))), flush=True)


if __name__ == "__main__":
    main()
