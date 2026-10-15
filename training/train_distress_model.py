"""Train and save the local multimodal logistic-regression pipeline from labeled JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from training.distress_logistic import (
    DEFAULT_MODEL_PATH,
    train_distress_logistic_model,
)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object.")
            if "label" not in item:
                raise ValueError(f"Line {line_number} is missing its explicit 'label'.")
            records.append(item)
    if not records:
        raise ValueError(f"No labeled feature records found in {path}.")
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the local fusion model from explicit labeled FeatureRecord JSONL. "
            "This does not use face-expression image labels as multimodal labels."
        )
    )
    parser.add_argument(
        "--records",
        required=True,
        type=Path,
        help="JSONL file: each line is a serialized FeatureRecord plus a ground-truth 'label'.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Output pipeline path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--allow-acted-synthetic",
        action="store_true",
        help="Explicitly permit acted/synthetic data for a pipeline-only demonstration.",
    )
    args = parser.parse_args(argv)

    if not args.records.is_file():
        parser.error(f"Labeled fusion records file does not exist: {args.records}")
    try:
        records = _load_jsonl(args.records)
        model = train_distress_logistic_model(
            records,
            model_path=args.model_path,
            use_group_split=True,
            allow_acted_synthetic=args.allow_acted_synthetic,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    metadata_path = args.model_path.with_suffix(".metadata.json")
    print(f"Saved multimodal fusion pipeline: {args.model_path}")
    print(f"Saved compatibility/training metadata: {metadata_path}")
    print(
        "Training counts:",
        json.dumps(model.metadata["training_record_counts_by_class"], sort_keys=True),
    )
    print(f"Record kinds: {', '.join(model.metadata['record_kinds'])}")
    print(str(model.metadata["limitations"]))
    if model.metadata["contains_acted_or_synthetic_records"]:
        print(model.metadata["synthetic_data_warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
