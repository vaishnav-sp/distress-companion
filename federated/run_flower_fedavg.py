"""Command-line entry point for the isolated Flower FedAvg experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from feedback.store import DEFAULT_FEEDBACK_PATH

from federated.flower_fedavg import run_federated_experiment
from federated.virtual_students import (
    REVIEWED_DATA_MODE,
    SYNTHETIC_TEST_DATA_MODE,
    VirtualStudentConfig,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline Flower FedAvg on virtual-student partitions. Synthetic "
            "mode is infrastructure-test data and not performance evidence."
        )
    )
    parser.add_argument(
        "--data-mode",
        choices=(SYNTHETIC_TEST_DATA_MODE, REVIEWED_DATA_MODE),
        default=SYNTHETIC_TEST_DATA_MODE,
    )
    parser.add_argument("--clients", type=int, choices=range(3, 6), default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feedback-db", type=Path, default=DEFAULT_FEEDBACK_PATH)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path("federated/output"))
    args = parser.parse_args(argv)

    if args.rounds < 1:
        parser.error("--rounds must be at least 1.")
    if args.data_mode == REVIEWED_DATA_MODE and args.review_manifest is None:
        parser.error("--review-manifest is required for reviewed data mode.")
    if args.review_manifest is not None and not args.review_manifest.is_file():
        parser.error(f"Review manifest does not exist: {args.review_manifest}")

    result = run_federated_experiment(
        config=VirtualStudentConfig(client_count=args.clients, seed=args.seed),
        rounds=args.rounds,
        seed=args.seed,
        data_mode=args.data_mode,
        feedback_db=args.feedback_db,
        review_manifest=args.review_manifest,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "artifact": str(result.artifact_path),
                "global_model_version": result.federated_metadata[
                    "global_model_version"
                ],
                "data_mode": result.federated_metadata["data_mode"],
                "synthetic_test_data": result.federated_metadata[
                    "synthetic_test_data"
                ],
                "initial_parameters": result.federated_metadata[
                    "initial_parameters"
                ],
                "final_parameters": result.federated_metadata["final_parameters"],
                "rounds": result.federated_metadata["rounds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
