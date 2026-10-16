"""Export explicitly reviewed caretaker feedback as local numerical examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from feedback.store import DEFAULT_FEEDBACK_PATH, FeedbackStorageError, FeedbackStore
from training.reviewed_feedback import load_reviewed_feedback, write_reviewed_examples


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export only explicitly reviewed, fresh, schema-compatible feedback "
            "as a local numerical-feature JSONL file."
        )
    )
    parser.add_argument(
        "--review-manifest",
        required=True,
        type=Path,
        help="Local JSONL manifest containing explicit review decisions and verified window timestamps.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Local output JSONL path for eligible numerical feature examples.",
    )
    parser.add_argument(
        "--feedback-db",
        type=Path,
        default=DEFAULT_FEEDBACK_PATH,
        help=f"Local feedback SQLite database (default: {DEFAULT_FEEDBACK_PATH}).",
    )
    args = parser.parse_args(argv)

    if not args.review_manifest.is_file():
        parser.error(f"Review manifest does not exist: {args.review_manifest}")
    if args.output.resolve() in {args.review_manifest.resolve(), args.feedback_db.resolve()}:
        parser.error("Output path must not overwrite the review manifest or feedback database.")
    try:
        result = load_reviewed_feedback(
            FeedbackStore(args.feedback_db),
            args.review_manifest,
        )
        for rejection in result.rejections:
            location = (
                f"line {rejection.manifest_line}"
                if rejection.manifest_line is not None
                else rejection.feedback_id or "manifest"
            )
            print(f"Rejected {location}: {rejection.reason}", file=sys.stderr)
        if not result.examples:
            print("No eligible reviewed feedback examples were exported.", file=sys.stderr)
            return 1
        output_path = write_reviewed_examples(result, args.output)
    except (FeedbackStorageError, OSError, TypeError, ValueError) as exc:
        print(f"Could not prepare reviewed feedback examples: {exc}", file=sys.stderr)
        return 1

    print(
        f"Exported {len(result.examples)} eligible numerical examples to {output_path}; "
        f"rejected {len(result.rejections)} records."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
