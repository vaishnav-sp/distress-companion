"""Local demonstration-only labels kept separate from training review manifests."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from feedback.store import FeedbackStore
from fusion.schema import FeatureRecord
from training.reviewed_feedback import (
    _validate_feature_summary,
    feedback_review_block_reason,
)

DEMO_REVIEW_VERSION = "distress-demo-feedback-review-v1"
DEMO_TRAINING_LABELS = {
    "potential_distress",
    "no_sufficient_distress_evidence",
}
_DEMO_REVIEW_KEYS = {
    "review_version",
    "feedback_id",
    "training_label",
    "reviewed_at_utc",
}
_DEMO_REVIEW_LOCK = threading.Lock()


@dataclass(frozen=True)
class DemoFeedbackReview:
    feedback_id: str
    training_label: str
    reviewed_at_utc: str


@dataclass(frozen=True)
class DemoTrainingExample:
    feedback_id: str
    snapshot_id: str
    training_label: str
    record: FeatureRecord


@dataclass(frozen=True)
class DemoTrainingDataset:
    examples: tuple[DemoTrainingExample, ...]
    ineligible_reasons: dict[str, str]


def _load_demo_reviews(path: Path) -> dict[str, DemoFeedbackReview]:
    if not path.exists():
        return {}
    if not path.is_file():
        raise ValueError("The demonstration-review path is not a regular file.")
    reviews: dict[str, DemoFeedbackReview] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise OSError(f"Could not read demonstration reviews: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Demonstration-review file has invalid JSON on line {line_number}."
            ) from exc
        if not isinstance(entry, dict) or set(entry) != _DEMO_REVIEW_KEYS:
            raise ValueError(
                f"Demonstration-review entry on line {line_number} has invalid fields."
            )
        feedback_id = entry.get("feedback_id")
        label = entry.get("training_label")
        reviewed_at = entry.get("reviewed_at_utc")
        if (
            entry.get("review_version") != DEMO_REVIEW_VERSION
            or not isinstance(feedback_id, str)
            or not feedback_id
            or not isinstance(label, str)
            or label not in DEMO_TRAINING_LABELS
            or not isinstance(reviewed_at, str)
        ):
            raise ValueError(
                f"Demonstration-review entry on line {line_number} is invalid."
            )
        try:
            timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"Demonstration-review timestamp on line {line_number} is invalid."
            ) from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(
                f"Demonstration-review timestamp on line {line_number} must include a timezone."
            )
        if feedback_id in reviews:
            raise ValueError(
                "Demonstration-review file contains duplicate feedback IDs."
            )
        reviews[feedback_id] = DemoFeedbackReview(
            feedback_id=feedback_id,
            training_label=label,
            reviewed_at_utc=reviewed_at,
        )
    return reviews


def load_demo_feedback_reviews(
    review_path: str | Path,
) -> dict[str, DemoFeedbackReview]:
    """Read manually assigned demonstration labels without modifying local data."""
    return _load_demo_reviews(Path(review_path).expanduser())


def load_demo_training_dataset(
    feedback_store: FeedbackStore,
    review_path: str | Path,
    *,
    feedback_rows: Sequence[Mapping[str, Any]] | None = None,
    reviews: Mapping[str, DemoFeedbackReview] | None = None,
) -> DemoTrainingDataset:
    """Build a validated dataset from manual demo labels, separate from v2 reviews."""
    if feedback_rows is None:
        feedback_rows = feedback_store.list_feedback()
    if reviews is None:
        reviews = load_demo_feedback_reviews(review_path)
    rows_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in feedback_rows:
        feedback_id = row.get("feedback_id")
        if isinstance(feedback_id, str):
            rows_by_id.setdefault(feedback_id, []).append(row)

    candidates: list[DemoTrainingExample] = []
    ineligible_reasons: dict[str, str] = {}
    for feedback_id, review in reviews.items():
        matching_rows = rows_by_id.get(feedback_id, [])
        if len(matching_rows) != 1:
            ineligible_reasons[feedback_id] = (
                "Feedback ID does not identify exactly one persisted feedback row."
            )
            continue

        row = matching_rows[0]
        try:
            summary = json.loads(row["feature_summary_json"])
            record, _, _, provenance = _validate_feature_summary(
                summary, feedback_id
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            ineligible_reasons[feedback_id] = (
                f"Persisted feature summary failed validation: {exc}"
            )
            continue

        provenance_reason = feedback_review_block_reason(row, feedback_id)
        if provenance_reason is not None:
            ineligible_reasons[feedback_id] = provenance_reason
            continue

        snapshot_id = provenance.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            ineligible_reasons[feedback_id] = (
                "Persisted snapshot provenance has no valid snapshot ID."
            )
            continue
        candidates.append(
            DemoTrainingExample(
                feedback_id=feedback_id,
                snapshot_id=snapshot_id,
                training_label=review.training_label,
                record=record,
            )
        )

    snapshot_counts: dict[str, int] = {}
    for example in candidates:
        snapshot_counts[example.snapshot_id] = (
            snapshot_counts.get(example.snapshot_id, 0) + 1
        )

    examples: list[DemoTrainingExample] = []
    for example in candidates:
        if snapshot_counts[example.snapshot_id] > 1:
            ineligible_reasons[example.feedback_id] = (
                "Snapshot ID is referenced by multiple demo reviews; duplicate "
                "snapshot examples are not included."
            )
        else:
            examples.append(example)
    return DemoTrainingDataset(tuple(examples), ineligible_reasons)


def append_demo_feedback_review(
    feedback_store: FeedbackStore,
    review_path: str | Path,
    feedback_id: str,
    training_label: str,
) -> DemoFeedbackReview:
    """Append one explicit demo label bound to an existing feedback row."""
    if not isinstance(feedback_id, str) or not feedback_id:
        raise ValueError("A valid selected feedback ID is required.")
    if (
        not isinstance(training_label, str)
        or training_label not in DEMO_TRAINING_LABELS
    ):
        raise ValueError(
            "Select potential_distress or no_sufficient_distress_evidence."
        )
    matching_rows = [
        row
        for row in feedback_store.list_feedback()
        if row.get("feedback_id") == feedback_id
    ]
    if len(matching_rows) != 1:
        raise ValueError(
            "The selected feedback row does not exist uniquely in local storage."
        )

    path = Path(review_path).expanduser()
    with _DEMO_REVIEW_LOCK:
        existing_reviews = _load_demo_reviews(path)
        if feedback_id in existing_reviews:
            raise ValueError(
                "A demonstration review already exists for this feedback row."
            )
        review = DemoFeedbackReview(
            feedback_id=feedback_id,
            training_label=training_label,
            reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        entry = {
            "review_version": DEMO_REVIEW_VERSION,
            "feedback_id": review.feedback_id,
            "training_label": review.training_label,
            "reviewed_at_utc": review.reviewed_at_utc,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8", newline="\n") as destination:
                if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
                    destination.write("\n")
                destination.write(json.dumps(entry, sort_keys=True) + "\n")
                destination.flush()
        except OSError as exc:
            raise OSError(f"Could not save demonstration review: {exc}") from exc
    return review
