"""Explicit local review and eligibility checks for caretaker feedback examples."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from feedback.store import FeedbackStore
from fusion.schema import (
    CANONICAL_MODES,
    FEATURE_SCHEMA_VERSION,
    FeatureRecord,
    feature_order,
)
from training.distress_logistic import (
    TARGET_LABEL_MAPPING,
    EvidenceSufficiencyPolicy,
)

REVIEW_MANIFEST_VERSION = "phase4-reviewed-feedback-manifest-v1"
REVIEW_DECISIONS = {"approved_for_training", "excluded", "ambiguous"}
MAX_REVIEWED_MODALITY_AGE_SECONDS = 3.0
_SUMMARY_KEYS = {
    "schema_version",
    "record_kind",
    "provenance",
    "feature_values",
    "feature_validity",
    "unavailable_reasons",
}
_MANIFEST_KEYS = {
    "manifest_version",
    "feedback_id",
    "review_decision",
    "reviewer_reference",
    "training_label",
    "reviewed_at_utc",
    "event_window_start_utc",
    "event_window_end_utc",
    "modality_timestamps_utc",
    "evidence_context_reference",
}
_SOURCE_VALUES = {
    "live_monitoring",
    "monitoring_diagnostic_snapshot",
    "unspecified",
}


@dataclass(frozen=True)
class ReviewedTrainingExample:
    feedback_id: str
    record: FeatureRecord
    label: str
    feedback_provenance: dict[str, Any]

    def to_training_record(self) -> dict[str, Any]:
        payload = self.record.to_dict()
        payload["label"] = self.label
        payload["feedback_provenance"] = dict(self.feedback_provenance)
        return payload


@dataclass(frozen=True)
class EligibilityRejection:
    feedback_id: str | None
    reason: str
    manifest_line: int | None = None


@dataclass(frozen=True)
class ReviewedFeedbackLoadResult:
    examples: tuple[ReviewedTrainingExample, ...]
    rejections: tuple[EligibilityRejection, ...]


def _parse_utc_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty timezone-aware ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone.")
    return parsed.astimezone(timezone.utc)


def _validate_feature_summary(
    payload: Any,
    feedback_id: str,
) -> tuple[FeatureRecord, str, str]:
    if not isinstance(payload, dict):
        raise ValueError("Feature summary must be a JSON object.")
    unknown = set(payload) - _SUMMARY_KEYS
    if unknown:
        raise ValueError(f"Feature summary contains unsupported fields: {sorted(unknown)}.")
    required = _SUMMARY_KEYS - {"unavailable_reasons"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Feature summary is missing fields: {sorted(missing)}.")
    if payload["schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Feature schema mismatch: expected {FEATURE_SCHEMA_VERSION!r}, "
            f"got {payload['schema_version']!r}."
        )
    names = feature_order()
    values = payload["feature_values"]
    validity = payload["feature_validity"]
    if not isinstance(values, dict) or set(values) != set(names):
        raise ValueError("Feature values do not match the canonical schema feature set.")
    if not isinstance(validity, dict) or set(validity) != set(names):
        raise ValueError("Feature-validity flags do not match the canonical schema feature set.")

    checked_values: dict[str, float | None] = {}
    checked_validity: dict[str, bool] = {}
    for name in names:
        valid = validity[name]
        value = values[name]
        if not isinstance(valid, bool):
            raise ValueError(f"Feature-validity flag for {name!r} must be boolean.")
        if valid:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Valid feature {name!r} must have a finite numeric value.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"Feature {name!r} has a non-finite numeric value.")
            checked_values[name] = numeric_value
        else:
            if value is not None:
                raise ValueError(f"Invalid feature {name!r} must be null, not an invented value.")
            checked_values[name] = None
        checked_validity[name] = valid

    source_info = payload["provenance"]
    if not isinstance(source_info, dict) or set(source_info) != {"source"}:
        raise ValueError("Feature provenance must contain only its recorded source.")
    source = source_info["source"]
    if not isinstance(source, str) or source not in _SOURCE_VALUES:
        raise ValueError("Feature provenance contains an unsupported or missing source.")

    record_kind = payload["record_kind"]
    if not isinstance(record_kind, str) or not record_kind:
        raise ValueError("Feature record kind must be a non-empty string.")
    reasons = payload.get("unavailable_reasons", {})
    if not isinstance(reasons, dict):
        raise ValueError("Unavailable-feature reasons must be a JSON object.")
    if set(reasons) - set(names):
        raise ValueError("Unavailable-feature reasons contain non-canonical feature names.")
    if any(not isinstance(reason, str) for reason in reasons.values()):
        raise ValueError("Unavailable-feature reasons must be strings.")
    if any(checked_validity[name] for name in reasons):
        raise ValueError("Unavailable-feature reasons cannot be attached to valid features.")

    record = FeatureRecord(
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=names,
        feature_values=checked_values,
        feature_validity=checked_validity,
        unavailable_reasons=dict(reasons),
        modality_status={modality: "not_recorded" for modality in CANONICAL_MODES},
        modality_messages={modality: "" for modality in CANONICAL_MODES},
        provenance={
            "source": source,
            "feedback_id": feedback_id,
            "feature_record_kind": record_kind,
        },
        record_kind="reviewed_feedback",
        created_at_utc=None,
    )
    sufficient, reason = EvidenceSufficiencyPolicy().evaluate(record)
    if not sufficient:
        raise ValueError(f"Insufficient evidence: {reason}")
    return record, source, record_kind


def _approved_example(
    feedback: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> ReviewedTrainingExample:
    feedback_id = feedback.get("feedback_id")
    if not isinstance(feedback_id, str) or not feedback_id:
        raise ValueError("Stored feedback record has no valid feedback ID.")
    if manifest.get("manifest_version") != REVIEW_MANIFEST_VERSION:
        raise ValueError(f"Manifest version must be {REVIEW_MANIFEST_VERSION!r}.")
    if manifest.get("feedback_id") != feedback_id:
        raise ValueError("Manifest feedback ID does not match the stored record.")

    decision = manifest.get("review_decision")
    if not isinstance(decision, str) or decision not in REVIEW_DECISIONS:
        raise ValueError("Review decision must be approved_for_training, excluded, or ambiguous.")
    if decision != "approved_for_training":
        raise ValueError(f"Review decision {decision!r} is not eligible for training.")

    label = manifest.get("training_label")
    if not isinstance(label, str) or label not in TARGET_LABEL_MAPPING:
        raise ValueError(
            "An explicitly reviewed training_label must be one of "
            f"{tuple(TARGET_LABEL_MAPPING)}."
        )
    reviewer_reference = manifest.get("reviewer_reference")
    if (
        not isinstance(reviewer_reference, str)
        or not reviewer_reference
        or len(reviewer_reference) > 128
        or not reviewer_reference.isascii()
        or any(not (char.isalnum() or char in "._:-") for char in reviewer_reference)
    ):
        raise ValueError(
            "reviewer_reference must be an opaque local reviewer code, not a name or free text."
        )
    action = feedback.get("action")
    if not isinstance(action, str) or action not in {
        "confirm_alert",
        "reject_alert",
        "missed_event",
    }:
        raise ValueError("Stored feedback action is missing or unsupported.")
    alert_id = feedback.get("alert_id")
    if action == "missed_event" and alert_id is not None:
        raise ValueError("A missed-event report cannot be associated with an alert ID.")
    if action in {"confirm_alert", "reject_alert"} and (
        not isinstance(alert_id, str) or not alert_id
    ):
        raise ValueError(f"Stored {action} feedback is missing its alert ID.")

    submitted_at = _parse_utc_timestamp(
        feedback.get("submitted_at_utc"),
        "stored submitted_at_utc",
    )
    reviewed_at = _parse_utc_timestamp(manifest.get("reviewed_at_utc"), "reviewed_at_utc")
    window_start = _parse_utc_timestamp(
        manifest.get("event_window_start_utc"),
        "event_window_start_utc",
    )
    window_end = _parse_utc_timestamp(
        manifest.get("event_window_end_utc"),
        "event_window_end_utc",
    )
    if window_start >= window_end:
        raise ValueError("The event window start must be earlier than its end.")
    if submitted_at < window_end:
        raise ValueError("The feedback submission predates the reviewed event window.")
    if reviewed_at < submitted_at:
        raise ValueError("The explicit review timestamp predates feedback submission.")

    context_reference = manifest.get("evidence_context_reference")
    if (
        not isinstance(context_reference, str)
        or not context_reference
        or len(context_reference) > 128
        or not context_reference.isascii()
        or any(not (char.isalnum() or char in "._:-") for char in context_reference)
    ):
        raise ValueError(
            "evidence_context_reference must be an opaque local reference, not a path or free text."
        )

    summary_json = feedback.get("feature_summary_json")
    if not isinstance(summary_json, str):
        raise ValueError("Stored feature summary is missing.")
    try:
        summary_payload = json.loads(summary_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Stored feature summary is malformed JSON: {exc.msg}.") from exc
    record, feature_source, feature_record_kind = _validate_feature_summary(
        summary_payload,
        feedback_id,
    )

    modality_timestamps = manifest.get("modality_timestamps_utc")
    if not isinstance(modality_timestamps, dict):
        raise ValueError("modality_timestamps_utc is required to verify evidence freshness.")
    available_modalities = {
        modality
        for modality, available in record.available_modalities.items()
        if available
    }
    if set(modality_timestamps) != available_modalities:
        raise ValueError(
            "Modality timestamps must be supplied for exactly the modalities with valid features."
        )
    normalized_modality_timestamps: dict[str, str] = {}
    for modality in CANONICAL_MODES:
        if modality not in available_modalities:
            continue
        timestamp = _parse_utc_timestamp(
            modality_timestamps[modality],
            f"modality_timestamps_utc[{modality!r}]",
        )
        if timestamp < window_start or timestamp > window_end:
            raise ValueError(f"{modality} timestamp falls outside the reviewed event window.")
        if (window_end - timestamp).total_seconds() > MAX_REVIEWED_MODALITY_AGE_SECONDS:
            raise ValueError(
                f"{modality} evidence is older than the "
                f"{MAX_REVIEWED_MODALITY_AGE_SECONDS:g}-second freshness limit."
            )
        normalized_modality_timestamps[modality] = timestamp.isoformat()

    provenance = {
        "feedback_id": feedback_id,
        "feedback_action": action,
        "alert_id": alert_id,
        "feedback_submitted_at_utc": submitted_at.isoformat(),
        "review_decision": decision,
        "reviewed_at_utc": reviewed_at.isoformat(),
        "event_window_start_utc": window_start.isoformat(),
        "event_window_end_utc": window_end.isoformat(),
        "modality_timestamps_utc": normalized_modality_timestamps,
        "evidence_context_reference": context_reference,
        "feature_source": feature_source,
        "feature_record_kind": feature_record_kind,
        "reviewer_reference": reviewer_reference,
    }
    record.provenance = {
        "source": "reviewed_feedback_manifest",
        "feedback_id": feedback_id,
        "feedback_action": action,
        "feedback_submitted_at_utc": submitted_at.isoformat(),
        "review_decision": decision,
        "reviewed_at_utc": reviewed_at.isoformat(),
        "evidence_context_reference": context_reference,
        "event_window_start_utc": window_start.isoformat(),
        "event_window_end_utc": window_end.isoformat(),
        "modality_timestamps_utc": normalized_modality_timestamps,
        "feature_source": feature_source,
        "feature_record_kind": feature_record_kind,
        "reviewer_reference": reviewer_reference,
    }
    return ReviewedTrainingExample(
        feedback_id=feedback_id,
        record=record,
        label=label,
        feedback_provenance=provenance,
    )


def _load_manifest(
    manifest_path: Path,
) -> tuple[dict[str, tuple[int, dict[str, Any]]], set[str], list[EligibilityRejection]]:
    entries: dict[str, tuple[int, dict[str, Any]]] = {}
    duplicate_ids: set[str] = set()
    rejections: list[EligibilityRejection] = []
    with manifest_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                rejections.append(
                    EligibilityRejection(
                        None,
                        f"Malformed manifest JSON: {exc.msg}.",
                        line_number,
                    )
                )
                continue
            if not isinstance(item, dict):
                rejections.append(
                    EligibilityRejection(
                        None,
                        "Manifest line must contain a JSON object.",
                        line_number,
                    )
                )
                continue
            unknown = set(item) - _MANIFEST_KEYS
            if unknown:
                rejections.append(
                    EligibilityRejection(
                        item.get("feedback_id") if isinstance(item.get("feedback_id"), str) else None,
                        f"Manifest contains unsupported fields: {sorted(unknown)}.",
                        line_number,
                    )
                )
                continue
            feedback_id = item.get("feedback_id")
            if not isinstance(feedback_id, str) or not feedback_id:
                rejections.append(
                    EligibilityRejection(
                        None,
                        "Manifest line is missing a valid feedback_id.",
                        line_number,
                    )
                )
                continue
            if feedback_id in entries or feedback_id in duplicate_ids:
                entries.pop(feedback_id, None)
                duplicate_ids.add(feedback_id)
                continue
            entries[feedback_id] = (line_number, item)
    return entries, duplicate_ids, rejections


def load_reviewed_feedback(
    feedback_store: FeedbackStore,
    manifest_path: str | Path,
) -> ReviewedFeedbackLoadResult:
    """Load only explicitly approved, fresh, schema-compatible local examples."""
    path = Path(manifest_path)
    entries, duplicate_ids, rejections = _load_manifest(path)
    feedback_rows = feedback_store.list_feedback()
    known_ids = {
        row.get("feedback_id")
        for row in feedback_rows
        if isinstance(row.get("feedback_id"), str)
    }

    for feedback_id in sorted(set(entries) - known_ids):
        line_number, _ = entries[feedback_id]
        rejections.append(
            EligibilityRejection(
                feedback_id,
                "Manifest entry does not reference a stored feedback record.",
                line_number,
            )
        )
    for feedback_id in sorted(duplicate_ids):
        rejections.append(
            EligibilityRejection(
                feedback_id,
                "Duplicate review-manifest entries are not eligible.",
            )
        )

    examples: list[ReviewedTrainingExample] = []
    for feedback in feedback_rows:
        feedback_id = feedback.get("feedback_id")
        if not isinstance(feedback_id, str) or not feedback_id:
            rejections.append(
                EligibilityRejection(None, "Stored feedback record has no valid feedback ID.")
            )
            continue
        if feedback_id in duplicate_ids:
            continue
        manifest_entry = entries.get(feedback_id)
        if manifest_entry is None:
            rejections.append(
                EligibilityRejection(
                    feedback_id,
                    "No separate review-manifest entry; feedback action is not a training label.",
                )
            )
            continue
        line_number, manifest = manifest_entry
        try:
            example = _approved_example(feedback, manifest)
        except (TypeError, ValueError) as exc:
            rejections.append(EligibilityRejection(feedback_id, str(exc), line_number))
            continue
        examples.append(example)

    return ReviewedFeedbackLoadResult(tuple(examples), tuple(rejections))


def write_reviewed_examples(
    result: ReviewedFeedbackLoadResult,
    output_path: str | Path,
) -> Path:
    """Write only accepted numerical feature records and explicit review provenance."""
    if not result.examples:
        raise ValueError("No eligible reviewed examples are available to export.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = [
        json.dumps(example.to_training_record(), sort_keys=True, allow_nan=False)
        for example in result.examples
    ]
    target.write_text("\n".join(serialized) + "\n", encoding="utf-8")
    return target
