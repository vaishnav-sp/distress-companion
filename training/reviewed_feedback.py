"""Explicit local review and eligibility checks for caretaker feedback examples."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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

REVIEW_MANIFEST_VERSION = "phase4-reviewed-feedback-manifest-v2"
REVIEW_DECISIONS = {"approved_for_training", "excluded", "ambiguous"}
MAX_REVIEWED_MODALITY_AGE_SECONDS = 3.0
_SUMMARY_KEYS = {
    "schema_version",
    "record_kind",
    "provenance",
    "feature_values",
    "feature_validity",
    "unavailable_reasons",
    "created_at_utc",
}
_MANIFEST_KEYS = {
    "manifest_version",
    "feedback_id",
    "snapshot_id",
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
    "live_rule_based_monitoring",
    "unspecified",
}
_REVIEW_WRITE_LOCK = threading.Lock()


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


@dataclass(frozen=True)
class FeedbackReviewStatus:
    status: str
    training_eligibility: str
    training_label: str | None = None
    reason: str | None = None


def feedback_review_block_reason(
    feedback_row: Mapping[str, Any],
    selected_feedback_id: str,
) -> str | None:
    """Return a safe explanation when a stored row lacks reviewable provenance."""
    feedback_id = feedback_row.get("feedback_id")
    if (
        not isinstance(selected_feedback_id, str)
        or not selected_feedback_id
        or feedback_id != selected_feedback_id
    ):
        return "The selected feedback row does not match the stored row being checked."

    summary_json = feedback_row.get("feature_summary_json")
    if not isinstance(summary_json, str):
        return "The stored feature summary is unavailable."
    try:
        summary = json.loads(summary_json)
    except json.JSONDecodeError:
        return "The stored feature summary is not valid JSON."
    if not isinstance(summary, dict):
        return "The stored feature summary is not a JSON object."
    if summary.get("schema_version") != FEATURE_SCHEMA_VERSION:
        return "The persisted feature-schema version is missing or unsupported."
    provenance_fields = summary.get("provenance")
    if not isinstance(provenance_fields, dict):
        return "The persisted provenance object is missing."

    problems: list[str] = []
    if provenance_fields.get("provenance_available") is not True:
        problems.append("provenance flag (availability)")
    snapshot_id = provenance_fields.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        problems.append("snapshot ID")

    action = feedback_row.get("action")
    alert_id = feedback_row.get("alert_id")
    if action not in {"confirm_alert", "reject_alert", "missed_event"}:
        return "The persisted feedback action is unsupported."
    if action == "missed_event":
        if alert_id is not None:
            return "The persisted missed-event row has an unexpected alert reference."
        expected_event_reference = "caretaker_reported_missed_event"
    else:
        if not isinstance(alert_id, str) or not alert_id:
            return "The persisted alert-feedback row has no alert reference."
        expected_event_reference = "alert_triggering_snapshot"
    if provenance_fields.get("event_reference_type") != expected_event_reference:
        problems.append("action-appropriate event reference")
    if provenance_fields.get("source_timestamp_clock") != "monotonic":
        problems.append("monotonic timestamp clock")

    utc_fields = provenance_fields.get("source_timestamps_utc")
    monotonic_fields = provenance_fields.get("source_timestamps_monotonic")
    if not isinstance(utc_fields, dict) or not isinstance(monotonic_fields, dict):
        problems.append("UTC and monotonic modality timestamps")
        utc_fields = {}
        monotonic_fields = {}
    elif not utc_fields:
        problems.append("modality timestamp entries")
    elif set(utc_fields) != set(monotonic_fields):
        problems.append("UTC/monotonic timestamp coverage")
    if set(utc_fields) - set(CANONICAL_MODES):
        problems.append("supported modality names")
    if summary.get("created_at_utc") is None:
        problems.append("feature-snapshot creation time")
    if provenance_fields.get("source_timestamp_anchor_utc") is None:
        problems.append("UTC clock anchor")
    raw_anchor_monotonic = provenance_fields.get(
        "source_timestamp_anchor_monotonic"
    )
    if (
        isinstance(raw_anchor_monotonic, bool)
        or not isinstance(raw_anchor_monotonic, (int, float))
        or not math.isfinite(float(raw_anchor_monotonic))
    ):
        problems.append("monotonic clock anchor")

    try:
        record, _, _, provenance = _validate_feature_summary(summary, feedback_id)
    except (TypeError, ValueError):
        if problems:
            return (
                "Missing or invalid persisted metadata: "
                + ", ".join(dict.fromkeys(problems))
                + "."
            )
        return (
            "The stored feature summary, schema, or timestamp fields are incomplete or invalid."
        )

    available_modalities = {
        modality
        for modality, available in record.available_modalities.items()
        if available
    }
    utc_timestamps = provenance["source_timestamps_utc"]
    monotonic_timestamps = provenance["source_timestamps_monotonic"]
    if (
        not available_modalities
        or available_modalities - set(CANONICAL_MODES)
        or set(utc_timestamps) != available_modalities
        or set(monotonic_timestamps) != available_modalities
    ):
        problems.append("timestamps for exactly the available modalities")

    created_at = provenance["created_at_utc"]
    anchor_utc_value = provenance["source_timestamp_anchor_utc"]
    anchor_monotonic = provenance["source_timestamp_anchor_monotonic"]
    if (
        created_at is not None
        and anchor_utc_value is not None
        and not isinstance(anchor_monotonic, bool)
        and isinstance(anchor_monotonic, (int, float))
        and math.isfinite(float(anchor_monotonic))
    ):
        created_at_time = _parse_utc_timestamp(
            created_at, "stored feature snapshot created_at_utc"
        )
        anchor_utc = _parse_utc_timestamp(
            anchor_utc_value, "stored source_timestamp_anchor_utc"
        )
        for modality in sorted(available_modalities):
            if modality not in utc_timestamps or modality not in monotonic_timestamps:
                continue
            utc_time = _parse_utc_timestamp(
                utc_timestamps[modality],
                f"stored source_timestamps_utc[{modality!r}]",
            )
            monotonic_time = monotonic_timestamps[modality]
            reconstructed = anchor_utc + timedelta(
                seconds=float(monotonic_time) - float(anchor_monotonic)
            )
            if (
                abs((utc_time - reconstructed).total_seconds()) > 0.002
                or utc_time > created_at_time
            ):
                problems.append("timestamps consistent with clock anchors and snapshot time")
                break

    if problems:
        return "Missing or invalid persisted metadata: " + ", ".join(dict.fromkeys(problems)) + "."
    return None


def feedback_review_provenance_diagnostic(
    feedback_row: Mapping[str, Any],
    selected_feedback_id: str,
) -> dict[str, str]:
    """Summarize provenance categories without exposing persisted values."""
    categories = {
        "Feedback row identity": ("selected feedback row",),
        "Feature schema": ("feature-schema version", "schema"),
        "Provenance availability": ("provenance flag", "provenance object"),
        "Snapshot identity": ("snapshot ID",),
        "Action-specific event reference": ("event reference", "alert reference"),
        "Source timestamp clock": ("timestamp clock",),
        "Modality timestamp coverage": (
            "timestamp entries",
            "timestamp coverage",
            "modality timestamps",
            "source timestamps",
            "timestamps for exactly",
        ),
        "Clock anchors": ("clock anchor",),
        "Snapshot creation time": ("creation time",),
        "Timestamp consistency": ("timestamps consistent", "timestamp provenance"),
    }
    reason = feedback_review_block_reason(feedback_row, selected_feedback_id)
    if reason is None:
        return {category: "Present" for category in categories}

    if "selected feedback row" in reason:
        return {
            category: (
                "Missing/invalid"
                if category == "Feedback row identity"
                else "Not independently verified"
            )
            for category in categories
        }

    if not reason.startswith("Missing or invalid persisted metadata:"):
        return {category: "Not independently verified" for category in categories}

    return {
        category: (
            "Missing/invalid"
            if any(token in reason for token in issue_tokens)
            else "Not independently verified"
        )
        for category, issue_tokens in categories.items()
    }


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
) -> tuple[FeatureRecord, str, str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Feature summary must be a JSON object.")
    unknown = set(payload) - _SUMMARY_KEYS
    if unknown:
        raise ValueError(f"Feature summary contains unsupported fields: {sorted(unknown)}.")
    required = _SUMMARY_KEYS - {"unavailable_reasons", "created_at_utc"}
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
    allowed_provenance_keys = {
        "source",
        "snapshot_id",
        "provenance_available",
        "event_reference_type",
        "source_timestamp_clock",
        "source_timestamps_utc",
        "source_timestamps_monotonic",
        "source_timestamp_anchor_utc",
        "source_timestamp_anchor_monotonic",
    }
    if (
        not isinstance(source_info, dict)
        or not {"source"}.issubset(source_info)
        or set(source_info) - allowed_provenance_keys
    ):
        raise ValueError("Feature provenance contains unsupported or missing fields.")
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

    created_at_utc = payload.get("created_at_utc")
    if created_at_utc is not None:
        created_at_utc = _parse_utc_timestamp(
            created_at_utc,
            "stored feature snapshot created_at_utc",
        ).isoformat()
    snapshot_id = source_info.get("snapshot_id")
    if snapshot_id is not None and (
        not isinstance(snapshot_id, str)
        or not snapshot_id
        or len(snapshot_id) > 512
        or not snapshot_id.isascii()
    ):
        raise ValueError("Stored feature snapshot ID is invalid.")
    provenance_available = source_info.get("provenance_available", False)
    if not isinstance(provenance_available, bool):
        raise ValueError("Stored snapshot provenance availability must be boolean.")
    event_reference_type = source_info.get("event_reference_type")
    if event_reference_type is not None and not isinstance(event_reference_type, str):
        raise ValueError("Stored snapshot event-reference type must be a string.")
    timestamp_clock = source_info.get("source_timestamp_clock")
    if timestamp_clock is not None and not isinstance(timestamp_clock, str):
        raise ValueError("Stored source timestamp clock must be a string.")

    source_timestamps_utc = source_info.get("source_timestamps_utc", {})
    source_timestamps_monotonic = source_info.get(
        "source_timestamps_monotonic", {}
    )
    if not isinstance(source_timestamps_utc, dict) or not isinstance(
        source_timestamps_monotonic, dict
    ):
        raise ValueError("Stored modality source timestamps must be JSON objects.")
    if set(source_timestamps_utc) != set(source_timestamps_monotonic):
        raise ValueError("UTC and monotonic modality timestamps do not match.")
    normalized_source_timestamps_utc: dict[str, str] = {}
    for modality, timestamp_value in source_timestamps_utc.items():
        if modality not in CANONICAL_MODES:
            raise ValueError("Stored modality source timestamp has an unknown modality.")
        timestamp = _parse_utc_timestamp(
            timestamp_value,
            f"stored source_timestamps_utc[{modality!r}]",
        )
        monotonic_value = source_timestamps_monotonic[modality]
        if (
            isinstance(monotonic_value, bool)
            or not isinstance(monotonic_value, (int, float))
            or not math.isfinite(float(monotonic_value))
        ):
            raise ValueError(
                f"Stored monotonic timestamp for {modality!r} must be finite."
            )
        normalized_source_timestamps_utc[modality] = timestamp.isoformat()

    timestamp_anchor_utc = source_info.get("source_timestamp_anchor_utc")
    timestamp_anchor_monotonic = source_info.get(
        "source_timestamp_anchor_monotonic"
    )
    if timestamp_anchor_utc is not None:
        timestamp_anchor_utc = _parse_utc_timestamp(
            timestamp_anchor_utc,
            "stored source_timestamp_anchor_utc",
        ).isoformat()
    if timestamp_anchor_monotonic is not None and (
        isinstance(timestamp_anchor_monotonic, bool)
        or not isinstance(timestamp_anchor_monotonic, (int, float))
        or not math.isfinite(float(timestamp_anchor_monotonic))
    ):
        raise ValueError("Stored monotonic timestamp anchor must be finite.")

    snapshot_provenance = {
        "snapshot_id": snapshot_id,
        "provenance_available": provenance_available,
        "event_reference_type": event_reference_type,
        "source_timestamp_clock": timestamp_clock,
        "source_timestamps_utc": normalized_source_timestamps_utc,
        "source_timestamps_monotonic": {
            modality: float(timestamp)
            for modality, timestamp in source_timestamps_monotonic.items()
        },
        "source_timestamp_anchor_utc": timestamp_anchor_utc,
        "source_timestamp_anchor_monotonic": (
            float(timestamp_anchor_monotonic)
            if timestamp_anchor_monotonic is not None
            else None
        ),
        "created_at_utc": created_at_utc,
    }
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
            **snapshot_provenance,
        },
        record_kind="reviewed_feedback",
        created_at_utc=created_at_utc,
    )
    sufficient, reason = EvidenceSufficiencyPolicy().evaluate(record)
    if not sufficient:
        raise ValueError(f"Insufficient evidence: {reason}")
    return record, source, record_kind, snapshot_provenance


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
    manifest_snapshot_id = manifest.get("snapshot_id")
    if not isinstance(manifest_snapshot_id, str) or not manifest_snapshot_id:
        raise ValueError("Manifest snapshot_id is required to identify the captured feature record.")

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
    if window_start > window_end:
        raise ValueError("The event window start must not be later than its end.")
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
    (
        record,
        feature_source,
        feature_record_kind,
        snapshot_provenance,
    ) = _validate_feature_summary(summary_payload, feedback_id)
    snapshot_id = snapshot_provenance["snapshot_id"]
    if not snapshot_provenance["provenance_available"]:
        raise ValueError("Stored feature snapshot provenance is unavailable.")
    if snapshot_id != manifest_snapshot_id:
        raise ValueError("Manifest snapshot ID does not match the stored feature snapshot.")
    expected_reference_type = (
        "caretaker_reported_missed_event"
        if action == "missed_event"
        else "alert_triggering_snapshot"
    )
    if snapshot_provenance["event_reference_type"] != expected_reference_type:
        raise ValueError(
            "Stored feature snapshot is not associated with this feedback action."
        )
    if snapshot_provenance["source_timestamp_clock"] != "monotonic":
        raise ValueError("Stored feature snapshot has an unsupported source timestamp clock.")
    created_at = snapshot_provenance["created_at_utc"]
    anchor_utc_value = snapshot_provenance["source_timestamp_anchor_utc"]
    anchor_monotonic = snapshot_provenance["source_timestamp_anchor_monotonic"]
    modality_times_utc = snapshot_provenance["source_timestamps_utc"]
    modality_times_monotonic = snapshot_provenance[
        "source_timestamps_monotonic"
    ]
    if (
        created_at is None
        or anchor_utc_value is None
        or anchor_monotonic is None
        or not modality_times_utc
    ):
        raise ValueError("Stored feature snapshot is missing timestamp provenance.")
    created_at_time = _parse_utc_timestamp(created_at, "stored feature snapshot created_at_utc")
    anchor_utc = _parse_utc_timestamp(
        anchor_utc_value,
        "stored source_timestamp_anchor_utc",
    )

    modality_timestamps = manifest.get("modality_timestamps_utc")
    if not isinstance(modality_timestamps, dict):
        raise ValueError("modality_timestamps_utc is required to verify evidence freshness.")
    available_modalities = {
        modality
        for modality, available in record.available_modalities.items()
        if available
    }
    if (
        set(modality_timestamps) != available_modalities
        or set(modality_times_utc) != available_modalities
        or set(modality_times_monotonic) != available_modalities
    ):
        raise ValueError(
            "Persisted and reviewed modality timestamps must cover exactly the available modalities."
        )
    normalized_modality_timestamps: dict[str, str] = {}
    for modality in CANONICAL_MODES:
        if modality not in available_modalities:
            continue
        timestamp = _parse_utc_timestamp(
            modality_timestamps[modality],
            f"modality_timestamps_utc[{modality!r}]",
        )
        persisted_timestamp = _parse_utc_timestamp(
            modality_times_utc[modality],
            f"stored source_timestamps_utc[{modality!r}]",
        )
        if timestamp != persisted_timestamp:
            raise ValueError(
                f"{modality} review timestamp does not match the stored feature snapshot."
            )
        reconstructed_timestamp = anchor_utc + timedelta(
            seconds=modality_times_monotonic[modality] - anchor_monotonic
        )
        if abs((persisted_timestamp - reconstructed_timestamp).total_seconds()) > 0.002:
            raise ValueError(
                f"{modality} UTC timestamp is inconsistent with its stored monotonic timestamp."
            )
        if persisted_timestamp > created_at_time:
            raise ValueError(
                f"{modality} source timestamp is later than the feature snapshot creation time."
            )
        if timestamp < window_start or timestamp > window_end:
            raise ValueError(f"{modality} timestamp falls outside the reviewed event window.")
        if (window_end - timestamp).total_seconds() > MAX_REVIEWED_MODALITY_AGE_SECONDS:
            raise ValueError(
                f"{modality} evidence is older than the "
                f"{MAX_REVIEWED_MODALITY_AGE_SECONDS:g}-second freshness limit."
            )
        normalized_modality_timestamps[modality] = timestamp.isoformat()
    source_times = [
        _parse_utc_timestamp(value, "stored modality source timestamp")
        for value in modality_times_utc.values()
    ]
    if window_start != min(source_times) or window_end != max(source_times):
        raise ValueError(
            "Reviewed event window must exactly span the persisted modality timestamps."
        )
    if window_end > created_at_time:
        raise ValueError(
            "Reviewed event window ends after the feature snapshot was created."
        )

    provenance = {
        "feedback_id": feedback_id,
        "snapshot_id": snapshot_id,
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
        "snapshot_created_at_utc": created_at_time.isoformat(),
        "source_timestamp_clock": snapshot_provenance["source_timestamp_clock"],
        "source_timestamps_monotonic": modality_times_monotonic,
        "source_timestamp_anchor_utc": anchor_utc.isoformat(),
        "source_timestamp_anchor_monotonic": anchor_monotonic,
    }
    record.provenance = {
        "source": "reviewed_feedback_manifest",
        "feedback_id": feedback_id,
        "feedback_action": action,
        "snapshot_id": snapshot_id,
        "snapshot_id": snapshot_id,
        "feedback_submitted_at_utc": submitted_at.isoformat(),
        "review_decision": decision,
        "reviewed_at_utc": reviewed_at.isoformat(),
        "evidence_context_reference": context_reference,
        "event_window_start_utc": window_start.isoformat(),
        "event_window_end_utc": window_end.isoformat(),
        "modality_timestamps_utc": normalized_modality_timestamps,
        "source_timestamps_monotonic": modality_times_monotonic,
        "source_timestamp_clock": snapshot_provenance["source_timestamp_clock"],
        "source_timestamp_anchor_utc": anchor_utc.isoformat(),
        "source_timestamp_anchor_monotonic": anchor_monotonic,
        "snapshot_created_at_utc": created_at_time.isoformat(),
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


def feedback_review_statuses(
    feedback_store: FeedbackStore,
    manifest_path: str | Path,
) -> dict[str, FeedbackReviewStatus]:
    """Report review and eligibility states from the stored feedback and v2 manifest."""
    feedback_rows = feedback_store.list_feedback()
    path = Path(manifest_path)
    if not path.is_file():
        return {
            str(row["feedback_id"]): FeedbackReviewStatus(
                "Awaiting review", "Not eligible"
            )
            for row in feedback_rows
            if isinstance(row.get("feedback_id"), str)
        }

    entries, duplicate_ids, manifest_rejections = _load_manifest(path)
    loaded = load_reviewed_feedback(feedback_store, path)
    examples_by_id = {example.feedback_id: example for example in loaded.examples}
    rejections_by_id: dict[str, list[str]] = {}
    for rejection in (*manifest_rejections, *loaded.rejections):
        if rejection.feedback_id is not None:
            if rejection.reason.startswith(
                "No separate review-manifest entry; feedback action is not a training label."
            ):
                continue
            rejections_by_id.setdefault(rejection.feedback_id, []).append(
                rejection.reason
            )

    statuses: dict[str, FeedbackReviewStatus] = {}
    for row in feedback_rows:
        feedback_id = row.get("feedback_id")
        if not isinstance(feedback_id, str):
            continue
        entry_info = entries.get(feedback_id)
        if feedback_id in duplicate_ids:
            statuses[feedback_id] = FeedbackReviewStatus(
                "Ineligible",
                "Not eligible",
                reason="Duplicate manifest entries are not eligible.",
            )
        elif entry_info is None:
            reasons = rejections_by_id.get(feedback_id, [])
            statuses[feedback_id] = FeedbackReviewStatus(
                "Ineligible" if reasons else "Awaiting review",
                "Not eligible",
                reason=reasons[0] if reasons else None,
            )
        else:
            entry = entry_info[1]
            decision = entry.get("review_decision")
            label = (
                entry.get("training_label")
                if isinstance(entry.get("training_label"), str)
                else None
            )
            if decision == "excluded":
                statuses[feedback_id] = FeedbackReviewStatus(
                    "Excluded", "Not eligible", label
                )
            elif decision == "ambiguous":
                statuses[feedback_id] = FeedbackReviewStatus(
                    "Ambiguous", "Not eligible", label
                )
            elif feedback_id in examples_by_id:
                statuses[feedback_id] = FeedbackReviewStatus(
                    "Approved for training", "Eligible", label
                )
            else:
                reasons = rejections_by_id.get(feedback_id, [])
                statuses[feedback_id] = FeedbackReviewStatus(
                    "Ineligible",
                    "Not eligible",
                    label,
                    reasons[0] if reasons else "Review entry did not pass eligibility checks.",
                )
    return statuses


def _validate_review_entry(
    feedback: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> None:
    if set(entry) != _MANIFEST_KEYS:
        raise ValueError(
            "Review entry fields must exactly match the reviewed-feedback v2 manifest schema."
        )
    if entry.get("manifest_version") != REVIEW_MANIFEST_VERSION:
        raise ValueError(f"Manifest version must be {REVIEW_MANIFEST_VERSION!r}.")
    feedback_id = feedback.get("feedback_id")
    if not isinstance(feedback_id, str) or entry.get("feedback_id") != feedback_id:
        raise ValueError("Review feedback_id must match an existing stored feedback row.")
    provenance_block = feedback_review_block_reason(feedback, feedback_id)
    if provenance_block is not None:
        raise ValueError(provenance_block)
    decision = entry.get("review_decision")
    if not isinstance(decision, str) or decision not in REVIEW_DECISIONS:
        raise ValueError("Review decision must be approved_for_training, excluded, or ambiguous.")
    label = entry.get("training_label")
    if not isinstance(label, str) or label not in TARGET_LABEL_MAPPING:
        raise ValueError(
            "training_label must be potential_distress or no_sufficient_distress_evidence."
        )
    for field_name in ("reviewer_reference", "evidence_context_reference"):
        value = entry.get(field_name)
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 128
            or not value.isascii()
            or any(not (char.isalnum() or char in "._:-") for char in value)
        ):
            raise ValueError(
                f"{field_name} must be an opaque reference using letters, numbers, '.', '_', ':', or '-'."
            )

    try:
        summary = json.loads(feedback.get("feature_summary_json", ""))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Stored feedback feature summary is malformed.") from exc
    if not isinstance(summary, dict) or not isinstance(summary.get("provenance"), dict):
        raise ValueError("Stored feedback snapshot provenance is unavailable.")
    provenance = summary["provenance"]
    snapshot_id = provenance.get("snapshot_id")
    if (
        not provenance.get("provenance_available")
        or not isinstance(snapshot_id, str)
        or not snapshot_id
        or len(snapshot_id) > 512
        or not snapshot_id.isascii()
        or entry.get("snapshot_id") != snapshot_id
    ):
        raise ValueError("Review snapshot_id must match the available stored snapshot exactly.")
    action = feedback.get("action")
    alert_id = feedback.get("alert_id")
    if action not in {"confirm_alert", "reject_alert", "missed_event"}:
        raise ValueError("Stored feedback action is missing or unsupported.")
    if action == "missed_event" and alert_id is not None:
        raise ValueError("A missed-event report cannot be associated with an alert ID.")
    if action in {"confirm_alert", "reject_alert"} and (
        not isinstance(alert_id, str) or not alert_id
    ):
        raise ValueError(f"Stored {action} feedback is missing its alert ID.")
    if provenance.get("event_reference_type") != (
        "caretaker_reported_missed_event"
        if action == "missed_event"
        else "alert_triggering_snapshot"
    ):
        raise ValueError("Stored snapshot provenance does not match the feedback action.")
    if provenance.get("source_timestamp_clock") != "monotonic":
        raise ValueError("Stored feature snapshot has an unsupported source timestamp clock.")

    modality_timestamps = provenance.get("source_timestamps_utc")
    monotonic_timestamps = provenance.get("source_timestamps_monotonic")
    reviewed_modality_timestamps = entry.get("modality_timestamps_utc")
    if (
        not isinstance(modality_timestamps, dict)
        or not modality_timestamps
        or not isinstance(monotonic_timestamps, dict)
        or set(monotonic_timestamps) != set(modality_timestamps)
        or not isinstance(reviewed_modality_timestamps, dict)
        or set(reviewed_modality_timestamps) != set(modality_timestamps)
    ):
        raise ValueError(
            "modality_timestamps_utc must cover exactly the stored snapshot modalities."
        )
    normalized_times: dict[str, datetime] = {}
    for modality, stored_value in modality_timestamps.items():
        stored_time = _parse_utc_timestamp(
            stored_value, f"stored source_timestamps_utc[{modality!r}]"
        )
        reviewed_time = _parse_utc_timestamp(
            reviewed_modality_timestamps.get(modality),
            f"modality_timestamps_utc[{modality!r}]",
        )
        if stored_time != reviewed_time:
            raise ValueError(
                f"{modality} review timestamp does not match the stored feature snapshot."
            )
        normalized_times[modality] = stored_time
        monotonic_value = monotonic_timestamps[modality]
        if (
            isinstance(monotonic_value, bool)
            or not isinstance(monotonic_value, (int, float))
            or not math.isfinite(float(monotonic_value))
        ):
            raise ValueError(f"Stored monotonic timestamp for {modality!r} is invalid.")

    window_start = _parse_utc_timestamp(
        entry.get("event_window_start_utc"), "event_window_start_utc"
    )
    window_end = _parse_utc_timestamp(
        entry.get("event_window_end_utc"), "event_window_end_utc"
    )
    if (
        window_start != min(normalized_times.values())
        or window_end != max(normalized_times.values())
    ):
        raise ValueError(
            "The reviewed event window must exactly span the stored modality timestamps."
        )
    created_at = _parse_utc_timestamp(
        summary.get("created_at_utc"), "stored feature snapshot created_at_utc"
    )
    anchor_utc = _parse_utc_timestamp(
        provenance.get("source_timestamp_anchor_utc"),
        "stored source_timestamp_anchor_utc",
    )
    anchor_monotonic = provenance.get("source_timestamp_anchor_monotonic")
    if (
        isinstance(anchor_monotonic, bool)
        or not isinstance(anchor_monotonic, (int, float))
        or not math.isfinite(float(anchor_monotonic))
        or window_end > created_at
    ):
        raise ValueError("Stored feature snapshot has invalid or incomplete timestamp provenance.")
    for modality, timestamp in normalized_times.items():
        monotonic_value = float(monotonic_timestamps[modality])
        reconstructed = anchor_utc + timedelta(
            seconds=monotonic_value - float(anchor_monotonic)
        )
        if (
            abs((timestamp - reconstructed).total_seconds()) > 0.002
            or timestamp > created_at
        ):
            raise ValueError(
                f"Stored timestamp provenance for {modality!r} is inconsistent."
            )
    submitted_at = _parse_utc_timestamp(
        feedback.get("submitted_at_utc"), "stored submitted_at_utc"
    )
    reviewed_at = _parse_utc_timestamp(entry.get("reviewed_at_utc"), "reviewed_at_utc")
    if reviewed_at < submitted_at:
        raise ValueError("reviewed_at_utc cannot predate feedback submission.")

    if decision == "approved_for_training":
        _approved_example(feedback, entry)


def append_review_entry(
    feedback_store: FeedbackStore,
    manifest_path: str | Path,
    entry: Mapping[str, Any],
) -> Path:
    """Validate and append one v2 review entry without replacing existing entries."""
    feedback_id = entry.get("feedback_id")
    if not isinstance(feedback_id, str) or not feedback_id:
        raise ValueError("A valid feedback_id is required.")
    feedback_rows = feedback_store.list_feedback()
    matching_rows = [
        row for row in feedback_rows if row.get("feedback_id") == feedback_id
    ]
    if len(matching_rows) != 1:
        raise ValueError("The selected feedback row does not exist uniquely in local storage.")
    _validate_review_entry(matching_rows[0], entry)

    path = Path(manifest_path)
    with _REVIEW_WRITE_LOCK:
        existing_ids: set[str] = set()
        if path.exists():
            if not path.is_file():
                raise ValueError("The review-manifest path is not a regular file.")
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                raise OSError(f"Could not read existing review manifest: {exc}") from exc
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Existing review manifest has invalid JSON on line {line_number}."
                    ) from exc
                if not isinstance(existing, dict) or set(existing) != _MANIFEST_KEYS:
                    raise ValueError(
                        f"Existing review manifest line {line_number} does not match the v2 schema."
                    )
                existing_id = existing.get("feedback_id")
                if not isinstance(existing_id, str) or not existing_id:
                    raise ValueError(
                        f"Existing review manifest line {line_number} has no feedback_id."
                    )
                if existing_id in existing_ids:
                    raise ValueError(
                        "Existing review manifest contains duplicate feedback IDs."
                    )
                existing_ids.add(existing_id)
        if feedback_id in existing_ids:
            raise ValueError(
                "A review entry already exists for this feedback row; existing entries are never overwritten."
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(dict(entry), sort_keys=True, allow_nan=False)
        with path.open("a", encoding="utf-8", newline="\n") as destination:
            if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
                destination.write("\n")
            destination.write(serialized + "\n")
            destination.flush()
    return path


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
