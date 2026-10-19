"""Alert feedback context and submission helpers for the dashboard."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from feedback.store import FeedbackStore, FeedbackSubmission
from fusion.schema import FeatureRecord


@dataclass(frozen=True)
class AlertFeedbackContext:
    alert_id: str
    alert: object
    record: FeatureRecord | None
    provenance_available: bool = False


@dataclass(frozen=True)
class FeedbackAlertReference:
    alert_id: str
    activated_at_utc: str | None = None


def record_for_score_snapshot(
    snapshot: Mapping[str, Any],
    score_record: FeatureRecord,
) -> tuple[FeatureRecord, bool]:
    """Reuse inference provenance only when it identifies this exact score input."""
    retained_record = snapshot.get("latest_feature_record")
    if not isinstance(retained_record, FeatureRecord):
        return score_record, False
    if (
        retained_record.schema_version != score_record.schema_version
        or retained_record.feature_names != score_record.feature_names
        or retained_record.feature_values != score_record.feature_values
        or retained_record.feature_validity != score_record.feature_validity
    ):
        return score_record, False

    provenance = retained_record.provenance
    timestamps = provenance.get("source_timestamps_monotonic")
    modalities = {
        modality
        for modality, available in retained_record.available_modalities.items()
        if available
    }
    if not isinstance(timestamps, dict) or set(timestamps) != modalities:
        return score_record, False
    for modality in modalities:
        snapshot_timestamp = snapshot.get(f"last_{modality}_success_timestamp")
        retained_timestamp = timestamps.get(modality)
        if (
            isinstance(snapshot_timestamp, bool)
            or not isinstance(snapshot_timestamp, (int, float))
            or not math.isfinite(float(snapshot_timestamp))
            or isinstance(retained_timestamp, bool)
            or not isinstance(retained_timestamp, (int, float))
            or not math.isfinite(float(retained_timestamp))
            or float(snapshot_timestamp) != float(retained_timestamp)
        ):
            return score_record, False
    return retained_record, True


def alert_feedback_context_for_snapshot(
    alert_id: str,
    record: FeatureRecord | None,
    *,
    activated_at_utc: str | None = None,
    provenance_available: bool = False,
) -> AlertFeedbackContext | None:
    if not isinstance(alert_id, str) or not alert_id:
        return None
    if record is not None and not isinstance(record, FeatureRecord):
        return None
    if not isinstance(provenance_available, bool):
        return None
    return AlertFeedbackContext(
        alert_id,
        FeedbackAlertReference(alert_id, activated_at_utc),
        record,
        provenance_available,
    )


def retained_alert_feedback_context(
    alert_snapshot: Any,
) -> AlertFeedbackContext | None:
    if not isinstance(alert_snapshot, dict):
        return None
    alert_id = alert_snapshot.get("alert_id")
    alert = alert_snapshot.get("alert")
    if (
        not isinstance(alert_id, str)
        or not alert_id
        or getattr(alert, "alert_id", None) != alert_id
    ):
        return None
    record = alert_snapshot.get("record")
    if record is not None and not isinstance(record, FeatureRecord):
        return None
    return AlertFeedbackContext(alert_id, alert, record, True)


def submit_alert_feedback(
    feedback_store: FeedbackStore,
    context: AlertFeedbackContext,
    action: Literal["confirm_alert", "reject_alert"],
) -> FeedbackSubmission:
    if context.record is None:
        raise ValueError("The alert's triggering feature snapshot is unavailable.")
    return feedback_store.submit(
        action,
        idempotency_key=f"{context.alert_id}:{action.removesuffix('_alert')}",
        alert_id=context.alert_id,
        record=context.record,
        provenance_available=context.provenance_available,
    )