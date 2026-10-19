"""Privacy-minimal local persistence for caretaker alert feedback."""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fusion.schema import FeatureRecord, ensure_record_schema

FeedbackAction = Literal["confirm_alert", "reject_alert", "missed_event"]
DEFAULT_FEEDBACK_PATH = Path.home() / ".distress_companion" / "feedback.sqlite3"


class FeedbackStorageError(RuntimeError):
    """Raised when feedback cannot be safely persisted locally."""


@dataclass(frozen=True)
class FeedbackSubmission:
    feedback_id: str
    inserted: bool
    action: FeedbackAction
    alert_id: str | None


def _minimal_feature_summary(
    record: FeatureRecord,
    *,
    action: FeedbackAction,
    provenance_available: bool | None,
) -> str:
    source = record.provenance.get("source")
    if not isinstance(source, str) or source not in {
        "live_monitoring",
        "monitoring_diagnostic_snapshot",
        "live_rule_based_monitoring",
    }:
        source = "unspecified"
    snapshot_id = record.provenance.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        snapshot_id = None
    source_timestamps_utc = record.provenance.get("source_timestamps_utc")
    if not isinstance(source_timestamps_utc, dict):
        source_timestamps_utc = {}
    source_timestamps_monotonic = record.provenance.get(
        "source_timestamps_monotonic"
    )
    if not isinstance(source_timestamps_monotonic, dict):
        source_timestamps_monotonic = {}
    modality_timestamps_utc = {
        str(name): value
        for name, value in source_timestamps_utc.items()
        if isinstance(name, str) and isinstance(value, str)
    }
    modality_timestamps_monotonic = {
        str(name): float(value)
        for name, value in source_timestamps_monotonic.items()
        if isinstance(name, str)
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    }
    available_modalities = {
        modality
        for modality, is_available in record.available_modalities.items()
        if is_available
    }
    complete_snapshot_provenance = bool(
        snapshot_id
        and isinstance(record.created_at_utc, str)
        and record.created_at_utc
        and record.provenance.get("source_timestamp_clock") == "monotonic"
        and modality_timestamps_utc
        and set(modality_timestamps_utc) == set(modality_timestamps_monotonic)
        and set(modality_timestamps_utc) == available_modalities
        and isinstance(
            record.provenance.get("source_timestamp_anchor_utc"), str
        )
        and isinstance(
            record.provenance.get("source_timestamp_anchor_monotonic"),
            (int, float),
        )
        and not isinstance(
            record.provenance.get("source_timestamp_anchor_monotonic"), bool
        )
        and math.isfinite(
            float(record.provenance.get("source_timestamp_anchor_monotonic"))
        )
    )
    provenance_is_available = complete_snapshot_provenance and (
        provenance_available is not False
    )
    if action == "missed_event":
        event_reference_type = (
            "caretaker_reported_missed_event"
            if provenance_is_available
            else "unavailable"
        )
    else:
        event_reference_type = (
            "alert_triggering_snapshot"
            if provenance_is_available
            else "unavailable"
        )
    summary = {
        "schema_version": record.schema_version,
        "record_kind": record.record_kind,
        "created_at_utc": record.created_at_utc,
        "provenance": {
            "source": source,
            "snapshot_id": snapshot_id,
            "provenance_available": provenance_is_available,
            "event_reference_type": event_reference_type,
            "source_timestamp_clock": record.provenance.get(
                "source_timestamp_clock"
            ),
            "source_timestamps_utc": modality_timestamps_utc,
            "source_timestamps_monotonic": modality_timestamps_monotonic,
            "source_timestamp_anchor_utc": record.provenance.get(
                "source_timestamp_anchor_utc"
            ),
            "source_timestamp_anchor_monotonic": record.provenance.get(
                "source_timestamp_anchor_monotonic"
            ),
        },
        "feature_values": {
            name: (
                float(record.feature_values[name])
                if record.feature_validity.get(name, False)
                and record.feature_values.get(name) is not None
                else None
            )
            for name in record.feature_names
        },
        "feature_validity": {
            name: bool(record.feature_validity.get(name, False))
            for name in record.feature_names
        },
        "unavailable_reasons": {
            name: str(
                record.unavailable_reasons.get(name)
                or "No valid feature value was available at submission."
            )
            for name in record.feature_names
            if not record.feature_validity.get(name, False)
        },
    }
    return json.dumps(summary, sort_keys=True, separators=(",", ":"))


class FeedbackStore:
    """Write confirm/reject/missed-event feedback without raw sensor data or PII."""

    def __init__(self, database_path: str | Path = DEFAULT_FEEDBACK_PATH) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database_path, timeout=5.0)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS caretaker_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    action TEXT NOT NULL CHECK (
                        action IN ('confirm_alert', 'reject_alert', 'missed_event')
                    ),
                    alert_id TEXT,
                    submitted_at_utc TEXT NOT NULL,
                    feature_summary_json TEXT NOT NULL
                )
                """
            )
            connection.commit()
            return connection
        except (OSError, sqlite3.Error) as exc:
            raise FeedbackStorageError(f"Could not initialize local feedback storage: {exc}") from exc

    def submit(
        self,
        action: FeedbackAction,
        *,
        idempotency_key: str,
        record: FeatureRecord,
        alert_id: str | None = None,
        provenance_available: bool | None = None,
    ) -> FeedbackSubmission:
        if action not in {"confirm_alert", "reject_alert", "missed_event"}:
            raise ValueError(f"Unsupported feedback action: {action!r}.")
        if action in {"confirm_alert", "reject_alert"} and not alert_id:
            raise ValueError(f"{action} requires an alert_id.")
        if not idempotency_key.strip():
            raise ValueError("An idempotency_key is required.")
        validated_record = ensure_record_schema(record)
        if provenance_available is not None and not isinstance(
            provenance_available, bool
        ):
            raise TypeError("provenance_available must be a boolean or None.")
        feature_summary = _minimal_feature_summary(
            validated_record,
            action=action,
            provenance_available=provenance_available,
        )
        feedback_id = str(uuid.uuid4())
        submitted_at = datetime.now(timezone.utc).isoformat()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT feedback_id, action, alert_id, feature_summary_json
                FROM caretaker_feedback WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing[1] != action or existing[2] != alert_id:
                    raise ValueError("The idempotency key was already used for different feedback.")
                if existing[3] != feature_summary:
                    raise ValueError(
                        "The idempotency key was already used for a different feature snapshot."
                    )
                return FeedbackSubmission(existing[0], False, action, alert_id)
            connection.execute(
                """
                INSERT INTO caretaker_feedback (
                    feedback_id, idempotency_key, action, alert_id,
                    submitted_at_utc, feature_summary_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    idempotency_key,
                    action,
                    alert_id,
                    submitted_at,
                    feature_summary,
                ),
            )
            connection.commit()
            return FeedbackSubmission(feedback_id, True, action, alert_id)
        except ValueError:
            connection.rollback()
            raise
        except sqlite3.Error as exc:
            connection.rollback()
            raise FeedbackStorageError(f"Could not save caretaker feedback: {exc}") from exc
        finally:
            connection.close()

    def list_feedback(self) -> list[dict[str, str | None]]:
        """Read stored feedback without creating a database, table, or directory."""
        if not self.database_path.is_file():
            return []
        try:
            database_uri = f"{self.database_path.resolve().as_uri()}?mode=ro"
            connection = sqlite3.connect(database_uri, uri=True, timeout=5.0)
        except (OSError, sqlite3.Error) as exc:
            raise FeedbackStorageError(f"Could not read local feedback storage: {exc}") from exc
        try:
            rows = connection.execute(
                """
                SELECT feedback_id, action, alert_id, submitted_at_utc, feature_summary_json
                FROM caretaker_feedback ORDER BY submitted_at_utc, feedback_id
                """
            ).fetchall()
            return [
                {
                    "feedback_id": row[0],
                    "action": row[1],
                    "alert_id": row[2],
                    "submitted_at_utc": row[3],
                    "feature_summary_json": row[4],
                }
                for row in rows
            ]
        except sqlite3.Error as exc:
            raise FeedbackStorageError(f"Could not read caretaker feedback: {exc}") from exc
        finally:
            connection.close()
