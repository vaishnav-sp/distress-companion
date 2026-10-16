"""Privacy-minimal local persistence for caretaker alert feedback."""

from __future__ import annotations

import json
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


def _minimal_feature_summary(record: FeatureRecord) -> str:
    source = record.provenance.get("source")
    if not isinstance(source, str) or source not in {
        "live_monitoring",
        "monitoring_diagnostic_snapshot",
    }:
        source = "unspecified"
    summary = {
        "schema_version": record.schema_version,
        "record_kind": record.record_kind,
        "provenance": {"source": source},
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
    ) -> FeedbackSubmission:
        if action not in {"confirm_alert", "reject_alert", "missed_event"}:
            raise ValueError(f"Unsupported feedback action: {action!r}.")
        if action in {"confirm_alert", "reject_alert"} and not alert_id:
            raise ValueError(f"{action} requires an alert_id.")
        if not idempotency_key.strip():
            raise ValueError("An idempotency_key is required.")
        validated_record = ensure_record_schema(record)
        feature_summary = _minimal_feature_summary(validated_record)
        feedback_id = str(uuid.uuid4())
        submitted_at = datetime.now(timezone.utc).isoformat()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT feedback_id, action, alert_id
                FROM caretaker_feedback WHERE idempotency_key = ?
                """,
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing[1] != action or existing[2] != alert_id:
                    raise ValueError("The idempotency key was already used for different feedback.")
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
        """Return stored feedback metadata without exposing idempotency keys."""
        connection = self._connect()
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
