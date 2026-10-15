import json

import pytest

from feedback import FeedbackStorageError, FeedbackStore
from fusion.schema import FEATURE_SCHEMA_VERSION, FeatureRecord, feature_order


def _record():
    names = feature_order()
    values = {name: None for name in names}
    validity = {name: False for name in names}
    values[names[0]] = 0.0
    validity[names[0]] = True
    return FeatureRecord(
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=names,
        feature_values=values,
        feature_validity=validity,
        unavailable_reasons={},
        modality_status={"face": "ok", "behavior": "unavailable", "audio": "unavailable"},
        modality_messages={"face": "", "behavior": "", "audio": ""},
    )


def test_feedback_stores_minimal_features_and_preserves_zero(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    saved = store.submit(
        "confirm_alert",
        idempotency_key="confirm-123",
        alert_id="alert-123",
        record=_record(),
    )
    row = store.list_feedback()[0]
    summary = json.loads(row["feature_summary_json"])
    feature = feature_order()[0]

    assert saved.inserted is True
    assert row["action"] == "confirm_alert"
    assert row["alert_id"] == "alert-123"
    assert summary["feature_values"][feature] == 0.0
    assert summary["feature_validity"][feature] is True
    assert "scenario_label" not in summary
    assert "raw_audio" not in row["feature_summary_json"]


def test_feedback_is_idempotent_and_allows_missed_event_without_alert(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    first = store.submit("missed_event", idempotency_key="event-1", record=_record())
    repeated = store.submit("missed_event", idempotency_key="event-1", record=_record())

    assert first.inserted is True
    assert repeated.inserted is False
    assert first.feedback_id == repeated.feedback_id
    assert len(store.list_feedback()) == 1


def test_feedback_validates_alert_and_idempotency_inputs(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    with pytest.raises(ValueError, match="requires an alert_id"):
        store.submit("reject_alert", idempotency_key="reject-1", record=_record())
    with pytest.raises(ValueError, match="idempotency_key"):
        store.submit("missed_event", idempotency_key=" ", record=_record())


def test_feedback_storage_failure_is_reported(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    store = FeedbackStore(blocker / "feedback.sqlite3")

    with pytest.raises(FeedbackStorageError, match="initialize"):
        store.submit("missed_event", idempotency_key="failure-1", record=_record())
