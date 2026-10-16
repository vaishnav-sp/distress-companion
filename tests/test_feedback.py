import json

import pytest

from distress_score import DistressScoreResult, score_alert_triggered
from feedback import FeedbackStorageError, FeedbackStore
from fusion.schema import FEATURE_SCHEMA_VERSION, FeatureRecord, feature_order
from rule_based_detection import RuleBasedCueDetector, RuleDetectionConfig


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
        unavailable_reasons={
            name: "Feature was unavailable at submission."
            for name in names
            if not validity[name]
        },
        modality_status={"face": "ok", "behavior": "unavailable", "audio": "unavailable"},
        modality_messages={"face": "", "behavior": "", "audio": ""},
        provenance={"source": "live_monitoring"},
    )


def _audio_record(rms, onset):
    names = feature_order()
    values = {name: None for name in names}
    validity = {name: False for name in names}
    for name, value in (
        ("audio.rms_loudness", rms),
        ("audio.onset_score", onset),
    ):
        values[name] = value
        validity[name] = True
    return FeatureRecord(
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=names,
        feature_values=values,
        feature_validity=validity,
        unavailable_reasons={
            name: "Feature unavailable in this detector fixture."
            for name in names
            if not validity[name]
        },
        modality_status={"face": "waiting", "behavior": "waiting", "audio": "ok"},
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
    unavailable_feature = next(name for name in feature_order() if name != feature)
    assert summary["feature_values"][unavailable_feature] is None
    assert summary["feature_validity"][unavailable_feature] is False
    assert (
        summary["unavailable_reasons"][unavailable_feature]
        == "Feature was unavailable at submission."
    )
    assert summary["provenance"] == {"source": "live_monitoring"}
    assert "scenario_label" not in summary
    assert "raw_audio" not in row["feature_summary_json"]


def test_rejection_is_stored_as_a_distinct_label_for_its_alert(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    saved = store.submit(
        "reject_alert",
        idempotency_key="alert-reject",
        alert_id="alert-456",
        record=_record(),
    )

    row = store.list_feedback()[0]
    assert saved.inserted is True
    assert row["action"] == "reject_alert"
    assert row["alert_id"] == "alert-456"


def test_feedback_is_idempotent_and_allows_missed_event_without_alert(tmp_path):
    database_path = tmp_path / "feedback.sqlite3"
    store = FeedbackStore(database_path)
    first = store.submit("missed_event", idempotency_key="event-1", record=_record())
    repeated = store.submit("missed_event", idempotency_key="event-1", record=_record())

    assert first.inserted is True
    assert first.action == "missed_event"
    assert first.alert_id is None
    assert repeated.inserted is False
    assert first.feedback_id == repeated.feedback_id
    reloaded_rows = FeedbackStore(database_path).list_feedback()
    assert len(reloaded_rows) == 1
    assert reloaded_rows[0]["action"] == "missed_event"
    assert reloaded_rows[0]["alert_id"] is None
    assert reloaded_rows[0]["submitted_at_utc"]


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


def test_missing_record_provenance_is_explicit_and_not_fabricated(tmp_path):
    record = _record()
    record.provenance = {}
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.submit("missed_event", idempotency_key="no-source", record=record)

    summary = json.loads(store.list_feedback()[0]["feature_summary_json"])
    assert summary["provenance"] == {"source": "unspecified"}
    assert "raw_audio" not in summary


def test_feedback_does_not_change_score_or_audio_peak_alerts(tmp_path):
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="audio_only")
    )
    detector.evaluate(
        _audio_record(0.01, 0.0),
        modality_source_timestamps={"audio": 10.0, "behavior": 10.0},
        now=10.0,
    )
    audio_peak = detector.evaluate(
        _audio_record(0.3, 0.8),
        modality_source_timestamps={"audio": 11.0, "behavior": 11.0},
        now=11.0,
    )
    score_result = DistressScoreResult(
        score=70.0,
        status="Available",
        modalities={},
        included_modalities=(),
        excluded_reasons={},
    )
    score_alert_before = score_alert_triggered(score_result)

    FeedbackStore(tmp_path / "feedback.sqlite3").submit(
        "missed_event",
        idempotency_key="independent-missed-event",
        record=_record(),
    )

    assert score_alert_before is True
    assert score_alert_triggered(score_result) is score_alert_before
    assert audio_peak.status == "potential_distress"
    assert audio_peak.cue_indicators["audio_peak"] is True
