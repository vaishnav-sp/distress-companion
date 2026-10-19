import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from distress_score import DistressScoreResult, score_alert_triggered
from feedback import FeedbackStorageError, FeedbackStore
from feedback.presentation import (
    alert_feedback_context_for_snapshot,
    record_for_score_snapshot,
    submit_alert_feedback,
)
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


def _record_with_snapshot(snapshot_id):
    record = _record()
    behavior_feature = "behavior.movement_energy"
    record.feature_values[behavior_feature] = 0.1
    record.feature_validity[behavior_feature] = True
    record.unavailable_reasons.pop(behavior_feature, None)
    anchor_utc = datetime.now(timezone.utc)
    anchor_monotonic = 100.0
    source_timestamps_monotonic = {
        "face": anchor_monotonic - 0.2,
        "behavior": anchor_monotonic - 0.1,
    }
    record.created_at_utc = (anchor_utc + timedelta(milliseconds=10)).isoformat()
    record.provenance.update(
        {
            "source": "live_rule_based_monitoring",
            "snapshot_id": snapshot_id,
            "source_timestamp_clock": "monotonic",
            "source_timestamp_anchor_utc": anchor_utc.isoformat(),
            "source_timestamp_anchor_monotonic": anchor_monotonic,
            "source_timestamps_monotonic": source_timestamps_monotonic,
            "source_timestamps_utc": {
                modality: (
                    anchor_utc
                    + timedelta(seconds=timestamp - anchor_monotonic)
                ).isoformat()
                for modality, timestamp in source_timestamps_monotonic.items()
            },
        }
    )
    return record


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
    assert summary["provenance"]["source"] == "live_monitoring"
    assert summary["provenance"]["provenance_available"] is False
    assert summary["provenance"]["snapshot_id"] is None
    assert summary["created_at_utc"] is None
    assert "scenario_label" not in summary
    assert "raw_audio" not in row["feature_summary_json"]


def test_score_popup_reuses_only_exact_provenance_bearing_snapshot(tmp_path):
    retained_record = _record_with_snapshot("inference-snapshot")
    score_record = deepcopy(retained_record)
    score_record.provenance = {"source": "monitoring_diagnostic_snapshot"}
    snapshot = {
        "latest_feature_record": retained_record,
        "last_face_success_timestamp": 99.8,
        "last_behavior_success_timestamp": 99.9,
    }

    selected_record, provenance_available = record_for_score_snapshot(
        snapshot,
        score_record,
    )

    assert selected_record is retained_record
    assert provenance_available is True
    context = alert_feedback_context_for_snapshot(
        "popup-alert",
        selected_record,
        provenance_available=provenance_available,
    )
    store = FeedbackStore(tmp_path / "exact-snapshot.sqlite3")
    submitted = submit_alert_feedback(store, context, "confirm_alert")
    duplicate = submit_alert_feedback(store, context, "confirm_alert")
    summary = json.loads(store.list_feedback()[0]["feature_summary_json"])

    assert submitted.inserted
    assert not duplicate.inserted
    assert summary["provenance"]["snapshot_id"] == "inference-snapshot"
    assert summary["provenance"]["provenance_available"] is True
    assert summary["provenance"]["event_reference_type"] == (
        "alert_triggering_snapshot"
    )


def test_score_popup_marks_mismatched_later_snapshot_provenance_unavailable():
    later_record = _record_with_snapshot("later-inference-snapshot")
    score_record = _record_with_snapshot("score-time-record")
    score_record.provenance = {"source": "monitoring_diagnostic_snapshot"}
    snapshot = {
        "latest_feature_record": later_record,
        "last_face_success_timestamp": 99.8,
        "last_behavior_success_timestamp": 99.9,
    }
    snapshot["latest_feature_record"].provenance[
        "source_timestamps_monotonic"
    ]["face"] = 99.7

    selected_record, provenance_available = record_for_score_snapshot(
        snapshot,
        score_record,
    )

    assert selected_record is score_record
    assert provenance_available is False
    assert "snapshot_id" not in selected_record.provenance


def test_app_binds_popup_feedback_to_the_captured_score_snapshot():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )
    popup_trigger = source.split("if popup_message:", 1)[1].split(
        'st.subheader("Multimodal Distress Alert")', 1
    )[0]

    assert "popup_evidence = build_fusion_evidence(snapshot)" in popup_trigger
    assert "record_for_score_snapshot(snapshot, score_snapshot_record)" in popup_trigger
    assert '"snapshot_id": popup_alert_id' not in popup_trigger


@pytest.mark.parametrize(
    ("action", "event_reference_type"),
    [
        ("confirm_alert", "alert_triggering_snapshot"),
        ("reject_alert", "alert_triggering_snapshot"),
        ("missed_event", "caretaker_reported_missed_event"),
    ],
)
def test_feedback_persists_snapshot_provenance_for_each_action(
    tmp_path, action, event_reference_type
):
    store = FeedbackStore(tmp_path / f"{action}.sqlite3")
    alert_id = "alert-123" if action != "missed_event" else None
    store.submit(
        action,
        idempotency_key=f"{action}-snapshot",
        alert_id=alert_id,
        record=_record_with_snapshot(f"snapshot-{action}"),
        provenance_available=True,
    )

    summary = json.loads(store.list_feedback()[0]["feature_summary_json"])
    provenance = summary["provenance"]

    assert summary["created_at_utc"]
    assert provenance["snapshot_id"] == f"snapshot-{action}"
    assert provenance["provenance_available"] is True
    assert provenance["event_reference_type"] == event_reference_type
    assert set(provenance["source_timestamps_utc"]) == {"face", "behavior"}
    assert set(provenance["source_timestamps_monotonic"]) == {"face", "behavior"}


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


def test_idempotency_rejects_reuse_for_a_different_snapshot(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.submit(
        "confirm_alert",
        idempotency_key="same-alert",
        alert_id="alert-1",
        record=_record(),
    )
    changed_record = _record()
    changed_record.feature_values[feature_order()[0]] = 0.5

    with pytest.raises(ValueError, match="different feature snapshot"):
        store.submit(
            "confirm_alert",
            idempotency_key="same-alert",
            alert_id="alert-1",
            record=changed_record,
        )


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
    assert summary["provenance"]["source"] == "unspecified"
    assert summary["provenance"]["provenance_available"] is False
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
