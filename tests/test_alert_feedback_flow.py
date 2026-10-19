import json
from types import SimpleNamespace
import time

from alert_management import AlertManager, AlertPolicy
from feedback import FeedbackStore
from feedback.presentation import (
    alert_feedback_context_for_snapshot,
    retained_alert_feedback_context,
    submit_alert_feedback,
)
from fusion import build_fusion_evidence, evidence_to_feature_record
from fusion.schema import FEATURE_SCHEMA_VERSION, FeatureRecord, feature_order
from monitoring import MonitoringManager
from training import infer_distress_status, train_distress_logistic_model


def _face_training_record(scenario, feature_values):
    names = feature_order()
    values = {name: None for name in names}
    validity = {name: False for name in names}
    for name, value in feature_values.items():
        values[name] = value
        validity[name] = True
    return FeatureRecord(
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=names,
        feature_values=values,
        feature_validity=validity,
        unavailable_reasons={},
        modality_status={"face": "ok", "behavior": "waiting", "audio": "waiting"},
        modality_messages={"face": "", "behavior": "", "audio": ""},
        scenario_label=scenario,
        record_kind="synthetic_acted",
        is_acted_synthetic=True,
    )


def test_feature_record_to_model_inference_alert_and_feedback(tmp_path):
    monitoring = MonitoringManager(
        face_branch_factory=lambda: None,
        behavior_branch_factory=lambda: None,
        audio_branch_factory=lambda: None,
    )
    monitoring_state_before_feedback = monitoring.snapshot()
    distress_profile = {
        "face.expression_angry": 0.8,
        "face.expression_fear": 0.1,
        "face.expression_happy": 0.02,
        "face.expression_neutral": 0.02,
        "face.expression_sad": 0.04,
        "face.expression_surprise": 0.02,
        "face.negative_affect_mean_angry_fear_sad": 0.31,
        "face.face_quality_score": 0.9,
    }
    baseline_profile = {
        "face.expression_angry": 0.05,
        "face.expression_fear": 0.03,
        "face.expression_happy": 0.7,
        "face.expression_neutral": 0.15,
        "face.expression_sad": 0.03,
        "face.expression_surprise": 0.04,
        "face.negative_affect_mean_angry_fear_sad": 0.0367,
        "face.face_quality_score": 0.9,
    }
    training_records = [
        _face_training_record("distress_case", distress_profile),
        _face_training_record("distress_case", distress_profile),
        _face_training_record("baseline_case", baseline_profile),
        _face_training_record("baseline_case", baseline_profile),
    ]
    model = train_distress_logistic_model(
        training_records,
        use_group_split=False,
        label_lookup={
            "distress_case": "potential_distress",
            "baseline_case": "no_sufficient_distress_evidence",
        },
        allow_acted_synthetic=True,
        model_path=tmp_path / "pipeline.joblib",
    )
    face_result = SimpleNamespace(
        expression_probabilities={
            "angry": 0.8,
            "disgust": None,
            "fear": 0.1,
            "happy": 0.02,
            "neutral": 0.02,
            "sad": 0.04,
            "surprise": 0.02,
        },
        class_validity={
            "angry": True,
            "disgust": False,
            "fear": True,
            "happy": True,
            "neutral": True,
            "sad": True,
            "surprise": True,
        },
        feature_vector_order=(
            "angry",
            "disgust",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
        ),
        negative_affect_feature=0.31,
        face_quality_score=0.9,
        status="ok",
        status_message="Face features are available.",
    )
    behavior_result = SimpleNamespace(
        feature_values={
            "movement_energy": 1.2,
            "sudden_movement": 0.8,
            "movement_change_from_baseline": 0.9,
            "shoulder_alignment": 1.0,
            "shoulder_asymmetry": 4.0,
            "left_upper_arm_angle_deg": 50.0,
            "right_upper_arm_angle_deg": 52.0,
            "left_forearm_angle_deg": 68.0,
            "right_forearm_angle_deg": 66.0,
            "head_offset_x": 2.0,
            "head_offset_y": 3.0,
            "head_orientation_deg": 5.0,
        },
        feature_validity={
            "movement_energy": True,
            "sudden_movement": True,
            "movement_change_from_baseline": True,
            "shoulder_alignment": True,
            "shoulder_asymmetry": True,
            "left_upper_arm_angle_deg": True,
            "right_upper_arm_angle_deg": True,
            "left_forearm_angle_deg": True,
            "right_forearm_angle_deg": True,
            "head_offset_x": True,
            "head_offset_y": True,
            "head_orientation_deg": True,
        },
        feature_names=tuple(
            [
                "movement_energy",
                "sudden_movement",
                "movement_change_from_baseline",
                "shoulder_alignment",
                "shoulder_asymmetry",
                "left_upper_arm_angle_deg",
                "right_upper_arm_angle_deg",
                "left_forearm_angle_deg",
                "right_forearm_angle_deg",
                "head_offset_x",
                "head_offset_y",
                "head_orientation_deg",
            ]
        ),
        unavailable_reasons={},
        status="ok",
        status_message="Behavior features are available.",
    )
    live_evidence = build_fusion_evidence(
        {
            "latest_face_result": face_result,
            "latest_behavior_result": behavior_result,
            "latest_audio_result": None,
        }
    )
    live_record = evidence_to_feature_record(live_evidence)
    inference = infer_distress_status(live_record, model=model)
    assert inference.status == "potential_distress"
    assert inference.potential_distress_score is not None

    alerts = AlertManager(
        AlertPolicy(
            raw_probability_threshold=0.4,
            release_threshold=0.2,
            smoothing_window=1,
        )
    )
    alert = alerts.update(inference, result_timestamp=10.0, now=10.0)
    assert alert.state == "potential_distress"
    assert alert.activated_at_utc is not None
    alert_state_before_feedback = alerts.diagnostic_state()
    model_coefficients_before_feedback = (
        model.pipeline.named_steps["classifier"].coef_.copy()
    )
    model_intercepts_before_feedback = (
        model.pipeline.named_steps["classifier"].intercept_.copy()
    )
    feature_record_before_feedback = live_record.to_json()

    saved = FeedbackStore(tmp_path / "feedback.sqlite3").submit(
        "confirm_alert",
        idempotency_key=f"{alert.alert_id}:confirm",
        alert_id=alert.alert_id,
        record=live_record,
    )
    assert saved.inserted
    rejected = FeedbackStore(tmp_path / "feedback.sqlite3").submit(
        "reject_alert",
        idempotency_key=f"{alert.alert_id}:reject",
        alert_id=alert.alert_id,
        record=live_record,
    )
    assert rejected.inserted
    assert rejected.alert_id == alert.alert_id
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    store.submit(
        "missed_event",
        idempotency_key="independent-missed-event",
        record=live_record,
    )

    assert alerts.diagnostic_state() == alert_state_before_feedback
    assert (
        model.pipeline.named_steps["classifier"].coef_
        == model_coefficients_before_feedback
    ).all()
    assert (
        model.pipeline.named_steps["classifier"].intercept_
        == model_intercepts_before_feedback
    ).all()
    assert live_record.to_json() == feature_record_before_feedback
    assert monitoring.snapshot() == monitoring_state_before_feedback
    stored_rows = store.list_feedback()
    assert {row["action"] for row in stored_rows} == {
        "confirm_alert",
        "reject_alert",
        "missed_event",
    }
    missed_report = next(row for row in stored_rows if row["action"] == "missed_event")
    assert missed_report["alert_id"] is None


def test_monitoring_retains_the_exact_record_that_activated_an_alert():
    records = []
    for snapshot_id in (
        "triggering-window",
        "later-active-window",
        "release-window-one",
        "release-window-two",
    ):
        record = _face_training_record(
            snapshot_id,
            {
                "face.expression_angry": 0.8,
                "face.expression_fear": 0.1,
                "behavior.movement_energy": 0.9,
                "behavior.sudden_movement": 0.8,
            },
        )
        record.provenance = {
            "source": "live_rule_based_monitoring",
            "snapshot_id": snapshot_id,
        }
        records.append(record)

    def process(snapshot):
        del snapshot
        record = records.pop(0)
        timestamp = time.monotonic()
        is_triggering_window = record.provenance["snapshot_id"] in {
            "triggering-window",
            "later-active-window",
        }
        return {
            "inference": SimpleNamespace(
                status=(
                    "potential_distress"
                    if is_triggering_window
                    else "no_sufficient_distress_evidence"
                ),
                potential_distress_score=0.99 if is_triggering_window else 0.1,
                valid_feature_count=4,
                available_modalities={
                    "face": True,
                    "behavior": True,
                    "audio": False,
                },
                reason=None,
            ),
            "feature_record": record,
            "source_timestamp": timestamp,
            "result_key": record.provenance["snapshot_id"],
            "updated_at_utc": record.created_at_utc,
        }

    manager = MonitoringManager(
        inference_processor=process,
        alert_manager=AlertManager(
            AlertPolicy(
                raw_probability_threshold=0.5,
                release_threshold=0.2,
                smoothing_window=1,
            )
        ),
    )
    manager.status = "running"
    manager.latest_face_result = object()
    manager.latest_behavior_result = object()
    manager.last_face_success_timestamp = time.monotonic()
    manager.last_behavior_success_timestamp = manager.last_face_success_timestamp

    manager.camera_capture_attempts += 1
    manager._refresh_live_inference()
    triggering_alert = manager.snapshot()["latest_alert"]
    assert triggering_alert.state == "potential_distress"

    manager.camera_capture_attempts += 1
    manager._refresh_live_inference()
    snapshot = manager.snapshot()

    assert snapshot["latest_feature_record"].provenance["snapshot_id"] == (
        "later-active-window"
    )
    assert snapshot["alert_feature_record"]["alert_id"] == triggering_alert.alert_id
    assert snapshot["alert_feature_record"]["alert"].alert_id == triggering_alert.alert_id
    assert snapshot["alert_feature_record"]["record"].provenance["snapshot_id"] == (
        "triggering-window"
    )

    manager.camera_capture_attempts += 1
    manager._refresh_live_inference()
    manager.camera_capture_attempts += 1
    manager._refresh_live_inference()
    released_snapshot = manager.snapshot()

    assert released_snapshot["latest_alert"].state == "cooldown"
    assert released_snapshot["latest_feature_record"].provenance["snapshot_id"] == (
        "release-window-two"
    )
    assert released_snapshot["alert_feature_record"]["alert_id"] == (
        triggering_alert.alert_id
    )
    assert released_snapshot["alert_feature_record"]["record"].provenance[
        "snapshot_id"
    ] == "triggering-window"


def test_resolved_alert_keeps_feedback_context_and_exact_triggering_snapshot(tmp_path):
    triggering_record = _face_training_record(
        "triggering-window", {"face.expression_angry": 0.8}
    )
    newer_record = _face_training_record(
        "later-window", {"face.expression_angry": 0.1}
    )
    alert = SimpleNamespace(
        alert_id="retained-alert-id",
        state="normal",
        activated_at_utc="2026-10-18T12:00:00+00:00",
    )
    context = retained_alert_feedback_context(
        {
            "alert_id": alert.alert_id,
            "alert": alert,
            "record": triggering_record,
        }
    )

    assert context is not None
    assert context.alert.state == "normal"
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    submitted = submit_alert_feedback(store, context, "confirm_alert")
    duplicate = submit_alert_feedback(store, context, "confirm_alert")

    assert submitted.inserted
    assert not duplicate.inserted
    rows = store.list_feedback()
    assert len(rows) == 1
    summary = json.loads(rows[0]["feature_summary_json"])
    assert summary["feature_values"]["face.expression_angry"] == 0.8
    assert summary["feature_values"]["face.expression_angry"] != (
        newer_record.feature_values["face.expression_angry"]
    )


def test_alert_feedback_requires_the_retained_record():
    alert = SimpleNamespace(alert_id="without-record", state="normal")
    context = retained_alert_feedback_context(
        {"alert_id": alert.alert_id, "alert": alert, "record": None}
    )

    assert context is not None
    try:
        submit_alert_feedback(FeedbackStore(":memory:"), context, "reject_alert")
    except ValueError as exc:
        assert "triggering feature snapshot is unavailable" in str(exc)
    else:
        raise AssertionError("Feedback without the retained alert snapshot was accepted.")


def test_alert_feedback_context_rejects_mismatched_alert_identity():
    alert = SimpleNamespace(alert_id="different-alert", state="potential_distress")

    assert retained_alert_feedback_context(
        {"alert_id": "requested-alert", "alert": alert, "record": None}
    ) is None


def test_app_renders_feedback_from_retained_alert_and_separate_missed_event_section():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )

    assert "if alert_feedback_context is not None:" in source
    assert 'st.subheader("Feedback for this Alert")' in source
    assert '"Confirm"' in source
    assert '"Reject"' in source
    assert 'st.subheader("Report a Missed Event")' in source
    assert 'key="feedback_missed_event"' in source


def test_score_popup_feedback_uses_captured_record_and_is_idempotent(tmp_path):
    captured_record = _face_training_record(
        "popup-trigger", {"face.expression_angry": 0.8}
    )
    later_record = _face_training_record(
        "later-dashboard", {"face.expression_angry": 0.1}
    )
    context = alert_feedback_context_for_snapshot(
        "score-popup-event",
        captured_record,
    )
    assert context is not None
    assert not context.provenance_available

    store = FeedbackStore(tmp_path / "popup-feedback.sqlite3")
    first = submit_alert_feedback(store, context, "confirm_alert")
    duplicate = submit_alert_feedback(store, context, "confirm_alert")

    assert first.inserted
    assert not duplicate.inserted
    summary = json.loads(store.list_feedback()[0]["feature_summary_json"])
    assert summary["feature_values"]["face.expression_angry"] == 0.8
    assert summary["provenance"]["provenance_available"] is False
    assert summary["feature_values"]["face.expression_angry"] != (
        later_record.feature_values["face.expression_angry"]
    )


def test_popup_dialog_contains_interactive_feedback_and_unavailable_state():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )
    popup = source.split('@st.dialog("New alert")', 1)[1].split(
        "st.title(", 1
    )[0]

    assert "def show_alert_popup(message: str, alert_id: str)" in popup
    assert "st.warning(message)" in popup
    assert '"Confirm"' in popup
    assert '"Reject"' in popup
    assert "disabled=not snapshot_available or already_submitted" in popup
    assert "feature snapshot associated with this alert is unavailable" in popup
    assert 'st.success(submitted_result["message"])' in popup
    assert 'st.error(submitted_result["message"])' in popup
