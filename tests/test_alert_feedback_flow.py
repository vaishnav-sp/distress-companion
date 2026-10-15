from types import SimpleNamespace

from alert_management import AlertManager, AlertPolicy
from feedback import FeedbackStore
from fusion import build_fusion_evidence, evidence_to_feature_record
from fusion.schema import FEATURE_SCHEMA_VERSION, FeatureRecord, feature_order
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
