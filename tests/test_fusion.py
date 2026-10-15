from types import SimpleNamespace

from fusion import (
    build_fusion_diagnostic_snapshot,
    build_fusion_evidence,
    evidence_to_feature_record,
)
from training import evidence_sufficient


def test_build_fusion_evidence_aggregates_current_results():
    snapshot = {
        "latest_face_result": SimpleNamespace(
            status="ok",
            status_message="Face features ready.",
            feature_values={"angry": 0.2, "happy": 0.1},
            feature_validity={"angry": True, "happy": True},
            feature_names=("angry", "happy"),
            unavailable_reasons={"happy": ""},
        ),
        "latest_behavior_result": SimpleNamespace(
            status="ok",
            status_message="Behavior features ready.",
            feature_values={"movement_energy": 1.5},
            feature_validity={"movement_energy": True},
            feature_names=("movement_energy",),
            unavailable_reasons={},
        ),
        "latest_audio_result": SimpleNamespace(
            status="warming_up",
            status_message="Audio baseline warming up.",
            feature_values={"rms_loudness": 0.4},
            feature_validity={"rms_loudness": False},
            feature_names=("rms_loudness",),
            unavailable_reasons={"rms_loudness": "Baseline not ready."},
        ),
    }

    evidence = build_fusion_evidence(snapshot)

    assert evidence.has_data is True
    assert "face.angry" in evidence.feature_values
    assert evidence.feature_values["behavior.movement_energy"] == 1.5
    assert evidence.feature_validity["face.angry"] is True
    assert evidence.feature_validity["audio.rms_loudness"] is False
    assert evidence.vector[0] == 0.2
    assert evidence.modality_status["audio"] == "warming_up"


def test_validity_flag_without_value_is_normalized_to_unavailable_with_reason():
    evidence = build_fusion_evidence(
        {
            "latest_face_result": None,
            "latest_behavior_result": SimpleNamespace(
                status="partial",
                status_message="One movement field is malformed.",
                feature_values={"sudden_movement": None},
                feature_validity={"sudden_movement": True},
                feature_names=("sudden_movement",),
                unavailable_reasons={},
            ),
            "latest_audio_result": None,
        }
    )

    record = evidence_to_feature_record(evidence)

    assert evidence.feature_validity["behavior.sudden_movement"] is False
    assert "supplied no numeric value" in evidence.unavailable_reasons[
        "behavior.sudden_movement"
    ]
    assert record.feature_validity["behavior.sudden_movement"] is False
    assert record.unavailable_reasons["behavior.sudden_movement"] == (
        "Branch marked the feature valid but supplied no numeric value."
    )


def test_build_fusion_evidence_handles_missing_results():
    evidence = build_fusion_evidence({
        "latest_face_result": None,
        "latest_behavior_result": None,
        "latest_audio_result": None,
    })

    assert evidence.has_data is False
    assert evidence.feature_names == ()
    assert evidence.vector == ()


def test_build_fusion_evidence_uses_face_probabilities_when_present():
    snapshot = {
        "latest_face_result": SimpleNamespace(
            status="ok",
            status_message="Face features ready.",
            expression_probabilities={"angry": 0.5, "happy": 0.2},
            class_validity={"angry": True, "happy": False},
            feature_vector_order=("angry", "happy"),
            abstention_reason="Not enough signal.",
        ),
        "latest_behavior_result": None,
        "latest_audio_result": None,
    }

    evidence = build_fusion_evidence(snapshot)

    assert evidence.feature_values["face.angry"] == 0.5
    assert evidence.feature_validity["face.angry"] is True
    assert evidence.feature_validity["face.happy"] is False
    assert evidence.unavailable_reasons["face.happy"] == "Not enough signal."


def test_face_result_without_status_uses_validity_for_modality_status():
    result = SimpleNamespace(
        face_detected=True,
        expression_probabilities={
            "angry": 0.2,
            "fear": 0.1,
            "happy": 0.2,
            "neutral": 0.2,
            "sad": 0.2,
            "surprise": 0.1,
            "disgust": None,
        },
        class_availability={
            "angry": True,
            "fear": True,
            "happy": True,
            "neutral": True,
            "sad": True,
            "surprise": True,
            "disgust": False,
        },
        class_validity={
            "angry": True,
            "fear": True,
            "happy": True,
            "neutral": True,
            "sad": True,
            "surprise": True,
            "disgust": False,
        },
        feature_vector_order=(
            "angry",
            "fear",
            "happy",
            "neutral",
            "sad",
            "surprise",
            "disgust",
        ),
        negative_affect_feature=0.1667,
        face_quality_score=0.9,
        abstention_reason=None,
        status_message=(
            "Expression probabilities are model estimates, not proof of emotion "
            "or a distress decision."
        ),
    )

    evidence = build_fusion_evidence(
        {
            "latest_face_result": result,
            "latest_behavior_result": None,
            "latest_audio_result": None,
        }
    )
    record = evidence_to_feature_record(evidence)
    diagnostic = build_fusion_diagnostic_snapshot(
        evidence,
        record,
        source_timestamps={"face": 12.0, "behavior": None, "audio": None},
        sufficient=False,
        sufficiency_reason="Only one modality is present.",
    )

    assert evidence.modality_status["face"] == "ok"
    assert evidence.modality_messages["face"] == result.status_message
    assert evidence.feature_validity["face.angry"] is True
    assert diagnostic["modality_status"]["face"] == "ok"
    assert diagnostic["modality_messages"]["face"] == result.status_message


def test_face_abstentions_and_partial_results_have_distinct_statuses():
    def face_result(*, valid_classes, abstention_reason, face_detected=True):
        labels = ("angry", "fear", "happy")
        return SimpleNamespace(
            face_detected=face_detected,
            expression_probabilities={
                name: 0.3 if valid_classes.get(name) else None for name in labels
            },
            class_availability={name: True for name in labels},
            class_validity={name: valid_classes.get(name, False) for name in labels},
            feature_vector_order=labels,
            negative_affect_feature=None,
            face_quality_score=None,
            abstention_reason=abstention_reason,
            status_message="Face result status message.",
        )

    partial = build_fusion_evidence(
        {
            "latest_face_result": face_result(
                valid_classes={"angry": True},
                abstention_reason=None,
            ),
            "latest_behavior_result": None,
            "latest_audio_result": None,
        }
    )
    absent = build_fusion_evidence(
        {
            "latest_face_result": face_result(
                valid_classes={},
                abstention_reason="no_face",
                face_detected=False,
            ),
            "latest_behavior_result": None,
            "latest_audio_result": None,
        }
    )

    assert partial.modality_status["face"] == "partial"
    assert partial.modality_messages["face"] == "Face result status message."
    assert absent.modality_status["face"] == "missing"
    assert absent.modality_messages["face"] == "Face result status message."


def test_fusion_diagnostic_snapshot_preserves_values_status_reasons_and_timestamps():
    evidence = build_fusion_evidence(
        {
            "latest_face_result": SimpleNamespace(
                status="ok",
                status_message="Face ready.",
                expression_probabilities={
                    "angry": 0.2,
                    "fear": 0.1,
                    "happy": 0.2,
                    "neutral": 0.2,
                    "sad": 0.2,
                    "surprise": 0.1,
                },
                class_validity={
                    "angry": True,
                    "fear": True,
                    "happy": True,
                    "neutral": True,
                    "sad": True,
                    "surprise": True,
                },
                feature_vector_order=(
                    "angry",
                    "fear",
                    "happy",
                    "neutral",
                    "sad",
                    "surprise",
                ),
                negative_affect_feature=0.1667,
                face_quality_score=0.9,
            ),
            "latest_behavior_result": SimpleNamespace(
                status="warming_up",
                status_message="Behavior baseline warming up.",
                feature_values={"movement_energy": None},
                feature_validity={"movement_energy": False},
                feature_names=("movement_energy",),
                unavailable_reasons={"movement_energy": "Needs a prior frame."},
            ),
            "latest_audio_result": SimpleNamespace(
                status="ok",
                status_message="Audio ready.",
                feature_values={"rms_loudness": 0.4},
                feature_validity={"rms_loudness": True},
                feature_names=("rms_loudness",),
                unavailable_reasons={},
            ),
        }
    )
    record = evidence_to_feature_record(evidence)
    sufficient, reason = evidence_sufficient(record)
    diagnostic = build_fusion_diagnostic_snapshot(
        evidence,
        record,
        source_timestamps={"face": 1.0, "behavior": 1.0, "audio": 1.5},
        sufficient=sufficient,
        sufficiency_reason=reason,
    )

    assert diagnostic["construction_status"] == "constructed_with_valid_multimodal_features"
    assert diagnostic["constructed_at_utc"] == record.created_at_utc
    assert diagnostic["source_timestamps"] == {
        "face": 1.0,
        "behavior": 1.0,
        "audio": 1.5,
    }
    assert diagnostic["feature_values"]["face.angry"] == 0.2
    assert diagnostic["feature_validity"]["face.angry"] is True
    assert diagnostic["feature_validity"]["behavior.movement_energy"] is False
    assert diagnostic["unavailable_reasons"]["behavior.movement_energy"] == "Needs a prior frame."
    assert diagnostic["modality_status"] == {
        "face": "ok",
        "behavior": "warming_up",
        "audio": "ok",
    }
    assert diagnostic["feature_record"]["feature_values"]["audio.rms_loudness"] == 0.4
    assert diagnostic["available_modalities"] == {
        "face": True,
        "behavior": False,
        "audio": True,
    }


def test_constructed_empty_fusion_evidence_is_insufficient_not_never_constructed():
    evidence = build_fusion_evidence(
        {
            "latest_face_result": None,
            "latest_behavior_result": None,
            "latest_audio_result": None,
        }
    )
    record = evidence_to_feature_record(evidence)
    sufficient, reason = evidence_sufficient(record)
    diagnostic = build_fusion_diagnostic_snapshot(
        evidence,
        record,
        source_timestamps={"face": None, "behavior": None, "audio": None},
        sufficient=sufficient,
        sufficiency_reason=reason,
    )

    assert diagnostic["construction_status"] == "constructed_but_insufficient"
    assert diagnostic["feature_names"] == ()
    assert diagnostic["valid_feature_count"] == 0
    assert reason is not None
