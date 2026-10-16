from types import SimpleNamespace
import math

import pytest

from alert_management import AlertManager
from branches.audio_branch import AudioFeatureResult, FEATURE_NAMES as AUDIO_FEATURE_NAMES
from branches.behavior_branch import (
    FEATURE_NAMES as BEHAVIOR_FEATURE_NAMES,
    BehaviorResult,
)
from branches.face_branch import (
    FEATURE_VECTOR_ORDER,
    FaceResult,
    PredictionConfidence,
)
from config import EXPERIMENTAL_DISTRESS_SCORE_WEIGHTS
from distress_score import calculate_distress_score


def _face_result(
    *,
    negative_affect=0.15,
    quality=0.9,
    quality_status="usable",
    entropy=None,
    probabilities=None,
):
    remaining_class_probability = (1.0 - 3.0 * negative_affect) / 3.0
    expressions = {
        "angry": 3.0 * negative_affect,
        "fear": 0.0,
        "happy": remaining_class_probability,
        "neutral": remaining_class_probability,
        "sad": 0.0,
        "surprise": remaining_class_probability,
        "disgust": None,
    }
    if probabilities is not None:
        expressions.update(probabilities)
    class_availability = {name: name != "disgust" for name in expressions}
    class_validity = {
        name: value is not None and name != "disgust"
        for name, value in expressions.items()
    }
    negative_mean = sum(expressions[name] for name in ("angry", "fear", "sad")) / 3
    valid_probabilities = [
        float(value)
        for name, value in expressions.items()
        if name != "disgust" and value is not None and math.isfinite(float(value))
    ]
    sorted_probabilities = sorted(valid_probabilities, reverse=True)
    calculated_entropy = (
        -sum(value * math.log(max(value, 1e-12)) for value in valid_probabilities)
        / math.log(len(valid_probabilities))
    )
    normalized_entropy = calculated_entropy if entropy is None else entropy
    return FaceResult(
        features={},
        provisional_distress=None,
        input_quality=True,
        face_detected=True,
        expression_probabilities=expressions,
        class_availability=class_availability,
        class_validity=class_validity,
        negative_affect_feature=negative_mean,
        face_quality_score=quality,
        face_quality_status=quality_status,
        prediction_confidence=PredictionConfidence(
            maximum_probability=max(valid_probabilities, default=0.0),
            normalized_entropy=normalized_entropy,
            top_two_margin=(
                sorted_probabilities[0] - sorted_probabilities[1]
                if len(sorted_probabilities) > 1
                else 0.0
            ),
        ),
        feature_vector=tuple(expressions.get(name) for name in FEATURE_VECTOR_ORDER),
        feature_vector_order=FEATURE_VECTOR_ORDER,
        abstention_reason=None,
        status_message="Test face result.",
    )


def _behavior_result(
    *,
    values=None,
    validity=None,
    regions=None,
    region_validity=None,
):
    feature_values = {
        "movement_energy": 0.0,
        "sudden_movement": 0.0,
        "wrist_peak_speed": 0.0,
        "wrist_peak_acceleration": 0.0,
        "wrist_movement_amplitude": 0.0,
        "wrist_repetition_rate_hz": 0.0,
        "wrist_elevated_movement_proportion": 0.0,
    }
    feature_values.update(values or {})
    feature_validity = {name: True for name in feature_values}
    feature_validity.update(validity or {})
    return BehaviorResult(
        features={},
        provisional_distress=None,
        input_quality=True,
        status="ok",
        status_message="Test behavior result.",
        landmark_validity={},
        feature_values=feature_values,
        feature_validity=feature_validity,
        feature_vector=tuple(None for _ in BEHAVIOR_FEATURE_NAMES),
        feature_names=BEHAVIOR_FEATURE_NAMES,
        unavailable_reasons={},
        quality_score=1.0,
        baseline_ready=True,
        baseline_sample_count=5,
        baseline_energy=0.0,
        repetition_status="measured",
        pose_diagnostic={},
        debug_overlay=None,
        region_metrics=regions or {},
        region_feature_validity=region_validity or {},
        region_unavailable_reasons={},
    )


def _audio_result(*, rms=0.2, contribution=0.4, status="ok", valid=True):
    values = {
        "rms_loudness": rms,
        "spectral_centroid_hz": None,
        "spectral_bandwidth_hz": None,
        "spectral_flatness": None,
        "onset_score": None,
        "baseline_relative_rms": None,
        "audio_contribution": contribution,
    }
    validity = {name: False for name in AUDIO_FEATURE_NAMES}
    validity["rms_loudness"] = valid
    validity["audio_contribution"] = valid
    return AudioFeatureResult(
        features={},
        provisional_distress=None,
        input_quality=valid,
        status=status,
        status_message="Test audio result.",
        quality_flag=valid,
        quality_status="usable" if valid else "unknown",
        rms_loudness=rms,
        rms_valid=valid,
        spectral_features={
            "spectral_centroid_hz": None,
            "spectral_bandwidth_hz": None,
            "spectral_flatness": None,
        },
        spectral_validity={
            "spectral_centroid_hz": False,
            "spectral_bandwidth_hz": False,
            "spectral_flatness": False,
        },
        onset_score=None,
        onset_valid=False,
        onset_flag=False,
        baseline_ready=True,
        baseline_sample_count=5,
        baseline_relative_rms=None,
        feature_values=values,
        feature_validity=validity,
        feature_vector=tuple(values[name] for name in AUDIO_FEATURE_NAMES),
        feature_names=AUDIO_FEATURE_NAMES,
        unavailable_reasons={},
        audio_contribution=contribution,
    )


def _snapshot(*, face=None, behavior=None, audio=None, timestamps=None, status="running"):
    timestamps = timestamps or {}
    return {
        "status": status,
        "latest_face_result": face,
        "latest_behavior_result": behavior,
        "latest_audio_result": audio,
        "last_face_success_timestamp": timestamps.get("face", 100.0 if face else None),
        "last_behavior_success_timestamp": timestamps.get(
            "behavior", 100.0 if behavior else None
        ),
        "last_audio_success_timestamp": timestamps.get("audio", 100.0 if audio else None),
    }


def _repetitive_behavior_result(
    *,
    repetition=0.5,
    amplitude=0.06,
    elevated=0.5,
    overrides=None,
):
    values = {
        "movement_energy": 0.0,
        "sudden_movement": 0.0,
        "wrist_peak_speed": 0.0,
        "wrist_peak_acceleration": 0.0,
        "wrist_repetition_rate_hz": repetition,
        "wrist_movement_amplitude": amplitude,
        "wrist_elevated_movement_proportion": elevated,
    }
    values.update(overrides or {})
    return _behavior_result(values=values)


def test_all_three_valid_modalities_are_weighted_by_configured_weights():
    result = calculate_distress_score(
        _snapshot(
            face=_face_result(negative_affect=0.15),
            behavior=_repetitive_behavior_result(
                repetition=0.25,
                amplitude=0.03,
                elevated=0.25,
            ),
            audio=_audio_result(contribution=0.4),
        ),
        now=100.0,
    )

    assert result.status == "Available"
    assert result.score == pytest.approx(39.5)
    assert result.included_modalities == ("emotion", "behavior", "audio")
    assert result.modalities["emotion"].score == pytest.approx(15.0)
    assert result.modalities["behavior"].score == pytest.approx(50.0)
    assert result.modalities["audio"].score == pytest.approx(40.0)
    assert result.modalities["emotion"].contribution == pytest.approx(3.0)
    assert result.modalities["behavior"].contribution == pytest.approx(22.5)
    assert result.modalities["audio"].contribution == pytest.approx(14.0)
    assert result.modalities["emotion"].weight == 20.0
    assert result.modalities["behavior"].weight == 45.0
    assert result.modalities["audio"].weight == 35.0


def test_two_valid_modalities_renormalize_weights_and_exclude_missing_branch():
    result = calculate_distress_score(
        _snapshot(
            face=_face_result(negative_affect=0.15),
            audio=_audio_result(contribution=0.2),
        ),
        now=100.0,
    )

    assert result.score == pytest.approx((20 * 15 + 35 * 20) / 55)
    assert result.included_modalities == ("emotion", "audio")
    assert result.modalities["emotion"].contribution == pytest.approx(20 / 55 * 15)
    assert result.modalities["audio"].contribution == pytest.approx(35 / 55 * 20)
    assert result.excluded_reasons["behavior"] == "No branch result is available."


def test_single_valid_modality_withholds_overall_score_and_contribution():
    result = calculate_distress_score(
        _snapshot(audio=_audio_result(contribution=0.7)),
        now=100.0,
    )

    assert result.score is None
    assert result.status == "Insufficient evidence"
    assert result.modalities["audio"].score == pytest.approx(70.0)
    assert result.modalities["audio"].available is True
    assert result.modalities["audio"].contributed is False
    assert result.modalities["audio"].contribution is None
    assert "minimum number" in result.excluded_reasons["audio"]


def test_no_usable_evidence_returns_insufficient_evidence():
    result = calculate_distress_score(_snapshot(), now=100.0)

    assert result.score is None
    assert result.status == "Insufficient evidence"
    assert not result.included_modalities
    assert all(not item.available for item in result.modalities.values())


def test_stale_branch_evidence_is_excluded_using_monotonic_timestamps():
    result = calculate_distress_score(
        _snapshot(
            face=_face_result(),
            audio=_audio_result(),
            timestamps={"face": 96.9, "audio": 100.0},
        ),
        now=100.0,
    )

    assert result.score is None
    assert not result.modalities["emotion"].available
    assert "stale" in result.excluded_reasons["emotion"]
    assert result.modalities["audio"].available


@pytest.mark.parametrize(
    ("quality", "quality_status", "expected"),
    [
        (None, "usable", "Face quality score"),
        (0.9, "poor", "Face quality is not marked usable"),
        (1.1, "usable", "outside [0, 1]"),
    ],
)
def test_missing_or_invalid_face_quality_excludes_emotion(
    quality, quality_status, expected
):
    result = calculate_distress_score(
        _snapshot(
            face=_face_result(quality=quality, quality_status=quality_status),
        ),
        now=100.0,
    )

    assert result.modalities["emotion"].score is None
    assert expected in result.excluded_reasons["emotion"]


def test_invalid_emotion_value_and_high_prediction_uncertainty_are_excluded():
    bad_probability = _face_result(
        probabilities={"angry": float("nan")}
    )
    malformed_distribution = _face_result(
        probabilities={"angry": 0.5, "fear": 0.5, "sad": 0.5}
    )
    uncertain = _face_result(
        negative_affect=1.0 / 6.0,
        entropy=1.0,
        probabilities={
            "angry": 1.0 / 6.0,
            "fear": 1.0 / 6.0,
            "sad": 1.0 / 6.0,
            "happy": 1.0 / 6.0,
            "neutral": 1.0 / 6.0,
            "surprise": 1.0 / 6.0,
        },
    )

    invalid_result = calculate_distress_score(
        _snapshot(face=bad_probability), now=100.0
    )
    malformed_result = calculate_distress_score(
        _snapshot(face=malformed_distribution), now=100.0
    )
    uncertain_result = calculate_distress_score(
        _snapshot(face=uncertain), now=100.0
    )

    assert "angry expression estimate" in invalid_result.excluded_reasons["emotion"]
    assert "do not sum to 1" in malformed_result.excluded_reasons["emotion"]
    assert "uncertainty" in uncertain_result.excluded_reasons["emotion"]


def test_behavior_rejects_invalid_wrist_values_but_uses_valid_region_fallback():
    behavior = _behavior_result(
        values={
            "wrist_repetition_rate_hz": float("nan"),
            "wrist_movement_amplitude": 0.2,
            "wrist_elevated_movement_proportion": 0.8,
        },
        validity={
            "wrist_repetition_rate_hz": True,
            "wrist_movement_amplitude": True,
            "wrist_elevated_movement_proportion": True,
        },
        regions={
            "torso": {
                "repetition_rate_hz": 0.5,
                "amplitude": 0.06,
                "elevated_movement_proportion": 0.5,
                "movement_energy": 0.0,
                "peak_speed": 0.0,
                "peak_acceleration": 0.0,
            }
        },
        region_validity={
            "torso": {
                "repetition_rate_hz": True,
                "amplitude": True,
                "elevated_movement_proportion": True,
                "movement_energy": True,
                "peak_speed": True,
                "peak_acceleration": True,
            }
        },
    )
    result = calculate_distress_score(
        _snapshot(behavior=behavior),
        now=100.0,
    )

    assert result.modalities["behavior"].available
    assert result.modalities["behavior"].score == pytest.approx(100.0)
    assert "torso repetitive movement" in result.modalities["behavior"].evidence[0]


def test_repetition_score_uses_weakest_required_valid_ratio():
    result = calculate_distress_score(
        _snapshot(
            behavior=_repetitive_behavior_result(
                repetition=0.4,
                amplitude=0.03,
                elevated=0.4,
            )
        ),
        now=100.0,
    )

    assert result.modalities["behavior"].score == pytest.approx(50.0)
    assert "weakest required ratio 0.500" in result.modalities["behavior"].evidence[0]


def test_valid_audio_requires_valid_rms_and_maps_contribution():
    result = calculate_distress_score(
        _snapshot(audio=_audio_result(rms=0.0, contribution=0.0, status="silent")),
        now=100.0,
    )

    assert result.modalities["audio"].available
    assert result.modalities["audio"].score == 0.0
    assert any("silent-window RMS" in text for text in result.modalities["audio"].evidence)


def test_missing_audio_is_not_treated_as_silent_zero_contribution():
    result = calculate_distress_score(
        _snapshot(audio=_audio_result(rms=None, contribution=0.0, valid=False)),
        now=100.0,
    )

    assert result.modalities["audio"].score is None
    assert "invalid" in result.excluded_reasons["audio"]


@pytest.mark.parametrize(
    "audio",
    [
        _audio_result(rms=-0.1, contribution=0.4),
        _audio_result(rms=0.2, contribution=1.1),
        _audio_result(rms=float("inf"), contribution=0.4),
        _audio_result(rms=0.2, contribution=float("nan")),
    ],
)
def test_invalid_audio_ranges_nonfinite_and_malformed_fields_are_excluded(audio):
    result = calculate_distress_score(_snapshot(audio=audio), now=100.0)

    assert result.modalities["audio"].score is None


def test_weights_are_configurable_and_normalized_over_contributors():
    result = calculate_distress_score(
        _snapshot(
            face=_face_result(negative_affect=0.15),
            behavior=_repetitive_behavior_result(
                repetition=0.5,
                amplitude=0.06,
                elevated=0.5,
            ),
        ),
        now=100.0,
        weights={"emotion": 1.0, "behavior": 3.0, "audio": 0.0},
    )

    assert result.score == pytest.approx(78.75)
    assert result.modalities["emotion"].contribution == pytest.approx(3.75)
    assert result.modalities["behavior"].contribution == pytest.approx(75.0)
    assert result.modalities["audio"].weight == 0.0


def test_scores_are_bounded_and_deterministic():
    snapshot = _snapshot(
        face=_face_result(negative_affect=1.0 / 3.0),
        behavior=_repetitive_behavior_result(
            repetition=100.0,
            amplitude=100.0,
            elevated=1.0,
        ),
        audio=_audio_result(contribution=1.0),
    )

    first = calculate_distress_score(snapshot, now=100.0)
    second = calculate_distress_score(snapshot, now=100.0)

    assert first == second
    assert 0.0 <= first.score <= 100.0
    assert all(
        item.score is None or 0.0 <= item.score <= 100.0
        for item in first.modalities.values()
    )
    assert "not a probability" in first.interpretation_warning


def test_score_does_not_access_or_mutate_alert_manager_state():
    manager = AlertManager()
    before = manager.diagnostic_state()
    snapshot = _snapshot(
        face=_face_result(),
        audio=_audio_result(),
    )
    snapshot["latest_alert"] = manager.snapshot()

    calculate_distress_score(snapshot, now=100.0)

    assert manager.diagnostic_state() == before


def test_stopped_or_malformed_monitoring_state_does_not_score():
    stopped = calculate_distress_score(
        _snapshot(face=_face_result(), audio=_audio_result(), status="stopped"),
        now=100.0,
    )
    malformed = calculate_distress_score(None, now=100.0)

    assert stopped.status == "Insufficient evidence"
    assert all("not running" in reason for reason in stopped.excluded_reasons.values())
    assert malformed.status == "Insufficient evidence"
    assert all("malformed" in reason for reason in malformed.excluded_reasons.values())


def test_invalid_timestamp_and_invalid_weight_inputs_are_rejected_safely():
    bad_timestamp = calculate_distress_score(
        _snapshot(audio=_audio_result(), timestamps={"audio": float("nan")}),
        now=100.0,
    )

    assert bad_timestamp.modalities["audio"].score is None
    assert "timestamp" in bad_timestamp.excluded_reasons["audio"]
    with pytest.raises(ValueError, match="finite and nonnegative"):
        calculate_distress_score(
            _snapshot(audio=_audio_result()),
            now=100.0,
            weights={"emotion": 30, "behavior": float("inf"), "audio": 30},
        )


def test_audio_feature_map_malformed_does_not_crash_or_score():
    malformed = SimpleNamespace(
        feature_values="not-a-map",
        feature_validity={},
        rms_valid=True,
        audio_contribution=0.0,
    )
    result = calculate_distress_score(_snapshot(audio=malformed), now=100.0)

    assert result.modalities["audio"].score is None
    assert "unavailable" in result.excluded_reasons["audio"]


def test_configured_demo_weights_are_the_approved_provisional_values():
    assert EXPERIMENTAL_DISTRESS_SCORE_WEIGHTS == {
        "emotion": 20.0,
        "behavior": 45.0,
        "audio": 35.0,
    }
