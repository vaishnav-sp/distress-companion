from types import SimpleNamespace

import pytest

from config import EXPERIMENTAL_DISTRESS_ALERT_THRESHOLD
from distress_score import (
    DistressScoreResult,
    calculate_distress_score,
    score_alert_triggered,
)
from fusion.schema import FEATURE_SCHEMA_VERSION, FeatureRecord, feature_order
from rule_based_detection import RuleBasedCueDetector, RuleDetectionConfig


def _score_result(score, status="Available"):
    return DistressScoreResult(
        score=score,
        status=status,
        modalities={},
        included_modalities=(),
        excluded_reasons={},
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [(69.9, False), (70.0, True), (70.1, True)],
)
def test_score_alert_uses_inclusive_configured_threshold(score, expected):
    assert score_alert_triggered(_score_result(score), threshold=70.0) is expected


def test_configured_default_threshold_is_70_and_is_used_by_default():
    assert EXPERIMENTAL_DISTRESS_ALERT_THRESHOLD == 70.0
    assert score_alert_triggered(_score_result(70.0)) is True


def test_insufficient_evidence_never_triggers_or_becomes_a_low_score():
    assert score_alert_triggered(_score_result(80.0, "Insufficient evidence")) is None


def _audio_snapshot(*, timestamp=100.0, rms=0.2, valid=True):
    audio = SimpleNamespace(
        feature_values={"rms_loudness": rms, "audio_contribution": 0.8},
        feature_validity={"rms_loudness": valid, "audio_contribution": valid},
        rms_valid=valid,
        audio_contribution=0.8,
    )
    return {
        "status": "running",
        "latest_audio_result": audio,
        "last_audio_success_timestamp": timestamp,
    }


@pytest.mark.parametrize(
    "snapshot",
    [
        {"status": "running"},
        _audio_snapshot(timestamp=96.9),
        _audio_snapshot(rms=float("nan")),
    ],
)
def test_missing_stale_or_invalid_branch_evidence_cannot_trigger(snapshot):
    result = calculate_distress_score(snapshot, now=100.0)

    assert result.status == "Insufficient evidence"
    assert result.score is None
    assert score_alert_triggered(result) is None


@pytest.mark.parametrize(
    "score",
    [None, float("nan"), float("inf"), float("-inf"), -0.1, 100.1, "invalid"],
)
def test_missing_invalid_or_nonfinite_score_cannot_trigger(score):
    assert score_alert_triggered(_score_result(score)) is None


@pytest.mark.parametrize(
    "threshold",
    [None, True, float("nan"), float("inf"), float("-inf"), -0.1, 100.1, "invalid"],
)
def test_invalid_thresholds_are_rejected(threshold):
    with pytest.raises(ValueError, match=r"finite and within \[0, 100\]"):
        score_alert_triggered(_score_result(70.0), threshold=threshold)


@pytest.mark.parametrize(
    ("score", "threshold", "expected"),
    [(0.0, 0.0, True), (0.0, 100.0, False), (100.0, 100.0, True)],
)
def test_threshold_accepts_both_configured_range_endpoints(score, threshold, expected):
    assert score_alert_triggered(
        _score_result(score), threshold=threshold
    ) is expected


def _feature_record(rms, onset):
    values = {name: None for name in feature_order()}
    validity = {name: False for name in feature_order()}
    reasons = {name: "Synthetic test feature unavailable." for name in feature_order()}
    for name, value in (
        ("audio.rms_loudness", rms),
        ("audio.onset_score", onset),
    ):
        values[name] = value
        validity[name] = True
        reasons.pop(name)
    return FeatureRecord(
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=feature_order(),
        feature_values=values,
        feature_validity=validity,
        unavailable_reasons=reasons,
        modality_status={"face": "waiting", "behavior": "waiting", "audio": "ok"},
        modality_messages={"face": "", "behavior": "", "audio": ""},
    )


def _evaluate_audio_peak(detector, record, timestamp):
    return detector.evaluate(
        record,
        modality_source_timestamps={"audio": timestamp, "behavior": timestamp},
        now=timestamp,
    )


def test_existing_audio_peak_can_trigger_while_score_alert_is_unavailable():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="audio_only")
    )
    _evaluate_audio_peak(detector, _feature_record(rms=0.01, onset=0.0), 10.0)
    audio_peak_result = _evaluate_audio_peak(
        detector,
        _feature_record(rms=0.3, onset=0.8),
        11.0,
    )
    score_result = calculate_distress_score({"status": "running"}, now=11.0)

    assert audio_peak_result.status == "potential_distress"
    assert audio_peak_result.cue_indicators["audio_peak"] is True
    assert score_alert_triggered(score_result) is None
