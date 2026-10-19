from types import SimpleNamespace
from datetime import datetime, timedelta
import time

import pytest

from alert_management import AlertManager, AlertPolicy
from fusion.schema import FEATURE_SCHEMA_VERSION, FeatureRecord, feature_order
from monitoring_history import create_alert_diagnostic_entry
from rule_based_detection import (
    RuleBasedCueDetector,
    RuleBasedInferenceProcessor,
    RuleDetectionConfig,
)


def _record(
    *,
    rms=0.01,
    onset=0.0,
    energy=0.1,
    sudden=0.0,
    speed=0.1,
    acceleration=0.1,
    amplitude=0.01,
    repetition=0.0,
    elevated=0.0,
    missing=(),
):
    provided = {
        "audio.rms_loudness": rms,
        "audio.onset_score": onset,
        "behavior.movement_energy": energy,
        "behavior.sudden_movement": sudden,
        "behavior.wrist_peak_speed": speed,
        "behavior.wrist_peak_acceleration": acceleration,
        "behavior.wrist_movement_amplitude": amplitude,
        "behavior.wrist_repetition_rate_hz": repetition,
        "behavior.wrist_elevated_movement_proportion": elevated,
    }
    values = {name: None for name in feature_order()}
    validity = {name: False for name in feature_order()}
    reasons = {name: "Synthetic test feature unavailable." for name in feature_order()}
    for name, value in provided.items():
        if name not in missing and value is not None:
            values[name] = float(value)
            validity[name] = True
            reasons.pop(name, None)
    return FeatureRecord(
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=feature_order(),
        feature_values=values,
        feature_validity=validity,
        unavailable_reasons=reasons,
        modality_status={"face": "waiting", "behavior": "ok", "audio": "ok"},
        modality_messages={"face": "", "behavior": "", "audio": ""},
    )


def _evaluate(
    detector,
    record,
    timestamp,
    *,
    audio_timestamp=None,
    behavior_timestamp=None,
    region_movement=None,
):
    return detector.evaluate(
        record,
        modality_source_timestamps={
            "audio": timestamp if audio_timestamp is None else audio_timestamp,
            "behavior": (
                timestamp if behavior_timestamp is None else behavior_timestamp
            ),
        },
        now=timestamp,
        region_movement=region_movement,
    )


def test_sustained_high_audio_rms_activates_only_after_configured_duration():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="audio_only", audio_sustained_seconds=2)
    )
    assert _evaluate(detector, _record(rms=0.12), 10).rule_triggered is False
    assert _evaluate(detector, _record(rms=0.12), 11).rule_triggered is False
    result = _evaluate(detector, _record(rms=0.12), 12)

    assert result.status == "potential_distress"
    assert result.cue_indicators["audio_sustained"] is True
    assert result.cue_indicators["movement_vigorous"] is False


def test_brief_audio_peak_follows_configured_onset_rule():
    detector = RuleBasedCueDetector(RuleDetectionConfig(combination_mode="audio_only"))
    _evaluate(detector, _record(rms=0.01, onset=0.0), 10)
    result = _evaluate(detector, _record(rms=0.3, onset=0.8), 11)

    assert result.status == "potential_distress"
    assert result.cue_indicators["audio_peak"] is True
    assert result.potential_distress_probability is None


def test_sudden_movement_is_an_explicit_single_sample_cue():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="movement_only")
    )
    result = _evaluate(detector, _record(sudden=1.0), 10)

    assert result.status == "potential_distress"
    assert result.cue_indicators["movement_sudden"] is True


def test_repetitive_movement_requires_rate_amplitude_and_persistence():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(
            combination_mode="movement_only",
            movement_repetition_seconds=2,
        )
    )
    values = dict(repetition=0.8, amplitude=0.2, elevated=0.8)
    assert _evaluate(detector, _record(**values), 10).rule_triggered is False
    assert _evaluate(detector, _record(**values), 11).rule_triggered is False
    result = _evaluate(detector, _record(**values), 12)

    assert result.status == "potential_distress"
    assert result.cue_indicators["movement_repetitive"] is True


def test_vigorous_movement_requires_sustained_energy_and_speed():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="movement_only")
    )
    vigorous = dict(energy=1.1, speed=1.2, acceleration=0.1)
    assert _evaluate(detector, _record(**vigorous), 10).rule_triggered is False
    result = _evaluate(detector, _record(**vigorous), 11)

    assert result.status == "potential_distress"
    assert result.cue_indicators["movement_vigorous"] is True


def test_ordinary_movement_alone_does_not_trigger_a_cue():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="movement_only")
    )
    result = _evaluate(
        detector,
        _record(energy=0.3, speed=0.4, repetition=0.2, amplitude=0.03),
        10,
    )

    assert result.status == "no_sufficient_distress_evidence"
    assert result.rule_triggered is False


def test_audio_and_movement_combination_requires_both_configured_cues():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="audio_and_movement")
    )
    neither = _evaluate(detector, _record(), 10)
    audio_only = _evaluate(detector, _record(rms=0.3, onset=0.8), 11)
    combined = _evaluate(
        detector,
        _record(rms=0.3, onset=0.8, sudden=1.0),
        12,
    )

    assert neither.rule_triggered is False
    assert audio_only.status == "no_sufficient_distress_evidence"
    assert combined.status == "potential_distress"
    assert combined.cue_indicators["audio_peak"] is True
    assert combined.cue_indicators["movement_sudden"] is True


def test_any_mode_uses_valid_audio_when_movement_features_are_missing():
    detector = RuleBasedCueDetector(RuleDetectionConfig(combination_mode="any"))
    record = _record(
        rms=0.01,
        onset=0.0,
        missing=(
            "behavior.movement_energy",
            "behavior.sudden_movement",
            "behavior.wrist_peak_speed",
            "behavior.wrist_peak_acceleration",
            "behavior.wrist_movement_amplitude",
            "behavior.wrist_repetition_rate_hz",
            "behavior.wrist_elevated_movement_proportion",
        ),
    )

    result = _evaluate(detector, record, 10)

    assert result.status == "no_sufficient_distress_evidence"
    assert result.rule_triggered is False
    assert result.rule_evidence_sufficient is True
    assert result.cue_indicators["movement_repetitive"] is None


def test_any_mode_uses_valid_movement_when_audio_features_are_missing():
    detector = RuleBasedCueDetector(RuleDetectionConfig(combination_mode="any"))
    record = _record(
        sudden=0.0,
        missing=("audio.rms_loudness", "audio.onset_score"),
    )

    result = _evaluate(detector, record, 10)

    assert result.status == "no_sufficient_distress_evidence"
    assert result.rule_triggered is False
    assert result.rule_evidence_sufficient is True
    assert result.cue_indicators["audio_sustained"] is None


def test_repetitive_region_movement_triggers_without_audio_or_wrist_features():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(
            combination_mode="movement_only",
            movement_regions=("torso",),
            movement_repetition_seconds=2.0,
        )
    )
    region_movement = {
        "torso": {
            "metrics": {
                "repetition_rate_hz": 0.8,
                "amplitude": 0.2,
                "elevated_movement_proportion": 0.8,
            },
            "validity": {
                "repetition_rate_hz": True,
                "amplitude": True,
                "elevated_movement_proportion": True,
            },
            "reasons": {},
        }
    }
    record = _record(
        missing=(
            "audio.rms_loudness",
            "audio.onset_score",
            "behavior.movement_energy",
            "behavior.sudden_movement",
            "behavior.wrist_peak_speed",
            "behavior.wrist_peak_acceleration",
            "behavior.wrist_movement_amplitude",
            "behavior.wrist_repetition_rate_hz",
            "behavior.wrist_elevated_movement_proportion",
        )
    )
    results = [
        _evaluate(
            detector,
            record,
            timestamp,
            region_movement=region_movement,
        )
        for timestamp in (10.0, 11.0, 12.0)
    ]

    assert results[-1].status == "potential_distress"
    assert results[-1].cue_indicators["movement_repetitive_torso"] is True
    assert results[-1].cue_indicators["movement_repetitive"] is True
    assert results[-1].rule_evidence_sufficient is True
    assert results[-1].available_modalities["behavior"] is True


def test_region_repetition_duplicate_timestamp_does_not_advance_persistence():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(
            combination_mode="movement_only",
            movement_regions=("torso",),
            movement_repetition_seconds=2.0,
        )
    )
    region_movement = {
        "torso": {
            "metrics": {
                "repetition_rate_hz": 0.8,
                "amplitude": 0.2,
                "elevated_movement_proportion": 0.8,
            },
            "validity": {
                "repetition_rate_hz": True,
                "amplitude": True,
                "elevated_movement_proportion": True,
            },
            "reasons": {},
        }
    }
    record = _record()
    first = _evaluate(detector, record, 10.0, region_movement=region_movement)
    second = _evaluate(detector, record, 11.0, region_movement=region_movement)
    duplicate = _evaluate(detector, record, 11.0, region_movement=region_movement)
    third = _evaluate(detector, record, 12.0, region_movement=region_movement)

    assert first.cue_indicators["movement_repetitive_torso"] is False
    assert second.cue_indicators["movement_repetitive_torso"] is False
    assert duplicate.cue_indicators["movement_repetitive_torso"] is False
    assert third.cue_indicators["movement_repetitive_torso"] is True


def test_audio_and_movement_mode_still_requires_both_with_region_cues():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(
            combination_mode="audio_and_movement",
            movement_regions=("torso",),
            movement_repetition_seconds=0.0,
        )
    )
    region_movement = {
        "torso": {
            "metrics": {
                "repetition_rate_hz": 0.8,
                "amplitude": 0.2,
                "elevated_movement_proportion": 0.8,
            },
            "validity": {
                "repetition_rate_hz": True,
                "amplitude": True,
                "elevated_movement_proportion": True,
            },
            "reasons": {},
        }
    }
    movement_without_audio = _evaluate(
        detector,
        _record(missing=("audio.rms_loudness", "audio.onset_score")),
        10.0,
        region_movement=region_movement,
    )
    both = _evaluate(
        detector,
        _record(rms=0.3, onset=0.8),
        11.0,
        region_movement=region_movement,
    )

    assert movement_without_audio.status == "unknown"
    assert movement_without_audio.rule_triggered is None
    assert both.status == "potential_distress"
    assert both.cue_indicators["movement_repetitive_torso"] is True
    assert both.cue_indicators["audio_peak"] is True


def test_any_mode_uses_audio_when_region_movement_is_unavailable():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="any", movement_regions=("torso",))
    )
    record = _record(
        rms=0.3,
        onset=0.8,
        missing=(
            "behavior.movement_energy",
            "behavior.sudden_movement",
            "behavior.wrist_peak_speed",
            "behavior.wrist_peak_acceleration",
            "behavior.wrist_movement_amplitude",
            "behavior.wrist_repetition_rate_hz",
            "behavior.wrist_elevated_movement_proportion",
        ),
    )
    unavailable_region = {
        "torso": {
            "metrics": {},
            "validity": {},
            "reasons": {"repetition_rate_hz": "Torso is not visible."},
        }
    }

    result = _evaluate(
        detector,
        record,
        10,
        region_movement=unavailable_region,
    )

    assert result.status == "potential_distress"
    assert result.cue_indicators["audio_peak"] is True
    assert result.cue_indicators["movement_repetitive_torso"] is None
    assert "Torso is not visible" in result.metadata["cue_unavailable_reasons"][
        "movement_repetitive_torso"
    ]


def test_inference_processor_reads_region_measurements_from_behavior_result():
    timestamp = time.monotonic()
    behavior = SimpleNamespace(
        status="partial",
        status_message="Wrist tracking unavailable; torso tracking available.",
        feature_values={"shoulder_alignment": 0.0},
        feature_validity={"shoulder_alignment": True},
        feature_names=("shoulder_alignment",),
        unavailable_reasons={},
        region_metrics={
            "torso": {
                "repetition_rate_hz": 0.8,
                "amplitude": 0.2,
                "elevated_movement_proportion": 0.8,
            }
        },
        region_feature_validity={
            "torso": {
                "repetition_rate_hz": True,
                "amplitude": True,
                "elevated_movement_proportion": True,
            }
        },
        region_unavailable_reasons={"torso": {}},
    )
    processor = RuleBasedInferenceProcessor(
        RuleDetectionConfig(
            combination_mode="movement_only",
            movement_regions=("torso",),
            movement_repetition_seconds=0.0,
        )
    )

    result = processor(
        {
            "status": "running",
            "latest_face_result": None,
            "latest_behavior_result": behavior,
            "latest_audio_result": None,
            "last_face_success_timestamp": None,
            "last_behavior_success_timestamp": timestamp,
            "last_audio_success_timestamp": None,
        }
    )

    assert result["inference"].cue_indicators["movement_repetitive_torso"] is True
    assert result["inference"].status == "potential_distress"
    assert result["feature_record"].schema_version == FEATURE_SCHEMA_VERSION


def test_any_mode_does_not_let_missing_repetition_hide_inactive_movement_cues():
    detector = RuleBasedCueDetector(RuleDetectionConfig(combination_mode="any"))
    record = _record(
        missing=(
            "behavior.wrist_repetition_rate_hz",
            "behavior.wrist_elevated_movement_proportion",
        )
    )

    result = _evaluate(detector, record, 10)

    assert result.status == "no_sufficient_distress_evidence"
    assert result.rule_triggered is False
    assert result.rule_evidence_sufficient is True
    assert result.modality_cues["behavior"] is False
    assert result.modality_evidence_status["behavior"] == "partially_available"
    assert result.cue_indicators["movement_repetitive"] is None
    assert "wrist_repetition_rate_hz" in result.metadata["cue_unavailable_reasons"][
        "movement_repetitive"
    ]


def test_any_mode_returns_inactive_when_both_modalities_are_usable_and_quiet():
    detector = RuleBasedCueDetector(RuleDetectionConfig(combination_mode="any"))
    result = _evaluate(detector, _record(), 10)

    assert result.status == "no_sufficient_distress_evidence"
    assert result.rule_triggered is False
    assert result.rule_evidence_sufficient is True


def test_any_mode_returns_unknown_when_both_modalities_are_unavailable():
    detector = RuleBasedCueDetector(RuleDetectionConfig(combination_mode="any"))
    record = _record(
        missing=(
            "audio.rms_loudness",
            "audio.onset_score",
            "behavior.movement_energy",
            "behavior.sudden_movement",
            "behavior.wrist_peak_speed",
            "behavior.wrist_peak_acceleration",
            "behavior.wrist_movement_amplitude",
            "behavior.wrist_repetition_rate_hz",
            "behavior.wrist_elevated_movement_proportion",
        )
    )
    result = _evaluate(detector, record, 10)

    assert result.status == "unknown"
    assert result.rule_triggered is None
    assert result.rule_evidence_sufficient is False
    assert result.modality_evidence_status["audio"] == "unavailable"
    assert result.modality_evidence_status["behavior"] == "unavailable"
    assert "audio" in result.reason.lower()
    assert "movement" in result.reason.lower()


def test_any_mode_uses_fresh_movement_when_audio_evidence_is_stale():
    detector = RuleBasedCueDetector(RuleDetectionConfig(combination_mode="any"))
    result = _evaluate(
        detector,
        _record(rms=0.3, onset=0.8, sudden=0.0),
        20,
        audio_timestamp=10,
    )

    assert result.status == "no_sufficient_distress_evidence"
    assert result.rule_triggered is False
    assert result.rule_evidence_sufficient is True
    assert result.modality_evidence_status["audio"] == "stale"
    assert result.modality_evidence_status["behavior"] == "evaluated_inactive"
    assert "maximum age" in result.modality_evidence_reasons["audio"]


def test_audio_and_movement_mode_is_unknown_without_both_evaluable_modalities():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="audio_and_movement")
    )
    record = _record(
        rms=0.01,
        onset=0.0,
        missing=(
            "behavior.movement_energy",
            "behavior.sudden_movement",
            "behavior.wrist_peak_speed",
            "behavior.wrist_peak_acceleration",
            "behavior.wrist_movement_amplitude",
            "behavior.wrist_repetition_rate_hz",
            "behavior.wrist_elevated_movement_proportion",
        ),
    )

    result = _evaluate(detector, record, 10)

    assert result.status == "unknown"
    assert result.rule_triggered is None
    assert result.rule_evidence_sufficient is False
    assert "behavior" in result.reason.lower()


def test_missing_wrist_landmarks_do_not_become_zero_or_trigger_movement():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="movement_only")
    )
    record = _record(
        missing=(
            "behavior.movement_energy",
            "behavior.sudden_movement",
            "behavior.wrist_peak_speed",
            "behavior.wrist_peak_acceleration",
            "behavior.wrist_movement_amplitude",
            "behavior.wrist_repetition_rate_hz",
            "behavior.wrist_elevated_movement_proportion",
        )
    )
    result = _evaluate(detector, record, 10)

    assert result.status == "unknown"
    assert result.rule_triggered is None
    assert result.cue_indicators["movement_vigorous"] is None
    assert result.cue_indicators["movement_repetitive"] is None


def test_mismatched_behavior_feature_name_is_reported_as_unpublished():
    from fusion import build_fusion_evidence, evidence_to_feature_record

    behavior = SimpleNamespace(
        status="partial",
        status_message="Synthetic branch result with an intentionally wrong feature name.",
        feature_values={"sudden_motion": 1.0},
        feature_validity={"sudden_motion": True},
        feature_names=("sudden_motion",),
        unavailable_reasons={},
    )
    record = evidence_to_feature_record(
        build_fusion_evidence(
            {
                "latest_face_result": None,
                "latest_behavior_result": behavior,
                "latest_audio_result": None,
            }
        )
    )
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="movement_only")
    )
    result = detector.evaluate(
        record,
        modality_source_timestamps={"behavior": 10.0, "audio": None},
        now=10.0,
    )

    assert result.status == "unknown"
    assert result.cue_indicators["movement_sudden"] is None
    assert "Unrecognized behavior feature name(s)" in result.cue_reason
    assert "behavior.sudden_movement" in result.cue_reason


def test_invalid_behavior_timestamps_are_treated_as_unfresh_evidence():
    processor = RuleBasedInferenceProcessor(
        RuleDetectionConfig(combination_mode="movement_only")
    )
    behavior = SimpleNamespace(
        status="ok",
        status_message="Movement data is present.",
        feature_values={
            "movement_energy": 1.1,
            "sudden_movement": 1.0,
            "wrist_peak_speed": 1.2,
            "wrist_peak_acceleration": 2.2,
            "wrist_movement_amplitude": 0.2,
            "wrist_repetition_rate_hz": 0.8,
            "wrist_elevated_movement_proportion": 0.8,
        },
        feature_validity={
            "movement_energy": True,
            "sudden_movement": True,
            "wrist_peak_speed": True,
            "wrist_peak_acceleration": True,
            "wrist_movement_amplitude": True,
            "wrist_repetition_rate_hz": True,
            "wrist_elevated_movement_proportion": True,
        },
        feature_names=(
            "movement_energy",
            "sudden_movement",
            "wrist_peak_speed",
            "wrist_peak_acceleration",
            "wrist_movement_amplitude",
            "wrist_repetition_rate_hz",
            "wrist_elevated_movement_proportion",
        ),
        unavailable_reasons={},
    )
    result = processor(
        {
            "latest_face_result": None,
            "latest_behavior_result": behavior,
            "latest_audio_result": None,
            "last_face_success_timestamp": None,
            "last_behavior_success_timestamp": "not-a-number",
            "last_audio_success_timestamp": None,
        }
    )

    assert result["inference"].status == "unknown"
    assert result["inference"].rule_triggered is None
    assert result["inference"].modality_evidence_status["behavior"] in {
        "missing_timestamp",
        "unavailable",
    }


def test_stale_audio_is_unavailable_not_silence():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="audio_only")
    )
    result = detector.evaluate(
        _record(rms=0.0),
        modality_source_timestamps={"audio": 10.0, "behavior": None},
        now=20.0,
    )

    assert result.status == "unknown"
    assert result.rule_triggered is None
    assert result.cue_indicators["audio_sustained"] is None


def test_stale_or_insufficient_rule_evidence_cannot_activate_alert():
    audio_only = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="audio_only")
    )
    stale = audio_only.evaluate(
        _record(rms=0.3, onset=0.8),
        modality_source_timestamps={"audio": 10.0, "behavior": None},
        now=20.0,
    )
    manager = AlertManager()
    stale_alert = manager.update(
        stale,
        result_timestamp=10,
        now=20,
        result_key="stale-rule",
    )

    missing = audio_only.evaluate(
        _record(missing=("audio.rms_loudness", "audio.onset_score")),
        modality_source_timestamps={"audio": 20.0, "behavior": None},
        now=20.0,
    )
    insufficient_alert = AlertManager().update(
        missing,
        result_timestamp=20,
        now=20,
        result_key="missing-rule",
    )

    assert stale_alert.state == "unknown"
    assert stale_alert.alert_id is None
    assert insufficient_alert.state == "unknown"
    assert insufficient_alert.alert_id is None


def test_irregular_timestamp_gap_resets_persistence_instead_of_inflating_duration():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(
            combination_mode="audio_only",
            audio_sustained_seconds=2,
            max_sample_gap_seconds=1.5,
        )
    )
    _evaluate(detector, _record(rms=0.12), 10)
    _evaluate(detector, _record(rms=0.12), 11)
    after_gap = _evaluate(detector, _record(rms=0.12), 14)

    assert after_gap.cue_indicators["audio_sustained"] is False
    assert after_gap.rule_triggered is False


def test_audio_only_rule_alert_uses_explicit_gate_and_releases_after_no_cues():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="audio_only")
    )
    manager = AlertManager(AlertPolicy(release_count=2, cooldown_seconds=5))
    loud = _evaluate(detector, _record(rms=0.3, onset=0.8), 10)
    before = manager.diagnostic_state()
    active = manager.update(
        loud,
        result_timestamp=10,
        now=10,
        result_key="loud",
    )
    assert active.state == "potential_distress"
    assert active.smoothed_score is None
    after = manager.diagnostic_state()
    diagnostic = create_alert_diagnostic_entry(
        loud,
        result_key="loud",
        inference_timestamp_utc="2026-01-01T00:00:00+00:00",
        source_timestamp=10,
        modality_source_timestamps={"face": None, "behavior": 10, "audio": 10},
        modality_results_present={"face": False, "behavior": True, "audio": True},
        observed_at_monotonic=10,
        evidence_sufficient=True,
        evidence_sufficiency_reason=None,
        before=before,
        after=after,
        alert_state_after=active.state,
        alert_reason_after=active.reason,
    )
    assert diagnostic["decision_source"] == "rule_based"
    assert diagnostic["potential_distress_probability"] is None
    assert diagnostic["raw_probability_threshold"] is None
    assert diagnostic["activation_qualified"] is True

    _evaluate(detector, _record(rms=0.01, onset=0.0), 11)
    quiet_one = _evaluate(detector, _record(rms=0.01, onset=0.0), 12)
    first_release = manager.update(
        quiet_one,
        result_timestamp=12,
        now=12,
        result_key="quiet-1",
    )
    quiet_two = _evaluate(detector, _record(rms=0.01, onset=0.0), 13)
    released = manager.update(
        quiet_two,
        result_timestamp=13,
        now=13,
        result_key="quiet-2",
    )

    assert first_release.state == "potential_distress"
    assert released.state == "cooldown"


def test_duplicate_rule_result_cannot_advance_release_state():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="audio_only")
    )
    manager = AlertManager(AlertPolicy(release_count=2))
    active_result = _evaluate(detector, _record(rms=0.3, onset=0.8), 10)
    active = manager.update(
        active_result, result_timestamp=10, now=10, result_key="same"
    )
    quiet_result = _evaluate(detector, _record(rms=0.01, onset=0.0), 11)
    duplicate = manager.update(
        quiet_result, result_timestamp=11, now=11, result_key="same"
    )

    assert active.state == "potential_distress"
    assert duplicate.state == "potential_distress"
    assert manager.diagnostic_state()["release_counter"] == 0


def test_monitoring_processor_builds_rule_result_without_classifier_probability():
    processor = RuleBasedInferenceProcessor(
        RuleDetectionConfig(combination_mode="audio_only")
    )
    timestamp = time.monotonic()
    audio = SimpleNamespace(
        status="ok",
        feature_names=("rms_loudness", "onset_score"),
        feature_values={"rms_loudness": 0.3, "onset_score": 0.8},
        feature_validity={"rms_loudness": True, "onset_score": True},
        unavailable_reasons={},
    )
    result = processor(
        {
            "latest_face_result": None,
            "latest_behavior_result": None,
            "latest_audio_result": audio,
            "last_face_success_timestamp": None,
            "last_behavior_success_timestamp": None,
            "last_audio_success_timestamp": timestamp,
        }
    )

    inference = result["inference"]
    assert inference.decision_source == "rule_based"
    assert inference.status == "potential_distress"
    assert inference.potential_distress_probability is None
    assert result["feature_record"].schema_version == FEATURE_SCHEMA_VERSION
    assert result["source_timestamp"] == pytest.approx(timestamp)
    record_provenance = result["feature_record"].provenance
    assert record_provenance["snapshot_id"]
    assert record_provenance["source_timestamps_monotonic"]["audio"] == timestamp
    anchor_utc = datetime.fromisoformat(
        record_provenance["source_timestamp_anchor_utc"]
    )
    expected_source_utc = anchor_utc + timedelta(
        seconds=timestamp
        - record_provenance["source_timestamp_anchor_monotonic"]
    )
    actual_source_utc = datetime.fromisoformat(
        record_provenance["source_timestamps_utc"]["audio"]
    )
    assert abs((actual_source_utc - expected_source_utc).total_seconds()) <= 0.001
    duplicate_snapshot = processor(
        {
            "latest_face_result": None,
            "latest_behavior_result": None,
            "latest_audio_result": audio,
            "last_face_success_timestamp": None,
            "last_behavior_success_timestamp": None,
            "last_audio_success_timestamp": timestamp,
        }
    )
    assert duplicate_snapshot["result_key"] == result["result_key"]
    assert (
        duplicate_snapshot["feature_record"].provenance["snapshot_id"]
        == record_provenance["snapshot_id"]
    )


def test_rule_inference_is_unknown_after_monitoring_stop_and_restart_resets_window():
    detector = RuleBasedCueDetector(
        RuleDetectionConfig(combination_mode="audio_only")
    )
    _evaluate(detector, _record(rms=0.12), 10)
    detector.reset()
    after_restart = _evaluate(detector, _record(rms=0.12), 11)

    assert after_restart.status == "no_sufficient_distress_evidence"
    assert after_restart.cue_indicators["audio_sustained"] is False


def test_monitoring_lifecycle_resets_stateful_rule_processor():
    class ResettableProcessor:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    from monitoring import MonitoringManager

    processor = ResettableProcessor()
    manager = MonitoringManager(inference_processor=processor)
    manager._reset_runtime_state()
    assert processor.reset_count == 1

    manager._set_stopped_state()
    assert processor.reset_count == 2
    assert manager.snapshot()["status"] == "stopped"


def test_monitoring_stop_clears_an_active_rule_alert():
    from monitoring import MonitoringManager

    inference = _evaluate(
        RuleBasedCueDetector(RuleDetectionConfig(combination_mode="audio_only")),
        _record(rms=0.3, onset=0.8),
        10,
    )
    alert_manager = AlertManager()
    active_alert = alert_manager.update(
        inference,
        result_timestamp=10,
        now=10,
        result_key="active-rule",
    )
    manager = MonitoringManager(alert_manager=alert_manager)
    manager.latest_inference = inference
    manager.latest_alert = active_alert

    manager._set_stopped_state()

    stopped = manager.snapshot()
    assert stopped["status"] == "stopped"
    assert stopped["latest_inference"] is None
    assert stopped["latest_alert"].state == "unknown"
    assert stopped["latest_alert"].alert_id is None
