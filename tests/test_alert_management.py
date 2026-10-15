from types import SimpleNamespace

import pytest

from alert_management import AlertManager, AlertPolicy


def _inference(status="potential_distress", score=0.9, **kwargs):
    fields = dict(
        status=status,
        model_score=score,
        potential_distress_score=(
            None
            if score is None
            else score
            if status == "potential_distress"
            else 1.0 - score
        ),
        potential_distress_probability=(
            None
            if score is None
            else score
            if status == "potential_distress"
            else 1.0 - score
        ),
        valid_feature_count=5,
        available_modalities={"face": True, "behavior": True, "audio": False},
        reason=None,
    )
    fields.update(kwargs)
    if "potential_distress_score" in kwargs and "potential_distress_probability" not in kwargs:
        fields["potential_distress_probability"] = kwargs["potential_distress_score"]
    elif (
        "potential_distress_probability" in kwargs
        and "potential_distress_score" not in kwargs
    ):
        fields["potential_distress_score"] = kwargs["potential_distress_probability"]
    return SimpleNamespace(**fields)


def test_alert_activates_on_one_unique_fresh_qualifying_result():
    policy = AlertPolicy(smoothing_window=2, cooldown_seconds=5)
    manager = AlertManager(policy)
    first = manager.update(_inference(), result_timestamp=10, now=10)
    duplicate = manager.update(
        _inference(
            potential_distress_score=0.01,
            potential_distress_probability=0.01,
        ),
        result_timestamp=10,
        now=10.1,
    )
    second = manager.update(_inference(), result_timestamp=11, now=11)

    assert first.state == "potential_distress"
    assert first.alert_id
    assert duplicate.state == "potential_distress"
    assert duplicate.smoothed_score == first.smoothed_score
    assert second.state == "potential_distress"
    assert second.alert_id == first.alert_id


def test_activation_uses_raw_probability_even_when_smoothed_score_is_low():
    manager = AlertManager(AlertPolicy())

    def submit(status, score, key):
        return manager.update(
            _inference(
                status=status,
                score=score,
                potential_distress_score=score,
                potential_distress_probability=score,
            ),
            result_timestamp=key,
            now=key,
            result_key=key,
        )

    for timestamp, score in enumerate((0.0023, 0.0026, 0.0049, 0.0049), start=1):
        state = submit("no_sufficient_distress_evidence", score, timestamp)
        assert state.state == "normal"

    first = submit("potential_distress", 0.9998, 5)
    assert first.smoothed_score == pytest.approx(0.2029)
    assert first.state == "potential_distress"
    assert first.alert_id


def test_alert_smoothed_distress_score_is_distinct_from_model_class_score():
    manager = AlertManager(AlertPolicy(smoothing_window=1))
    inference = _inference(
        status="no_sufficient_distress_evidence",
        score=0.91,
    )

    alert = manager.update(inference, result_timestamp=5, now=5)

    assert inference.model_score == 0.91
    assert inference.potential_distress_score == pytest.approx(0.09)
    assert alert.smoothed_score == pytest.approx(0.09)
    assert alert.state == "normal"


def test_high_raw_probability_with_non_positive_label_does_not_activate():
    manager = AlertManager()
    result = manager.update(
        _inference(
            status="no_sufficient_distress_evidence",
            score=0.01,
            potential_distress_probability=0.99,
        ),
        result_timestamp=5,
        now=5,
    )

    assert result.state == "normal"
    assert result.alert_id is None


def test_activation_uses_raw_explicit_probability_not_predicted_class_score():
    manager = AlertManager(
        AlertPolicy(smoothing_window=1)
    )
    non_triggering = manager.update(
        _inference(
            status="potential_distress",
            score=0.99,
            potential_distress_score=0.74,
            potential_distress_probability=0.74,
        ),
        result_timestamp=1,
        now=1,
    )
    assert non_triggering.state == "normal"
    assert non_triggering.smoothed_score == pytest.approx(0.74)

    triggering = manager.update(
        _inference(
            status="potential_distress",
            score=0.01,
            potential_distress_score=0.75,
            potential_distress_probability=0.75,
        ),
        result_timestamp=2,
        now=2,
    )
    assert triggering.state == "potential_distress"
    assert triggering.smoothed_score == pytest.approx(0.75)
    assert triggering.alert_id
    assert triggering.activated_at_utc


def test_raw_probability_threshold_is_configurable():
    manager = AlertManager(AlertPolicy(raw_probability_threshold=0.65))
    result = manager.update(
        _inference(
            potential_distress_score=0.65,
            potential_distress_probability=0.65,
        ),
        result_timestamp=3,
        now=3,
    )

    assert result.state == "potential_distress"


def test_stopping_clears_active_alert_and_stale_evidence_cannot_reactivate_it():
    manager = AlertManager(
        AlertPolicy(
            smoothing_window=1,
            max_result_age_seconds=2,
        )
    )
    active = manager.update(_inference(), result_timestamp=10, now=10)
    assert active.state == "potential_distress"

    stopped = manager.report_unknown(
        "Monitoring is stopped; no fresh inference is being produced."
    )
    assert stopped.state == "unknown"
    assert "stopped" in stopped.reason.lower()
    assert stopped.alert_id is None
    assert stopped.activated_at_utc is None
    assert stopped.smoothed_score is None

    stale = manager.update(_inference(), result_timestamp=10, now=13)
    assert stale.state == "unknown"
    assert "stale" in stale.reason.lower()


def test_stop_then_restart_requires_new_fresh_result():
    manager = AlertManager(
        AlertPolicy(smoothing_window=1, cooldown_seconds=10)
    )
    active = manager.update(_inference(), result_timestamp=10, now=10)

    reset = manager.report_unknown("Monitoring stopped.")
    assert reset.state == "unknown"
    assert reset.alert_id is None

    after_restart = manager.update(
        _inference(), result_timestamp=12, now=12, result_key=12
    )
    assert after_restart.state == "potential_distress"
    assert after_restart.alert_id != active.alert_id


def test_stale_or_insufficient_result_is_unknown_not_normal():
    manager = AlertManager(AlertPolicy(max_result_age_seconds=2))
    stale = manager.update(_inference(), result_timestamp=1, now=4)
    insufficient = manager.update(
        _inference("unknown", None, valid_feature_count=0, reason="No evidence."),
        result_timestamp=4,
        now=4,
    )
    assert stale.state == "unknown"
    assert "stale" in stale.reason.lower()
    assert insufficient.state == "unknown"


def test_invalid_raw_probability_is_rejected_without_fabricating_an_alert():
    manager = AlertManager()

    invalid = manager.update(
        _inference(
            potential_distress_probability="not-a-probability",
        ),
        result_timestamp=5,
        now=5,
    )

    assert invalid.state == "unknown"
    assert "unavailable or invalid" in invalid.reason
    assert invalid.alert_id is None


def test_unavailable_modalities_cannot_trigger_alert():
    manager = AlertManager()
    no_modalities = _inference(
        available_modalities={"face": False, "behavior": False, "audio": False}
    )

    result = manager.update(no_modalities, result_timestamp=5, now=5)

    assert result.state == "unknown"
    assert "modality" in result.reason.lower()


def test_single_modality_evidence_is_rejected_by_default_policy():
    manager = AlertManager()
    audio_only = _inference(
        status="potential_distress",
        score=0.91,
        valid_feature_count=5,
        available_modalities={"face": False, "behavior": False, "audio": True},
    )

    result = manager.update(audio_only, result_timestamp=6, now=6)

    assert result.state == "unknown"
    assert "modality" in result.reason.lower()


def test_hysteresis_and_cooldown_are_clock_controlled():
    policy = AlertPolicy(
        release_count=2,
        smoothing_window=1,
        cooldown_seconds=10,
    )
    manager = AlertManager(policy)
    active = manager.update(_inference(), result_timestamp=1, now=1)
    assert active.state == "potential_distress"

    manager.update(
        _inference(
            "potential_distress",
            0.9,
            potential_distress_score=0.54,
            potential_distress_probability=0.54,
        ),
        result_timestamp=2,
        now=2,
    )
    released = manager.update(
        _inference(
            "potential_distress",
            0.9,
            potential_distress_score=0.54,
            potential_distress_probability=0.54,
        ),
        result_timestamp=3,
        now=3,
    )
    assert released.state == "cooldown"
    still_cooling = manager.update(
        _inference(), result_timestamp=4, now=4
    )
    assert still_cooling.state == "cooldown"
    assert manager.snapshot(now=13).state == "normal"
    reactivated = manager.update(_inference(), result_timestamp=14, now=14)
    assert reactivated.state == "potential_distress"


def test_policy_rejects_ambiguous_thresholds():
    with pytest.raises(ValueError, match="Thresholds"):
        AlertPolicy(raw_probability_threshold=0.4, release_threshold=0.5)
