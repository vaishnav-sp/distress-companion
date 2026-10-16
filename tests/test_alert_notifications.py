import pytest

from alert_notifications import score_popup_transition
from config import EXPERIMENTAL_DISTRESS_ALERT_THRESHOLD
from distress_score import DistressScoreResult, score_alert_triggered


def _score_alert(score):
    result = DistressScoreResult(
        score=score,
        status="Available" if score is not None else "Insufficient evidence",
        modalities={},
        included_modalities=(),
        excluded_reasons={},
    )
    active = score_alert_triggered(
        result,
        threshold=EXPERIMENTAL_DISTRESS_ALERT_THRESHOLD,
    )
    return score_popup_transition(
        score_alert_active=active,
        previous_score_alert_active=False,
        score=result.score,
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (59.0, None),
        (60.0, "Potential distress detected\n\nDistress score: 60/100"),
        (75.0, "Potential distress detected\n\nDistress score: 75/100"),
    ],
)
def test_popup_uses_score_threshold_and_current_score(score, expected):
    popup_message, active = _score_alert(score)

    assert popup_message == expected
    assert active is (score >= EXPERIMENTAL_DISTRESS_ALERT_THRESHOLD)


@pytest.mark.parametrize("score", [None, 75.0])
@pytest.mark.parametrize("score_active", [None, False])
def test_unavailable_or_inactive_score_never_shows_popup(score, score_active):
    popup_message, active = score_popup_transition(
        score_alert_active=score_active,
        previous_score_alert_active=False,
        score=score,
    )

    assert popup_message is None
    assert active is False


def test_live_rule_cue_and_audio_peak_alone_cannot_trigger_popup():
    popup_message, active = score_popup_transition(
        score_alert_active=None,
        previous_score_alert_active=False,
        score=None,
    )

    assert popup_message is None
    assert active is False


def test_same_score_alert_does_not_repeat_on_rerun():
    first_message, active = _score_alert(75.0)
    rerun_message, active_after_rerun = score_popup_transition(
        score_alert_active=True,
        previous_score_alert_active=active,
        score=75.0,
    )

    assert first_message == "Potential distress detected\n\nDistress score: 75/100"
    assert rerun_message is None
    assert active_after_rerun is True


def test_new_score_alert_after_clear_shows_popup_again():
    clear_message, cleared = score_popup_transition(
        score_alert_active=False,
        previous_score_alert_active=True,
        score=59.0,
    )
    new_message, active = score_popup_transition(
        score_alert_active=True,
        previous_score_alert_active=cleared,
        score=75.0,
    )

    assert clear_message is None
    assert cleared is False
    assert new_message == "Potential distress detected\n\nDistress score: 75/100"
    assert active is True
