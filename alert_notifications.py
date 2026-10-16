"""Score-based popup transition handling for the Streamlit dashboard."""

from __future__ import annotations


def score_popup_transition(
    *,
    score_alert_active: bool | None,
    previous_score_alert_active: bool,
    score: float | None,
) -> tuple[str | None, bool]:
    """Return a popup message only on a newly active, available score alert."""
    current_score_alert_active = score_alert_active is True and score is not None
    if current_score_alert_active and not previous_score_alert_active:
        return (
            "Potential distress detected\n\n"
            f"Distress score: {score:g}/100",
            True,
        )
    return None, current_score_alert_active
