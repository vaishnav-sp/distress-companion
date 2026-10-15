import pytest

from alert_management import AlertManager, AlertPolicy
from simulated_alert_test import (
    SimulatedDemoAlertManager,
    handle_simulated_feedback,
    run_simulated_alert_test,
)


def test_simulated_alert_exercises_activation_release_and_cooldown_only():
    policy = AlertPolicy(
        smoothing_window=5,
        release_count=2,
        cooldown_seconds=10,
    )

    result = run_simulated_alert_test(policy)

    assert result["active"].state == "potential_distress"
    assert result["active"].alert_id
    assert result["released"].state == "cooldown"
    assert result["cooldown_expired"].state == "normal"


@pytest.mark.parametrize("action", ["confirm", "reject", "report_missed_event"])
def test_simulated_feedback_is_in_memory_and_bound_to_simulated_alert(action):
    result = handle_simulated_feedback(action, "simulated-alert-id")

    assert result == {
        "action": action,
        "alert_id": "simulated-alert-id",
        "simulated": True,
        "persisted": False,
    }


def test_simulated_feedback_rejects_missing_alert_identity():
    with pytest.raises(ValueError, match="simulated alert ID"):
        handle_simulated_feedback("confirm", None)


def test_demo_alert_activates_immediately_and_reset_dismisses_it():
    manager = SimulatedDemoAlertManager()

    alert = manager.activate()

    assert alert.label == "SIMULATED DEMO ALERT"
    assert alert.state == "potential_distress"
    assert alert.issued_at_utc
    assert manager.alert == alert

    manager.reset()

    assert manager.alert is None
    assert manager.feedback == ()


def test_demo_feedback_is_transient_for_all_caretaker_actions():
    manager = SimulatedDemoAlertManager()
    alert = manager.activate()

    for action in ("confirm", "reject", "report_missed_event"):
        feedback = manager.record_feedback(action)
        assert feedback == {
            "action": action,
            "alert_id": alert.alert_id,
            "simulated": True,
            "persisted": False,
        }
    assert len(manager.feedback) == 3


def test_demo_alert_does_not_mutate_live_manager_or_live_history():
    live_manager = AlertManager()
    live_state_before = live_manager.diagnostic_state()
    live_history = [{"result_key": "existing-live-result"}]
    live_history_before = list(live_history)
    demo_manager = SimulatedDemoAlertManager()

    demo_manager.activate()
    demo_manager.record_feedback("report_missed_event")

    assert live_manager.diagnostic_state() == live_state_before
    assert live_history == live_history_before
    assert demo_manager.feedback[0]["persisted"] is False


def test_demo_feedback_requires_active_demo_alert():
    with pytest.raises(ValueError, match="must be active"):
        SimulatedDemoAlertManager().record_feedback("confirm")
