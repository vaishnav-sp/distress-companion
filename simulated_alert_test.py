"""Deterministic alert-policy exercise, isolated from all live monitoring state."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from alert_management import AlertManager, AlertPolicy


@dataclass(frozen=True)
class SimulatedDemoAlert:
    """A clearly identified demo alert, independent of live alert state."""

    alert_id: str
    state: str
    label: str
    issued_at_utc: str


class SimulatedDemoAlertManager:
    """Hold an immediate simulated alert and transient, non-persistent feedback."""

    def __init__(self) -> None:
        self._alert: SimulatedDemoAlert | None = None
        self._feedback: list[dict[str, Any]] = []

    @property
    def alert(self) -> SimulatedDemoAlert | None:
        return self._alert

    @property
    def feedback(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._feedback)

    def activate(self) -> SimulatedDemoAlert:
        self._alert = SimulatedDemoAlert(
            alert_id=f"simulated-{uuid.uuid4()}",
            state="potential_distress",
            label="SIMULATED DEMO ALERT",
            issued_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self._feedback.clear()
        return self._alert

    def record_feedback(self, action: str) -> dict[str, Any]:
        if self._alert is None:
            raise ValueError("A simulated demo alert must be active before feedback.")
        feedback = handle_simulated_feedback(action, self._alert.alert_id)
        self._feedback.append(feedback)
        return dict(feedback)

    def reset(self) -> None:
        self._alert = None
        self._feedback.clear()


def run_simulated_alert_test(policy: AlertPolicy) -> dict[str, Any]:
    """Drive synthetic outputs through a separate real AlertManager instance."""
    manager = AlertManager(policy)
    timestamp = time.monotonic()
    active = None
    for index in range(policy.smoothing_window + 1):
        timestamp += 0.1
        simulated_positive = SimpleNamespace(
            status="potential_distress",
            predicted_label="potential_distress",
            model_score=0.99,
            potential_distress_score=0.99,
            valid_feature_count=max(1, policy.minimum_valid_features),
            available_modalities={"face": True, "behavior": True, "audio": False},
            reason=None,
        )
        active = manager.update(
            simulated_positive,
            result_timestamp=timestamp,
            now=timestamp,
            result_key=f"sim-positive-{index}",
        )
        if active.state == "potential_distress":
            break

    if active is None or active.state != "potential_distress":
        raise RuntimeError("The alert policy did not activate for deterministic test scores.")

    released = active
    for index in range(policy.release_count + policy.smoothing_window):
        timestamp += 0.1
        simulated_negative = SimpleNamespace(
            status="no_sufficient_distress_evidence",
            predicted_label="no_sufficient_distress_evidence",
            model_score=0.99,
            potential_distress_score=0.01,
            valid_feature_count=max(1, policy.minimum_valid_features),
            available_modalities={"face": True, "behavior": True, "audio": False},
            reason=None,
        )
        released = manager.update(
            simulated_negative,
            result_timestamp=timestamp,
            now=timestamp,
            result_key=f"sim-negative-{index}",
        )
        if released.state in {"cooldown", "normal"}:
            break

    if released.state not in {"cooldown", "normal"}:
        raise RuntimeError("The alert policy did not release the simulated alert.")
    cooldown_expired = manager.snapshot(
        now=timestamp + policy.cooldown_seconds + 0.1
    )
    return {
        "active": active,
        "released": released,
        "cooldown_expired": cooldown_expired,
        "alert_id": active.alert_id,
    }


def handle_simulated_feedback(action: str, alert_id: str | None) -> dict[str, Any]:
    """Acknowledge a test-only alert in memory without touching live feedback storage."""
    if action not in {"confirm", "reject", "report_missed_event"}:
        raise ValueError(f"Unsupported simulated feedback action: {action!r}.")
    if not alert_id:
        raise ValueError("Simulated feedback requires the simulated alert ID.")
    return {
        "action": action,
        "alert_id": alert_id,
        "simulated": True,
        "persisted": False,
    }


__all__ = [
    "SimulatedDemoAlert",
    "SimulatedDemoAlertManager",
    "handle_simulated_feedback",
    "run_simulated_alert_test",
]
