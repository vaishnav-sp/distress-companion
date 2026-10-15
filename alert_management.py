"""Local, non-clinical alert-state management for model inference results."""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable


@dataclass(frozen=True)
class AlertPolicy:
    """Engineering controls; thresholds are not clinically validated."""

    raw_probability_threshold: float = 0.75
    release_threshold: float = 0.55
    smoothing_window: int = 5
    release_count: int = 2
    cooldown_seconds: float = 30.0
    max_result_age_seconds: float = 3.0
    minimum_valid_features: int = 1
    minimum_available_modalities: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.release_threshold < self.raw_probability_threshold <= 1.0:
            raise ValueError(
                "Thresholds must satisfy 0 <= release < raw probability <= 1."
            )
        if self.smoothing_window < 1 or self.release_count < 1:
            raise ValueError("Smoothing window and release count must be positive.")
        if self.cooldown_seconds < 0 or self.max_result_age_seconds <= 0:
            raise ValueError("Cooldown must be nonnegative and maximum result age positive.")
        if self.minimum_valid_features < 1 or self.minimum_available_modalities < 1:
            raise ValueError("Evidence requirements must be positive.")


@dataclass(frozen=True)
class AlertSnapshot:
    state: str
    reason: str
    smoothed_score: float | None
    alert_id: str | None
    activated_at_utc: str | None
    status: str
    valid_feature_count: int
    available_modalities: tuple[str, ...]


class AlertManager:
    """Apply freshness/evidence gates, model or rule activation, and alert lifecycle."""

    def __init__(
        self,
        policy: AlertPolicy | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy or AlertPolicy()
        self._clock = clock
        self._scores: deque[float] = deque(maxlen=self.policy.smoothing_window)
        self._last_result_key: float | str | None = None
        self._release_count = 0
        self._cooldown_until: float | None = None
        self._state = "unknown"
        self._reason = "Waiting for a model inference result."
        self._smoothed_score: float | None = None
        self._alert_id: str | None = None
        self._activated_at_utc: str | None = None
        self._status = "unknown"
        self._valid_feature_count = 0
        self._available_modalities: tuple[str, ...] = ()

    def _snapshot(self) -> AlertSnapshot:
        return AlertSnapshot(
            state=self._state,
            reason=self._reason,
            smoothed_score=self._smoothed_score,
            alert_id=self._alert_id,
            activated_at_utc=self._activated_at_utc,
            status=self._status,
            valid_feature_count=self._valid_feature_count,
            available_modalities=self._available_modalities,
        )

    def diagnostic_state(self) -> dict[str, object]:
        """Return a read-only copy of alert-decision state for diagnostics."""
        return {
            "state": self._state,
            "reason": self._reason,
            "smoothed_score": self._smoothed_score,
            "release_counter": self._release_count,
            "last_result_key": self._last_result_key,
            "cooldown_until": self._cooldown_until,
            "raw_probability_threshold": self.policy.raw_probability_threshold,
            "release_threshold": self.policy.release_threshold,
            "release_persistence_count": self.policy.release_count,
            "cooldown_seconds": self.policy.cooldown_seconds,
            "max_result_age_seconds": self.policy.max_result_age_seconds,
            "minimum_valid_features": self.policy.minimum_valid_features,
            "minimum_available_modalities": self.policy.minimum_available_modalities,
        }

    def update(
        self,
        inference: object,
        *,
        result_timestamp: float | None,
        now: float | None = None,
        result_key: float | str | None = None,
    ) -> AlertSnapshot:
        """Consume one inference result; duplicate sample keys do not change state."""
        current_time = self._clock() if now is None else float(now)
        if self._state == "cooldown":
            if self._cooldown_until is not None and current_time < self._cooldown_until:
                self._reason = "An alert was recently issued; cooldown is active."
                return self._snapshot()
            self._state = "normal"
            self._reason = "Cooldown ended."
            self._cooldown_until = None

        status = str(getattr(inference, "status", "unknown"))
        is_rule_based = getattr(inference, "decision_source", None) == "rule_based"
        valid_count = int(getattr(inference, "valid_feature_count", 0) or 0)
        modalities = getattr(inference, "available_modalities", {}) or {}
        available = tuple(sorted(name for name, is_available in modalities.items() if is_available))
        self._status = status
        self._valid_feature_count = valid_count
        self._available_modalities = available

        invalid_reason = None
        if result_timestamp is None:
            invalid_reason = "No source timestamp is available; result freshness cannot be verified."
        elif not math.isfinite(float(result_timestamp)):
            invalid_reason = "Source timestamp is invalid."
        elif current_time - float(result_timestamp) > self.policy.max_result_age_seconds:
            invalid_reason = "Inference result is stale."
        elif current_time < float(result_timestamp):
            invalid_reason = "Source timestamp is ahead of the monitoring clock."
        elif status == "unknown":
            invalid_reason = getattr(inference, "reason", None) or "Inference evidence is insufficient."
        elif valid_count < self.policy.minimum_valid_features:
            invalid_reason = "Valid-feature count is below the alert policy minimum."
        elif is_rule_based and getattr(
            inference, "rule_evidence_sufficient", None
        ) is not True:
            invalid_reason = (
                getattr(inference, "reason", None)
                or "Rule cue evidence is insufficient."
            )
        elif (
            not is_rule_based
            and len(available) < self.policy.minimum_available_modalities
        ):
            invalid_reason = "Available-modality count is below the alert policy minimum."
        elif status not in {"potential_distress", "no_sufficient_distress_evidence", "normal"}:
            invalid_reason = f"Unsupported inference status {status!r}."

        if invalid_reason:
            self._state = "unknown"
            self._reason = str(invalid_reason)
            self._scores.clear()
            self._release_count = 0
            self._smoothed_score = None
            self._alert_id = None
            self._activated_at_utc = None
            return self._snapshot()

        key = result_key if result_key is not None else result_timestamp
        if key == self._last_result_key:
            return self._snapshot()
        self._last_result_key = key

        if is_rule_based:
            rule_triggered = getattr(inference, "rule_triggered", None)
            if not isinstance(rule_triggered, bool):
                self._state = "unknown"
                self._reason = "Rule cue result is unavailable or invalid."
                self._scores.clear()
                self._release_count = 0
                self._smoothed_score = None
                self._alert_id = None
                self._activated_at_utc = None
                return self._snapshot()
            raw_score = None
            self._scores.clear()
            self._smoothed_score = None
            activation_qualified = (
                status == "potential_distress" and rule_triggered
            )
        else:
            raw_score = getattr(
                inference,
                "potential_distress_probability",
                getattr(inference, "potential_distress_score", None),
            )
            try:
                raw_score = float(raw_score)
            except (TypeError, ValueError):
                raw_score = math.nan
            if not math.isfinite(raw_score) or not 0.0 <= raw_score <= 1.0:
                self._state = "unknown"
                self._reason = "Potential-distress class probability is unavailable or invalid."
                self._scores.clear()
                self._release_count = 0
                self._smoothed_score = None
                self._alert_id = None
                self._activated_at_utc = None
                return self._snapshot()

            is_positive = status == "potential_distress"
            self._scores.append(raw_score)
            self._smoothed_score = sum(self._scores) / len(self._scores)
            activation_qualified = (
                is_positive and raw_score >= self.policy.raw_probability_threshold
            )

        if self._state == "potential_distress":
            if is_rule_based:
                should_release = not activation_qualified
            else:
                should_release = (
                    not is_positive or raw_score < self.policy.release_threshold
                )
            if should_release:
                self._release_count += 1
            else:
                self._release_count = 0
            if self._release_count >= self.policy.release_count:
                self._state = "cooldown" if self.policy.cooldown_seconds else "normal"
                self._reason = (
                    "Configured cue persistence ended; alert released."
                    if is_rule_based
                    else "Positive inference persistence ended; alert released."
                )
                self._cooldown_until = (
                    current_time + self.policy.cooldown_seconds
                    if self.policy.cooldown_seconds
                    else None
                )
                self._release_count = 0
            else:
                self._reason = "Alert remains active under hysteresis."
        elif activation_qualified:
            self._state = "potential_distress"
            self._alert_id = str(uuid.uuid4())
            self._activated_at_utc = datetime.now(timezone.utc).isoformat()
            self._reason = (
                "Alert issued immediately: configured potential-distress rule cue "
                "was active."
                if is_rule_based
                else (
                    "Alert issued immediately: raw potential-distress probability "
                    "met the configured threshold."
                )
            )
            self._release_count = 0
        else:
            self._state = "normal"
            if is_rule_based:
                self._reason = getattr(inference, "reason", None) or (
                    "No configured potential-distress cue is active."
                )
            else:
                self._reason = (
                    "No sufficient distress evidence."
                    if status in {"normal", "no_sufficient_distress_evidence"}
                    else "Raw potential-distress probability is below the activation threshold."
                )
        return self._snapshot()

    def snapshot(self, *, now: float | None = None) -> AlertSnapshot:
        current_time = self._clock() if now is None else float(now)
        if self._state == "cooldown" and self._cooldown_until is not None:
            if current_time >= self._cooldown_until:
                self._state = "normal"
                self._reason = "Cooldown ended."
                self._cooldown_until = None
        return self._snapshot()

    def report_unknown(self, reason: str) -> AlertSnapshot:
        """Explicitly set unknown when inference could not be run."""
        self._state = "unknown"
        self._reason = reason
        self._scores.clear()
        self._release_count = 0
        self._cooldown_until = None
        self._last_result_key = None
        self._smoothed_score = None
        self._alert_id = None
        self._activated_at_utc = None
        self._status = "unknown"
        self._valid_feature_count = 0
        self._available_modalities = ()
        return self._snapshot()
