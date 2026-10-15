"""Configurable, non-clinical rule cues for live audio and movement features."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import time
from typing import Any, Mapping

from fusion.schema import FeatureRecord


@dataclass(frozen=True)
class RuleDetectionConfig:
    """Tunable demonstration settings; these values are not clinically validated."""

    combination_mode: str = "any"
    max_result_age_seconds: float = 3.0
    cue_window_seconds: float = 4.0
    max_sample_gap_seconds: float = 1.5
    audio_sustained_rms_threshold: float = 0.08
    audio_sustained_seconds: float = 2.0
    audio_peak_rms_threshold: float = 0.25
    audio_onset_threshold: float = 0.35
    audio_peak_window_seconds: float = 1.5
    audio_peak_min_events: int = 1
    movement_repetition_min_rate_hz: float = 0.5
    movement_repetition_min_amplitude: float = 0.06
    movement_repetition_min_elevated_proportion: float = 0.5
    movement_repetition_seconds: float = 2.0
    movement_vigorous_min_energy: float = 0.8
    movement_vigorous_min_peak_speed: float = 1.0
    movement_vigorous_min_acceleration: float = 2.0
    movement_vigorous_seconds: float = 1.0
    movement_sudden_threshold: float = 0.75
    movement_regions: tuple[str, ...] = (
        "head",
        "left_arm",
        "right_arm",
        "torso",
        "left_leg",
        "right_leg",
        "whole_body",
    )

    def __post_init__(self) -> None:
        if self.combination_mode not in {
            "any",
            "audio_only",
            "movement_only",
            "audio_and_movement",
        }:
            raise ValueError("Unsupported rule combination mode.")
        valid_regions = {
            "head",
            "left_arm",
            "right_arm",
            "torso",
            "left_leg",
            "right_leg",
            "whole_body",
        }
        if any(region not in valid_regions for region in self.movement_regions):
            raise ValueError("Unsupported movement region in rule configuration.")
        if (
            self.max_result_age_seconds <= 0
            or self.cue_window_seconds <= 0
            or self.max_sample_gap_seconds <= 0
            or self.audio_sustained_seconds < 0
            or self.audio_peak_window_seconds <= 0
            or self.audio_peak_min_events < 1
            or self.movement_repetition_seconds < 0
            or self.movement_vigorous_seconds < 0
        ):
            raise ValueError("Rule windows and evidence requirements must be positive.")
        thresholds = (
            self.audio_sustained_rms_threshold,
            self.audio_peak_rms_threshold,
            self.audio_onset_threshold,
            self.movement_repetition_min_rate_hz,
            self.movement_repetition_min_amplitude,
            self.movement_vigorous_min_energy,
            self.movement_vigorous_min_peak_speed,
            self.movement_vigorous_min_acceleration,
            self.movement_sudden_threshold,
        )
        if any(not math.isfinite(value) or value < 0 for value in thresholds):
            raise ValueError("Rule thresholds must be finite and nonnegative.")
        proportions = (
            self.movement_repetition_min_elevated_proportion,
        )
        if any(not 0 <= value <= 1 for value in proportions):
            raise ValueError("Movement proportions must be between zero and one.")


@dataclass(frozen=True)
class RuleInferenceResult:
    """Rule-cue result; deliberately has no model probability."""

    status: str
    predicted_label: str
    valid_feature_count: int
    available_modalities: dict[str, bool]
    reason: str
    decision_source: str
    rule_triggered: bool | None
    rule_evidence_sufficient: bool
    cue_indicators: dict[str, bool | None]
    cue_reason: str
    modality_cues: dict[str, bool | None]
    modality_evidence_status: dict[str, str]
    modality_evidence_reasons: dict[str, str]
    metadata: dict[str, Any]
    model_score: None = None
    potential_distress_score: None = None
    potential_distress_probability: None = None


class RuleBasedCueDetector:
    """Evaluate validity-gated, timestamped audio and movement cue rules."""

    def __init__(self, config: RuleDetectionConfig | None = None) -> None:
        self.config = config or RuleDetectionConfig()
        self.reset()

    def reset(self) -> None:
        self._audio_history: deque[
            tuple[float, float, bool, bool | None]
        ] = deque()
        self._movement_history: deque[
            tuple[float, bool | None, bool | None, bool | None]
        ] = deque()
        self._region_repetition_history: dict[
            str, deque[tuple[float, bool | None]]
        ] = {region: deque() for region in self.config.movement_regions}
        self._last_audio_timestamp: float | None = None
        self._last_movement_timestamp: float | None = None
        self._previous_audio_rms: float | None = None

    @staticmethod
    def _valid(record: FeatureRecord, name: str) -> bool:
        value = record.feature_values.get(name)
        return bool(record.feature_validity.get(name, False)) and value is not None and math.isfinite(
            float(value)
        )

    @staticmethod
    def _value(record: FeatureRecord, name: str) -> float | None:
        if not RuleBasedCueDetector._valid(record, name):
            return None
        value = record.feature_values[name]
        return float(value) if value is not None else None

    @staticmethod
    def _coerce_timestamp(timestamp: Any) -> float | None:
        if timestamp is None:
            return None
        try:
            value = float(timestamp)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value

    def _fresh(
        self,
        modality: str,
        timestamp: float | None,
        now: float,
    ) -> bool:
        normalized = self._coerce_timestamp(timestamp)
        if normalized is None:
            return False
        age = now - normalized
        return 0.0 <= age <= self.config.max_result_age_seconds

    def _evidence_status(
        self,
        modality: str,
        timestamp: float | None,
        now: float,
        cue: bool | None,
        *,
        partial: bool,
    ) -> tuple[str, str]:
        if timestamp is None:
            return "missing_timestamp", f"{modality} has no source timestamp."
        age = now - float(timestamp)
        if age < 0:
            return "future_timestamp", f"{modality} source timestamp is in the future."
        if age > self.config.max_result_age_seconds:
            return (
                "stale",
                f"{modality} evidence is {age:.2f}s old; maximum age is "
                f"{self.config.max_result_age_seconds:.2f}s.",
            )
        if cue is None:
            return "unavailable", f"{modality} cue inputs are missing or invalid."
        if cue:
            return "triggered", f"{modality} has an active configured cue."
        if partial:
            return (
                "partially_available",
                f"{modality} had usable inactive cue evidence; one or more other "
                "cue inputs were unavailable.",
            )
        return "evaluated_inactive", f"{modality} cues were evaluated and inactive."

    @staticmethod
    def _unavailable_feature_reason(
        record: FeatureRecord,
        feature_names: tuple[str, ...],
    ) -> str:
        unavailable: list[str] = []
        for name in feature_names:
            if name not in record.feature_names:
                unavailable.append(
                    f"{name}: expected feature name is absent from the feature record "
                    "(branch-to-schema name mismatch)."
                )
                continue
            if RuleBasedCueDetector._valid(record, name):
                continue
            reason = record.unavailable_reasons.get(
                name, "feature validity is false or its value is missing"
            )
            if reason.startswith("No current evidence was published for"):
                reason = (
                    f"{reason} Check that the behavior branch publishes this "
                    "feature under its canonical name."
                )
            unavailable.append(f"{name}: {reason}")
        return "; ".join(unavailable) or "Required cue input is unavailable."

    @staticmethod
    def _aggregate_cues(values: tuple[bool | None, ...]) -> bool | None:
        if True in values:
            return True
        if any(value is False for value in values):
            return False
        return None

    @staticmethod
    def _run_duration(
        history: deque[tuple[float, Any]],
        *,
        max_gap: float,
    ) -> float:
        duration = 0.0
        previous: float | None = None
        for sample_time, qualifies in reversed(history):
            if not qualifies:
                break
            if previous is not None:
                gap = previous - sample_time
                if gap <= 0 or gap > max_gap:
                    break
                duration += gap
            previous = sample_time
        return duration

    def _trim(self, history: deque, timestamp: float) -> None:
        cutoff = timestamp - self.config.cue_window_seconds
        while history and history[0][0] < cutoff:
            history.popleft()

    def evaluate(
        self,
        record: FeatureRecord,
        *,
        modality_source_timestamps: Mapping[str, float | None],
        now: float | None = None,
        region_movement: Mapping[str, Any] | None = None,
    ) -> RuleInferenceResult:
        current_time = time.monotonic() if now is None else float(now)
        timestamps = {
            modality: self._coerce_timestamp(value)
            for modality, value in dict(modality_source_timestamps).items()
        }
        audio_fresh = self._fresh("audio", timestamps.get("audio"), current_time)
        movement_fresh = self._fresh(
            "behavior", timestamps.get("behavior"), current_time
        )

        audio_rms = self._value(record, "audio.rms_loudness") if audio_fresh else None
        audio_onset = self._value(record, "audio.onset_score") if audio_fresh else None
        audio_timestamp = timestamps.get("audio")
        audio_available = audio_rms is not None
        audio_sustained: bool | None = None
        audio_peak: bool | None = None
        if audio_available and audio_timestamp is not None:
            sample_time = float(audio_timestamp)
            if self._last_audio_timestamp is None or sample_time > self._last_audio_timestamp:
                if (
                    self._last_audio_timestamp is not None
                    and sample_time - self._last_audio_timestamp
                    > self.config.max_sample_gap_seconds
                ):
                    self._audio_history.clear()
                    self._previous_audio_rms = None
                peak: bool | None = (
                    audio_onset is not None
                    and audio_onset >= self.config.audio_onset_threshold
                ) or (
                    self._previous_audio_rms is not None
                    and self._previous_audio_rms
                    < self.config.audio_peak_rms_threshold
                    <= audio_rms
                )
                if (
                    audio_onset is None
                    and audio_rms >= self.config.audio_peak_rms_threshold
                    and self._previous_audio_rms is None
                ):
                    peak = None
                above_sustained = audio_rms >= self.config.audio_sustained_rms_threshold
                self._audio_history.append(
                    (sample_time, audio_rms, above_sustained, peak)
                )
                self._trim(self._audio_history, sample_time)
                self._last_audio_timestamp = sample_time
                self._previous_audio_rms = audio_rms
                peak_history: deque[tuple[float, Any]] = deque(
                    (item_time, is_peak)
                    for item_time, _, _, is_peak in self._audio_history
                    if sample_time - item_time <= self.config.audio_peak_window_seconds
                )
                peak_count = sum(
                    is_peak is True for _, is_peak in peak_history
                )
                audio_peak = (
                    True
                    if peak_count >= self.config.audio_peak_min_events
                    else None
                    if any(is_peak is None for _, is_peak in peak_history)
                    else False
                )
            elif sample_time == self._last_audio_timestamp:
                peak_events = [
                    is_peak
                    for item_time, _, _, is_peak in self._audio_history
                    if item_time == sample_time
                ]
                audio_peak = (
                    peak_events[-1] if peak_events else None
                )
            else:
                audio_peak = None
            if self._audio_history and audio_timestamp is not None:
                timed_audio_history: deque[tuple[float, Any]] = deque(
                    (item_time, qualifies)
                    for item_time, _, qualifies, _ in self._audio_history
                )
                duration = self._run_duration(
                    timed_audio_history,
                    max_gap=self.config.max_sample_gap_seconds,
                )
                last_sample = self._audio_history[-1]
                audio_sustained = bool(
                    last_sample[2]
                    and duration >= self.config.audio_sustained_seconds
                )

        movement_timestamp = timestamps.get("behavior")
        movement_values: dict[str, float | None] = {}
        if movement_fresh:
            movement_values = {
                name: self._value(record, f"behavior.{name}")
                for name in (
                    "movement_energy",
                    "sudden_movement",
                    "wrist_peak_speed",
                    "wrist_peak_acceleration",
                    "wrist_movement_amplitude",
                    "wrist_repetition_rate_hz",
                    "wrist_elevated_movement_proportion",
                )
            }
        movement_timestamp_value = (
            float(movement_timestamp) if movement_fresh and movement_timestamp is not None else None
        )
        vigorous_sample: bool | None = None
        repetitive_sample: bool | None = None
        sudden_sample: bool | None = None
        if movement_timestamp_value is not None:
            energy = movement_values["movement_energy"]
            peak_speed = movement_values["wrist_peak_speed"]
            peak_acceleration = movement_values["wrist_peak_acceleration"]
            if energy is not None:
                if energy < self.config.movement_vigorous_min_energy:
                    vigorous_sample = False
                elif peak_speed is not None:
                    if peak_speed >= self.config.movement_vigorous_min_peak_speed:
                        vigorous_sample = True
                    elif peak_acceleration is not None:
                        vigorous_sample = (
                            peak_acceleration
                            >= self.config.movement_vigorous_min_acceleration
                        )
                elif peak_acceleration is not None:
                    vigorous_sample = (
                        peak_acceleration
                        >= self.config.movement_vigorous_min_acceleration
                    )
            rate = movement_values["wrist_repetition_rate_hz"]
            amplitude = movement_values["wrist_movement_amplitude"]
            elevated = movement_values["wrist_elevated_movement_proportion"]
            if rate is not None and amplitude is not None and elevated is not None:
                repetitive_sample = (
                    rate >= self.config.movement_repetition_min_rate_hz
                    and amplitude >= self.config.movement_repetition_min_amplitude
                    and elevated
                    >= self.config.movement_repetition_min_elevated_proportion
                )
            sudden = movement_values["sudden_movement"]
            if sudden is not None:
                sudden_sample = sudden >= self.config.movement_sudden_threshold
            if (
                self._last_movement_timestamp is None
                or movement_timestamp_value > self._last_movement_timestamp
            ):
                if (
                    self._last_movement_timestamp is not None
                    and movement_timestamp_value - self._last_movement_timestamp
                    > self.config.max_sample_gap_seconds
                ):
                    self._movement_history.clear()
                self._movement_history.append(
                    (
                        movement_timestamp_value,
                        repetitive_sample,
                        vigorous_sample,
                        sudden_sample,
                    )
                )
                self._trim(self._movement_history, movement_timestamp_value)
                self._last_movement_timestamp = movement_timestamp_value

        repetitive: bool | None = None
        vigorous: bool | None = None
        sudden_cue: bool | None = None
        if movement_fresh and movement_timestamp_value is not None:
            if repetitive_sample is not None and self._movement_history:
                repetitive_history: deque[tuple[float, Any]] = deque(
                    (sample_time, value)
                    for sample_time, value, _, _ in self._movement_history
                )
                repetitive = bool(
                    repetitive_sample
                    and self._run_duration(
                        repetitive_history,
                        max_gap=self.config.max_sample_gap_seconds,
                    )
                    >= self.config.movement_repetition_seconds
                )
            if vigorous_sample is not None and self._movement_history:
                vigorous_history: deque[tuple[float, Any]] = deque(
                    (sample_time, value)
                    for sample_time, _, value, _ in self._movement_history
                )
                vigorous = bool(
                    vigorous_sample
                    and self._run_duration(
                        vigorous_history,
                        max_gap=self.config.max_sample_gap_seconds,
                    )
                    >= self.config.movement_vigorous_seconds
                )
            if sudden_sample is not None:
                sudden_cue = sudden_sample

        region_samples: dict[str, bool | None] = {}
        region_sample_reasons: dict[str, str] = {}
        region_cues: dict[str, bool | None] = {}
        for region in (
            self.config.movement_regions if region_movement is not None else ()
        ):
            source = (region_movement or {}).get(region)
            metrics = source.get("metrics", {}) if isinstance(source, Mapping) else {}
            validity = source.get("validity", {}) if isinstance(source, Mapping) else {}
            reasons = source.get("reasons", {}) if isinstance(source, Mapping) else {}
            required = (
                "repetition_rate_hz",
                "amplitude",
                "elevated_movement_proportion",
            )
            values: dict[str, float] = {}
            unavailable: list[str] = []
            for name in required:
                value = metrics.get(name)
                is_valid = validity.get(name, False)
                try:
                    numeric_value = float(value) if value is not None else None
                except (TypeError, ValueError):
                    numeric_value = None
                if not is_valid or numeric_value is None or not math.isfinite(numeric_value):
                    unavailable.append(
                        f"{name}: {reasons.get(name, 'value is missing or invalid')}"
                    )
                else:
                    values[name] = numeric_value
            sample: bool | None = None
            reason = ""
            if not movement_fresh:
                reason = "Behavior evidence is missing or stale."
            elif unavailable:
                reason = "; ".join(unavailable)
            else:
                sample = (
                    values["repetition_rate_hz"]
                    >= self.config.movement_repetition_min_rate_hz
                    and values["amplitude"]
                    >= self.config.movement_repetition_min_amplitude
                    and values["elevated_movement_proportion"]
                    >= self.config.movement_repetition_min_elevated_proportion
                )
            region_samples[region] = sample
            region_sample_reasons[region] = reason
            history = self._region_repetition_history.setdefault(region, deque())
            if movement_timestamp_value is not None and (
                self._last_movement_timestamp is None
                or movement_timestamp_value >= self._last_movement_timestamp
            ):
                if history and history[-1][0] == movement_timestamp_value:
                    sample = history[-1][1]
                    region_samples[region] = sample
                elif not history or movement_timestamp_value > history[-1][0]:
                    if (
                        history
                        and movement_timestamp_value - history[-1][0]
                        > self.config.max_sample_gap_seconds
                    ):
                        history.clear()
                    history.append((movement_timestamp_value, sample))
                    self._trim(history, movement_timestamp_value)
            cue: bool | None = None
            if sample is not None and movement_timestamp_value is not None:
                cue = bool(
                    sample
                    and self._run_duration(
                        history,
                        max_gap=self.config.max_sample_gap_seconds,
                    )
                    >= self.config.movement_repetition_seconds
                )
            region_cues[region] = cue

        combined_repetitive = self._aggregate_cues(
            (repetitive, *region_cues.values())
        )
        cues: dict[str, bool | None] = {
            "audio_sustained": audio_sustained,
            "audio_peak": audio_peak,
            "movement_repetitive": combined_repetitive,
            "movement_repetitive_wrist": repetitive,
            "movement_vigorous": vigorous,
            "movement_sudden": sudden_cue,
            **{
                f"movement_repetitive_{region}": value
                for region, value in region_cues.items()
            },
        }
        audio_cue = self._aggregate_cues((audio_sustained, audio_peak))
        movement_cue_values = (
            combined_repetitive,
            vigorous,
            sudden_cue,
            *region_cues.values(),
        )
        movement_cue = self._aggregate_cues(movement_cue_values)
        modality_cues = {"audio": audio_cue, "behavior": movement_cue}
        modality_status: dict[str, str] = {}
        modality_reasons: dict[str, str] = {}
        for modality, cue, fresh in (
            ("audio", audio_cue, audio_fresh),
            ("behavior", movement_cue, movement_fresh),
        ):
            partial = cue is not None and any(
                value is None
                for value in (
                    (audio_sustained, audio_peak)
                    if modality == "audio"
                    else movement_cue_values
                )
            )
            status_value, reason_value = self._evidence_status(
                modality,
                timestamps.get(modality),
                current_time,
                cue if fresh else None,
                partial=partial,
            )
            modality_status[modality] = status_value
            modality_reasons[modality] = reason_value

        cue_unavailable_reasons = {
            "audio_sustained": (
                self._unavailable_feature_reason(record, ("audio.rms_loudness",))
                if audio_sustained is None and audio_fresh
                else modality_reasons["audio"]
                if audio_sustained is None
                else ""
            ),
            "audio_peak": (
                self._unavailable_feature_reason(record, ("audio.onset_score",))
                if audio_peak is None and audio_fresh
                else modality_reasons["audio"]
                if audio_peak is None
                else ""
            ),
            "movement_repetitive": (
                self._unavailable_feature_reason(
                    record,
                    (
                        "behavior.wrist_repetition_rate_hz",
                        "behavior.wrist_movement_amplitude",
                        "behavior.wrist_elevated_movement_proportion",
                    ),
                )
                if repetitive is None and movement_fresh
                else modality_reasons["behavior"]
                if repetitive is None
                else ""
            ),
            "movement_vigorous": (
                self._unavailable_feature_reason(
                    record,
                    (
                        "behavior.movement_energy",
                        "behavior.wrist_peak_speed",
                        "behavior.wrist_peak_acceleration",
                    ),
                )
                if vigorous is None and movement_fresh
                else modality_reasons["behavior"]
                if vigorous is None
                else ""
            ),
            "movement_sudden": (
                self._unavailable_feature_reason(
                    record, ("behavior.sudden_movement",)
                )
                if sudden_cue is None and movement_fresh
                else modality_reasons["behavior"]
                if sudden_cue is None
                else ""
            ),
        }
        cue_unavailable_reasons["movement_repetitive_wrist"] = (
            cue_unavailable_reasons["movement_repetitive"]
        )
        region_cue_unavailable_reasons = {
            f"movement_repetitive_{region}": (
                region_sample_reasons[region]
                or modality_reasons["behavior"]
                if cue is None
                else ""
            )
            for region, cue in region_cues.items()
        }
        cue_unavailable_reasons.update(region_cue_unavailable_reasons)
        if combined_repetitive is None:
            cue_unavailable_reasons["movement_repetitive"] = "; ".join(
                detail
                for detail in (
                    cue_unavailable_reasons["movement_repetitive_wrist"],
                    *region_cue_unavailable_reasons.values(),
                )
                if detail
            )
        else:
            cue_unavailable_reasons["movement_repetitive"] = ""
        if audio_cue is None:
            audio_details = tuple(
                detail
                for cue_name in ("audio_sustained", "audio_peak")
                if (detail := cue_unavailable_reasons[cue_name])
            )
            modality_reasons["audio"] = (
                "No evaluable audio cue. " + "; ".join(audio_details)
                if audio_details
                else "Audio cue inputs are missing or invalid."
            )
        if movement_cue is None:
            movement_details = tuple(
                detail
                for cue_name in (
                    "movement_repetitive",
                    "movement_vigorous",
                    "movement_sudden",
                    *region_cue_unavailable_reasons,
                )
                if (detail := cue_unavailable_reasons[cue_name])
            )
            modality_reasons["behavior"] = (
                "No evaluable movement cue. " + "; ".join(movement_details)
                if movement_details
                else "Behavior cue inputs are missing or invalid."
            )
        mode = self.config.combination_mode
        if mode == "audio_only":
            selected_available = audio_cue is not None
            triggered = audio_cue
        elif mode == "movement_only":
            selected_available = movement_cue is not None
            triggered = movement_cue
        elif mode == "audio_and_movement":
            selected_available = audio_cue is not None and movement_cue is not None
            triggered = (
                bool(audio_cue and movement_cue)
                if selected_available
                else None
            )
        else:
            triggered = self._aggregate_cues((audio_cue, movement_cue))
            selected_available = triggered is not None

        sufficient = bool(selected_available)
        region_evidence_sufficient = any(
            cue is not None for cue in region_cues.values()
        )
        available_modalities = dict(record.available_modalities)
        if region_evidence_sufficient:
            available_modalities["behavior"] = True
        status = (
            "potential_distress"
            if triggered is True
            else "no_sufficient_distress_evidence"
            if triggered is False
            else "unknown"
        )
        reason = self._reason(
            status,
            mode,
            cues,
            modality_cues=modality_cues,
            modality_reasons=modality_reasons,
        )
        cue_reason = (
            reason
            if status == "unknown"
            else "; ".join(
                f"{modality}: {modality_reasons[modality]}"
                for modality in ("audio", "behavior")
            )
        )
        return RuleInferenceResult(
            status=status,
            predicted_label=status,
            valid_feature_count=record.valid_feature_count,
            available_modalities=available_modalities,
            reason=reason,
            decision_source="rule_based",
            rule_triggered=triggered,
            rule_evidence_sufficient=(
                sufficient
                and (
                    record.valid_feature_count > 0
                    or region_evidence_sufficient
                )
            ),
            cue_indicators=cues,
            cue_reason=cue_reason,
            modality_cues=modality_cues,
            modality_evidence_status=modality_status,
            modality_evidence_reasons=modality_reasons,
            metadata={
                "detection_mode": "rule_based",
                "combination_mode": mode,
                "rule_thresholds": {
                    name: value
                    for name, value in vars(self.config).items()
                    if name != "combination_mode"
                },
                "limitations": (
                    "Configured audio and movement cues are demonstration rules, "
                    "not a distress diagnosis or clinically validated detector."
                ),
                "modality_evidence_status": modality_status,
                "modality_evidence_reasons": modality_reasons,
                "cue_unavailable_reasons": cue_unavailable_reasons,
                "region_cue_states": region_cues,
                "region_cue_reasons": region_sample_reasons,
            },
        )

    @staticmethod
    def _reason(
        status: str,
        mode: str,
        cues: Mapping[str, bool | None],
        *,
        modality_cues: Mapping[str, bool | None],
        modality_reasons: Mapping[str, str],
    ) -> str:
        active = [name for name, value in cues.items() if value is True]
        if status == "potential_distress":
            return (
                "Configured potential-distress cue(s) active: "
                + ", ".join(active)
                + f" (combination mode: {mode})."
            )
        if status == "unknown":
            if mode == "any":
                unavailable = [
                    f"{'movement' if name == 'behavior' else name}: "
                    f"{modality_reasons[name]}"
                    for name, value in modality_cues.items()
                    if value is None
                ]
                detail = "; ".join(unavailable)
            elif mode == "audio_and_movement":
                unavailable = [
                    f"{'movement' if name == 'behavior' else name}: "
                    f"{modality_reasons[name]}"
                    for name, value in modality_cues.items()
                    if value is None
                ]
                detail = "; ".join(unavailable)
            else:
                modality = "audio" if mode == "audio_only" else "behavior"
                detail = modality_reasons[modality]
            return (
                f"Insufficient usable evidence for combination mode {mode}. "
                f"{detail} Missing or stale evidence is not treated as inactive."
            )
        return f"No configured cue met its rule in combination mode {mode}."


class RuleBasedInferenceProcessor:
    """Build fresh schema-v2 evidence and evaluate it without loading a classifier."""

    def __init__(self, config: RuleDetectionConfig | None = None) -> None:
        self.detector = RuleBasedCueDetector(config)

    def reset(self) -> None:
        self.detector.reset()

    def __call__(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        from fusion import build_fusion_evidence, evidence_to_feature_record

        now = time.monotonic()
        fresh_snapshot = dict(snapshot)
        modality_timestamps: dict[str, float | None] = {}
        fresh_timestamps: dict[str, float] = {}
        for modality in ("face", "behavior", "audio"):
            result_key = f"latest_{modality}_result"
            timestamp_key = f"last_{modality}_success_timestamp"
            result = snapshot.get(result_key)
            timestamp = snapshot.get(timestamp_key)
            numeric_timestamp = self.detector._coerce_timestamp(timestamp)
            is_fresh = (
                result is not None
                and numeric_timestamp is not None
                and 0.0 <= now - numeric_timestamp <= self.detector.config.max_result_age_seconds
            )
            if not is_fresh:
                fresh_snapshot[result_key] = None
            else:
                fresh_timestamps[modality] = numeric_timestamp
            modality_timestamps[modality] = numeric_timestamp

        evidence = build_fusion_evidence(fresh_snapshot)
        record = evidence_to_feature_record(
            evidence,
            provenance={"source": "live_rule_based_monitoring"},
        )
        behavior_result = fresh_snapshot.get("latest_behavior_result")
        if isinstance(behavior_result, Mapping):
            get_behavior_value = behavior_result.get
        else:
            get_behavior_value = lambda name, default=None: getattr(
                behavior_result,
                name,
                default,
            )
        region_metrics = get_behavior_value("region_metrics", {})
        region_validity = get_behavior_value("region_feature_validity", {})
        region_reasons = get_behavior_value("region_unavailable_reasons", {})
        region_movement = (
            {
                region: {
                    "metrics": region_metrics.get(region, {}),
                    "validity": region_validity.get(region, {}),
                    "reasons": region_reasons.get(region, {}),
                }
                for region in self.detector.config.movement_regions
            }
            if region_metrics
            else None
        )
        evaluation_started = time.perf_counter()
        inference = self.detector.evaluate(
            record,
            modality_source_timestamps=modality_timestamps,
            now=now,
            region_movement=region_movement,
        )
        rule_evaluation_ms = (
            time.perf_counter() - evaluation_started
        ) * 1000.0
        source_timestamp = max(fresh_timestamps.values(), default=None)
        result_key = "|".join(
            f"{modality}:{fresh_timestamps.get(modality, 'missing')}"
            for modality in ("face", "behavior", "audio")
        )
        return {
            "inference": inference,
            "feature_record": record,
            "stage_timings_ms": {"rule_evaluation": rule_evaluation_ms},
            "source_timestamp": source_timestamp,
            "modality_source_timestamps": modality_timestamps,
            "evidence_sufficient": inference.rule_evidence_sufficient,
            "evidence_sufficiency_reason": (
                None if inference.rule_evidence_sufficient else inference.reason
            ),
            "result_key": result_key,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }


__all__ = [
    "RuleBasedCueDetector",
    "RuleDetectionConfig",
    "RuleBasedInferenceProcessor",
    "RuleInferenceResult",
]
