"""Experimental, read-only branch evidence score for dashboard display.

This score is not a probability, diagnosis, or validated distress measurement.
It consumes existing branch outputs and does not affect detector or alert-manager
decisions; a separate dashboard-only score alert may consume its result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import time
from typing import Any

from config import (
    EXPERIMENTAL_DISTRESS_SCORE_FACE_MAX_ENTROPY,
    EXPERIMENTAL_DISTRESS_SCORE_FRESHNESS_SECONDS,
    EXPERIMENTAL_DISTRESS_SCORE_MIN_MODALITIES,
    EXPERIMENTAL_DISTRESS_SCORE_WEIGHTS,
    EXPERIMENTAL_DISTRESS_ALERT_THRESHOLD,
    RULE_MOVEMENT_REPETITION_MIN_AMPLITUDE,
    RULE_MOVEMENT_REPETITION_MIN_ELEVATED_PROPORTION,
    RULE_MOVEMENT_REPETITION_MIN_RATE_HZ,
    RULE_MOVEMENT_SUDDEN_THRESHOLD,
    RULE_MOVEMENT_VIGOROUS_MIN_ACCELERATION,
    RULE_MOVEMENT_VIGOROUS_MIN_ENERGY,
    RULE_MOVEMENT_VIGOROUS_MIN_PEAK_SPEED,
)

MODALITIES = ("emotion", "behavior", "audio")
_BRANCH_NAMES = {
    "emotion": "face",
    "behavior": "behavior",
    "audio": "audio",
}
_NEGATIVE_CLASSES = ("angry", "fear", "sad")
INTERPRETATION_WARNING = (
    "Experimental distress-related evidence score only — not a probability, "
    "diagnosis, or clinically validated distress measurement. Facial expression, "
    "movement, and acoustic activity are indirect contextual cues."
)


@dataclass(frozen=True)
class ModalityScore:
    """One branch's bounded evidence score and its use in the weighted result."""

    score: float | None
    available: bool
    weight: float
    contributed: bool
    contribution: float | None
    evidence: tuple[str, ...]
    reason: str | None


@dataclass(frozen=True)
class DistressScoreResult:
    """Read-only score result suitable for display and testing."""

    score: float | None
    status: str
    modalities: Mapping[str, ModalityScore]
    included_modalities: tuple[str, ...]
    excluded_reasons: Mapping[str, str]
    interpretation_warning: str = INTERPRETATION_WARNING


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _valid_feature(result: Any, name: str) -> tuple[float | None, str]:
    values = _field(result, "feature_values")
    validity = _field(result, "feature_validity")
    if not isinstance(values, Mapping) or not isinstance(validity, Mapping):
        return None, f"{name} feature values or validity metadata are unavailable."
    if validity.get(name) is not True:
        reasons = _field(result, "unavailable_reasons", {})
        detail = reasons.get(name) if isinstance(reasons, Mapping) else None
        return None, str(detail or f"{name} is marked invalid or unavailable.")
    number = _finite_number(values.get(name))
    if number is None:
        return None, f"{name} is missing, malformed, or non-finite."
    return number, ""


def _threshold(value: Any, name: str) -> float:
    number = _finite_number(value)
    if number is None or number <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return number


def _feature_score(
    result: Any,
    name: str,
    threshold: float,
    *,
    maximum: float | None = None,
) -> tuple[float | None, str]:
    value, reason = _valid_feature(result, name)
    if value is None:
        return None, reason
    if value < 0 or (maximum is not None and value > maximum):
        return None, f"{name} is outside its documented feature range."
    return min(value / threshold, 1.0) * 100.0, ""


def _emotion_score(
    result: Any,
    *,
    maximum_entropy: float,
) -> tuple[float | None, tuple[str, ...], str | None]:
    if result is None:
        return None, (), "No face-branch result is available."
    if _field(result, "face_detected") is not True:
        return None, (), "A usable face was not detected."
    if _field(result, "face_quality_status") != "usable":
        return None, (), "Face quality is not marked usable."
    quality = _finite_number(_field(result, "face_quality_score"))
    if quality is None or not 0.0 <= quality <= 1.0:
        return None, (), "Face quality score is missing or outside [0, 1]."

    probabilities = _field(result, "expression_probabilities")
    validity = _field(result, "class_validity")
    availability = _field(result, "class_availability")
    if not all(isinstance(item, Mapping) for item in (probabilities, validity, availability)):
        return None, (), "Expression probabilities or class metadata are unavailable."
    class_values: list[float] = []
    for label in _NEGATIVE_CLASSES:
        if availability.get(label) is not True:
            return None, (), f"The {label} expression class is not available."
        if validity.get(label) is not True:
            return None, (), f"The {label} expression estimate is invalid."
        value = _finite_number(probabilities.get(label))
        if value is None or not 0.0 <= value <= 1.0:
            return None, (), f"The {label} expression estimate is outside [0, 1]."
        class_values.append(value)

    negative_affect = _finite_number(_field(result, "negative_affect_feature"))
    expected = sum(class_values) / len(class_values)
    if (
        negative_affect is None
        or not 0.0 <= negative_affect <= 1.0
        or not math.isclose(negative_affect, expected, rel_tol=0.0, abs_tol=1e-6)
    ):
        return None, (), (
            "The negative-affect summary is missing, invalid, or inconsistent with "
            "the valid angry/fear/sad estimates."
        )

    supported_labels = tuple(
        label
        for label, is_available in availability.items()
        if is_available is True
    )
    if supported_labels and all(
        validity.get(label) is True
        and (
            (value := _finite_number(probabilities.get(label))) is not None
            and 0.0 <= value <= 1.0
        )
        for label in supported_labels
    ):
        supported_values = [
            float(probabilities[label]) for label in supported_labels
        ]
        if not math.isclose(
            sum(supported_values),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            return None, (), "Supported expression probabilities do not sum to 1."

    confidence = _field(result, "prediction_confidence")
    entropy = _finite_number(_field(confidence, "normalized_entropy"))
    if entropy is None or not 0.0 <= entropy <= 1.0:
        return None, (), "Normalized prediction entropy is missing or outside [0, 1]."
    if supported_labels and all(
        validity.get(label) is True
        and _finite_number(probabilities.get(label)) is not None
        for label in supported_labels
    ):
        supported_values = [
            float(probabilities[label]) for label in supported_labels
        ]
        calculated_entropy = (
            -sum(
                value * math.log(max(value, 1e-12))
                for value in supported_values
            )
            / math.log(len(supported_values))
        )
        if not math.isclose(
            entropy,
            calculated_entropy,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            return None, (), (
                "Normalized prediction entropy is inconsistent with the supported "
                "expression probabilities."
            )
    if entropy > maximum_entropy:
        return None, (), (
            f"Expression prediction uncertainty ({entropy:.3f} normalized entropy) "
            "exceeds the score-only cutoff."
        )

    return (
        negative_affect * 100.0,
        (
            "negative_affect_feature: mean of valid angry/fear/sad expression estimates",
            f"usable face quality {quality:.3f}",
            f"normalized entropy {entropy:.3f}",
        ),
        None,
    )


def _behavior_score(result: Any) -> tuple[float | None, tuple[str, ...], str | None]:
    if result is None:
        return None, (), "No behavior-branch result is available."

    repetition_thresholds = {
        "repetition_rate_hz": _threshold(
            RULE_MOVEMENT_REPETITION_MIN_RATE_HZ,
            "RULE_MOVEMENT_REPETITION_MIN_RATE_HZ",
        ),
        "amplitude": _threshold(
            RULE_MOVEMENT_REPETITION_MIN_AMPLITUDE,
            "RULE_MOVEMENT_REPETITION_MIN_AMPLITUDE",
        ),
        "elevated_movement_proportion": _threshold(
            RULE_MOVEMENT_REPETITION_MIN_ELEVATED_PROPORTION,
            "RULE_MOVEMENT_REPETITION_MIN_ELEVATED_PROPORTION",
        ),
    }
    energy_threshold = _threshold(
        RULE_MOVEMENT_VIGOROUS_MIN_ENERGY, "RULE_MOVEMENT_VIGOROUS_MIN_ENERGY"
    )
    speed_threshold = _threshold(
        RULE_MOVEMENT_VIGOROUS_MIN_PEAK_SPEED,
        "RULE_MOVEMENT_VIGOROUS_MIN_PEAK_SPEED",
    )
    acceleration_threshold = _threshold(
        RULE_MOVEMENT_VIGOROUS_MIN_ACCELERATION,
        "RULE_MOVEMENT_VIGOROUS_MIN_ACCELERATION",
    )
    sudden_threshold = _threshold(
        RULE_MOVEMENT_SUDDEN_THRESHOLD, "RULE_MOVEMENT_SUDDEN_THRESHOLD"
    )

    candidates: list[tuple[str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    candidates.append((
        "wrist",
        _field(result, "feature_values", {}),
        _field(result, "feature_validity", {}),
        _field(result, "unavailable_reasons", {}),
    ))
    region_metrics = _field(result, "region_metrics", {})
    region_validity = _field(result, "region_feature_validity", {})
    region_reasons = _field(result, "region_unavailable_reasons", {})
    if isinstance(region_metrics, Mapping):
        for region in sorted(
            key for key in region_metrics if isinstance(key, str)
        ):
            metrics = region_metrics.get(region)
            valid = region_validity.get(region, {}) if isinstance(region_validity, Mapping) else {}
            reasons = region_reasons.get(region, {}) if isinstance(region_reasons, Mapping) else {}
            if isinstance(metrics, Mapping) and isinstance(valid, Mapping):
                candidates.append((str(region), metrics, valid, reasons if isinstance(reasons, Mapping) else {}))

    family_scores: dict[str, tuple[float, str]] = {}
    exclusion_details: list[str] = []
    for source_name, values, validity, reasons in candidates:
        if not isinstance(values, Mapping) or not isinstance(validity, Mapping):
            continue

        repetition_names = (
            ("wrist_repetition_rate_hz", "repetition_rate_hz"),
            ("wrist_movement_amplitude", "amplitude"),
            (
                "wrist_elevated_movement_proportion",
                "elevated_movement_proportion",
            ),
        )
        repetition_values: list[float] = []
        repetition_evidence: list[str] = []
        repetition_invalid: list[str] = []
        for wrist_name, region_name in repetition_names:
            feature_name = wrist_name if source_name == "wrist" else region_name
            if validity.get(feature_name) is not True:
                detail = reasons.get(feature_name) if isinstance(reasons, Mapping) else None
                repetition_invalid.append(
                    f"{feature_name}: {detail or 'marked invalid or unavailable'}"
                )
                continue
            number = _finite_number(values.get(feature_name))
            maximum = 1.0 if feature_name == "elevated_movement_proportion" else None
            if number is None or number < 0 or (maximum is not None and number > maximum):
                repetition_invalid.append(f"{feature_name}: invalid or out of range")
                continue
            ratio = min(number / repetition_thresholds[region_name], 1.0)
            repetition_values.append(ratio)
            repetition_evidence.append(f"{feature_name} ratio {ratio:.3f}")
        if len(repetition_values) == len(repetition_names):
            repetition_score = min(repetition_values) * 100.0
            if (
                "repetitive" not in family_scores
                or repetition_score > family_scores["repetitive"][0]
            ):
                family_scores["repetitive"] = (
                    repetition_score,
                    f"{source_name} repetitive movement; weakest required ratio "
                    f"{min(repetition_values):.3f} ({', '.join(repetition_evidence)})",
                )
        elif repetition_invalid:
            exclusion_details.append(
                f"{source_name} repetitive movement unavailable: "
                + "; ".join(repetition_invalid)
            )

        energy_name = "movement_energy" if source_name == "wrist" else "movement_energy"
        speed_name = "wrist_peak_speed" if source_name == "wrist" else "peak_speed"
        acceleration_name = (
            "wrist_peak_acceleration"
            if source_name == "wrist"
            else "peak_acceleration"
        )
        energy_ratio, energy_reason = _feature_score(
            _FeatureView(values, validity, reasons),
            energy_name,
            energy_threshold,
        )
        speed_ratio, speed_reason = _feature_score(
            _FeatureView(values, validity, reasons),
            speed_name,
            speed_threshold,
        )
        acceleration_ratio, acceleration_reason = _feature_score(
            _FeatureView(values, validity, reasons),
            acceleration_name,
            acceleration_threshold,
        )
        if energy_ratio is not None:
            speed_gate = speed_ratio is not None and speed_ratio >= 100.0
            acceleration_gate = (
                acceleration_ratio is not None and acceleration_ratio >= 100.0
            )
            criterion_conclusive = (
                speed_gate
                or acceleration_gate
                or (speed_ratio is not None and acceleration_ratio is not None)
            )
            if criterion_conclusive:
                vigor_score = min(
                    energy_ratio,
                    max(
                        speed_ratio or 0.0,
                        acceleration_ratio or 0.0,
                    ),
                )
                if (
                    "vigorous" not in family_scores
                    or vigor_score > family_scores["vigorous"][0]
                ):
                    family_scores["vigorous"] = (
                        vigor_score,
                        f"{source_name} vigorous movement; energy ratio "
                        f"{energy_ratio / 100.0:.3f}, speed ratio "
                        f"{(speed_ratio or 0.0) / 100.0:.3f}, acceleration ratio "
                        f"{(acceleration_ratio or 0.0) / 100.0:.3f}",
                    )
            else:
                exclusion_details.append(
                    f"{source_name} vigorous movement could not evaluate the "
                    f"speed-or-acceleration criterion ({speed_reason or acceleration_reason})."
                )

        if source_name == "wrist":
            sudden_name = "sudden_movement"
        else:
            sudden_name = ""
        if sudden_name:
            sudden_score, _ = _feature_score(
                _FeatureView(values, validity, reasons),
                sudden_name,
                sudden_threshold,
                maximum=1.0,
            )
            if sudden_score is not None and (
                "sudden" not in family_scores
                or sudden_score > family_scores["sudden"][0]
            ):
                family_scores["sudden"] = (
                    sudden_score,
                    f"sudden_movement ratio "
                    f"{sudden_score / 100.0:.3f} of its configured threshold",
                )

    if not family_scores:
        reason = (
            "; ".join(dict.fromkeys(exclusion_details))
            or "No validity-gated movement feature set is usable."
        )
        return None, (), reason

    # Use the strongest independently defined movement family rather than
    # summing overlapping measurements of the same underlying movement.
    family_name = max(family_scores, key=lambda name: family_scores[name][0])
    score, explanation = family_scores[family_name]
    other_families = sorted(name for name in family_scores if name != family_name)
    evidence = (explanation,)
    if other_families:
        evidence += (
            "Other valid movement families were not added to avoid double-counting "
            "overlapping motion evidence.",
        )
    return min(max(score, 0.0), 100.0), evidence, None


class _FeatureView:
    """Small adapter letting the shared validity reader inspect regional metrics."""

    def __init__(
        self,
        values: Mapping[str, Any],
        validity: Mapping[str, Any],
        reasons: Any,
    ) -> None:
        self.feature_values = values
        self.feature_validity = validity
        self.unavailable_reasons = reasons


def _audio_score(result: Any) -> tuple[float | None, tuple[str, ...], str | None]:
    if result is None:
        return None, (), "No audio-branch result is available."
    rms, rms_reason = _valid_feature(result, "rms_loudness")
    if rms is None or _field(result, "rms_valid") is not True or rms < 0:
        return None, (), rms_reason or "RMS evidence is unavailable or invalid."
    contribution, contribution_reason = _valid_feature(result, "audio_contribution")
    branch_contribution = _finite_number(_field(result, "audio_contribution"))
    if (
        contribution is None
        or branch_contribution is None
        or not 0.0 <= contribution <= 1.0
        or not math.isclose(contribution, branch_contribution, rel_tol=0.0, abs_tol=1e-9)
    ):
        return None, (), (
            contribution_reason
            or "Audio contribution is missing, invalid, or inconsistent."
        )
    return (
        contribution * 100.0,
        (
            f"valid RMS loudness {rms:.6g}",
            f"bounded signal-activity audio_contribution {contribution:.3f}",
            (
                "valid silent-window RMS is retained as acoustic activity score 0"
                if _field(result, "status") == "silent" and contribution == 0.0
                else "audio contribution describes acoustic activity, not distress"
            ),
        ),
        None,
    )


def _validated_weights(weights: Mapping[str, Any] | None) -> dict[str, float]:
    configured = EXPERIMENTAL_DISTRESS_SCORE_WEIGHTS if weights is None else weights
    if not isinstance(configured, Mapping) or set(configured) != set(MODALITIES):
        raise ValueError(f"Weights must define exactly these modalities: {MODALITIES}.")
    parsed: dict[str, float] = {}
    for modality in MODALITIES:
        value = _finite_number(configured[modality])
        if value is None or value < 0:
            raise ValueError("Weights must be finite and nonnegative.")
        parsed[modality] = value
    if sum(parsed.values()) <= 0:
        raise ValueError("At least one modality weight must be positive.")
    return parsed


def calculate_distress_score(
    snapshot: Mapping[str, Any],
    *,
    now: float | None = None,
    weights: Mapping[str, Any] | None = None,
    freshness_seconds: float = EXPERIMENTAL_DISTRESS_SCORE_FRESHNESS_SECONDS,
    face_max_normalized_entropy: float = EXPERIMENTAL_DISTRESS_SCORE_FACE_MAX_ENTROPY,
    minimum_modalities: int = EXPERIMENTAL_DISTRESS_SCORE_MIN_MODALITIES,
) -> DistressScoreResult:
    """Calculate a deterministic dashboard score from one monitoring snapshot.

    Only fresh, explicitly valid branch evidence contributes. At least two
    positively weighted modalities are required for an overall score.
    """
    configured_weights = _validated_weights(weights)
    freshness = _finite_number(freshness_seconds)
    entropy_cutoff = _finite_number(face_max_normalized_entropy)
    if freshness is None or freshness <= 0:
        raise ValueError("Freshness limit must be a finite positive number.")
    if entropy_cutoff is None or not 0.0 <= entropy_cutoff <= 1.0:
        raise ValueError("Face entropy cutoff must be within [0, 1].")
    if isinstance(minimum_modalities, bool) or not isinstance(minimum_modalities, int):
        raise ValueError("Minimum modality count must be an integer.")
    if not 2 <= minimum_modalities <= len(MODALITIES):
        raise ValueError("Minimum modality count must be between 2 and 3.")

    current_time = time.monotonic() if now is None else _finite_number(now)
    if current_time is None:
        raise ValueError("Current monotonic time must be finite.")
    valid_snapshot = isinstance(snapshot, Mapping)
    monitoring_running = valid_snapshot and snapshot.get("status") == "running"

    modality_results: dict[str, ModalityScore] = {}
    exclusions: dict[str, str] = {}
    for modality in MODALITIES:
        branch_name = _BRANCH_NAMES[modality]
        branch_result = (
            snapshot.get(f"latest_{branch_name}_result")
            if valid_snapshot
            else None
        )
        timestamp = (
            _finite_number(snapshot.get(f"last_{branch_name}_success_timestamp"))
            if valid_snapshot
            else None
        )
        reason: str | None = None
        score: float | None = None
        evidence: tuple[str, ...] = ()
        if not valid_snapshot:
            reason = "Monitoring snapshot is malformed."
        elif not monitoring_running:
            reason = "Monitoring is not running; live branch evidence is excluded."
        elif branch_result is None:
            reason = "No branch result is available."
        elif timestamp is None:
            reason = "Branch success timestamp is missing or invalid."
        else:
            age = current_time - timestamp
            if age < 0:
                reason = "Branch success timestamp is in the future."
            elif age > freshness:
                reason = f"Branch evidence is stale ({age:.2f}s old)."
            else:
                if modality == "emotion":
                    score, evidence, reason = _emotion_score(
                        branch_result,
                        maximum_entropy=entropy_cutoff,
                    )
                elif modality == "behavior":
                    score, evidence, reason = _behavior_score(branch_result)
                else:
                    score, evidence, reason = _audio_score(branch_result)

        available = score is not None and reason is None
        if not available and reason:
            exclusions[modality] = reason
        modality_results[modality] = ModalityScore(
            score=score,
            available=available,
            weight=configured_weights[modality],
            contributed=False,
            contribution=None,
            evidence=evidence,
            reason=reason,
        )

    included = tuple(
        modality
        for modality in MODALITIES
        if modality_results[modality].available
        and modality_results[modality].weight > 0
    )
    overall: float | None = None
    if len(included) >= minimum_modalities:
        total_weight = sum(modality_results[name].weight for name in included)
        weighted_score = sum(
            modality_results[name].weight * float(modality_results[name].score)
            for name in included
        )
        overall = weighted_score / total_weight
        if not math.isfinite(overall):
            overall = None
        else:
            overall = min(max(overall, 0.0), 100.0)

    if overall is None:
        for modality in included:
            exclusions[modality] = (
                "Evidence is valid, but the minimum number of weighted modalities "
                f"({minimum_modalities}) was not met."
            )
        included_for_score: tuple[str, ...] = ()
        status = "Insufficient evidence"
    else:
        total_weight = sum(modality_results[name].weight for name in included)
        for modality in included:
            item = modality_results[modality]
            normalized_weight = item.weight / total_weight
            modality_results[modality] = ModalityScore(
                score=item.score,
                available=item.available,
                weight=item.weight,
                contributed=True,
                contribution=normalized_weight * float(item.score),
                evidence=item.evidence,
                reason=item.reason,
            )
        included_for_score = included
        status = "Available"

    return DistressScoreResult(
        score=overall,
        status=status,
        modalities=modality_results,
        included_modalities=included_for_score,
        excluded_reasons=exclusions,
    )


def score_alert_triggered(
    result: DistressScoreResult,
    threshold: Any = EXPERIMENTAL_DISTRESS_ALERT_THRESHOLD,
) -> bool | None:
    """Return the threshold decision, or None when no usable score exists."""
    threshold_value = _finite_number(threshold)
    if threshold_value is None or not 0.0 <= threshold_value <= 100.0:
        raise ValueError("Distress alert threshold must be finite and within [0, 100].")
    if _field(result, "status") != "Available":
        return None
    score = _finite_number(_field(result, "score"))
    if score is None or not 0.0 <= score <= 100.0:
        return None
    return score >= threshold_value


__all__ = [
    "DistressScoreResult",
    "INTERPRETATION_WARNING",
    "ModalityScore",
    "calculate_distress_score",
    "score_alert_triggered",
]
