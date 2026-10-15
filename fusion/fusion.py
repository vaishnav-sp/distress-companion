"""Phase 3 evidence aggregation for multimodal monitoring results.

This module does not classify distress. It defines a stable, explicit schema for
combining the current face, behavior, and audio branch outputs into a single
feature view that downstream fusion or training code can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class FusionEvidence:
    """A normalized multimodal feature bundle for downstream fusion and training."""

    feature_names: tuple[str, ...]
    feature_values: dict[str, float | None] = field(default_factory=dict)
    feature_validity: dict[str, bool] = field(default_factory=dict)
    unavailable_reasons: dict[str, str] = field(default_factory=dict)
    modality_status: dict[str, str] = field(default_factory=dict)
    modality_messages: dict[str, str] = field(default_factory=dict)
    modality_quality: dict[str, float | None] = field(default_factory=dict)
    vector: tuple[float | None, ...] = ()

    @property
    def has_data(self) -> bool:
        return bool(self.feature_names)


def _normalize_feature_result(prefix: str, result: Any) -> tuple[dict[str, float | None], dict[str, bool], dict[str, str], tuple[str, ...], dict[str, float | None]]:
    if result is None:
        return {}, {}, {}, (), {}

    values: dict[str, float | None] = {}
    validities: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    names: tuple[str, ...] = ()

    feature_values = getattr(result, "feature_values", None)
    if isinstance(feature_values, Mapping):
        values = {str(key): float(value) if value is not None else None for key, value in feature_values.items()}
        declared_validity = getattr(result, "feature_validity", {})
        validities = {}
        for name, value in values.items():
            declared_valid = bool(declared_validity.get(name, False))
            validities[name] = declared_valid and value is not None
            if declared_valid and value is None:
                reasons[name] = (
                    "Branch marked the feature valid but supplied no numeric value."
                )
        names = tuple(str(name) for name in (getattr(result, "feature_names", tuple(values.keys())) or tuple(values.keys())))
        branch_reasons = getattr(result, "unavailable_reasons", {}) or {}
        for name in names:
            if not validities.get(name, False) and name not in reasons:
                reasons[name] = str(branch_reasons.get(name, ""))
        quality_map: dict[str, float | None] = {}
        for name in names:
            if name in values:
                quality_map[name] = values[name]
        return values, validities, reasons, names, quality_map

    expression_probabilities = getattr(result, "expression_probabilities", None)
    if isinstance(expression_probabilities, Mapping):
        names_list = [
            str(name)
            for name in (
                getattr(result, "feature_vector_order", tuple(expression_probabilities.keys()))
                or tuple(expression_probabilities.keys())
            )
        ]
        for name in names_list:
            value = expression_probabilities.get(name)
            values[str(name)] = float(value) if value is not None else None
            valid = bool((getattr(result, "class_validity", {}) or {}).get(name, False))
            validities[str(name)] = valid
            if not valid:
                reasons[str(name)] = str((getattr(result, "abstention_reason", None) or "Probabilities unavailable in this modality."))
        supplementary = {
            "negative_affect_mean_angry_fear_sad": getattr(
                result, "negative_affect_feature", None
            ),
            "face_quality_score": getattr(result, "face_quality_score", None),
        }
        for name, value in supplementary.items():
            values[name] = float(value) if value is not None else None
            validities[name] = value is not None
            names_list.append(name)
            if value is None:
                reasons[name] = (
                    getattr(result, "abstention_reason", None)
                    or "This face feature is unavailable for the current result."
                )
        names = tuple(names_list)
        return values, validities, reasons, names, {str(name): values.get(name) for name in names}

    return values, validities, reasons, names, {}


def _modality_name(label: str) -> str:
    return label.lower().strip().replace(" ", "_")


def _face_modality_status(
    result: Any,
    validity: Mapping[str, bool],
) -> str:
    """Derive face status for FaceResult, which has no generic status field."""
    explicit_status = getattr(result, "status", None)
    if explicit_status:
        return str(explicit_status)

    class_validity = getattr(result, "class_validity", {}) or {}
    class_availability = getattr(result, "class_availability", {}) or {}
    supported_classes = [
        name
        for name in class_validity
        if class_availability.get(name, True)
    ]
    valid_classes = [
        name for name in supported_classes if bool(class_validity.get(name, False))
    ]
    if valid_classes:
        abstention_reason = getattr(result, "abstention_reason", None)
        if abstention_reason or len(valid_classes) < len(supported_classes):
            return "partial"
        return "ok"

    abstention_reason = getattr(result, "abstention_reason", None)
    if abstention_reason == "no_face":
        return "missing"
    if abstention_reason in {"model_unavailable", "face_detector_unavailable"}:
        return "unavailable"
    if abstention_reason in {"inference_error", "face_detection_error"}:
        return "error"
    if not any(validity.values()):
        return "unknown"
    return "partial"


def build_fusion_evidence(snapshot: Mapping[str, Any] | None) -> FusionEvidence:
    """Flatten the current monitoring snapshot into a single multimodal evidence bag."""
    if snapshot is None:
        return FusionEvidence(feature_names=(), feature_values={}, feature_validity={}, unavailable_reasons={}, modality_status={}, modality_messages={}, modality_quality={}, vector=())

    supported = {
        "face": snapshot.get("latest_face_result"),
        "behavior": snapshot.get("latest_behavior_result"),
        "audio": snapshot.get("latest_audio_result"),
    }

    combined_values: dict[str, float | None] = {}
    combined_validity: dict[str, bool] = {}
    combined_reasons: dict[str, str] = {}
    modality_status: dict[str, str] = {}
    modality_messages: dict[str, str] = {}
    modality_quality: dict[str, float | None] = {}
    ordered_names: list[str] = []

    for modality, result in supported.items():
        if result is None:
            modality_status[modality] = "waiting"
        elif modality == "face":
            _, face_validity, _, _, _ = _normalize_feature_result(
                modality,
                result,
            )
            modality_status[modality] = _face_modality_status(
                result,
                face_validity,
            )
        else:
            modality_status[modality] = getattr(result, "status", "unknown")
        modality_messages[modality] = (
            getattr(result, "status_message", "Waiting for first result.")
            if result is not None
            else "Waiting for first result."
        )

        values, validity, reasons, names, quality = _normalize_feature_result(modality, result)
        if not names:
            continue

        prefixed_names = [f"{modality}.{name}" for name in names]
        ordered_names.extend(prefixed_names)

        for name in names:
            feature_key = f"{modality}.{name}"
            combined_values[feature_key] = values.get(name)
            combined_validity[feature_key] = bool(validity.get(name, False))
            if feature_key not in combined_reasons and not combined_validity[feature_key]:
                combined_reasons[feature_key] = reasons.get(name, f"{modality} feature unavailable.")
            if feature_key not in modality_quality and name in quality:
                modality_quality[feature_key] = quality.get(name)

    ordered_names = list(dict.fromkeys(ordered_names))
    vector = tuple(
        combined_values.get(name) if combined_validity.get(name, False) else None
        for name in ordered_names
    )

    for name in ordered_names:
        if name not in combined_validity:
            combined_validity[name] = False
        if name not in combined_values:
            combined_values[name] = None
        if name not in combined_reasons and not combined_validity[name]:
            combined_reasons[name] = "Feature unavailable."

    return FusionEvidence(
        feature_names=tuple(ordered_names),
        feature_values=combined_values,
        feature_validity=combined_validity,
        unavailable_reasons=combined_reasons,
        modality_status=modality_status,
        modality_messages=modality_messages,
        modality_quality=modality_quality,
        vector=vector,
    )


__all__ = ["FusionEvidence", "build_fusion_evidence"]
