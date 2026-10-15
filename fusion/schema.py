"""Stable, versioned feature schema and record layer for multimodal evidence.

This module defines the canonical, deterministic feature layout used by the
future logistic-regression trainer and any downstream inference code. It is
intentionally limited to the shared schema and record layer: no distress
classification, no thresholds, no alerts, and no federated-learning logic.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from fusion.fusion import FusionEvidence

FEATURE_SCHEMA_VERSION = "phase3-feature-schema-v2"

# Canonical feature ordering and metadata for the current local multimodal
# evidence sources. The names intentionally preserve the modality prefix so that a
# single feature vector is ordered deterministically and does not blur the origin
# of each reading.
FEATURE_SPECIFICATIONS: dict[str, dict[str, str | None]] = {
    "face.expression_angry": {
        "modality": "face",
        "type": "float",
        "unit": None,
        "description": "Model-estimated probability for the angry class.",
    },
    "face.expression_fear": {
        "modality": "face",
        "type": "float",
        "unit": None,
        "description": "Model-estimated probability for the fear class.",
    },
    "face.expression_happy": {
        "modality": "face",
        "type": "float",
        "unit": None,
        "description": "Model-estimated probability for the happy class.",
    },
    "face.expression_neutral": {
        "modality": "face",
        "type": "float",
        "unit": None,
        "description": "Model-estimated probability for the neutral class.",
    },
    "face.expression_sad": {
        "modality": "face",
        "type": "float",
        "unit": None,
        "description": "Model-estimated probability for the sad class.",
    },
    "face.expression_surprise": {
        "modality": "face",
        "type": "float",
        "unit": None,
        "description": "Model-estimated probability for the surprise class.",
    },
    "face.negative_affect_mean_angry_fear_sad": {
        "modality": "face",
        "type": "float",
        "unit": None,
        "description": "Simple negative-affect summary across angry, fear, and sad estimates.",
    },
    "face.face_quality_score": {
        "modality": "face",
        "type": "float",
        "unit": None,
        "description": "Face-image usability score in the range [0, 1].",
    },
    "behavior.movement_energy": {
        "modality": "behavior",
        "type": "float",
        "unit": "shoulder_widths_per_second",
        "description": "Movement energy derived from displacement relative to shoulder width.",
    },
    "behavior.sudden_movement": {
        "modality": "behavior",
        "type": "float",
        "unit": None,
        "description": "Binary or bounded movement-change indicator for sudden motion.",
    },
    "behavior.movement_change_from_baseline": {
        "modality": "behavior",
        "type": "float",
        "unit": None,
        "description": "Movement change relative to its short rolling baseline.",
    },
    "behavior.shoulder_alignment": {
        "modality": "behavior",
        "type": "float",
        "unit": "pixels",
        "description": "Mean vertical shoulder alignment over the detected frame.",
    },
    "behavior.shoulder_asymmetry": {
        "modality": "behavior",
        "type": "float",
        "unit": "pixels",
        "description": "Absolute shoulder alignment asymmetry.",
    },
    "behavior.left_upper_arm_angle_deg": {
        "modality": "behavior",
        "type": "float",
        "unit": "degrees",
        "description": "Left upper-arm angle relative to the shoulder line.",
    },
    "behavior.right_upper_arm_angle_deg": {
        "modality": "behavior",
        "type": "float",
        "unit": "degrees",
        "description": "Right upper-arm angle relative to the shoulder line.",
    },
    "behavior.left_forearm_angle_deg": {
        "modality": "behavior",
        "type": "float",
        "unit": "degrees",
        "description": "Left forearm angle relative to the upper arm.",
    },
    "behavior.right_forearm_angle_deg": {
        "modality": "behavior",
        "type": "float",
        "unit": "degrees",
        "description": "Right forearm angle relative to the upper arm.",
    },
    "behavior.head_offset_x": {
        "modality": "behavior",
        "type": "float",
        "unit": "pixels",
        "description": "Horizontal displacement of the head relative to shoulder center.",
    },
    "behavior.head_offset_y": {
        "modality": "behavior",
        "type": "float",
        "unit": "pixels",
        "description": "Vertical displacement of the head relative to shoulder center.",
    },
    "behavior.head_orientation_deg": {
        "modality": "behavior",
        "type": "float",
        "unit": "degrees",
        "description": "Approximate head orientation angle.",
    },
    "behavior.repetition_change": {
        "modality": "behavior",
        "type": "float",
        "unit": None,
        "description": "Deprecated and unavailable; use wrist_repetition_rate_hz.",
    },
    "behavior.wrist_average_speed": {
        "modality": "behavior",
        "type": "float",
        "unit": "shoulder_widths_per_second",
        "description": "Time-weighted mean speed of visible wrists relative to shoulder center and width.",
    },
    "behavior.wrist_peak_speed": {
        "modality": "behavior",
        "type": "float",
        "unit": "shoulder_widths_per_second",
        "description": "95th-percentile visible-wrist speed after camera-jitter deadband filtering.",
    },
    "behavior.wrist_peak_acceleration": {
        "modality": "behavior",
        "type": "float",
        "unit": "shoulder_widths_per_second_squared",
        "description": "95th-percentile change in filtered wrist speed over time.",
    },
    "behavior.wrist_movement_amplitude": {
        "modality": "behavior",
        "type": "float",
        "unit": "shoulder_widths",
        "description": "Largest robust 10th-to-90th percentile wrist excursion in the rolling window.",
    },
    "behavior.wrist_repetition_rate_hz": {
        "modality": "behavior",
        "type": "float",
        "unit": "hz",
        "description": "Estimated repeated wrist oscillation rate; movement is not itself distress.",
    },
    "behavior.wrist_elevated_movement_proportion": {
        "modality": "behavior",
        "type": "float",
        "unit": None,
        "description": "Fraction of valid wrist-observation time above the configured speed level.",
    },
    "behavior.left_wrist_visibility": {
        "modality": "behavior",
        "type": "float",
        "unit": "confidence",
        "description": "Mean left-wrist landmark confidence during the rolling window.",
    },
    "behavior.right_wrist_visibility": {
        "modality": "behavior",
        "type": "float",
        "unit": "confidence",
        "description": "Mean right-wrist landmark confidence during the rolling window.",
    },
    "audio.rms_loudness": {
        "modality": "audio",
        "type": "float",
        "unit": "amplitude",
        "description": "Root-mean-square amplitude of the captured audio window.",
    },
    "audio.spectral_centroid_hz": {
        "modality": "audio",
        "type": "float",
        "unit": "hz",
        "description": "Weighted spectral centroid of the short audio window.",
    },
    "audio.spectral_bandwidth_hz": {
        "modality": "audio",
        "type": "float",
        "unit": "hz",
        "description": "Second moment bandwidth of the spectral distribution.",
    },
    "audio.spectral_flatness": {
        "modality": "audio",
        "type": "float",
        "unit": None,
        "description": "Spectral flatness value describing tonal vs noisy energy.",
    },
    "audio.onset_score": {
        "modality": "audio",
        "type": "float",
        "unit": None,
        "description": "Relative energy shift from the rolling audio baseline.",
    },
    "audio.baseline_relative_rms": {
        "modality": "audio",
        "type": "float",
        "unit": None,
        "description": "Current RMS relative to the short baseline mean.",
    },
    "audio.audio_contribution": {
        "modality": "audio",
        "type": "float",
        "unit": None,
        "description": "Normalized local audio contribution summary.",
    },
}

CANONICAL_MODES = ("face", "behavior", "audio")
FEATURE_ORDER = tuple(FEATURE_SPECIFICATIONS)


def feature_order() -> tuple[str, ...]:
    """Return the canonical feature order used for deterministic schema records."""
    return tuple(FEATURE_SPECIFICATIONS.keys())


def feature_definitions() -> dict[str, dict[str, str | None]]:
    """Return a stable, copy-safe schema definition map."""
    return {name: dict(spec) for name, spec in FEATURE_SPECIFICATIONS.items()}


@dataclass(frozen=True)
class FeatureSchema:
    """Versioned metadata describing the canonical multimodal evidence layout."""

    version: str = FEATURE_SCHEMA_VERSION
    modality_order: tuple[str, ...] = CANONICAL_MODES
    feature_order: tuple[str, ...] = field(default_factory=feature_order)
    feature_definitions: dict[str, dict[str, str | None]] = field(default_factory=feature_definitions)


@dataclass
class FeatureRecord:
    """A schema-compliant multimodal evidence record for training or evaluation.

    A value of None means the feature is missing or invalid for this record; a
    genuine numeric zero remains valid and must not be converted to null.
    """

    schema_version: str
    feature_names: tuple[str, ...]
    feature_values: dict[str, float | None]
    feature_validity: dict[str, bool]
    unavailable_reasons: dict[str, str]
    modality_status: dict[str, str]
    modality_messages: dict[str, str]
    scenario_label: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    record_kind: str = "live_monitoring"
    is_acted_synthetic: bool = False
    created_at_utc: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        _validate_record(self)

    @property
    def feature_vector(self) -> tuple[float | None, ...]:
        return tuple(self.feature_values.get(name) for name in self.feature_names)

    @property
    def valid_feature_count(self) -> int:
        return sum(1 for name in self.feature_names if self.feature_validity.get(name, False))

    @property
    def available_modalities(self) -> dict[str, bool]:
        return {
            modality: any(name.startswith(f"{modality}.") for name in self.feature_names)
            and any(
                self.feature_validity.get(name, False)
                for name in self.feature_names
                if name.startswith(f"{modality}.")
            )
            for modality in CANONICAL_MODES
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "feature_values": {key: value for key, value in self.feature_values.items()},
            "feature_validity": {key: bool(value) for key, value in self.feature_validity.items()},
            "unavailable_reasons": {key: str(value) for key, value in self.unavailable_reasons.items()},
            "modality_status": {key: str(value) for key, value in self.modality_status.items()},
            "modality_messages": {key: str(value) for key, value in self.modality_messages.items()},
            "scenario_label": self.scenario_label,
            "provenance": dict(self.provenance),
            "record_kind": self.record_kind,
            "is_acted_synthetic": bool(self.is_acted_synthetic),
            "created_at_utc": self.created_at_utc,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureRecord":
        if not isinstance(payload, Mapping):
            raise TypeError("FeatureRecord payload must be a mapping.")
        schema_version = payload.get("schema_version")
        if schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: expected {FEATURE_SCHEMA_VERSION!r}, got {schema_version!r}."
            )
        if "feature_names" not in payload:
            raise ValueError("Malformed feature record: missing 'feature_names'.")
        feature_names = tuple(payload["feature_names"])
        feature_values = payload.get("feature_values", {})
        feature_validity = payload.get("feature_validity", {})
        if not isinstance(feature_values, Mapping):
            raise ValueError("Malformed feature record: 'feature_values' must be a mapping.")
        if not isinstance(feature_validity, Mapping):
            raise ValueError("Malformed feature record: 'feature_validity' must be a mapping.")

        record = cls(
            schema_version=schema_version,
            feature_names=feature_names,
            feature_values={str(key): _coerce_numeric(value) for key, value in feature_values.items()},
            feature_validity={str(key): bool(value) for key, value in feature_validity.items()},
            unavailable_reasons={
                str(key): str(value) for key, value in payload.get("unavailable_reasons", {}).items()
            },
            modality_status={
                str(key): str(value) for key, value in payload.get("modality_status", {}).items()
            },
            modality_messages={
                str(key): str(value) for key, value in payload.get("modality_messages", {}).items()
            },
            scenario_label=payload.get("scenario_label"),
            provenance=dict(payload.get("provenance", {})),
            record_kind=str(payload.get("record_kind", "live_monitoring")),
            is_acted_synthetic=bool(payload.get("is_acted_synthetic", False)),
            created_at_utc=payload.get("created_at_utc"),
            notes=payload.get("notes"),
        )
        return record

    def to_json(self, *, sort_keys: bool = True) -> str:
        return json.dumps(self.to_dict(), sort_keys=sort_keys, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> "FeatureRecord":
        decoded = json.loads(payload)
        return cls.from_dict(decoded)


def _coerce_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
        raise ValueError("Feature values must be finite real numbers or None.")
    raise TypeError("Feature record values must be numeric or None.")


def _validate_record(record: FeatureRecord) -> None:
    if record.schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version {record.schema_version!r}; expected {FEATURE_SCHEMA_VERSION!r}."
        )
    if not record.feature_names:
        raise ValueError("FeatureRecord requires at least one feature name.")
    if len(set(record.feature_names)) != len(record.feature_names):
        raise ValueError("Feature names must be unique and deterministic.")
    if set(record.feature_names) - set(FEATURE_SPECIFICATIONS.keys()):
        unknown = sorted(set(record.feature_names) - set(FEATURE_SPECIFICATIONS.keys()))
        raise ValueError(f"Unknown feature name(s) in record: {unknown}")
    if tuple(record.feature_names) != tuple(dict.fromkeys(record.feature_names)):
        raise ValueError("Feature names must preserve their canonical order without duplicates.")
    if set(record.feature_values) != set(record.feature_names):
        missing = sorted(set(record.feature_names) - set(record.feature_values))
        extra = sorted(set(record.feature_values) - set(record.feature_names))
        raise ValueError(f"Feature values mismatch: missing={missing}, extra={extra}")
    for name in record.feature_names:
        if name not in record.feature_validity:
            raise ValueError(f"Feature validity missing for {name!r}.")
        valid = record.feature_validity.get(name, False)
        value = record.feature_values.get(name)
        if valid and value is None:
            raise ValueError(f"Feature {name!r} is marked valid but is None.")
        if (not valid) and value is not None:
            raise ValueError(
                f"Feature {name!r} is invalid and should use None rather than a fabricated numeric value."
            )
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"Feature {name!r} has a non-finite numeric value.")
    for modality in CANONICAL_MODES:
        if modality not in record.modality_status:
            record.modality_status.setdefault(modality, "unknown")
        if modality not in record.modality_messages:
            record.modality_messages.setdefault(modality, "No message recorded.")


def ensure_record_schema(record: Mapping[str, Any] | FeatureRecord) -> FeatureRecord:
    """Normalize a raw payload into a validated FeatureRecord."""
    if isinstance(record, FeatureRecord):
        return record
    return FeatureRecord.from_dict(record)


def evidence_to_feature_record(
    evidence: FusionEvidence,
    *,
    scenario_label: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    record_kind: str = "live_monitoring",
    is_acted_synthetic: bool = False,
    notes: str | None = None,
) -> FeatureRecord:
    """Convert a current FusionEvidence snapshot into a deterministic schema record."""
    feature_names = feature_order()
    ordered_values: dict[str, float | None] = {}
    ordered_validity: dict[str, bool] = {}
    unavailable: dict[str, str] = {}

    for name in feature_names:
        source_names = [name]
        if name.startswith("face.expression_"):
            source_names.append(f"face.{name.removeprefix('face.expression_')}")
        source_name = next(
            (candidate for candidate in source_names if candidate in evidence.feature_values),
            None,
        )
        is_valid = (
            source_name is not None
            and bool(evidence.feature_validity.get(source_name, False))
            and evidence.feature_values.get(source_name) is not None
        )
        ordered_values[name] = (
            float(evidence.feature_values[source_name])
            if is_valid and source_name is not None
            else None
        )
        ordered_validity[name] = is_valid
        if not is_valid:
            if source_name is not None:
                unavailable[name] = str(
                    evidence.unavailable_reasons.get(source_name, "")
                )
            else:
                modality = name.partition(".")[0]
                unexpected_names = sorted(
                    candidate
                    for candidate in evidence.feature_names
                    if candidate.startswith(f"{modality}.")
                    and candidate not in FEATURE_SPECIFICATIONS
                )
                unavailable[name] = (
                    f"No current evidence was published for {name}."
                    + (
                        f" Unrecognized {modality} feature name(s) were published: "
                        f"{', '.join(unexpected_names)}."
                        if unexpected_names
                        else ""
                    )
                )

    status = {modality: str(evidence.modality_status.get(modality, "waiting")) for modality in CANONICAL_MODES}
    messages = {modality: str(evidence.modality_messages.get(modality, "Waiting for first result.")) for modality in CANONICAL_MODES}

    return FeatureRecord(
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=feature_names,
        feature_values=ordered_values,
        feature_validity=ordered_validity,
        unavailable_reasons=unavailable,
        modality_status=status,
        modality_messages=messages,
        scenario_label=scenario_label,
        provenance=dict(provenance or {}),
        record_kind=record_kind,
        is_acted_synthetic=is_acted_synthetic,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        notes=notes,
    )


def build_fusion_diagnostic_snapshot(
    evidence: FusionEvidence,
    record: FeatureRecord,
    *,
    source_timestamps: Mapping[str, float | None],
    sufficient: bool,
    sufficiency_reason: str | None,
    runtime_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a plain-data diagnostic copy that is separate from live inference."""
    return {
        "construction_status": (
            "constructed_with_valid_multimodal_features"
            if sufficient
            else "constructed_but_insufficient"
        ),
        "constructed_at_utc": record.created_at_utc,
        "source_timestamps": dict(source_timestamps),
        "sufficiency_reason": sufficiency_reason,
        "feature_names": tuple(evidence.feature_names),
        "feature_values": dict(evidence.feature_values),
        "feature_validity": dict(evidence.feature_validity),
        "unavailable_reasons": dict(evidence.unavailable_reasons),
        "modality_status": dict(evidence.modality_status),
        "modality_messages": dict(evidence.modality_messages),
        "modality_quality": dict(evidence.modality_quality),
        "feature_record": record.to_dict(),
        "valid_feature_count": record.valid_feature_count,
        "available_modalities": dict(record.available_modalities),
        "runtime_diagnostics": dict(runtime_diagnostics or {}),
    }


def generate_acted_synthetic_records(
    label: str,
    *,
    count: int = 4,
    seed: int = 42,
    scenario_description: str | None = None,
) -> list[FeatureRecord]:
    """Create deterministic acted/synthetic records for testing and future trainer work.

    These are intentionally clearly marked as synthetic/acted and are not claimed to
    be actual clinical observations of autistic students. They are only for schema
    validation and future model-development scaffolding.
    """
    if count < 1:
        raise ValueError("count must be at least 1.")
    if not label:
        raise ValueError("A scenario label is required.")

    import random

    rng = random.Random(seed)
    prototype = {
        "face.expression_angry": 0.15,
        "face.expression_fear": 0.10,
        "face.expression_happy": 0.20,
        "face.expression_neutral": 0.35,
        "face.expression_sad": 0.12,
        "face.expression_surprise": 0.08,
        "face.negative_affect_mean_angry_fear_sad": 0.12,
        "face.face_quality_score": 0.92,
        "behavior.movement_energy": 0.60,
        "behavior.sudden_movement": 0.10,
        "behavior.movement_change_from_baseline": 0.15,
        "behavior.shoulder_alignment": 0.0,
        "behavior.shoulder_asymmetry": 0.05,
        "behavior.left_upper_arm_angle_deg": 18.0,
        "behavior.right_upper_arm_angle_deg": 20.0,
        "behavior.left_forearm_angle_deg": 35.0,
        "behavior.right_forearm_angle_deg": 34.0,
        "behavior.head_offset_x": 0.0,
        "behavior.head_offset_y": 0.0,
        "behavior.head_orientation_deg": 0.0,
        "behavior.repetition_change": None,
        "audio.rms_loudness": 0.45,
        "audio.spectral_centroid_hz": 520.0,
        "audio.spectral_bandwidth_hz": 180.0,
        "audio.spectral_flatness": 0.6,
        "audio.onset_score": 0.12,
        "audio.baseline_relative_rms": 0.0,
        "audio.audio_contribution": 0.28,
    }

    records: list[FeatureRecord] = []
    for index in range(count):
        values: dict[str, float | None] = {}
        validity: dict[str, bool] = {}
        reasons: dict[str, str] = {}
        for name in feature_order():
            base = prototype.get(name)
            if base is None:
                values[name] = None
                validity[name] = False
                reasons[name] = "Unavailable in the current phase."
                continue
            jitter = (rng.random() - 0.5) * 0.12 * max(abs(float(base)), 0.01)
            values[name] = float(base + jitter)
            validity[name] = True
        records.append(
            FeatureRecord(
                schema_version=FEATURE_SCHEMA_VERSION,
                feature_names=feature_order(),
                feature_values=values,
                feature_validity=validity,
                unavailable_reasons=reasons,
                modality_status={
                    "face": "ok",
                    "behavior": "ok",
                    "audio": "ok",
                },
                modality_messages={
                    "face": "Synthetic validation scenario. Not a real observation.",
                    "behavior": "Synthetic validation scenario. Not a real observation.",
                    "audio": "Synthetic validation scenario. Not a real observation.",
                },
                scenario_label=f"acted_synthetic:{label}:{index}",
                provenance={
                    "generator": "acted_synthetic_dataset",
                    "seed": seed,
                    "scenario_label": label,
                    "count": count,
                    "scenario_description": scenario_description or "Documented toy scenario used for schema validation.",
                },
                record_kind="synthetic_acted",
                is_acted_synthetic=True,
                created_at_utc=datetime.now(timezone.utc).isoformat(),
                notes="This record is synthetic and not a genuine observation of autistic students.",
            )
        )
    return records


__all__ = [
    "CANONICAL_MODES",
    "FEATURE_ORDER",
    "FEATURE_SCHEMA_VERSION",
    "FEATURE_SPECIFICATIONS",
    "FeatureRecord",
    "FeatureSchema",
    "generate_acted_synthetic_records",
    "evidence_to_feature_record",
    "ensure_record_schema",
    "feature_definitions",
    "feature_order",
    "FeatureRecord",
    "FusionEvidence",
]
