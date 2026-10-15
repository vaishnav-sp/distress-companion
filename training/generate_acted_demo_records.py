"""Create deterministic, explicitly labeled synthetic records for pipeline demos."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

from fusion.schema import FEATURE_SCHEMA_VERSION, FeatureRecord, feature_order

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "acted" / "phase3_demo_records.jsonl"
LABEL_PROFILES = {
    "potential_distress": {
        "expression_probabilities": (0.48, 0.12, 0.04, 0.10, 0.20, 0.06),
        "behavior": {
            "movement_energy": (0.24, 0.08),
            "sudden_movement": (0.08, 0.04),
            "movement_change_from_baseline": (0.10, 0.05),
            "wrist_average_speed": (0.16, 0.06),
            "wrist_peak_speed": (0.30, 0.09),
            "wrist_peak_acceleration": (0.45, 0.15),
            "wrist_movement_amplitude": (0.14, 0.05),
            "wrist_repetition_rate_hz": (0.08, 0.05),
            "wrist_elevated_movement_proportion": (0.05, 0.04),
            "left_wrist_visibility": (0.91, 0.05),
            "right_wrist_visibility": (0.90, 0.05),
            "shoulder_alignment": (1.0, 1.5),
            "shoulder_asymmetry": (2.0, 1.0),
            "left_upper_arm_angle_deg": (25.0, 5.0),
            "right_upper_arm_angle_deg": (26.0, 5.0),
            "left_forearm_angle_deg": (38.0, 6.0),
            "right_forearm_angle_deg": (39.0, 6.0),
            "head_offset_x": (2.0, 1.5),
            "head_offset_y": (3.0, 2.0),
            "head_orientation_deg": (3.0, 2.0),
        },
        "audio": {
            "rms_loudness": (0.32, 0.06),
            "spectral_centroid_hz": (1250.0, 180.0),
            "spectral_bandwidth_hz": (700.0, 120.0),
            "spectral_flatness": (0.68, 0.08),
            "onset_score": (0.72, 0.12),
            "baseline_relative_rms": (0.75, 0.14),
            "audio_contribution": (0.72, 0.10),
        },
    },
    "no_sufficient_distress_evidence": {
        "expression_probabilities": (0.04, 0.03, 0.48, 0.34, 0.05, 0.06),
        "behavior": {
            "movement_energy": (0.24, 0.08),
            "sudden_movement": (0.08, 0.04),
            "movement_change_from_baseline": (0.10, 0.05),
            "wrist_average_speed": (0.16, 0.06),
            "wrist_peak_speed": (0.30, 0.09),
            "wrist_peak_acceleration": (0.45, 0.15),
            "wrist_movement_amplitude": (0.14, 0.05),
            "wrist_repetition_rate_hz": (0.08, 0.05),
            "wrist_elevated_movement_proportion": (0.05, 0.04),
            "left_wrist_visibility": (0.91, 0.05),
            "right_wrist_visibility": (0.90, 0.05),
            "shoulder_alignment": (1.0, 1.5),
            "shoulder_asymmetry": (2.0, 1.0),
            "left_upper_arm_angle_deg": (25.0, 5.0),
            "right_upper_arm_angle_deg": (26.0, 5.0),
            "left_forearm_angle_deg": (38.0, 6.0),
            "right_forearm_angle_deg": (39.0, 6.0),
            "head_offset_x": (2.0, 1.5),
            "head_offset_y": (3.0, 2.0),
            "head_orientation_deg": (3.0, 2.0),
        },
        "audio": {
            "rms_loudness": (0.06, 0.02),
            "spectral_centroid_hz": (520.0, 90.0),
            "spectral_bandwidth_hz": (280.0, 60.0),
            "spectral_flatness": (0.34, 0.08),
            "onset_score": (0.10, 0.05),
            "baseline_relative_rms": (0.08, 0.04),
            "audio_contribution": (0.18, 0.06),
        },
    },
}
FACE_FEATURES = (
    "face.expression_angry",
    "face.expression_fear",
    "face.expression_happy",
    "face.expression_neutral",
    "face.expression_sad",
    "face.expression_surprise",
)


def _make_record(
    label: str,
    group_index: int,
    sample_index: int,
    rng: random.Random,
    seed: int,
) -> dict[str, object]:
    profile = LABEL_PROFILES[label]
    values: dict[str, float | None] = {name: None for name in feature_order()}
    validity = {name: False for name in feature_order()}

    raw_face = [
        max(0.001, probability + rng.uniform(-0.012, 0.012))
        for probability in profile["expression_probabilities"]
    ]
    total = sum(raw_face)
    for name, value in zip(FACE_FEATURES, raw_face):
        values[name] = value / total
        validity[name] = True
    values["face.negative_affect_mean_angry_fear_sad"] = (
        float(values["face.expression_angry"])
        + float(values["face.expression_fear"])
        + float(values["face.expression_sad"])
    ) / 3
    values["face.face_quality_score"] = rng.uniform(0.82, 0.98)
    validity["face.negative_affect_mean_angry_fear_sad"] = True
    validity["face.face_quality_score"] = True

    for name, (mean, spread) in profile["behavior"].items():
        key = f"behavior.{name}"
        values[key] = max(0.0, rng.gauss(mean, spread))
        validity[key] = True
    for name, (mean, spread) in profile["audio"].items():
        key = f"audio.{name}"
        values[key] = max(0.0, rng.gauss(mean, spread))
        validity[key] = True
    # The current behavior branch does not provide this measurement.
    values["behavior.repetition_change"] = None
    validity["behavior.repetition_change"] = False
    unavailable = {
        name: "Not measured in this phase."
        for name, is_valid in validity.items()
        if not is_valid
    }

    group = f"acted_demo_{label}_{group_index:02d}"
    record = FeatureRecord(
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=feature_order(),
        feature_values=values,
        feature_validity=validity,
        unavailable_reasons=unavailable,
        modality_status={"face": "synthetic", "behavior": "synthetic", "audio": "synthetic"},
        modality_messages={
            modality: "Deterministic synthetic demonstration data; not a real observation."
            for modality in ("face", "behavior", "audio")
        },
        scenario_label=f"{group}:sample_{sample_index:02d}",
        provenance={
            "scenario_group": group,
            "generator": "training.generate_acted_demo_records",
            "seed": seed,
            "source": "synthetic_profile",
            "label_rationale": (
                "Author-assigned pipeline-demo label for the synthetic face/audio "
                "profile; movement values intentionally overlap the baseline profile. "
                "This is not empirically justified distress evidence."
                if label == "potential_distress"
                else "Author-assigned synthetic baseline label for pipeline testing; "
                "not a clinical or behavioral ground-truth judgment."
            ),
            "description": (
                "Human-authored toy feature profile for exercising the training "
                "pipeline only; not measured from a person or recording."
            ),
        },
        record_kind="synthetic_acted",
        is_acted_synthetic=True,
        created_at_utc=None,
        notes="Synthetic demonstration only. Not a real observation and not validated distress evidence.",
    )
    payload = record.to_dict()
    payload["label"] = label
    return payload


def generate_records(
    *,
    seed: int = 20261014,
    groups_per_class: int = 8,
    records_per_group: int = 2,
) -> list[dict[str, object]]:
    """Return reproducible schema-valid examples with explicit synthetic labels."""
    if groups_per_class < 2:
        raise ValueError("At least two independent groups per class are required.")
    if records_per_group < 1:
        raise ValueError("records_per_group must be positive.")
    rng = random.Random(seed)
    records = []
    for label in LABEL_PROFILES:
        for group_index in range(groups_per_class):
            for sample_index in range(records_per_group):
                records.append(_make_record(label, group_index, sample_index, rng, seed))
    return records


MOVEMENT_DEMO_SCENARIOS = {
    "quiet_baseline": {
        "label": "no_sufficient_distress_evidence",
        "description": "Quiet baseline with minimal movement.",
        "label_rationale": "Synthetic baseline case; no positive demonstration cue is specified.",
        "behavior": (0.02, 0.02, 0.00, 0.04, 0.06, 0.08, 0.01, 0.00, 0.00),
        "audio_profile": "no_sufficient_distress_evidence",
    },
    "ordinary_hand_gestures": {
        "label": "no_sufficient_distress_evidence",
        "description": "Ordinary intermittent hand gestures.",
        "label_rationale": "Ordinary gestures alone are not assigned a positive label in this synthetic case.",
        "behavior": (0.16, 0.13, 0.03, 0.28, 0.35, 0.48, 0.17, 0.00, 0.06),
        "audio_profile": "no_sufficient_distress_evidence",
    },
    "sustained_repetitive_hand_movement": {
        "label": "no_sufficient_distress_evidence",
        "description": "Sustained repetitive hand movement without distress label.",
        "label_rationale": "Repetitive movement alone is explicitly not assigned a positive label.",
        "behavior": (0.36, 0.31, 0.48, 0.54, 0.48, 0.58, 0.24, 0.92, 0.40),
        "audio_profile": "no_sufficient_distress_evidence",
    },
    "sustained_vigorous_hand_arm_movement": {
        "label": "no_sufficient_distress_evidence",
        "description": "Sustained vigorous arm movement alone is not labeled distress.",
        "label_rationale": "Vigorous movement alone is explicitly not assigned a positive label.",
        "behavior": (0.82, 0.74, 0.68, 1.35, 2.10, 2.80, 0.72, 0.65, 0.78),
        "audio_profile": "no_sufficient_distress_evidence",
    },
    "sudden_isolated_movement": {
        "label": "no_sufficient_distress_evidence",
        "description": "One sudden gesture, not sustained repetition.",
        "label_rationale": "An isolated gesture alone is not assigned a positive label.",
        "behavior": (0.25, 0.14, 0.12, 1.10, 3.20, 0.18, 0.15, 0.00, 0.09),
        "audio_profile": "no_sufficient_distress_evidence",
    },
    "high_energy_audio_little_movement": {
        "label": "potential_distress",
        "description": "High-energy audio with little body movement.",
        "label_rationale": (
            "The synthetic scenario author designates the high-energy audio profile "
            "as the positive pipeline-test cue; this is not a validated distress label."
        ),
        "behavior": (0.03, 0.02, 0.01, 0.05, 0.07, 0.08, 0.02, 0.00, 0.00),
        "audio_profile": "potential_distress",
    },
    "combined_movement_and_high_energy_audio": {
        "label": "potential_distress",
        "description": "Combined movement and high-energy audio.",
        "label_rationale": (
            "The synthetic scenario author designates high-energy audio as the "
            "positive pipeline-test cue; movement is context only and is not by "
            "itself the reason for the label. This is not validated distress evidence."
        ),
        "behavior": (0.80, 0.72, 0.70, 1.30, 2.00, 2.65, 0.68, 0.62, 0.75),
        "audio_profile": "potential_distress",
    },
}


def generate_movement_demo_scenarios(
    *,
    seed: int = 20261015,
) -> list[dict[str, object]]:
    """Generate reproducible toy scenario records, not evidence of model validity."""
    rng = random.Random(seed)
    generated: list[dict[str, object]] = []
    behavior_feature_names = (
        "movement_energy",
        "wrist_average_speed",
        "wrist_repetition_rate_hz",
        "wrist_peak_speed",
        "wrist_peak_acceleration",
        "wrist_movement_amplitude",
        "movement_change_from_baseline",
        "wrist_elevated_movement_proportion",
        "sudden_movement",
    )
    baseline_expression = LABEL_PROFILES[
        "no_sufficient_distress_evidence"
    ]["expression_probabilities"]
    for index, (scenario_name, scenario) in enumerate(MOVEMENT_DEMO_SCENARIOS.items()):
        label = str(scenario["label"])
        payload = _make_record(label, 100 + index, 0, rng, seed)
        values = payload["feature_values"]
        validity = payload["feature_validity"]
        raw_face = [
            max(0.001, probability + rng.uniform(-0.012, 0.012))
            for probability in baseline_expression
        ]
        face_total = sum(raw_face)
        for name, value in zip(FACE_FEATURES, raw_face):
            values[name] = value / face_total
            validity[name] = True
        values["face.negative_affect_mean_angry_fear_sad"] = (
            values["face.expression_angry"]
            + values["face.expression_fear"]
            + values["face.expression_sad"]
        ) / 3
        values["face.face_quality_score"] = 0.9
        validity["face.negative_affect_mean_angry_fear_sad"] = True
        validity["face.face_quality_score"] = True
        for name, value in zip(behavior_feature_names, scenario["behavior"]):
            key = f"behavior.{name}"
            values[key] = float(value)
            validity[key] = True
        for side in ("left", "right"):
            key = f"behavior.{side}_wrist_visibility"
            values[key] = 0.94
            validity[key] = True
        audio_profile = LABEL_PROFILES[str(scenario["audio_profile"])]["audio"]
        for name, (mean, _spread) in audio_profile.items():
            key = f"audio.{name}"
            values[key] = float(mean)
            validity[key] = True

        group = f"acted_movement_{scenario_name}"
        payload["scenario_label"] = f"{group}:sample_00"
        payload["provenance"] = {
            "scenario_group": group,
            "generator": "training.generate_acted_demo_records",
            "seed": seed,
            "scenario_name": scenario_name,
            "description": str(scenario["description"]),
            "label_rationale": str(scenario["label_rationale"]),
            "source": "synthetic_acted_scenario",
            "limitation": (
                "Human-authored toy values for pipeline demonstration only; "
                "not measured behavior or distress evidence."
            ),
        }
        payload["notes"] = (
            "Synthetic acted scenario only. Repetitive movement alone is not "
            "labeled as distress; these values do not establish model accuracy."
        )
        generated.append(payload)
    return generated


def write_records(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            output.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write deterministic, explicitly labeled synthetic FeatureRecord JSONL "
            "for a pipeline demonstration only."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--groups-per-class", type=int, default=8)
    parser.add_argument("--records-per-group", type=int, default=2)
    parser.add_argument(
        "--include-movement-scenarios",
        action="store_true",
        help="Append seven labeled toy movement/audio scenarios to the generated records.",
    )
    args = parser.parse_args(argv)

    records = generate_records(
        seed=args.seed,
        groups_per_class=args.groups_per_class,
        records_per_group=args.records_per_group,
    )
    if args.include_movement_scenarios:
        records.extend(generate_movement_demo_scenarios(seed=args.seed))
    write_records(args.output, records)
    print(f"Wrote {len(records)} explicitly labeled synthetic demonstration records to {args.output}")
    print("These records are not real observations or validated distress evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
