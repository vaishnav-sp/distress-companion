"""Isolated, versioned personalization experiments for the local dashboard."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from feedback.store import FeedbackStore
from federated.virtual_students import (
    SyntheticTrainingExample,
    VirtualStudentConfig,
    create_synthetic_clients,
)
from training.distress_logistic import TARGET_LABEL_MAPPING, TARGET_LABELS
from training.fusion_head import (
    FUSION_HEAD_MODEL_VERSION,
    fine_tune_fusion_head,
    initialize_fusion_head,
    save_fusion_head,
)
from training.reviewed_feedback import (
    ReviewedTrainingExample,
    load_reviewed_feedback,
)

MINIMUM_EXAMPLES_PER_CLASS = 4


@dataclass(frozen=True)
class PersonalizationResult:
    data_mode: str
    sample_count: int
    class_counts: dict[str, int]
    training_sample_count: int
    held_out_sample_count: int
    model_version: str
    artifact_path: Path
    run_metadata_path: Path
    before_metrics: dict[str, float]
    after_metrics: dict[str, float]


def _metrics(head: Any, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    predicted = head.pipeline.predict(features)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
    }


def _split_reviewed_examples(
    examples: Sequence[ReviewedTrainingExample],
) -> tuple[list[ReviewedTrainingExample], list[ReviewedTrainingExample]]:
    grouped = {
        label: [example for example in examples if example.label == label]
        for label in TARGET_LABELS
    }
    insufficient = {
        label: len(rows)
        for label, rows in grouped.items()
        if len(rows) < MINIMUM_EXAMPLES_PER_CLASS
    }
    if insufficient:
        available = ", ".join(
            f"{label}: {count}" for label, count in insufficient.items()
        )
        raise ValueError(
            "Insufficient eligible class diversity for a held-out evaluation. "
            f"Need at least {MINIMUM_EXAMPLES_PER_CLASS} per label; available: {available}."
        )

    parents = list(range(len(examples)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        parents[find(left)] = find(right)

    first_for_context: dict[str, int] = {}
    first_for_window: dict[tuple[Any, ...], int] = {}
    for index, example in enumerate(examples):
        provenance = example.feedback_provenance
        context = provenance["evidence_context_reference"]
        window = (
            provenance["event_window_start_utc"],
            provenance["event_window_end_utc"],
            tuple(sorted(provenance["modality_timestamps_utc"].items())),
        )
        if context in first_for_context:
            union(index, first_for_context[context])
        else:
            first_for_context[context] = index
        if window in first_for_window:
            union(index, first_for_window[window])
        else:
            first_for_window[window] = index

    grouped_by_source: dict[int, list[ReviewedTrainingExample]] = {}
    for index, example in enumerate(examples):
        grouped_by_source.setdefault(find(index), []).append(example)
    independent_groups: dict[str, list[list[ReviewedTrainingExample]]] = {
        label: [] for label in TARGET_LABELS
    }
    for group in grouped_by_source.values():
        labels = {example.label for example in group}
        if len(labels) != 1:
            raise ValueError(
                "Conflicting labels refer to the same reviewed evidence context or event window."
            )
        independent_groups[next(iter(labels))].append(group)

    rng = np.random.default_rng(42)
    training: list[ReviewedTrainingExample] = []
    held_out: list[ReviewedTrainingExample] = []
    for label in TARGET_LABELS:
        groups = independent_groups[label]
        if len(groups) < 2:
            raise ValueError(
                "Insufficient independent evidence contexts for a held-out split; "
                f"label {label!r} needs at least two distinct reviewed events."
            )
        rng.shuffle(groups)
        held_out_count = min(
            len(groups) - 1,
            max(1, round(len(groups) * 0.25)),
        )
        for group in groups[:held_out_count]:
            held_out.extend(group)
        for group in groups[held_out_count:]:
            training.extend(group)
    rng.shuffle(training)
    rng.shuffle(held_out)
    return training, held_out


def _reviewed_features(
    examples: Sequence[ReviewedTrainingExample],
) -> tuple[np.ndarray, np.ndarray]:
    from training.distress_logistic import _record_to_feature_vector

    features = np.vstack(
        [_record_to_feature_vector(example.record)[0] for example in examples]
    )
    labels = np.asarray(
        [TARGET_LABEL_MAPPING[example.label] for example in examples],
        dtype=int,
    )
    return features, labels


def reviewed_training_block_reason(
    examples: Sequence[ReviewedTrainingExample],
) -> str | None:
    """Explain whether approved examples can support both training and held-out tests."""
    if not examples:
        return "No eligible reviewed examples are available."
    try:
        _split_reviewed_examples(examples)
    except ValueError as exc:
        return str(exc)
    return None


def _write_experiment(
    *,
    output_directory: str | Path,
    data_mode: str,
    sample_count: int,
    class_counts: dict[str, int],
    training_sample_count: int,
    held_out_sample_count: int,
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
    trained_head: Any,
) -> PersonalizationResult:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    model_prefix = (
        "fusion-head-reviewed"
        if data_mode == "reviewed"
        else "fusion-head-synthetic-demo"
    )
    while True:
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        artifact_path = output_directory / f"{model_prefix}-{run_id}.joblib"
        run_metadata_path = output_directory / f"{model_prefix}-{run_id}.run.json"
        if not any(
            candidate.exists()
            for candidate in (
                artifact_path,
                artifact_path.with_suffix(".metadata.json"),
                run_metadata_path,
            )
        ):
            break
    save_fusion_head(trained_head, artifact_path)
    run_metadata = {
        "run_metadata_version": "local-personalization-run-v1",
        "data_mode": data_mode,
        "sample_count": sample_count,
        "class_counts": class_counts,
        "training_sample_count": training_sample_count,
        "held_out_sample_count": held_out_sample_count,
        "model_version": FUSION_HEAD_MODEL_VERSION,
        "artifact_file": artifact_path.name,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "live_model_activated": False,
    }
    with run_metadata_path.open("x", encoding="utf-8") as metadata_file:
        json.dump(run_metadata, metadata_file, indent=2, sort_keys=True, allow_nan=False)
        metadata_file.write("\n")
    return PersonalizationResult(
        data_mode=data_mode,
        sample_count=sample_count,
        class_counts=class_counts,
        training_sample_count=training_sample_count,
        held_out_sample_count=held_out_sample_count,
        model_version=FUSION_HEAD_MODEL_VERSION,
        artifact_path=artifact_path,
        run_metadata_path=run_metadata_path,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
    )


def run_reviewed_personalization(
    feedback_store: FeedbackStore,
    manifest_path: str | Path,
    output_directory: str | Path,
) -> PersonalizationResult:
    """Fine-tune and evaluate only eligible records from the explicit v2 manifest."""
    if not Path(manifest_path).is_file():
        raise ValueError("No v2 review manifest exists; submit an explicit review first.")
    loaded = load_reviewed_feedback(feedback_store, manifest_path)
    examples = list(loaded.examples)
    if not examples:
        raise ValueError(
            "No eligible reviewed examples are available. Feedback actions alone "
            "are not training labels."
        )
    class_counts = {
        label: sum(example.label == label for example in examples)
        for label in TARGET_LABELS
    }
    training_examples, held_out_examples = _split_reviewed_examples(examples)
    held_out_features, held_out_labels = _reviewed_features(held_out_examples)
    baseline = initialize_fusion_head()
    before_metrics = _metrics(baseline, held_out_features, held_out_labels)
    trained = fine_tune_fusion_head(baseline, training_examples)
    after_metrics = _metrics(trained, held_out_features, held_out_labels)
    return _write_experiment(
        output_directory=output_directory,
        data_mode="reviewed",
        sample_count=len(examples),
        class_counts=class_counts,
        training_sample_count=len(training_examples),
        held_out_sample_count=len(held_out_examples),
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        trained_head=trained,
    )


def run_synthetic_personalization_demo(
    output_directory: str | Path,
    *,
    seed: int = 20261018,
) -> PersonalizationResult:
    """Run a deterministic synthetic-only experiment in a separate artifact namespace."""
    partitions = create_synthetic_clients(
        VirtualStudentConfig(seed=seed),
        samples_per_client=40,
        evaluation_samples_per_class=20,
    )
    training_examples = [
        example
        for client in partitions.clients
        for example in client.load_training_partition()
        if isinstance(example, SyntheticTrainingExample)
    ]
    held_out_examples = [
        example
        for example in partitions.evaluation_examples
        if isinstance(example, SyntheticTrainingExample)
    ]
    if not training_examples or not held_out_examples:
        raise ValueError("The deterministic synthetic demo produced no usable examples.")

    training_features = np.asarray(
        [example.features for example in training_examples], dtype=float
    )
    training_labels = np.asarray(
        [TARGET_LABEL_MAPPING[example.label] for example in training_examples],
        dtype=int,
    )
    held_out_features = np.asarray(
        [example.features for example in held_out_examples], dtype=float
    )
    held_out_labels = np.asarray(
        [TARGET_LABEL_MAPPING[example.label] for example in held_out_examples],
        dtype=int,
    )
    class_counts = {
        label: sum(
            example.label == label
            for example in (*training_examples, *held_out_examples)
        )
        for label in TARGET_LABELS
    }
    baseline = initialize_fusion_head()
    before_metrics = _metrics(baseline, held_out_features, held_out_labels)
    trained = initialize_fusion_head()
    trained.pipeline.fit(training_features, training_labels)
    after_metrics = _metrics(trained, held_out_features, held_out_labels)
    return _write_experiment(
        output_directory=Path(output_directory) / "synthetic_demo",
        data_mode="DEMO/SYNTHETIC ONLY",
        sample_count=len(training_examples) + len(held_out_examples),
        class_counts=class_counts,
        training_sample_count=len(training_examples),
        held_out_sample_count=len(held_out_examples),
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        trained_head=trained,
    )
