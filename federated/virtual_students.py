"""Deterministic local-only virtual-student client partitions."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from feedback.store import FeedbackStore
from fusion.schema import feature_order
from training.distress_logistic import TARGET_LABELS
from training.fusion_head import (
    FUSION_HEAD_INPUT_WIDTH,
    initialize_fusion_head,
)
from training.reviewed_feedback import (
    EligibilityRejection,
    ReviewedTrainingExample,
    load_reviewed_feedback,
)

REVIEWED_DATA_MODE = "reviewed"
SYNTHETIC_TEST_DATA_MODE = "synthetic_test"
MIN_CLIENT_COUNT = 3
MAX_CLIENT_COUNT = 5


@dataclass(frozen=True)
class VirtualStudentConfig:
    """Stable client count and random seed for a simulated cohort."""

    client_count: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.client_count, bool) or not isinstance(self.client_count, int):
            raise TypeError("client_count must be an integer.")
        if not MIN_CLIENT_COUNT <= self.client_count <= MAX_CLIENT_COUNT:
            raise ValueError(
                f"client_count must be between {MIN_CLIENT_COUNT} and {MAX_CLIENT_COUNT}."
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")


@dataclass(frozen=True)
class SyntheticTrainingExample:
    """Explicitly labelled numerical infrastructure-test data, never reviewed data."""

    sample_id: str
    label: str
    features: tuple[float, ...]
    data_mode: str = SYNTHETIC_TEST_DATA_MODE


TrainingExample = ReviewedTrainingExample | SyntheticTrainingExample


@dataclass(frozen=True)
class ClientDataQuality:
    sample_count: int
    class_counts: Mapping[str, int]
    feature_validity_rate: float
    data_mode: str
    synthetic_test_data: bool


@dataclass(frozen=True)
class VirtualStudentClient:
    """Client-facing local partition and shared fusion-head contract."""

    client_id: str
    data_mode: str
    _training_examples: tuple[TrainingExample, ...]
    _fusion_head_contract: Mapping[str, object]

    def load_training_partition(self) -> tuple[TrainingExample, ...]:
        """Return this client's training examples only."""
        return self._training_examples

    @property
    def fusion_head_contract(self) -> dict[str, object]:
        """Return a copy of the Milestone 1 model contract."""
        return copy.deepcopy(dict(self._fusion_head_contract))

    def data_quality(self) -> ClientDataQuality:
        counts = {label: 0 for label in TARGET_LABELS}
        valid_features = 0
        total_features = 0
        for example in self._training_examples:
            counts[example.label] += 1
            if isinstance(example, ReviewedTrainingExample):
                validity = example.record.feature_validity
                valid_features += sum(bool(validity[name]) for name in feature_order())
                total_features += len(feature_order())
            else:
                valid_features += sum(flag == 1.0 for flag in example.features[1::2])
                total_features += len(example.features[1::2])
        rate = valid_features / total_features if total_features else 0.0
        return ClientDataQuality(
            sample_count=len(self._training_examples),
            class_counts=counts,
            feature_validity_rate=rate,
            data_mode=self.data_mode,
            synthetic_test_data=self.data_mode == SYNTHETIC_TEST_DATA_MODE,
        )


@dataclass(frozen=True)
class VirtualStudentPartitions:
    clients: tuple[VirtualStudentClient, ...]
    evaluation_examples: tuple[TrainingExample, ...]
    data_mode: str
    eligibility_rejections: tuple[EligibilityRejection, ...] = ()


def _new_client(client_index: int, mode: str, examples: list[TrainingExample]) -> VirtualStudentClient:
    contract = initialize_fusion_head().metadata
    return VirtualStudentClient(
        client_id=f"virtual-student-{client_index + 1:02d}",
        data_mode=mode,
        _training_examples=tuple(examples),
        _fusion_head_contract=contract,
    )


def _synthetic_feature_vector(rng: np.random.Generator, location: float) -> tuple[float, ...]:
    feature_values = rng.normal(
        loc=location,
        scale=0.2,
        size=FUSION_HEAD_INPUT_WIDTH // 2,
    )
    vector = np.empty(FUSION_HEAD_INPUT_WIDTH, dtype=float)
    vector[0::2] = feature_values
    vector[1::2] = 1.0
    return tuple(float(value) for value in vector)


def create_synthetic_clients(
    config: VirtualStudentConfig = VirtualStudentConfig(),
    *,
    samples_per_client: int = 40,
    evaluation_samples_per_class: int = 2,
) -> VirtualStudentPartitions:
    """Generate intentionally non-IID, explicitly synthetic infrastructure-test data."""
    if isinstance(samples_per_client, bool) or not isinstance(samples_per_client, int):
        raise TypeError("samples_per_client must be an integer.")
    if samples_per_client < 2:
        raise ValueError("samples_per_client must be at least 2 to cover both classes.")
    if isinstance(evaluation_samples_per_class, bool) or not isinstance(
        evaluation_samples_per_class, int
    ):
        raise TypeError("evaluation_samples_per_class must be an integer.")
    if evaluation_samples_per_class < 1:
        raise ValueError("evaluation_samples_per_class must be at least 1.")

    rng = np.random.default_rng(config.seed)
    clients: list[VirtualStudentClient] = []
    for client_index in range(config.client_count):
        second_class_fraction = (
            0.2 + 0.6 * client_index / (config.client_count - 1)
        )
        positive_count = min(
            samples_per_client - 1,
            max(1, round(samples_per_client * second_class_fraction)),
        )
        class_counts = {
            TARGET_LABELS[0]: samples_per_client - positive_count,
            TARGET_LABELS[1]: positive_count,
        }
        examples: list[TrainingExample] = []
        for label_index, label in enumerate(TARGET_LABELS):
            for sample_index in range(class_counts[label]):
                features = _synthetic_feature_vector(
                    rng,
                    location=client_index * 2.5 + label_index * 0.4,
                )
                examples.append(
                    SyntheticTrainingExample(
                        sample_id=(
                            f"synthetic-client-{client_index + 1:02d}-"
                            f"train-{label_index}-{sample_index:05d}"
                        ),
                        label=label,
                        features=tuple(float(value) for value in features),
                    )
                )
        clients.append(
            _new_client(client_index, SYNTHETIC_TEST_DATA_MODE, examples)
        )

    evaluation: list[TrainingExample] = []
    for client_index in range(config.client_count):
        for label_index, label in enumerate(TARGET_LABELS):
            for sample_index in range(evaluation_samples_per_class):
                features = _synthetic_feature_vector(
                    rng,
                    location=client_index * 2.5 + label_index * 0.4,
                )
                evaluation.append(
                    SyntheticTrainingExample(
                        sample_id=(
                            f"synthetic-client-{client_index + 1:02d}-"
                            f"evaluation-{label_index}-{sample_index:05d}"
                        ),
                        label=label,
                        features=tuple(float(value) for value in features),
                    )
                )

    return VirtualStudentPartitions(
        clients=tuple(clients),
        evaluation_examples=tuple(evaluation),
        data_mode=SYNTHETIC_TEST_DATA_MODE,
    )


def _reviewed_window_key(example: ReviewedTrainingExample) -> tuple[str, str]:
    provenance = example.feedback_provenance
    start = provenance.get("event_window_start_utc")
    end = provenance.get("event_window_end_utc")
    if not isinstance(start, str) or not isinstance(end, str):
        raise ValueError(
            f"Reviewed example {example.feedback_id!r} has no verified event-window identity."
        )
    return start, end


def _allocation_counts(total: int, client_count: int, label_index: int) -> list[int]:
    counts = [1] * client_count
    remaining = total - client_count
    if remaining <= 0:
        return counts
    weights = np.asarray(
        [
            1.0 + 0.6 * ((client_index + label_index) % client_count)
            for client_index in range(client_count)
        ],
        dtype=float,
    )
    exact = remaining * weights / weights.sum()
    extra = np.floor(exact).astype(int)
    for index, count in enumerate(extra):
        counts[index] += int(count)
    remainder = remaining - int(extra.sum())
    order = sorted(
        range(client_count),
        key=lambda index: (-(exact[index] - extra[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def _feature_signature(example: ReviewedTrainingExample, feature_name: str) -> float:
    value = example.record.feature_values.get(feature_name)
    return float(value) if isinstance(value, (int, float)) else 0.0


def create_reviewed_clients(
    feedback_store: FeedbackStore,
    review_manifest: str | Path,
    config: VirtualStudentConfig = VirtualStudentConfig(),
    *,
    evaluation_fraction: float = 0.2,
) -> VirtualStudentPartitions:
    """Load eligible reviewed data through Milestone 1, then partition locally."""
    if isinstance(evaluation_fraction, bool) or not 0.0 <= evaluation_fraction < 1.0:
        raise ValueError("evaluation_fraction must be at least 0 and less than 1.")

    loaded = load_reviewed_feedback(feedback_store, review_manifest)
    examples = list(loaded.examples)
    if not examples:
        rejection_count = len(loaded.rejections)
        raise ValueError(
            "No eligible reviewed examples are available for virtual clients "
            f"({rejection_count} eligibility rejection(s)); review the manifest and "
            "provide explicitly approved, fresh, schema-compatible records."
        )

    windows: dict[tuple[str, str], str] = {}
    event_contexts: dict[str, str] = {}
    for example in examples:
        key = _reviewed_window_key(example)
        context = example.feedback_provenance.get("evidence_context_reference")
        if not isinstance(context, str) or not context:
            raise ValueError(
                f"Reviewed example {example.feedback_id!r} has no verified source-event identity."
            )
        if key in windows or context in event_contexts:
            raise ValueError(
                "Duplicate reviewed source event or window would leak data across "
                f"partitions: {context!r} / {key[0]} through {key[1]}."
            )
        windows[key] = example.feedback_id
        event_contexts[context] = example.feedback_id

    grouped = {
        label: [example for example in examples if example.label == label]
        for label in TARGET_LABELS
    }
    training_by_label: dict[str, list[ReviewedTrainingExample]] = {}
    evaluation: list[ReviewedTrainingExample] = []
    required_evaluation = 1 if evaluation_fraction > 0 else 0
    minimum = config.client_count + required_evaluation
    missing = {
        label: len(class_examples)
        for label, class_examples in grouped.items()
        if len(class_examples) < minimum
    }
    if missing:
        details = ", ".join(
            f"{label}: {available} available, {minimum} required"
            for label, available in missing.items()
        )
        raise ValueError(
            f"Insufficient eligible class coverage for {config.client_count} clients "
            f"with evaluation_fraction={evaluation_fraction:g} ({details}). Add eligible "
            "reviewed examples with explicit labels, or request fewer clients; labels "
            "will not be inferred or fabricated."
        )

    rng = np.random.default_rng(config.seed)
    allocations_by_label: dict[str, list[int]] = {}
    feature_variances: dict[str, float] = {}
    for name in feature_order():
        values = [
            example.record.feature_values[name]
            for example in examples
            if example.record.feature_validity[name]
        ]
        feature_variances[name] = float(np.var(values)) if values else -1.0
    signature_feature = max(feature_order(), key=lambda name: feature_variances[name])
    tie_breakers = {
        example.feedback_id: float(rng.random())
        for example in examples
    }

    for label_index, label in enumerate(TARGET_LABELS):
        label_examples = grouped[label]
        shuffled = list(label_examples)
        rng.shuffle(shuffled)
        evaluation_count = (
            max(1, round(len(shuffled) * evaluation_fraction))
            if evaluation_fraction > 0
            else 0
        )
        if len(shuffled) - evaluation_count < config.client_count:
            raise ValueError(
                f"Insufficient training coverage for label {label!r}: "
                f"{len(shuffled)} eligible examples leave fewer than "
                f"{config.client_count} after holding out {evaluation_count}."
            )
        evaluation.extend(shuffled[:evaluation_count])
        training = sorted(
            shuffled[evaluation_count:],
            key=lambda example: (
                _feature_signature(example, signature_feature),
                tie_breakers[example.feedback_id],
            ),
        )
        allocations = _allocation_counts(
            len(training),
            config.client_count,
            label_index,
        )
        training_by_label[label] = training
        allocations_by_label[label] = allocations

    client_examples: list[list[TrainingExample]] = [
        [] for _ in range(config.client_count)
    ]
    for label in TARGET_LABELS:
        allocations = allocations_by_label[label]
        start = 0
        for client_index, count in enumerate(allocations):
            client_examples[client_index].extend(
                training_by_label[label][start : start + count]
            )
            start += count

    clients = tuple(
        _new_client(index, REVIEWED_DATA_MODE, examples_for_client)
        for index, examples_for_client in enumerate(client_examples)
    )
    return VirtualStudentPartitions(
        clients=clients,
        evaluation_examples=tuple(evaluation),
        data_mode=REVIEWED_DATA_MODE,
        eligibility_rejections=loaded.rejections,
    )
