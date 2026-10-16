"""Isolated schema-v2 fusion-head initialization and local fine-tuning."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fusion.schema import FEATURE_SCHEMA_VERSION, ensure_record_schema, feature_order
from training.distress_logistic import (
    TARGET_LABEL_MAPPING,
    TARGET_LABELS,
    EvidenceSufficiencyPolicy,
    _record_to_feature_vector,
)
from training.reviewed_feedback import ReviewedTrainingExample

FUSION_HEAD_MODEL_VERSION = "phase4-fusion-head-v1"
FUSION_HEAD_METADATA_VERSION = "phase4-fusion-head-metadata-v1"
FUSION_HEAD_INPUT_WIDTH = len(feature_order()) * 2
FUSION_HEAD_PREPROCESSING_CONTRACT = {
    "contract_version": "fixed-zero-imputation-no-scaling-v1",
    "feature_vector_layout": "interleaved_value_then_validity_flag",
    "missing_value_input": "NaN value coordinate with validity flag 0.0",
    "missing_value_replacement": 0.0,
    "validity_flag_values": [0.0, 1.0],
    "scaling": "none",
    "data_dependent_fit_statistics": False,
    "client_specific_preprocessing": False,
}


@dataclass
class FusionHeadModel:
    pipeline: Pipeline
    metadata: dict[str, Any]
    model_path: Path | None = None


def _expected_metadata() -> dict[str, Any]:
    return {
        "metadata_version": FUSION_HEAD_METADATA_VERSION,
        "model_version": FUSION_HEAD_MODEL_VERSION,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_order": list(feature_order()),
        "input_width": FUSION_HEAD_INPUT_WIDTH,
        "preprocessing_contract": copy.deepcopy(FUSION_HEAD_PREPROCESSING_CONTRACT),
        "target_label_mapping": {
            label: int(value) for label, value in TARGET_LABEL_MAPPING.items()
        },
        "estimator": {
            "class": "sklearn.linear_model.LogisticRegression",
            "solver": "lbfgs",
            "warm_start": True,
            "C": 1.0,
            "class_weight": "balanced",
        },
        "limitations": (
            "Offline experimental head only. It is not validated for clinical use "
            "and is not loaded by live inference."
        ),
    }


def _build_pipeline() -> Pipeline:
    imputer = SimpleImputer(
        strategy="constant",
        fill_value=0.0,
        keep_empty_features=True,
    )
    imputer.fit(np.zeros((1, FUSION_HEAD_INPUT_WIDTH), dtype=float))
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
        warm_start=True,
    )
    classifier.classes_ = np.asarray([0, 1], dtype=int)
    classifier.coef_ = np.zeros((1, FUSION_HEAD_INPUT_WIDTH), dtype=float)
    classifier.intercept_ = np.zeros(1, dtype=float)
    classifier.n_iter_ = np.zeros(1, dtype=np.int32)
    classifier.n_features_in_ = FUSION_HEAD_INPUT_WIDTH
    return Pipeline(steps=[("imputer", imputer), ("classifier", classifier)])


def _validate_pipeline(pipeline: Any) -> None:
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Fusion head artifact must contain a scikit-learn Pipeline.")
    if [name for name, _ in pipeline.steps] != ["imputer", "classifier"]:
        raise ValueError("Fusion head pipeline steps do not match the fixed preprocessing contract.")
    imputer = pipeline.named_steps["imputer"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(imputer, SimpleImputer):
        raise TypeError("Fusion head pipeline must use the fixed SimpleImputer.")
    if (
        imputer.strategy != "constant"
        or imputer.fill_value != 0.0
        or not imputer.keep_empty_features
        or np.asarray(imputer.statistics_).shape != (FUSION_HEAD_INPUT_WIDTH,)
        or not np.all(np.asarray(imputer.statistics_) == 0.0)
    ):
        raise ValueError("Fusion head imputer is incompatible with the fixed zero-fill contract.")
    if not isinstance(classifier, LogisticRegression):
        raise TypeError("Fusion head pipeline must contain a logistic-regression classifier.")
    if (
        not np.array_equal(classifier.classes_, np.asarray([0, 1]))
        or np.asarray(classifier.coef_).shape != (1, FUSION_HEAD_INPUT_WIDTH)
        or np.asarray(classifier.intercept_).shape != (1,)
        or classifier.n_features_in_ != FUSION_HEAD_INPUT_WIDTH
        or classifier.solver != "lbfgs"
        or not classifier.warm_start
        or classifier.C != 1.0
        or classifier.class_weight != "balanced"
    ):
        raise ValueError("Fusion head classifier is incompatible with the metadata contract.")
    if not np.isfinite(classifier.coef_).all() or not np.isfinite(classifier.intercept_).all():
        raise ValueError("Fusion head parameters must be finite.")
    if pipeline.n_features_in_ != FUSION_HEAD_INPUT_WIDTH:
        raise ValueError("Fusion head input width does not match the canonical schema.")


def initialize_fusion_head() -> FusionHeadModel:
    """Create a zero-parameter starter head under the shared schema-v2 contract."""
    return FusionHeadModel(
        pipeline=_build_pipeline(),
        metadata=_expected_metadata(),
    )


def save_fusion_head(model: FusionHeadModel, model_path: str | Path) -> Path:
    """Save a dedicated fusion-head artifact and its versioned metadata sidecar."""
    if model.metadata != _expected_metadata():
        raise ValueError("Fusion head metadata is incompatible with this model version.")
    _validate_pipeline(model.pipeline)
    target = Path(model_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.pipeline, target)
    metadata_path = target.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(model.metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    model.model_path = target
    return target


def load_fusion_head(model_path: str | Path) -> FusionHeadModel:
    """Load and strictly validate a dedicated schema-v2 fusion-head artifact."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Fusion head not found at {path}.")
    metadata_path = path.with_suffix(".metadata.json")
    if not metadata_path.is_file():
        raise ValueError(f"Fusion head metadata file is missing: {metadata_path}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata != _expected_metadata():
        raise ValueError("Fusion head metadata is missing or incompatible.")
    pipeline = joblib.load(path)
    _validate_pipeline(pipeline)
    return FusionHeadModel(pipeline, metadata, path)


def _validate_example(example: ReviewedTrainingExample) -> None:
    if not isinstance(example, ReviewedTrainingExample):
        raise TypeError("Fine-tuning accepts only ReviewedTrainingExample values.")
    if example.label not in TARGET_LABELS:
        raise ValueError("Reviewed example has an unsupported explicit binary label.")
    provenance = example.feedback_provenance
    if (
        provenance.get("feedback_id") != example.feedback_id
        or provenance.get("review_decision") != "approved_for_training"
        or provenance.get("feedback_action")
        not in {"confirm_alert", "reject_alert", "missed_event"}
        or not provenance.get("reviewer_reference")
        or not provenance.get("reviewed_at_utc")
        or not provenance.get("event_window_start_utc")
        or not provenance.get("event_window_end_utc")
        or not provenance.get("modality_timestamps_utc")
        or not provenance.get("evidence_context_reference")
    ):
        raise ValueError("Reviewed example is missing explicit review or event-window provenance.")
    record = ensure_record_schema(example.record)
    sufficient, reason = EvidenceSufficiencyPolicy().evaluate(record)
    if not sufficient:
        raise ValueError(f"Reviewed example is insufficient for training: {reason}")


def fine_tune_fusion_head(
    starting_head: FusionHeadModel | str | Path,
    examples: Sequence[ReviewedTrainingExample],
) -> FusionHeadModel:
    """Fine-tune a compatible supplied head using explicitly reviewed local examples."""
    if isinstance(starting_head, (str, Path)):
        starting_head = load_fusion_head(starting_head)
    if not isinstance(starting_head, FusionHeadModel):
        raise TypeError("A compatible FusionHeadModel or saved head path is required.")
    if starting_head.metadata != _expected_metadata():
        raise ValueError("Starting fusion head is incompatible with the current metadata contract.")
    _validate_pipeline(starting_head.pipeline)
    if not examples:
        raise ValueError("At least one reviewed example is required for fine-tuning.")

    seen_feedback_ids: set[str] = set()
    for example in examples:
        _validate_example(example)
        if example.feedback_id in seen_feedback_ids:
            raise ValueError("Duplicate feedback IDs cannot be used twice in one fine-tuning run.")
        seen_feedback_ids.add(example.feedback_id)
    labels = {example.label for example in examples}
    if labels != set(TARGET_LABELS):
        raise ValueError("Fine-tuning requires explicitly reviewed examples for both binary labels.")

    matrix = np.vstack(
        [_record_to_feature_vector(example.record)[0] for example in examples]
    )
    targets = np.asarray(
        [TARGET_LABEL_MAPPING[example.label] for example in examples],
        dtype=int,
    )
    pipeline = copy.deepcopy(starting_head.pipeline)
    pipeline.fit(matrix, targets)
    _validate_pipeline(pipeline)
    return FusionHeadModel(pipeline, _expected_metadata())
