"""Minimal scikit-learn trainer and inference interface for schema-compliant evidence records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fusion.schema import (
    FEATURE_SCHEMA_VERSION,
    FeatureRecord,
    ensure_record_schema,
    feature_order,
)

TARGET_LABELS = (
    "potential_distress",
    "no_sufficient_distress_evidence",
)

TARGET_LABEL_MAPPING = {
    "potential_distress": 1,
    "no_sufficient_distress_evidence": 0,
}

LABEL_ALIASES = {
    "potential_distress": "potential_distress",
    "distress": "potential_distress",
    "distressed": "potential_distress",
    "stressed": "potential_distress",
    "panic": "potential_distress",
    "crying": "potential_distress",
    "tantrum": "potential_distress",
    "no_distress": "no_sufficient_distress_evidence",
    "no_sufficient_distress_evidence": "no_sufficient_distress_evidence",
    "normal": "no_sufficient_distress_evidence",
    "baseline": "no_sufficient_distress_evidence",
    "neutral": "no_sufficient_distress_evidence",
    "calm": "no_sufficient_distress_evidence",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "distress_logistic"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "pipeline.joblib"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_PATH.with_suffix(".metadata.json")


@dataclass(frozen=True)
class EvidenceSufficiencyPolicy:
    """Simple, configurable evidence policy for model inference.

    A model prediction is only allowed when the record passes schema validation and
    has enough valid evidence to support the claim. This is not a clinical
    threshold and should be treated as a safety gate for local model use.

    The default requires evidence across at least two independent modalities so a
    single-modality feature bundle cannot masquerade as a multimodal distress
    prediction.
    """

    min_valid_features: int = 4
    min_available_modalities: int = 2
    required_modalities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.min_valid_features < 1:
            raise ValueError("min_valid_features must be at least 1.")
        if self.min_available_modalities < 1:
            raise ValueError("min_available_modalities must be at least 1.")

    def evaluate(self, record: FeatureRecord) -> tuple[bool, str | None]:
        valid_features = [
            name for name in record.feature_names if record.feature_validity.get(name, False)
        ]
        if not valid_features:
            return False, "No valid features are available for inference."
        if len(valid_features) < self.min_valid_features:
            return False, (
                f"Only {len(valid_features)} valid features are available; the minimum "
                f"is {self.min_valid_features}."
            )
        available_modalities = {
            name.split(".", 1)[0] for name in valid_features if "." in name
        }
        if self.min_available_modalities and len(available_modalities) < self.min_available_modalities:
            return False, (
                f"Only {len(available_modalities)} modality(s) have valid features; the minimum "
                f"is {self.min_available_modalities}."
            )
        if self.required_modalities and not set(self.required_modalities).issubset(available_modalities):
            missing = sorted(set(self.required_modalities) - available_modalities)
            return False, f"Required modalities missing: {missing}."
        return True, None


@dataclass
class DistressInferenceResult:
    status: str
    predicted_label: str | None
    model_score: float | None
    valid_feature_count: int
    available_modalities: dict[str, bool]
    reason: str | None
    model_schema_version: str | None = None
    model_feature_order: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    potential_distress_score: float | None = None

    @property
    def potential_distress_probability(self) -> float | None:
        """Probability for the explicit potential-distress class, not predicted class."""
        return self.potential_distress_score


@dataclass
class TrainedDistressModel:
    pipeline: Pipeline
    metadata: dict[str, Any]
    model_path: Path | None = None

    @property
    def schema_version(self) -> str:
        return str(self.metadata.get("schema_version", FEATURE_SCHEMA_VERSION))

    @property
    def feature_order(self) -> tuple[str, ...]:
        return tuple(self.metadata.get("feature_order") or feature_order())

    @property
    def target_label_mapping(self) -> dict[str, int]:
        return dict(self.metadata.get("target_label_mapping", TARGET_LABEL_MAPPING))

    @property
    def target_label_mapping_inverse(self) -> dict[int, str]:
        return {int(value): str(key) for key, value in self.target_label_mapping.items()}

    @property
    def is_demo_model(self) -> bool:
        return bool(
            self.metadata.get("contains_acted_or_synthetic_records")
            or self.metadata.get("synthetic_data_warning")
        )

    def save(self, model_path: str | Path | None = None) -> Path:
        target_path = Path(model_path) if model_path else self.model_path or DEFAULT_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, target_path)
        metadata_path = target_path.with_suffix(".metadata.json")
        metadata_path.write_text(json.dumps(self.metadata, indent=2, sort_keys=True), encoding="utf-8")
        return target_path


    @classmethod
    def load(
        cls,
        model_path: str | Path | None = None,
        *,
        strict: bool = True,
    ) -> "TrainedDistressModel | None":
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.exists():
            if strict:
                raise FileNotFoundError(f"Distress model not found at {path}.")
            return None

        pipeline = joblib.load(path)
        metadata_path = path.with_suffix(".metadata.json")
        if not metadata_path.exists():
            if strict:
                raise ValueError(f"Model metadata file is missing: {metadata_path}")
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"Model metadata at {metadata_path} must be a JSON object.")
        if not isinstance(pipeline, Pipeline):
            raise TypeError(f"Saved object at {path} is not a scikit-learn Pipeline.")
        return cls(pipeline=pipeline, metadata=metadata, model_path=path)


def _validate_training_label(label: str) -> str:
    normalized = str(label).strip()
    if not normalized:
        raise ValueError("Training labels cannot be empty.")
    lowered = normalized.lower()
    if lowered == "unknown":
        raise ValueError("'unknown' is an inference/evidence-sufficiency state, not a training label.")
    if lowered in LABEL_ALIASES:
        return LABEL_ALIASES[lowered]
    if normalized in TARGET_LABELS:
        return normalized
    raise ValueError(
        f"Unsupported training label {normalized!r}. Supported labels are {TARGET_LABELS}."
    )


def _scenario_group(record: FeatureRecord) -> str:
    provenance = record.provenance or {}
    scenario_group = provenance.get("scenario_group")
    if scenario_group:
        return str(scenario_group)
    scenario_label = record.scenario_label or ""
    if scenario_label:
        parts = scenario_label.split(":")
        if len(parts) > 2:
            return ":".join(parts[:-1])
        return scenario_label
    return "record_group"


def _validate_feature_order(record: FeatureRecord) -> None:
    if record.schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Schema version mismatch: expected {FEATURE_SCHEMA_VERSION!r}, got {record.schema_version!r}."
        )
    if tuple(record.feature_names) != tuple(feature_order()):
        raise ValueError(
            "Feature order mismatch: record feature names do not match the canonical schema order."
        )


def _record_to_feature_vector(record: FeatureRecord) -> np.ndarray:
    _validate_feature_order(record)
    feature_values: list[float] = []
    for name in feature_order():
        value = record.feature_values.get(name)
        valid = bool(record.feature_validity.get(name, False))
        feature_values.append(float(value) if valid and value is not None else np.nan)
        feature_values.append(1.0 if valid and value is not None else 0.0)
    return np.asarray(feature_values, dtype=float).reshape(1, -1)


def _records_to_matrix(records: Sequence[FeatureRecord]) -> np.ndarray:
    matrix = [_record_to_feature_vector(record)[0] for record in records]
    return np.asarray(matrix, dtype=float)


def _split_records_by_group(
    records: Sequence[FeatureRecord],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[FeatureRecord], list[FeatureRecord]]:
    if not records:
        raise ValueError("No records were provided for training.")
    if len(records) < 2:
        raise ValueError("At least two records are required to create a train/test split.")
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(_scenario_group(record), []).append(index)
    if len(groups) < 2:
        raise ValueError("Not enough distinct scenario groups are available for leakage-safe splitting.")

    rng = np.random.default_rng(random_state)
    selected_test_indices: set[int] = set()
    group_keys = list(groups)
    rng.shuffle(group_keys)
    test_group_count = max(1, int(round(len(group_keys) * test_size)))
    test_groups = set(group_keys[:test_group_count])
    for group_name, indexes in groups.items():
        if group_name in test_groups:
            selected_test_indices.update(indexes)
    if not selected_test_indices:
        raise ValueError("Training split produced an empty test set.")
    train_indices = [index for index in range(len(records)) if index not in selected_test_indices]
    test_indices = [index for index in range(len(records)) if index in selected_test_indices]
    if not train_indices or not test_indices:
        raise ValueError("Training split produced an empty train or test set.")
    return [records[index] for index in train_indices], [records[index] for index in test_indices]


def _prepare_record_set(
    records: Iterable[FeatureRecord | Mapping[str, Any]],
    *,
    label_lookup: Mapping[str, str] | None = None,
) -> list[tuple[FeatureRecord, str]]:
    prepared: list[tuple[FeatureRecord, str]] = []
    for record in records:
        validated = ensure_record_schema(record)
        explicit_label = record.get("label") if isinstance(record, Mapping) else None
        if explicit_label is not None:
            normalized = _validate_training_label(str(explicit_label))
        elif label_lookup is not None and validated.scenario_label is not None:
            key = str(validated.scenario_label)
            if key not in label_lookup:
                raise ValueError(f"No training label mapped for scenario label {key!r}.")
            normalized = _validate_training_label(label_lookup[key])
        else:
            raise ValueError(
                "Every training record requires an explicit ground-truth 'label' "
                "or an explicit label_lookup entry for its scenario_label."
            )
        prepared.append((validated, normalized))
    return prepared


def train_distress_logistic_model(
    records: Iterable[FeatureRecord | Mapping[str, Any]],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    use_group_split: bool = True,
    label_lookup: Mapping[str, str] | None = None,
    model_path: str | Path | None = None,
    class_weight: str | None = "balanced",
    allow_acted_synthetic: bool = False,
) -> TrainedDistressModel:
    """Train a small logistic-regression model using schema-compliant FeatureRecord objects."""

    prepared = _prepare_record_set(records, label_lookup=label_lookup)
    if not prepared:
        raise ValueError("No valid records were provided for training.")
    acted_records = [
        record
        for record, _ in prepared
        if record.is_acted_synthetic or record.record_kind == "synthetic_acted"
    ]
    if acted_records and not allow_acted_synthetic:
        raise ValueError(
            "Acted/synthetic records are not enabled for training. Pass "
            "allow_acted_synthetic=True only for an explicitly identified pipeline demonstration."
        )

    labels = [label for _, label in prepared]
    unique_labels = sorted(set(labels))
    if set(unique_labels) != set(TARGET_LABELS):
        missing = sorted(set(TARGET_LABELS) - set(unique_labels))
        raise ValueError(
            f"Training data must include both target labels. Missing: {missing}. "
            "Binary distress labels are required for this logistic-regression phase."
        )

    if use_group_split and len(prepared) > 2:
        train_records, test_records = _split_records_by_group(
            [record for record, _ in prepared], test_size=test_size, random_state=random_state
        )
        labels_by_record_id = {id(record): label for record, label in prepared}
        train_pairs = [
            (record, labels_by_record_id[id(record)])
            for record in train_records
        ]
        test_pairs = [
            (record, labels_by_record_id[id(record)])
            for record in test_records
        ]
    else:
        train_pairs = prepared
        test_pairs = []

    if {label for _, label in train_pairs} != set(TARGET_LABELS):
        raise ValueError(
            "The leakage-safe training split must contain both target labels. "
            "Add more distinct provenance.scenario_group values for each class."
        )

    X_train = _records_to_matrix([record for record, _ in train_pairs])
    y_train = np.asarray([TARGET_LABEL_MAPPING[label] for _, label in train_pairs], dtype=int)
    X_test = _records_to_matrix([record for record, _ in test_pairs]) if test_pairs else None
    y_test = np.asarray([TARGET_LABEL_MAPPING[label] for _, label in test_pairs], dtype=int) if test_pairs else None

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", missing_values=np.nan)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                    class_weight=class_weight,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    def source_counts(pairs: Sequence[tuple[FeatureRecord, str]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record, _ in pairs:
            source = str(record.provenance.get("source", "unspecified"))
            counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items()))

    train_groups = {_scenario_group(record) for record, _ in train_pairs}
    test_groups = {_scenario_group(record) for record, _ in test_pairs}
    all_records_synthetic = all(
        record.is_acted_synthetic or record.record_kind == "synthetic_acted"
        for record, _ in prepared
    )
    metadata = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_order": list(feature_order()),
        "target_label_mapping": {label: int(value) for label, value in TARGET_LABEL_MAPPING.items()},
        "training_record_counts_by_class": {
            label: int(sum(1 for _, candidate in train_pairs if candidate == label))
            for label in TARGET_LABELS
        },
        "test_record_counts_by_class": {
            label: int(sum(1 for _, candidate in test_pairs if candidate == label)) if test_pairs else 0
            for label in TARGET_LABELS
        },
        "record_kinds": sorted({record.record_kind for record, _ in prepared}),
        "contains_acted_or_synthetic_records": bool(
            any(
                record.is_acted_synthetic or record.record_kind == "synthetic_acted"
                for record, _ in prepared
            )
        ),
        "record_provenance": {
            "synthetic_acted": int(
                sum(
                    1
                    for record, _ in prepared
                    if record.is_acted_synthetic or record.record_kind == "synthetic_acted"
                )
            ),
            "live_monitoring": int(
                sum(1 for record, _ in prepared if record.record_kind == "live_monitoring")
            ),
        },
        "training_provenance": {
            "dataset_record_count": len(prepared),
            "all_records_synthetic_or_acted": all_records_synthetic,
            "source_counts_by_split": {
                "train": source_counts(train_pairs),
                "test": source_counts(test_pairs),
            },
            "scenario_group_counts_by_split": {
                "train": len(train_groups),
                "test": len(test_groups),
            },
            "scenario_groups_disjoint": train_groups.isdisjoint(test_groups),
        },
        "preprocessing": {
            "missing_value_strategy": "median_imputation_with_validity_indicators",
            "feature_expansion": "value_plus_validity_flag_per_feature",
            "scaler": "standard",
            "class_weight": class_weight,
        },
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": "This model is a local pipeline demonstration only and is not validated clinical distress detection.",
        "synthetic_data_warning": (
            "Acted/synthetic training records are pipeline demonstrations only, not "
            "validated observations or clinical distress detection."
            if acted_records
            else "Training records were explicitly labeled; their provenance and validity require independent review."
        ),
        "record_split_by_group": bool(use_group_split and test_pairs),
    }

    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    trained = TrainedDistressModel(pipeline=pipeline, metadata=metadata, model_path=target_path)
    trained.save(target_path)
    return trained


def load_distress_model(
    model_path: str | Path | None = None,
    *,
    strict: bool = False,
) -> TrainedDistressModel | None:
    """Load a saved distress model and metadata from disk."""
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    candidate = TrainedDistressModel.load(path, strict=strict)
    if candidate is None:
        return None
    metadata = candidate.metadata or {}
    if metadata.get("schema_version") != FEATURE_SCHEMA_VERSION:
        if strict:
            raise ValueError(
                f"Incompatible schema version in model metadata: {metadata.get('schema_version')!r}. "
                f"Expected {FEATURE_SCHEMA_VERSION!r}."
            )
        return None
    feature_order_from_metadata = tuple(metadata.get("feature_order") or ())
    if tuple(feature_order_from_metadata) != tuple(feature_order()):
        if strict:
            raise ValueError(
                "Saved model feature order does not match the current schema order. "
                "This model is incompatible with the current feature schema."
            )
        return None
    if metadata.get("target_label_mapping") != TARGET_LABEL_MAPPING:
        if strict:
            raise ValueError("Saved model target label mapping is missing or incompatible.")
        return None
    classifier = candidate.pipeline.named_steps.get("classifier")
    if not isinstance(classifier, LogisticRegression) or set(classifier.classes_) != {0, 1}:
        if strict:
            raise ValueError(
                "Saved pipeline does not contain the expected binary logistic-regression classifier."
            )
        return None
    expected_input_features = len(feature_order()) * 2
    if candidate.pipeline.n_features_in_ != expected_input_features:
        if strict:
            raise ValueError(
                "Saved pipeline input width does not match values-plus-validity schema features."
            )
        return None
    return candidate


def evidence_sufficient(
    record: FeatureRecord | Mapping[str, Any],
    *,
    policy: EvidenceSufficiencyPolicy | None = None,
) -> tuple[bool, str | None]:
    validated = ensure_record_schema(record)
    selected_policy = policy or EvidenceSufficiencyPolicy()
    return selected_policy.evaluate(validated)


def infer_distress_status(
    record: FeatureRecord | Mapping[str, Any],
    *,
    model: TrainedDistressModel | None = None,
    model_path: str | Path | None = None,
    policy: EvidenceSufficiencyPolicy | None = None,
    strict: bool = False,
) -> DistressInferenceResult:
    """Infer a local distress status from a validated FeatureRecord."""

    validated = ensure_record_schema(record)
    available_modalities = {
        modality: bool(
            any(
                name.startswith(f"{modality}.") and validated.feature_validity.get(name, False)
                for name in validated.feature_names
            )
        )
        for modality in ("face", "behavior", "audio")
    }
    valid_feature_count = sum(
        1 for name in validated.feature_names if bool(validated.feature_validity.get(name, False))
    )

    if model is None:
        try:
            model = load_distress_model(
                model_path or DEFAULT_MODEL_PATH,
                strict=True,
            )
        except FileNotFoundError:
            if strict:
                raise
            model = None
        except (OSError, TypeError, ValueError) as exc:
            if strict:
                raise
            return DistressInferenceResult(
                status="unknown",
                predicted_label=None,
                model_score=None,
                valid_feature_count=valid_feature_count,
                available_modalities=available_modalities,
                reason=f"Saved distress model is incompatible or invalid: {exc}",
                model_schema_version=None,
                model_feature_order=None,
                metadata={},
            )
    if model is None:
        return DistressInferenceResult(
            status="unknown",
            predicted_label=None,
            model_score=None,
            valid_feature_count=valid_feature_count,
            available_modalities=available_modalities,
            reason="No trained distress model is available for inference.",
            model_schema_version=None,
            model_feature_order=None,
            metadata={},
        )

    if model.schema_version != FEATURE_SCHEMA_VERSION:
        return DistressInferenceResult(
            status="unknown",
            predicted_label=None,
            model_score=None,
            valid_feature_count=valid_feature_count,
            available_modalities=available_modalities,
            reason=f"Model schema mismatch: {model.schema_version!r} != {FEATURE_SCHEMA_VERSION!r}.",
            model_schema_version=model.schema_version,
            model_feature_order=model.feature_order,
            metadata=model.metadata,
        )

    if tuple(validated.feature_names) != tuple(model.feature_order):
        return DistressInferenceResult(
            status="unknown",
            predicted_label=None,
            model_score=None,
            valid_feature_count=valid_feature_count,
            available_modalities=available_modalities,
            reason="Record feature ordering does not match the trained model schema order.",
            model_schema_version=model.schema_version,
            model_feature_order=model.feature_order,
            metadata=model.metadata,
        )

    classifier = model.pipeline.named_steps.get("classifier")
    if (
        model.target_label_mapping != TARGET_LABEL_MAPPING
        or not isinstance(classifier, LogisticRegression)
        or set(classifier.classes_) != {0, 1}
        or model.pipeline.n_features_in_ != len(feature_order()) * 2
    ):
        return DistressInferenceResult(
            status="unknown",
            predicted_label=None,
            model_score=None,
            valid_feature_count=valid_feature_count,
            available_modalities=available_modalities,
            reason="The loaded model pipeline or target mapping is incompatible with this schema.",
            model_schema_version=model.schema_version,
            model_feature_order=model.feature_order,
            metadata=model.metadata,
        )

    selected_policy = policy or EvidenceSufficiencyPolicy()
    sufficient, reason = evidence_sufficient(validated, policy=selected_policy)
    if not sufficient:
        return DistressInferenceResult(
            status="unknown",
            predicted_label=None,
            model_score=None,
            valid_feature_count=valid_feature_count,
            available_modalities=available_modalities,
            reason=reason,
            model_schema_version=model.schema_version,
            model_feature_order=model.feature_order,
            metadata=model.metadata,
        )

    inputs = _record_to_feature_vector(validated)
    probabilities = model.pipeline.predict_proba(inputs)[0]
    target_lookup = model.target_label_mapping_inverse
    predicted_index = int(np.argmax(probabilities))
    predicted_label = target_lookup.get(predicted_index, "no_sufficient_distress_evidence")
    model_score = float(probabilities[predicted_index])
    distress_class_id = model.target_label_mapping["potential_distress"]
    distress_class_index = next(
        index
        for index, class_id in enumerate(classifier.classes_)
        if int(class_id) == distress_class_id
    )
    potential_distress_score = float(probabilities[distress_class_index])
    status = predicted_label

    return DistressInferenceResult(
        status=status,
        predicted_label=predicted_label,
        model_score=model_score,
        valid_feature_count=valid_feature_count,
        available_modalities=available_modalities,
        reason=None,
        model_schema_version=model.schema_version,
        model_feature_order=model.feature_order,
        metadata=model.metadata,
        potential_distress_score=potential_distress_score,
    )


__all__ = [
    "DEFAULT_MODEL_DIR",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_METADATA_PATH",
    "EvidenceSufficiencyPolicy",
    "DistressInferenceResult",
    "TrainedDistressModel",
    "TARGET_LABELS",
    "TARGET_LABEL_MAPPING",
    "evidence_sufficient",
    "infer_distress_status",
    "load_distress_model",
    "train_distress_logistic_model",
]
