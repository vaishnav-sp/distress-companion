"""Flower ClientApp/ServerApp workflow for offline fusion-head FedAvg."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import (
    Code,
    Context,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from feedback.store import DEFAULT_FEEDBACK_PATH, FeedbackStore
from fusion.schema import FEATURE_SCHEMA_VERSION, feature_order
from training.distress_logistic import TARGET_LABEL_MAPPING, TARGET_LABELS, _record_to_feature_vector
from training.fusion_head import (
    FUSION_HEAD_INPUT_WIDTH,
    FUSION_HEAD_MODEL_VERSION,
    FUSION_HEAD_PREPROCESSING_CONTRACT,
    FusionHeadModel,
    _validate_pipeline,
    initialize_fusion_head,
    load_fusion_head,
    save_fusion_head,
)
from training.reviewed_feedback import ReviewedTrainingExample

from federated.virtual_students import (
    REVIEWED_DATA_MODE,
    SYNTHETIC_TEST_DATA_MODE,
    SyntheticTrainingExample,
    VirtualStudentClient,
    VirtualStudentConfig,
    create_reviewed_clients,
    create_synthetic_clients,
)

MIN_PARTICIPATING_CLIENTS = 3
FEDERATED_METADATA_VERSION = "phase4-federated-fusion-head-metadata-v1"
FEDAVG_STRATEGY_NAME = "Flower FedAvg"
GLOBAL_MODEL_PREFIX = "phase4-global-r"


@dataclass(frozen=True)
class LocalTrainingConfig:
    solver: str = "lbfgs"
    max_iter: int = 1000
    tolerance: float = 1e-4
    regularization_c: float = 1.0
    class_weight: str = "balanced"
    warm_start: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "max_iter": self.max_iter,
            "tolerance": self.tolerance,
            "regularization_c": self.regularization_c,
            "class_weight": self.class_weight,
            "warm_start": self.warm_start,
        }


DEFAULT_LOCAL_TRAINING_CONFIG = LocalTrainingConfig()


@dataclass(frozen=True)
class FederatedRunResult:
    artifact_path: Path
    model: FusionHeadModel
    federated_metadata: Mapping[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parameter_arrays(parameters: Sequence[np.ndarray]) -> list[np.ndarray]:
    if len(parameters) != 2:
        raise ValueError("Fusion-head parameters must contain exactly coefficients and intercept.")
    arrays = [np.asarray(array, dtype=np.float64) for array in parameters]
    expected_shapes = [(1, FUSION_HEAD_INPUT_WIDTH), (1,)]
    for index, (array, expected_shape) in enumerate(zip(arrays, expected_shapes)):
        if array.shape != expected_shape:
            raise ValueError(
                f"Fusion-head parameter {index} has shape {array.shape}; "
                f"expected {expected_shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Fusion-head parameter {index} contains non-finite values.")
    return arrays


def parameter_digest(parameters: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in _parameter_arrays(parameters):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def parameter_summary(parameters: Sequence[np.ndarray]) -> dict[str, Any]:
    arrays = _parameter_arrays(parameters)
    return {
        "parameter_shapes": [list(array.shape) for array in arrays],
        "parameter_l2_norm": float(
            math.sqrt(sum(float(np.square(array).sum()) for array in arrays))
        ),
        "parameter_sha256": parameter_digest(arrays),
    }


def initial_fusion_parameters() -> list[np.ndarray]:
    head = initialize_fusion_head()
    classifier = head.pipeline.named_steps["classifier"]
    return _parameter_arrays([classifier.coef_.copy(), classifier.intercept_.copy()])


def _set_parameters(head: FusionHeadModel, parameters: Sequence[np.ndarray]) -> None:
    coefficients, intercept = _parameter_arrays(parameters)
    classifier = head.pipeline.named_steps["classifier"]
    classifier.coef_ = coefficients.copy()
    classifier.intercept_ = intercept.copy()
    classifier.classes_ = np.asarray([0, 1], dtype=int)
    classifier.n_features_in_ = FUSION_HEAD_INPUT_WIDTH
    classifier.n_iter_ = np.zeros(1, dtype=np.int32)
    _validate_pipeline(head.pipeline)


def _training_matrix(
    examples: Sequence[ReviewedTrainingExample | SyntheticTrainingExample],
) -> tuple[np.ndarray, np.ndarray]:
    if not examples:
        raise ValueError("A Flower client requires a non-empty local training partition.")
    rows: list[np.ndarray] = []
    targets: list[int] = []
    for example in examples:
        if example.label not in TARGET_LABEL_MAPPING:
            raise ValueError("Local partition contains an unsupported explicit binary label.")
        if isinstance(example, ReviewedTrainingExample):
            rows.append(_record_to_feature_vector(example.record)[0])
        elif isinstance(example, SyntheticTrainingExample):
            if example.data_mode != SYNTHETIC_TEST_DATA_MODE:
                raise ValueError("Synthetic examples must be visibly marked synthetic_test.")
            rows.append(np.asarray(example.features, dtype=np.float64))
        else:
            raise TypeError("Flower local training accepts only reviewed or marked synthetic data.")
        targets.append(TARGET_LABEL_MAPPING[example.label])

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(examples), FUSION_HEAD_INPUT_WIDTH):
        raise ValueError(
            f"Local feature matrix has shape {matrix.shape}; expected "
            f"({len(examples)}, {FUSION_HEAD_INPUT_WIDTH})."
        )
    if set(targets) != {0, 1}:
        raise ValueError(
            "Each client requires eligible local examples for both explicit binary labels."
        )
    if np.isinf(matrix).any():
        raise ValueError("Local features contain infinite values.")
    return matrix, np.asarray(targets, dtype=np.int64)


def _partition_for_client(
    partition_id: int,
    config: VirtualStudentConfig,
    data_mode: str,
    feedback_db: str | Path,
    review_manifest: str | Path | None,
) -> VirtualStudentClient:
    if partition_id < 0 or partition_id >= config.client_count:
        raise ValueError(
            f"Flower partition ID {partition_id} is outside the configured "
            f"{config.client_count}-client cohort."
        )
    if data_mode == SYNTHETIC_TEST_DATA_MODE:
        partitions = create_synthetic_clients(config)
    elif data_mode == REVIEWED_DATA_MODE:
        if review_manifest is None:
            raise ValueError("Reviewed-data mode requires a local review manifest path.")
        partitions = create_reviewed_clients(
            FeedbackStore(feedback_db),
            review_manifest,
            config,
        )
    else:
        raise ValueError(f"Unsupported federated data mode {data_mode!r}.")
    return partitions.clients[partition_id]


def _expected_protocol_config(
    round_number: int,
    global_model_version: str,
    global_parameters_digest: str,
    seed: int,
    training_config: LocalTrainingConfig,
) -> dict[str, Scalar]:
    expected_model_version = f"{GLOBAL_MODEL_PREFIX}{round_number - 1:04d}"
    if global_model_version != expected_model_version:
        raise ValueError(
            f"Global model version {global_model_version!r} is not compatible with "
            f"round {round_number}; expected {expected_model_version!r}."
        )
    return {
        "protocol_round": round_number,
        "global_model_version": global_model_version,
        "global_parameters_sha256": global_parameters_digest,
        "model_architecture_version": FUSION_HEAD_MODEL_VERSION,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_order_json": _canonical_json(feature_order()),
        "input_width": FUSION_HEAD_INPUT_WIDTH,
        "preprocessing_contract_json": _canonical_json(
            FUSION_HEAD_PREPROCESSING_CONTRACT
        ),
        "target_label_mapping_json": _canonical_json(TARGET_LABEL_MAPPING),
        "training_seed": seed,
        "training_config_json": _canonical_json(training_config.to_dict()),
    }


class _LocalFusionHeadClient(NumPyClient):
    def __init__(
        self,
        client: VirtualStudentClient,
        *,
        seed: int,
        training_config: LocalTrainingConfig,
    ) -> None:
        self._client = client
        self._seed = seed
        self._training_config = training_config

    def get_parameters(self, config: Mapping[str, Scalar]) -> list[np.ndarray]:
        del config
        return initial_fusion_parameters()

    def fit(
        self,
        parameters: list[np.ndarray],
        config: Mapping[str, Scalar],
    ) -> tuple[list[np.ndarray], int, dict[str, Scalar]]:
        round_number = config.get("protocol_round")
        if not isinstance(round_number, int) or round_number < 1:
            raise ValueError("Flower fit config is missing a valid protocol round.")
        model_version = config.get("global_model_version")
        parameter_hash = config.get("global_parameters_sha256")
        if not isinstance(model_version, str) or not isinstance(parameter_hash, str):
            raise ValueError("Flower fit config is missing global model compatibility metadata.")
        expected = _expected_protocol_config(
            round_number,
            model_version,
            parameter_hash,
            self._seed,
            self._training_config,
        )
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"Flower fit config has incompatible {key}.")

        global_arrays = _parameter_arrays(parameters)
        if parameter_digest(global_arrays) != parameter_hash:
            raise ValueError("Received global parameters do not match their validated digest.")

        examples = self._client.load_training_partition()
        matrix, targets = _training_matrix(examples)
        if len(examples) != self._client.data_quality().sample_count:
            raise ValueError("Local partition sample count changed during client training.")
        counts = {
            label: sum(example.label == label for example in examples)
            for label in TARGET_LABELS
        }
        if any(count == 0 for count in counts.values()):
            raise ValueError("Client local training partition is missing required class coverage.")

        head = initialize_fusion_head()
        _set_parameters(head, global_arrays)
        classifier = head.pipeline.named_steps["classifier"]
        transformed = head.pipeline.named_steps["imputer"].transform(matrix)
        classifier.set_params(
            C=self._training_config.regularization_c,
            class_weight=self._training_config.class_weight,
            max_iter=self._training_config.max_iter,
            random_state=self._seed,
            solver=self._training_config.solver,
            tol=self._training_config.tolerance,
            warm_start=self._training_config.warm_start,
        )
        classifier.fit(transformed, targets)
        local_arrays = _parameter_arrays(
            [classifier.coef_.copy(), classifier.intercept_.copy()]
        )
        _validate_pipeline(head.pipeline)

        metrics: dict[str, Scalar] = {
            "client_id": self._client.client_id,
            "protocol_round": round_number,
            "global_model_version": model_version,
            "model_architecture_version": FUSION_HEAD_MODEL_VERSION,
            "schema_version": FEATURE_SCHEMA_VERSION,
            "feature_order_json": _canonical_json(feature_order()),
            "input_width": FUSION_HEAD_INPUT_WIDTH,
            "preprocessing_contract_json": _canonical_json(
                FUSION_HEAD_PREPROCESSING_CONTRACT
            ),
            "target_label_mapping_json": _canonical_json(TARGET_LABEL_MAPPING),
            "training_seed": self._seed,
            "training_config_json": _canonical_json(self._training_config.to_dict()),
            "global_parameters_sha256": parameter_hash,
            "class_count_potential_distress": counts["potential_distress"],
            "class_count_no_sufficient_distress_evidence": counts[
                "no_sufficient_distress_evidence"
            ],
            "data_mode": self._client.data_mode,
            "synthetic_test_data": self._client.data_mode == SYNTHETIC_TEST_DATA_MODE,
        }
        return local_arrays, len(examples), metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: Mapping[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        del parameters, config
        return 0.0, 0, {}


class ValidatingFlowerFedAvg(FedAvg):
    """Flower FedAvg with strict protocol checks and auditable round records."""

    def __init__(
        self,
        client_count: int,
        rounds: int,
        seed: int,
        data_mode: str,
        training_config: LocalTrainingConfig = DEFAULT_LOCAL_TRAINING_CONFIG,
    ) -> None:
        if not MIN_PARTICIPATING_CLIENTS <= client_count <= 5:
            raise ValueError("Flower simulation requires between 3 and 5 clients.")
        if rounds < 1:
            raise ValueError("At least one Flower aggregation round is required.")
        self.client_count = client_count
        self.rounds = rounds
        self.seed = seed
        self.data_mode = data_mode
        self.training_config = training_config
        self.round_records: list[dict[str, Any]] = []
        self.current_model_version = f"{GLOBAL_MODEL_PREFIX}0000"
        self.current_parameters = initial_fusion_parameters()
        self.current_parameters_digest = parameter_digest(self.current_parameters)
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=client_count,
            min_evaluate_clients=client_count,
            min_available_clients=client_count,
            on_fit_config_fn=self._fit_config,
            fit_metrics_aggregation_fn=self._aggregate_safe_metrics,
            accept_failures=False,
            initial_parameters=ndarrays_to_parameters(self.current_parameters),
            inplace=False,
        )

    def _fit_config(self, round_number: int) -> dict[str, Scalar]:
        return _expected_protocol_config(
            round_number,
            self.current_model_version,
            self.current_parameters_digest,
            self.seed,
            self.training_config,
        )

    @staticmethod
    def _aggregate_safe_metrics(
        results: list[tuple[int, dict[str, Scalar]]],
    ) -> dict[str, Scalar]:
        return {
            "eligible_examples": sum(num_examples for num_examples, _ in results),
            "participating_clients": len(results),
        }

    def _validate_update(
        self,
        round_number: int,
        fit_res: FitRes,
        used_client_ids: set[str],
    ) -> tuple[str | None, str | None]:
        metrics = fit_res.metrics
        if fit_res.status.code != Code.OK:
            return None, "Client update status is not successful."
        client_id = metrics.get("client_id")
        if not isinstance(client_id, str) or not client_id:
            return None, "Client update is missing a stable client ID."
        if client_id in used_client_ids:
            return client_id, "Duplicate client update in the same round."
        if client_id not in {
            f"virtual-student-{index + 1:02d}"
            for index in range(self.client_count)
        }:
            return client_id, "Client ID is not part of the configured cohort."
        if fit_res.num_examples < 2:
            return client_id, "Client update has too few eligible local examples."

        expected_config = self._fit_config(round_number)
        expected_metrics: dict[str, Scalar] = {
            key: value
            for key, value in expected_config.items()
            if key != "global_parameters_sha256"
        }
        expected_metrics["global_parameters_sha256"] = self.current_parameters_digest
        for key, expected_value in expected_metrics.items():
            if metrics.get(key) != expected_value:
                return client_id, f"Client update has incompatible {key}."
        if metrics.get("global_model_version") != self.current_model_version:
            return client_id, "Client update started from an incompatible global model version."

        expected_class_counts = (
            metrics.get("class_count_potential_distress"),
            metrics.get("class_count_no_sufficient_distress_evidence"),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in expected_class_counts
        ):
            return client_id, "Client update has invalid aggregate class counts."
        if sum(expected_class_counts) != fit_res.num_examples:
            return client_id, "Client class counts do not match its eligible sample count."
        if metrics.get("data_mode") != self.data_mode:
            return client_id, "Client update has a data mode incompatible with this run."
        if metrics.get("synthetic_test_data") != (
            metrics.get("data_mode") == SYNTHETIC_TEST_DATA_MODE
        ):
            return client_id, "Client synthetic-data marker is inconsistent."

        try:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            _parameter_arrays(arrays)
        except (TypeError, ValueError) as exc:
            return client_id, f"Client update parameters are invalid: {exc}"
        used_client_ids.add(client_id)
        return client_id, None

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        participants: list[dict[str, Any]] = []
        used_client_ids: set[str] = set()
        errors: list[str] = []
        for _, fit_res in results:
            client_id, error = self._validate_update(
                server_round,
                fit_res,
                used_client_ids,
            )
            if error is not None:
                errors.append(error)
            elif client_id is not None:
                participants.append(
                    {
                        "client_id": client_id,
                        "eligible_training_examples": fit_res.num_examples,
                        "class_counts": {
                            "potential_distress": fit_res.metrics[
                                "class_count_potential_distress"
                            ],
                            "no_sufficient_distress_evidence": fit_res.metrics[
                                "class_count_no_sufficient_distress_evidence"
                            ],
                        },
                        "data_mode": fit_res.metrics["data_mode"],
                        "synthetic_test_data": fit_res.metrics["synthetic_test_data"],
                        "starting_model_version": fit_res.metrics[
                            "global_model_version"
                        ],
                    }
                )

        if failures:
            errors.append(f"{len(failures)} Flower client failure(s) were reported.")
        if len(results) < MIN_PARTICIPATING_CLIENTS:
            errors.append(
                f"Only {len(results)} client(s) participated; at least "
                f"{MIN_PARTICIPATING_CLIENTS} are required."
            )
        if len(results) != self.client_count:
            errors.append(
                f"Expected {self.client_count} client updates but received {len(results)}."
            )
        if len(used_client_ids) != len(results):
            errors.append("Client participation IDs are missing or duplicated.")

        round_record = {
            "round": server_round,
            "status": "failed" if errors else "validating",
            "participants": sorted(
                participants,
                key=lambda item: item["client_id"],
            ),
            "failure_count": len(failures),
            "validation_errors": errors,
            "starting_global_model_version": self.current_model_version,
            "starting_parameters_sha256": self.current_parameters_digest,
        }
        self.round_records.append(round_record)
        if errors:
            return None, {}

        ordered_results = sorted(
            results,
            key=lambda item: item[1].metrics["client_id"],
        )
        aggregated, aggregate_metrics = super().aggregate_fit(
            server_round,
            ordered_results,
            failures,
        )
        if aggregated is None:
            round_record["status"] = "failed"
            round_record["validation_errors"].append(
                "Flower FedAvg did not return aggregated parameters."
            )
            return None, {}

        try:
            aggregated_arrays = _parameter_arrays(parameters_to_ndarrays(aggregated))
        except (TypeError, ValueError) as exc:
            round_record["status"] = "failed"
            round_record["validation_errors"].append(
                f"Flower FedAvg returned invalid parameters: {exc}"
            )
            return None, {}

        self.current_parameters = aggregated_arrays
        self.current_parameters_digest = parameter_digest(aggregated_arrays)
        self.current_model_version = f"{GLOBAL_MODEL_PREFIX}{server_round:04d}"
        round_record.update(
            {
                "status": "success",
                "aggregated_global_model_version": self.current_model_version,
                "aggregated_parameters_sha256": self.current_parameters_digest,
                "eligible_training_examples": sum(
                    item["eligible_training_examples"] for item in participants
                ),
                "aggregation_strategy": FEDAVG_STRATEGY_NAME,
                "aggregation_weight": "eligible local training-example count",
                "flower_aggregate_metrics": aggregate_metrics,
            }
        )
        return aggregated, aggregate_metrics


def create_flower_client_app(
    config: VirtualStudentConfig,
    *,
    training_seed: int | None = None,
    data_mode: str = SYNTHETIC_TEST_DATA_MODE,
    feedback_db: str | Path = DEFAULT_FEEDBACK_PATH,
    review_manifest: str | Path | None = None,
    training_config: LocalTrainingConfig = DEFAULT_LOCAL_TRAINING_CONFIG,
) -> ClientApp:
    """Create a Flower ClientApp that materializes only its selected local client."""
    client_seed = config.seed if training_seed is None else training_seed
    if isinstance(client_seed, bool) or not isinstance(client_seed, int):
        raise TypeError("training_seed must be an integer.")
    partition_cache: dict[int, VirtualStudentClient] = {}

    def client_fn(context: Context) -> Client:
        partition_value = context.node_config.get(
            "partition-id",
            context.node_config.get("partition_id", context.node_id),
        )
        if isinstance(partition_value, bool) or not isinstance(partition_value, int):
            raise ValueError("Flower client Context has no integer partition ID.")
        if partition_value not in partition_cache:
            partition_cache[partition_value] = _partition_for_client(
                partition_value,
                config,
                data_mode,
                feedback_db,
                review_manifest,
            )
        return _LocalFusionHeadClient(
            partition_cache[partition_value],
            seed=client_seed,
            training_config=training_config,
        ).to_client()

    return ClientApp(client_fn=client_fn)


def create_local_flower_client(
    partition_id: int,
    config: VirtualStudentConfig,
    *,
    training_seed: int | None = None,
    data_mode: str = SYNTHETIC_TEST_DATA_MODE,
    feedback_db: str | Path = DEFAULT_FEEDBACK_PATH,
    review_manifest: str | Path | None = None,
    training_config: LocalTrainingConfig = DEFAULT_LOCAL_TRAINING_CONFIG,
) -> NumPyClient:
    """Materialize one local virtual-student partition for Flower client training."""
    client_seed = config.seed if training_seed is None else training_seed
    if isinstance(client_seed, bool) or not isinstance(client_seed, int):
        raise TypeError("training_seed must be an integer.")
    return _LocalFusionHeadClient(
        _partition_for_client(
            partition_id,
            config,
            data_mode,
            feedback_db,
            review_manifest,
        ),
        seed=client_seed,
        training_config=training_config,
    )


def _fusion_head_from_parameters(parameters: Sequence[np.ndarray]) -> FusionHeadModel:
    head = initialize_fusion_head()
    _set_parameters(head, parameters)
    return head


def _federated_metadata(
    strategy: ValidatingFlowerFedAvg,
    *,
    data_mode: str,
    run_config: Mapping[str, Any],
    initial_parameters: Sequence[np.ndarray],
) -> dict[str, Any]:
    final_parameters = strategy.current_parameters
    return {
        "metadata_version": FEDERATED_METADATA_VERSION,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_order": list(feature_order()),
        "input_width": FUSION_HEAD_INPUT_WIDTH,
        "model_architecture_version": FUSION_HEAD_MODEL_VERSION,
        "initial_global_model_version": f"{GLOBAL_MODEL_PREFIX}0000",
        "preprocessing_contract": FUSION_HEAD_PREPROCESSING_CONTRACT,
        "target_label_mapping": TARGET_LABEL_MAPPING,
        "round_number": strategy.rounds,
        "global_model_version": strategy.current_model_version,
        "aggregation_strategy": FEDAVG_STRATEGY_NAME,
        "aggregation_weight": "eligible local training-example count",
        "client_count": strategy.client_count,
        "minimum_successful_participants": MIN_PARTICIPATING_CLIENTS,
        "data_mode": data_mode,
        "synthetic_test_data": data_mode == SYNTHETIC_TEST_DATA_MODE,
        "training_configuration": {
            "seed": strategy.seed,
            "local_optimizer": strategy.training_config.to_dict(),
            **dict(run_config),
        },
        "initial_parameters": parameter_summary(initial_parameters),
        "final_parameters": parameter_summary(final_parameters),
        "rounds": strategy.round_records,
        "limitations": (
            "Offline federated experiment only. Synthetic data is infrastructure-test "
            "data, not real-world or clinical performance evidence. This artifact is "
            "not activated by live inference."
        ),
    }


def save_federated_fusion_head(
    model: FusionHeadModel,
    federated_metadata: Mapping[str, Any],
    model_path: str | Path,
) -> Path:
    """Persist a new standard fusion head plus a separately validated FL sidecar."""
    target = Path(model_path)
    federated_sidecar = target.with_suffix(".federated.json")
    if target.exists() or target.with_suffix(".metadata.json").exists() or federated_sidecar.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing federated model artifact at {target}."
        )
    if federated_metadata.get("metadata_version") != FEDERATED_METADATA_VERSION:
        raise ValueError("Federated metadata version is missing or incompatible.")
    if federated_metadata.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Federated metadata schema version is incompatible.")
    if federated_metadata.get("feature_order") != list(feature_order()):
        raise ValueError("Federated metadata feature order is incompatible.")
    if federated_metadata.get("input_width") != FUSION_HEAD_INPUT_WIDTH:
        raise ValueError("Federated metadata model input width is incompatible.")
    if federated_metadata.get("model_architecture_version") != FUSION_HEAD_MODEL_VERSION:
        raise ValueError("Federated model architecture version is incompatible.")
    if federated_metadata.get("preprocessing_contract") != FUSION_HEAD_PREPROCESSING_CONTRACT:
        raise ValueError("Federated preprocessing contract is incompatible.")
    if federated_metadata.get("target_label_mapping") != TARGET_LABEL_MAPPING:
        raise ValueError("Federated target-label mapping is incompatible.")
    round_number = federated_metadata.get("round_number")
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
        raise ValueError("Federated metadata has an invalid round number.")
    if federated_metadata.get("global_model_version") != f"{GLOBAL_MODEL_PREFIX}{round_number:04d}":
        raise ValueError("Federated global model version does not match its round number.")
    if federated_metadata.get("aggregation_strategy") != FEDAVG_STRATEGY_NAME:
        raise ValueError("Federated aggregation strategy metadata is incompatible.")
    if federated_metadata.get("minimum_successful_participants", 0) < MIN_PARTICIPATING_CLIENTS:
        raise ValueError("Federated metadata does not require three successful participants.")
    rounds = federated_metadata.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != round_number:
        raise ValueError("Federated round history does not match the final round number.")
    for expected_round, record in enumerate(rounds, start=1):
        if (
            not isinstance(record, dict)
            or record.get("round") != expected_round
            or record.get("status") != "success"
            or len(record.get("participants", [])) < MIN_PARTICIPATING_CLIENTS
        ):
            raise ValueError("Federated metadata contains an unsuccessful or incomplete round.")

    arrays = _parameter_arrays(
        [
            model.pipeline.named_steps["classifier"].coef_,
            model.pipeline.named_steps["classifier"].intercept_,
        ]
    )
    if parameter_digest(arrays) != federated_metadata["final_parameters"].get(
        "parameter_sha256"
    ):
        raise ValueError("Federated metadata does not match the fusion-head parameters.")
    save_fusion_head(model, target)
    federated_sidecar.write_text(
        json.dumps(dict(federated_metadata), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return target


def load_federated_fusion_head(model_path: str | Path) -> FederatedRunResult:
    """Load the existing fusion-head format and validate its FedAvg sidecar."""
    target = Path(model_path)
    model = load_fusion_head(target)
    sidecar = target.with_suffix(".federated.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Federated metadata sidecar is missing: {sidecar}.")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Federated metadata sidecar must contain a JSON object.")
    required = {
        "metadata_version",
        "schema_version",
        "feature_order",
        "input_width",
        "model_architecture_version",
        "initial_global_model_version",
        "preprocessing_contract",
        "target_label_mapping",
        "round_number",
        "global_model_version",
        "aggregation_strategy",
        "aggregation_weight",
        "client_count",
        "minimum_successful_participants",
        "data_mode",
        "synthetic_test_data",
        "training_configuration",
        "initial_parameters",
        "final_parameters",
        "rounds",
        "limitations",
    }
    if set(metadata) != required:
        raise ValueError("Federated metadata fields do not match the versioned contract.")
    _validate_federated_metadata(metadata)
    arrays = _parameter_arrays(
        [
            model.pipeline.named_steps["classifier"].coef_,
            model.pipeline.named_steps["classifier"].intercept_,
        ]
    )
    if parameter_digest(arrays) != metadata["final_parameters"].get("parameter_sha256"):
        raise ValueError("Loaded fusion-head parameters do not match federated metadata.")
    return FederatedRunResult(target, model, metadata)


def _validate_federated_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("metadata_version") != FEDERATED_METADATA_VERSION:
        raise ValueError("Federated metadata version is incompatible.")
    if metadata.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Federated metadata schema version is incompatible.")
    if metadata.get("feature_order") != list(feature_order()):
        raise ValueError("Federated metadata feature order is incompatible.")
    if metadata.get("input_width") != FUSION_HEAD_INPUT_WIDTH:
        raise ValueError("Federated metadata model input width is incompatible.")
    if metadata.get("model_architecture_version") != FUSION_HEAD_MODEL_VERSION:
        raise ValueError("Federated model architecture version is incompatible.")
    if metadata.get("initial_global_model_version") != f"{GLOBAL_MODEL_PREFIX}0000":
        raise ValueError("Federated initial global model version is incompatible.")
    if metadata.get("preprocessing_contract") != FUSION_HEAD_PREPROCESSING_CONTRACT:
        raise ValueError("Federated preprocessing contract is incompatible.")
    if metadata.get("target_label_mapping") != TARGET_LABEL_MAPPING:
        raise ValueError("Federated target-label mapping is incompatible.")
    round_number = metadata.get("round_number")
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
        raise ValueError("Federated metadata has an invalid round number.")
    if metadata.get("global_model_version") != f"{GLOBAL_MODEL_PREFIX}{round_number:04d}":
        raise ValueError("Federated global model version does not match its round number.")
    if metadata.get("aggregation_strategy") != FEDAVG_STRATEGY_NAME:
        raise ValueError("Federated aggregation strategy metadata is incompatible.")
    if metadata.get("aggregation_weight") != "eligible local training-example count":
        raise ValueError("Federated aggregation weighting metadata is incompatible.")
    client_count = metadata.get("client_count")
    if (
        isinstance(client_count, bool)
        or not isinstance(client_count, int)
        or not MIN_PARTICIPATING_CLIENTS <= client_count <= 5
    ):
        raise ValueError("Federated metadata has an invalid client count.")
    if metadata.get("minimum_successful_participants") != MIN_PARTICIPATING_CLIENTS:
        raise ValueError("Federated metadata has an incompatible participant minimum.")
    data_mode = metadata.get("data_mode")
    if data_mode not in {REVIEWED_DATA_MODE, SYNTHETIC_TEST_DATA_MODE}:
        raise ValueError("Federated metadata has an unsupported data mode.")
    if metadata.get("synthetic_test_data") != (data_mode == SYNTHETIC_TEST_DATA_MODE):
        raise ValueError("Federated synthetic-data marker is inconsistent.")
    if not isinstance(metadata.get("training_configuration"), dict):
        raise ValueError("Federated training configuration is missing.")
    for key in ("initial_parameters", "final_parameters"):
        summary = metadata.get(key)
        if (
            not isinstance(summary, dict)
            or summary.get("parameter_shapes") != [[1, FUSION_HEAD_INPUT_WIDTH], [1]]
            or not isinstance(summary.get("parameter_l2_norm"), (float, int))
            or not math.isfinite(summary["parameter_l2_norm"])
            or not isinstance(summary.get("parameter_sha256"), str)
        ):
            raise ValueError(f"Federated {key} summary is invalid.")
    rounds = metadata.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != round_number:
        raise ValueError("Federated round history does not match the final round number.")
    for expected_round, record in enumerate(rounds, start=1):
        if not isinstance(record, dict) or record.get("round") != expected_round:
            raise ValueError("Federated round history is malformed.")
        participants = record.get("participants")
        if (
            record.get("status") != "success"
            or not isinstance(participants, list)
            or len(participants) < MIN_PARTICIPATING_CLIENTS
            or record.get("aggregated_global_model_version")
            != f"{GLOBAL_MODEL_PREFIX}{expected_round:04d}"
        ):
            raise ValueError("Federated artifact includes an unsuccessful round.")
        participant_ids = [item.get("client_id") for item in participants if isinstance(item, dict)]
        if (
            len(participant_ids) != len(participants)
            or len(set(participant_ids)) != len(participant_ids)
            or any(not isinstance(client_id, str) for client_id in participant_ids)
        ):
            raise ValueError("Federated round participation IDs are invalid.")


def run_federated_experiment(
    *,
    config: VirtualStudentConfig = VirtualStudentConfig(),
    rounds: int = 2,
    seed: int | None = None,
    data_mode: str = SYNTHETIC_TEST_DATA_MODE,
    feedback_db: str | Path = DEFAULT_FEEDBACK_PATH,
    review_manifest: str | Path | None = None,
    output_directory: str | Path = "federated/output",
    training_config: LocalTrainingConfig = DEFAULT_LOCAL_TRAINING_CONFIG,
) -> FederatedRunResult:
    """Run actual Flower FedAvg rounds and persist an immutable global artifact."""
    training_seed = config.seed if seed is None else seed
    if isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise TypeError("seed must be an integer.")
    if data_mode == REVIEWED_DATA_MODE and review_manifest is None:
        raise ValueError("Reviewed-data mode requires a local review manifest path.")

    initial_parameters = initial_fusion_parameters()
    strategy = ValidatingFlowerFedAvg(
        config.client_count,
        rounds,
        training_seed,
        data_mode,
        training_config,
    )
    client_app = create_flower_client_app(
        config,
        training_seed=training_seed,
        data_mode=data_mode,
        feedback_db=feedback_db,
        review_manifest=review_manifest,
        training_config=training_config,
    )
    def server_fn(context: Context) -> ServerAppComponents:
        del context
        return ServerAppComponents(
            config=ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )

    server_app = ServerApp(server_fn=server_fn)

    from flwr.simulation import run_simulation

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=config.client_count,
        backend_name="ray",
        backend_config={
            "client_resources": {"num_cpus": 1},
            "init_args": {
                "num_cpus": config.client_count,
                "include_dashboard": False,
            },
        },
    )
    if len(strategy.round_records) != rounds or any(
        record.get("status") != "success" for record in strategy.round_records
    ):
        raise RuntimeError(
            "Flower simulation did not complete every round with at least three "
            "validated client updates; no global model artifact was saved."
        )

    head = _fusion_head_from_parameters(strategy.current_parameters)
    run_metadata = {
        "configured_rounds": rounds,
        "configured_clients": config.client_count,
        "seed": training_seed,
    }
    metadata = _federated_metadata(
        strategy,
        data_mode=data_mode,
        run_config=run_metadata,
        initial_parameters=initial_parameters,
    )
    versioned_path = (
        Path(output_directory)
        / f"global-model-r{rounds:04d}-{strategy.current_parameters_digest[:12]}.joblib"
    )
    save_federated_fusion_head(head, metadata, versioned_path)
    return FederatedRunResult(versioned_path, head, metadata)
