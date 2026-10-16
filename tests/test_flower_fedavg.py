import json
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("flwr")

from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from federated.flower_fedavg import (
    FEDAVG_STRATEGY_NAME,
    LocalTrainingConfig,
    ValidatingFlowerFedAvg,
    create_local_flower_client,
    initial_fusion_parameters,
    load_federated_fusion_head,
    parameter_digest,
    run_federated_experiment,
)
from federated.virtual_students import (
    SYNTHETIC_TEST_DATA_MODE,
    VirtualStudentConfig,
    create_synthetic_clients,
)
from fusion.schema import feature_order
from training.fusion_head import FUSION_HEAD_INPUT_WIDTH


def _strategy(client_count=3, rounds=1, mode=SYNTHETIC_TEST_DATA_MODE):
    return ValidatingFlowerFedAvg(
        client_count=client_count,
        rounds=rounds,
        seed=17,
        data_mode=mode,
        training_config=LocalTrainingConfig(),
    )


def _fit_result(strategy, client_number, parameters, num_examples=4, **metric_updates):
    metrics = strategy._fit_config(1)
    metrics.update(
        {
            "client_id": f"virtual-student-{client_number:02d}",
            "class_count_potential_distress": num_examples // 2,
            "class_count_no_sufficient_distress_evidence": num_examples // 2,
            "data_mode": strategy.data_mode,
            "synthetic_test_data": strategy.data_mode == SYNTHETIC_TEST_DATA_MODE,
        }
    )
    metrics.update(metric_updates)
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(parameters),
        num_examples=num_examples,
        metrics=metrics,
    )


def _proxy(client_number):
    return Mock(spec=ClientProxy, cid=f"flower-node-{client_number}")


def test_flower_fedavg_computes_known_example_weighted_average():
    strategy = _strategy()
    updates = []
    for client_number, coefficient, intercept, count in (
        (1, 1.0, 2.0, 2),
        (2, 3.0, 4.0, 4),
        (3, 5.0, 6.0, 6),
    ):
        arrays = [
            np.full((1, FUSION_HEAD_INPUT_WIDTH), coefficient),
            np.asarray([intercept]),
        ]
        updates.append(
            (_proxy(client_number), _fit_result(strategy, client_number, arrays, count))
        )

    aggregated, metrics = strategy.aggregate_fit(1, updates, [])
    result = parameters_to_ndarrays(aggregated)

    assert np.allclose(result[0], 44.0 / 12.0)
    assert np.allclose(result[1], 56.0 / 12.0)
    assert metrics == {"eligible_examples": 12, "participating_clients": 3}
    assert strategy.round_records[0]["status"] == "success"
    assert strategy.round_records[0]["aggregation_strategy"] == FEDAVG_STRATEGY_NAME


@pytest.mark.parametrize(
    ("mutations", "message"),
    [
        ([np.zeros((1, 4)), np.zeros(1)], "shape"),
        (
            [np.full((1, FUSION_HEAD_INPUT_WIDTH), np.nan), np.zeros(1)],
            "non-finite",
        ),
    ],
)
def test_flower_rejects_invalid_parameter_shapes_and_nonfinite_values(mutations, message):
    strategy = _strategy()
    results = [
        (_proxy(1), _fit_result(strategy, 1, mutations)),
        (
            _proxy(2),
            _fit_result(strategy, 2, initial_fusion_parameters()),
        ),
        (
            _proxy(3),
            _fit_result(strategy, 3, initial_fusion_parameters()),
        ),
    ]

    aggregate, _ = strategy.aggregate_fit(1, results, [])

    assert aggregate is None
    assert strategy.round_records[0]["status"] == "failed"
    assert message in " ".join(strategy.round_records[0]["validation_errors"])


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("schema_version", "phase1-schema", "schema_version"),
        ("model_architecture_version", "phase4-other-head", "model_architecture_version"),
        ("feature_order_json", json.dumps(list(reversed(feature_order()))), "feature_order_json"),
        ("global_model_version", "phase4-global-r0999", "global_model_version"),
        ("protocol_round", 2, "protocol_round"),
    ],
)
def test_flower_rejects_incompatible_schema_feature_order_and_model_version(
    metric, value, message
):
    strategy = _strategy()
    results = [
        (
            _proxy(index),
            _fit_result(
                strategy,
                index,
                initial_fusion_parameters(),
                **{metric: value},
            ),
        )
        for index in range(1, 4)
    ]

    aggregate, _ = strategy.aggregate_fit(1, results, [])

    assert aggregate is None
    assert strategy.round_records[0]["status"] == "failed"
    assert message in " ".join(strategy.round_records[0]["validation_errors"])


def test_fewer_than_three_clients_cannot_complete_a_round():
    strategy = _strategy()
    results = [
        (_proxy(index), _fit_result(strategy, index, initial_fusion_parameters()))
        for index in (1, 2)
    ]

    aggregate, _ = strategy.aggregate_fit(1, results, [])

    assert aggregate is None
    assert strategy.round_records[0]["status"] == "failed"
    assert "at least 3" in " ".join(strategy.round_records[0]["validation_errors"])


def test_client_update_contains_only_parameters_and_aggregate_protocol_metadata():
    config = VirtualStudentConfig(client_count=3, seed=23)
    partitions = create_synthetic_clients(config)
    for index, partition in enumerate(partitions.clients):
        client = create_local_flower_client(index, config)
        local_ids = {
            example.sample_id for example in client._client.load_training_partition()
        }
        assert local_ids == {
            example.sample_id for example in partition.load_training_partition()
        }
        strategy = _strategy()
        strategy.seed = config.seed
        arrays, sample_count, metrics = client.fit(
            initial_fusion_parameters(),
            strategy._fit_config(1),
        )
        assert sample_count == len(local_ids)
        assert metrics["client_id"] == partition.client_id
        assert metrics["global_parameters_sha256"] == parameter_digest(
            initial_fusion_parameters()
        )
        assert len(arrays) == 2
        assert arrays[0].shape == (1, FUSION_HEAD_INPUT_WIDTH)
        assert arrays[1].shape == (1,)
        assert all(np.isfinite(array).all() for array in arrays)
        assert "class_count_potential_distress" in metrics
        assert "class_count_no_sufficient_distress_evidence" in metrics

        payload_text = repr(metrics)
        for sensitive_field in (
            "feedback_id",
            "event_window",
            "evidence_context",
            "raw_audio",
            "raw_video",
            "feature_values",
            "feature_validity",
        ):
            assert sensitive_field not in payload_text
        assert all(sample_id not in payload_text for sample_id in local_ids)


def test_real_flower_simulation_trains_clients_aggregates_and_persists_global_head(
    tmp_path,
):
    config = VirtualStudentConfig(client_count=3, seed=61)
    result = run_federated_experiment(
        config=config,
        rounds=2,
        seed=61,
        output_directory=tmp_path,
    )
    metadata = result.federated_metadata
    rounds = metadata["rounds"]

    assert result.artifact_path.is_file()
    assert result.artifact_path.with_suffix(".metadata.json").is_file()
    assert result.artifact_path.with_suffix(".federated.json").is_file()
    assert metadata["aggregation_strategy"] == FEDAVG_STRATEGY_NAME
    assert metadata["synthetic_test_data"] is True
    assert metadata["initial_parameters"]["parameter_l2_norm"] == 0.0
    assert metadata["final_parameters"]["parameter_l2_norm"] > 0.0
    assert (
        metadata["initial_parameters"]["parameter_sha256"]
        != metadata["final_parameters"]["parameter_sha256"]
    )
    assert [record["status"] for record in rounds] == ["success", "success"]
    assert [record["round"] for record in rounds] == [1, 2]
    for record in rounds:
        assert len(record["participants"]) == 3
        assert len({item["client_id"] for item in record["participants"]}) == 3
        assert all(item["eligible_training_examples"] > 0 for item in record["participants"])
        assert record["aggregation_weight"] == "eligible local training-example count"
    assert rounds[0]["starting_global_model_version"] == "phase4-global-r0000"
    assert rounds[1]["starting_global_model_version"] == "phase4-global-r0001"
    assert (
        rounds[1]["starting_parameters_sha256"]
        == rounds[0]["aggregated_parameters_sha256"]
    )
    assert metadata["feature_order"] == list(feature_order())
    assert metadata["input_width"] == 72
    assert metadata["initial_global_model_version"] == "phase4-global-r0000"
    assert metadata["training_configuration"]["seed"] == 61
    assert metadata["training_configuration"]["local_optimizer"] == (
        LocalTrainingConfig().to_dict()
    )

    loaded = load_federated_fusion_head(result.artifact_path)
    example = create_synthetic_clients(config).clients[0].load_training_partition()[0]
    input_matrix = np.asarray(example.features, dtype=float).reshape(1, -1)
    expected = result.model.pipeline.predict_proba(input_matrix)
    actual = loaded.model.pipeline.predict_proba(input_matrix)
    assert loaded.federated_metadata == metadata
    assert loaded.federated_metadata["global_model_version"] == "phase4-global-r0002"
    assert np.allclose(actual, expected)

    sidecar = result.artifact_path.with_suffix(".federated.json")
    malformed_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    malformed_metadata["feature_order"] = list(reversed(feature_order()))
    sidecar.write_text(json.dumps(malformed_metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="feature order is incompatible"):
        load_federated_fusion_head(result.artifact_path)
