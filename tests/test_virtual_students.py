import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from feedback.store import FeedbackStore
from federated.virtual_students import (
    REVIEWED_DATA_MODE,
    SYNTHETIC_TEST_DATA_MODE,
    SyntheticTrainingExample,
    VirtualStudentConfig,
    create_reviewed_clients,
    create_synthetic_clients,
)
from fusion.schema import FEATURE_SCHEMA_VERSION, FeatureRecord, feature_order
from training.distress_logistic import TARGET_LABELS
from training.fusion_head import initialize_fusion_head
from training.reviewed_feedback import REVIEW_MANIFEST_VERSION


def _reviewed_record(index):
    names = feature_order()
    valid_names = {
        "face.expression_angry",
        "face.expression_fear",
        "behavior.movement_energy",
        "behavior.sudden_movement",
    }
    values = {
        name: (index / 100.0 if name == "face.expression_angry" else 0.25)
        if name in valid_names
        else None
        for name in names
    }
    validity = {name: name in valid_names for name in names}
    created_at = datetime.now(timezone.utc)
    anchor_utc = created_at - timedelta(milliseconds=10 + index * 10)
    anchor_monotonic = 1000.0 + index
    source_timestamps_monotonic = {
        "face": anchor_monotonic - 0.2,
        "behavior": anchor_monotonic - 0.1,
    }
    source_timestamps_utc = {
        modality: (
            anchor_utc
            + timedelta(seconds=timestamp - anchor_monotonic)
        ).isoformat()
        for modality, timestamp in source_timestamps_monotonic.items()
    }
    return FeatureRecord(
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=names,
        feature_values=values,
        feature_validity=validity,
        unavailable_reasons={
            name: "Unavailable in virtual-client test data."
            for name in names
            if not validity[name]
        },
        modality_status={"face": "ok", "behavior": "ok", "audio": "waiting"},
        modality_messages={"face": "", "behavior": "", "audio": ""},
        provenance={
            "source": "live_rule_based_monitoring",
            "snapshot_id": f"student-window-{index}",
            "source_timestamp_clock": "monotonic",
            "source_timestamp_anchor_utc": anchor_utc.isoformat(),
            "source_timestamp_anchor_monotonic": anchor_monotonic,
            "source_timestamps_monotonic": source_timestamps_monotonic,
            "source_timestamps_utc": source_timestamps_utc,
        },
        created_at_utc=created_at.isoformat(),
    )


def _review_manifest(feedback, index, label, **updates):
    provenance = json.loads(feedback["feature_summary_json"])["provenance"]
    modality_timestamps = provenance["source_timestamps_utc"]
    source_times = [
        datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
        for timestamp in modality_timestamps.values()
    ]
    item = {
        "manifest_version": REVIEW_MANIFEST_VERSION,
        "feedback_id": feedback["feedback_id"],
        "snapshot_id": provenance["snapshot_id"],
        "review_decision": "approved_for_training",
        "reviewer_reference": "local-reviewer-1",
        "training_label": label,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_window_start_utc": min(source_times).isoformat(),
        "event_window_end_utc": max(source_times).isoformat(),
        "modality_timestamps_utc": modality_timestamps,
        "evidence_context_reference": f"event-{index}",
    }
    item.update(updates)
    return item


def _reviewed_store_and_manifest(tmp_path, examples_per_label=14):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    manifests = []
    index = 0
    for label in TARGET_LABELS:
        for _ in range(examples_per_label):
            submission = store.submit(
                "missed_event",
                idempotency_key=f"virtual-client-{index}",
                record=_reviewed_record(index),
            )
            feedback = next(
                row
                for row in store.list_feedback()
                if row["feedback_id"] == submission.feedback_id
            )
            manifests.append(_review_manifest(feedback, index, label))
            index += 1
    manifest_path = tmp_path / "review.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item, allow_nan=False) + "\n" for item in manifests),
        encoding="utf-8",
    )
    return store, manifest_path


@pytest.mark.parametrize("client_count", [3, 4, 5])
def test_synthetic_clients_have_stable_ids_coverage_and_separate_partitions(client_count):
    config = VirtualStudentConfig(client_count=client_count, seed=41)
    first = create_synthetic_clients(config, samples_per_client=50)
    second = create_synthetic_clients(config, samples_per_client=50)

    assert [client.client_id for client in first.clients] == [
        f"virtual-student-{index + 1:02d}" for index in range(client_count)
    ]
    train_ids = []
    means = []
    positive_proportions = []
    for client in first.clients:
        examples = client.load_training_partition()
        assert len(examples) == 50
        assert all(isinstance(example, SyntheticTrainingExample) for example in examples)
        assert all(example.data_mode == SYNTHETIC_TEST_DATA_MODE for example in examples)
        counts = client.data_quality().class_counts
        assert all(counts[label] > 0 for label in TARGET_LABELS)
        assert client.data_quality().synthetic_test_data is True
        assert client.data_quality().data_mode == SYNTHETIC_TEST_DATA_MODE
        assert client.data_quality().feature_validity_rate == 1.0
        assert all(
            flag in {0.0, 1.0}
            for example in examples
            for flag in example.features[1::2]
        )
        train_ids.extend(example.sample_id for example in examples)
        means.append(np.mean([example.features[0] for example in examples]))
        positive_proportions.append(counts[TARGET_LABELS[1]] / len(examples))
        assert client.fusion_head_contract == initialize_fusion_head().metadata

    evaluation_ids = [
        example.sample_id for example in first.evaluation_examples
    ]
    assert len(train_ids) == len(set(train_ids))
    assert set(train_ids).isdisjoint(evaluation_ids)
    assert len(evaluation_ids) == len(set(evaluation_ids))
    assert means == sorted(means)
    assert len(set(positive_proportions)) == client_count
    assert [
        (example.sample_id, example.features)
        for client in first.clients
        for example in client.load_training_partition()
    ] == [
        (example.sample_id, example.features)
        for client in second.clients
        for example in client.load_training_partition()
    ]
    assert [
        (example.sample_id, example.features)
        for example in first.evaluation_examples
    ] == [
        (example.sample_id, example.features)
        for example in second.evaluation_examples
    ]


def test_client_partition_loader_exposes_only_its_own_data():
    partitions = create_synthetic_clients(
        VirtualStudentConfig(client_count=3, seed=4),
        samples_per_client=12,
    )
    all_ids = {
        client.client_id: {
            example.sample_id for example in client.load_training_partition()
        }
        for client in partitions.clients
    }

    for client in partitions.clients:
        loaded = {example.sample_id for example in client.load_training_partition()}
        assert loaded == all_ids[client.client_id]
        assert all(
            sample_id.startswith(f"synthetic-client-{client.client_id[-2:]}-")
            for sample_id in loaded
        )
        for other_id, other_samples in all_ids.items():
            if other_id != client.client_id:
                assert loaded.isdisjoint(other_samples)


@pytest.mark.parametrize("client_count", [3, 4, 5])
def test_reviewed_partitions_are_deterministic_and_window_disjoint(
    tmp_path, client_count
):
    store, manifest = _reviewed_store_and_manifest(tmp_path)
    config = VirtualStudentConfig(client_count=client_count, seed=9)

    first = create_reviewed_clients(store, manifest, config)
    second = create_reviewed_clients(store, manifest, config)

    assert first.data_mode == REVIEWED_DATA_MODE
    assert [client.client_id for client in first.clients] == [
        client.client_id for client in second.clients
    ]
    identities = []
    training_windows = []
    feature_means = []
    class_distributions = []
    for first_client, second_client in zip(first.clients, second.clients):
        first_examples = first_client.load_training_partition()
        second_examples = second_client.load_training_partition()
        assert [example.feedback_id for example in first_examples] == [
            example.feedback_id for example in second_examples
        ]
        assert all(first_client.data_quality().class_counts[label] > 0 for label in TARGET_LABELS)
        feature_means.append(
            np.mean(
                [
                    example.record.feature_values["face.expression_angry"]
                    for example in first_examples
                ]
            )
        )
        class_distributions.append(
            tuple(first_client.data_quality().class_counts[label] for label in TARGET_LABELS)
        )
        identities.extend(example.feedback_id for example in first_examples)
        training_windows.extend(
            (
                example.feedback_provenance["evidence_context_reference"],
                example.feedback_provenance["event_window_start_utc"],
                example.feedback_provenance["event_window_end_utc"],
            )
            for example in first_examples
        )
    evaluation_windows = [
        (
            example.feedback_provenance["evidence_context_reference"],
            example.feedback_provenance["event_window_start_utc"],
            example.feedback_provenance["event_window_end_utc"],
        )
        for example in first.evaluation_examples
    ]

    assert len(identities) == len(set(identities))
    assert len(training_windows) == len(set(training_windows))
    assert set(training_windows).isdisjoint(evaluation_windows)
    assert len(evaluation_windows) == len(set(evaluation_windows))
    assert feature_means == sorted(feature_means)
    assert len(set(feature_means)) == client_count
    assert len(set(class_distributions)) > 1
    assert set(identities).isdisjoint(
        {example.feedback_id for example in second.evaluation_examples}
    )
    assert all(client.fusion_head_contract["model_version"] for client in first.clients)


def test_reviewed_mode_uses_only_eligible_loader_examples_and_reports_rejections(tmp_path):
    store, manifest_path = _reviewed_store_and_manifest(tmp_path, examples_per_label=8)
    ineligible = store.submit(
        "missed_event",
        idempotency_key="not-reviewed",
        record=_reviewed_record(999),
    )
    feedback = next(
        row for row in store.list_feedback() if row["feedback_id"] == ineligible.feedback_id
    )
    excluded = _review_manifest(
        feedback,
        999,
        TARGET_LABELS[0],
        review_decision="excluded",
    )
    with manifest_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(excluded, allow_nan=False) + "\n")

    partitions = create_reviewed_clients(
        store,
        manifest_path,
        VirtualStudentConfig(client_count=3, seed=11),
    )

    training_ids = {
        example.feedback_id
        for client in partitions.clients
        for example in client.load_training_partition()
    }
    evaluation_ids = {example.feedback_id for example in partitions.evaluation_examples}
    assert ineligible.feedback_id not in training_ids | evaluation_ids
    assert any(
        "not eligible for training" in rejection.reason
        for rejection in partitions.eligibility_rejections
    )


def test_reviewed_mode_rejects_duplicate_source_event_across_windows(tmp_path):
    store, manifest_path = _reviewed_store_and_manifest(tmp_path)
    items = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    items[1]["evidence_context_reference"] = items[0]["evidence_context_reference"]
    manifest_path.write_text(
        "".join(json.dumps(item, allow_nan=False) + "\n" for item in items),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate reviewed source event or window"):
        create_reviewed_clients(
            store,
            manifest_path,
            VirtualStudentConfig(client_count=3, seed=3),
        )


def test_reviewed_mode_fails_actionably_when_class_coverage_is_insufficient(tmp_path):
    store, manifest = _reviewed_store_and_manifest(
        tmp_path,
        examples_per_label=3,
    )

    with pytest.raises(ValueError, match="Insufficient eligible class coverage") as error:
        create_reviewed_clients(
            store,
            manifest,
            VirtualStudentConfig(client_count=3, seed=1),
        )

    assert "required" in str(error.value)
    assert "labels will not be inferred or fabricated" in str(error.value)


@pytest.mark.parametrize("client_count", [2, 6])
def test_client_count_outside_supported_range_is_rejected(client_count):
    with pytest.raises(ValueError, match="client_count must be between 3 and 5"):
        VirtualStudentConfig(client_count=client_count)


def test_synthetic_data_is_unambiguously_not_real_world_performance_data():
    partitions = create_synthetic_clients(
        VirtualStudentConfig(client_count=3, seed=2),
        samples_per_client=6,
    )

    assert partitions.data_mode == "synthetic_test"
    assert all(
        client.data_quality().synthetic_test_data
        for client in partitions.clients
    )
    assert all(
        isinstance(example, SyntheticTrainingExample)
        and len(example.features) == 72
        and all(np.isfinite(value) for value in example.features)
        for client in partitions.clients
        for example in client.load_training_partition()
    )
