import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from feedback.store import FeedbackStore
from fusion.schema import FEATURE_SCHEMA_VERSION, FeatureRecord, feature_order
from training.distress_logistic import TARGET_LABELS, _record_to_feature_vector
from training.fusion_head import (
    FUSION_HEAD_INPUT_WIDTH,
    FUSION_HEAD_PREPROCESSING_CONTRACT,
    fine_tune_fusion_head,
    initialize_fusion_head,
    load_fusion_head,
    save_fusion_head,
)
from training.prepare_reviewed_feedback import main as prepare_reviewed_feedback_cli
from training.reviewed_feedback import (
    REVIEW_MANIFEST_VERSION,
    load_reviewed_feedback,
    write_reviewed_examples,
)


def _record(
    valid_features=(
        "face.expression_angry",
        "face.expression_fear",
        "behavior.movement_energy",
        "behavior.sudden_movement",
    )
):
    names = feature_order()
    valid_names = set(valid_features)
    values = {
        name: (float(index + 1) / 10 if name in valid_names else None)
        for index, name in enumerate(names)
    }
    validity = {name: name in valid_names for name in names}
    return FeatureRecord(
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=names,
        feature_values=values,
        feature_validity=validity,
        unavailable_reasons={
            name: "Unavailable in test record."
            for name in names
            if not validity[name]
        },
        modality_status={"face": "ok", "behavior": "ok", "audio": "waiting"},
        modality_messages={"face": "", "behavior": "", "audio": ""},
        provenance={"source": "live_monitoring"},
    )


def _submit(store, action="missed_event"):
    options = {
        "idempotency_key": f"test-{action}-{len(store.list_feedback())}",
        "record": _record(),
    }
    if action in {"confirm_alert", "reject_alert"}:
        options["alert_id"] = f"alert-{action}"
    submission = store.submit(action, **options)
    return next(
        row
        for row in store.list_feedback()
        if row["feedback_id"] == submission.feedback_id
    )


def _timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _manifest(feedback, *, label="potential_distress", **updates):
    submitted_at = _timestamp(feedback["submitted_at_utc"])
    window_end = submitted_at - timedelta(milliseconds=250)
    window_start = window_end - timedelta(seconds=1)
    modality_timestamps = {
        "face": (window_end - timedelta(milliseconds=100)).isoformat(),
        "behavior": (window_end - timedelta(milliseconds=200)).isoformat(),
    }
    item = {
        "manifest_version": REVIEW_MANIFEST_VERSION,
        "feedback_id": feedback["feedback_id"],
        "review_decision": "approved_for_training",
        "reviewer_reference": "reviewer-local-1",
        "training_label": label,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_window_start_utc": window_start.isoformat(),
        "event_window_end_utc": window_end.isoformat(),
        "modality_timestamps_utc": modality_timestamps,
        "evidence_context_reference": "local-event-42",
    }
    item.update(updates)
    return item


def _manifest_file(path, *items):
    path.write_text(
        "".join(json.dumps(item, allow_nan=False) + "\n" for item in items),
        encoding="utf-8",
    )
    return path


def _rewrite_summary(store, feedback, mutate):
    payload = json.loads(feedback["feature_summary_json"])
    mutate(payload)
    connection = sqlite3.connect(store.database_path)
    try:
        connection.execute(
            "UPDATE caretaker_feedback SET feature_summary_json = ? WHERE feedback_id = ?",
            (json.dumps(payload, allow_nan=True), feedback["feedback_id"]),
        )
        connection.commit()
    finally:
        connection.close()


def test_explicitly_reviewed_eligible_example_preserves_action_and_numerical_values(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store, "missed_event")
    manifest = _manifest(feedback, label="no_sufficient_distress_evidence")
    path = _manifest_file(tmp_path / "review.jsonl", manifest)

    result = load_reviewed_feedback(store, path)

    assert len(result.examples) == 1
    assert result.rejections == ()
    example = result.examples[0]
    assert example.label == "no_sufficient_distress_evidence"
    assert example.feedback_provenance["feedback_action"] == "missed_event"
    assert example.feedback_provenance["alert_id"] is None
    assert example.record.created_at_utc is None
    assert example.record.provenance["review_decision"] == "approved_for_training"
    exported = example.to_training_record()
    assert exported["feature_values"]["face.expression_angry"] == 0.1
    assert "raw_image" not in json.dumps(exported).lower()
    assert "raw_audio" not in json.dumps(exported).lower()


@pytest.mark.parametrize("action", ["confirm_alert", "reject_alert", "missed_event"])
def test_action_only_feedback_never_becomes_a_training_label(tmp_path, action):
    store = FeedbackStore(tmp_path / f"{action}.sqlite3")
    _submit(store, action)
    path = _manifest_file(tmp_path / f"{action}.jsonl")

    result = load_reviewed_feedback(store, path)

    assert not result.examples
    assert len(result.rejections) == 1
    assert "feedback action is not a training label" in result.rejections[0].reason


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("event_window_start_utc", None, "event_window_start_utc must be"),
        ("modality_timestamps_utc", None, "modality_timestamps_utc is required"),
    ],
)
def test_missing_window_or_freshness_context_is_rejected(tmp_path, field, value, reason):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    path = _manifest_file(
        tmp_path / "review.jsonl",
        _manifest(feedback, **{field: value}),
    )

    result = load_reviewed_feedback(store, path)

    assert not result.examples
    assert reason in result.rejections[0].reason


def test_stale_modality_timestamp_is_rejected(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    manifest = _manifest(feedback)
    window_end = _timestamp(manifest["event_window_end_utc"])
    manifest["event_window_start_utc"] = (
        window_end - timedelta(seconds=5)
    ).isoformat()
    manifest["modality_timestamps_utc"]["face"] = (
        window_end - timedelta(seconds=4)
    ).isoformat()

    result = load_reviewed_feedback(store, _manifest_file(tmp_path / "review.jsonl", manifest))

    assert not result.examples
    assert "freshness limit" in result.rejections[0].reason


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            lambda payload: payload["feature_values"].__setitem__(
                "face.expression_angry", "not-a-number"
            ),
            "finite numeric value",
        ),
        (
            lambda payload: payload["feature_values"].__setitem__(
                "face.expression_angry", float("nan")
            ),
            "non-finite",
        ),
        (
            lambda payload: payload.__setitem__("schema_version", "phase3-feature-schema-v1"),
            "Feature schema mismatch",
        ),
        (
            lambda payload: (
                payload.__setitem__(
                    "feature_values",
                    {name: None for name in feature_order()},
                ),
                payload.__setitem__(
                    "feature_validity",
                    {name: False for name in feature_order()},
                ),
            ),
            "Insufficient evidence",
        ),
    ],
)
def test_invalid_or_insufficient_feature_summaries_are_rejected(tmp_path, mutate, reason):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    _rewrite_summary(store, feedback, mutate)
    path = _manifest_file(tmp_path / "review.jsonl", _manifest(feedback))

    result = load_reviewed_feedback(store, path)

    assert not result.examples
    assert reason in result.rejections[0].reason


def test_manifest_label_must_be_explicit_and_ambiguous_review_is_rejected(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    manifest = _manifest(feedback)
    manifest["review_decision"] = "ambiguous"

    result = load_reviewed_feedback(
        store,
        _manifest_file(tmp_path / "review.jsonl", manifest),
    )

    assert not result.examples
    assert "not eligible for training" in result.rejections[0].reason


def test_canonical_order_has_36_features_and_interleaved_72_column_input(tmp_path):
    record = _record()
    vector = _record_to_feature_vector(record)

    assert len(feature_order()) == 36
    assert vector.shape == (1, 72)
    assert FUSION_HEAD_INPUT_WIDTH == 72
    assert vector[0, :4].tolist() == [0.1, 1.0, 0.2, 1.0]
    assert np.isnan(vector[0, 4])
    assert vector[0, 5] == 0.0


def test_fixed_preprocessing_contract_does_not_depend_on_client_distribution():
    client_a = initialize_fusion_head()
    client_b = initialize_fusion_head()
    matrix_a = np.zeros((2, FUSION_HEAD_INPUT_WIDTH), dtype=float)
    matrix_b = np.full((2, FUSION_HEAD_INPUT_WIDTH), 17.0, dtype=float)
    matrix_a[0, 0] = np.nan
    matrix_b[1, 35] = np.nan

    transformed_a = client_a.pipeline.named_steps["imputer"].transform(matrix_a)
    transformed_b = client_b.pipeline.named_steps["imputer"].transform(matrix_b)

    assert client_a.metadata["preprocessing_contract"] == FUSION_HEAD_PREPROCESSING_CONTRACT
    assert client_b.metadata["preprocessing_contract"] == FUSION_HEAD_PREPROCESSING_CONTRACT
    assert np.array_equal(
        client_a.pipeline.named_steps["imputer"].statistics_,
        client_b.pipeline.named_steps["imputer"].statistics_,
    )
    assert transformed_a.shape == transformed_b.shape == (2, 72)
    assert transformed_a[0, 0] == 0.0
    assert transformed_b[0, 0] == 17.0


def test_fine_tuning_uses_a_supplied_compatible_head_without_mutating_it(
    tmp_path, monkeypatch
):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    examples = []
    for index, label in enumerate(TARGET_LABELS):
        feedback = _submit(store, "missed_event")
        manifest = _manifest(feedback, label=label)
        manifest_path = _manifest_file(tmp_path / f"review-{index}.jsonl", manifest)
        result = load_reviewed_feedback(store, manifest_path)
        examples.extend(
            example
            for example in result.examples
            if example.feedback_id == feedback["feedback_id"]
        )
    assert len(examples) == 2

    starting = initialize_fusion_head()
    starting.pipeline.named_steps["classifier"].coef_[0, 0] = 0.25
    starting.pipeline.named_steps["classifier"].intercept_[0] = -0.1
    starting_path = save_fusion_head(starting, tmp_path / "starter.joblib")
    loaded_start = load_fusion_head(starting_path)
    supplied_weights = loaded_start.pipeline.named_steps["classifier"].coef_.copy()
    observed_initial_weights = []
    original_fit = LogisticRegression.fit

    def capture_initial_weights(classifier, *args, **kwargs):
        observed_initial_weights.append(classifier.coef_.copy())
        return original_fit(classifier, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", capture_initial_weights)
    trained = fine_tune_fusion_head(loaded_start, examples)

    assert np.array_equal(observed_initial_weights[0], supplied_weights)
    assert np.array_equal(
        loaded_start.pipeline.named_steps["classifier"].coef_,
        supplied_weights,
    )
    assert np.isfinite(trained.pipeline.named_steps["classifier"].coef_).all()


def test_save_load_round_trip_preserves_head_metadata_and_predictions(tmp_path):
    starter = initialize_fusion_head()
    starter.pipeline.named_steps["classifier"].coef_[0, 0] = 0.75
    starter.pipeline.named_steps["classifier"].intercept_[0] = 0.2
    path = save_fusion_head(starter, tmp_path / "head.joblib")

    loaded = load_fusion_head(path)

    assert loaded.metadata == starter.metadata
    assert loaded.metadata["schema_version"] == FEATURE_SCHEMA_VERSION
    assert loaded.metadata["feature_order"] == list(feature_order())
    assert loaded.metadata["input_width"] == 72
    assert loaded.metadata["target_label_mapping"] == {
        "potential_distress": 1,
        "no_sufficient_distress_evidence": 0,
    }
    assert np.array_equal(
        loaded.pipeline.predict_proba(_record_to_feature_vector(_record())),
        starter.pipeline.predict_proba(_record_to_feature_vector(_record())),
    )


def test_reviewed_export_contains_only_accepted_feature_records(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    manifest = _manifest(feedback)
    result = load_reviewed_feedback(
        store,
        _manifest_file(tmp_path / "review.jsonl", manifest),
    )
    output = write_reviewed_examples(result, tmp_path / "features.jsonl")
    exported = json.loads(output.read_text(encoding="utf-8"))

    assert exported["label"] == "potential_distress"
    assert exported["feedback_provenance"]["feedback_action"] == "missed_event"
    assert "raw_video" not in output.read_text(encoding="utf-8").lower()


def test_prepare_cli_reports_no_examples_and_does_not_create_output(tmp_path, capsys):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    _submit(store)
    manifest_path = _manifest_file(tmp_path / "review.jsonl")
    output_path = tmp_path / "out.jsonl"

    result = prepare_reviewed_feedback_cli(
        [
            "--feedback-db",
            str(store.database_path),
            "--review-manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 1
    assert not output_path.exists()
    assert "feedback action is not a training label" in capsys.readouterr().err


def test_duplicate_manifest_entries_are_rejected(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    manifest = _manifest(feedback)

    result = load_reviewed_feedback(
        store,
        _manifest_file(tmp_path / "review.jsonl", manifest, manifest),
    )

    assert not result.examples
    assert any("Duplicate review-manifest" in rejection.reason for rejection in result.rejections)


def test_manifest_rejects_unapproved_extra_fields_that_could_carry_media(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    manifest = _manifest(feedback)
    manifest["raw_audio"] = "not-allowed"

    result = load_reviewed_feedback(
        store,
        _manifest_file(tmp_path / "review.jsonl", manifest),
    )

    assert not result.examples
    assert any("unsupported fields" in rejection.reason for rejection in result.rejections)
