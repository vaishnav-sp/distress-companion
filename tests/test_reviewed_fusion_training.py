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
from training.personalization_demo import (
    run_reviewed_personalization,
    run_synthetic_personalization_demo,
)
from training.prepare_reviewed_feedback import main as prepare_reviewed_feedback_cli
from training.reviewed_feedback import (
    REVIEW_MANIFEST_VERSION,
    append_review_entry,
    feedback_review_block_reason,
    feedback_review_statuses,
    load_reviewed_feedback,
    write_reviewed_examples,
)


def _record(
    snapshot_id="review-test-snapshot",
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
    created_at = datetime.now(timezone.utc)
    anchor_utc = created_at - timedelta(milliseconds=10)
    anchor_monotonic = 100.0
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
            name: "Unavailable in test record."
            for name in names
            if not validity[name]
        },
        modality_status={"face": "ok", "behavior": "ok", "audio": "waiting"},
        modality_messages={"face": "", "behavior": "", "audio": ""},
        provenance={
            "source": "live_rule_based_monitoring",
            "snapshot_id": snapshot_id,
            "source_timestamp_clock": "monotonic",
            "source_timestamp_anchor_utc": anchor_utc.isoformat(),
            "source_timestamp_anchor_monotonic": anchor_monotonic,
            "source_timestamps_monotonic": source_timestamps_monotonic,
            "source_timestamps_utc": source_timestamps_utc,
        },
        created_at_utc=created_at.isoformat(),
    )


def _submit(store, action="missed_event"):
    idempotency_key = f"test-{action}-{len(store.list_feedback())}"
    options = {
        "idempotency_key": idempotency_key,
        "record": _record(snapshot_id=idempotency_key),
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
    summary = json.loads(feedback["feature_summary_json"])
    provenance = summary["provenance"]
    modality_timestamps = provenance["source_timestamps_utc"]
    source_times = [_timestamp(value) for value in modality_timestamps.values()]
    item = {
        "manifest_version": REVIEW_MANIFEST_VERSION,
        "feedback_id": feedback["feedback_id"],
        "snapshot_id": provenance["snapshot_id"],
        "review_decision": "approved_for_training",
        "reviewer_reference": "reviewer-local-1",
        "training_label": label,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "event_window_start_utc": min(source_times).isoformat(),
        "event_window_end_utc": max(source_times).isoformat(),
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
    assert example.record.created_at_utc == json.loads(
        feedback["feature_summary_json"]
    )["created_at_utc"]
    assert example.feedback_provenance["snapshot_id"] == manifest["snapshot_id"]
    assert example.record.provenance["review_decision"] == "approved_for_training"
    exported = example.to_training_record()
    assert exported["feature_values"]["face.expression_angry"] == 0.1
    assert "raw_image" not in json.dumps(exported).lower()
    assert "raw_audio" not in json.dumps(exported).lower()


@pytest.mark.parametrize(
    "action",
    ["confirm_alert", "reject_alert", "missed_event"],
)
def test_each_feedback_action_keeps_its_explicitly_reviewed_snapshot(
    tmp_path, action
):
    store = FeedbackStore(tmp_path / f"{action}.sqlite3")
    feedback = _submit(store, action)
    manifest = _manifest(feedback, label="potential_distress")

    result = load_reviewed_feedback(
        store,
        _manifest_file(tmp_path / f"{action}.jsonl", manifest),
    )

    assert len(result.examples) == 1
    example = result.examples[0]
    assert example.feedback_id == feedback["feedback_id"]
    assert example.feedback_provenance["feedback_action"] == action
    assert example.feedback_provenance["snapshot_id"] == manifest["snapshot_id"]
    assert example.feedback_provenance["event_window_start_utc"] == (
        manifest["event_window_start_utc"]
    )


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
    status = feedback_review_statuses(store, path)[feedback["feedback_id"]]
    assert status.training_eligibility == "Not eligible"


def test_mismatched_modality_timestamp_is_rejected(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    manifest = _manifest(feedback)
    manifest["modality_timestamps_utc"]["face"] = (
        _timestamp(manifest["modality_timestamps_utc"]["face"])
        + timedelta(milliseconds=1)
    ).isoformat()

    result = load_reviewed_feedback(store, _manifest_file(tmp_path / "review.jsonl", manifest))

    assert not result.examples
    assert "does not match the stored feature snapshot" in result.rejections[0].reason


def test_reviewed_event_window_must_exactly_span_persisted_source_times(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    manifest = _manifest(feedback)
    manifest["event_window_start_utc"] = (
        _timestamp(manifest["event_window_start_utc"]) - timedelta(milliseconds=1)
    ).isoformat()

    result = load_reviewed_feedback(store, _manifest_file(tmp_path / "review.jsonl", manifest))

    assert not result.examples
    assert "exactly span the persisted modality timestamps" in result.rejections[0].reason


def test_legacy_feedback_remains_readable_but_is_not_training_eligible(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    manifest = _manifest(feedback)

    _rewrite_summary(
        store,
        feedback,
        lambda payload: (
            payload.pop("created_at_utc"),
            payload.__setitem__("provenance", {"source": "live_monitoring"}),
        ),
    )
    legacy_row = store.list_feedback()[0]
    result = load_reviewed_feedback(
        store,
        _manifest_file(tmp_path / "review.jsonl", manifest),
    )

    assert legacy_row["feedback_id"] == feedback["feedback_id"]
    assert not result.examples
    assert "provenance is unavailable" in result.rejections[0].reason


def test_legacy_review_manifest_version_is_not_eligible(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    manifest = _manifest(feedback)
    manifest["manifest_version"] = "phase4-reviewed-feedback-manifest-v1"

    result = load_reviewed_feedback(
        store,
        _manifest_file(tmp_path / "review.jsonl", manifest),
    )

    assert not result.examples
    assert "manifest-v2" in result.rejections[0].reason


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


def test_feedback_history_reads_do_not_create_or_modify_local_data(tmp_path):
    database_path = tmp_path / "private" / "feedback.sqlite3"
    store = FeedbackStore(database_path)

    assert store.list_feedback() == []
    assert feedback_review_statuses(store, tmp_path / "review.jsonl") == {}
    assert not database_path.parent.exists()

    feedback = _submit(store)
    before = database_path.read_bytes()
    statuses = feedback_review_statuses(store, tmp_path / "review.jsonl")

    assert statuses[feedback["feedback_id"]].status == "Awaiting review"
    assert database_path.read_bytes() == before
    assert not (tmp_path / "review.jsonl").exists()


def test_review_form_provenance_gate_accepts_complete_row_and_matches_row_id(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)

    assert feedback_review_block_reason(feedback, feedback["feedback_id"]) is None
    assert feedback_review_block_reason(feedback, "different-row") is not None


@pytest.mark.parametrize(
    ("provenance_field", "value", "expected_reason"),
    [
        ("provenance_available", False, "provenance flag"),
        ("snapshot_id", None, "snapshot ID"),
        ("event_reference_type", "unavailable", "event reference"),
        ("source_timestamp_clock", "wall_clock", "timestamp clock"),
        ("source_timestamps_monotonic", {}, "timestamp coverage"),
        ("source_timestamp_anchor_utc", None, "UTC clock anchor"),
        ("source_timestamp_anchor_monotonic", None, "monotonic clock anchor"),
    ],
)
def test_review_form_provenance_gate_explains_incomplete_metadata(
    tmp_path, provenance_field, value, expected_reason
):
    store = FeedbackStore(tmp_path / f"{provenance_field}.sqlite3")
    feedback = _submit(store)

    def mutate(summary):
        summary["provenance"][provenance_field] = value

    _rewrite_summary(store, feedback, mutate)
    updated_row = next(
        row
        for row in store.list_feedback()
        if row["feedback_id"] == feedback["feedback_id"]
    )

    reason = feedback_review_block_reason(
        updated_row, feedback["feedback_id"]
    )

    assert reason is not None
    assert expected_reason in reason


def test_review_form_provenance_gate_checks_schema_modality_and_creation_time(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)

    def set_schema(summary):
        summary["schema_version"] = "unsupported-schema"

    _rewrite_summary(store, feedback, set_schema)
    row = next(
        item
        for item in store.list_feedback()
        if item["feedback_id"] == feedback["feedback_id"]
    )
    assert "feature-schema version" in feedback_review_block_reason(
        row, feedback["feedback_id"]
    )

    feedback = _submit(store)

    def remove_modality_timestamp(summary):
        summary["provenance"]["source_timestamps_utc"].pop("face")

    _rewrite_summary(store, feedback, remove_modality_timestamp)
    row = next(
        item
        for item in store.list_feedback()
        if item["feedback_id"] == feedback["feedback_id"]
    )
    assert "timestamp coverage" in feedback_review_block_reason(
        row, feedback["feedback_id"]
    )

    feedback = _submit(store)

    def remove_creation_time(summary):
        summary["created_at_utc"] = None

    _rewrite_summary(store, feedback, remove_creation_time)
    row = next(
        item
        for item in store.list_feedback()
        if item["feedback_id"] == feedback["feedback_id"]
    )
    assert "creation time" in feedback_review_block_reason(
        row, feedback["feedback_id"]
    )


def test_review_submission_rejects_incomplete_provenance_for_every_decision(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    entry = _manifest(feedback)
    entry["review_decision"] = "excluded"

    def remove_provenance(summary):
        summary["provenance"]["provenance_available"] = False

    _rewrite_summary(store, feedback, remove_provenance)
    manifest_path = tmp_path / "review.jsonl"

    with pytest.raises(ValueError, match="provenance flag"):
        append_review_entry(store, manifest_path, entry)
    assert not manifest_path.exists()


def test_explicit_review_append_is_v2_exact_snapshot_and_duplicate_safe(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    manifest_path = tmp_path / "private" / "review.jsonl"
    entry = _manifest(feedback)

    assert append_review_entry(store, manifest_path, entry) == manifest_path
    saved_content = manifest_path.read_text(encoding="utf-8")
    loaded = load_reviewed_feedback(store, manifest_path)

    assert [example.feedback_id for example in loaded.examples] == [
        feedback["feedback_id"]
    ]
    assert loaded.examples[0].feedback_provenance["snapshot_id"] == entry["snapshot_id"]
    assert feedback_review_statuses(store, manifest_path)[
        feedback["feedback_id"]
    ].status == "Approved for training"
    with pytest.raises(ValueError, match="already exists"):
        append_review_entry(store, manifest_path, entry)
    assert manifest_path.read_text(encoding="utf-8") == saved_content


def test_feedback_action_without_review_label_is_awaiting_not_training(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store, "confirm_alert")
    manifest_path = _manifest_file(tmp_path / "review.jsonl")

    result = load_reviewed_feedback(store, manifest_path)

    assert not result.examples
    assert any("action is not a training label" in item.reason for item in result.rejections)
    assert feedback_review_statuses(store, manifest_path)[
        feedback["feedback_id"]
    ].status == "Awaiting review"


def test_invalid_persisted_review_is_shown_as_ineligible(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    entry = _manifest(feedback)
    entry["training_label"] = "confirm_alert"
    manifest_path = _manifest_file(tmp_path / "review.jsonl", entry)

    status = feedback_review_statuses(store, manifest_path)[feedback["feedback_id"]]

    assert status.status == "Ineligible"
    assert status.training_eligibility == "Not eligible"


@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [("excluded", "Excluded"), ("ambiguous", "Ambiguous")],
)
def test_excluded_and_ambiguous_reviews_never_become_training_examples(
    tmp_path, decision, expected_status
):
    store = FeedbackStore(tmp_path / f"{decision}.sqlite3")
    feedback = _submit(store)
    manifest_path = tmp_path / f"{decision}.jsonl"
    entry = _manifest(feedback)
    entry["review_decision"] = decision

    append_review_entry(store, manifest_path, entry)
    loaded = load_reviewed_feedback(store, manifest_path)

    assert not loaded.examples
    status = feedback_review_statuses(store, manifest_path)[feedback["feedback_id"]]
    assert status.status == expected_status
    assert status.training_eligibility == "Not eligible"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_label", "confirm_alert"),
        ("snapshot_id", "a-different-snapshot"),
    ],
)
def test_review_submission_rejects_action_labels_and_snapshot_rebinding(
    tmp_path, field, value
):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    entry = _manifest(feedback)
    entry[field] = value
    manifest_path = tmp_path / "review.jsonl"

    with pytest.raises(ValueError):
        append_review_entry(store, manifest_path, entry)
    assert not manifest_path.exists()


def test_reviewed_training_blocks_insufficient_class_diversity(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    feedback = _submit(store)
    manifest_path = _manifest_file(
        tmp_path / "review.jsonl",
        _manifest(feedback, label="potential_distress"),
    )
    output_directory = tmp_path / "personalized"

    with pytest.raises(ValueError, match="at least 4 per label"):
        run_reviewed_personalization(store, manifest_path, output_directory)
    assert not output_directory.exists()


def test_reviewed_personalization_uses_only_explicit_reviews_and_versions_artifacts(
    tmp_path,
):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    manifest_path = tmp_path / "review.jsonl"
    for index in range(8):
        feedback = _submit(store)
        label = (
            "potential_distress"
            if index < 4
            else "no_sufficient_distress_evidence"
        )
        entry = _manifest(
            feedback,
            label=label,
            evidence_context_reference=f"event-{index}",
        )
        append_review_entry(store, manifest_path, entry)

    output_directory = tmp_path / "personalized"
    first = run_reviewed_personalization(store, manifest_path, output_directory)
    second = run_reviewed_personalization(store, manifest_path, output_directory)

    assert first.data_mode == "reviewed"
    assert first.sample_count == 8
    assert first.class_counts == {
        "potential_distress": 4,
        "no_sufficient_distress_evidence": 4,
    }
    assert first.held_out_sample_count == 2
    assert first.artifact_path != second.artifact_path
    assert first.artifact_path.is_file() and second.artifact_path.is_file()
    assert first.run_metadata_path.is_file() and second.run_metadata_path.is_file()
    metadata = json.loads(first.run_metadata_path.read_text(encoding="utf-8"))
    assert metadata["data_mode"] == "reviewed"
    assert metadata["live_model_activated"] is False


def test_synthetic_personalization_is_deterministic_versioned_and_isolated(tmp_path):
    output_directory = tmp_path / "private-models"

    first = run_synthetic_personalization_demo(output_directory)
    second = run_synthetic_personalization_demo(output_directory)

    assert first.data_mode == "DEMO/SYNTHETIC ONLY"
    assert first.artifact_path != second.artifact_path
    assert first.artifact_path.is_file() and second.artifact_path.is_file()
    assert first.run_metadata_path.is_file() and second.run_metadata_path.is_file()
    assert first.artifact_path.parent.name == "synthetic_demo"
    assert first.class_counts == second.class_counts
    assert first.after_metrics == second.after_metrics
    metadata = json.loads(first.run_metadata_path.read_text(encoding="utf-8"))
    assert metadata["data_mode"] == "DEMO/SYNTHETIC ONLY"
    assert metadata["live_model_activated"] is False
