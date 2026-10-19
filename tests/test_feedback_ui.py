from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from feedback.store import DEFAULT_FEEDBACK_PATH, FeedbackStore
from tests.test_reviewed_fusion_training import _record
from training.demo_feedback_reviews import (
    append_demo_feedback_review,
    load_demo_feedback_reviews,
    load_demo_training_dataset,
)
from training.reviewed_feedback import (
    feedback_review_block_reason,
    feedback_review_provenance_diagnostic,
    feedback_review_statuses,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rendered_text_in_order(node):
    result = []
    children = getattr(node, "children", None)
    if not isinstance(children, dict):
        return result
    for child in children.values():
        value = getattr(child, "value", None)
        if isinstance(value, str):
            result.append(value)
        label = getattr(child, "label", None)
        if isinstance(label, str):
            result.append(label)
        result.extend(_rendered_text_in_order(child))
    return result


def test_feedback_table_and_review_form_render_after_monitoring_is_stopped(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "feedback.sqlite3"
    store = FeedbackStore(database_path)
    store.submit(
        "missed_event",
        idempotency_key="stopped-monitoring-ui-test",
        record=_record(snapshot_id="stopped-monitoring-ui-snapshot"),
    )

    original_init = FeedbackStore.__init__

    def use_test_database(self, database_path=DEFAULT_FEEDBACK_PATH):
        target_path = (
            tmp_path / "feedback.sqlite3"
            if database_path == DEFAULT_FEEDBACK_PATH
            else database_path
        )
        original_init(self, target_path)

    monkeypatch.setattr(FeedbackStore, "__init__", use_test_database)
    monkeypatch.setenv(
        "DISTRESS_REVIEW_MANIFEST", str(tmp_path / "review_manifest_v2.jsonl")
    )
    monkeypatch.setenv(
        "DISTRESS_DEMO_REVIEW_FILE", str(tmp_path / "demo_reviews.jsonl")
    )
    monkeypatch.setenv(
        "DISTRESS_PERSONALIZATION_DIR", str(tmp_path / "personalized_models")
    )

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=90)

    assert not app.exception
    assert app.session_state["monitoring_manager"] is None
    rendered_text = _rendered_text_in_order(app.main)
    section_indices = [
        next(
            index
            for index, text in enumerate(rendered_text)
            if heading in text
        )
        for heading in (
            "Autism Distress Companion",
            "Monitoring controls",
            "Experimental Multimodal Distress-Related Evidence Score",
            "Multimodal Distress Alert",
            "Feedback History and Review",
            "Review a Feedback Row",
            "Train / Personalize",
            "Monitoring and Sensors",
            "Audio-feature test",
        )
    ]
    assert section_indices == sorted(section_indices)
    assert sum(button.label == "Start monitoring" for button in app.button) == 1
    assert sum(button.label == "Stop monitoring" for button in app.button) == 1
    assert any(button.label == "Save Review" for button in app.button)
    assert any("Submitted (UTC)" in str(table.value) for table in app.table)

    missed_event = next(
        button
        for button in app.button
        if button.key == "feedback_missed_event"
    )
    missed_event.click().run(timeout=90)

    assert not app.exception
    assert len(FeedbackStore(database_path).list_feedback()) == 2
    feedback_selector = next(
        widget for widget in app.selectbox if widget.label == "Feedback row"
    )
    assert len(feedback_selector.options) == 2


def test_review_workflow_saves_demo_label_without_changing_training_eligibility(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "feedback.sqlite3"
    store = FeedbackStore(database_path)
    incomplete = store.submit(
        "missed_event",
        idempotency_key="ui-incomplete",
        record=_record(snapshot_id="ui-incomplete-snapshot"),
        provenance_available=False,
    )
    first_reviewable = store.submit(
        "missed_event",
        idempotency_key="ui-reviewable-first",
        record=_record(snapshot_id="ui-reviewable-first-snapshot"),
    )
    second_reviewable = store.submit(
        "missed_event",
        idempotency_key="ui-reviewable-second",
        record=_record(snapshot_id="ui-reviewable-second-snapshot"),
    )

    original_init = FeedbackStore.__init__

    def use_test_database(self, path=DEFAULT_FEEDBACK_PATH):
        original_init(
            self,
            database_path if path == DEFAULT_FEEDBACK_PATH else path,
        )

    manifest_path = tmp_path / "private" / "review_manifest_v2.jsonl"
    demo_review_path = tmp_path / "private" / "demo_reviews.jsonl"
    monkeypatch.setattr(FeedbackStore, "__init__", use_test_database)
    monkeypatch.setenv("DISTRESS_REVIEW_MANIFEST", str(manifest_path))
    monkeypatch.setenv(
        "DISTRESS_DEMO_REVIEW_FILE", str(demo_review_path)
    )
    monkeypatch.setenv(
        "DISTRESS_PERSONALIZATION_DIR", str(tmp_path / "personalized_models")
    )

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=90)

    assert not app.exception
    assert app.session_state["monitoring_manager"] is None
    rendered_text = _rendered_text_in_order(app.main)
    headings = (
        "Experimental Multimodal Distress-Related Evidence Score",
        "Multimodal Distress Alert",
        "Feedback History and Review",
        "Review a Feedback Row",
        "Train / Personalize",
        "Monitoring and Sensors",
        "Simulated caretaker workflow demo",
        "Audio-feature test",
    )
    heading_positions = [
        next(index for index, text in enumerate(rendered_text) if heading in text)
        for heading in headings
    ]
    assert heading_positions == sorted(heading_positions)
    for heading in headings[2:5]:
        assert sum(text == heading for text in rendered_text) == 1

    selector = next(
        widget for widget in app.selectbox if widget.label == "Feedback row"
    )
    assert selector.value == second_reviewable.feedback_id
    assert len(selector.options) == 3
    history_table = next(
        table
        for table in app.table
        if "Persisted provenance" in table.value.columns
    )
    assert history_table.value.shape[0] == 3
    assert any(
        value.startswith("Incomplete:")
        for value in history_table.value["Persisted provenance"]
    )
    complete_diagnostic = next(
        table
        for table in app.table
        if "Required category" in table.value.columns
    )
    assert set(complete_diagnostic.value["Status"]) == {"Present"}
    assert feedback_review_block_reason(
        next(
            row
            for row in store.list_feedback()
            if row["feedback_id"] == incomplete.feedback_id
        ),
        incomplete.feedback_id,
    )

    selector.select(incomplete.feedback_id).run(timeout=90)
    submit_button = next(item for item in app.button if item.label == "Save Review")
    assert not submit_button.disabled
    diagnostic_table = next(
        table
        for table in app.table
        if "Required category" in table.value.columns
    )
    assert "Missing/invalid" in set(diagnostic_table.value["Status"])
    assert not manifest_path.exists()

    next(
        widget for widget in app.selectbox if widget.label == "Demonstration label"
    ).select("potential_distress")
    next(item for item in app.button if item.label == "Save Review").click().run(
        timeout=90
    )
    assert not app.exception
    demo_reviews = load_demo_feedback_reviews(demo_review_path)
    assert demo_reviews[incomplete.feedback_id].training_label == "potential_distress"
    assert not manifest_path.exists()
    refreshed_history = next(
        table
        for table in app.table
        if "Review status" in table.value.columns
    ).value
    reviewed_row = refreshed_history.loc[
        refreshed_history["Training label"] == "potential_distress"
    ].iloc[0]
    assert reviewed_row["Review status"] == "Reviewed"
    assert reviewed_row["Training eligibility"].startswith(
        "Not eligible for demo training:"
    )
    assert reviewed_row["Persisted provenance"].startswith("Incomplete:")
    assert any(
        "not eligible for training" in item.value
        for item in app.success
    )
    assert next(item for item in app.button if item.label == "Save Review").disabled

    with pytest.raises(ValueError, match="already exists"):
        append_demo_feedback_review(
            store,
            demo_review_path,
            incomplete.feedback_id,
            "no_sufficient_distress_evidence",
        )
    assert len(demo_review_path.read_text(encoding="utf-8").splitlines()) == 1
    assert not manifest_path.exists()

    selector = next(
        widget for widget in app.selectbox if widget.label == "Feedback row"
    )
    selector.select(second_reviewable.feedback_id).run(timeout=90)
    assert not any(
        widget.label
        in {"Opaque reviewer reference", "Opaque evidence-context reference"}
        for widget in app.text_input
    )
    assert not next(
        item for item in app.button if item.label == "Save Review"
    ).disabled
    assert feedback_review_provenance_diagnostic(
        next(
            row
            for row in store.list_feedback()
            if row["feedback_id"] == second_reviewable.feedback_id
        ),
        second_reviewable.feedback_id,
    )["Timestamp consistency"] == "Present"
    next(
        widget for widget in app.selectbox if widget.label == "Demonstration label"
    ).select("no_sufficient_distress_evidence")
    next(item for item in app.button if item.label == "Save Review").click().run(
        timeout=90
    )
    assert not app.exception
    refreshed_history = next(
        table
        for table in app.table
        if "Review status" in table.value.columns
    ).value
    eligible_row = refreshed_history.loc[
        refreshed_history["Training label"] == "no_sufficient_distress_evidence"
    ].iloc[0]
    assert eligible_row["Review status"] == "Reviewed"
    assert (
        eligible_row["Training eligibility"]
        == "Eligible for demo training only (manual label; unverified)"
    )


def test_demo_training_dataset_validates_provenance_and_feature_sufficiency(
    tmp_path,
):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    valid = store.submit(
        "missed_event",
        idempotency_key="demo-dataset-valid",
        record=_record(snapshot_id="demo-dataset-valid-snapshot"),
    )
    incomplete = store.submit(
        "missed_event",
        idempotency_key="demo-dataset-incomplete",
        record=_record(snapshot_id="demo-dataset-incomplete-snapshot"),
        provenance_available=False,
    )
    insufficient = store.submit(
        "missed_event",
        idempotency_key="demo-dataset-insufficient",
        record=_record(
            snapshot_id="demo-dataset-insufficient-snapshot",
            valid_features=(
                "face.expression_angry",
                "behavior.movement_energy",
                "behavior.sudden_movement",
            ),
        ),
    )
    review_path = tmp_path / "demo_reviews.jsonl"
    append_demo_feedback_review(
        store, review_path, valid.feedback_id, "potential_distress"
    )
    append_demo_feedback_review(
        store, review_path, incomplete.feedback_id, "potential_distress"
    )
    append_demo_feedback_review(
        store, review_path, insufficient.feedback_id, "no_sufficient_distress_evidence"
    )

    dataset = load_demo_training_dataset(store, review_path)

    assert [example.feedback_id for example in dataset.examples] == [
        valid.feedback_id
    ]
    assert dataset.examples[0].snapshot_id == "demo-dataset-valid-snapshot"
    assert dataset.examples[0].training_label == "potential_distress"
    assert "provenance flag" in dataset.ineligible_reasons[incomplete.feedback_id]
    assert "Only 3 valid features" in dataset.ineligible_reasons[
        insufficient.feedback_id
    ]


def test_demo_training_dataset_excludes_duplicate_snapshots(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.sqlite3")
    first = store.submit(
        "missed_event",
        idempotency_key="demo-duplicate-snapshot-first",
        record=_record(snapshot_id="shared-demo-snapshot"),
    )
    second = store.submit(
        "missed_event",
        idempotency_key="demo-duplicate-snapshot-second",
        record=_record(snapshot_id="shared-demo-snapshot"),
    )
    review_path = tmp_path / "demo_reviews.jsonl"
    for feedback_id in (first.feedback_id, second.feedback_id):
        append_demo_feedback_review(
            store, review_path, feedback_id, "potential_distress"
        )

    dataset = load_demo_training_dataset(store, review_path)

    assert dataset.examples == ()
    assert len(dataset.ineligible_reasons) == 2
    assert all(
        "duplicate snapshot" in reason
        for reason in dataset.ineligible_reasons.values()
    )
