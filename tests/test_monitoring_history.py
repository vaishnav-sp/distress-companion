from types import SimpleNamespace

from monitoring_history import create_history_entry, inference_history_csv


def test_history_entry_keeps_score_semantics_and_csv_export_columns():
    inference = SimpleNamespace(
        status="potential_distress",
        predicted_label="potential_distress",
        model_score=0.8,
        potential_distress_score=0.7,
        available_modalities={"face": True, "behavior": True, "audio": False},
    )
    alert = SimpleNamespace(state="potential_distress", smoothed_score=0.65)

    entry = create_history_entry(
        inference,
        alert,
        result_key="live-1",
        timestamp="2026-10-15T10:00:00+00:00",
    )
    exported = inference_history_csv([entry])

    assert entry["predicted_class_probability"] == 0.8
    assert entry["potential_distress_probability"] == 0.7
    assert exported.splitlines()[0].split(",") == [
        "timestamp",
        "prediction",
        "predicted_class_probability",
        "potential_distress_probability",
        "smoothed_distress_score",
        "alert_state",
        "available_modalities",
        "inference_status",
    ]
    assert '"behavior, face"' in exported
    assert "live-1" not in exported
    assert "camera" not in exported.lower()
    assert "audio" not in exported.lower()
