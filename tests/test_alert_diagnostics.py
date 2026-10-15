from types import SimpleNamespace
import json
import time

import pytest

from alert_management import AlertManager, AlertPolicy
from monitoring import MonitoringManager
from monitoring_history import (
    ALERT_DIAGNOSTIC_FIELDS,
    alert_diagnostics_csv,
)


def _inference(
    status="potential_distress",
    score=0.99,
    *,
    valid_feature_count=5,
    available_modalities=None,
    reason=None,
):
    return SimpleNamespace(
        status=status,
        predicted_label=status,
        potential_distress_score=score,
        potential_distress_probability=score,
        valid_feature_count=valid_feature_count,
        available_modalities=available_modalities
        or {"face": True, "behavior": True, "audio": False},
        reason=reason,
    )


def _manager_for_results(results):
    pending_results = list(results)

    def process(snapshot):
        result = pending_results.pop(0)
        source_timestamp = result["source_timestamp"]
        return {
            "inference": result["inference"],
            "feature_record": None,
            "source_timestamp": source_timestamp,
            "modality_source_timestamps": result["modality_source_timestamps"],
            "evidence_sufficient": result["evidence_sufficient"],
            "evidence_sufficiency_reason": result["evidence_sufficiency_reason"],
            "result_key": result["result_key"],
            "updated_at_utc": "2026-10-15T06:25:48.000000+00:00",
        }

    manager = MonitoringManager(
        inference_processor=process,
        alert_manager=AlertManager(
            AlertPolicy(
                raw_probability_threshold=0.75,
                smoothing_window=5,
                max_result_age_seconds=3.0,
            )
        ),
    )
    manager.status = "running"
    manager.latest_face_result = object()
    manager.latest_behavior_result = object()
    manager.latest_audio_result = object()
    return manager


def _result(
    key,
    *,
    score=0.99,
    status="potential_distress",
    age_seconds=0.0,
    sufficient=True,
    modalities=None,
    valid_feature_count=5,
    reason=None,
):
    timestamp = time.monotonic() - age_seconds
    return {
        "inference": _inference(
            status,
            score,
            valid_feature_count=valid_feature_count,
            available_modalities=modalities,
            reason=reason,
        ),
        "source_timestamp": timestamp,
        "modality_source_timestamps": {
            "face": timestamp,
            "behavior": timestamp,
            "audio": None,
        },
        "evidence_sufficient": sufficient,
        "evidence_sufficiency_reason": (
            None if sufficient else "Two modalities are required."
        ),
        "result_key": key,
    }


def _process_next(manager):
    manager.camera_capture_attempts += 1
    manager._refresh_live_inference()
    return manager.snapshot()


def test_duplicate_result_is_exported_as_ignored_without_changing_alert_state():
    manager = _manager_for_results(
        [
            _result("duplicate-key", score=0.99),
            _result("duplicate-key", score=0.01),
        ]
    )

    first = _process_next(manager)
    duplicate = _process_next(manager)

    assert first["latest_alert"].smoothed_score == pytest.approx(0.99)
    assert first["latest_alert"].state == "potential_distress"
    assert duplicate["latest_alert"].smoothed_score == pytest.approx(0.99)
    assert duplicate["latest_alert"].state == "potential_distress"
    assert duplicate["latest_alert"].alert_id == first["latest_alert"].alert_id
    diagnostic = duplicate["alert_diagnostics"][-1]
    assert diagnostic["decision"] == "ignored_duplicate"
    assert diagnostic["release_counter_before"] == 0
    assert diagnostic["release_counter_after"] == 0


def test_diagnostic_recording_failure_does_not_change_alert_update(monkeypatch):
    import monitoring

    manager = _manager_for_results([_result("diagnostic-failure")])

    def fail_diagnostic(*args, **kwargs):
        raise RuntimeError("diagnostics unavailable")

    monkeypatch.setattr(monitoring, "create_alert_diagnostic_entry", fail_diagnostic)

    snapshot = _process_next(manager)

    assert snapshot["latest_alert"].state == "potential_distress"
    assert snapshot["latest_alert"].smoothed_score == pytest.approx(0.99)
    assert snapshot["alert_diagnostics"] == []


def test_stale_and_insufficient_results_are_diagnosed_without_activation():
    manager = _manager_for_results(
        [
            _result("stale", age_seconds=10.0),
            _result(
                "insufficient",
                modalities={"face": True, "behavior": False, "audio": False},
                sufficient=False,
                reason="Only one modality is available.",
            ),
        ]
    )

    stale = _process_next(manager)["alert_diagnostics"][-1]
    insufficient_snapshot = _process_next(manager)
    insufficient = insufficient_snapshot["alert_diagnostics"][-1]

    assert stale["evidence_freshness"] == "stale"
    assert stale["decision"] == "rejected_stale_or_insufficient"
    assert stale["alert_state_after"] == "unknown"
    assert insufficient["evidence_sufficient"] is False
    assert insufficient["available_modalities"] == "face"
    assert json.loads(insufficient["modality_availability"]) == {
        "audio": False,
        "behavior": False,
        "face": True,
    }
    assert insufficient["decision"] == "rejected_stale_or_insufficient"
    assert insufficient["release_counter_after"] == 0
    assert insufficient_snapshot["latest_alert"].state == "unknown"


def test_missing_timestamps_and_unavailable_modalities_are_explicit():
    missing_timestamp = _result(
        "missing-time",
        modalities={"face": True, "behavior": False, "audio": False},
        sufficient=False,
    )
    missing_timestamp["source_timestamp"] = None
    missing_timestamp["modality_source_timestamps"] = {
        "face": None,
        "behavior": None,
        "audio": None,
    }
    manager = _manager_for_results([missing_timestamp])

    diagnostic = _process_next(manager)["alert_diagnostics"][-1]
    assert diagnostic["source_timestamp_monotonic"] is None
    assert diagnostic["evidence_age_seconds"] is None
    assert diagnostic["evidence_freshness"] == "missing_timestamp"
    assert json.loads(diagnostic["modality_freshness"]) == {
        "audio": "missing_timestamp",
        "behavior": "missing_timestamp",
        "face": "missing_timestamp",
    }
    assert diagnostic["decision"] == "rejected_stale_or_insufficient"


def test_threshold_qualification_counter_reset_and_activation_are_recorded():
    manager = _manager_for_results(
        [
            *[
                _result(
                    f"warmup-{index}",
                    score=0.01,
                    status="no_sufficient_distress_evidence",
                )
                for index in range(4)
            ],
            _result("below-threshold", score=0.74),
            _result("high", score=0.99),
        ]
    )

    for _ in range(4):
        snapshot = _process_next(manager)
    warmup = snapshot["alert_diagnostics"][-1]
    below = _process_next(manager)["alert_diagnostics"][-1]
    high_snapshot = _process_next(manager)
    high = high_snapshot["alert_diagnostics"][-1]

    assert warmup["activation_qualified"] is False
    assert below["potential_distress_probability"] == pytest.approx(0.74)
    assert below["activation_qualified"] is False
    assert below["alert_state_after"] == "normal"
    assert high["potential_distress_probability"] == pytest.approx(0.99)
    assert high["smoothed_score_after"] < 0.75
    assert high["activation_qualified"] is True
    assert high["activation_criterion"] == (
        "status == potential_distress and raw potential_distress_probability "
        ">= raw_probability_threshold"
    )
    assert high["raw_probability_threshold"] == pytest.approx(0.75)
    assert high["release_counter_before"] == 0
    assert high["release_counter_after"] == 0
    assert high["alert_state_after"] == "potential_distress"
    assert high_snapshot["latest_alert"].state == "potential_distress"


def test_release_count_and_cooldown_are_diagnosed_using_raw_probability():
    manager = _manager_for_results(
        [
            _result("active", score=0.99),
            _result("release-first", score=0.54),
            _result("release-second", score=0.54),
        ]
    )
    _process_next(manager)
    first_release = _process_next(manager)["alert_diagnostics"][-1]
    second_release = _process_next(manager)["alert_diagnostics"][-1]

    assert first_release["release_counter_after"] == 1
    assert first_release["alert_state_after"] == "potential_distress"
    assert second_release["release_counter_after"] == 0
    assert second_release["alert_state_after"] == "cooldown"
    assert second_release["release_threshold"] == pytest.approx(0.55)
    assert second_release["release_requirement"] == 2


def test_diagnostic_export_has_separate_versioned_schema_and_no_media_fields():
    manager = _manager_for_results([_result("csv-key")])
    snapshot = _process_next(manager)
    csv_text = alert_diagnostics_csv(snapshot["alert_diagnostics"])
    header = csv_text.splitlines()[0].split(",")

    assert tuple(header) == ALERT_DIAGNOSTIC_FIELDS
    assert "live-alert-diagnostic-v3" in csv_text
    assert "csv-key" in csv_text
    assert "frame" not in header
    assert "audio_recording" not in header


def test_diagnostics_remain_available_after_monitoring_stop():
    manager = _manager_for_results([_result("retained-key")])
    active = _process_next(manager)
    manager.stop()
    stopped = manager.snapshot()

    assert stopped["status"] == "stopped"
    assert stopped["alert_diagnostics"] == active["alert_diagnostics"]
