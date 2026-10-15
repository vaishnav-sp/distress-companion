import time
from pathlib import Path
from threading import Event, Thread, current_thread
from types import SimpleNamespace

import monitoring
import numpy as np
import pytest
from alert_management import AlertManager, AlertPolicy
from fusion import (
    build_fusion_diagnostic_snapshot,
    build_fusion_evidence,
    evidence_to_feature_record,
)
from fusion.live_status import live_fusion_status, live_modality_summary
from training import evidence_sufficient, infer_distress_status


class _CameraCaptureAdapter:
    def read(self):
        return monitoring.capture_frame()

    def close(self):
        return None


class _MicrophoneCaptureAdapter:
    def read_window(self, stop_event=None):
        return monitoring.capture_audio_window()

    def close(self):
        return None


@pytest.fixture(autouse=True)
def fake_monitoring_resources(monkeypatch):
    monkeypatch.setattr(monitoring, "CameraCaptureSession", _CameraCaptureAdapter)
    monkeypatch.setattr(monitoring, "MicrophoneCaptureStream", _MicrophoneCaptureAdapter)
    monkeypatch.setattr(
        monitoring,
        "capture_frame",
        lambda: SimpleNamespace(
            available=True,
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
            error=None,
            camera_opened=True,
            camera_index=0,
        ),
    )
    monkeypatch.setattr(
        monitoring,
        "capture_audio_window",
        lambda: SimpleNamespace(
            status="valid",
            message="Synthetic in-memory audio window.",
            available=True,
        ),
    )


class DummyFaceBranch:
    def __init__(self):
        self.calls = 0

    def analyze(self, frame):
        self.calls += 1
        return SimpleNamespace(
            input_quality=True,
            abstention_reason=None,
            status_message=f"Face features processed ({self.calls}).",
            feature_values={"angry": 0.2},
            feature_validity={"angry": True},
            feature_names=("angry",),
            unavailable_reasons={},
            baseline_ready=True,
            baseline_sample_count=self.calls,
            quality_score=0.9,
            quality_status="usable",
        )


class DummyBehaviorBranch:
    def __init__(self):
        self.calls = 0

    def analyze(self, frame, timestamp_seconds=None):
        self.calls += 1
        return SimpleNamespace(
            status="ok",
            status_message=f"Behavior features processed ({self.calls}).",
            feature_values={"movement_energy": 1.5},
            feature_validity={"movement_energy": True},
            feature_names=("movement_energy",),
            unavailable_reasons={},
            quality_score=0.85,
            baseline_ready=True,
            baseline_sample_count=self.calls,
        )

    def reset(self):
        return None


class DummyAudioBranch:
    def __init__(self):
        self.calls = 0

    def analyze(self, capture):
        self.calls += 1
        return SimpleNamespace(
            status="ok",
            status_message=f"Audio features processed ({self.calls}).",
            feature_values={"rms_loudness": 0.5},
            feature_validity={"rms_loudness": True},
            feature_names=("rms_loudness",),
            unavailable_reasons={},
            baseline_ready=True,
            baseline_sample_count=self.calls,
            quality_flag=True,
            quality_status="usable",
        )

    def reset(self):
        return None


def test_monitoring_manager_starts_once_and_stops_cleanly(monkeypatch):
    monkeypatch.setattr(
        monitoring,
        "capture_frame",
        lambda: SimpleNamespace(available=True, frame=object(), error=None),
    )
    monkeypatch.setattr(
        monitoring,
        "capture_audio_window",
        lambda: SimpleNamespace(status="ok", message="Audio window captured."),
    )

    def infer(_state):
        return {
            "inference": SimpleNamespace(
                status="unknown",
                reason="Test inference has no fresh evidence.",
            ),
            "source_timestamp": None,
            "result_key": "test-inference",
            "updated_at_utc": "2026-10-15T00:00:00+00:00",
            "stage_timings_ms": {"rule_evaluation": 0.01},
        }

    manager = monitoring.MonitoringManager(
        camera_interval_seconds=0.05,
        audio_interval_seconds=0.05,
        shutdown_timeout_seconds=0.2,
        face_branch_factory=lambda: DummyFaceBranch(),
        behavior_branch_factory=lambda: DummyBehaviorBranch(),
        audio_branch_factory=lambda: DummyAudioBranch(),
        inference_processor=infer,
    )

    assert manager.start() is True
    time.sleep(0.1)
    deadline = time.monotonic() + 1.0
    while (
        "camera_intercapture" not in manager.snapshot()["stage_timings"]
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)
    assert manager.is_running is True
    assert manager.start() is False
    assert manager.camera_success_count >= 1 or manager.audio_success_count >= 1
    snapshot = manager.snapshot()
    assert snapshot["last_face_success_timestamp"] is not None
    assert snapshot["last_behavior_success_timestamp"] is not None
    assert snapshot["last_audio_success_timestamp"] is not None
    assert snapshot["face_analysis_attempts"] >= 1
    assert snapshot["face_results_returned"] >= 1
    assert snapshot["behavior_analysis_attempts"] >= 1
    assert snapshot["behavior_results_returned"] >= 1
    assert snapshot["audio_analysis_attempts"] >= 1
    assert snapshot["audio_results_returned"] >= 1
    timings = snapshot["stage_timings"]
    assert {
        "camera_capture",
        "camera_intercapture",
        "face_branch",
        "behavior_branch",
        "audio_capture",
        "audio_feature_extraction",
        "rule_evaluation",
    } <= timings.keys()
    assert all(
        timing["count"] >= 1
        and timing["last_ms"] >= 0
        and timing["mean_ms"] >= 0
        for timing in timings.values()
    )

    manager.stop()
    time.sleep(0.05)
    assert manager.is_running is False
    assert manager.status == "stopped"
    stopped_timings = manager.snapshot()["stage_timings"]
    assert timings.keys() <= stopped_timings.keys()


def test_pose_diagnostics_toggle_targets_existing_behavior_branch():
    behavior_branch = DummyBehaviorBranch()
    manager = monitoring.MonitoringManager(
        behavior_branch=behavior_branch,
    )

    manager.set_pose_diagnostics_enabled(True)

    assert behavior_branch.diagnostics_enabled is True
    assert manager.pose_diagnostics_enabled is True
    manager.set_pose_diagnostics_enabled(False)
    assert behavior_branch.diagnostics_enabled is False


def test_live_pose_overlay_comes_from_the_existing_monitoring_worker(monkeypatch):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    overlay = np.ones((8, 8, 3), dtype=np.uint8)
    calls = {"capture": 0, "pose": 0}

    class PoseBranch(DummyBehaviorBranch):
        def analyze(self, captured_frame, timestamp_seconds=None):
            calls["pose"] += 1
            assert captured_frame is frame
            return SimpleNamespace(
                status="ok",
                status_message="Pose result published by monitoring.",
                feature_values={},
                feature_validity={},
                feature_names=(),
                unavailable_reasons={},
                quality_score=1.0,
                baseline_ready=True,
                baseline_sample_count=1,
                pose_diagnostic={"pose_detected": True},
                debug_overlay=overlay,
                region_metrics={"torso": {"movement_energy": 0.1}},
                region_feature_validity={"torso": {"movement_energy": True}},
                region_unavailable_reasons={"torso": {}},
            )

    def capture():
        calls["capture"] += 1
        return SimpleNamespace(
            available=True,
            frame=frame,
            error=None,
            camera_opened=True,
            camera_index=0,
        )

    monkeypatch.setattr(monitoring, "capture_frame", capture)
    branch = PoseBranch()
    manager = monitoring.MonitoringManager(
        face_branch=None,
        behavior_branch=branch,
        audio_branch=None,
    )

    manager._check_camera()
    snapshot = manager.snapshot()

    assert calls == {"capture": 1, "pose": 1}
    assert snapshot["latest_behavior_result"].debug_overlay is overlay
    assert snapshot["latest_behavior_result"].region_metrics["torso"]["movement_energy"] == 0.1


def test_worker_reuses_sensor_resources_and_publishes_unique_background_history():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    resources = {"camera_open": 0, "camera_close": 0, "mic_open": 0, "mic_close": 0}
    inference_threads = []

    class CameraSession:
        is_opened = True

        def __init__(self):
            resources["camera_open"] += 1

        def read(self):
            return SimpleNamespace(
                available=True,
                frame=frame,
                error=None,
                camera_opened=True,
                camera_index=0,
            )

        def close(self):
            resources["camera_close"] += 1

    class MicrophoneStream:
        def __init__(self):
            resources["mic_open"] += 1

        def read_window(self, stop_event=None):
            return SimpleNamespace(
                status="valid",
                available=True,
                message="Fresh test window.",
            )

        def close(self):
            resources["mic_close"] += 1

    def infer(snapshot):
        inference_threads.append(current_thread().name)
        source_timestamp = max(
            timestamp
            for timestamp in (
                snapshot["last_face_success_timestamp"],
                snapshot["last_behavior_success_timestamp"],
                snapshot["last_audio_success_timestamp"],
            )
            if timestamp is not None
        )
        return {
            "inference": SimpleNamespace(
                status="no_sufficient_distress_evidence",
                predicted_label="no_sufficient_distress_evidence",
                model_score=0.9,
                potential_distress_score=0.1,
                valid_feature_count=5,
                available_modalities={"face": True, "behavior": True, "audio": False},
                reason=None,
            ),
            "feature_record": None,
            "source_timestamp": source_timestamp,
            "result_key": f"result-{source_timestamp}",
            "updated_at_utc": "2026-10-15T00:00:00+00:00",
        }

    manager = monitoring.MonitoringManager(
        camera_interval_seconds=0.1,
        audio_interval_seconds=0.5,
        shutdown_timeout_seconds=1.0,
        face_branch=DummyFaceBranch(),
        behavior_branch=DummyBehaviorBranch(),
        audio_branch=DummyAudioBranch(),
        camera_session_factory=CameraSession,
        microphone_stream_factory=MicrophoneStream,
        inference_processor=infer,
    )

    assert manager.start() is True
    assert manager.start() is False
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        snapshot = manager.snapshot()
        if snapshot["audio_success_count"] and snapshot["inference_history"]:
            break
        time.sleep(0.02)

    assert manager.is_running is True
    assert resources["camera_open"] == 1
    assert resources["mic_open"] == 1
    assert inference_threads and all(name != "MainThread" for name in inference_threads)
    history = manager.snapshot()["inference_history"]
    result_keys = [entry["result_key"] for entry in history]
    assert result_keys
    assert len(result_keys) == len(set(result_keys))
    assert set(history[-1]) >= {
        "timestamp",
        "prediction",
        "predicted_class_probability",
        "potential_distress_probability",
        "smoothed_distress_score",
        "alert_state",
        "available_modalities",
        "inference_status",
    }

    manager.stop()
    stopped = manager.snapshot()
    assert stopped["worker_alive"] is False
    assert stopped["status"] == "stopped"
    assert stopped["latest_inference"] is None
    assert stopped["inference_history"] == history
    assert resources == {"camera_open": 1, "camera_close": 1, "mic_open": 1, "mic_close": 1}


def test_regional_rule_result_is_the_result_consumed_by_alert_manager():
    from rule_based_detection import RuleBasedInferenceProcessor, RuleDetectionConfig

    now = time.monotonic()
    behavior = SimpleNamespace(
        status="partial",
        feature_values={"shoulder_alignment": 0.0},
        feature_validity={"shoulder_alignment": True},
        feature_names=("shoulder_alignment",),
        unavailable_reasons={},
        region_metrics={
            "torso": {
                "repetition_rate_hz": 1.0,
                "amplitude": 0.1,
                "elevated_movement_proportion": 0.8,
            }
        },
        region_feature_validity={
            "torso": {
                "repetition_rate_hz": True,
                "amplitude": True,
                "elevated_movement_proportion": True,
            }
        },
        region_unavailable_reasons={"torso": {}},
    )
    processor = RuleBasedInferenceProcessor(
        RuleDetectionConfig(
            combination_mode="movement_only",
            movement_regions=("torso",),
            movement_repetition_seconds=0.0,
        )
    )
    result = processor(
        {
            "status": "running",
            "latest_face_result": None,
            "latest_behavior_result": behavior,
            "latest_audio_result": None,
            "last_face_success_timestamp": None,
            "last_behavior_success_timestamp": now,
            "last_audio_success_timestamp": None,
        }
    )
    alert_manager = AlertManager(
        AlertPolicy(
            minimum_valid_features=1,
            max_result_age_seconds=3.0,
        )
    )
    alert = alert_manager.update(
        result["inference"],
        result_timestamp=result["source_timestamp"],
        result_key=result["result_key"],
        now=now,
    )

    assert result["inference"].cue_indicators["movement_repetitive_torso"] is True
    assert result["inference"].rule_triggered is True
    assert alert.state == "potential_distress"


def test_branch_factories_run_once_per_monitoring_lifecycle(monkeypatch):
    monkeypatch.setattr(
        monitoring,
        "capture_frame",
        lambda: SimpleNamespace(available=False, frame=None, error="Camera unavailable."),
    )
    monkeypatch.setattr(
        monitoring,
        "capture_audio_window",
        lambda: SimpleNamespace(status="missing", message="Microphone unavailable."),
    )
    calls = {"face": 0, "behavior": 0, "audio": 0}

    def factory(name, branch_type):
        def create():
            calls[name] += 1
            return branch_type()

        return create

    manager = monitoring.MonitoringManager(
        camera_interval_seconds=0.01,
        audio_interval_seconds=0.01,
        face_branch_factory=factory("face", DummyFaceBranch),
        behavior_branch_factory=factory("behavior", DummyBehaviorBranch),
        audio_branch_factory=factory("audio", DummyAudioBranch),
    )

    assert manager.start() is True
    assert manager.start() is False
    time.sleep(0.04)
    manager.stop()
    assert calls == {"face": 1, "behavior": 1, "audio": 1}

    assert manager.start() is True
    time.sleep(0.04)
    manager.stop()
    assert calls == {"face": 2, "behavior": 2, "audio": 2}


def test_monitoring_manager_tracks_errors_without_duplicate_workers(monkeypatch):
    def fail_capture_frame():
        raise RuntimeError("camera unavailable")

    monkeypatch.setattr(monitoring, "capture_frame", fail_capture_frame)
    monkeypatch.setattr(
        monitoring,
        "capture_audio_window",
        lambda: SimpleNamespace(status="ok", message="Audio window captured."),
    )

    manager = monitoring.MonitoringManager(
        camera_interval_seconds=0.01,
        audio_interval_seconds=0.01,
        shutdown_timeout_seconds=0.2,
        face_branch_factory=lambda: DummyFaceBranch(),
        behavior_branch_factory=lambda: DummyBehaviorBranch(),
        audio_branch_factory=lambda: DummyAudioBranch(),
    )

    assert manager.start() is True
    time.sleep(0.12)
    assert manager.is_running is True
    assert manager._worker is not None

    manager.stop()
    assert manager._worker is None
    assert manager.status == "stopped"

    manager.start()
    assert manager.is_running is True
    manager.stop()
    assert manager.is_running is False


def test_monitoring_manager_uses_provided_branch_instances(monkeypatch):
    monkeypatch.setattr(
        monitoring,
        "capture_frame",
        lambda: SimpleNamespace(available=True, frame=object(), error=None),
    )
    monkeypatch.setattr(
        monitoring,
        "capture_audio_window",
        lambda: SimpleNamespace(status="valid", message="Audio window captured.", available=True),
    )

    face_branch = DummyFaceBranch()
    behavior_branch = DummyBehaviorBranch()
    audio_branch = DummyAudioBranch()
    manager = monitoring.MonitoringManager(
        camera_interval_seconds=0.01,
        audio_interval_seconds=0.01,
        shutdown_timeout_seconds=0.2,
        face_branch=face_branch,
        behavior_branch=behavior_branch,
        audio_branch=audio_branch,
    )

    assert manager.face_branch is face_branch
    assert manager.behavior_branch is behavior_branch
    assert manager.audio_branch is audio_branch

    assert manager.start() is True
    time.sleep(0.08)
    assert manager.latest_face_result is not None
    assert manager.latest_behavior_result is not None
    assert manager.latest_audio_result is not None

    manager.stop()
    assert manager.is_running is False


def test_monitoring_manager_reports_missing_branch_initialization(monkeypatch):
    def fail_face():
        raise RuntimeError("face model failed")

    monkeypatch.setattr(
        monitoring,
        "capture_frame",
        lambda: SimpleNamespace(available=True, frame=object(), error=None),
    )
    monkeypatch.setattr(
        monitoring,
        "capture_audio_window",
        lambda: SimpleNamespace(status="valid", message="Audio window captured.", available=True),
    )

    manager = monitoring.MonitoringManager(
        camera_interval_seconds=0.01,
        audio_interval_seconds=0.01,
        shutdown_timeout_seconds=0.2,
        face_branch_factory=fail_face,
        behavior_branch_factory=lambda: DummyBehaviorBranch(),
        audio_branch_factory=lambda: DummyAudioBranch(),
    )

    assert manager.start() is True
    time.sleep(0.08)
    assert manager.last_error["face"] is not None
    assert "Face branch" in (manager.last_error["face"] or "")
    assert "RuntimeError" in (manager.last_error["face"] or "")
    assert "face model failed" in (manager.last_error["face"] or "")
    assert manager.latest_behavior_result is not None or manager.latest_audio_result is not None

    manager.stop()


def test_monitoring_manager_reports_branch_factory_returning_none(monkeypatch):
    monkeypatch.setattr(
        monitoring,
        "capture_frame",
        lambda: SimpleNamespace(available=False, frame=None, error="Camera unavailable."),
    )
    monkeypatch.setattr(
        monitoring,
        "capture_audio_window",
        lambda: SimpleNamespace(status="missing", message="Microphone unavailable."),
    )
    manager = monitoring.MonitoringManager(
        face_branch_factory=lambda: None,
        behavior_branch_factory=DummyBehaviorBranch,
        audio_branch_factory=DummyAudioBranch,
    )

    assert manager.start() is True
    assert manager.last_error["face"] == "Face branch initialization returned None."
    manager.stop()


def test_monitoring_stop_keeps_live_worker_and_branches_until_worker_exits(monkeypatch):
    capture_started = Event()
    release_capture = Event()

    def slow_capture_frame():
        capture_started.set()
        release_capture.wait(timeout=1.0)
        return SimpleNamespace(available=True, frame=object(), error=None)

    monkeypatch.setattr(monitoring, "capture_frame", slow_capture_frame)
    monkeypatch.setattr(
        monitoring,
        "capture_audio_window",
        lambda: SimpleNamespace(status="valid", message="Audio window captured.", available=True),
    )

    face_branch = DummyFaceBranch()
    manager = monitoring.MonitoringManager(
        camera_interval_seconds=0.01,
        audio_interval_seconds=0.01,
        shutdown_timeout_seconds=0.02,
        face_branch=face_branch,
        behavior_branch=DummyBehaviorBranch(),
        audio_branch=DummyAudioBranch(),
    )

    assert manager.start() is True
    assert capture_started.wait(timeout=0.5)
    worker = manager._worker
    snapshot = manager.snapshot()
    assert snapshot["camera_capture_attempts"] == 1
    assert snapshot["current_operation"] == "capturing camera frame"

    manager.stop()

    assert manager.status == "stopping"
    assert manager.is_running is True
    assert manager._worker is worker
    assert manager.face_branch is face_branch

    release_capture.set()
    worker.join(timeout=1.0)

    assert manager.is_running is False
    assert manager.status == "stopped"
    assert manager.face_branch is None
    assert manager.behavior_branch is None
    assert manager.audio_branch is None


def test_capture_failures_explain_missing_branch_results_without_calling_branches(monkeypatch):
    monkeypatch.setattr(
        monitoring,
        "capture_frame",
        lambda: SimpleNamespace(
            available=False,
            frame=None,
            error="Could not open the camera.",
            camera_opened=False,
        ),
    )
    monkeypatch.setattr(
        monitoring,
        "capture_audio_window",
        lambda: SimpleNamespace(
            status="missing",
            available=False,
            message="No microphone input is available.",
        ),
    )
    face_branch = DummyFaceBranch()
    behavior_branch = DummyBehaviorBranch()
    audio_branch = DummyAudioBranch()
    manager = monitoring.MonitoringManager(
        face_branch=face_branch,
        behavior_branch=behavior_branch,
        audio_branch=audio_branch,
    )

    manager._check_camera()
    manager._check_microphone()
    snapshot = manager.snapshot()

    assert snapshot["camera_capture_attempts"] == 1
    assert snapshot["audio_capture_attempts"] == 1
    assert snapshot["camera_status"] == "unavailable"
    assert snapshot["camera_message"] == "Could not open the camera."
    assert snapshot["microphone_status"] == "unavailable"
    assert snapshot["microphone_message"] == "No microphone input is available."
    assert snapshot["latest_face_result"] is None
    assert snapshot["latest_behavior_result"] is None
    assert snapshot["latest_audio_result"] is None
    assert snapshot["face_analysis_attempts"] == 0
    assert snapshot["behavior_analysis_attempts"] == 0
    assert snapshot["audio_analysis_attempts"] == 0
    assert snapshot["face_inference_invocations"] == 0
    assert snapshot["behavior_inference_invocations"] == 0
    assert snapshot["last_camera_opened"] is False
    assert snapshot["diagnostics"]["camera"]["stage"] == "sensor_capture_failure"
    assert face_branch.calls == 0
    assert behavior_branch.calls == 0
    assert audio_branch.calls == 0


def test_valid_camera_frame_records_capture_and_branch_publication_diagnostics(
    monkeypatch,
):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    monkeypatch.setattr(
        monitoring,
        "capture_frame",
        lambda: SimpleNamespace(
            available=True,
            frame=frame,
            error=None,
            camera_opened=True,
        ),
    )
    face_branch = DummyFaceBranch()
    behavior_branch = DummyBehaviorBranch()
    manager = monitoring.MonitoringManager(
        face_branch=face_branch,
        behavior_branch=behavior_branch,
        audio_branch=DummyAudioBranch(),
    )

    manager._check_camera()
    snapshot = manager.snapshot()

    assert snapshot["camera_capture_attempts"] == 1
    assert snapshot["camera_frames_captured"] == 1
    assert snapshot["last_camera_opened"] is True
    assert snapshot["last_camera_frame_info"] == {
        "shape": (480, 640, 3),
        "channels": 3,
        "dtype": "uint8",
        "channel_order": "BGR",
        "camera_index": None,
        "valid_for_branch_input": True,
    }
    assert snapshot["face_inference_invocations"] == 1
    assert snapshot["behavior_inference_invocations"] == 1
    assert face_branch.calls == 1
    assert behavior_branch.calls == 1
    assert snapshot["face_results_returned"] == 1
    assert snapshot["behavior_results_returned"] == 1
    assert snapshot["last_face_success_timestamp"] is not None
    assert snapshot["last_behavior_success_timestamp"] is not None
    assert snapshot["diagnostics"]["face"]["stage"] == "result_published"
    assert snapshot["diagnostics"]["behavior"]["stage"] == "result_published"


def test_stop_clears_live_results_but_retains_publication_and_stop_diagnostics(monkeypatch):
    monkeypatch.setattr(
        monitoring,
        "capture_frame",
        lambda: SimpleNamespace(available=True, frame=object(), error=None),
    )
    monkeypatch.setattr(
        monitoring,
        "capture_audio_window",
        lambda: SimpleNamespace(status="valid", message="Audio window captured."),
    )
    manager = monitoring.MonitoringManager(
        camera_interval_seconds=0.01,
        audio_interval_seconds=0.01,
        shutdown_timeout_seconds=0.5,
        face_branch=DummyFaceBranch(),
        behavior_branch=DummyBehaviorBranch(),
        audio_branch=DummyAudioBranch(),
    )

    assert manager.start() is True
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        snapshot = manager.snapshot()
        if (
            snapshot["face_results_returned"]
            and snapshot["behavior_results_returned"]
            and snapshot["audio_results_returned"]
        ):
            break
        time.sleep(0.01)
    assert manager.is_running is True

    manager.stop()
    snapshot = manager.snapshot()

    assert snapshot["status"] == "stopped"
    assert snapshot["worker_alive"] is False
    assert snapshot["worker_exit_reason"] == "stop_requested"
    assert snapshot["latest_face_result"] is None
    assert snapshot["latest_behavior_result"] is None
    assert snapshot["latest_audio_result"] is None
    assert snapshot["face_results_returned"] > 0
    assert snapshot["behavior_results_returned"] > 0
    assert snapshot["audio_results_returned"] > 0
    assert snapshot["diagnostics"]["worker"]["stage"] == "stopped"


def test_fusion_diagnostic_survives_stop_but_cannot_feed_live_inference_or_alerts(
    monkeypatch,
):
    class DiagnosticFaceBranch:
        def analyze(self, frame):
            labels = ("angry", "fear", "happy", "neutral", "sad", "surprise")
            return SimpleNamespace(
                status="ok",
                status_message="Synthetic face features.",
                expression_probabilities={
                    "angry": 0.3,
                    "fear": 0.2,
                    "happy": 0.1,
                    "neutral": 0.1,
                    "sad": 0.2,
                    "surprise": 0.1,
                },
                class_validity={name: True for name in labels},
                feature_vector_order=labels,
                negative_affect_feature=0.23,
                face_quality_score=0.9,
            )

    class DiagnosticBehaviorBranch:
        def analyze(self, frame, timestamp_seconds=None):
            return SimpleNamespace(
                status="ok",
                status_message="Synthetic behavior features.",
                feature_values={"movement_energy": 0.2},
                feature_validity={"movement_energy": True},
                feature_names=("movement_energy",),
                unavailable_reasons={},
            )

    class DiagnosticAudioBranch:
        def analyze(self, capture):
            return SimpleNamespace(
                status="ok",
                status_message="Synthetic audio features.",
                feature_values={"rms_loudness": 0.3},
                feature_validity={"rms_loudness": True},
                feature_names=("rms_loudness",),
                unavailable_reasons={},
            )

    monkeypatch.setattr(
        monitoring,
        "capture_frame",
        lambda: SimpleNamespace(available=True, frame=object(), error=None),
    )
    monkeypatch.setattr(
        monitoring,
        "capture_audio_window",
        lambda: SimpleNamespace(status="valid", message="Synthetic audio capture."),
    )
    manager = monitoring.MonitoringManager(
        camera_interval_seconds=0.01,
        audio_interval_seconds=0.01,
        shutdown_timeout_seconds=0.5,
        face_branch=DiagnosticFaceBranch(),
        behavior_branch=DiagnosticBehaviorBranch(),
        audio_branch=DiagnosticAudioBranch(),
    )

    assert manager.snapshot()["fusion_diagnostic_snapshot"] is None
    assert manager.start() is True
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        active_snapshot = manager.snapshot()
        if all(
            active_snapshot[f"{modality}_results_returned"] > 0
            for modality in ("face", "behavior", "audio")
        ):
            break
        time.sleep(0.01)
    assert manager.is_running is True

    evidence = build_fusion_evidence(active_snapshot)
    record = evidence_to_feature_record(evidence)
    sufficient, reason = evidence_sufficient(record)
    assert sufficient is True
    diagnostic = build_fusion_diagnostic_snapshot(
        evidence,
        record,
        source_timestamps={
            "face": active_snapshot["last_face_success_timestamp"],
            "behavior": active_snapshot["last_behavior_success_timestamp"],
            "audio": active_snapshot["last_audio_success_timestamp"],
        },
        sufficient=sufficient,
        sufficiency_reason=reason,
        runtime_diagnostics={
            "camera_frames_captured": 1,
            "face_inference_invocations": 1,
            "face_results_returned": 1,
            "behavior_inference_invocations": 1,
            "behavior_results_returned": 1,
        },
    )
    manager.store_fusion_diagnostic_snapshot(diagnostic)

    manager.stop()
    stopped_snapshot = manager.snapshot()

    assert stopped_snapshot["fusion_diagnostic_snapshot"] == diagnostic
    assert stopped_snapshot["fusion_diagnostic_snapshot"]["construction_status"] == (
        "constructed_with_valid_multimodal_features"
    )
    historical_runtime = stopped_snapshot["fusion_diagnostic_snapshot"][
        "runtime_diagnostics"
    ]
    assert historical_runtime["camera_frames_captured"] > 0
    assert historical_runtime["face_inference_invocations"] > 0
    assert historical_runtime["behavior_inference_invocations"] > 0
    assert stopped_snapshot["latest_face_result"] is None
    assert stopped_snapshot["latest_behavior_result"] is None
    assert stopped_snapshot["latest_audio_result"] is None

    live_evidence = build_fusion_evidence(stopped_snapshot)
    live_record = evidence_to_feature_record(live_evidence)
    inference = infer_distress_status(live_record)
    assert live_evidence.has_data is False
    assert inference.status == "unknown"
    assert inference.valid_feature_count == 0
    alert = AlertManager(AlertPolicy()).report_unknown(
        "Monitoring is stopped; no fresh inference is being produced."
    )
    assert alert.state == "unknown"


def test_monitoring_stop_resets_live_alert_manager_state():
    alert_manager = AlertManager(AlertPolicy(smoothing_window=1))
    active = alert_manager.update(
        SimpleNamespace(
            status="potential_distress",
            predicted_label="potential_distress",
            model_score=0.8,
            potential_distress_score=0.8,
            potential_distress_probability=0.8,
            valid_feature_count=5,
            available_modalities={"face": True, "behavior": True, "audio": False},
            reason=None,
        ),
        result_timestamp=10.0,
        now=10.0,
        result_key="active-before-stop",
    )
    assert active.state == "potential_distress"
    manager = monitoring.MonitoringManager(alert_manager=alert_manager)
    manager.status = "running"
    manager.latest_alert = active

    manager.stop()
    snapshot = manager.snapshot()

    assert snapshot["status"] == "stopped"
    assert snapshot["latest_alert"].state == "unknown"
    assert snapshot["latest_alert"].alert_id is None
    assert snapshot["latest_inference"] is None


def test_unexpected_worker_exit_retains_exception_type_and_message(monkeypatch):
    exit_recorded = Event()

    def fail_worker_sleep(_duration):
        exit_recorded.set()
        raise RuntimeError("worker pacing failed")

    monkeypatch.setattr(
        monitoring,
        "capture_frame",
        lambda: SimpleNamespace(
            available=False,
            frame=None,
            error="Camera unavailable.",
        ),
    )
    monkeypatch.setattr(
        monitoring,
        "capture_audio_window",
        lambda: SimpleNamespace(status="missing", message="Microphone unavailable."),
    )
    monkeypatch.setattr(monitoring.time, "sleep", fail_worker_sleep)
    manager = monitoring.MonitoringManager(
        face_branch=DummyFaceBranch(),
        behavior_branch=DummyBehaviorBranch(),
        audio_branch=DummyAudioBranch(),
    )

    assert manager.start() is True
    assert exit_recorded.wait(timeout=1.0)
    worker = manager._worker
    assert worker is not None
    worker.join(timeout=1.0)

    snapshot = manager.snapshot()
    assert snapshot["worker_alive"] is False
    assert snapshot["status"] == "error"
    assert snapshot["worker_exit_reason"] == "unexpected_exit"
    assert snapshot["diagnostics"]["worker"]["stage"] == "unexpected_worker_exit"
    assert snapshot["diagnostics"]["worker"]["exception_type"] == "RuntimeError"
    assert snapshot["diagnostics"]["worker"]["message"] == "worker pacing failed"
    assert "RuntimeError" in snapshot["message"]


def test_face_inference_exception_is_diagnostic_and_not_published(monkeypatch):
    class FailingFaceBranch:
        def analyze(self, frame):
            raise RuntimeError("face inference blocked by a model error")

    monkeypatch.setattr(
        monitoring,
        "capture_frame",
        lambda: SimpleNamespace(available=True, frame=object(), error=None),
    )
    manager = monitoring.MonitoringManager(
        face_branch=FailingFaceBranch(),
        behavior_branch=DummyBehaviorBranch(),
        audio_branch=DummyAudioBranch(),
    )

    manager._check_camera()
    snapshot = manager.snapshot()

    assert snapshot["face_analysis_attempts"] == 1
    assert snapshot["face_results_returned"] == 0
    assert snapshot["latest_face_result"] is None
    assert "face inference blocked by a model error" in snapshot["last_error"]["face"]
    assert snapshot["diagnostics"]["face"]["stage"] == "branch_inference_exception"
    assert snapshot["diagnostics"]["face"]["exception_type"] == "RuntimeError"
    assert snapshot["behavior_results_returned"] == 1


def test_valid_face_abstention_is_published_and_marked_as_no_usable_result(monkeypatch):
    class AbstainingFaceBranch:
        def analyze(self, frame):
            return SimpleNamespace(
                abstention_reason="no_face",
                status_message="No usable face was detected.",
                feature_validity={},
            )

    monkeypatch.setattr(
        monitoring,
        "capture_frame",
        lambda: SimpleNamespace(available=True, frame=object(), error=None),
    )
    manager = monitoring.MonitoringManager(
        face_branch=AbstainingFaceBranch(),
        behavior_branch=DummyBehaviorBranch(),
        audio_branch=DummyAudioBranch(),
    )

    manager._check_camera()
    snapshot = manager.snapshot()

    assert snapshot["latest_face_result"] is not None
    assert snapshot["face_analysis_attempts"] == 1
    assert snapshot["face_results_returned"] == 1
    assert snapshot["diagnostics"]["face"]["stage"] == "valid_inference_no_usable_result"
    assert snapshot["diagnostics"]["face"]["message"] == "No usable face was detected."


def test_result_publication_failure_is_reported_with_exception_type():
    class FailingPublicationManager(monitoring.MonitoringManager):
        @property
        def latest_face_result(self):
            return getattr(self, "_stored_face_result", None)

        @latest_face_result.setter
        def latest_face_result(self, result):
            if result is not None:
                raise RuntimeError("face result store unavailable")
            self._stored_face_result = result

    manager = FailingPublicationManager(
        face_branch=DummyFaceBranch(),
        behavior_branch=DummyBehaviorBranch(),
        audio_branch=DummyAudioBranch(),
    )

    published = manager._publish_result("face", object(), time.monotonic())
    snapshot = manager.snapshot()

    assert published is False
    assert snapshot["latest_face_result"] is None
    assert snapshot["face_results_returned"] == 1
    assert snapshot["face_results_published"] == 0
    assert snapshot["diagnostics"]["face"]["stage"] == "result_publication_failure"
    assert snapshot["diagnostics"]["face"]["exception_type"] == "RuntimeError"
    assert snapshot["diagnostics"]["face"]["message"] == "face result store unavailable"


def test_app_passes_branch_factories_to_monitoring_manager():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    for dependency in (
        "face_branch_factory=get_face_branch",
        "behavior_branch_factory=get_behavior_branch",
        "audio_branch_factory=get_audio_branch",
    ):
        assert dependency in source
    for dependency in (
        "face_branch=get_face_branch()",
        "behavior_branch=get_behavior_branch()",
        "audio_branch=get_audio_branch()",
    ):
        assert dependency not in source
    assert "status == \"stopped\" and results_returned > 0" in source
    assert "stopped before this branch returned a result" in source
    assert 'worker_snapshot.get("latest_inference")' in source
    assert "inference_processor=RuleBasedInferenceProcessor(" in source
    assert "get_background_distress_model" not in source
    assert "Inference status: **{inference.status}**" in source
    assert 'alert.state == "potential_distress"' in source
    assert "Alert lifecycle: NO ACTIVE ALERT — {alert.reason}" in source
    assert 'if status == "stopped":' in source
    assert "Monitoring status: **STOPPED**" in source
    assert "Live inference: **NOT RUNNING**" in source
    assert "Current live features were cleared when monitoring stopped" in source
    assert "Alert state: UNKNOWN because monitoring is stopped." in source
    assert "this stopped state does not describe the previous run" in source


def test_stale_published_results_are_excluded_live_but_retained_in_history():
    manager = monitoring.MonitoringManager()
    now = time.monotonic()
    manager._publish_result(
        "face",
        SimpleNamespace(
            status="ok",
            status_message="Face features were published.",
            feature_values={"angry": 0.2, "fear": 0.1, "happy": 0.3},
            feature_validity={"angry": True, "fear": True, "happy": True},
            feature_names=("angry", "fear", "happy"),
            unavailable_reasons={},
        ),
        now - 10,
    )
    manager._publish_result(
        "behavior",
        SimpleNamespace(
            status="ok",
            status_message="Behavior features were published.",
            feature_values={"movement_energy": 0.4},
            feature_validity={"movement_energy": True},
            feature_names=("movement_energy",),
            unavailable_reasons={},
        ),
        now - 10,
    )
    with manager._lock:
        manager.status = "running"
        manager._worker = SimpleNamespace(is_alive=lambda: True)

    raw_snapshot = manager.snapshot()
    assert raw_snapshot["worker_alive"] is True
    assert raw_snapshot["face_results_published"] == 1
    assert raw_snapshot["behavior_results_published"] == 1

    live_snapshot = dict(raw_snapshot)
    freshness = {}
    for modality in ("face", "behavior", "audio"):
        result = raw_snapshot[f"latest_{modality}_result"]
        timestamp = raw_snapshot.get(f"last_{modality}_success_timestamp")
        stale = result is not None and (
            timestamp is None or now - float(timestamp) > 3.0
        )
        freshness[modality] = not stale
        if stale:
            live_snapshot[f"latest_{modality}_result"] = None

    live_evidence = build_fusion_evidence(live_snapshot)
    historical_evidence = build_fusion_evidence(raw_snapshot)
    historical_record = evidence_to_feature_record(
        historical_evidence,
        provenance={"source": "monitoring_diagnostic_snapshot"},
    )
    historical_sufficient, reason = evidence_sufficient(historical_record)
    historical_diagnostic = build_fusion_diagnostic_snapshot(
        historical_evidence,
        historical_record,
        source_timestamps={
            modality: (
                raw_snapshot.get(f"last_{modality}_success_timestamp")
                if raw_snapshot.get(f"latest_{modality}_result") is not None
                else None
            )
            for modality in ("face", "behavior", "audio")
        },
        sufficient=historical_sufficient,
        sufficiency_reason=reason,
        runtime_diagnostics={
            "monitor_status": raw_snapshot["status"],
            "worker_alive": raw_snapshot["worker_alive"],
            "face_results_published": raw_snapshot["face_results_published"],
            "behavior_results_published": raw_snapshot["behavior_results_published"],
        },
    )

    assert freshness == {"face": False, "behavior": False, "audio": True}
    assert live_evidence.has_data is False
    live_record = evidence_to_feature_record(live_evidence)
    sufficient, insufficiency_reason = evidence_sufficient(live_record)
    assert sufficient is False
    assert live_fusion_status(
        "running",
        model_available=True,
        evidence_sufficient=sufficient,
        inference_reason=insufficiency_reason,
    )[0] == "WAITING FOR SUFFICIENT FRESH EVIDENCE"
    assert live_modality_summary(
        "running",
        results={
            modality: raw_snapshot[f"latest_{modality}_result"]
            for modality in ("face", "behavior", "audio")
        },
        freshness=freshness,
        available_modalities=live_record.available_modalities,
    ) == {
        "face": "stale; excluded from live inference",
        "behavior": "stale; excluded from live inference",
        "audio": "missing",
    }
    assert historical_evidence.modality_status == {
        "face": "ok",
        "behavior": "ok",
        "audio": "waiting",
    }
    assert historical_diagnostic["valid_feature_count"] == 4
    assert historical_diagnostic["available_modalities"] == {
        "face": True,
        "behavior": True,
        "audio": False,
    }
    assert historical_diagnostic["runtime_diagnostics"]["face_results_published"] == 1
    assert historical_diagnostic["runtime_diagnostics"]["behavior_results_published"] == 1

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "fusion_evidence = build_fusion_evidence(snapshot)" in source
    assert "diagnostic_evidence = build_fusion_evidence(raw_snapshot)" in source
