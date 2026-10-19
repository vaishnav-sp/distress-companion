"""Continuous local capture, branch processing, and background inference orchestration."""

from __future__ import annotations

from copy import deepcopy
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

import numpy as np

from alert_management import AlertManager
from branches.audio_branch import AudioFeatureBranch
from branches.behavior_branch import UpperBodyBehaviorBranch
from branches.face_branch import FaceExpressionBranch
from config import LIVE_AUDIO_INTERVAL_SECONDS, LIVE_CAMERA_INTERVAL_SECONDS
from microphone import MicrophoneCaptureStream, capture_audio_window
from monitoring_history import (
    create_alert_diagnostic_entry,
    create_history_entry,
)
from webcam import CameraCaptureSession, capture_frame

logger = logging.getLogger(__name__)


class MonitoringManager:
    """Manage a single bounded monitoring loop for local camera and microphone inputs."""

    def __init__(
        self,
        camera_interval_seconds: float = LIVE_CAMERA_INTERVAL_SECONDS,
        audio_interval_seconds: float = LIVE_AUDIO_INTERVAL_SECONDS,
        shutdown_timeout_seconds: float = 2.0,
        face_branch_factory: Callable[[], Any] | None = None,
        behavior_branch_factory: Callable[[], Any] | None = None,
        audio_branch_factory: Callable[[], Any] | None = None,
        face_branch: Any | None = None,
        behavior_branch: Any | None = None,
        audio_branch: Any | None = None,
        camera_session_factory: Callable[[], Any] | None = None,
        microphone_stream_factory: Callable[[], Any] | None = None,
        inference_processor: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        alert_manager: AlertManager | None = None,
    ) -> None:
        self.camera_interval_seconds = max(0.1, float(camera_interval_seconds))
        self.audio_interval_seconds = max(0.5, float(audio_interval_seconds))
        self.shutdown_timeout_seconds = max(0.1, float(shutdown_timeout_seconds))
        self.face_branch_factory = face_branch_factory or FaceExpressionBranch
        self.behavior_branch_factory = behavior_branch_factory or UpperBodyBehaviorBranch
        self.audio_branch_factory = audio_branch_factory or AudioFeatureBranch
        self.camera_session_factory = camera_session_factory
        self.microphone_stream_factory = microphone_stream_factory
        self.inference_processor = inference_processor
        self.alert_manager = alert_manager or AlertManager()

        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._camera_session: Any | None = None
        self._microphone_stream: Any | None = None
        self._capture_resources_initialized = False
        self.status = "stopped"
        self.message = "Monitoring is idle."
        self.camera_status = "idle"
        self.camera_message = "Camera has not been tested in monitoring mode."
        self.microphone_status = "idle"
        self.microphone_message = "Microphone has not been tested in monitoring mode."
        self.last_camera_timestamp: float | None = None
        self.last_microphone_timestamp: float | None = None
        self.last_camera_success_timestamp: float | None = None
        self.last_microphone_success_timestamp: float | None = None
        self.last_face_success_timestamp: float | None = None
        self.last_behavior_success_timestamp: float | None = None
        self.last_audio_success_timestamp: float | None = None
        self.camera_capture_attempts = 0
        self.camera_frames_captured = 0
        self.last_camera_opened: bool | None = None
        self.last_camera_frame_info: dict[str, Any] | None = None
        self.stage_timings: dict[str, dict[str, float | int]] = {}
        self._last_camera_capture_monotonic: float | None = None
        self.camera_success_count = 0
        self.audio_capture_attempts = 0
        self.audio_success_count = 0
        self.face_analysis_attempts = 0
        self.face_inference_invocations = 0
        self.face_results_returned = 0
        self.face_results_published = 0
        self.behavior_analysis_attempts = 0
        self.behavior_inference_invocations = 0
        self.behavior_results_returned = 0
        self.behavior_results_published = 0
        self.audio_analysis_attempts = 0
        self.audio_inference_invocations = 0
        self.audio_results_returned = 0
        self.audio_results_published = 0
        self.processing_iterations = 0
        self.current_operation = "idle"
        self.worker_exit_reason: str | None = None
        self._fusion_diagnostic_snapshot: dict[str, Any] | None = None
        self.latest_face_result: Any | None = None
        self.latest_behavior_result: Any | None = None
        self.latest_audio_result: Any | None = None
        self.latest_inference: Any | None = None
        self.latest_feature_record: Any | None = None
        self.latest_inference_key: str | None = None
        self.latest_inference_timestamp: float | None = None
        self.latest_inference_updated_at_utc: str | None = None
        self.latest_inference_error: str | None = None
        self.latest_alert = self.alert_manager.snapshot()
        self.alert_feature_record: dict[str, Any] | None = None
        self._inference_input_key: tuple[Any, ...] | None = None
        self.inference_history: list[dict[str, Any]] = []
        self._last_history_key: str | None = None
        self.alert_diagnostics: list[dict[str, Any]] = []
        self.face_branch: Any | None = face_branch
        self.behavior_branch: Any | None = behavior_branch
        self.audio_branch: Any | None = audio_branch
        self.pose_diagnostics_enabled = False
        self.last_error: dict[str, str | None] = {
            "camera": None,
            "microphone": None,
            "face": None,
            "behavior": None,
            "audio": None,
        }
        self.diagnostics: dict[str, dict[str, str | None]] = {
            name: {"stage": "idle", "exception_type": None, "message": None}
            for name in ("camera", "microphone", "face", "behavior", "audio", "worker")
        }
        self._last_camera_processed = 0.0
        self._last_audio_processed = 0.0

    def _initialize_branch(
        self,
        branch_name: str,
        branch: Any | None,
        branch_factory: Callable[[], Any] | None,
    ) -> tuple[Any | None, str | None, str | None]:
        if branch is not None:
            return branch, None, None
        if branch_factory is None:
            return None, None, None
        try:
            created = branch_factory()
        except Exception as exc:
            message = (
                f"{branch_name.capitalize()} branch initialization failed "
                f"({type(exc).__name__}): {exc}"
            )
            logger.exception("%s branch initialization failed", branch_name)
            return None, message, type(exc).__name__
        if created is None:
            message = f"{branch_name.capitalize()} branch initialization returned None."
            logger.error(message)
            return None, message, None
        initialization_error = getattr(created, "initialization_error", None)
        if initialization_error:
            message = (
                f"{branch_name.capitalize()} branch initialization failed: "
                f"{initialization_error}"
            )
            logger.error(message)
            return (
                created,
                message,
                getattr(created, "initialization_exception_type", None),
            )
        return created, None, None

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def snapshot(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of the current monitoring state."""
        with self._lock:
            return {
                "status": self.status,
                "message": self.message,
                "camera_status": self.camera_status,
                "camera_message": self.camera_message,
                "microphone_status": self.microphone_status,
                "microphone_message": self.microphone_message,
                "last_camera_timestamp": self.last_camera_timestamp,
                "last_microphone_timestamp": self.last_microphone_timestamp,
                "last_camera_success_timestamp": self.last_camera_success_timestamp,
                "last_microphone_success_timestamp": self.last_microphone_success_timestamp,
                "last_face_success_timestamp": self.last_face_success_timestamp,
                "last_behavior_success_timestamp": self.last_behavior_success_timestamp,
                "last_audio_success_timestamp": self.last_audio_success_timestamp,
                "camera_capture_attempts": self.camera_capture_attempts,
                "camera_frames_captured": self.camera_frames_captured,
                "last_camera_opened": self.last_camera_opened,
                "last_camera_frame_info": (
                    dict(self.last_camera_frame_info)
                    if self.last_camera_frame_info is not None
                    else None
                ),
                "stage_timings": {
                    name: dict(values)
                    for name, values in self.stage_timings.items()
                },
                "camera_success_count": self.camera_success_count,
                "audio_capture_attempts": self.audio_capture_attempts,
                "audio_success_count": self.audio_success_count,
                "face_analysis_attempts": self.face_analysis_attempts,
                "face_inference_invocations": self.face_inference_invocations,
                "face_results_returned": self.face_results_returned,
                "face_results_published": self.face_results_published,
                "behavior_analysis_attempts": self.behavior_analysis_attempts,
                "behavior_inference_invocations": self.behavior_inference_invocations,
                "behavior_results_returned": self.behavior_results_returned,
                "behavior_results_published": self.behavior_results_published,
                "audio_analysis_attempts": self.audio_analysis_attempts,
                "audio_inference_invocations": self.audio_inference_invocations,
                "audio_results_returned": self.audio_results_returned,
                "audio_results_published": self.audio_results_published,
                "processing_iterations": self.processing_iterations,
                "current_operation": self.current_operation,
                "worker_alive": self.is_running,
                "worker_exit_reason": self.worker_exit_reason,
                "fusion_diagnostic_snapshot": deepcopy(
                    self._fusion_diagnostic_snapshot
                ),
                "latest_face_result": self.latest_face_result,
                "latest_behavior_result": self.latest_behavior_result,
                "latest_audio_result": self.latest_audio_result,
                "latest_inference": self.latest_inference,
                "latest_feature_record": self.latest_feature_record,
                "latest_inference_key": self.latest_inference_key,
                "latest_inference_timestamp": self.latest_inference_timestamp,
                "latest_inference_updated_at_utc": self.latest_inference_updated_at_utc,
                "latest_inference_error": self.latest_inference_error,
                "latest_alert": self.latest_alert,
                "alert_feature_record": deepcopy(self.alert_feature_record),
                "inference_history": deepcopy(self.inference_history),
                "alert_diagnostics": deepcopy(self.alert_diagnostics),
                "last_error": dict(self.last_error),
                "diagnostics": {
                    name: dict(diagnostic)
                    for name, diagnostic in self.diagnostics.items()
                },
            }

    def store_fusion_diagnostic_snapshot(
        self,
        diagnostic_snapshot: Mapping[str, Any],
    ) -> None:
        """Retain a detached diagnostic copy without exposing it to live inference."""
        with self._lock:
            self._fusion_diagnostic_snapshot = deepcopy(dict(diagnostic_snapshot))

    def _set_diagnostic(
        self,
        name: str,
        stage: str,
        message: str | None,
        exception_type: str | None = None,
    ) -> None:
        self.diagnostics[name] = {
            "stage": stage,
            "exception_type": exception_type,
            "message": message,
        }

    def _record_timing(self, stage: str, duration_ms: float) -> None:
        """Retain aggregate stage timings only; no sensor data is stored."""
        with self._lock:
            previous = self.stage_timings.get(stage)
            if previous is None:
                self.stage_timings[stage] = {
                    "last_ms": duration_ms,
                    "mean_ms": duration_ms,
                    "max_ms": duration_ms,
                    "count": 1,
                }
                return
            count = int(previous["count"]) + 1
            mean = float(previous["mean_ms"]) + (
                duration_ms - float(previous["mean_ms"])
            ) / count
            self.stage_timings[stage] = {
                "last_ms": duration_ms,
                "mean_ms": mean,
                "max_ms": max(float(previous["max_ms"]), duration_ms),
                "count": count,
            }

    def _reset_runtime_state(self) -> None:
        """Reset the live monitoring state so the UI reflects the next real execution."""
        self.status = "starting"
        self.message = "Starting local monitoring..."
        self.camera_status = "starting"
        self.camera_message = "Waiting for first camera frame."
        self.microphone_status = "starting"
        self.microphone_message = "Waiting for first audio window."
        self.last_camera_timestamp = None
        self.last_microphone_timestamp = None
        self.last_camera_success_timestamp = None
        self.last_microphone_success_timestamp = None
        self.last_face_success_timestamp = None
        self.last_behavior_success_timestamp = None
        self.last_audio_success_timestamp = None
        self.camera_capture_attempts = 0
        self.camera_frames_captured = 0
        self.last_camera_opened = None
        self.last_camera_frame_info = None
        self.stage_timings.clear()
        self._last_camera_capture_monotonic = None
        self.camera_success_count = 0
        self.audio_capture_attempts = 0
        self.audio_success_count = 0
        self.face_analysis_attempts = 0
        self.face_inference_invocations = 0
        self.face_results_returned = 0
        self.face_results_published = 0
        self.behavior_analysis_attempts = 0
        self.behavior_inference_invocations = 0
        self.behavior_results_returned = 0
        self.behavior_results_published = 0
        self.audio_analysis_attempts = 0
        self.audio_inference_invocations = 0
        self.audio_results_returned = 0
        self.audio_results_published = 0
        self.processing_iterations = 0
        self.current_operation = "starting"
        self.worker_exit_reason = None
        self.latest_face_result = None
        self.latest_behavior_result = None
        self.latest_audio_result = None
        self.latest_inference = None
        self.latest_feature_record = None
        self.latest_inference_key = None
        self.latest_inference_timestamp = None
        self.latest_inference_updated_at_utc = None
        self.latest_inference_error = None
        self.latest_alert = self.alert_manager.report_unknown(
            "Monitoring is starting; waiting for fresh evidence."
        )
        self._inference_input_key = None
        self.last_face_success_timestamp = None
        self.last_behavior_success_timestamp = None
        self.last_audio_success_timestamp = None
        self.last_error = {"camera": None, "microphone": None, "face": None, "behavior": None, "audio": None}
        self.diagnostics = {
            name: {"stage": "idle", "exception_type": None, "message": None}
            for name in ("camera", "microphone", "face", "behavior", "audio", "worker")
        }
        self._last_camera_processed = 0.0
        self._last_audio_processed = 0.0
        self._capture_resources_initialized = False
        self._reset_inference_processor()

    def _reset_inference_processor(self) -> None:
        reset = getattr(self.inference_processor, "reset", None)
        if not callable(reset):
            return
        try:
            reset()
        except Exception as exc:
            logger.exception("Unable to reset the live inference processor")
            message = f"{type(exc).__name__}: {exc}"
            self.latest_inference_error = message
            self._set_diagnostic(
                "worker",
                "inference_processor_reset_failure",
                str(exc),
                type(exc).__name__,
            )

    def set_pose_diagnostics_enabled(self, enabled: bool) -> None:
        """Enable temporary diagnostics on the existing live behavior branch."""
        with self._lock:
            self.pose_diagnostics_enabled = bool(enabled)
            if self.behavior_branch is not None:
                self.behavior_branch.diagnostics_enabled = self.pose_diagnostics_enabled

    def start(self) -> bool:
        """Start the monitoring loop once and only once per manager instance."""
        with self._lock:
            if self.is_running:
                return False
            self._stop_event.clear()
            self._reset_runtime_state()
            self.face_branch, face_error, face_error_type = self._initialize_branch(
                "face",
                self.face_branch,
                self.face_branch_factory,
            )
            self.behavior_branch, behavior_error, behavior_error_type = self._initialize_branch(
                "behavior",
                self.behavior_branch,
                self.behavior_branch_factory,
            )
            if self.behavior_branch is not None:
                self.behavior_branch.diagnostics_enabled = (
                    self.pose_diagnostics_enabled
                )
            self.audio_branch, audio_error, audio_error_type = self._initialize_branch(
                "audio",
                self.audio_branch,
                self.audio_branch_factory,
            )
            if face_error is not None:
                self.last_error["face"] = face_error
                self._set_diagnostic(
                    "face", "branch_initialization_failure", face_error, face_error_type
                )
            if behavior_error is not None:
                self.last_error["behavior"] = behavior_error
                self._set_diagnostic(
                    "behavior",
                    "branch_initialization_failure",
                    behavior_error,
                    behavior_error_type,
                )
            if audio_error is not None:
                self.last_error["audio"] = audio_error
                self._set_diagnostic(
                    "audio", "branch_initialization_failure", audio_error, audio_error_type
                )
            self._worker = threading.Thread(
                target=self._run_loop,
                name="distress-monitoring",
                daemon=True,
            )
            self._worker.start()
            return True

    def stop(self) -> None:
        """Request a clean stop and release owned resources."""
        with self._lock:
            self._stop_event.set()
            unexpected_exit = self.worker_exit_reason == "unexpected_exit"
            if not unexpected_exit:
                self.worker_exit_reason = "stop_requested"
                self._set_diagnostic(
                    "worker",
                    "stop_requested",
                    "Monitoring was stopped by an explicit stop request.",
                )
            worker = self._worker
            if worker is not None and worker.is_alive():
                self.status = "stopping"
                self.message = "Waiting for the monitoring worker to stop."
            else:
                self._set_stopped_state()

        if worker is not None and worker.is_alive():
            worker.join(timeout=self.shutdown_timeout_seconds)
            if worker.is_alive():
                with self._lock:
                    self.status = "stopping"
                    self.message = (
                        "The monitoring worker has not stopped yet; waiting for the "
                        "in-progress capture to finish."
                    )
                return

            with self._lock:
                if not unexpected_exit:
                    self._set_stopped_state()
                if self._worker is worker:
                    self._worker = None
            self._release_branches()
            return

        with self._lock:
            if self._worker is worker:
                self._worker = None
            unexpected_exit = self.worker_exit_reason == "unexpected_exit"
        if not unexpected_exit:
            self._release_branches()

    def _set_stopped_state(self) -> None:
        self.status = "stopped"
        self.message = "Monitoring stopped and resources were released."
        self.camera_status = "stopped"
        self.camera_message = "Monitoring stopped; no new frames are being captured."
        self.microphone_status = "stopped"
        self.microphone_message = "Monitoring stopped; no new audio windows are being processed."
        self.latest_face_result = None
        self.latest_behavior_result = None
        self.latest_audio_result = None
        self.latest_inference = None
        self.latest_feature_record = None
        self.latest_inference_key = None
        self.latest_inference_timestamp = None
        self.latest_inference_updated_at_utc = None
        self.latest_inference_error = None
        self.latest_alert = self.alert_manager.report_unknown(
            "Monitoring is stopped; no fresh inference is being produced."
        )
        self.last_face_success_timestamp = None
        self.last_behavior_success_timestamp = None
        self.last_audio_success_timestamp = None
        self._last_camera_processed = 0.0
        self._last_audio_processed = 0.0
        self.current_operation = "stopped"
        self._set_diagnostic(
            "worker",
            "stopped",
            "Monitoring stopped; live results were cleared to prevent stale inference.",
        )
        self._reset_inference_processor()

    def _release_branches(self) -> None:
        with self._lock:
            branches = (
                ("face", self.face_branch),
                ("behavior", self.behavior_branch),
                ("audio", self.audio_branch),
            )
            self.face_branch = None
            self.behavior_branch = None
            self.audio_branch = None

        for branch_name, branch in branches:
            try:
                close = getattr(branch, "close", None)
                if callable(close):
                    close()
            except Exception as exc:
                logger.exception("Unable to close the %s branch", branch_name)
                with self._lock:
                    message = f"{type(exc).__name__}: {exc}"
                    self.last_error[branch_name] = message
                    self._set_diagnostic(
                        branch_name,
                        "branch_resource_release_failure",
                        str(exc),
                        type(exc).__name__,
                    )

    def _open_capture_resources(self) -> None:
        with self._lock:
            self._capture_resources_initialized = True
        for label, factory in (
            ("camera", self.camera_session_factory or CameraCaptureSession),
            ("microphone", self.microphone_stream_factory or MicrophoneCaptureStream),
        ):
            if self._stop_event.is_set():
                return
            try:
                resource = factory()
                with self._lock:
                    if label == "camera":
                        self._camera_session = resource
                        self.last_camera_opened = getattr(resource, "is_opened", None)
                        self.camera_status = (
                            "running" if self.last_camera_opened is not False else "unavailable"
                        )
                        self.camera_message = (
                            "Camera resource is open for continuous capture."
                            if self.last_camera_opened is not False
                            else "Camera is unavailable; monitoring will retry capture."
                        )
                    else:
                        self._microphone_stream = resource
                        self.microphone_status = "running"
                        self.microphone_message = "Microphone stream is open for continuous capture."
            except Exception as exc:
                logger.exception("Unable to open the %s monitoring resource", label)
                with self._lock:
                    message = f"{type(exc).__name__}: {exc}"
                    if label == "camera":
                        self.camera_status = "unavailable"
                        self.camera_message = f"Camera unavailable: {message}"
                        self.last_error["camera"] = message
                    else:
                        self.microphone_status = "unavailable"
                        self.microphone_message = f"Microphone unavailable: {message}"
                        self.last_error["microphone"] = message
                    self._set_diagnostic(
                        label,
                        "sensor_capture_failure",
                        message,
                        type(exc).__name__,
                    )

    def _release_capture_resources(self) -> None:
        with self._lock:
            resources = (
                ("camera", self._camera_session),
                ("microphone", self._microphone_stream),
            )
            self._camera_session = None
            self._microphone_stream = None
        for label, resource in resources:
            if resource is None:
                continue
            try:
                resource.close()
            except Exception as exc:
                logger.exception("Unable to close the %s monitoring resource", label)
                with self._lock:
                    message = f"{type(exc).__name__}: {exc}"
                    self.last_error[label] = message
                    self._set_diagnostic(
                        label,
                        "sensor_resource_release_failure",
                        str(exc),
                        type(exc).__name__,
                    )

    def reset(self) -> None:
        """Clear the live results without altering active capture state."""
        with self._lock:
            self.latest_face_result = None
            self.latest_behavior_result = None
            self.latest_audio_result = None
            self.camera_message = "Monitoring reset; next frame will refresh the latest status."
            self.microphone_message = "Monitoring reset; next audio window will refresh the latest status."
            if self.behavior_branch is not None:
                try:
                    self.behavior_branch.reset()
                except Exception as exc:
                    logger.exception("Unable to reset the behavior baseline during monitoring reset")
                    message = f"{type(exc).__name__}: {exc}"
                    self.last_error["behavior"] = message
                    self._set_diagnostic(
                        "behavior",
                        "branch_reset_failure",
                        str(exc),
                        type(exc).__name__,
                    )
            if self.audio_branch is not None:
                try:
                    self.audio_branch.reset()
                except Exception as exc:
                    logger.exception("Unable to reset the audio baseline during monitoring reset")
                    message = f"{type(exc).__name__}: {exc}"
                    self.last_error["audio"] = message
                    self._set_diagnostic(
                        "audio",
                        "branch_reset_failure",
                        str(exc),
                        type(exc).__name__,
                    )

    def _run_loop(self) -> None:
        """Process camera and audio windows in a single bounded loop until stopped."""
        self._open_capture_resources()
        with self._lock:
            if not self._stop_event.is_set():
                self.status = "running"
                self.message = "Monitoring is active."
                if self.camera_status != "unavailable":
                    self.camera_status = "running"
                    self.camera_message = "Waiting for first camera frame."
                if self.microphone_status != "unavailable":
                    self.microphone_status = "running"
                    self.microphone_message = "Waiting for first audio window."
                self.current_operation = "waiting for next capture"
        last_camera_update = 0.0
        last_audio_update = 0.0

        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                with self._lock:
                    self.processing_iterations += 1
                if now - last_camera_update >= self.camera_interval_seconds:
                    self._check_camera()
                    last_camera_update = now
                if (
                    not self._stop_event.is_set()
                    and now - last_audio_update >= self.audio_interval_seconds
                ):
                    self._check_microphone()
                    last_audio_update = now
                self._refresh_live_inference()
                time.sleep(0.05)
        except Exception as exc:  # pragma: no cover - defensive guard for worker lifecycle
            logger.exception("Monitoring worker encountered an unexpected exception")
            with self._lock:
                self.status = "error"
                self.message = (
                    "Monitoring stopped because of an unexpected worker error "
                    f"({type(exc).__name__}): {exc}"
                )
                self.worker_exit_reason = "unexpected_exit"
                self.current_operation = "worker exited unexpectedly"
                self._set_diagnostic(
                    "worker",
                    "unexpected_worker_exit",
                    str(exc),
                    type(exc).__name__,
                )
        finally:
            self._release_capture_resources()
            self._release_branches()
            with self._lock:
                if self._stop_event.is_set() and self.status != "error":
                    self._set_stopped_state()

    def _check_camera(self) -> None:
        camera_cycle_started = time.perf_counter()
        capture_started = time.perf_counter()
        capture_timing_recorded = False
        with self._lock:
            self.camera_capture_attempts += 1
            self.last_camera_timestamp = time.monotonic()
            self.camera_status = "capturing"
            self.camera_message = "Attempting to capture a camera frame."
            self.current_operation = "capturing camera frame"
        try:
            if self._capture_resources_initialized:
                if self._camera_session is None and not self._stop_event.is_set():
                    factory = self.camera_session_factory or CameraCaptureSession
                    self._camera_session = factory()
                capture = (
                    self._camera_session.read()
                    if self._camera_session is not None
                    else None
                )
            else:
                capture = capture_frame()
            camera_timestamp = time.monotonic()
            self._record_timing(
                "camera_capture",
                (time.perf_counter() - capture_started) * 1000.0,
            )
            capture_timing_recorded = True
            with self._lock:
                previous_capture = self._last_camera_capture_monotonic
                self._last_camera_capture_monotonic = camera_timestamp
            if previous_capture is not None and camera_timestamp > previous_capture:
                self._record_timing(
                    "camera_intercapture",
                    (camera_timestamp - previous_capture) * 1000.0,
                )
            with self._lock:
                self.last_camera_timestamp = camera_timestamp
                self.last_camera_opened = getattr(capture, "camera_opened", None)
            if self._stop_event.is_set():
                return
            if capture is None or capture.frame is None:
                with self._lock:
                    self.camera_status = "unavailable"
                    self.camera_message = capture.error if capture is not None and capture.error else "No frame was captured."
                    self.last_error["camera"] = self.camera_message
                    self._set_diagnostic(
                        "camera",
                        "sensor_capture_failure",
                        self.camera_message,
                        getattr(capture, "error_type", None),
                    )
                return
            frame = capture.frame
            shape = getattr(frame, "shape", None)
            shape_tuple = tuple(int(value) for value in shape) if shape is not None else ()
            channels = (
                int(shape_tuple[2])
                if len(shape_tuple) == 3
                else (1 if len(shape_tuple) == 2 else None)
            )
            frame_dtype = str(getattr(frame, "dtype", "unknown"))
            valid_frame = (
                isinstance(frame, np.ndarray)
                and frame.dtype == np.uint8
                and (
                    frame.ndim == 2
                    or (frame.ndim == 3 and frame.shape[2] in {3, 4})
                )
                and frame.shape[0] > 0
                and frame.shape[1] > 0
            )
            with self._lock:
                self.camera_frames_captured += 1
                self.last_camera_frame_info = {
                    "shape": shape_tuple,
                    "channels": channels,
                    "dtype": frame_dtype,
                    "channel_order": (
                        "BGR"
                        if len(shape_tuple) == 3 and channels == 3
                        else "BGRA"
                        if len(shape_tuple) == 3 and channels == 4
                        else "grayscale"
                        if len(shape_tuple) == 2
                        else "unknown"
                    ),
                    "camera_index": getattr(capture, "camera_index", None),
                    "valid_for_branch_input": bool(valid_frame),
                }
                self._set_diagnostic(
                    "camera",
                    "frame_captured",
                    (
                        f"Captured frame shape={shape_tuple}, channels={channels}, "
                        f"dtype={frame_dtype}, valid_for_branch_input={valid_frame}."
                    ),
                )

            face_result = None
            if self.face_branch is not None:
                branch_started = time.perf_counter()
                try:
                    with self._lock:
                        self.face_analysis_attempts += 1
                        self.face_inference_invocations += 1
                        self.current_operation = "running face analysis"
                        self._set_diagnostic(
                            "face",
                            "inference_invoked",
                            "Face branch analyze() was invoked with the captured frame.",
                        )
                    face_result = self.face_branch.analyze(frame)
                    self._publish_result("face", face_result, camera_timestamp)
                    with self._lock:
                        if getattr(face_result, "status_message", None):
                            self.camera_message = face_result.status_message
                        self.camera_status = "ok" if getattr(face_result, "input_quality", True) else "partial"
                        if getattr(face_result, "abstention_reason", None):
                            self.camera_message = face_result.status_message
                except Exception as exc:
                    logger.exception("Face-inference step failed during monitoring")
                    with self._lock:
                        self.camera_status = "error"
                        self.camera_message = f"Face analysis failed: {exc}"
                        self.last_error["face"] = f"{type(exc).__name__}: {exc}"
                        self._set_diagnostic(
                            "face",
                            "branch_inference_exception",
                            str(exc),
                            type(exc).__name__,
                        )
                finally:
                    self._record_timing(
                        "face_branch",
                        (time.perf_counter() - branch_started) * 1000.0,
                    )
            else:
                with self._lock:
                    self.last_error["face"] = self.last_error.get("face") or (
                        "Face branch is not initialized; camera processing continues "
                        "without face inference."
                    )
                    self._set_diagnostic(
                        "face",
                        "branch_initialization_failure",
                        self.last_error["face"],
                    )
                    self.camera_message = self.last_error["face"]

            behavior_result = None
            if self.behavior_branch is not None:
                branch_started = time.perf_counter()
                try:
                    with self._lock:
                        self.behavior_analysis_attempts += 1
                        self.behavior_inference_invocations += 1
                        self.current_operation = "running behavior analysis"
                        self._set_diagnostic(
                            "behavior",
                            "inference_invoked",
                            "Behavior branch analyze() was invoked with the captured frame.",
                        )
                    behavior_result = self.behavior_branch.analyze(
                        frame,
                        timestamp_seconds=camera_timestamp,
                    )
                    pose_diagnostics = getattr(
                        behavior_result, "pose_diagnostic", {}
                    )
                    for stage, duration_ms in pose_diagnostics.get(
                        "stage_timings_ms", {}
                    ).items():
                        self._record_timing(stage, duration_ms)
                    self._publish_result("behavior", behavior_result, camera_timestamp)
                    with self._lock:
                        if behavior_result is not None and getattr(behavior_result, "status_message", None):
                            self.camera_message = behavior_result.status_message
                        if behavior_result is not None and behavior_result.status in {"partial", "warming_up", "ok"}:
                            self.camera_status = behavior_result.status
                except Exception as exc:
                    logger.exception("Behavior inference failed during monitoring")
                    with self._lock:
                        self.camera_status = "error"
                        self.camera_message = f"Behavior analysis failed: {exc}"
                        self.last_error["behavior"] = f"{type(exc).__name__}: {exc}"
                        self._set_diagnostic(
                            "behavior",
                            "branch_inference_exception",
                            str(exc),
                            type(exc).__name__,
                        )
                finally:
                    self._record_timing(
                        "behavior_branch",
                        (time.perf_counter() - branch_started) * 1000.0,
                    )
            else:
                with self._lock:
                    self.last_error["behavior"] = self.last_error.get("behavior") or (
                        "Behavior branch is not initialized; camera processing continues "
                        "without behavior inference."
                    )
                    self._set_diagnostic(
                        "behavior",
                        "branch_initialization_failure",
                        self.last_error["behavior"],
                    )
                    self.camera_message = self.last_error["behavior"]

            with self._lock:
                if face_result is not None or behavior_result is not None:
                    self.camera_success_count += 1
                    self.last_camera_success_timestamp = camera_timestamp
                    self.last_error["camera"] = None
        except Exception as exc:
            if not capture_timing_recorded:
                self._record_timing(
                    "camera_capture",
                    (time.perf_counter() - capture_started) * 1000.0,
                )
            logger.exception("Camera acquisition failed during monitoring")
            with self._lock:
                self.camera_status = "error"
                self.camera_message = f"Camera capture failed: {exc}"
                self.last_error["camera"] = f"{type(exc).__name__}: {exc}"
                self._set_diagnostic(
                    "camera",
                    "sensor_capture_failure",
                    str(exc),
                    type(exc).__name__,
                )
        finally:
            self._record_timing(
                "camera_processing_cycle",
                (time.perf_counter() - camera_cycle_started) * 1000.0,
            )
            with self._lock:
                if not self._stop_event.is_set():
                    self.current_operation = "waiting for next capture"

    def _check_microphone(self) -> None:
        microphone_cycle_started = time.perf_counter()
        if self.audio_branch is None:
            with self._lock:
                self.microphone_status = "unavailable"
                self.microphone_message = self.last_error.get("audio") or (
                    "Audio branch is not initialized; microphone processing continues "
                    "without audio inference."
                )
                self.last_error["audio"] = self.microphone_message
                self.last_error["microphone"] = (
                    self.last_error.get("microphone") or self.microphone_message
                )
            return

        with self._lock:
            self.audio_capture_attempts += 1
            self.last_microphone_timestamp = time.monotonic()
            self.microphone_status = "capturing"
            self.microphone_message = "Attempting to capture an audio window."
            self.current_operation = "capturing microphone audio"
        capture_started = time.perf_counter()
        capture_timing_recorded = False
        try:
            if self._capture_resources_initialized:
                if self._microphone_stream is None and not self._stop_event.is_set():
                    factory = self.microphone_stream_factory or MicrophoneCaptureStream
                    self._microphone_stream = factory()
                capture = (
                    self._microphone_stream.read_window(self._stop_event)
                    if self._microphone_stream is not None
                    else None
                )
            else:
                capture = capture_audio_window()
            self._record_timing(
                "audio_capture",
                (time.perf_counter() - capture_started) * 1000.0,
            )
            capture_timing_recorded = True
            microphone_timestamp = time.monotonic()
            with self._lock:
                self.last_microphone_timestamp = microphone_timestamp
            if self._stop_event.is_set():
                return
            if capture is None:
                with self._lock:
                    self.microphone_status = "unavailable"
                    self.microphone_message = "Audio capture returned no data."
                    self.last_error["microphone"] = self.microphone_message
                    self._set_diagnostic(
                        "microphone",
                        "sensor_capture_failure",
                        self.microphone_message,
                    )
                return
            if capture.status in {"missing", "unknown"}:
                with self._lock:
                    self.microphone_status = "unavailable"
                    self.microphone_message = capture.message
                    self.last_error["microphone"] = capture.message
                    self._set_diagnostic(
                        "microphone",
                        "sensor_capture_failure",
                        capture.message,
                        getattr(capture, "error_type", None),
                    )
                return
            try:
                with self._lock:
                    self.audio_analysis_attempts += 1
                    self.audio_inference_invocations += 1
                    self.current_operation = "running audio analysis"
                branch_started = time.perf_counter()
                try:
                    audio_result = self.audio_branch.analyze(capture)
                finally:
                    self._record_timing(
                        "audio_feature_extraction",
                        (time.perf_counter() - branch_started) * 1000.0,
                    )
                self._publish_result("audio", audio_result, microphone_timestamp)
                with self._lock:
                    if audio_result is not None:
                        self.audio_success_count += 1
                        self.last_microphone_success_timestamp = microphone_timestamp
                    if capture.status == "silent":
                        self.microphone_status = "silent"
                    elif audio_result is not None:
                        self.microphone_status = audio_result.status
                    else:
                        self.microphone_status = "error"
                    self.microphone_message = (
                        audio_result.status_message
                        if audio_result is not None and getattr(audio_result, "status_message", None)
                        else capture.message
                    )
                    if audio_result is not None:
                        self.last_error["microphone"] = None
                    else:
                        self.last_error["microphone"] = "Audio analysis returned no result."
            except Exception as exc:
                logger.exception("Audio-analysis step failed during monitoring")
                with self._lock:
                    self.microphone_status = "error"
                    self.microphone_message = f"Audio analysis failed: {exc}"
                    self.last_error["audio"] = f"{type(exc).__name__}: {exc}"
                    self._set_diagnostic(
                        "audio",
                        "branch_inference_exception",
                        str(exc),
                        type(exc).__name__,
                    )
        except Exception as exc:
            if not capture_timing_recorded:
                self._record_timing(
                    "audio_capture",
                    (time.perf_counter() - capture_started) * 1000.0,
                )
            logger.exception("Microphone capture processing failed during monitoring")
            with self._lock:
                self.microphone_status = "error"
                self.microphone_message = f"Microphone processing failed: {exc}"
                self.last_error["microphone"] = f"{type(exc).__name__}: {exc}"
                self._set_diagnostic(
                    "microphone",
                    "sensor_capture_failure",
                    str(exc),
                    type(exc).__name__,
                )
        finally:
            self._record_timing(
                "audio_processing_cycle",
                (time.perf_counter() - microphone_cycle_started) * 1000.0,
            )
            with self._lock:
                if not self._stop_event.is_set():
                    self.current_operation = "waiting for next capture"

    def _publish_result(self, modality: str, result: Any, timestamp: float) -> bool:
        """Publish a branch result atomically and retain publication diagnostics."""
        try:
            with self._lock:
                if modality == "face":
                    if result is not None:
                        self.face_results_returned += 1
                    self.latest_face_result = result
                    if result is not None:
                        self.face_results_published += 1
                        self.last_face_success_timestamp = timestamp
                elif modality == "behavior":
                    if result is not None:
                        self.behavior_results_returned += 1
                    self.latest_behavior_result = result
                    if result is not None:
                        self.behavior_results_published += 1
                        self.last_behavior_success_timestamp = timestamp
                elif modality == "audio":
                    if result is not None:
                        self.audio_results_returned += 1
                    self.latest_audio_result = result
                    if result is not None:
                        self.audio_results_published += 1
                        self.last_audio_success_timestamp = timestamp
                else:
                    raise ValueError(f"Unsupported result modality: {modality}")

                if result is None:
                    message = f"{modality.capitalize()} analysis returned no result."
                    self.last_error[modality] = message
                    self._set_diagnostic(modality, "valid_inference_no_result", message)
                    return False
                category, message = self._classify_result(modality, result)
                exception_type = (
                    self._exception_type_from_message(message)
                    if category in {
                        "branch_initialization_failure",
                        "branch_inference_exception",
                    }
                    else None
                )
                self._set_diagnostic(
                    modality,
                    category,
                    message,
                    exception_type,
                )
                self.last_error[modality] = (
                    message
                    if category in {
                        "branch_initialization_failure",
                        "branch_inference_exception",
                    }
                    else None
                )
                return True
        except Exception as exc:
            logger.exception("Unable to publish the %s branch result", modality)
            with self._lock:
                message = f"{type(exc).__name__}: {exc}"
                if modality in self.last_error:
                    self.last_error[modality] = message
                    self._set_diagnostic(
                        modality,
                        "result_publication_failure",
                        str(exc),
                        type(exc).__name__,
                    )
            return False

    def _refresh_live_inference(self) -> None:
        if self.inference_processor is None or self._stop_event.is_set():
            return
        with self._lock:
            state = {
                "status": self.status,
                "latest_face_result": self.latest_face_result,
                "latest_behavior_result": self.latest_behavior_result,
                "latest_audio_result": self.latest_audio_result,
                "last_face_success_timestamp": self.last_face_success_timestamp,
                "last_behavior_success_timestamp": self.last_behavior_success_timestamp,
                "last_audio_success_timestamp": self.last_audio_success_timestamp,
                "camera_capture_attempts": self.camera_capture_attempts,
                "audio_capture_attempts": self.audio_capture_attempts,
                "camera_status": self.camera_status,
                "microphone_status": self.microphone_status,
            }
            input_key = (
                state["last_face_success_timestamp"],
                state["last_behavior_success_timestamp"],
                state["last_audio_success_timestamp"],
                state["camera_capture_attempts"],
                state["audio_capture_attempts"],
            )
            changed = input_key != self._inference_input_key
            previous_inference = self.latest_inference
            previous_timestamp = self.latest_inference_timestamp
            previous_key = self.latest_inference_key
        if changed:
            inference_started = time.perf_counter()
            try:
                result = dict(self.inference_processor(state))
                self._record_timing(
                    "evidence_and_rule_pipeline",
                    (time.perf_counter() - inference_started) * 1000.0,
                )
                for stage, duration_ms in result.get(
                    "stage_timings_ms", {}
                ).items():
                    self._record_timing(stage, duration_ms)
                inference = result.get("inference")
                source_timestamp = result.get("source_timestamp")
                result_key = result.get("result_key")
                previous_alert = self.latest_alert
                alert_before = self.alert_manager.diagnostic_state()
                alert = self.alert_manager.update(
                    inference,
                    result_timestamp=source_timestamp,
                    result_key=result_key,
                )
                alert_after = self.alert_manager.diagnostic_state()
                diagnostic_entry = None
                try:
                    diagnostic_entry = create_alert_diagnostic_entry(
                        inference,
                        result_key=result_key,
                        inference_timestamp_utc=result.get("updated_at_utc"),
                        source_timestamp=source_timestamp,
                        modality_source_timestamps=result.get(
                            "modality_source_timestamps",
                            {
                                modality: state.get(
                                    f"last_{modality}_success_timestamp"
                                )
                                for modality in ("face", "behavior", "audio")
                            },
                        ),
                        modality_results_present={
                            modality: state.get(f"latest_{modality}_result") is not None
                            for modality in ("face", "behavior", "audio")
                        },
                        observed_at_monotonic=time.monotonic(),
                        evidence_sufficient=result.get("evidence_sufficient"),
                        evidence_sufficiency_reason=result.get(
                            "evidence_sufficiency_reason"
                        ),
                        before=alert_before,
                        after=alert_after,
                        alert_state_after=alert.state,
                        alert_reason_after=alert.reason,
                    )
                except Exception:
                    logger.exception("Unable to record alert decision diagnostics")
                with self._lock:
                    self._inference_input_key = input_key
                    self.latest_inference = inference
                    self.latest_feature_record = result.get("feature_record")
                    self.latest_inference_key = (
                        str(result_key) if result_key is not None else None
                    )
                    self.latest_inference_timestamp = source_timestamp
                    self.latest_inference_updated_at_utc = result.get(
                        "updated_at_utc"
                    )
                    self.latest_inference_error = result.get("error")
                    self.latest_alert = alert
                    if (
                        alert.state == "potential_distress"
                        and alert.alert_id
                        and alert.alert_id != previous_alert.alert_id
                    ):
                        self.alert_feature_record = {
                            "alert_id": alert.alert_id,
                            "alert": deepcopy(alert),
                            "record": deepcopy(result.get("feature_record")),
                        }
                    if diagnostic_entry is not None:
                        self.alert_diagnostics.append(diagnostic_entry)
                    key_text = str(result_key) if result_key is not None else None
                    if inference is not None and key_text != self._last_history_key:
                        self.inference_history.append(
                            create_history_entry(
                                inference,
                                alert,
                                result_key=key_text,
                                timestamp=result.get("updated_at_utc")
                                or datetime.now(timezone.utc).isoformat(),
                            )
                        )
                        self._last_history_key = key_text
                    self._set_diagnostic(
                        "worker",
                        "inference_published",
                        "Latest inference and alert state were updated in the monitoring worker.",
                    )
            except Exception as exc:
                self._record_timing(
                    "evidence_and_rule_pipeline",
                    (time.perf_counter() - inference_started) * 1000.0,
                )
                logger.exception("Background fusion inference failed")
                with self._lock:
                    self._inference_input_key = input_key
                    self.latest_inference = None
                    self.latest_feature_record = None
                    self.latest_inference_key = None
                    self.latest_inference_timestamp = None
                    self.latest_inference_updated_at_utc = None
                    self.latest_inference_error = f"{type(exc).__name__}: {exc}"
                    self.latest_alert = self.alert_manager.report_unknown(
                        f"Background inference failed ({type(exc).__name__}): {exc}"
                    )
                    self._set_diagnostic(
                        "worker",
                        "inference_exception",
                        str(exc),
                        type(exc).__name__,
                    )
        elif previous_inference is not None:
            alert = self.alert_manager.update(
                previous_inference,
                result_timestamp=previous_timestamp,
                result_key=previous_key,
            )
            with self._lock:
                self.latest_alert = alert

    @staticmethod
    def _classify_result(modality: str, result: Any) -> tuple[str, str | None]:
        """Distinguish a published usable result from an explicit abstention."""
        status_message = getattr(result, "status_message", None)
        if modality == "face":
            reason = getattr(result, "abstention_reason", None)
            if reason == "model_unavailable":
                return "branch_initialization_failure", status_message
            if reason in {"inference_error", "face_detection_error"}:
                return "branch_inference_exception", status_message
            if reason is not None:
                return "valid_inference_no_usable_result", status_message
        elif modality == "behavior":
            reasons = getattr(result, "unavailable_reasons", {})
            reason_values = set(reasons.values()) if isinstance(reasons, dict) else set()
            if "landmarker_unavailable" in reason_values:
                return "branch_initialization_failure", status_message
            if "inference_error" in reason_values:
                return "branch_inference_exception", status_message
            if getattr(result, "status", None) in {"unknown", "missing"}:
                return "valid_inference_no_usable_result", status_message
        elif getattr(result, "status", None) in {"unknown", "missing"}:
            return "valid_inference_no_usable_result", status_message

        validity = getattr(result, "feature_validity", None)
        if isinstance(validity, dict) and not any(validity.values()):
            return "valid_inference_no_usable_result", status_message
        return "result_published", status_message

    @staticmethod
    def _exception_type_from_message(message: str | None) -> str | None:
        if message is None:
            return None
        start = message.find("(")
        end = message.find("):", start + 1)
        if start < 0 or end < 0:
            return None
        return message[start + 1 : end]


__all__ = ["MonitoringManager"]
