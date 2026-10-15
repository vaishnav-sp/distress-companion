"""Local upper-body landmark and movement feature extraction."""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from branches.interfaces import BranchOutput, BranchProcessor
from config import (
    BEHAVIOR_BASELINE_MIN_SAMPLES,
    BEHAVIOR_BASELINE_WINDOW,
    BEHAVIOR_ELEVATED_SPEED_THRESHOLD,
    BEHAVIOR_LANDMARK_VISIBILITY_THRESHOLD,
    BEHAVIOR_MIN_VALID_LANDMARKS,
    BEHAVIOR_MOVEMENT_BASELINE_FLOOR,
    BEHAVIOR_MOVEMENT_JITTER_FLOOR,
    BEHAVIOR_MOVEMENT_MAX_GAP_SECONDS,
    BEHAVIOR_MOVEMENT_WINDOW_SECONDS,
    BEHAVIOR_REGION_MIN_LANDMARKS,
    BEHAVIOR_REGION_MIN_OBSERVATIONS,
    BEHAVIOR_REGION_MIN_SPAN_SECONDS,
    BEHAVIOR_REPETITION_MIN_AMPLITUDE,
    BEHAVIOR_SUDDEN_MOVEMENT_THRESHOLD,
    BEHAVIOR_WHOLE_BODY_MIN_LANDMARKS,
    POSE_LANDMARKER_MODEL_PATH,
)

logger = logging.getLogger(__name__)

LANDMARK_INDICES = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

REGION_LANDMARKS = {
    "head": ("nose", "left_ear", "right_ear", "neck"),
    "left_arm": (
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "left_pinky",
        "left_index",
        "left_thumb",
    ),
    "right_arm": (
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "right_pinky",
        "right_index",
        "right_thumb",
    ),
    "torso": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
    "left_leg": (
        "left_hip",
        "left_knee",
        "left_ankle",
        "left_heel",
        "left_foot_index",
    ),
    "right_leg": (
        "right_hip",
        "right_knee",
        "right_ankle",
        "right_heel",
        "right_foot_index",
    ),
    "whole_body": tuple(LANDMARK_INDICES),
}

REGION_FEATURES = (
    "amplitude",
    "movement_energy",
    "peak_speed",
    "peak_acceleration",
    "repetition_rate_hz",
    "elevated_movement_proportion",
)

FEATURE_NAMES = (
    "movement_energy",
    "sudden_movement",
    "movement_change_from_baseline",
    "shoulder_alignment",
    "shoulder_asymmetry",
    "left_upper_arm_angle_deg",
    "right_upper_arm_angle_deg",
    "left_forearm_angle_deg",
    "right_forearm_angle_deg",
    "head_offset_x",
    "head_offset_y",
    "head_orientation_deg",
    "repetition_change",
    "wrist_average_speed",
    "wrist_peak_speed",
    "wrist_peak_acceleration",
    "wrist_movement_amplitude",
    "wrist_repetition_rate_hz",
    "wrist_elevated_movement_proportion",
    "left_wrist_visibility",
    "right_wrist_visibility",
)


@dataclass(frozen=True)
class _PoseObservation:
    timestamp: float
    wrists: Mapping[str, tuple[float, float] | None]
    visibility: Mapping[str, float | None]
    regions: Mapping[str, tuple[float, float] | None]


@dataclass(frozen=True)
class BehaviorResult(BranchOutput):
    """Observable behavior-branch values and explicit validity metadata."""

    status: str
    status_message: str
    landmark_validity: Mapping[str, bool]
    feature_values: Mapping[str, float | None]
    feature_validity: Mapping[str, bool]
    feature_vector: tuple[float | None, ...]
    feature_names: tuple[str, ...]
    unavailable_reasons: Mapping[str, str]
    quality_score: float
    baseline_ready: bool
    baseline_sample_count: int
    baseline_energy: float | None
    repetition_status: str
    pose_diagnostic: Mapping[str, Any]
    debug_overlay: np.ndarray | None
    region_metrics: Mapping[str, Mapping[str, float | None]] = field(
        default_factory=dict
    )
    region_feature_validity: Mapping[str, Mapping[str, bool]] = field(
        default_factory=dict
    )
    region_unavailable_reasons: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )


class UpperBodyBehaviorBranch(BranchProcessor):
    """Extract pose geometry and rolling wrist movement evidence from frames.

    Only bounded, normalized wrist observations and scalar movement measurements
    are retained in memory; image frames are never retained.
    """

    def __init__(
        self,
        model_path: Path = POSE_LANDMARKER_MODEL_PATH,
        landmarker: Any | None = None,
        visibility_threshold: float = BEHAVIOR_LANDMARK_VISIBILITY_THRESHOLD,
        min_valid_landmarks: int = BEHAVIOR_MIN_VALID_LANDMARKS,
        baseline_window: int = BEHAVIOR_BASELINE_WINDOW,
        baseline_min_samples: int = BEHAVIOR_BASELINE_MIN_SAMPLES,
        sudden_movement_threshold: float = BEHAVIOR_SUDDEN_MOVEMENT_THRESHOLD,
        movement_baseline_floor: float = BEHAVIOR_MOVEMENT_BASELINE_FLOOR,
        movement_window_seconds: float = BEHAVIOR_MOVEMENT_WINDOW_SECONDS,
        movement_max_gap_seconds: float = BEHAVIOR_MOVEMENT_MAX_GAP_SECONDS,
        movement_jitter_floor: float = BEHAVIOR_MOVEMENT_JITTER_FLOOR,
        elevated_speed_threshold: float = BEHAVIOR_ELEVATED_SPEED_THRESHOLD,
        repetition_min_amplitude: float = BEHAVIOR_REPETITION_MIN_AMPLITUDE,
        region_min_observations: int = BEHAVIOR_REGION_MIN_OBSERVATIONS,
        region_min_span_seconds: float = BEHAVIOR_REGION_MIN_SPAN_SECONDS,
        region_min_landmarks: int = BEHAVIOR_REGION_MIN_LANDMARKS,
        whole_body_min_landmarks: int = BEHAVIOR_WHOLE_BODY_MIN_LANDMARKS,
    ) -> None:
        if baseline_window < 1 or not 1 <= baseline_min_samples <= baseline_window:
            raise ValueError("Baseline sample limits must satisfy 1 <= minimum <= window.")
        if min_valid_landmarks < 1:
            raise ValueError("At least one valid landmark must be required.")
        if movement_baseline_floor <= 0:
            raise ValueError("Movement baseline floor must be positive.")
        if movement_window_seconds < 1.0:
            raise ValueError("Movement window must be at least one second.")
        if movement_max_gap_seconds <= 0 or movement_jitter_floor < 0:
            raise ValueError("Movement gap must be positive and jitter floor nonnegative.")
        if elevated_speed_threshold <= 0 or repetition_min_amplitude <= 0:
            raise ValueError("Movement thresholds must be positive.")
        if region_min_observations < 2 or region_min_span_seconds <= 0:
            raise ValueError("Region history requirements must be positive.")
        if region_min_landmarks < 1 or whole_body_min_landmarks < 1:
            raise ValueError("Region landmark requirements must be positive.")

        self.model_path = Path(model_path)
        self.visibility_threshold = visibility_threshold
        self.min_valid_landmarks = min_valid_landmarks
        self.baseline_min_samples = baseline_min_samples
        self.sudden_movement_threshold = sudden_movement_threshold
        self.movement_baseline_floor = movement_baseline_floor
        self.movement_window_seconds = movement_window_seconds
        self.movement_max_gap_seconds = movement_max_gap_seconds
        self.movement_jitter_floor = movement_jitter_floor
        self.elevated_speed_threshold = elevated_speed_threshold
        self.repetition_min_amplitude = repetition_min_amplitude
        self.region_min_observations = region_min_observations
        self.region_min_span_seconds = region_min_span_seconds
        self.region_min_landmarks = region_min_landmarks
        self.whole_body_min_landmarks = whole_body_min_landmarks
        self._movement_history: deque[float] = deque(maxlen=baseline_window)
        self._pose_observations: deque[_PoseObservation] = deque(maxlen=256)
        self._pose_observation_issue: str | None = None
        self._previous_timestamp: float | None = None
        self._closed = False
        self.diagnostics_enabled = False
        self.initialization_error: str | None = None
        self.initialization_exception_type: str | None = None
        self.landmarker = landmarker
        if self.landmarker is None:
            try:
                self.landmarker = self._create_landmarker(self.model_path)
            except Exception as exc:
                self.initialization_exception_type = type(exc).__name__
                self.initialization_error = (
                    "Could not initialize MediaPipe Pose Landmarker "
                    f"({type(exc).__name__}): {exc}"
                )
                logger.exception("Unable to initialize the local Pose Landmarker")

    @property
    def is_ready(self) -> bool:
        return self.landmarker is not None and not self._closed

    def process(self, input_data: object) -> BranchOutput:
        """Implement the common future-branch interface for a single frame."""
        return self.analyze(input_data)

    def analyze(
        self,
        frame: object,
        timestamp_seconds: float | None = None,
    ) -> BehaviorResult:
        """Analyze one OpenCV BGR frame without storing the image."""
        input_received = frame is not None
        input_dimensions = (
            tuple(int(value) for value in getattr(frame, "shape", ()))
            if isinstance(frame, np.ndarray)
            else None
        )
        diagnostic: dict[str, Any] = {
            "label": "TEMPORARY LIVE POSE DIAGNOSTIC",
            "frame_received": input_received,
            "frame_dimensions": input_dimensions,
            "pose_landmarks_returned": False,
            "pose_detected": False,
            "landmark_count": 0,
            "visibility_threshold": self.visibility_threshold,
            "wrists": {},
        }
        if not self.is_ready:
            self._log_pose_diagnostic(diagnostic)
            return self._unknown(
                "landmarker_unavailable",
                self.initialization_error or "Pose tracking has been closed.",
                pose_diagnostic=diagnostic,
            )

        image = self._normalize_frame(frame)
        if image is None:
            self._clear_previous_frame()
            self._log_pose_diagnostic(diagnostic)
            return self._unknown(
                "invalid_frame",
                "Expected a non-empty uint8 BGR, grayscale, or BGRA frame.",
                pose_diagnostic=diagnostic,
            )

        captured_at = time.monotonic() if timestamp_seconds is None else timestamp_seconds
        if not math.isfinite(captured_at):
            self._clear_previous_frame()
            self._log_pose_diagnostic(diagnostic)
            return self._unknown(
                "invalid_timestamp",
                "Frame timestamp must be finite.",
                debug_overlay=(
                    self._draw_pose_diagnostic(image, {}, {})
                    if self.diagnostics_enabled
                    else None
                ),
                pose_diagnostic=diagnostic,
            )
        if self._previous_timestamp is not None and captured_at <= self._previous_timestamp:
            self._clear_previous_frame()
            self._log_pose_diagnostic(diagnostic)
            return self._unknown(
                "invalid_timestamp",
                "Frame timestamps must increase monotonically; movement was not measured.",
                debug_overlay=(
                    self._draw_pose_diagnostic(image, {}, {})
                    if self.diagnostics_enabled
                    else None
                ),
                pose_diagnostic=diagnostic,
            )

        pose_inference_started = time.perf_counter()
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            import mediapipe as mp

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)
            diagnostic["stage_timings_ms"] = {
                "pose_inference": (
                    time.perf_counter() - pose_inference_started
                )
                * 1000.0
            }
        except Exception as exc:
            self._clear_previous_frame()
            logger.exception("Pose Landmarker inference failed")
            diagnostic["stage_timings_ms"] = {
                "pose_inference": (
                    time.perf_counter() - pose_inference_started
                )
                * 1000.0
            }
            diagnostic["pose_detection_error"] = f"{type(exc).__name__}: {exc}"
            self._log_pose_diagnostic(diagnostic)
            return self._unknown(
                "inference_error",
                f"Pose inference failed ({type(exc).__name__}): {exc}",
                debug_overlay=(
                    self._draw_pose_diagnostic(image, {}, {})
                    if self.diagnostics_enabled
                    else None
                ),
                pose_diagnostic=diagnostic,
            )

        poses = getattr(result, "pose_landmarks", None)
        pose_detected = bool(poses and poses[0])
        diagnostic["pose_landmarks_returned"] = poses is not None
        diagnostic["landmark_count"] = (
            len(poses[0]) if pose_detected else 0
        )
        diagnostic["pose_detected"] = pose_detected
        if not pose_detected:
            self._clear_previous_frame()
            for side in ("left", "right"):
                diagnostic["wrists"][side] = {
                    "landmark_index": LANDMARK_INDICES[f"{side}_wrist"],
                    "normalized_xy": None,
                    "visibility": None,
                    "presence": None,
                    "passed_threshold": False,
                    "rejection_reason": "Pose Landmarker did not detect a pose.",
                }
            self._log_pose_diagnostic(diagnostic)
            return self._unknown(
                "person_not_detected",
                "No person was detected; movement and posture are unknown.",
                debug_overlay=(
                    self._draw_pose_diagnostic(image, {}, {})
                    if self.diagnostics_enabled
                    else None
                ),
                pose_diagnostic=diagnostic,
            )

        try:
            pose = poses[0]
            landmarks, landmark_validity, landmark_confidence, wrist_diagnostics = (
                self._read_landmarks(pose)
            )
            diagnostic["wrists"] = wrist_diagnostics
        except (IndexError, TypeError, ValueError, AttributeError) as exc:
            self._clear_previous_frame()
            logger.warning("Pose Landmarker returned unusable landmarks: %s", exc)
            diagnostic["landmark_error"] = f"{type(exc).__name__}: {exc}"
            self._log_pose_diagnostic(diagnostic)
            return self._unknown(
                "invalid_landmarks",
                "Pose tracking returned an invalid landmark set.",
                debug_overlay=(
                    self._draw_pose_diagnostic(image, {}, {})
                    if self.diagnostics_enabled
                    else None
                ),
                pose_diagnostic=diagnostic,
            )

        valid_count = sum(landmark_validity.values())
        quality_score = valid_count / len(LANDMARK_INDICES)
        if valid_count < self.min_valid_landmarks:
            self._clear_previous_frame()
            self._log_pose_diagnostic(diagnostic)
            return self._unknown(
                "insufficient_landmarks",
                "Too few upper-body landmarks are visible to calculate features.",
                landmark_validity=landmark_validity,
                quality_score=quality_score,
                debug_overlay=(
                    self._draw_pose_diagnostic(image, landmarks, wrist_diagnostics)
                    if self.diagnostics_enabled
                    else None
                ),
                pose_diagnostic=diagnostic,
            )

        values: dict[str, float | None] = {name: None for name in FEATURE_NAMES}
        reasons: dict[str, str] = {}
        self._set_posture_features(landmarks, values, reasons)
        self._append_pose_observation(
            landmarks,
            landmark_confidence,
            captured_at,
        )
        region_features_started = time.perf_counter()
        region_metrics, region_validity, region_reasons, region_tracking = (
            self._region_movement_features(landmarks, landmark_confidence)
        )
        diagnostic.setdefault("stage_timings_ms", {})[
            "region_feature_calculation"
        ] = (time.perf_counter() - region_features_started) * 1000.0
        diagnostic["regions"] = region_tracking
        movement_features, movement_reasons = self._movement_features()
        values.update(movement_features)
        reasons.update(movement_reasons)
        movement_energy = values["movement_energy"]

        baseline_ready = len(self._movement_history) >= self.baseline_min_samples
        baseline_energy = (
            float(np.mean(self._movement_history)) if self._movement_history else None
        )
        baseline_sample_count = len(self._movement_history)
        if movement_energy is not None and baseline_ready and baseline_energy is not None:
            values["movement_change_from_baseline"] = (
                (movement_energy - baseline_energy)
                / max(baseline_energy, self.movement_baseline_floor)
            )
            values["sudden_movement"] = float(
                movement_energy - baseline_energy >= self.sudden_movement_threshold
            )
        else:
            baseline_reason = (
                f"Movement baseline has {baseline_sample_count} valid sample(s); "
                f"{self.baseline_min_samples} are required."
            )
            if movement_energy is None:
                baseline_reason += (
                    f" Current movement energy is unavailable: "
                    f"{reasons.get('movement_energy', 'wrist movement could not be measured')}."
                )
            reasons["movement_change_from_baseline"] = baseline_reason
            reasons["sudden_movement"] = (
                f"Sudden movement is unavailable until the movement baseline is "
                f"ready ({baseline_sample_count}/{self.baseline_min_samples} valid "
                f"sample(s))."
                + (
                    f" Current movement energy is unavailable: "
                    f"{reasons.get('movement_energy', 'wrist movement could not be measured')}."
                    if movement_energy is None
                    else ""
                )
            )

        reasons["repetition_change"] = (
            "Deprecated in schema v2; use wrist_repetition_rate_hz."
        )
        feature_validity = {
            name: value is not None and math.isfinite(value)
            for name, value in values.items()
        }
        for name, is_valid in feature_validity.items():
            if not is_valid:
                values[name] = None
            else:
                value = values[name]
                if value is not None and not math.isfinite(value):
                    values[name] = None
                    feature_validity[name] = False
                    reasons[name] = "Feature calculation produced a non-finite value."

        if movement_energy is not None:
            self._movement_history.append(movement_energy)
        self._previous_timestamp = captured_at

        partial = valid_count < len(LANDMARK_INDICES)
        if partial:
            status = "partial"
            message = "Some upper-body landmarks are unavailable; only valid features are shown."
        elif not baseline_ready:
            status = "warming_up"
            message = "Collecting valid movement samples to establish a short personal baseline."
        else:
            status = "ok"
            message = "Pose features are available; they do not determine distress."

        available_features = {
            name: float(value)
            for name, value in values.items()
            if feature_validity[name] and value is not None
        }
        debug_overlay = (
            self._draw_pose_diagnostic(image, landmarks, wrist_diagnostics)
            if self.diagnostics_enabled
            else None
        )
        self._log_pose_diagnostic(diagnostic)
        return BehaviorResult(
            features=available_features,
            provisional_distress=None,
            input_quality=True,
            status=status,
            status_message=message,
            landmark_validity=landmark_validity,
            feature_values=values,
            feature_validity=feature_validity,
            feature_vector=tuple(values[name] for name in FEATURE_NAMES),
            feature_names=FEATURE_NAMES,
            unavailable_reasons=reasons,
            quality_score=quality_score,
            baseline_ready=baseline_ready,
            baseline_sample_count=baseline_sample_count,
            baseline_energy=baseline_energy,
            repetition_status=(
                "measured"
                if feature_validity["wrist_repetition_rate_hz"]
                else "warming_up"
            ),
            pose_diagnostic=diagnostic,
            debug_overlay=debug_overlay,
            region_metrics=region_metrics,
            region_feature_validity=region_validity,
            region_unavailable_reasons=region_reasons,
        )

    def _log_pose_diagnostic(self, diagnostic: Mapping[str, Any]) -> None:
        if self.diagnostics_enabled:
            logger.info("TEMPORARY LIVE POSE DIAGNOSTIC: %s", dict(diagnostic))

    @staticmethod
    def _draw_pose_diagnostic(
        image: np.ndarray,
        landmarks: Mapping[str, tuple[float, float]],
        wrists: Mapping[str, Mapping[str, Any]],
    ) -> np.ndarray:
        overlay = image.copy()
        height, width = overlay.shape[:2]
        pose_points = dict(landmarks)
        left_shoulder = landmarks.get("left_shoulder")
        right_shoulder = landmarks.get("right_shoulder")
        if left_shoulder is not None and right_shoulder is not None:
            pose_points["neck"] = (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2,
            )
        pose_connections = (
            ("nose", "left_ear"),
            ("nose", "right_ear"),
            ("nose", "neck"),
            ("neck", "left_shoulder"),
            ("neck", "right_shoulder"),
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("left_wrist", "left_pinky"),
            ("left_wrist", "left_index"),
            ("left_wrist", "left_thumb"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("right_wrist", "right_pinky"),
            ("right_wrist", "right_index"),
            ("right_wrist", "right_thumb"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("left_ankle", "left_heel"),
            ("left_heel", "left_foot_index"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
            ("right_ankle", "right_heel"),
            ("right_heel", "right_foot_index"),
        )

        def pixel(point: tuple[float, float]) -> tuple[int, int] | None:
            x, y = point
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return None
            return (
                int(round(x * (width - 1))),
                int(round(y * (height - 1))),
            )

        for start_name, end_name in pose_connections:
            start = pose_points.get(start_name)
            end = pose_points.get(end_name)
            if start is None or end is None:
                continue
            start_pixel = pixel(start)
            end_pixel = pixel(end)
            if start_pixel is not None and end_pixel is not None:
                cv2.line(overlay, start_pixel, end_pixel, (255, 180, 0), 2)
        for point in pose_points.values():
            point_pixel = pixel(point)
            if point_pixel is not None:
                cv2.circle(overlay, point_pixel, 4, (255, 180, 0), -1)
        for side, diagnostic in wrists.items():
            point = diagnostic.get("normalized_xy")
            if point is None or not isinstance(point, tuple):
                continue
            point_pixel = pixel(point)
            if point_pixel is None:
                continue
            passed = diagnostic.get("passed_threshold", False)
            color = (0, 200, 0) if passed else (0, 0, 255)
            cv2.circle(overlay, point_pixel, 8, color, 2)
            cv2.putText(
                overlay,
                f"{side} wrist vis={diagnostic.get('visibility')} "
                f"pres={diagnostic.get('presence')} pass={passed}",
                (max(5, point_pixel[0] - 100), max(20, point_pixel[1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            overlay,
            "TEMPORARY LIVE POSE DIAGNOSTIC",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def reset(self) -> None:
        """Clear per-session measurements while keeping the local detector open."""
        self._movement_history.clear()
        self._pose_observations.clear()
        self._clear_previous_frame()

    def close(self) -> None:
        """Release MediaPipe resources and clear all in-memory tracking state."""
        self.reset()
        self._closed = True
        close = getattr(self.landmarker, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Unable to close the Pose Landmarker cleanly")

    @staticmethod
    def _create_landmarker(model_path: Path) -> Any:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Pose model asset not found at {model_path}. "
                "Place pose_landmarker_lite.task in the project's models folder."
            )
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=False,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def _read_landmarks(
        self,
        pose: Any,
    ) -> tuple[
        dict[str, tuple[float, float]],
        dict[str, bool],
        dict[str, float | None],
        dict[str, dict[str, Any]],
    ]:
        coordinates: dict[str, tuple[float, float]] = {}
        validity: dict[str, bool] = {}
        confidence: dict[str, float | None] = {}
        wrist_diagnostics: dict[str, dict[str, Any]] = {}
        for name, index in LANDMARK_INDICES.items():
            landmark = pose[index]
            raw_visibility = getattr(landmark, "visibility", None)
            raw_presence = getattr(landmark, "presence", None)
            visibility = (
                float(raw_visibility) if raw_visibility is not None else None
            )
            presence = float(raw_presence) if raw_presence is not None else None
            scores = [
                score
                for score in (visibility, presence)
                if score is not None
            ]
            confidence[name] = (
                min(scores)
                if scores
                and all(
                    math.isfinite(score) and 0.0 <= score <= 1.0
                    for score in scores
                )
                else None
            )
            x = float(landmark.x)
            y = float(landmark.y)
            coordinates_finite = math.isfinite(x) and math.isfinite(y)
            score_values_valid = bool(scores) and all(
                math.isfinite(score) and 0.0 <= score <= 1.0
                for score in scores
            )
            inside_image = coordinates_finite and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
            passed_threshold = (
                score_values_valid
                and min(scores) >= self.visibility_threshold
            )
            valid = (
                coordinates_finite
                and inside_image
                and passed_threshold
            )
            validity[name] = valid
            if valid:
                coordinates[name] = (x, y)
            if name in {"left_wrist", "right_wrist"}:
                if not coordinates_finite:
                    rejection_reason = "Landmark coordinates are non-finite."
                elif not inside_image:
                    rejection_reason = (
                        "Landmark normalized x/y is outside the camera image."
                    )
                elif not scores:
                    rejection_reason = (
                        "Neither visibility nor presence confidence was returned."
                    )
                elif not score_values_valid:
                    rejection_reason = (
                        "Visibility or presence confidence is non-finite or outside "
                        "the expected [0, 1] range."
                    )
                elif not passed_threshold:
                    rejection_reason = (
                        f"Lowest available confidence {min(scores):.3f} is below "
                        f"the configured threshold {self.visibility_threshold:.2f}."
                    )
                else:
                    rejection_reason = None
                wrist_diagnostics[name.removesuffix("_wrist")] = {
                    "landmark_index": index,
                    "normalized_xy": (x, y) if coordinates_finite else None,
                    "visibility": visibility,
                    "presence": presence,
                    "passed_threshold": valid,
                    "rejection_reason": rejection_reason,
                }
        return coordinates, validity, confidence, wrist_diagnostics

    def _append_pose_observation(
        self,
        landmarks: Mapping[str, tuple[float, float]],
        confidence: Mapping[str, float | None],
        captured_at: float,
    ) -> None:
        self._pose_observation_issue = None
        left_shoulder = landmarks.get("left_shoulder")
        right_shoulder = landmarks.get("right_shoulder")
        left_hip = landmarks.get("left_hip")
        right_hip = landmarks.get("right_hip")
        shoulder_center = (
            (
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2,
            )
            if left_shoulder is not None and right_shoulder is not None
            else None
        )
        hip_center = (
            (
                (left_hip[0] + right_hip[0]) / 2,
                (left_hip[1] + right_hip[1]) / 2,
            )
            if left_hip is not None and right_hip is not None
            else None
        )
        shoulder_width = (
            math.dist(left_shoulder, right_shoulder)
            if left_shoulder is not None and right_shoulder is not None
            else None
        )
        hip_width = (
            math.dist(left_hip, right_hip)
            if left_hip is not None and right_hip is not None
            else None
        )
        torso_length = (
            math.dist(shoulder_center, hip_center)
            if shoulder_center is not None and hip_center is not None
            else None
        )
        scale = next(
            (
                candidate
                for candidate in (shoulder_width, hip_width, torso_length)
                if candidate is not None and candidate > 1e-6
            ),
            None,
        )
        if scale is None:
            self._pose_observation_issue = (
                "Movement normalization is unavailable: visible shoulders or hips "
                "are required to establish body scale."
            )
        if self._pose_observations:
            gap = captured_at - self._pose_observations[-1].timestamp
            if gap <= 0 or gap > self.movement_max_gap_seconds:
                self._pose_observations.clear()
                self._pose_observation_issue = (
                    f"Pose history reset because the frame gap was {gap:.2f}s; "
                    f"the maximum movement gap is "
                    f"{self.movement_max_gap_seconds:.2f}s."
                )

        anchor = shoulder_center or hip_center
        wrists = {
            side: (
                (
                    (landmarks[f"{side}_wrist"][0] - anchor[0]) / scale,
                    (landmarks[f"{side}_wrist"][1] - anchor[1]) / scale,
                )
                if (
                    scale is not None
                    and anchor is not None
                    and shoulder_center is not None
                    and f"{side}_wrist" in landmarks
                )
                else None
            )
            for side in ("left", "right")
        }
        regions: dict[str, tuple[float, float] | None] = {}
        for region, names in REGION_LANDMARKS.items():
            points = []
            for name in names:
                if name == "neck":
                    if shoulder_center is not None:
                        points.append(shoulder_center)
                elif name in landmarks:
                    points.append(landmarks[name])
            minimum = (
                self.whole_body_min_landmarks
                if region == "whole_body"
                else max(self.region_min_landmarks, 2)
                if region.endswith("_arm") or region.endswith("_leg") or region == "torso"
                else self.region_min_landmarks
            )
            if scale is None or len(points) < minimum:
                regions[region] = None
                continue
            center_x = sum(point[0] for point in points) / len(points)
            center_y = sum(point[1] for point in points) / len(points)
            if region == "whole_body":
                regions[region] = (center_x / scale, center_y / scale)
            elif anchor is not None:
                regions[region] = (
                    (center_x - anchor[0]) / scale,
                    (center_y - anchor[1]) / scale,
                )
            else:
                regions[region] = None
        self._pose_observations.append(
            _PoseObservation(
                timestamp=captured_at,
                wrists=wrists,
                visibility={
                    f"{side}_wrist": confidence.get(f"{side}_wrist")
                    for side in ("left", "right")
                },
                regions=regions,
            )
        )
        cutoff = captured_at - self.movement_window_seconds
        while (
            self._pose_observations
            and self._pose_observations[0].timestamp < cutoff
        ):
            self._pose_observations.popleft()

    @staticmethod
    def _estimate_repetition_rate_hz(
        timestamps: list[float],
        signal: np.ndarray,
        *,
        jitter_floor: float,
        min_amplitude: float,
    ) -> float:
        if len(signal) < 2 or len(timestamps) != len(signal):
            return 0.0
        amplitude = float(np.percentile(signal, 90) - np.percentile(signal, 10))
        if amplitude < min_amplitude:
            return 0.0
        centered = np.asarray(signal, dtype=float) - float(np.median(signal))
        hysteresis = max(float(jitter_floor), amplitude * 0.1)
        previous_sign = 0
        crossings = 0
        for sample in centered:
            if sample >= hysteresis:
                current_sign = 1
            elif sample <= -hysteresis:
                current_sign = -1
            else:
                current_sign = 0
            if current_sign != 0 and previous_sign != 0 and current_sign != previous_sign:
                crossings += 1
            if current_sign != 0:
                previous_sign = current_sign
        span = float(timestamps[-1] - timestamps[0])
        if crossings < 2 or span <= 0:
            return 0.0
        return max(0.0, crossings / (2.0 * span))

    def _region_movement_features(
        self,
        landmarks: Mapping[str, tuple[float, float]],
        confidence: Mapping[str, float | None],
    ) -> tuple[
        dict[str, dict[str, float | None]],
        dict[str, dict[str, bool]],
        dict[str, dict[str, str]],
        dict[str, dict[str, Any]],
    ]:
        metrics_by_region: dict[str, dict[str, float | None]] = {}
        validity_by_region: dict[str, dict[str, bool]] = {}
        reasons_by_region: dict[str, dict[str, str]] = {}
        tracking: dict[str, dict[str, Any]] = {}
        for region, names in REGION_LANDMARKS.items():
            minimum = (
                self.whole_body_min_landmarks
                if region == "whole_body"
                else max(self.region_min_landmarks, 2)
                if region.endswith("_arm") or region.endswith("_leg") or region == "torso"
                else self.region_min_landmarks
            )
            visible_names = [name for name in names if name in landmarks]
            if (
                region == "head"
                and "left_shoulder" in landmarks
                and "right_shoulder" in landmarks
            ):
                visible_names.append("derived_neck")
            positions = [
                (item.timestamp, item.regions[region])
                for item in self._pose_observations
                if item.regions[region] is not None
            ]
            positions = [
                (stamp, point) for stamp, point in positions if point is not None
            ]
            count = len(positions)
            span = positions[-1][0] - positions[0][0] if count >= 2 else 0.0
            current_region_available = bool(
                self._pose_observations
                and self._pose_observations[-1].regions[region] is not None
            )
            frame_tracking_valid = (
                len(visible_names) >= minimum and current_region_available
            )
            if (
                region == "torso"
                and not (
                    any(name.endswith("shoulder") for name in visible_names)
                    and any(name.endswith("hip") for name in visible_names)
                )
            ):
                frame_tracking_valid = False
            if not frame_tracking_valid:
                tracking_reason = (
                    "Torso tracking requires at least one visible shoulder and one "
                    "visible hip landmark."
                    if region == "torso"
                    and not (
                        any(name.endswith("shoulder") for name in visible_names)
                        and any(name.endswith("hip") for name in visible_names)
                    )
                    else
                    f"{region} tracking has {len(visible_names)} valid landmark(s); "
                    f"{minimum} required."
                )
            elif not positions:
                tracking_reason = self._pose_observation_issue or (
                    "No timestamped normalized region observations are available; "
                    "body scale could not be established."
                )
            else:
                tracking_reason = ""

            values: dict[str, float | None] = {
                name: None for name in REGION_FEATURES
            }
            reasons: dict[str, str] = {}
            validity: dict[str, bool] = {name: False for name in REGION_FEATURES}
            speeds: list[tuple[float, float, float]] = []
            if count >= 2:
                for (previous_time, previous), (current_time, current) in zip(
                    positions,
                    positions[1:],
                ):
                    elapsed = current_time - previous_time
                    if elapsed <= 0 or elapsed > self.movement_max_gap_seconds:
                        continue
                    speed = max(
                        0.0,
                        math.dist(previous, current) - self.movement_jitter_floor,
                    ) / elapsed
                    speeds.append(
                        ((previous_time + current_time) / 2, speed, elapsed)
                    )
            if count >= 2 and speeds:
                x_values = [point[0] for _, point in positions]
                y_values = [point[1] for _, point in positions]
                x_amplitude = float(
                    np.percentile(x_values, 90) - np.percentile(x_values, 10)
                )
                y_amplitude = float(
                    np.percentile(y_values, 90) - np.percentile(y_values, 10)
                )
                values["amplitude"] = math.hypot(x_amplitude, y_amplitude)
                duration = sum(elapsed for _, _, elapsed in speeds)
                if duration > 0:
                    values["movement_energy"] = math.sqrt(
                        sum(speed * speed * elapsed for _, speed, elapsed in speeds)
                        / duration
                    )
                    values["peak_speed"] = float(
                        np.percentile([speed for _, speed, _ in speeds], 95)
                    )
                    values["elevated_movement_proportion"] = sum(
                        elapsed
                        for _, speed, elapsed in speeds
                        if speed >= self.elevated_speed_threshold
                    ) / duration
                    for feature in (
                        "amplitude",
                        "movement_energy",
                        "peak_speed",
                        "elevated_movement_proportion",
                    ):
                        validity[feature] = values[feature] is not None
            else:
                speed_reason = tracking_reason or (
                    f"{region} has {count} valid observation(s); at least two "
                    "with an acceptable timestamp gap are required to measure speed."
                )
                for feature in (
                    "amplitude",
                    "movement_energy",
                    "peak_speed",
                    "elevated_movement_proportion",
                ):
                    reasons[feature] = speed_reason

            speed_samples = [(sample_time, speed) for sample_time, speed, _ in speeds]
            accelerations = [
                abs(current_speed - previous_speed)
                / (current_time - previous_time)
                for (previous_time, previous_speed), (current_time, current_speed)
                in zip(speed_samples, speed_samples[1:])
                if 0 < current_time - previous_time <= self.movement_max_gap_seconds
            ]
            if accelerations:
                values["peak_acceleration"] = float(
                    np.percentile(accelerations, 95)
                )
                validity["peak_acceleration"] = True
            else:
                reasons["peak_acceleration"] = (
                    f"{region} acceleration requires at least two consecutive "
                    "valid speed samples."
                )

            if count < self.region_min_observations or span < self.region_min_span_seconds:
                reasons["repetition_rate_hz"] = (
                    f"{region} repetition is warming up: {count}/"
                    f"{self.region_min_observations} observations spanning "
                    f"{span:.2f}/{self.region_min_span_seconds:.2f}s."
                )
            else:
                x_values = np.asarray(
                    [point[0] for _, point in positions],
                    dtype=float,
                )
                y_values = np.asarray(
                    [point[1] for _, point in positions],
                    dtype=float,
                )
                signal = (
                    x_values
                    if np.ptp(x_values) >= np.ptp(y_values)
                    else y_values
                )
                values["repetition_rate_hz"] = self._estimate_repetition_rate_hz(
                    [timestamp for timestamp, _ in positions],
                    signal,
                    jitter_floor=self.movement_jitter_floor,
                    min_amplitude=self.repetition_min_amplitude,
                )
                validity["repetition_rate_hz"] = True

            if count < 2:
                region_reason = tracking_reason or self._pose_observation_issue or (
                    f"{region} has insufficient timestamped history for movement features."
                )
            elif span < self.region_min_span_seconds:
                region_reason = (
                    f"{region} history spans {span:.2f}s; "
                    f"{self.region_min_span_seconds:.2f}s is required for repetition."
                )
            else:
                region_reason = tracking_reason
            if not validity["repetition_rate_hz"]:
                reasons.setdefault("repetition_rate_hz", region_reason)
            if not frame_tracking_valid:
                unavailable_reason = tracking_reason or (
                    f"No current {region} observation was available."
                )
                for feature in REGION_FEATURES:
                    values[feature] = None
                    validity[feature] = False
                    reasons[feature] = unavailable_reason
            metrics_by_region[region] = values
            validity_by_region[region] = validity
            reasons_by_region[region] = reasons
            tracking[region] = {
                "available": frame_tracking_valid and bool(positions),
                "valid_landmark_count": len(visible_names),
                "required_landmark_count": minimum,
                "confidence": {
                    name: confidence.get(name)
                    for name in names
                    if confidence.get(name) is not None
                },
                "observation_count": count,
                "span_seconds": span,
                "reason": (
                    tracking_reason
                    or self._pose_observation_issue
                    or (
                        "Region tracking available; temporal repetition may still be warming up."
                        if count < self.region_min_observations
                        or span < self.region_min_span_seconds
                        else ""
                    )
                ),
            }
            if (
                region == "head"
                and confidence.get("left_shoulder") is not None
                and confidence.get("right_shoulder") is not None
            ):
                tracking[region]["confidence"]["derived_neck"] = min(
                    float(confidence["left_shoulder"]),
                    float(confidence["right_shoulder"]),
                )
        return metrics_by_region, validity_by_region, reasons_by_region, tracking

    def _movement_features(self) -> tuple[dict[str, float | None], dict[str, str]]:
        movement_feature_names = (
            "movement_energy",
            "sudden_movement",
            "movement_change_from_baseline",
            "repetition_change",
            "wrist_average_speed",
            "wrist_peak_speed",
            "wrist_peak_acceleration",
            "wrist_movement_amplitude",
            "wrist_repetition_rate_hz",
            "wrist_elevated_movement_proportion",
            "left_wrist_visibility",
            "right_wrist_visibility",
        )
        values: dict[str, float | None] = {
            name: None for name in movement_feature_names
        }
        reasons: dict[str, str] = {}
        for side in ("left", "right"):
            visibility_name = f"{side}_wrist_visibility"
            visibility_values = [
                observation.visibility[f"{side}_wrist"]
                for observation in self._pose_observations
                if observation.visibility[f"{side}_wrist"] is not None
            ]
            if visibility_values:
                values[visibility_name] = float(np.mean(visibility_values))
            else:
                reasons[visibility_name] = "Wrist visibility confidence is unavailable."

        intervals: list[tuple[float, float]] = []
        per_wrist_speeds: dict[str, list[tuple[float, float, float]]] = {
            "left": [],
            "right": [],
        }
        per_wrist_positions: dict[str, list[tuple[float, float, float]]] = {
            "left": [],
            "right": [],
        }
        for observation in self._pose_observations:
            for side in ("left", "right"):
                position = observation.wrists[side]
                if position is not None:
                    per_wrist_positions[side].append(
                        (observation.timestamp, position[0], position[1])
                    )
        usable_counts = {
            side: len(points) for side, points in per_wrist_positions.items()
        }
        usable_spans = {
            side: (
                points[-1][0] - points[0][0]
                if len(points) >= 2
                else 0.0
            )
            for side, points in per_wrist_positions.items()
        }
        low_confidence_count = sum(
            1
            for observation in self._pose_observations
            if any(
                observation.visibility[f"{side}_wrist"] is not None
                and observation.visibility[f"{side}_wrist"]
                < self.visibility_threshold
                for side in ("left", "right")
            )
        )
        maximum_usable_count = max(usable_counts.values(), default=0)
        maximum_usable_span = max(usable_spans.values(), default=0.0)

        if maximum_usable_count == 0:
            if low_confidence_count:
                wrist_observation_reason = (
                    f"No wrist observations met the visibility threshold "
                    f"{self.visibility_threshold:.2f} in "
                    f"{low_confidence_count} low-confidence frame(s)."
                )
            elif self._pose_observation_issue:
                wrist_observation_reason = self._pose_observation_issue
            else:
                wrist_observation_reason = (
                    "No valid wrist landmarks were available in the rolling window."
                )
        elif maximum_usable_count < 2:
            wrist_observation_reason = (
                f"Only {maximum_usable_count} valid wrist observation is available; "
                "at least two are required to measure movement."
            )
            if self._pose_observation_issue:
                wrist_observation_reason += f" {self._pose_observation_issue}"
        else:
            wrist_observation_reason = (
                f"Valid wrist observations span {maximum_usable_span:.2f}s, but no "
                "consecutive pair passed the movement-gap check."
            )

        for side, points in per_wrist_positions.items():
            for previous, current in zip(points, points[1:]):
                elapsed = current[0] - previous[0]
                if elapsed <= 0 or elapsed > self.movement_max_gap_seconds:
                    continue
                displacement = math.dist(previous[1:], current[1:])
                filtered = max(0.0, displacement - self.movement_jitter_floor)
                speed = filtered / elapsed
                per_wrist_speeds[side].append(
                    ((previous[0] + current[0]) / 2, speed, elapsed)
                )
                intervals.append((speed, elapsed))

        all_speeds = [speed for samples in per_wrist_speeds.values() for _, speed, _ in samples]
        total_duration = sum(duration for _, duration in intervals)
        if all_speeds and total_duration > 0:
            values["wrist_average_speed"] = sum(
                speed * duration
                for samples in per_wrist_speeds.values()
                for _, speed, duration in samples
            ) / total_duration
            values["movement_energy"] = math.sqrt(
                sum(
                    speed * speed * duration
                    for samples in per_wrist_speeds.values()
                    for _, speed, duration in samples
                )
                / total_duration
            )
            values["wrist_peak_speed"] = float(np.percentile(all_speeds, 95))
            elevated_duration = sum(
                duration
                for speed, duration in intervals
                if speed >= self.elevated_speed_threshold
            )
            values["wrist_elevated_movement_proportion"] = (
                elevated_duration / total_duration
            )
        else:
            reasons["movement_energy"] = wrist_observation_reason
            for name in (
                "wrist_average_speed",
                "wrist_peak_speed",
                "wrist_elevated_movement_proportion",
            ):
                reasons[name] = wrist_observation_reason

        if all_speeds:
            accelerations: list[float] = []
            for samples in per_wrist_speeds.values():
                for previous, current in zip(samples, samples[1:]):
                    elapsed = current[0] - previous[0]
                    if 0 < elapsed <= self.movement_max_gap_seconds:
                        accelerations.append(abs(current[1] - previous[1]) / elapsed)
            if accelerations:
                values["wrist_peak_acceleration"] = float(
                    np.percentile(accelerations, 95)
                )
            else:
                reasons["wrist_peak_acceleration"] = (
                    "At least two consecutive wrist-speed samples are required "
                    "to measure acceleration."
                )
        else:
            reasons["wrist_peak_acceleration"] = wrist_observation_reason

        amplitudes: list[float] = []
        repetition_rates: list[float] = []
        for points in per_wrist_positions.values():
            if len(points) >= 2:
                x_span = float(np.percentile([point[1] for point in points], 90) - np.percentile([point[1] for point in points], 10))
                y_span = float(np.percentile([point[2] for point in points], 90) - np.percentile([point[2] for point in points], 10))
                amplitudes.append(math.hypot(x_span, y_span))
            span = points[-1][0] - points[0][0] if len(points) >= 2 else 0.0
            if len(points) < 6 or span < 1.5:
                continue
            x_values = np.asarray([point[1] for point in points], dtype=float)
            y_values = np.asarray([point[2] for point in points], dtype=float)
            signal = x_values if np.ptp(x_values) >= np.ptp(y_values) else y_values
            repetition_rates.append(
                self._estimate_repetition_rate_hz(
                    [point[0] for point in points],
                    signal,
                    jitter_floor=self.movement_jitter_floor,
                    min_amplitude=self.repetition_min_amplitude,
                )
            )

        if amplitudes:
            values["wrist_movement_amplitude"] = max(amplitudes)
        else:
            reasons["wrist_movement_amplitude"] = (
                f"{wrist_observation_reason} Wrist amplitude requires at least two "
                "valid positions."
            )
        if repetition_rates:
            values["wrist_repetition_rate_hz"] = max(repetition_rates)
        else:
            reasons["wrist_repetition_rate_hz"] = (
                f"Repetition estimate has {maximum_usable_count} valid wrist "
                f"observation(s) spanning {maximum_usable_span:.2f}s; it requires "
                "at least six valid observations spanning 1.5s."
                if maximum_usable_count
                else f"{wrist_observation_reason} Repetition requires at least six "
                "valid observations spanning 1.5s."
            )

        return values, reasons

    @staticmethod
    def _set_posture_features(
        landmarks: Mapping[str, tuple[float, float]],
        values: dict[str, float | None],
        reasons: dict[str, str],
    ) -> None:
        left_shoulder = landmarks.get("left_shoulder")
        right_shoulder = landmarks.get("right_shoulder")
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_width = math.dist(left_shoulder, right_shoulder)
            if shoulder_width > 1e-6:
                alignment = (right_shoulder[1] - left_shoulder[1]) / shoulder_width
                values["shoulder_alignment"] = alignment
                values["shoulder_asymmetry"] = abs(alignment)
            else:
                reasons["shoulder_alignment"] = "Shoulder landmarks are too close together."
                reasons["shoulder_asymmetry"] = "Shoulder landmarks are too close together."
        else:
            reasons["shoulder_alignment"] = "Requires both visible shoulders."
            reasons["shoulder_asymmetry"] = "Requires both visible shoulders."

        for side in ("left", "right"):
            shoulder = landmarks.get(f"{side}_shoulder")
            elbow = landmarks.get(f"{side}_elbow")
            wrist = landmarks.get(f"{side}_wrist")
            upper_name = f"{side}_upper_arm_angle_deg"
            forearm_name = f"{side}_forearm_angle_deg"
            if shoulder is not None and elbow is not None:
                values[upper_name] = math.degrees(
                    math.atan2(elbow[1] - shoulder[1], elbow[0] - shoulder[0])
                )
            else:
                reasons[upper_name] = f"Requires visible {side} shoulder and elbow."
            if elbow is not None and wrist is not None:
                values[forearm_name] = math.degrees(
                    math.atan2(wrist[1] - elbow[1], wrist[0] - elbow[0])
                )
            else:
                reasons[forearm_name] = f"Requires visible {side} elbow and wrist."

        nose = landmarks.get("nose")
        if nose is not None and left_shoulder is not None and right_shoulder is not None:
            shoulder_width = math.dist(left_shoulder, right_shoulder)
            if shoulder_width > 1e-6:
                shoulder_midpoint = (
                    (left_shoulder[0] + right_shoulder[0]) / 2,
                    (left_shoulder[1] + right_shoulder[1]) / 2,
                )
                offset_x = (nose[0] - shoulder_midpoint[0]) / shoulder_width
                offset_y = (nose[1] - shoulder_midpoint[1]) / shoulder_width
                values["head_offset_x"] = offset_x
                values["head_offset_y"] = offset_y
                values["head_orientation_deg"] = math.degrees(
                    math.atan2(offset_y, offset_x)
                )
            else:
                for name in ("head_offset_x", "head_offset_y", "head_orientation_deg"):
                    reasons[name] = "Shoulder landmarks are too close together."
        else:
            for name in ("head_offset_x", "head_offset_y", "head_orientation_deg"):
                reasons[name] = "Requires a visible nose and both shoulders."

    @staticmethod
    def _normalize_frame(frame: object) -> np.ndarray | None:
        if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.dtype != np.uint8:
            return None
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
            return None
        if frame.shape[2] == 3:
            return frame
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return None

    def _clear_previous_frame(self) -> None:
        self._pose_observations.clear()
        self._pose_observation_issue = None
        self._previous_timestamp = None

    def _unknown(
        self,
        reason: str,
        message: str,
        landmark_validity: Mapping[str, bool] | None = None,
        quality_score: float = 0.0,
        pose_diagnostic: Mapping[str, Any] | None = None,
        debug_overlay: np.ndarray | None = None,
    ) -> BehaviorResult:
        values: dict[str, float | None] = {name: None for name in FEATURE_NAMES}
        validities = {name: False for name in FEATURE_NAMES}
        unavailable_reasons = {name: reason for name in FEATURE_NAMES}
        features: dict[str, float] = {}
        return BehaviorResult(
            features=features,
            provisional_distress=None,
            input_quality=False,
            status="unknown",
            status_message=message,
            landmark_validity=landmark_validity
            or {name: False for name in LANDMARK_INDICES},
            feature_values=values,
            feature_validity=validities,
            feature_vector=tuple(None for _ in FEATURE_NAMES),
            feature_names=FEATURE_NAMES,
            unavailable_reasons=unavailable_reasons,
            quality_score=quality_score,
            baseline_ready=len(self._movement_history) >= self.baseline_min_samples,
            baseline_sample_count=len(self._movement_history),
            baseline_energy=(
                float(np.mean(self._movement_history)) if self._movement_history else None
            ),
            repetition_status="unavailable",
            pose_diagnostic=pose_diagnostic or {
                "label": "TEMPORARY LIVE POSE DIAGNOSTIC",
                "frame_received": False,
                "frame_dimensions": None,
                "pose_landmarks_returned": False,
                "pose_detected": False,
                "landmark_count": 0,
                "visibility_threshold": self.visibility_threshold,
                "wrists": {},
            },
            debug_overlay=debug_overlay,
            region_metrics={
                region: {name: None for name in REGION_FEATURES}
                for region in REGION_LANDMARKS
            },
            region_feature_validity={
                region: {name: False for name in REGION_FEATURES}
                for region in REGION_LANDMARKS
            },
            region_unavailable_reasons={
                region: {name: message for name in REGION_FEATURES}
                for region in REGION_LANDMARKS
            },
        )
