import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from branches.behavior_branch import (
    FEATURE_NAMES,
    LANDMARK_INDICES,
    UpperBodyBehaviorBranch,
)
from fusion import build_fusion_evidence, evidence_to_feature_record
from rule_based_detection import RuleBasedCueDetector, RuleDetectionConfig


def pose_landmarks(
    shift_x: float = 0.0,
    visibility_overrides: dict[str, float] | None = None,
    wrist_positions: dict[str, tuple[float, float]] | None = None,
) -> list[SimpleNamespace]:
    points = {
        "nose": (0.50, 0.20),
        "left_ear": (0.46, 0.21),
        "right_ear": (0.54, 0.21),
        "left_shoulder": (0.40, 0.40),
        "right_shoulder": (0.60, 0.40),
        "left_elbow": (0.36, 0.53),
        "right_elbow": (0.64, 0.53),
        "left_wrist": (0.34, 0.68),
        "right_wrist": (0.66, 0.68),
        "left_pinky": (0.32, 0.67),
        "right_pinky": (0.68, 0.67),
        "left_index": (0.33, 0.66),
        "right_index": (0.67, 0.66),
        "left_thumb": (0.35, 0.66),
        "right_thumb": (0.65, 0.66),
        "left_hip": (0.44, 0.63),
        "right_hip": (0.56, 0.63),
        "left_knee": (0.43, 0.82),
        "right_knee": (0.57, 0.82),
        "left_ankle": (0.42, 0.97),
        "right_ankle": (0.58, 0.97),
        "left_heel": (0.41, 0.98),
        "right_heel": (0.59, 0.98),
        "left_foot_index": (0.40, 0.98),
        "right_foot_index": (0.60, 0.98),
    }
    points.update(wrist_positions or {})
    overrides = visibility_overrides or {}
    landmarks = [
        SimpleNamespace(x=0.5, y=0.5, visibility=1.0, presence=1.0)
        for _ in range(33)
    ]
    for name, index in LANDMARK_INDICES.items():
        x, y = points[name]
        confidence = overrides.get(name, 1.0)
        landmarks[index] = SimpleNamespace(
            x=x + shift_x,
            y=y,
            visibility=confidence,
            presence=confidence,
        )
    return landmarks


def region_trajectory(
    region: str,
    count: int = 32,
    *,
    interval: float = 0.15,
    repeat_hz: float = 0.8,
) -> list[list[SimpleNamespace]]:
    poses = []
    for index in range(count):
        timestamp = index * interval
        offset = 0.045 * math.sin(2 * math.pi * repeat_hz * timestamp)
        confidence = {}
        if region == "torso":
            confidence = {
                "left_wrist": 0.1,
                "right_wrist": 0.1,
            }
        elif region == "head":
            confidence = {
                name: 0.1
                for name in (
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_pinky",
                    "right_pinky",
                    "left_index",
                    "right_index",
                    "left_thumb",
                    "right_thumb",
                )
            }
        landmarks = pose_landmarks(visibility_overrides=confidence)
        if region == "torso":
            for name in ("left_hip", "right_hip"):
                point = landmarks[LANDMARK_INDICES[name]]
                point.y += offset
        elif region == "head":
            for name in ("nose", "left_ear", "right_ear"):
                point = landmarks[LANDMARK_INDICES[name]]
                point.x += offset
        elif region == "left_arm":
            for name, factor in (
                ("left_elbow", 0.6),
                ("left_wrist", 1.0),
                ("left_pinky", 1.1),
                ("left_index", 1.0),
                ("left_thumb", 0.9),
            ):
                point = landmarks[LANDMARK_INDICES[name]]
                point.x += offset * factor
        poses.append(landmarks)
    return poses


class QueueLandmarker:
    def __init__(self, poses: list[list[SimpleNamespace] | None]) -> None:
        self.poses = iter(poses)
        self.closed = False

    def detect(self, image: object) -> SimpleNamespace:
        pose = next(self.poses)
        return SimpleNamespace(pose_landmarks=[] if pose is None else [pose])

    def close(self) -> None:
        self.closed = True


class FailingLandmarker:
    def detect(self, image: object) -> SimpleNamespace:
        raise RuntimeError("mock pose inference failure")


class BehaviorBranchTests(unittest.TestCase):
    def make_branch(
        self,
        poses: list[list[SimpleNamespace] | None] | None = None,
        **kwargs: object,
    ) -> UpperBodyBehaviorBranch:
        return UpperBodyBehaviorBranch(
            landmarker=QueueLandmarker(poses or []),
            **kwargs,
        )

    @staticmethod
    def frame() -> np.ndarray:
        return np.zeros((120, 160, 3), dtype=np.uint8)

    def analyze_trajectory(
        self,
        offsets: list[float],
        *,
        interval: float = 0.15,
        visibility: float = 1.0,
    ):
        poses = [
            pose_landmarks(
                visibility_overrides={
                    "left_wrist": visibility,
                    "right_wrist": visibility,
                },
                wrist_positions={
                    "left_wrist": (0.34, 0.68 + offset * 0.2),
                    "right_wrist": (0.66, 0.68 - offset * 0.2),
                },
            )
            for offset in offsets
        ]
        branch = self.make_branch(
            poses,
            movement_window_seconds=4.0,
            baseline_min_samples=3,
            baseline_window=16,
        )
        results = [
            branch.analyze(self.frame(), timestamp_seconds=1.0 + index * interval)
            for index in range(len(poses))
        ]
        return branch, results

    def analyze_region_trajectory(self, region: str, **kwargs: object):
        poses = region_trajectory(region, **kwargs)
        branch = self.make_branch(
            poses,
            region_min_observations=6,
            region_min_span_seconds=1.5,
        )
        results = [
            branch.analyze(self.frame(), timestamp_seconds=10 + index * 0.15)
            for index in range(len(poses))
        ]
        return branch, results

    def test_repetitive_torso_is_measured_when_wrists_are_unavailable(self) -> None:
        _, results = self.analyze_region_trajectory("torso")
        result = results[-1]

        self.assertFalse(result.feature_validity["wrist_repetition_rate_hz"])
        self.assertTrue(result.region_feature_validity["torso"]["repetition_rate_hz"])
        self.assertGreaterEqual(
            result.region_metrics["torso"]["repetition_rate_hz"] or 0,
            0.5,
        )
        self.assertTrue(result.region_feature_validity["torso"]["movement_energy"])
        self.assertTrue(result.region_feature_validity["torso"]["peak_speed"])
        self.assertTrue(result.region_feature_validity["torso"]["peak_acceleration"])
        self.assertTrue(result.pose_diagnostic["regions"]["torso"]["available"])

    def test_repetitive_head_is_measured_when_arms_are_unavailable(self) -> None:
        _, results = self.analyze_region_trajectory("head")
        result = results[-1]

        self.assertFalse(result.pose_diagnostic["regions"]["left_arm"]["available"])
        self.assertFalse(result.pose_diagnostic["regions"]["right_arm"]["available"])
        self.assertTrue(result.region_feature_validity["head"]["repetition_rate_hz"])
        self.assertGreaterEqual(
            result.region_metrics["head"]["repetition_rate_hz"] or 0,
            0.5,
        )

    def test_repetitive_arm_is_measured_from_available_landmarks(self) -> None:
        _, results = self.analyze_region_trajectory("left_arm")
        result = results[-1]

        self.assertTrue(
            result.region_feature_validity["left_arm"]["repetition_rate_hz"]
        )
        self.assertGreaterEqual(
            result.region_metrics["left_arm"]["repetition_rate_hz"] or 0,
            0.5,
        )
        self.assertTrue(result.region_feature_validity["right_arm"]["repetition_rate_hz"])
        self.assertEqual(
            result.region_metrics["right_arm"]["repetition_rate_hz"],
            0.0,
        )

    def test_isolated_head_motion_does_not_become_repetitive(self) -> None:
        poses = region_trajectory("head", repeat_hz=0.0)
        for index, landmarks in enumerate(poses):
            if index == 12:
                for name in ("nose", "left_ear", "right_ear"):
                    landmarks[LANDMARK_INDICES[name]].x += 0.09
        result = self.make_branch(
            poses,
            region_min_observations=6,
            region_min_span_seconds=1.5,
        )
        analyzed = [
            result.analyze(self.frame(), timestamp_seconds=10 + index * 0.15)
            for index in range(len(poses))
        ][-1]

        self.assertTrue(
            analyzed.region_feature_validity["head"]["repetition_rate_hz"]
        )
        self.assertEqual(
            analyzed.region_metrics["head"]["repetition_rate_hz"],
            0.0,
        )

    def test_region_repetition_has_explicit_warmup_and_gap_reset(self) -> None:
        branch = self.make_branch(
            region_trajectory("torso", count=8),
            region_min_observations=6,
            region_min_span_seconds=1.5,
        )
        first = branch.analyze(self.frame(), timestamp_seconds=1.0)
        self.assertFalse(
            first.region_feature_validity["torso"]["repetition_rate_hz"]
        )
        self.assertIn(
            "warming up",
            first.region_unavailable_reasons["torso"]["repetition_rate_hz"],
        )

        branch = self.make_branch(
            region_trajectory("torso", count=10),
            region_min_observations=6,
            region_min_span_seconds=1.0,
        )
        for index in range(7):
            branch.analyze(self.frame(), timestamp_seconds=1.0 + index * 0.15)
        after_gap = branch.analyze(self.frame(), timestamp_seconds=4.0)
        self.assertFalse(
            after_gap.region_feature_validity["torso"]["repetition_rate_hz"]
        )
        self.assertIn("history reset", after_gap.pose_diagnostic["regions"]["torso"]["reason"])

    def test_missing_region_landmarks_do_not_invalidate_other_regions(self) -> None:
        landmarks = pose_landmarks()
        for name in (
            "nose",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
        ):
            landmarks[LANDMARK_INDICES[name]].visibility = 0.1
            landmarks[LANDMARK_INDICES[name]].presence = 0.1
        result = self.make_branch([landmarks]).analyze(
            self.frame(),
            timestamp_seconds=1.0,
        )

        self.assertFalse(result.pose_diagnostic["regions"]["head"]["available"])
        self.assertIn("valid landmark(s)", result.pose_diagnostic["regions"]["head"]["reason"])
        self.assertTrue(result.pose_diagnostic["regions"]["left_leg"]["available"])

    def test_valid_pose_has_stable_named_finite_posture_features(self) -> None:
        branch = self.make_branch([pose_landmarks()])

        result = branch.analyze(self.frame(), timestamp_seconds=1.0)

        self.assertEqual(result.status, "warming_up")
        self.assertEqual(result.feature_names, FEATURE_NAMES)
        self.assertEqual(len(result.feature_vector), len(FEATURE_NAMES))
        self.assertTrue(result.input_quality)
        self.assertTrue(all(result.landmark_validity.values()))
        self.assertTrue(result.feature_validity["shoulder_alignment"])
        self.assertTrue(result.feature_validity["left_upper_arm_angle_deg"])
        self.assertTrue(all(
            value is None or math.isfinite(value)
            for value in result.feature_vector
        ))
        self.assertIsNone(result.provisional_distress)
        assert result.pose_diagnostic["stage_timings_ms"]["pose_inference"] >= 0
        assert (
            result.pose_diagnostic["stage_timings_ms"][
                "region_feature_calculation"
            ]
            >= 0
        )

    def test_mediapipe_wrist_indices_and_confidence_fields_are_not_swapped(self) -> None:
        landmarks = pose_landmarks()
        landmarks[15] = SimpleNamespace(
            x=0.23, y=0.71, visibility=0.91, presence=0.82
        )
        landmarks[16] = SimpleNamespace(
            x=0.77, y=0.69, visibility=0.63, presence=0.94
        )
        result = self.make_branch([landmarks]).analyze(
            self.frame(),
            timestamp_seconds=1.0,
        )

        left = result.pose_diagnostic["wrists"]["left"]
        right = result.pose_diagnostic["wrists"]["right"]
        self.assertEqual(LANDMARK_INDICES["left_wrist"], 15)
        self.assertEqual(LANDMARK_INDICES["right_wrist"], 16)
        self.assertEqual(left["landmark_index"], 15)
        self.assertEqual(right["landmark_index"], 16)
        self.assertEqual(left["normalized_xy"], (0.23, 0.71))
        self.assertEqual(right["normalized_xy"], (0.77, 0.69))
        self.assertEqual(left["visibility"], 0.91)
        self.assertEqual(left["presence"], 0.82)
        self.assertEqual(right["visibility"], 0.63)
        self.assertEqual(right["presence"], 0.94)
        self.assertTrue(left["passed_threshold"])
        self.assertTrue(right["passed_threshold"])

    def test_wrist_fails_if_any_reported_confidence_is_below_threshold(self) -> None:
        landmarks = pose_landmarks()
        landmarks[15] = SimpleNamespace(
            x=0.34, y=0.68, visibility=0.9, presence=0.2
        )

        result = self.make_branch([landmarks]).analyze(
            self.frame(),
            timestamp_seconds=1.0,
        )

        left = result.pose_diagnostic["wrists"]["left"]
        self.assertFalse(result.landmark_validity["left_wrist"])
        self.assertEqual(left["visibility"], 0.9)
        self.assertEqual(left["presence"], 0.2)
        self.assertFalse(left["passed_threshold"])
        self.assertIn("0.200 is below", left["rejection_reason"])

    def test_missing_wrist_scores_are_not_defaulted_to_a_confident_value(self) -> None:
        landmarks = pose_landmarks()
        landmarks[15] = SimpleNamespace(x=0.34, y=0.68)

        result = self.make_branch([landmarks]).analyze(
            self.frame(),
            timestamp_seconds=1.0,
        )

        left = result.pose_diagnostic["wrists"]["left"]
        self.assertFalse(result.landmark_validity["left_wrist"])
        self.assertIsNone(left["visibility"])
        self.assertIsNone(left["presence"])
        self.assertEqual(left["normalized_xy"], (0.34, 0.68))
        self.assertEqual(
            left["rejection_reason"],
            "Neither visibility nor presence confidence was returned.",
        )

    def test_temporary_diagnostic_outputs_overlay_and_per_frame_log(self) -> None:
        branch = self.make_branch([pose_landmarks()])
        branch.diagnostics_enabled = True

        with self.assertLogs("branches.behavior_branch", level="INFO") as captured:
            result = branch.analyze(self.frame(), timestamp_seconds=1.0)

        self.assertEqual(
            result.pose_diagnostic["label"],
            "TEMPORARY LIVE POSE DIAGNOSTIC",
        )
        self.assertTrue(result.pose_diagnostic["frame_received"])
        self.assertEqual(result.pose_diagnostic["frame_dimensions"], (120, 160, 3))
        self.assertTrue(result.pose_diagnostic["pose_detected"])
        self.assertIsNotNone(result.debug_overlay)
        self.assertIn("TEMPORARY LIVE POSE DIAGNOSTIC", captured.output[0])

    def test_no_pose_diagnostic_still_includes_camera_overlay(self) -> None:
        branch = self.make_branch([None])
        branch.diagnostics_enabled = True

        with self.assertLogs("branches.behavior_branch", level="INFO"):
            result = branch.analyze(self.frame(), timestamp_seconds=1.0)

        self.assertFalse(result.pose_diagnostic["pose_detected"])
        self.assertEqual(
            result.pose_diagnostic["wrists"]["left"]["rejection_reason"],
            "Pose Landmarker did not detect a pose.",
        )
        self.assertIsNotNone(result.debug_overlay)

    def test_first_frame_has_no_fabricated_movement(self) -> None:
        branch = self.make_branch([pose_landmarks()])

        result = branch.analyze(self.frame(), timestamp_seconds=1.0)

        self.assertIsNone(result.feature_values["movement_energy"])
        self.assertFalse(result.feature_validity["movement_energy"])
        self.assertEqual(result.baseline_sample_count, 0)
        self.assertFalse(result.baseline_ready)

    def test_acceleration_stays_unavailable_until_two_speed_samples_exist(self) -> None:
        branch = self.make_branch(
            [
                pose_landmarks(),
                pose_landmarks(
                    wrist_positions={
                        "left_wrist": (0.38, 0.68),
                        "right_wrist": (0.62, 0.68),
                    }
                ),
            ]
        )
        branch.analyze(self.frame(), timestamp_seconds=1.0)
        result = branch.analyze(self.frame(), timestamp_seconds=1.1)

        self.assertTrue(result.feature_validity["movement_energy"])
        self.assertIsNone(result.feature_values["wrist_peak_acceleration"])
        self.assertFalse(result.feature_validity["wrist_peak_acceleration"])
        self.assertIn(
            "two consecutive wrist-speed samples",
            result.unavailable_reasons["wrist_peak_acceleration"],
        )

    def test_stable_observations_warm_up_bounded_movement_baseline(self) -> None:
        branch = self.make_branch(
            [pose_landmarks() for _ in range(8)],
            baseline_min_samples=3,
            baseline_window=4,
        )
        results = [
            branch.analyze(self.frame(), timestamp_seconds=1.0 + index * 0.1)
            for index in range(8)
        ]

        self.assertTrue(results[-1].baseline_ready)
        self.assertEqual(results[-1].baseline_sample_count, 4)
        self.assertEqual(results[-1].status, "ok")
        self.assertEqual(results[3].status, "warming_up")
        self.assertEqual(results[4].status, "ok")
        self.assertAlmostEqual(results[-1].baseline_energy or 0.0, 0.0)

    def test_movement_and_sudden_rule_compare_against_recent_baseline(self) -> None:
        poses = [pose_landmarks() for _ in range(7)]
        poses.append(
            pose_landmarks(
                wrist_positions={
                    "left_wrist": (0.44, 0.68),
                    "right_wrist": (0.56, 0.68),
                }
            )
        )
        branch = self.make_branch(
            poses,
            baseline_min_samples=3,
            baseline_window=5,
            sudden_movement_threshold=0.75,
        )
        results = [
            branch.analyze(self.frame(), timestamp_seconds=1.0 + index * 0.1)
            for index in range(len(poses))
        ]
        movement_result = results[-1]

        self.assertGreater(movement_result.feature_values["movement_energy"] or 0, 0)
        self.assertEqual(movement_result.feature_values["sudden_movement"], 1.0)
        self.assertGreater(
            movement_result.feature_values["movement_change_from_baseline"] or 0,
            0,
        )

    def test_partial_visibility_keeps_valid_features_and_marks_missing_ones(self) -> None:
        pose = pose_landmarks(visibility_overrides={"left_wrist": 0.1})
        result = self.make_branch([pose]).analyze(
            self.frame(),
            timestamp_seconds=1.0,
        )

        self.assertEqual(result.status, "partial")
        self.assertTrue(result.input_quality)
        self.assertFalse(result.landmark_validity["left_wrist"])
        self.assertFalse(result.feature_validity["left_forearm_angle_deg"])
        self.assertIn("left_forearm_angle_deg", result.unavailable_reasons)

    def test_no_person_and_too_few_visible_landmarks_are_unknown(self) -> None:
        no_person = self.make_branch([None]).analyze(
            self.frame(),
            timestamp_seconds=1.0,
        )
        low_visibility_pose = pose_landmarks(
            visibility_overrides={
                name: 0.1 for name in LANDMARK_INDICES
            }
        )
        unusable = self.make_branch([low_visibility_pose]).analyze(
            self.frame(),
            timestamp_seconds=1.0,
        )

        self.assertEqual(no_person.status, "unknown")
        self.assertEqual(no_person.unavailable_reasons["movement_energy"], "person_not_detected")
        self.assertEqual(unusable.status, "unknown")
        self.assertEqual(unusable.unavailable_reasons["movement_energy"], "insufficient_landmarks")
        self.assertFalse(unusable.input_quality)

    def test_invalid_frame_and_timestamp_return_unknown(self) -> None:
        branch = self.make_branch([pose_landmarks(), pose_landmarks()])
        invalid_frame = branch.analyze(np.zeros((0, 0, 3), dtype=np.uint8))
        valid_frame = branch.analyze(self.frame(), timestamp_seconds=1.0)
        invalid_timestamp = branch.analyze(
            self.frame(),
            timestamp_seconds=float("nan"),
        )

        self.assertEqual(invalid_frame.status, "unknown")
        self.assertEqual(invalid_frame.unavailable_reasons["movement_energy"], "invalid_frame")
        self.assertEqual(valid_frame.status, "warming_up")
        self.assertEqual(invalid_timestamp.unavailable_reasons["movement_energy"], "invalid_timestamp")

    def test_deprecated_repetition_change_stays_unavailable_until_rate_is_measured(self) -> None:
        result = self.make_branch([pose_landmarks()]).analyze(
            self.frame(),
            timestamp_seconds=1.0,
        )

        self.assertEqual(result.repetition_status, "warming_up")
        self.assertIsNone(result.feature_values["repetition_change"])
        self.assertFalse(result.feature_validity["repetition_change"])
        self.assertIn("Deprecated", result.unavailable_reasons["repetition_change"])

    def test_stationary_wrists_produce_low_valid_movement_features(self) -> None:
        _, results = self.analyze_trajectory([0.0] * 30)
        result = results[-1]

        self.assertTrue(result.feature_validity["movement_energy"])
        self.assertAlmostEqual(result.feature_values["movement_energy"] or 0.0, 0.0)
        self.assertAlmostEqual(result.feature_values["wrist_average_speed"] or 0.0, 0.0)
        self.assertAlmostEqual(
            result.feature_values["wrist_repetition_rate_hz"] or 0.0,
            0.0,
        )
        self.assertTrue(result.feature_validity["left_wrist_visibility"])
        self.assertTrue(result.feature_validity["right_wrist_visibility"])

    def test_repeated_hand_movement_increases_energy_and_repetition_rate(self) -> None:
        import math

        offsets = [
            0.18 * math.sin(2 * math.pi * 1.0 * index * 0.15)
            for index in range(30)
        ]
        _, results = self.analyze_trajectory(offsets)
        result = results[-1]

        self.assertGreater(result.feature_values["movement_energy"] or 0.0, 0.2)
        self.assertGreater(result.feature_values["wrist_repetition_rate_hz"] or 0.0, 0.3)
        self.assertGreater(
            result.feature_values["wrist_elevated_movement_proportion"] or 0.0,
            0.0,
        )
        self.assertEqual(result.repetition_status, "measured")

    def test_vigorous_movement_has_stronger_speed_and_amplitude_than_gentle(self) -> None:
        import math

        gentle = [
            0.035 * math.sin(2 * math.pi * 0.8 * index * 0.15)
            for index in range(30)
        ]
        vigorous = [
            0.22 * math.sin(2 * math.pi * 0.8 * index * 0.15)
            for index in range(30)
        ]
        _, gentle_results = self.analyze_trajectory(gentle)
        _, vigorous_results = self.analyze_trajectory(vigorous)

        self.assertGreater(
            vigorous_results[-1].feature_values["wrist_average_speed"] or 0.0,
            gentle_results[-1].feature_values["wrist_average_speed"] or 0.0,
        )
        self.assertGreater(
            vigorous_results[-1].feature_values["wrist_movement_amplitude"] or 0.0,
            gentle_results[-1].feature_values["wrist_movement_amplitude"] or 0.0,
        )

    def test_isolated_gesture_does_not_register_as_sustained_repetition(self) -> None:
        offsets = [0.0] * 30
        offsets[14] = 0.9
        offsets[15] = 0.0
        _, results = self.analyze_trajectory(offsets)
        result = results[-1]

        self.assertAlmostEqual(
            result.feature_values["wrist_repetition_rate_hz"] or 0.0,
            0.0,
        )
        self.assertLess(
            result.feature_values["wrist_elevated_movement_proportion"] or 0.0,
            0.25,
        )

    def test_low_confidence_wrists_are_not_used_as_movement(self) -> None:
        _, results = self.analyze_trajectory([0.0, 1.0, -1.0, 1.0], visibility=0.1)
        result = results[-1]

        self.assertFalse(result.feature_validity["movement_energy"])
        self.assertIsNone(result.feature_values["movement_energy"])
        self.assertTrue(result.feature_validity["left_wrist_visibility"])
        self.assertAlmostEqual(result.feature_values["left_wrist_visibility"] or 0.0, 0.1)
        self.assertIn(
            "visibility threshold",
            result.unavailable_reasons["movement_energy"],
        )

    def test_valid_branch_features_survive_fusion_and_trigger_movement_rule(self) -> None:
        import math

        offsets = [
            0.18 * math.sin(2 * math.pi * index * 0.15)
            for index in range(30)
        ]
        _, results = self.analyze_trajectory(offsets)
        behavior = results[-1]
        timestamp = 1.0 + (len(results) - 1) * 0.15
        evidence = build_fusion_evidence(
            {
                "latest_face_result": None,
                "latest_behavior_result": behavior,
                "latest_audio_result": None,
            }
        )
        record = evidence_to_feature_record(evidence)
        detector = RuleBasedCueDetector(
            RuleDetectionConfig(
                combination_mode="movement_only",
                movement_repetition_seconds=0,
            )
        )

        inference = detector.evaluate(
            record,
            modality_source_timestamps={"behavior": timestamp, "audio": None},
            now=timestamp,
        )

        self.assertTrue(record.feature_validity["behavior.movement_energy"])
        self.assertTrue(
            record.feature_validity["behavior.wrist_repetition_rate_hz"]
        )
        self.assertTrue(
            record.feature_validity["behavior.wrist_movement_amplitude"]
        )
        self.assertEqual(inference.cue_indicators["movement_repetitive"], True)
        self.assertEqual(inference.status, "potential_distress")

    def test_insufficient_wrist_history_has_specific_reason(self) -> None:
        branch = self.make_branch(
            [pose_landmarks(), pose_landmarks()],
            movement_max_gap_seconds=0.25,
        )
        branch.analyze(self.frame(), timestamp_seconds=1.0)
        result = branch.analyze(self.frame(), timestamp_seconds=1.5)

        self.assertFalse(result.feature_validity["movement_energy"])
        self.assertIn(
            "Only 1 valid wrist observation",
            result.unavailable_reasons["movement_energy"],
        )
        self.assertIn(
            "frame gap was 0.50s",
            result.unavailable_reasons["movement_energy"],
        )

    def test_camera_translation_jitter_is_removed_by_torso_normalization(self) -> None:
        import math

        poses = [
            pose_landmarks(shift_x=0.01 * math.sin(index * 1.7))
            for index in range(24)
        ]
        branch = self.make_branch(poses)
        results = [
            branch.analyze(self.frame(), timestamp_seconds=1.0 + index * 0.15)
            for index in range(len(poses))
        ]

        self.assertLess(results[-1].feature_values["movement_energy"] or 0.0, 1e-6)

    def test_irregular_timestamps_use_elapsed_time_and_long_gaps_reset_tracking(self) -> None:
        poses = [
            pose_landmarks(wrist_positions={"left_wrist": (0.34 + index * 0.02, 0.68)})
            for index in range(4)
        ]
        poses.append(
            pose_landmarks(wrist_positions={"left_wrist": (0.50, 0.68)})
        )
        branch = self.make_branch(poses, movement_max_gap_seconds=0.8)
        times = (1.0, 1.1, 1.35, 1.6, 3.0)
        results = [
            branch.analyze(self.frame(), timestamp_seconds=timestamp)
            for timestamp in times
        ]

        self.assertGreater(results[3].feature_values["wrist_average_speed"] or 0.0, 0.0)
        self.assertFalse(results[4].feature_validity["movement_energy"])
        self.assertIsNone(results[4].feature_values["movement_energy"])

    def test_reset_clears_temporal_history_and_close_releases_landmarker(self) -> None:
        landmarker = QueueLandmarker([pose_landmarks(), pose_landmarks()])
        branch = UpperBodyBehaviorBranch(
            landmarker=landmarker,
            baseline_min_samples=1,
            baseline_window=2,
        )
        branch.analyze(self.frame(), timestamp_seconds=1.0)
        branch.analyze(self.frame(), timestamp_seconds=1.1)
        self.assertEqual(len(branch._movement_history), 1)

        branch.reset()
        self.assertEqual(len(branch._movement_history), 0)
        self.assertEqual(len(branch._pose_observations), 0)
        branch.close()
        self.assertTrue(landmarker.closed)
        self.assertFalse(branch.is_ready)

    def test_initialization_and_inference_failures_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            branch = UpperBodyBehaviorBranch(model_path=Path(directory) / "missing.task")
        initialization_result = branch.analyze(self.frame(), timestamp_seconds=1.0)
        inference_result = UpperBodyBehaviorBranch(
            landmarker=FailingLandmarker()
        ).analyze(self.frame(), timestamp_seconds=1.0)

        self.assertEqual(initialization_result.status, "unknown")
        self.assertIn("model asset not found", initialization_result.status_message)
        self.assertEqual(inference_result.status, "unknown")
        self.assertIn("mock pose inference failure", inference_result.status_message)


if __name__ == "__main__":
    unittest.main()
