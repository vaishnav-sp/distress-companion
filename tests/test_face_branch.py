import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn

from branches.face_branch import (
    FEATURE_VECTOR_ORDER,
    MODEL_SETUP_MESSAGE,
    FaceExpressionBranch,
)
from branches.face_model import ARCHITECTURE, preprocessing_metadata
from config import FACE_CLASS_NAMES, FACE_DISPLAY_LABELS, FACE_DISPLAY_ORDER


class FakeDetector:
    def __init__(self, boxes: list[tuple[int, int, int, int]]) -> None:
        self.boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)

    def detectMultiScale(self, image: np.ndarray, **kwargs: object) -> np.ndarray:
        return self.boxes


class FixedLogitsModel(nn.Module):
    def __init__(self, values: list[float] | None = None, fail: bool = False) -> None:
        super().__init__()
        self.values = values or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.fail = fail

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self.fail:
            raise RuntimeError("mock inference failure")
        return torch.tensor([self.values], dtype=torch.float32)


class CheckpointModel(FixedLogitsModel):
    def load_state_dict(self, state_dict: object, strict: bool = True) -> None:
        return None


def usable_frame() -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.integers(40, 210, size=(160, 160, 3), dtype=np.uint8)


class FaceBranchTests(unittest.TestCase):
    def make_branch(
        self,
        boxes: list[tuple[int, int, int, int]] | None = None,
        model: nn.Module | None = None,
        weights_path: Path | None = None,
    ) -> FaceExpressionBranch:
        return FaceExpressionBranch(
            weights_path=weights_path or Path("missing-face-model.pt"),
            detector=FakeDetector(boxes if boxes is not None else [(16, 16, 128, 128)]),
            model=model,
            preprocess=lambda image: torch.zeros((3, 224, 224), dtype=torch.float32),
            device="cpu",
        )

    def test_probability_output_is_finite_normalized_and_in_expected_order(self) -> None:
        result = self.make_branch(model=FixedLogitsModel()).analyze(usable_frame())

        self.assertTrue(result.input_quality)
        self.assertTrue(result.face_detected)
        self.assertEqual(result.feature_vector_order, FEATURE_VECTOR_ORDER)
        self.assertEqual(result.feature_vector_order, FACE_DISPLAY_ORDER)
        values = [
            value for value in result.expression_probabilities.values() if value is not None
        ]
        self.assertEqual(len(values), 6)
        self.assertTrue(all(np.isfinite(values)))
        self.assertAlmostEqual(sum(values), 1.0)
        self.assertEqual(result.class_validity["disgust"], False)
        self.assertEqual(result.expression_probabilities["disgust"], None)
        self.assertTrue(all(result.class_validity[label] for label in (
            "angry", "fear", "happy", "neutral", "sad", "surprise"
        )))
        self.assertFalse(result.class_availability["disgust"])

    def test_class_display_mapping_is_explicit_and_order_stable(self) -> None:
        self.assertEqual(
            FACE_DISPLAY_LABELS,
            {
                "Natural": "neutral",
                "joy": "happy",
                "sadness": "sad",
                "anger": "angry",
                "fear": "fear",
                "surprise": "surprise",
            },
        )
        self.assertEqual(
            tuple(FACE_DISPLAY_LABELS[name] for name in FACE_CLASS_NAMES),
            ("angry", "fear", "happy", "neutral", "sad", "surprise"),
        )

    def test_negative_affect_is_mean_of_angry_fear_and_sad(self) -> None:
        result = self.make_branch(model=FixedLogitsModel()).analyze(usable_frame())
        expected = sum(
            result.expression_probabilities[label] or 0.0
            for label in ("angry", "fear", "sad")
        ) / 3

        self.assertAlmostEqual(result.negative_affect_feature or 0.0, expected)
        self.assertEqual(result.provisional_distress, result.negative_affect_feature)

    def test_feature_vector_has_seven_entries_with_disgust_unavailable(self) -> None:
        result = self.make_branch(model=FixedLogitsModel()).analyze(usable_frame())

        self.assertEqual(len(result.feature_vector), 7)
        self.assertEqual(result.feature_vector_order[1], "disgust")
        self.assertIsNone(result.feature_vector[1])
        self.assertEqual(
            tuple(value is not None for value in result.feature_vector),
            (True, False, True, True, True, True, True),
        )

    def test_no_face_abstains_without_classifying_as_neutral(self) -> None:
        result = self.make_branch(boxes=[], model=FixedLogitsModel()).analyze(usable_frame())

        self.assertFalse(result.face_detected)
        self.assertFalse(result.input_quality)
        self.assertEqual(result.abstention_reason, "no_face")
        self.assertIsNone(result.negative_affect_feature)
        self.assertTrue(all(value is None for value in result.expression_probabilities.values()))
        self.assertIsNone(result.expression_probabilities["neutral"])
        self.assertIsNone(result.prediction_confidence)

    def test_invalid_image_abstains(self) -> None:
        branch = self.make_branch(model=FixedLogitsModel())
        for frame in (None, np.zeros((8, 8, 3), dtype=np.float32), np.zeros((8, 8, 2), dtype=np.uint8)):
            with self.subTest(frame_type=type(frame).__name__):
                result = branch.analyze(frame)
                self.assertEqual(result.abstention_reason, "invalid_image")
                self.assertTrue(all(value is None for value in result.feature_vector))

    def test_small_or_low_quality_face_abstains(self) -> None:
        small_result = self.make_branch(
            boxes=[(10, 10, 32, 32)],
            model=FixedLogitsModel(),
        ).analyze(usable_frame())
        self.assertEqual(small_result.abstention_reason, "face_too_small")
        self.assertEqual(small_result.face_quality_status, "poor")
        self.assertIsNone(small_result.expression_probabilities["neutral"])

        dark = np.zeros((160, 160, 3), dtype=np.uint8)
        dark_result = self.make_branch(model=FixedLogitsModel()).analyze(dark)
        self.assertEqual(dark_result.abstention_reason, "face_bad_exposure")
        self.assertIsNone(dark_result.negative_affect_feature)

        blurred = cv2.GaussianBlur(usable_frame(), (41, 41), 0)
        blurred_result = self.make_branch(model=FixedLogitsModel()).analyze(blurred)
        self.assertEqual(blurred_result.abstention_reason, "face_too_blurry")

    def test_missing_weights_return_actionable_training_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.make_branch(
                model=None,
                weights_path=Path(directory) / "not-trained.pt",
            ).analyze(usable_frame())

        self.assertEqual(result.abstention_reason, "model_unavailable")
        self.assertIn("python -m training.train_face_model", result.status_message)
        self.assertEqual(result.status_message.splitlines()[0], MODEL_SETUP_MESSAGE.splitlines()[0])
        self.assertTrue(all(value is None for value in result.expression_probabilities.values()))

    def test_inference_error_abstains_instead_of_returning_fake_probabilities(self) -> None:
        result = self.make_branch(model=FixedLogitsModel(fail=True)).analyze(usable_frame())

        self.assertEqual(result.abstention_reason, "inference_error")
        self.assertIn("mock inference failure", result.status_message)
        self.assertIsNone(result.negative_affect_feature)
        self.assertTrue(all(value is None for value in result.expression_probabilities.values()))

    def test_nonfinite_model_output_abstains(self) -> None:
        model = FixedLogitsModel([float("nan"), 0, 0, 0, 0, 0])
        result = self.make_branch(model=model).analyze(usable_frame())

        self.assertEqual(result.abstention_reason, "inference_error")
        self.assertTrue(all(value is None for value in result.feature_vector))

    def test_corrupt_local_checkpoint_reports_load_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "model.pt"
            weights_path.write_bytes(b"not a torch checkpoint")
            result = self.make_branch(weights_path=weights_path).analyze(usable_frame())

        self.assertEqual(result.abstention_reason, "model_unavailable")
        self.assertIn("Could not load", result.status_message)

    def test_loads_local_checkpoint_after_verifying_its_label_and_preprocess_metadata(self) -> None:
        class_to_index = {name: index for index, name in enumerate(FACE_CLASS_NAMES)}
        checkpoint = {
            "metadata": {
                "architecture": ARCHITECTURE,
                "class_names": list(FACE_CLASS_NAMES),
                "class_to_index": class_to_index,
                "display_label_mapping": FACE_DISPLAY_LABELS,
                "preprocessing": preprocessing_metadata(),
            },
            "state_dict": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "model.pt"
            torch.save(checkpoint, weights_path)
            branch = self.make_branch(weights_path=weights_path)
            with patch("branches.face_model.build_face_model", return_value=CheckpointModel()):
                result = branch.analyze(usable_frame())

        self.assertTrue(result.input_quality)
        self.assertTrue(result.class_validity["angry"])
        self.assertIsNone(result.expression_probabilities["disgust"])

    def test_rejects_checkpoint_with_incorrect_class_mapping(self) -> None:
        checkpoint = {
            "metadata": {
                "architecture": ARCHITECTURE,
                "class_names": list(FACE_CLASS_NAMES),
                "class_to_index": {name: index for index, name in enumerate(FACE_CLASS_NAMES[::-1])},
                "display_label_mapping": FACE_DISPLAY_LABELS,
                "preprocessing": preprocessing_metadata(),
            },
            "state_dict": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "model.pt"
            torch.save(checkpoint, weights_path)
            branch = self.make_branch(weights_path=weights_path)
            with patch("branches.face_model.build_face_model", return_value=CheckpointModel()):
                result = branch.analyze(usable_frame())

        self.assertEqual(result.abstention_reason, "model_unavailable")
        self.assertIn("class-index mapping is invalid", result.status_message)


if __name__ == "__main__":
    unittest.main()
