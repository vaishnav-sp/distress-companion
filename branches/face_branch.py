"""Local face detection and six-class expression inference.

The expression output is a weak feature only. It is not a distress decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from PIL import Image

from branches.face_model import (
    ARCHITECTURE,
    make_evaluation_transform,
    preprocessing_metadata,
)
from branches.interfaces import BranchOutput, BranchProcessor
from config import (
    FACE_CLASS_NAMES,
    FACE_DISPLAY_LABELS,
    FACE_DISPLAY_ORDER,
    FACE_MAX_BRIGHTNESS,
    FACE_MIN_BLUR_VARIANCE,
    FACE_MIN_BRIGHTNESS,
    FACE_MIN_SIZE_PIXELS,
    FACE_MODEL_PATH,
)

logger = logging.getLogger(__name__)
NEGATIVE_AFFECT_ORDER = ("angry", "fear", "sad")
FEATURE_VECTOR_ORDER = FACE_DISPLAY_ORDER
MODEL_SETUP_MESSAGE = (
    "The local face-expression model has not been trained. From the project folder, "
    "run: python -m training.train_face_model"
)


@dataclass(frozen=True)
class PredictionConfidence:
    """Softmax summary statistics; these describe uncertainty, not correctness."""

    maximum_probability: float
    normalized_entropy: float
    top_two_margin: float


@dataclass(frozen=True)
class FaceResult(BranchOutput):
    """Face-branch output with nullable seven-label probabilities and abstention data."""

    face_detected: bool
    expression_probabilities: Mapping[str, float | None]
    class_availability: Mapping[str, bool]
    class_validity: Mapping[str, bool]
    negative_affect_feature: float | None
    face_quality_score: float | None
    face_quality_status: str
    prediction_confidence: PredictionConfidence | None
    feature_vector: tuple[float | None, ...]
    feature_vector_order: tuple[str, ...]
    abstention_reason: str | None
    status_message: str


class FaceExpressionBranch(BranchProcessor):
    """Detect a face and infer expression probabilities with local trained weights."""

    def __init__(
        self,
        weights_path: Path = FACE_MODEL_PATH,
        detector: Any | None = None,
        model: Any | None = None,
        preprocess: Any | None = None,
        device: str | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.initialization_exception_type: str | None = None
        try:
            self.detector = (
                detector if detector is not None else self._create_default_detector()
            )
        except Exception as exc:
            self.detector = None
            self.initialization_exception_type = type(exc).__name__
            logger.exception("Unable to initialize the OpenCV Haar face detector")
        self.model = model
        self.preprocess = preprocess or make_evaluation_transform()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model_loaded = model is not None
        self.initialization_error = (
            None
            if self._detector_is_ready()
            else (
                "The local OpenCV Haar face detector could not be initialized"
                + (
                    f" ({self.initialization_exception_type})."
                    if self.initialization_exception_type
                    else "."
                )
            )
        )

    def process(self, input_data: object) -> BranchOutput:
        """Implement the shared future-branch protocol."""
        result = self.analyze(input_data)
        return BranchOutput(
            features=result.features,
            provisional_distress=result.provisional_distress,
            input_quality=result.input_quality,
        )

    def analyze(self, frame: object) -> FaceResult:
        """Analyze one in-memory OpenCV BGR frame, abstaining on unusable evidence."""
        invalid = self._normalize_frame(frame)
        if invalid is None:
            return self._abstain(
                face_detected=False,
                reason="invalid_image",
                message="Expected a non-empty uint8 BGR, grayscale, or BGRA image array.",
            )
        color_frame = invalid
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

        if not self._detector_is_ready():
            logger.error("OpenCV Haar face detector is not available")
            return self._abstain(
                face_detected=False,
                reason="face_detector_unavailable",
                message="The local OpenCV face detector could not be initialized.",
            )
        try:
            detections = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(FACE_MIN_SIZE_PIXELS, FACE_MIN_SIZE_PIXELS),
            )
        except Exception as exc:
            logger.exception("Face detection failed")
            return self._abstain(
                face_detected=False,
                reason="face_detection_error",
                message=f"Face detection failed ({type(exc).__name__}): {exc}",
            )

        boxes = np.asarray(detections)
        if boxes.size == 0:
            return self._abstain(
                face_detected=False,
                reason="no_face",
                message="No usable face was detected; expression output is unknown.",
            )
        try:
            boxes = boxes.reshape(-1, 4).astype(int)
        except (TypeError, ValueError):
            return self._abstain(
                face_detected=False,
                reason="invalid_face_detection",
                message="The face detector returned invalid coordinates.",
            )

        x, y, width, height = max(boxes, key=lambda box: int(box[2]) * int(box[3]))
        frame_height, frame_width = color_frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame_width, x + width), min(frame_height, y + height)
        if x1 <= x0 or y1 <= y0:
            return self._abstain(
                face_detected=False,
                reason="invalid_face_crop",
                message="The detected face lies outside the image.",
            )

        face_crop = color_frame[y0:y1, x0:x1]
        quality_score, quality_problem = self._assess_quality(face_crop)
        if quality_problem:
            return self._abstain(
                face_detected=True,
                reason=quality_problem,
                message=self._quality_message(quality_problem),
                quality_score=quality_score,
                quality_status="poor",
            )

        model_status = self._ensure_model_loaded()
        if model_status is not None:
            return self._abstain(
                face_detected=True,
                reason="model_unavailable",
                message=model_status,
                quality_score=quality_score,
                quality_status="usable",
            )

        try:
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            transformed = self.preprocess(Image.fromarray(rgb_crop))
            batch = transformed.unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logits = self.model(batch)
                if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (
                    1,
                    len(FACE_CLASS_NAMES),
                ):
                    raise ValueError("Model output must have shape (1, 6).")
                if not torch.isfinite(logits).all():
                    raise ValueError("Model returned non-finite scores.")
                probabilities = torch.softmax(logits[0].float(), dim=0).cpu().numpy()
            if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
                raise ValueError("Model returned invalid class probabilities.")
            probabilities = probabilities / probabilities.sum()
        except Exception as exc:
            logger.exception("Face-expression inference failed")
            return self._abstain(
                face_detected=True,
                reason="inference_error",
                message=f"Face-expression inference failed ({type(exc).__name__}): {exc}",
                quality_score=quality_score,
                quality_status="usable",
            )

        probability_by_source = {
            FACE_DISPLAY_LABELS[name]: float(probabilities[index])
            for index, name in enumerate(FACE_CLASS_NAMES)
        }
        displayed: dict[str, float | None] = {
            label: probability_by_source.get(label) for label in FACE_DISPLAY_ORDER
        }
        valid = {label: displayed[label] is not None for label in FACE_DISPLAY_ORDER}
        negative_affect = float(
            sum(float(displayed[label]) for label in NEGATIVE_AFFECT_ORDER) / 3.0
        )
        sorted_probabilities = np.sort(probabilities)
        entropy = float(
            -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
            / np.log(len(FACE_CLASS_NAMES))
        )
        confidence = PredictionConfidence(
            maximum_probability=float(sorted_probabilities[-1]),
            normalized_entropy=entropy,
            top_two_margin=float(sorted_probabilities[-1] - sorted_probabilities[-2]),
        )
        vector = tuple(displayed[label] for label in FEATURE_VECTOR_ORDER)
        features = {
            f"expression_{label}": float(value)
            for label, value in displayed.items()
            if value is not None
        }
        features["negative_affect_mean_angry_fear_sad"] = negative_affect
        features["face_quality_score"] = quality_score
        return FaceResult(
            features=features,
            provisional_distress=negative_affect,
            input_quality=True,
            face_detected=True,
            expression_probabilities=displayed,
            class_availability=self._class_availability(),
            class_validity=valid,
            negative_affect_feature=negative_affect,
            face_quality_score=quality_score,
            face_quality_status="usable",
            prediction_confidence=confidence,
            feature_vector=vector,
            feature_vector_order=FEATURE_VECTOR_ORDER,
            abstention_reason=None,
            status_message=(
                "Expression probabilities are model estimates, not proof of emotion "
                "or a distress decision."
            ),
        )

    @staticmethod
    def _create_default_detector() -> Any | None:
        cascade_factory = getattr(cv2, "CascadeClassifier", None)
        cascade_directory = getattr(getattr(cv2, "data", None), "haarcascades", None)
        if not callable(cascade_factory) or not cascade_directory:
            logger.error("Installed OpenCV does not expose the Haar CascadeClassifier API")
            return None
        return cascade_factory(
            str(Path(cascade_directory) / "haarcascade_frontalface_default.xml")
        )

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

    def _detector_is_ready(self) -> bool:
        if self.detector is None:
            return False
        is_empty = getattr(self.detector, "empty", None)
        return not callable(is_empty) or not is_empty()

    @staticmethod
    def _assess_quality(face_crop: np.ndarray) -> tuple[float, str | None]:
        height, width = face_crop.shape[:2]
        if min(height, width) < FACE_MIN_SIZE_PIXELS:
            return 0.0, "face_too_small"
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        size_score = min(1.0, min(height, width) / 128.0)
        brightness_score = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
        sharpness_score = min(1.0, blur_variance / 100.0)
        quality_score = float(
            0.4 * size_score + 0.3 * brightness_score + 0.3 * sharpness_score
        )
        if brightness < FACE_MIN_BRIGHTNESS or brightness > FACE_MAX_BRIGHTNESS:
            return quality_score, "face_bad_exposure"
        if blur_variance < FACE_MIN_BLUR_VARIANCE:
            return quality_score, "face_too_blurry"
        return quality_score, None

    @staticmethod
    def _quality_message(reason: str) -> str:
        messages = {
            "face_too_small": "The detected face is too small for a reliable expression estimate.",
            "face_bad_exposure": "The face is too dark or overexposed; improve lighting and retry.",
            "face_too_blurry": "The detected face is too blurry; hold still and retry.",
        }
        return messages.get(reason, "The detected face is not usable.")

    def _ensure_model_loaded(self) -> str | None:
        if self._model_loaded:
            return None
        if not self.weights_path.is_file():
            return MODEL_SETUP_MESSAGE
        try:
            checkpoint = torch.load(
                self.weights_path,
                map_location="cpu",
                weights_only=True,
            )
            metadata = checkpoint["metadata"]
            expected_mapping = {name: index for index, name in enumerate(FACE_CLASS_NAMES)}
            if metadata.get("architecture") != ARCHITECTURE:
                raise ValueError("Saved checkpoint architecture does not match this branch.")
            if metadata.get("class_names") != list(FACE_CLASS_NAMES):
                raise ValueError("Saved checkpoint class order does not match this branch.")
            if metadata.get("class_to_index") != expected_mapping:
                raise ValueError("Saved checkpoint class-index mapping is invalid.")
            if metadata.get("display_label_mapping") != FACE_DISPLAY_LABELS:
                raise ValueError("Saved checkpoint display-label mapping is invalid.")
            if metadata.get("preprocessing") != preprocessing_metadata():
                raise ValueError("Saved checkpoint preprocessing does not match this branch.")

            from branches.face_model import build_face_model

            model = build_face_model(num_classes=len(FACE_CLASS_NAMES), pretrained=False)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            self.model = model.to(self.device).eval()
            self._model_loaded = True
            return None
        except Exception as exc:
            logger.exception("Unable to load local face-expression checkpoint %s", self.weights_path)
            return (
                f"Could not load the local face-expression model "
                f"({type(exc).__name__}): {exc}"
            )

    @staticmethod
    def _class_availability() -> dict[str, bool]:
        """Six trained labels are supported; disgust is absent from the model."""
        return {label: label != "disgust" for label in FACE_DISPLAY_ORDER}

    @classmethod
    def _abstain(
        cls,
        face_detected: bool,
        reason: str,
        message: str,
        quality_score: float | None = None,
        quality_status: str = "unknown",
    ) -> FaceResult:
        probabilities = {label: None for label in FACE_DISPLAY_ORDER}
        return FaceResult(
            features=None,
            provisional_distress=None,
            input_quality=False,
            face_detected=face_detected,
            expression_probabilities=probabilities,
            class_availability=cls._class_availability(),
            class_validity={label: False for label in FACE_DISPLAY_ORDER},
            negative_affect_feature=None,
            face_quality_score=quality_score,
            face_quality_status=quality_status,
            prediction_confidence=None,
            feature_vector=tuple(None for _ in FEATURE_VECTOR_ORDER),
            feature_vector_order=FEATURE_VECTOR_ORDER,
            abstention_reason=reason,
            status_message=message,
        )
