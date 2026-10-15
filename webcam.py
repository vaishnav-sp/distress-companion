"""One-shot webcam diagnostics and reusable local camera capture sessions."""

import logging
from dataclasses import dataclass
import time

import cv2
import numpy as np

from config import (
    BEHAVIOR_CAPTURE_FRAME_COUNT,
    BEHAVIOR_CAPTURE_INTERVAL_SECONDS,
    CAMERA_INDEX,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraResult:
    """Result of trying to open a webcam and read a single frame."""

    frame: np.ndarray | None
    error: str | None = None
    error_type: str | None = None
    camera_opened: bool | None = None
    camera_index: int | None = None

    @property
    def available(self) -> bool:
        return self.frame is not None and self.error is None


@dataclass(frozen=True)
class CameraSequenceResult:
    """A short in-memory sequence of frames and monotonic capture times."""

    frames: tuple[tuple[np.ndarray, float], ...]
    error: str | None = None
    error_type: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.frames)


class CameraCaptureSession:
    """Own one camera handle for repeated, in-memory frame reads."""

    def __init__(self, camera_index: int = CAMERA_INDEX) -> None:
        self.camera_index = camera_index
        self._camera: cv2.VideoCapture | None = None

    @property
    def is_opened(self) -> bool:
        return self._camera is not None and bool(self._camera.isOpened())

    def read(self) -> CameraResult:
        try:
            if self._camera is None or not self._camera.isOpened():
                if self._camera is not None:
                    self._camera.release()
                self._camera = cv2.VideoCapture(self.camera_index)
            camera_opened = bool(self._camera.isOpened())
            if not camera_opened:
                self._camera.release()
                self._camera = None
                return CameraResult(
                    frame=None,
                    error=(
                        "Could not open the camera. Check that it is connected, not in use "
                        "by another app, and allowed in Windows privacy settings."
                    ),
                    camera_opened=False,
                    camera_index=self.camera_index,
                )
            success, frame = self._camera.read()
            if not success or frame is None:
                return CameraResult(
                    frame=None,
                    error="The camera opened, but no frame could be captured.",
                    camera_opened=True,
                    camera_index=self.camera_index,
                )
            return CameraResult(
                frame=frame,
                camera_opened=True,
                camera_index=self.camera_index,
            )
        except Exception as exc:
            logger.exception("Camera index %s failed while reading a frame", self.camera_index)
            return CameraResult(
                frame=None,
                error=f"Camera capture failed: {exc}",
                error_type=type(exc).__name__,
                camera_opened=self.is_opened,
                camera_index=self.camera_index,
            )

    def close(self) -> None:
        camera, self._camera = self._camera, None
        if camera is not None:
            camera.release()


def capture_frame(camera_index: int = CAMERA_INDEX) -> CameraResult:
    """Capture one frame, always releasing the camera afterwards."""
    camera: CameraCaptureSession | None = None
    try:
        camera = CameraCaptureSession(camera_index)
        return camera.read()
    except Exception as exc:
        logger.exception("Unexpected error while capturing from camera index %s", camera_index)
        return CameraResult(
            frame=None,
            error=f"Camera capture failed: {exc}",
            error_type=type(exc).__name__,
            camera_opened=None,
            camera_index=camera_index,
        )
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception:
                logger.exception("Failed to release camera index %s", camera_index)


def capture_frame_sequence(
    camera_index: int = CAMERA_INDEX,
    frame_count: int = BEHAVIOR_CAPTURE_FRAME_COUNT,
    interval_seconds: float = BEHAVIOR_CAPTURE_INTERVAL_SECONDS,
) -> CameraSequenceResult:
    """Capture a bounded, user-requested frame burst using one camera session."""
    if frame_count < 1 or interval_seconds < 0:
        return CameraSequenceResult(
            frames=(),
            error="Frame count must be positive and interval cannot be negative.",
        )

    camera: cv2.VideoCapture | None = None
    frames: list[tuple[np.ndarray, float]] = []
    try:
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            message = (
                "Could not open the camera. Check that it is connected, "
                "not in use by another app, and allowed in Windows privacy settings."
            )
            logger.warning("Unable to open camera index %s for a frame sequence", camera_index)
            return CameraSequenceResult(frames=(), error=message)

        for frame_index in range(frame_count):
            if frame_index:
                time.sleep(interval_seconds)
            success, frame = camera.read()
            captured_at = time.monotonic()
            if not success or frame is None:
                logger.warning(
                    "Camera index %s stopped after %s frames",
                    camera_index,
                    len(frames),
                )
                break
            frames.append((frame, captured_at))

        if not frames:
            return CameraSequenceResult(
                frames=(),
                error="The camera opened, but no frame could be captured.",
            )
        return CameraSequenceResult(frames=tuple(frames))
    except Exception as exc:
        logger.exception(
            "Unexpected error while capturing a frame sequence from camera index %s",
            camera_index,
        )
        message = f"Camera capture failed: {exc}"
        return CameraSequenceResult(
            frames=tuple(frames),
            error=message,
            error_type=type(exc).__name__,
        )
    finally:
        if camera is not None:
            try:
                camera.release()
            except Exception:
                logger.exception("Failed to release camera index %s", camera_index)
