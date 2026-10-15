"""Microphone diagnostics and bounded, in-memory continuous audio capture."""

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd

from config import (
    AUDIO_CAPTURE_WINDOW_SECONDS,
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    AUDIO_SILENCE_THRESHOLD,
    AUDIO_TEST_DURATION_SECONDS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MicrophoneStatus:
    """Whether a default audio input device is available."""

    available: bool
    message: str
    error_type: str | None = None


@dataclass(frozen=True)
class AudioTestResult:
    """Status and simple signal levels from a temporary test recording."""

    available: bool
    message: str
    rms_level: float | None = None
    peak_level: float | None = None


@dataclass(frozen=True)
class AudioCaptureResult:
    """Short in-memory audio window with explicit validity and quality states."""

    status: str
    message: str
    available: bool = False
    sample_rate: int = AUDIO_SAMPLE_RATE
    channels: int = AUDIO_CHANNELS
    duration_seconds: float = 0.0
    samples: np.ndarray | None = None
    rms_level: float | None = None
    peak_level: float | None = None
    quality_flag: bool = False
    error_type: str | None = None


class MicrophoneCaptureStream:
    """Continuously sample into a bounded ring buffer and expose short windows."""

    def __init__(
        self,
        duration_seconds: float = AUDIO_CAPTURE_WINDOW_SECONDS,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        silence_threshold: float = AUDIO_SILENCE_THRESHOLD,
    ) -> None:
        if duration_seconds <= 0 or sample_rate <= 0 or channels <= 0:
            raise ValueError("Audio window, sample rate, and channels must be positive.")
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        self.silence_threshold = silence_threshold
        self._window_frames = int(duration_seconds * sample_rate)
        if self._window_frames < 1:
            raise ValueError("Audio window must contain at least one sample.")
        self._buffer = np.zeros(self._window_frames, dtype=np.float32)
        self._write_position = 0
        self._total_samples = 0
        self._last_window_sample_count = 0
        self._lock = threading.Lock()
        self._updated = threading.Event()
        self._closed = False
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            callback=self._on_audio,
        )
        try:
            self._stream.start()
        except Exception:
            self._stream.close()
            self._stream = None
            raise

    def _on_audio(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            logger.warning("Microphone input stream status: %s", status)
        samples = np.asarray(indata, dtype=np.float32)
        if samples.ndim == 2:
            samples = np.mean(samples, axis=1)
        samples = samples.reshape(-1)
        if not samples.size:
            return
        with self._lock:
            if samples.size >= self._window_frames:
                self._buffer[:] = samples[-self._window_frames :]
                self._write_position = 0
            else:
                end = self._write_position + samples.size
                if end <= self._window_frames:
                    self._buffer[self._write_position : end] = samples
                else:
                    split = self._window_frames - self._write_position
                    self._buffer[self._write_position :] = samples[:split]
                    self._buffer[: end % self._window_frames] = samples[split:]
                self._write_position = end % self._window_frames
            self._total_samples += samples.size
            self._updated.set()

    def read_window(
        self,
        stop_event: threading.Event | None = None,
        *,
        timeout_seconds: float = 0.1,
    ) -> AudioCaptureResult:
        """Wait for a new complete window, returning only its latest samples."""
        while not self._closed:
            if stop_event is not None and stop_event.is_set():
                return AudioCaptureResult(
                    status="missing",
                    message="Monitoring stopped while waiting for a microphone window.",
                    sample_rate=self.sample_rate,
                    channels=self.channels,
                    duration_seconds=self.duration_seconds,
                )
            with self._lock:
                ready = (
                    self._total_samples >= self._window_frames
                    and self._total_samples - self._last_window_sample_count
                    >= self._window_frames
                )
                if ready:
                    start = self._write_position
                    samples = np.concatenate(
                        (self._buffer[start:], self._buffer[:start])
                    ).astype(np.float32, copy=True)
                    self._last_window_sample_count = self._total_samples
                    break
            self._updated.wait(timeout_seconds)
            self._updated.clear()
        else:
            return AudioCaptureResult(
                status="missing",
                message="Microphone stream is closed.",
                sample_rate=self.sample_rate,
                channels=self.channels,
                duration_seconds=self.duration_seconds,
            )

        if not np.all(np.isfinite(samples)):
            return AudioCaptureResult(
                status="unknown",
                message="The captured audio buffer contained non-finite values and was discarded.",
                sample_rate=self.sample_rate,
                channels=self.channels,
                duration_seconds=self.duration_seconds,
            )
        rms_level = float(np.sqrt(np.mean(np.square(samples))))
        peak_level = float(np.max(np.abs(samples)))
        silent = rms_level <= self.silence_threshold
        return AudioCaptureResult(
            status="silent" if silent else "valid",
            message=(
                "The microphone captured a window below the local silence threshold."
                if silent
                else f"Captured a {self.duration_seconds:g}-second in-memory audio window."
            ),
            available=True,
            sample_rate=self.sample_rate,
            channels=self.channels,
            duration_seconds=self.duration_seconds,
            samples=samples,
            rms_level=rms_level,
            peak_level=peak_level,
            quality_flag=not silent,
        )

    def close(self) -> None:
        self._closed = True
        self._updated.set()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()


def check_microphone() -> MicrophoneStatus:
    """Check for an input-capable default device without recording audio."""
    try:
        devices = sd.query_devices(kind="input")
        if isinstance(devices, dict):
            devices = [devices]
        if not devices:
            message = (
                "No microphone input is available. Connect or enable a microphone "
                "and check Windows microphone privacy settings."
            )
            logger.info("No default audio input device is available")
            return MicrophoneStatus(available=False, message=message)
        if not any(
            int(device.get("max_input_channels", 0)) >= 1 for device in devices if isinstance(device, dict)
        ):
            message = (
                "No microphone input is available. Connect or enable a microphone "
                "and check Windows microphone privacy settings."
            )
            logger.info("No default audio input device is available")
            return MicrophoneStatus(available=False, message=message)
        return MicrophoneStatus(available=True, message="Microphone input is available.")
    except Exception as exc:
        logger.exception("Unable to query the default microphone")
        return MicrophoneStatus(
            available=False,
            message=f"Could not access microphone input ({type(exc).__name__}): {exc}",
            error_type=type(exc).__name__,
        )


def capture_audio_window(
    duration_seconds: float = AUDIO_CAPTURE_WINDOW_SECONDS,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int = AUDIO_CHANNELS,
    silence_threshold: float = AUDIO_SILENCE_THRESHOLD,
) -> AudioCaptureResult:
    """Capture a short in-memory audio window and label silent versus unavailable input."""
    if duration_seconds <= 0:
        return AudioCaptureResult(
            status="missing",
            message="Recording duration must be greater than zero.",
            available=False,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=duration_seconds,
        )
    if sample_rate <= 0 or channels <= 0:
        return AudioCaptureResult(
            status="missing",
            message="Recording configuration must use positive sample rate and channel counts.",
            available=False,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=duration_seconds,
        )

    status = check_microphone()
    if not status.available:
        return AudioCaptureResult(
            status="missing",
            message=status.message,
            available=False,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=duration_seconds,
            error_type=status.error_type,
        )

    try:
        frame_count = int(duration_seconds * sample_rate)
        recording = sd.rec(
            frame_count,
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
        )
        sd.wait()
        samples = np.asarray(recording, dtype=np.float32)
        if samples.size == 0:
            message = "The microphone returned an empty recording."
            logger.warning(message)
            return AudioCaptureResult(
                status="missing",
                message=message,
                available=False,
                sample_rate=sample_rate,
                channels=channels,
                duration_seconds=duration_seconds,
            )

        if samples.ndim == 2:
            if samples.shape[1] > 1:
                samples = np.mean(samples, axis=1)
            else:
                samples = samples[:, 0]
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(samples)):
            message = "The captured audio buffer contained non-finite values and was discarded."
            logger.warning(message)
            return AudioCaptureResult(
                status="unknown",
                message=message,
                available=False,
                sample_rate=sample_rate,
                channels=channels,
                duration_seconds=duration_seconds,
                samples=None,
            )

        rms_level = float(np.sqrt(np.mean(np.square(samples))))
        peak_level = float(np.max(np.abs(samples)))
        if rms_level <= silence_threshold:
            return AudioCaptureResult(
                status="silent",
                message=(
                    "The microphone captured a short window but the signal remained below "
                    "the local silence threshold."
                ),
                available=True,
                sample_rate=sample_rate,
                channels=channels,
                duration_seconds=duration_seconds,
                samples=samples,
                rms_level=rms_level,
                peak_level=peak_level,
                quality_flag=False,
            )

        return AudioCaptureResult(
            status="valid",
            message=f"Captured a {duration_seconds:g}-second in-memory audio sample.",
            available=True,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=duration_seconds,
            samples=samples,
            rms_level=rms_level,
            peak_level=peak_level,
            quality_flag=True,
        )
    except Exception as exc:
        logger.exception("Unable to record a microphone test sample")
        return AudioCaptureResult(
            status="missing",
            message=f"Microphone capture failed: {exc}",
            available=False,
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=duration_seconds,
            error_type=type(exc).__name__,
        )


def test_microphone(
    duration_seconds: float = AUDIO_TEST_DURATION_SECONDS,
) -> AudioTestResult:
    """Record a short sample in memory and report its RMS and peak levels."""
    result = capture_audio_window(duration_seconds=duration_seconds)
    if not result.available:
        return AudioTestResult(available=False, message=result.message)
    return AudioTestResult(
        available=result.available,
        message=result.message,
        rms_level=result.rms_level,
        peak_level=result.peak_level,
    )
