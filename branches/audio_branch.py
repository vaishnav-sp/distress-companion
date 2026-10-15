"""Local microphone-based acoustic feature extraction.

This phase does not classify distress. It only records a short buffer, computes
local signal features, updates a bounded rolling baseline, and returns a
transparent audio contribution summary for later fusion logic.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from branches.interfaces import BranchOutput, BranchProcessor
from config import (
    AUDIO_BASELINE_MIN_SAMPLES,
    AUDIO_BASELINE_WINDOW,
    AUDIO_CHANNELS,
    AUDIO_ONSET_BASELINE_FLOOR,
    AUDIO_ONSET_THRESHOLD,
    AUDIO_SAMPLE_RATE,
    AUDIO_SILENCE_THRESHOLD,
)
from microphone import AudioCaptureResult, capture_audio_window

logger = logging.getLogger(__name__)

FEATURE_NAMES = (
    "rms_loudness",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_flatness",
    "onset_score",
    "baseline_relative_rms",
    "audio_contribution",
)


@dataclass(frozen=True)
class AudioFeatureResult(BranchOutput):
    """Structured audio branch output with explicit validity metadata."""

    status: str
    status_message: str
    quality_flag: bool
    quality_status: str
    rms_loudness: float | None
    rms_valid: bool
    spectral_features: Mapping[str, float | None]
    spectral_validity: Mapping[str, bool]
    onset_score: float | None
    onset_valid: bool
    onset_flag: bool
    baseline_ready: bool
    baseline_sample_count: int
    baseline_relative_rms: float | None
    feature_values: Mapping[str, float | None]
    feature_validity: Mapping[str, bool]
    feature_vector: tuple[float | None, ...]
    feature_names: tuple[str, ...]
    unavailable_reasons: Mapping[str, str]
    audio_contribution: float | None


class AudioFeatureBranch(BranchProcessor):
    """Extract short-window acoustic features with a local rolling baseline."""

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        baseline_window: int = AUDIO_BASELINE_WINDOW,
        baseline_min_samples: int = AUDIO_BASELINE_MIN_SAMPLES,
        silence_threshold: float = AUDIO_SILENCE_THRESHOLD,
        onset_threshold: float = AUDIO_ONSET_THRESHOLD,
        onset_baseline_floor: float = AUDIO_ONSET_BASELINE_FLOOR,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")
        if channels <= 0:
            raise ValueError("At least one channel is required.")
        if baseline_window < 1:
            raise ValueError("Baseline window must be at least 1.")
        if not 1 <= baseline_min_samples <= baseline_window:
            raise ValueError("Baseline minimum must satisfy 1 <= baseline_min <= baseline_window.")
        if silence_threshold < 0:
            raise ValueError("Silence threshold must be non-negative.")
        if onset_threshold < 0:
            raise ValueError("Onset threshold must be non-negative.")
        if onset_baseline_floor <= 0:
            raise ValueError("Onset floor must be positive.")

        self.sample_rate = sample_rate
        self.channels = channels
        self.silence_threshold = silence_threshold
        self.onset_threshold = onset_threshold
        self.onset_baseline_floor = onset_baseline_floor
        self.baseline_window = baseline_window
        self.baseline_min_samples = baseline_min_samples
        self._rms_history: deque[float] = deque(maxlen=baseline_window)
        self._energy_history: deque[float] = deque(maxlen=baseline_window)

    def process(self, input_data: object) -> BranchOutput:
        """Implement the shared future-branch protocol."""
        result = self.analyze(input_data)
        return BranchOutput(
            features=result.features,
            provisional_distress=result.provisional_distress,
            input_quality=result.input_quality,
        )

    def analyze(self, input_data: object) -> AudioFeatureResult:
        """Extract local signal features from a short in-memory audio window."""
        signal = self._coerce_signal(input_data)
        if signal is None:
            return self._missing(
                status="missing",
                message="Expected a non-empty NumPy array or AudioCaptureResult with in-memory samples.",
            )

        finite = np.asarray(signal, dtype=np.float32)
        if finite.size == 0 or not np.all(np.isfinite(finite)):
            return self._missing(
                status="unknown",
                message="The captured audio window was empty or contained non-finite samples.",
            )

        rms_loudness = self._rms_amplitude(finite)
        if not math.isfinite(rms_loudness):
            return self._missing(
                status="unknown",
                message="The local RMS loudness is not finite; feature extraction was not possible.",
            )

        if rms_loudness <= self.silence_threshold:
            return self._silent_window(rms_loudness)

        spectral_values: dict[str, float | None] = {
            "spectral_centroid_hz": None,
            "spectral_bandwidth_hz": None,
            "spectral_flatness": None,
        }
        spectral_validity: dict[str, bool] = {
            name: False for name in spectral_values
        }
        reasons: dict[str, str] = {}

        spectral = self._spectral_features(finite)
        if spectral is not None:
            spectral_values["spectral_centroid_hz"] = spectral["centroid_hz"]
            spectral_values["spectral_bandwidth_hz"] = spectral["bandwidth_hz"]
            spectral_values["spectral_flatness"] = spectral["flatness"]
            for name, value in spectral_values.items():
                spectral_validity[name] = value is not None and math.isfinite(value)
                if not spectral_validity[name]:
                    reasons[name] = "FFT spectral estimate was not reliable for this short window."

        onset_score = self._onset_score(finite)
        onset_valid = onset_score is not None and math.isfinite(onset_score)
        onset_flag = bool(onset_valid and onset_score >= self.onset_threshold)

        self._rms_history.append(float(rms_loudness))
        self._energy_history.append(float(np.mean(np.square(finite))))
        baseline_mean_rms = self._baseline_mean_rms()
        baseline_ready = len(self._rms_history) >= self.baseline_min_samples
        baseline_relative_rms = None
        if baseline_ready and baseline_mean_rms is not None and baseline_mean_rms > 0:
            baseline_relative_rms = (rms_loudness - baseline_mean_rms) / baseline_mean_rms
        elif baseline_ready:
            baseline_relative_rms = 0.0 if rms_loudness == 0.0 else None

        feature_values: dict[str, float | None] = {
            "rms_loudness": float(rms_loudness),
            "spectral_centroid_hz": spectral_values["spectral_centroid_hz"],
            "spectral_bandwidth_hz": spectral_values["spectral_bandwidth_hz"],
            "spectral_flatness": spectral_values["spectral_flatness"],
            "onset_score": onset_score,
            "baseline_relative_rms": baseline_relative_rms,
        }
        feature_validity: dict[str, bool] = {
            "rms_loudness": True,
            "spectral_centroid_hz": spectral_validity["spectral_centroid_hz"],
            "spectral_bandwidth_hz": spectral_validity["spectral_bandwidth_hz"],
            "spectral_flatness": spectral_validity["spectral_flatness"],
            "onset_score": onset_valid,
            "baseline_relative_rms": baseline_relative_rms is not None,
        }

        if onset_score is not None and not math.isfinite(onset_score):
            reasons["onset_score"] = "The onset score was non-finite and therefore invalid."
            feature_validity["onset_score"] = False
            feature_values["onset_score"] = None
        if baseline_relative_rms is None and baseline_ready:
            reasons["baseline_relative_rms"] = (
                "Baseline comparison is unavailable because the recent local baseline is zero or unstable."
            )

        audio_contribution = self._audio_contribution(rms_loudness, onset_score)
        feature_values["audio_contribution"] = audio_contribution
        feature_validity["audio_contribution"] = (
            audio_contribution is not None and math.isfinite(audio_contribution)
        )
        if not feature_validity["audio_contribution"]:
            reasons["audio_contribution"] = "Audio contribution could not be computed reliably."

        valid_count = sum(1 for ok in feature_validity.values() if ok)
        if not baseline_ready:
            status = "warming_up"
            status_message = "Valid audio is available but the local rolling baseline is still warming up."
            quality_status = "warming_up"
            quality_flag = True
        elif valid_count == 0:
            status = "unknown"
            status_message = "The audio window was valid but no stable features could be computed."
            quality_status = "unknown"
            quality_flag = False
        elif valid_count < len(feature_values):
            status = "partial"
            status_message = "Some acoustic features are unavailable; the rest remain valid."
            quality_status = "partial"
            quality_flag = True
        else:
            status = "ok"
            status_message = "Audio features are available and are not a distress classification."
            quality_status = "usable"
            quality_flag = True

        available_features = {
            name: float(value)
            for name, value in feature_values.items()
            if feature_validity.get(name, False) and value is not None and math.isfinite(float(value))
        }
        return AudioFeatureResult(
            features=available_features,
            provisional_distress=None,
            input_quality=quality_flag,
            status=status,
            status_message=status_message,
            quality_flag=quality_flag,
            quality_status=quality_status,
            rms_loudness=float(rms_loudness),
            rms_valid=True,
            spectral_features=spectral_values,
            spectral_validity=spectral_validity,
            onset_score=onset_score,
            onset_valid=onset_valid,
            onset_flag=onset_flag,
            baseline_ready=baseline_ready,
            baseline_sample_count=len(self._rms_history),
            baseline_relative_rms=baseline_relative_rms,
            feature_values=feature_values,
            feature_validity=feature_validity,
            feature_vector=tuple(feature_values[name] for name in FEATURE_NAMES),
            feature_names=FEATURE_NAMES,
            unavailable_reasons=reasons,
            audio_contribution=audio_contribution,
        )

    def reset(self) -> None:
        """Clear the rolling baseline while keeping the branch ready for future captures."""
        self._rms_history.clear()
        self._energy_history.clear()

    @staticmethod
    def _rms_amplitude(signal: np.ndarray) -> float:
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        return float(np.sqrt(np.mean(np.square(signal))))

    def _spectral_features(self, signal: np.ndarray) -> dict[str, float] | None:
        if signal.size < 2:
            return None
        frame = np.asarray(signal, dtype=np.float32)
        window = np.hanning(frame.size)
        windowed = frame * window
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        if np.sum(magnitude) <= 0.0:
            return None

        freqs = np.fft.rfftfreq(frame.size, d=1.0 / float(self.sample_rate))
        centroid = float(np.sum(freqs * magnitude) / np.sum(magnitude))
        if not math.isfinite(centroid):
            return None
        bandwidth = float(
            np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude))
        )
        mean_magnitude = float(np.mean(magnitude))
        if mean_magnitude <= 0.0:
            flatness = 0.0
        else:
            flatness = float(np.exp(np.mean(np.log(magnitude + 1e-12))) / mean_magnitude)
        if not np.isfinite(flatness):
            flatness = 0.0
        return {
            "centroid_hz": centroid,
            "bandwidth_hz": bandwidth,
            "flatness": flatness,
        }

    def _baseline_mean_rms(self) -> float | None:
        if not self._rms_history:
            return None
        return float(np.mean(np.asarray(list(self._rms_history), dtype=np.float32)))

    def _onset_score(self, signal: np.ndarray) -> float | None:
        energy = float(np.mean(np.square(signal)))
        if not math.isfinite(energy):
            return None
        if not self._energy_history:
            return None
        previous = float(np.mean(np.asarray(list(self._energy_history), dtype=np.float32)))
        if previous <= self.onset_baseline_floor:
            return None
        score = (energy - previous) / previous
        return float(score) if math.isfinite(score) else None

    def _audio_contribution(self, rms_loudness: float, onset_score: float | None) -> float | None:
        loudness_term = np.clip(rms_loudness / max(rms_loudness + 0.05, 1e-6), 0.0, 1.0)
        if onset_score is not None:
            onset_term = np.clip(onset_score / max(self.onset_threshold * 2.0, 1e-6), 0.0, 1.0)
            contribution = float(np.clip(0.65 * loudness_term + 0.35 * onset_term, 0.0, 1.0))
            return contribution
        return float(np.clip(loudness_term, 0.0, 1.0))

    def _coerce_signal(self, input_data: object) -> np.ndarray | None:
        if input_data is None:
            return None
        if isinstance(input_data, AudioCaptureResult):
            samples = input_data.samples
            if samples is None:
                return None
            return np.asarray(samples, dtype=np.float32).reshape(-1)
        if isinstance(input_data, np.ndarray):
            signal = input_data
        else:
            try:
                signal = np.asarray(input_data, dtype=np.float32)
            except (TypeError, ValueError):
                return None
        if signal.size == 0:
            return None
        if signal.ndim == 2:
            if signal.shape[1] > 1:
                signal = np.mean(signal, axis=1)
            else:
                signal = signal[:, 0]
        return np.asarray(signal, dtype=np.float32).reshape(-1)

    def _silent_window(self, rms_loudness: float) -> AudioFeatureResult:
        feature_values = {
            "rms_loudness": float(rms_loudness),
            "spectral_centroid_hz": None,
            "spectral_bandwidth_hz": None,
            "spectral_flatness": None,
            "onset_score": None,
            "baseline_relative_rms": None,
            "audio_contribution": 0.0,
        }
        feature_validity = {
            "rms_loudness": True,
            "spectral_centroid_hz": False,
            "spectral_bandwidth_hz": False,
            "spectral_flatness": False,
            "onset_score": False,
            "baseline_relative_rms": False,
            "audio_contribution": True,
        }
        feature_names = FEATURE_NAMES
        return AudioFeatureResult(
            features={"rms_loudness": float(rms_loudness), "audio_contribution": 0.0},
            provisional_distress=None,
            input_quality=False,
            status="silent",
            status_message=(
                "The recorded microphone window is below the local silence threshold and is treated as silent."
            ),
            quality_flag=False,
            quality_status="silent",
            rms_loudness=float(rms_loudness),
            rms_valid=True,
            spectral_features={
                "spectral_centroid_hz": None,
                "spectral_bandwidth_hz": None,
                "spectral_flatness": None,
            },
            spectral_validity={
                "spectral_centroid_hz": False,
                "spectral_bandwidth_hz": False,
                "spectral_flatness": False,
            },
            onset_score=None,
            onset_valid=False,
            onset_flag=False,
            baseline_ready=len(self._rms_history) >= self.baseline_min_samples,
            baseline_sample_count=len(self._rms_history),
            baseline_relative_rms=None,
            feature_values=feature_values,
            feature_validity=feature_validity,
            feature_vector=tuple(feature_values[name] for name in feature_names),
            feature_names=feature_names,
            unavailable_reasons={
                "spectral_centroid_hz": "Silent input does not support a reliable centroid estimate.",
                "spectral_bandwidth_hz": "Silent input does not support a reliable bandwidth estimate.",
                "spectral_flatness": "Silent input does not support a reliable flatness estimate.",
                "onset_score": "Silent input does not support a reliable onset estimate.",
                "baseline_relative_rms": "Silent input is not used to update the rolling baseline.",
            },
            audio_contribution=0.0,
        )

    def _missing(self, status: str, message: str) -> AudioFeatureResult:
        missing_feature_values = {name: None for name in FEATURE_NAMES}
        missing_feature_values["audio_contribution"] = 0.0
        missing_validity = {name: False for name in FEATURE_NAMES}
        missing_validity["audio_contribution"] = True
        return AudioFeatureResult(
            features={},
            provisional_distress=None,
            input_quality=False,
            status=status,
            status_message=message,
            quality_flag=False,
            quality_status="missing" if status == "missing" else "unknown",
            rms_loudness=None,
            rms_valid=False,
            spectral_features={
                "spectral_centroid_hz": None,
                "spectral_bandwidth_hz": None,
                "spectral_flatness": None,
            },
            spectral_validity={
                "spectral_centroid_hz": False,
                "spectral_bandwidth_hz": False,
                "spectral_flatness": False,
            },
            onset_score=None,
            onset_valid=False,
            onset_flag=False,
            baseline_ready=False,
            baseline_sample_count=len(self._rms_history),
            baseline_relative_rms=None,
            feature_values=missing_feature_values,
            feature_validity=missing_validity,
            feature_vector=tuple(missing_feature_values[name] for name in FEATURE_NAMES),
            feature_names=FEATURE_NAMES,
            unavailable_reasons={
                name: "No valid measurement was available for this feature." for name in FEATURE_NAMES
            },
            audio_contribution=0.0,
        )


AudioResult = AudioFeatureResult
AudioBranch = AudioFeatureBranch
