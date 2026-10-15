import numpy as np

from branches.audio_branch import AudioFeatureBranch
from microphone import AudioCaptureResult


def test_valid_synthetic_audio_produces_named_finite_features():
    sample_rate = 16_000
    samples = np.arange(1_600, dtype=np.float32)
    signal = (0.2 * np.sin(2 * np.pi * 440 * samples / sample_rate)).astype(
        np.float32
    )
    capture = AudioCaptureResult(
        status="valid",
        message="Synthetic test signal.",
        available=True,
        sample_rate=sample_rate,
        channels=1,
        duration_seconds=0.1,
        samples=signal,
    )
    branch = AudioFeatureBranch(
        sample_rate=sample_rate,
        baseline_window=3,
        baseline_min_samples=2,
    )

    first = branch.analyze(capture)
    second = branch.analyze(capture)

    assert first.status == "warming_up"
    assert first.feature_validity["rms_loudness"] is True
    assert first.feature_values["spectral_centroid_hz"] is not None
    assert second.baseline_ready is True
    assert second.status in {"ok", "partial"}
    assert second.features
    assert all(np.isfinite(value) for value in second.features.values())
