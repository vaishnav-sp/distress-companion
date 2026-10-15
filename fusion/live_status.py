"""Presentation helpers for current, freshness-gated fusion inference."""

from __future__ import annotations

from typing import Mapping


MODALITIES = ("face", "behavior", "audio")


def live_fusion_status(
    monitoring_status: str,
    *,
    model_available: bool,
    evidence_sufficient: bool,
    inference_reason: str | None = None,
    model_error: str | None = None,
) -> tuple[str, str]:
    """Return an explicit display state without changing inference or alert policy."""
    if monitoring_status == "stopped":
        return "STOPPED", "Monitoring is stopped; live inference is not running."
    if monitoring_status == "error":
        return "UNAVAILABLE", model_error or "The monitoring worker exited with an error."
    if monitoring_status != "running":
        return "WAITING", f"Monitoring is {monitoring_status}; no live prediction is running."
    if not model_available:
        return "UNAVAILABLE", model_error or "No trained local model is available."
    if not evidence_sufficient:
        return (
            "WAITING FOR SUFFICIENT FRESH EVIDENCE",
            inference_reason or "Fresh evidence does not meet the inference requirements.",
        )
    if inference_reason:
        return "UNAVAILABLE", inference_reason
    return "RUNNING", "Prediction produced from sufficient fresh evidence."


def live_modality_summary(
    monitoring_status: str,
    *,
    results: Mapping[str, object | None],
    freshness: Mapping[str, bool],
    available_modalities: Mapping[str, bool],
) -> dict[str, str]:
    """Describe which modalities are currently contributing fresh valid features."""
    if monitoring_status == "stopped":
        return {name: "cleared when monitoring stopped" for name in MODALITIES}
    if monitoring_status != "running":
        return {name: "not evaluated" for name in MODALITIES}

    summary: dict[str, str] = {}
    for name in MODALITIES:
        if results.get(name) is None:
            summary[name] = "missing"
        elif not freshness.get(name, False):
            summary[name] = "stale; excluded from live inference"
        elif available_modalities.get(name, False):
            summary[name] = "valid fresh evidence used"
        else:
            summary[name] = "result present; no valid features"
    return summary


def live_sensor_status(
    sensor_status: str,
    last_success_timestamp: float | None,
    monitoring_status: str,
    *,
    now: float,
    stale_after_seconds: float = 3.0,
) -> str:
    """Map operational sensor diagnostics to stable RUNNING/UNAVAILABLE/STALE/STOPPED states."""
    if monitoring_status in {"stopped", "stopping"} or sensor_status == "stopped":
        return "STOPPED"
    if monitoring_status == "error" or sensor_status in {"unavailable", "error"}:
        return "UNAVAILABLE"
    if (
        last_success_timestamp is not None
        and now - float(last_success_timestamp) > stale_after_seconds
    ):
        return "STALE"
    return "RUNNING"


__all__ = ["live_fusion_status", "live_modality_summary", "live_sensor_status"]
