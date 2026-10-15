"""Privacy-minimal helpers for live inference history and CSV export."""

from __future__ import annotations

import csv
import json
import math
from io import StringIO
from typing import Any, Iterable, Mapping


HISTORY_FIELDS = (
    "timestamp",
    "prediction",
    "predicted_class_probability",
    "potential_distress_probability",
    "smoothed_distress_score",
    "alert_state",
    "available_modalities",
    "inference_status",
)

ALERT_DIAGNOSTIC_SCHEMA_VERSION = "live-alert-diagnostic-v3"
ALERT_DIAGNOSTIC_FIELDS = (
    "schema_version",
    "result_key",
    "inference_timestamp_utc",
    "source_timestamp_monotonic",
    "modality_source_timestamps_monotonic",
    "modality_evidence_age_seconds",
    "modality_freshness",
    "evidence_age_seconds",
    "evidence_freshness",
    "evidence_sufficient",
    "evidence_sufficiency_reason",
    "valid_feature_count",
    "available_modalities",
    "modality_availability",
    "inference_status",
    "predicted_label",
    "potential_distress_probability",
    "activation_criterion",
    "activation_qualified",
    "smoothed_score_before",
    "smoothed_score_after",
    "release_counter_before",
    "release_counter_after",
    "raw_probability_threshold",
    "release_threshold",
    "release_requirement",
    "cooldown_seconds",
    "maximum_result_age_seconds",
    "minimum_valid_features",
    "minimum_available_modalities",
    "alert_state_before",
    "alert_state_after",
    "alert_reason_before",
    "alert_reason_after",
    "decision",
    "decision_source",
    "cue_indicators",
    "cue_reason",
)


def create_alert_diagnostic_entry(
    inference: object,
    *,
    result_key: str | float | None,
    inference_timestamp_utc: str | None,
    source_timestamp: float | None,
    modality_source_timestamps: Mapping[str, float | None],
    modality_results_present: Mapping[str, bool],
    observed_at_monotonic: float,
    evidence_sufficient: bool | None,
    evidence_sufficiency_reason: str | None,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    alert_state_after: str,
    alert_reason_after: str,
) -> dict[str, Any]:
    """Build a privacy-minimal, observational record of one alert update."""
    max_age = float(before["max_result_age_seconds"])
    modality_ages: dict[str, float | None] = {}
    modality_freshness: dict[str, str] = {}
    for modality in ("face", "behavior", "audio"):
        timestamp = modality_source_timestamps.get(modality)
        present = modality_results_present.get(modality, False)
        if not present:
            modality_ages[modality] = None
            modality_freshness[modality] = "no_result"
        elif timestamp is None:
            modality_ages[modality] = None
            modality_freshness[modality] = "missing_timestamp"
        else:
            age = float(observed_at_monotonic) - float(timestamp)
            modality_ages[modality] = age
            modality_freshness[modality] = (
                "future_timestamp"
                if age < 0
                else "fresh"
                if age <= max_age
                else "stale"
            )

    if source_timestamp is None:
        evidence_age = None
        evidence_freshness = "missing_timestamp"
    else:
        evidence_age = float(observed_at_monotonic) - float(source_timestamp)
        evidence_freshness = (
            "future_timestamp"
            if evidence_age < 0
            else "fresh"
            if evidence_age <= max_age
            else "stale"
        )

    modalities = getattr(inference, "available_modalities", {}) or {}
    available = tuple(sorted(name for name, value in modalities.items() if value))
    valid_count = int(getattr(inference, "valid_feature_count", 0) or 0)
    status = str(getattr(inference, "status", "unknown"))
    effective_key = result_key if result_key is not None else source_timestamp
    cooldown_active = (
        before.get("state") == "cooldown"
        and before.get("cooldown_until") is not None
        and float(observed_at_monotonic) < float(before["cooldown_until"])
    )

    after_reason = alert_reason_after.lower()
    if cooldown_active:
        decision = "ignored_cooldown"
    elif alert_state_after == "unknown":
        if any(
            marker in after_reason
            for marker in (
                "stale",
                "timestamp",
                "freshness",
                "insufficient",
                "below the alert policy minimum",
                "modality",
                "feature",
            )
        ):
            decision = "rejected_stale_or_insufficient"
        elif "probability is unavailable or invalid" in after_reason:
            decision = "rejected_invalid_score"
        else:
            decision = "rejected_unknown"
    elif effective_key == before.get("last_result_key"):
        decision = "ignored_duplicate"
    else:
        decision = "accepted"

    score = getattr(
        inference,
        "potential_distress_probability",
        getattr(inference, "potential_distress_score", None),
    )
    decision_source = getattr(inference, "decision_source", "model")
    rule_triggered = getattr(inference, "rule_triggered", None)
    is_rule_based = decision_source == "rule_based"
    if is_rule_based:
        activation_qualified = (
            decision == "accepted"
            and status == "potential_distress"
            and rule_triggered is True
        )
        activation_criterion = (
            "status == potential_distress and rule_triggered is true "
            "with fresh, sufficient cue evidence"
        )
    else:
        try:
            activation_qualified = (
                decision == "accepted"
                and status == "potential_distress"
                and score is not None
                and math.isfinite(float(score))
                and float(score) >= float(before["raw_probability_threshold"])
            )
        except (TypeError, ValueError):
            activation_qualified = False
        activation_criterion = (
            "status == potential_distress and raw potential_distress_probability "
            ">= raw_probability_threshold"
        )

    return {
        "schema_version": ALERT_DIAGNOSTIC_SCHEMA_VERSION,
        "result_key": effective_key,
        "inference_timestamp_utc": inference_timestamp_utc,
        "source_timestamp_monotonic": source_timestamp,
        "modality_source_timestamps_monotonic": json.dumps(
            dict(modality_source_timestamps), sort_keys=True
        ),
        "modality_evidence_age_seconds": json.dumps(modality_ages, sort_keys=True),
        "modality_freshness": json.dumps(modality_freshness, sort_keys=True),
        "evidence_age_seconds": evidence_age,
        "evidence_freshness": evidence_freshness,
        "evidence_sufficient": evidence_sufficient,
        "evidence_sufficiency_reason": evidence_sufficiency_reason,
        "valid_feature_count": valid_count,
        "available_modalities": ", ".join(available),
        "modality_availability": json.dumps(
            {name: bool(modalities.get(name, False)) for name in ("face", "behavior", "audio")},
            sort_keys=True,
        ),
        "inference_status": status,
        "predicted_label": getattr(inference, "predicted_label", None),
        "potential_distress_probability": score,
        "activation_criterion": activation_criterion,
        "activation_qualified": activation_qualified,
        "smoothed_score_before": before.get("smoothed_score"),
        "smoothed_score_after": after.get("smoothed_score"),
        "release_counter_before": before.get("release_counter"),
        "release_counter_after": after.get("release_counter"),
        "raw_probability_threshold": (
            None if is_rule_based else before.get("raw_probability_threshold")
        ),
        "release_threshold": None if is_rule_based else before.get("release_threshold"),
        "release_requirement": before.get("release_persistence_count"),
        "cooldown_seconds": before.get("cooldown_seconds"),
        "maximum_result_age_seconds": before.get("max_result_age_seconds"),
        "minimum_valid_features": before.get("minimum_valid_features"),
        "minimum_available_modalities": (
            None
            if is_rule_based
            else before.get("minimum_available_modalities")
        ),
        "alert_state_before": before.get("state"),
        "alert_state_after": alert_state_after,
        "alert_reason_before": before.get("reason"),
        "alert_reason_after": alert_reason_after,
        "decision": decision,
        "decision_source": decision_source,
        "cue_indicators": json.dumps(
            getattr(inference, "cue_indicators", {}) or {}, sort_keys=True
        ),
        "cue_reason": getattr(inference, "cue_reason", None),
    }


def create_history_entry(
    inference: object,
    alert: object,
    *,
    result_key: str | None,
    timestamp: str,
) -> dict[str, Any]:
    modalities = getattr(inference, "available_modalities", {}) or {}
    metadata = getattr(inference, "metadata", {}) or {}
    return {
        "result_key": result_key,
        "timestamp": timestamp,
        "prediction": getattr(inference, "predicted_label", None),
        "predicted_class_probability": getattr(inference, "model_score", None),
        "potential_distress_probability": getattr(
            inference,
            "potential_distress_probability",
            getattr(inference, "potential_distress_score", None),
        ),
        "smoothed_distress_score": getattr(alert, "smoothed_score", None),
        "alert_state": getattr(alert, "state", "unknown"),
        "available_modalities": ", ".join(
            sorted(name for name, available in modalities.items() if available)
        ),
        "inference_status": getattr(inference, "status", "unknown"),
        "detection_mode": (
            metadata.get("detection_mode", "model")
            if isinstance(metadata, Mapping)
            else "model"
        ),
        "cue_indicators": json.dumps(
            getattr(inference, "cue_indicators", {}) or {}, sort_keys=True
        ),
        "cue_reason": getattr(inference, "cue_reason", None),
    }


def inference_history_csv(
    history: Iterable[Mapping[str, Any]],
) -> str:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=HISTORY_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(history)
    return output.getvalue()


def alert_diagnostics_csv(
    diagnostics: Iterable[Mapping[str, Any]],
) -> str:
    """Serialize the versioned alert-decision diagnostics independently of history."""
    output = StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=ALERT_DIAGNOSTIC_FIELDS,
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(diagnostics)
    return output.getvalue()


__all__ = [
    "ALERT_DIAGNOSTIC_FIELDS",
    "ALERT_DIAGNOSTIC_SCHEMA_VERSION",
    "HISTORY_FIELDS",
    "alert_diagnostics_csv",
    "create_alert_diagnostic_entry",
    "create_history_entry",
    "inference_history_csv",
]
