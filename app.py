"""Local dashboard for device checks and optional face-expression inference."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING
import uuid

import cv2
import streamlit as st

from alert_management import AlertManager, AlertPolicy
from alert_notifications import score_popup_transition
from config import (
    BEHAVIOR_MOVEMENT_MAX_GAP_SECONDS,
    BEHAVIOR_MOVEMENT_WINDOW_SECONDS,
    EXPERIMENTAL_DISTRESS_SCORE_FACE_MAX_ENTROPY,
    EXPERIMENTAL_DISTRESS_SCORE_FRESHNESS_SECONDS,
    EXPERIMENTAL_DISTRESS_SCORE_MIN_MODALITIES,
    EXPERIMENTAL_DISTRESS_SCORE_WEIGHTS,
    EXPERIMENTAL_DISTRESS_ALERT_THRESHOLD,
    LIVE_AUDIO_INTERVAL_SECONDS,
    LIVE_CAMERA_INTERVAL_SECONDS,
    LIVE_DASHBOARD_REFRESH_SECONDS,
    RULE_AUDIO_ONSET_THRESHOLD,
    RULE_AUDIO_PEAK_MIN_EVENTS,
    RULE_AUDIO_PEAK_RMS_THRESHOLD,
    RULE_AUDIO_PEAK_WINDOW_SECONDS,
    RULE_AUDIO_SUSTAINED_RMS_THRESHOLD,
    RULE_AUDIO_SUSTAINED_SECONDS,
    RULE_COMBINATION_MODE,
    RULE_MOVEMENT_REPETITION_MIN_AMPLITUDE,
    RULE_MOVEMENT_REPETITION_MIN_ELEVATED_PROPORTION,
    RULE_MOVEMENT_REPETITION_MIN_RATE_HZ,
    RULE_MOVEMENT_REPETITION_SECONDS,
    RULE_MOVEMENT_REGIONS,
    RULE_MOVEMENT_SUDDEN_THRESHOLD,
    RULE_MOVEMENT_VIGOROUS_MIN_ACCELERATION,
    RULE_MOVEMENT_VIGOROUS_MIN_ENERGY,
    RULE_MOVEMENT_VIGOROUS_MIN_PEAK_SPEED,
    RULE_MOVEMENT_VIGOROUS_SECONDS,
)
from feedback import FeedbackStorageError, FeedbackStore
from feedback.presentation import (
    alert_feedback_context_for_snapshot,
    record_for_score_snapshot,
    retained_alert_feedback_context,
    submit_alert_feedback,
)
from training.personalization_demo import (
    PersonalizationResult,
    run_reviewed_personalization,
    run_synthetic_personalization_demo,
    reviewed_training_block_reason,
)
from training.reviewed_feedback import (
    feedback_review_block_reason,
    feedback_review_provenance_diagnostic,
    feedback_review_statuses,
    load_reviewed_feedback,
)
from training.demo_feedback_reviews import (
    DEMO_TRAINING_LABELS,
    append_demo_feedback_review,
    load_demo_feedback_reviews,
    load_demo_training_dataset,
)
from distress_score import calculate_distress_score, score_alert_triggered
from fusion import (
    build_fusion_diagnostic_snapshot,
    build_fusion_evidence,
    evidence_to_feature_record,
)
from fusion.live_status import (
    live_fusion_status,
    live_modality_summary,
    live_sensor_status,
)
from microphone import check_microphone, test_microphone
from monitoring_history import alert_diagnostics_csv, inference_history_csv
from simulated_alert_test import (
    SimulatedDemoAlertManager,
)
from rule_based_detection import RuleBasedInferenceProcessor, RuleDetectionConfig
from webcam import capture_frame, capture_frame_sequence

if TYPE_CHECKING:
    from branches.behavior_branch import BehaviorResult, UpperBodyBehaviorBranch
    from branches.face_branch import FaceExpressionBranch, FaceResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
REVIEW_MANIFEST_PATH = Path(
    os.environ.get(
        "DISTRESS_REVIEW_MANIFEST",
        str(Path.home() / ".distress_companion" / "review_manifest_v2.jsonl"),
    )
).expanduser()
DEMO_REVIEW_PATH = Path(
    os.environ.get(
        "DISTRESS_DEMO_REVIEW_FILE",
        str(Path.home() / ".distress_companion" / "demo_feedback_reviews.jsonl"),
    )
).expanduser()
PERSONALIZATION_DIRECTORY = Path(
    os.environ.get(
        "DISTRESS_PERSONALIZATION_DIR",
        str(Path.home() / ".distress_companion" / "personalized_models"),
    )
).expanduser()


def get_face_branch() -> FaceExpressionBranch:
    from branches.face_branch import FaceExpressionBranch

    return FaceExpressionBranch()


def get_behavior_branch() -> UpperBodyBehaviorBranch:
    from branches.behavior_branch import UpperBodyBehaviorBranch

    return UpperBodyBehaviorBranch()


def get_audio_branch() -> "AudioFeatureBranch":
    from branches.audio_branch import AudioFeatureBranch

    return AudioFeatureBranch()


def create_rule_detection_config() -> RuleDetectionConfig:
    return RuleDetectionConfig(
        combination_mode=RULE_COMBINATION_MODE,
        cue_window_seconds=BEHAVIOR_MOVEMENT_WINDOW_SECONDS,
        max_sample_gap_seconds=BEHAVIOR_MOVEMENT_MAX_GAP_SECONDS,
        audio_sustained_rms_threshold=RULE_AUDIO_SUSTAINED_RMS_THRESHOLD,
        audio_sustained_seconds=RULE_AUDIO_SUSTAINED_SECONDS,
        audio_peak_rms_threshold=RULE_AUDIO_PEAK_RMS_THRESHOLD,
        audio_onset_threshold=RULE_AUDIO_ONSET_THRESHOLD,
        audio_peak_window_seconds=RULE_AUDIO_PEAK_WINDOW_SECONDS,
        audio_peak_min_events=RULE_AUDIO_PEAK_MIN_EVENTS,
        movement_repetition_min_rate_hz=RULE_MOVEMENT_REPETITION_MIN_RATE_HZ,
        movement_repetition_min_amplitude=RULE_MOVEMENT_REPETITION_MIN_AMPLITUDE,
        movement_repetition_min_elevated_proportion=(
            RULE_MOVEMENT_REPETITION_MIN_ELEVATED_PROPORTION
        ),
        movement_repetition_seconds=RULE_MOVEMENT_REPETITION_SECONDS,
        movement_vigorous_min_energy=RULE_MOVEMENT_VIGOROUS_MIN_ENERGY,
        movement_vigorous_min_peak_speed=RULE_MOVEMENT_VIGOROUS_MIN_PEAK_SPEED,
        movement_vigorous_min_acceleration=RULE_MOVEMENT_VIGOROUS_MIN_ACCELERATION,
        movement_vigorous_seconds=RULE_MOVEMENT_VIGOROUS_SECONDS,
        movement_sudden_threshold=RULE_MOVEMENT_SUDDEN_THRESHOLD,
        movement_regions=RULE_MOVEMENT_REGIONS,
    )


def process_monitoring_inference(snapshot):
    """Compatibility entry point for the stateful, classifier-free rule processor."""
    global _standalone_rule_processor
    if _standalone_rule_processor is None:
        _standalone_rule_processor = RuleBasedInferenceProcessor(
            create_rule_detection_config()
        )
    return _standalone_rule_processor(snapshot)


_standalone_rule_processor: RuleBasedInferenceProcessor | None = None


def create_monitoring_manager(alert_manager=None):
    from monitoring import MonitoringManager

    return MonitoringManager(
        camera_interval_seconds=LIVE_CAMERA_INTERVAL_SECONDS,
        audio_interval_seconds=LIVE_AUDIO_INTERVAL_SECONDS,
        face_branch_factory=get_face_branch,
        behavior_branch_factory=get_behavior_branch,
        audio_branch_factory=get_audio_branch,
        inference_processor=RuleBasedInferenceProcessor(create_rule_detection_config()),
        alert_manager=alert_manager,
    )


def _render_monitoring_result_block(title: str, result: object, waiting_message: str) -> None:
    if result is None:
        st.info(f"{title}: {waiting_message}")
        return

    status = getattr(result, "status", "unknown")
    status_message = getattr(result, "status_message", "") or "No status message available."
    if status == "ok":
        st.success(f"{title}: {status_message}")
    elif status in {"warming_up", "partial", "silent"}:
        st.warning(f"{title}: {status_message}")
    else:
        st.error(f"{title}: {status_message}")

    if hasattr(result, "feature_validity"):
        rows = [
            {
                "Feature": name,
                "Value": (
                    f"{result.feature_values[name]:.6f}"
                    if result.feature_validity.get(name, False)
                    else "Unavailable"
                ),
                "Valid": "Yes" if result.feature_validity.get(name, False) else "No",
                "Reason": result.unavailable_reasons.get(name, ""),
            }
            for name in getattr(result, "feature_names", ())
        ]
        if rows:
            st.table(rows)
    elif hasattr(result, "expression_probabilities"):
        rows = [
            {
                "Class": label,
                "Probability": (
                    f"{result.expression_probabilities[label]:.4f}"
                    if result.expression_probabilities.get(label) is not None
                    else "Unavailable"
                ),
                "Valid estimate": "Yes" if result.class_validity.get(label, False) else "No",
                "Supported by model": "Yes" if result.class_availability.get(label, False) else "No",
            }
            for label in result.feature_vector_order
        ]
        if rows:
            st.table(rows)

    if hasattr(result, "baseline_ready"):
        st.write(
            f"Baseline ready: {'Yes' if result.baseline_ready else 'No'} "
            f"({getattr(result, 'baseline_sample_count', 0)} samples)"
        )
    if hasattr(result, "quality_score"):
        st.write(f"Quality score: {result.quality_score:.2%}")
    if hasattr(result, "quality_flag"):
        st.write(f"Quality flag: {'Yes' if result.quality_flag else 'No'}")
    if hasattr(result, "quality_status"):
        st.write(f"Quality status: {result.quality_status}")


def _render_current_distress_status(
    *,
    status: str,
    message: str,
    inference: object | None,
    alert: object | None,
    rule_based: bool,
    inference_timestamp: object | None,
) -> None:
    if alert is not None and getattr(alert, "state", None) == "potential_distress":
        label = "Potential Distress"
        sublabel = "Alert active"
        tone = "warning"
        narrative = getattr(alert, "reason", None) or "A configured distress cue remains active."
        evidence_time = getattr(alert, "activated_at_utc", None) or inference_timestamp
    elif status == "stopped":
        label = "Monitoring Stopped"
        sublabel = "No live inference is running"
        tone = "secondary"
        narrative = "Monitoring is stopped; no fresh evidence is being produced."
        evidence_time = None
    elif inference is not None and getattr(inference, "status", "") == "potential_distress":
        label = "Potential Distress"
        sublabel = "Detector result"
        tone = "warning"
        narrative = getattr(inference, "reason", None) or "A configured cue has been triggered."
        evidence_time = inference_timestamp
    elif inference is not None and getattr(inference, "status", "") == "no_sufficient_distress_evidence":
        label = "Monitoring — No Alert"
        sublabel = "Sufficient evidence not met"
        tone = "info"
        narrative = getattr(inference, "reason", None) or "The latest evidence does not meet the configured trigger criteria."
        evidence_time = inference_timestamp
    elif inference is not None and getattr(inference, "status", "") == "unknown":
        label = "Insufficient Evidence"
        sublabel = "Unknown or unavailable"
        tone = "secondary"
        narrative = getattr(inference, "reason", None) or "Evidence is missing, stale, or insufficient to evaluate the selected mode."
        evidence_time = inference_timestamp
    else:
        label = "Monitoring — No Alert"
        sublabel = "Monitoring is healthy"
        tone = "success"
        narrative = message or "No configured distress cue is active right now."
        evidence_time = inference_timestamp

    if tone == "warning":
        accent = "#f0a500"
        bg = "rgba(240,165,0,0.12)"
        border = "rgba(240,165,0,0.5)"
    elif tone == "success":
        accent = "#1f9d55"
        bg = "rgba(31,157,85,0.10)"
        border = "rgba(31,157,85,0.4)"
    elif tone == "secondary":
        accent = "#6c757d"
        bg = "rgba(108,117,125,0.10)"
        border = "rgba(108,117,125,0.4)"
    else:
        accent = "#3b82f6"
        bg = "rgba(59,130,246,0.10)"
        border = "rgba(59,130,246,0.4)"

    if rule_based:
        score_block = "Rule-based detector result; no calibrated probability is produced."
    elif getattr(inference, "potential_distress_score", None) is not None:
        score_block = f"Potential-distress score: {inference.potential_distress_score:.3f}"
    elif getattr(alert, "smoothed_score", None) is not None:
        score_block = f"Smoothed distress score: {alert.smoothed_score:.3f}"
    else:
        score_block = "No calibrated distress score is available for this detector."

    st.markdown(
        f"<div style='padding:1rem 1.25rem; border:1px solid {border}; border-radius:12px; background:{bg}; margin-bottom:1rem;'>"
        f"<div style='font-size:0.82rem; text-transform:uppercase; letter-spacing:0.08em; color:{accent}; font-weight:700;'>Current Distress Status</div>"
        f"<div style='font-size:2rem; font-weight:700; margin:0.35rem 0; color:{accent};'>{label}</div>"
        f"<div style='font-size:0.95rem; color:var(--text-color); opacity:0.92; margin-bottom:0.35rem;'>{sublabel}</div>"
        f"<div style='font-size:0.9rem; color:var(--text-color); opacity:0.82; margin-bottom:0.2rem;'>{narrative}</div>"
        f"<div style='display:flex; gap:1rem; flex-wrap:wrap; margin-top:0.5rem; font-size:0.82rem; color:var(--text-color); opacity:0.78;'>"
        f"<span><strong>Evidence:</strong> {score_block}</span>"
        f"<span><strong>Updated:</strong> {evidence_time if evidence_time is not None else 'not available'}</span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Autism Distress Companion", page_icon="🧩")
st.markdown(
    """
    <style>
    :root {
        --dashboard-accent: #148b83;
        --dashboard-border: rgba(128, 145, 165, 0.24);
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1500px;
        padding-top: 1.35rem;
        padding-bottom: 2rem;
    }

    [data-testid="stAppViewContainer"] .main h1 {
        font-size: clamp(1.8rem, 2.5vw, 2.25rem);
        line-height: 1.2;
        letter-spacing: -0.025em;
        margin-bottom: 0.3rem;
    }

    [data-testid="stAppViewContainer"] .main h2 {
        font-size: 1.38rem;
        line-height: 1.3;
        letter-spacing: -0.012em;
        margin-top: 0.35rem;
        margin-bottom: 0.3rem;
    }

    [data-testid="stAppViewContainer"] .main h3 {
        font-size: 1.08rem;
        line-height: 1.35;
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }

    [data-testid="stAppViewContainer"] .main [data-testid="stVerticalBlock"] {
        gap: 0.8rem;
    }

    [data-testid="stAppViewContainer"] .main [data-testid="stMetric"] {
        padding: 0.7rem 0.85rem;
        border: 1px solid var(--dashboard-border);
        border-radius: 0.7rem;
        background: var(--secondary-background-color);
    }

    [data-testid="stAppViewContainer"] .main [data-testid="stMetricLabel"] {
        font-size: 0.82rem;
    }

    [data-testid="stAppViewContainer"] .main [data-testid="stMetricValue"] {
        font-size: 1.45rem;
        line-height: 1.25;
    }

    [data-testid="stAppViewContainer"] .main [data-testid="stTable"] {
        border: 1px solid var(--dashboard-border);
        border-radius: 0.65rem;
        overflow: hidden;
    }

    [data-testid="stAppViewContainer"] .main [data-testid="stTable"] th,
    [data-testid="stAppViewContainer"] .main [data-testid="stTable"] td {
        padding: 0.42rem 0.62rem;
        font-size: 0.88rem;
        line-height: 1.35;
    }

    [data-testid="stAppViewContainer"] .main [data-testid="stAlert"] {
        border-radius: 0.65rem;
    }

    [data-testid="stAppViewContainer"] .main [data-testid="stButton"] button,
    [data-testid="stAppViewContainer"] .main [data-testid="stFormSubmitButton"] button {
        min-height: 2.55rem;
        border-radius: 0.55rem;
        font-weight: 600;
        transition: filter 120ms ease, border-color 120ms ease;
    }

    [data-testid="stAppViewContainer"] .main [data-testid="stButton"] button:hover,
    [data-testid="stAppViewContainer"] .main [data-testid="stFormSubmitButton"] button:hover {
        border-color: var(--dashboard-accent);
        filter: brightness(1.04);
    }

    [data-testid="stAppViewContainer"] .main button[kind="primary"] {
        background: var(--dashboard-accent);
        border-color: var(--dashboard-accent);
    }

    [data-testid="stAppViewContainer"] .main [data-testid="stHorizontalBlock"] {
        gap: 0.9rem;
    }

    [data-testid="stAppViewContainer"] .main hr {
        margin: 0.5rem 0 0.8rem;
        border-color: var(--dashboard-border);
    }

    @media (max-width: 700px) {
        [data-testid="stAppViewContainer"] .main .block-container {
            padding: 1rem 1rem 1.5rem;
        }

        [data-testid="stAppViewContainer"] .main h2 {
            font-size: 1.22rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.dialog("New alert")
def show_alert_popup(message: str, alert_id: str) -> None:
    st.warning(message)
    feedback_context = st.session_state.get("score_popup_feedback_context")
    context_matches = (
        feedback_context is not None
        and feedback_context.alert_id == alert_id
    )
    snapshot_available = (
        context_matches and feedback_context.record is not None
    )
    if not snapshot_available:
        st.info(
            "The feature snapshot associated with this alert is unavailable; "
            "feedback buttons are disabled."
        )

    submitted_result = st.session_state.get("score_popup_feedback_result") or {}
    already_submitted = (
        submitted_result.get("alert_id") == alert_id
        and submitted_result.get("submitted") is True
    )
    feedback_columns = st.columns(2)
    for column, action, label in (
        (feedback_columns[0], "confirm_alert", "Confirm"),
        (feedback_columns[1], "reject_alert", "Reject"),
    ):
        with column:
            if st.button(
                label,
                key=f"score_popup_{action}_{alert_id}",
                disabled=not snapshot_available or already_submitted,
            ):
                try:
                    result = submit_alert_feedback(
                        st.session_state.feedback_store,
                        feedback_context,
                        action,
                    )
                    st.session_state.score_popup_feedback_result = {
                        "alert_id": alert_id,
                        "submitted": True,
                        "message": (
                            "Feedback saved locally."
                            if result.inserted
                            else "Feedback for this alert was already saved."
                        ),
                        "is_error": False,
                    }
                    st.rerun()
                except (FeedbackStorageError, ValueError) as exc:
                    st.session_state.score_popup_feedback_result = {
                        "alert_id": alert_id,
                        "submitted": False,
                        "message": f"Feedback storage failed: {exc}",
                        "is_error": True,
                    }

    submitted_result = st.session_state.get("score_popup_feedback_result") or {}
    if submitted_result.get("alert_id") == alert_id:
        if submitted_result.get("is_error"):
            st.error(submitted_result["message"])
        elif submitted_result.get("submitted"):
            st.success(submitted_result["message"])


def _render_feedback_history_and_personalization() -> None:
    st.header("Feedback History and Review")
    review_notice = st.session_state.pop("review_notice", None)
    review_notice_kind = st.session_state.pop("review_notice_kind", "success")
    if review_notice:
        if review_notice_kind == "warning":
            st.warning(review_notice)
        elif review_notice_kind == "error":
            st.error(review_notice)
        else:
            st.success(review_notice)
    st.caption(
        "Feedback records are read-only here. Confirm, Reject, and Report Missed "
        "Event are caregiver actions, not verified labels. Raw feature vectors and "
        "private identifiers are not shown."
    )
    st.caption(
        "Demonstration labels are manually assigned, locally stored, and not "
        "independently verified ground truth. They do not enter the v2 training loader."
    )

    feedback_store = st.session_state.feedback_store
    try:
        feedback_rows = feedback_store.list_feedback()
        review_statuses = feedback_review_statuses(
            feedback_store, REVIEW_MANIFEST_PATH
        )
        demo_reviews = load_demo_feedback_reviews(DEMO_REVIEW_PATH)
        demo_training_dataset = load_demo_training_dataset(
            feedback_store,
            DEMO_REVIEW_PATH,
            feedback_rows=feedback_rows,
            reviews=demo_reviews,
        )
    except FeedbackStorageError as exc:
        st.error(f"Could not read local feedback history: {exc}")
        return
    except (OSError, ValueError) as exc:
        st.error(f"Could not read local review status: {exc}")
        return

    table_rows: list[dict[str, str]] = []
    selector_labels: dict[str, str] = {}
    demo_examples_by_id = {
        example.feedback_id: example for example in demo_training_dataset.examples
    }
    history_rows = sorted(
        feedback_rows,
        key=lambda row: str(row.get("submitted_at_utc") or ""),
        reverse=True,
    )
    reviewability: dict[str, tuple[bool, str | None]] = {}
    for row in history_rows:
        feedback_id = row.get("feedback_id")
        if not isinstance(feedback_id, str):
            continue
        status = review_statuses.get(feedback_id)
        if status is None:
            continue
        block_reason = feedback_review_block_reason(row, feedback_id)
        reviewability[feedback_id] = (block_reason is None, block_reason)
        demo_review = demo_reviews.get(feedback_id)
        demo_ineligible_reason = demo_training_dataset.ineligible_reasons.get(
            feedback_id
        )
        if status.status == "Excluded":
            displayed_review_status = "Excluded"
        elif demo_review is not None or status.status != "Awaiting review":
            displayed_review_status = "Reviewed"
        else:
            displayed_review_status = "Awaiting review"
        action = str(row.get("action") or "unknown")
        submitted_at = str(row.get("submitted_at_utc") or "Unavailable")
        try:
            summary = json.loads(row.get("feature_summary_json") or "{}")
        except json.JSONDecodeError:
            summary = {}
        provenance = summary.get("provenance", {}) if isinstance(summary, dict) else {}
        if not isinstance(provenance, dict):
            provenance = {}
        modality_times = provenance.get("source_timestamps_utc", {})
        modality_names = (
            ", ".join(sorted(modality_times))
            if isinstance(modality_times, dict) and modality_times
            else "Unavailable"
        )
        safe_summary = (
            f"Schema {summary.get('schema_version', 'unavailable')}; "
            f"record {summary.get('record_kind', 'unavailable')}; "
            f"snapshot provenance "
            f"{'available' if provenance.get('provenance_available') is True else 'unavailable'}; "
            f"modalities: {modality_names}"
        )
        table_rows.append(
            {
                "Submitted (UTC)": submitted_at,
                "Action": action,
                "Review status": displayed_review_status,
                "Training eligibility": (
                    "Eligible (v2 reviewed data)"
                    if status.training_eligibility == "Eligible"
                    else "Eligible for demo training only (manual label; unverified)"
                    if feedback_id in demo_examples_by_id
                    else (
                        f"Not eligible for demo training: {demo_ineligible_reason}"
                        if demo_review is not None and demo_ineligible_reason
                        else "Not eligible"
                    )
                ),
                "Training label": (
                    demo_review.training_label
                    if demo_review is not None
                    else status.training_label or "Not assigned"
                ),
                "Persisted provenance": (
                    "Complete"
                    if block_reason is None
                    else f"Incomplete: {block_reason}"
                ),
                "Safe metadata summary": safe_summary,
            }
        )
        selector_labels[feedback_id] = (
            f"{submitted_at} · {action} · {displayed_review_status}"
        )

    if table_rows:
        st.table(table_rows)
    else:
        st.info("No locally stored caregiver feedback is available yet.")

    st.subheader("Review a Feedback Row")
    st.write(
        "Choose a demonstration label manually. Confirm, Reject, and Report Missed "
        "Event actions are not converted into labels. This label is not verified "
        "ground truth and is not a training approval."
    )
    if history_rows:
        pending_ids = [
            row["feedback_id"]
            for row in history_rows
            if isinstance(row.get("feedback_id"), str)
            and review_statuses[row["feedback_id"]].status == "Awaiting review"
            and row["feedback_id"] not in demo_reviews
        ]
        default_feedback_id = next(
            iter(pending_ids or selector_labels),
            None,
        )
        selector_options = (
            [default_feedback_id]
            + [
                feedback_id
                for feedback_id in selector_labels
                if feedback_id != default_feedback_id
            ]
            if default_feedback_id is not None
            else []
        )
        selected_feedback_id = st.selectbox(
            "Feedback row",
            options=selector_options,
            format_func=lambda feedback_id: selector_labels[feedback_id],
            key="review_feedback_row",
        )
        selected_row = next(
            row
            for row in feedback_rows
            if row.get("feedback_id") == selected_feedback_id
        )
        provenance_diagnostic = feedback_review_provenance_diagnostic(
            selected_row, selected_feedback_id
        )
        st.caption("Persisted provenance diagnostic (read-only; not used to assign labels):")
        st.table(
            [
                {"Required category": category, "Status": category_status}
                for category, category_status in provenance_diagnostic.items()
            ]
        )
        selected_review_status = review_statuses.get(selected_feedback_id)
        selected_demo_review = demo_reviews.get(selected_feedback_id)
        already_reviewed = (
            selected_demo_review is not None
            or (
                selected_review_status is not None
                and selected_review_status.status != "Awaiting review"
            )
        )
        if already_reviewed:
            st.info(
                "This feedback row already has a saved review. Reviews are "
                "append-only and cannot be replaced."
            )
        with st.form("review_feedback_form", clear_on_submit=False):
            training_label = st.selectbox(
                "Demonstration label",
                options=[
                    "— Select a label —",
                    "potential_distress",
                    "no_sufficient_distress_evidence",
                ],
                key="demo_review_training_label",
            )
            st.caption(
                "One demonstration review per feedback row. The original feedback "
                "snapshot and record are not modified."
            )
            save_review = st.form_submit_button(
                "Save Review",
                disabled=already_reviewed,
                type="primary",
            )
        if save_review:
            if training_label not in DEMO_TRAINING_LABELS:
                st.error("Select potential_distress or no_sufficient_distress_evidence.")
            else:
                try:
                    append_demo_feedback_review(
                        feedback_store,
                        DEMO_REVIEW_PATH,
                        selected_feedback_id,
                        training_label,
                    )
                except (
                    ValueError,
                    OSError,
                    FeedbackStorageError,
                ) as exc:
                    st.error(f"Review was not saved: {exc}")
                else:
                    st.session_state.review_notice_kind = "success"
                    st.session_state.review_notice = (
                        "Demonstration review saved locally. This manually assigned "
                        "label is not independently verified ground truth and is not "
                        "eligible for training."
                    )
                    st.rerun()
    else:
        st.info("Save feedback first to enable explicit review.")

    st.subheader("Train / Personalize")
    try:
        reviewed = (
            load_reviewed_feedback(feedback_store, REVIEW_MANIFEST_PATH)
            if REVIEW_MANIFEST_PATH.is_file()
            else None
        )
    except (OSError, ValueError, FeedbackStorageError) as exc:
        reviewed = None
        st.error(f"Reviewed-data eligibility could not be checked: {exc}")
    eligible_examples = list(reviewed.examples) if reviewed is not None else []
    class_counts = {
        label: sum(example.label == label for example in eligible_examples)
        for label in ("potential_distress", "no_sufficient_distress_evidence")
    }
    training_block_reason = reviewed_training_block_reason(eligible_examples)
    demo_class_counts = {
        label: sum(
            example.training_label == label
            for example in demo_training_dataset.examples
        )
        for label in sorted(DEMO_TRAINING_LABELS)
    }
    st.caption(
        "Separate demo-training dataset from explicitly reviewed, persisted feedback: "
        f"{len(demo_training_dataset.examples)} eligible; "
        f"{len(demo_training_dataset.ineligible_reasons)} reviewed row(s) excluded "
        f"by validation. Class counts: {demo_class_counts}. Demo labels are manual "
        "and unverified; this dataset is not used by the v2 reviewed-data loader."
    )
    if not REVIEW_MANIFEST_PATH.is_file():
        st.info(
            "Training blocked: there is no v2 review manifest. Feedback actions alone "
            "cannot supply labels."
        )
    elif not eligible_examples:
        st.info(
            "Training blocked: no genuinely approved, eligible reviewed examples are available."
        )
    elif training_block_reason:
        st.info(f"Training blocked: {training_block_reason} Current class counts: {class_counts}.")
    else:
        st.write(f"Eligible reviewed examples by label: {class_counts}")

    reviewed_train_clicked = st.button(
        "Train / Personalize from reviewed data",
        key="train_reviewed_personalization",
        disabled=training_block_reason is not None,
    )
    if reviewed_train_clicked:
        try:
            result = run_reviewed_personalization(
                feedback_store,
                REVIEW_MANIFEST_PATH,
                PERSONALIZATION_DIRECTORY,
            )
            st.session_state.personalization_result = result
        except (OSError, ValueError, FeedbackStorageError) as exc:
            st.error(f"Personalization did not run: {exc}")

    st.warning(
        "**DEMO/SYNTHETIC ONLY:** this deterministic experiment demonstrates the "
        "training mechanism; it is not real-user personalization or evidence of "
        "real-world distress-detection performance. Synthetic data stays separate "
        "from reviewed feedback and live monitoring."
    )
    if st.button(
        "Run DEMO/SYNTHETIC ONLY experiment",
        key="train_synthetic_personalization",
    ):
        try:
            st.session_state.synthetic_personalization_result = (
                run_synthetic_personalization_demo(PERSONALIZATION_DIRECTORY)
            )
        except (OSError, ValueError) as exc:
            st.error(f"Synthetic demonstration did not run: {exc}")

    for result_key in (
        "personalization_result",
        "synthetic_personalization_result",
    ):
        result = st.session_state.get(result_key)
        if not isinstance(result, PersonalizationResult):
            continue
        if result.data_mode != "reviewed":
            st.info("Result type: **DEMO/SYNTHETIC ONLY**.")
        st.write(
            f"Samples: {result.sample_count} · class counts: {result.class_counts} · "
            f"training: {result.training_sample_count} · "
            f"held out: {result.held_out_sample_count}"
        )
        st.write(f"Model version: `{result.model_version}`")
        st.write("Held-out baseline metrics:")
        st.json(result.before_metrics)
        st.write("Held-out personalized metrics:")
        st.json(result.after_metrics)
        st.caption(
            f"New isolated artifact: `{result.artifact_path}`. Run metadata: "
            f"`{result.run_metadata_path}`. It was not activated by live inference."
        )


st.title("Autism Distress Companion")
st.caption(
    "Local device checks, continuous multimodal monitoring, background rule-cue evaluation, "
    "and non-clinical alert-state management. Raw audio and video are not stored."
)

if "monitoring_manager" not in st.session_state:
    st.session_state.monitoring_manager = None
if "simulated_demo_alert_manager" not in st.session_state:
    st.session_state.simulated_demo_alert_manager = SimulatedDemoAlertManager()
if "alert_manager" not in st.session_state:
    st.session_state.alert_manager = AlertManager(
        AlertPolicy(
            raw_probability_threshold=0.75,
            release_threshold=0.55,
            smoothing_window=5,
            release_count=2,
            cooldown_seconds=30.0,
            max_result_age_seconds=3.0,
        )
    )
if "feedback_store" not in st.session_state:
    st.session_state.feedback_store = FeedbackStore()
if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = None
if "feedback_message_kind" not in st.session_state:
    st.session_state.feedback_message_kind = None
if "feedback_message_target" not in st.session_state:
    st.session_state.feedback_message_target = None
if "score_alert_was_active" not in st.session_state:
    st.session_state.score_alert_was_active = False
if "score_popup_feedback_context" not in st.session_state:
    st.session_state.score_popup_feedback_context = None
if "score_popup_feedback_result" not in st.session_state:
    st.session_state.score_popup_feedback_result = None
if "missed_event_idempotency_key" not in st.session_state:
    st.session_state.missed_event_idempotency_key = str(uuid.uuid4())
if "missed_event_submitted" not in st.session_state:
    st.session_state.missed_event_submitted = False
if "camera_message" not in st.session_state:
    st.session_state.camera_message = "Not tested yet."
if "camera_available" not in st.session_state:
    st.session_state.camera_available = None
if "microphone_test_message" not in st.session_state:
    st.session_state.microphone_test_message = "Not tested yet."
if "microphone_test_available" not in st.session_state:
    st.session_state.microphone_test_available = None
if "microphone_levels" not in st.session_state:
    st.session_state.microphone_levels = None
if "audio_branch" not in st.session_state:
    st.session_state.audio_branch = None
if "audio_result" not in st.session_state:
    st.session_state.audio_result = None
if "audio_capture_message" not in st.session_state:
    st.session_state.audio_capture_message = None
if "face_expression_result" not in st.session_state:
    st.session_state.face_expression_result = None
if "face_capture_message" not in st.session_state:
    st.session_state.face_capture_message = None
if "behavior_branch" not in st.session_state:
    st.session_state.behavior_branch = None
if "behavior_result" not in st.session_state:
    st.session_state.behavior_result = None
if "behavior_capture_message" not in st.session_state:
    st.session_state.behavior_capture_message = None
if "pose_live_diagnostics_enabled" not in st.session_state:
    st.session_state.pose_live_diagnostics_enabled = False

if "microphone_status" not in st.session_state:
    status = check_microphone()
    st.session_state.microphone_status = status.message
    st.session_state.microphone_available = status.available

monitoring_manager_for_tests = st.session_state.monitoring_manager
device_tests_disabled = bool(
    monitoring_manager_for_tests is not None
    and (
        monitoring_manager_for_tests.is_running
        or monitoring_manager_for_tests.status == "stopping"
    )
)


@st.fragment(run_every=LIVE_DASHBOARD_REFRESH_SECONDS)
def _render_distress_score_and_threshold() -> None:
    manager = st.session_state.monitoring_manager
    snapshot = manager.snapshot() if manager is not None else {}

    score_result = calculate_distress_score(
        snapshot,
        weights=EXPERIMENTAL_DISTRESS_SCORE_WEIGHTS,
        freshness_seconds=EXPERIMENTAL_DISTRESS_SCORE_FRESHNESS_SECONDS,
        face_max_normalized_entropy=(
            EXPERIMENTAL_DISTRESS_SCORE_FACE_MAX_ENTROPY
        ),
        minimum_modalities=EXPERIMENTAL_DISTRESS_SCORE_MIN_MODALITIES,
    )
    st.subheader("Experimental Multimodal Distress-Related Evidence Score")
    st.caption(
        "Separate read-only dashboard indicator; it does not affect detector "
        "results or alert actions. This is not a probability or clinically "
        "validated measurement."
    )
    if score_result.score is None:
        st.info(
            "Insufficient evidence — at least "
            f"{EXPERIMENTAL_DISTRESS_SCORE_MIN_MODALITIES} fresh, valid, weighted "
            "modalities are required."
        )
    else:
        st.metric("Experimental evidence score (0–100)", f"{score_result.score:.1f}")
    modality_columns = st.columns(3)
    modality_labels = {
        "emotion": "Emotion evidence",
        "behavior": "Behaviour evidence",
        "audio": "Audio activity evidence",
    }
    for column, modality in zip(modality_columns, ("emotion", "behavior", "audio")):
        item = score_result.modalities[modality]
        with column:
            st.metric(
                modality_labels[modality],
                f"{item.score:.1f}" if item.score is not None else "Unavailable",
            )
            st.caption(
                f"Configured weight: {item.weight:g}% · "
                f"{'Contributed' if item.contributed else 'Not included'}"
                + (
                    f" · contribution {item.contribution:.1f}"
                    if item.contribution is not None
                    else ""
                )
            )
    with st.expander("Score evidence and exclusions"):
        st.caption(score_result.interpretation_warning)
        for modality in ("emotion", "behavior", "audio"):
            item = score_result.modalities[modality]
            state_label = (
                "contributed"
                if item.contributed
                else "valid but withheld by the minimum-evidence rule"
                if item.available
                else "unavailable"
            )
            st.write(f"**{modality_labels[modality]} — {state_label}**")
            for detail in item.evidence:
                st.caption(detail)
            reason = score_result.excluded_reasons.get(modality) or item.reason
            if reason:
                st.caption(reason)

    score_alert = score_alert_triggered(
        score_result,
        threshold=EXPERIMENTAL_DISTRESS_ALERT_THRESHOLD,
    )
    score_alert_message = "Potential Distress — Caregiver Attention Recommended"
    popup_message, score_alert_was_active = score_popup_transition(
        score_alert_active=score_alert,
        previous_score_alert_active=st.session_state.score_alert_was_active,
        score=score_result.score,
    )
    st.session_state.score_alert_was_active = score_alert_was_active
    if popup_message:
        popup_alert_id = str(uuid.uuid4())
        popup_evidence = build_fusion_evidence(snapshot)
        score_snapshot_record = (
            evidence_to_feature_record(
                popup_evidence,
                provenance={"source": "monitoring_diagnostic_snapshot"},
            )
            if popup_evidence.has_data
            else None
        )
        popup_record, popup_provenance_available = (
            record_for_score_snapshot(snapshot, score_snapshot_record)
            if score_snapshot_record is not None
            else (None, False)
        )
        st.session_state.score_popup_feedback_context = (
            alert_feedback_context_for_snapshot(
                popup_alert_id,
                popup_record,
                provenance_available=popup_provenance_available,
            )
        )
        st.session_state.score_popup_feedback_result = None
        show_alert_popup(popup_message, popup_alert_id)

    st.subheader("Multimodal Distress Alert")
    st.caption(
        "Experimental score-based dashboard indicator using indirect cues only. "
        "It is not a diagnosis or validated distress probability and does not "
        "change the existing detector, alert manager, or audio-peak alert."
    )
    if score_alert is True:
        st.warning(score_alert_message)
    elif score_alert is False:
        st.info("No score-based alert currently triggered")
    else:
        st.info("Unable to assess — insufficient evidence")
    score_value = (
        f"{score_result.score:.1f}/100"
        if score_alert is not None
        else "Unavailable"
    )
    score_alert_columns = st.columns(2)
    score_alert_columns[0].metric("Current multimodal score", score_value)
    score_alert_columns[1].metric(
        "Configured alert threshold",
        f"{EXPERIMENTAL_DISTRESS_ALERT_THRESHOLD:g}/100",
    )


def _render_monitoring_controls() -> None:
    st.subheader("Monitoring controls")
    st.caption(
        "Sensors and rule-cue evaluation run in the background. This dashboard refreshes "
        "about every 1.5 seconds. Rule settings are not clinically validated."
    )
    previous_render_ms = st.session_state.get("monitoring_panel_render_ms")
    if previous_render_ms is not None:
        st.caption(
            f"Previous monitoring-panel render time: {previous_render_ms:.1f} ms "
            "(measured on the prior dashboard refresh)."
        )

    manager = st.session_state.monitoring_manager
    start_column, stop_column = st.columns(2)
    with start_column:
        if st.button(
            "Start monitoring",
            use_container_width=True,
            type="primary",
        ):
            if manager is None:
                manager = create_monitoring_manager(st.session_state.alert_manager)
                st.session_state.monitoring_manager = manager
            manager.start()
            st.rerun(scope="app")
    with stop_column:
        if st.button("Stop monitoring", use_container_width=True):
            if manager is not None:
                manager.stop()
            st.rerun(scope="app")


@st.fragment(run_every=LIVE_DASHBOARD_REFRESH_SECONDS)
def render_monitoring_panel() -> None:
    render_started_at = time.perf_counter()
    worker_snapshot = {}
    inference = None
    inference_timestamp = None
    inference_stale = False
    inference_error = None
    inference_metadata = None
    detection_mode = "model"
    rule_based = False

    monitoring_manager_for_tests = st.session_state.monitoring_manager
    device_tests_disabled = bool(
        monitoring_manager_for_tests is not None
        and (
            monitoring_manager_for_tests.is_running
            or monitoring_manager_for_tests.status == "stopping"
        )
    )

    manager = st.session_state.monitoring_manager
    if manager is None:
        status = "stopped"
        message = "Monitoring is idle; start to process camera and microphone signals."
        snapshot = {}
        inference = None
        inference_timestamp = None
        alert = st.session_state.alert_manager.report_unknown(
            "Monitoring is stopped; no fresh inference is being produced."
        )
        rule_based = False
    else:
        snapshot = manager.snapshot()
        status = snapshot.get("status", "stopped")
        message = snapshot.get("message", "Monitoring is running.")
        inference = snapshot.get("latest_inference")
        inference_timestamp = snapshot.get("latest_inference_timestamp")
        alert = snapshot.get("latest_alert")
        metadata = getattr(inference, "metadata", None) if inference is not None else None
        rule_based = isinstance(metadata, dict) and metadata.get("detection_mode") == "rule_based"
    _render_current_distress_status(
        status=status,
        message=message,
        inference=inference,
        alert=alert,
        rule_based=rule_based,
        inference_timestamp=inference_timestamp,
    )

    st.subheader("Monitoring and Sensors")
    camera_column, microphone_column = st.columns(2)
    with camera_column:
        st.subheader("Webcam")
        camera_status = st.session_state.camera_available
        if camera_status is None:
            st.info(f"Status: {st.session_state.camera_message}")
        elif camera_status:
            st.success(f"Status: {st.session_state.camera_message}")
        else:
            st.error(f"Status: {st.session_state.camera_message}")

        if st.button(
            "Test webcam",
            use_container_width=True,
            disabled=device_tests_disabled,
        ):
            result = capture_frame()
            st.session_state.camera_available = result.available
            st.session_state.camera_message = (
                "Camera opened and captured one frame."
                if result.available
                else result.error or "Camera test failed."
            )
            if result.available and result.frame is not None:
                rgb_frame = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_frame, caption="Current webcam test frame")
            else:
                logger.warning("Webcam test unavailable: %s", st.session_state.camera_message)
                st.error(st.session_state.camera_message)

    with microphone_column:
        st.subheader("Microphone")
        if st.session_state.microphone_available:
            st.success(f"Status: {st.session_state.microphone_status}")
        else:
            st.error(f"Status: {st.session_state.microphone_status}")

        if st.button(
            "Test microphone",
            use_container_width=True,
            disabled=device_tests_disabled,
        ):
            result = test_microphone()
            st.session_state.microphone_test_available = result.available
            st.session_state.microphone_test_message = result.message
            st.session_state.microphone_levels = (
                (result.rms_level, result.peak_level) if result.available else None
            )
            if result.available:
                logger.info("Microphone test completed")
            else:
                logger.warning("Microphone test unavailable: %s", result.message)

        microphone_test_available = st.session_state.microphone_test_available
        if microphone_test_available is True:
            st.success(st.session_state.microphone_test_message)
            rms_level, peak_level = st.session_state.microphone_levels
            st.write(f"Input level — RMS: {rms_level:.5f}; peak: {peak_level:.5f}")
        elif microphone_test_available is False:
            st.error(st.session_state.microphone_test_message)
        else:
            st.caption(f"Test: {st.session_state.microphone_test_message}")

    manager = st.session_state.monitoring_manager
    if manager is None:
        status = "stopped"
        message = "Monitoring is idle; start to process camera and microphone signals."
        snapshot = {
            "latest_face_result": None,
            "latest_behavior_result": None,
            "latest_audio_result": None,
            "camera_status": "idle",
            "camera_capture_attempts": 0,
            "camera_success_count": 0,
            "last_camera_success_timestamp": None,
            "microphone_status": "idle",
            "audio_capture_attempts": 0,
            "audio_success_count": 0,
            "last_microphone_success_timestamp": None,
            "camera_message": "Camera has not been tested in monitoring mode.",
            "microphone_message": "Microphone has not been tested in monitoring mode.",
            "processing_iterations": 0,
            "current_operation": "idle",
            "face_analysis_attempts": 0,
            "face_inference_invocations": 0,
            "face_results_returned": 0,
            "face_results_published": 0,
            "behavior_analysis_attempts": 0,
            "behavior_inference_invocations": 0,
            "behavior_results_returned": 0,
            "behavior_results_published": 0,
            "audio_analysis_attempts": 0,
            "audio_inference_invocations": 0,
            "audio_results_returned": 0,
            "audio_results_published": 0,
            "camera_frames_captured": 0,
            "last_camera_opened": None,
            "last_camera_frame_info": None,
            "worker_alive": False,
            "worker_exit_reason": None,
            "last_error": {},
            "diagnostics": {},
            "stage_timings": {},
            "alert_diagnostics": [],
        }
    else:
        snapshot = manager.snapshot()
        status = snapshot["status"]
        message = snapshot["message"]

    if status == "running":
        st.success(f"Monitoring: RUNNING — {message}")
    elif status == "starting":
        st.info(f"Monitoring: STARTING — {message}")
    elif status == "error":
        st.error(f"Monitoring: ERROR — {message}")
    elif status == "stopping":
        st.warning(f"Monitoring: STOPPING — {message}")
    else:
        st.caption(f"Monitoring: {status.upper()} — {message}")

    pose_diagnostics_enabled = st.checkbox(
        "Enable live pose analysis overlay and diagnostics",
        key="pose_live_diagnostics_enabled",
        help=(
            "Shows the latest frame processed by the existing monitoring camera, "
            "with pose skeleton, wrist confidence, and region tracking values. It does not open "
            "another camera or save images."
        ),
    )

    if manager is not None:
        manager.set_pose_diagnostics_enabled(pose_diagnostics_enabled)
        snapshot = manager.snapshot()
        status = snapshot["status"]
        st.write(
            "Camera: "
            f"{snapshot['camera_status']} | attempts={snapshot['camera_capture_attempts']} | "
            f"branch results returned={snapshot['camera_success_count']} | "
            f"last success={snapshot['last_camera_success_timestamp']}"
        )
        st.write(
            "Microphone: "
            f"{snapshot['microphone_status']} | attempts={snapshot['audio_capture_attempts']} | "
            f"branch results returned={snapshot['audio_success_count']} | "
            f"last success={snapshot['last_microphone_success_timestamp']}"
        )
        st.caption(
            "Capture and returned-result counters are operational diagnostics only. "
            "A returned branch result can be unknown or abstain and is not proof of "
            "valid inference or alert evidence."
        )
        st.write(f"Last camera message: {snapshot['camera_message']}")
        st.write(f"Last microphone message: {snapshot['microphone_message']}")
        st.write(f"Monitoring iterations: {snapshot['processing_iterations']}")
        st.write(f"Current worker operation: {snapshot['current_operation']}")
        stage_timings = snapshot.get("stage_timings", {})
        if stage_timings:
            timing_rows = []
            for stage, timing in stage_timings.items():
                mean_ms = float(timing["mean_ms"])
                timing_rows.append(
                    {
                        "Stage": stage.replace("_", " ").title(),
                        "Last (ms)": round(float(timing["last_ms"]), 2),
                        "Mean (ms)": round(mean_ms, 2),
                        "Max (ms)": round(float(timing["max_ms"]), 2),
                        "Samples": int(timing["count"]),
                        "Rate (Hz)": (
                            round(1000.0 / mean_ms, 2)
                            if stage == "camera_intercapture" and mean_ms > 0
                            else None
                        ),
                    }
                )
            st.caption(
                "Worker stage timings are in-memory aggregates. Camera rate is "
                "calculated from successive captured-frame timestamps."
            )
            st.table(timing_rows)
        st.write(
            f"Worker alive: {'Yes' if snapshot['worker_alive'] else 'No'} | "
            f"Exit reason: {snapshot['worker_exit_reason'] or 'not exited'}"
        )
        st.caption(
            "Branch attempts/invoked/results returned/results published — "
            f"face {snapshot['face_analysis_attempts']}/"
            f"{snapshot['face_inference_invocations']}/"
            f"{snapshot['face_results_returned']}/"
            f"{snapshot['face_results_published']}, "
            f"behavior {snapshot['behavior_analysis_attempts']}/"
            f"{snapshot['behavior_inference_invocations']}/"
            f"{snapshot['behavior_results_returned']}/"
            f"{snapshot['behavior_results_published']}, "
            f"audio {snapshot['audio_analysis_attempts']}/"
            f"{snapshot['audio_inference_invocations']}/"
            f"{snapshot['audio_results_returned']}/"
            f"{snapshot['audio_results_published']}."
        )
        for modality, error in snapshot["last_error"].items():
            if error:
                st.warning(f"{modality.capitalize()} error: {error}")
        for modality, diagnostic in snapshot["diagnostics"].items():
            stage = diagnostic["stage"]
            if stage != "idle":
                exception_type = diagnostic["exception_type"]
                detail = diagnostic["message"] or "No additional diagnostic message."
                exception = f" ({exception_type})" if exception_type else ""
                st.caption(
                    f"{modality.capitalize()} diagnostic: {stage}{exception} — {detail}"
                )

        raw_snapshot = snapshot
        freshness: dict[str, bool] = {}
        now_monotonic = time.monotonic()
        for modality, title in (
            ("face", "Latest face result"),
            ("behavior", "Latest behavior result"),
            ("audio", "Latest audio result"),
        ):
            st.subheader(title)
            result = raw_snapshot[f"latest_{modality}_result"]
            timestamp = raw_snapshot.get(f"last_{modality}_success_timestamp")
            stale = result is not None and (
                timestamp is None or now_monotonic - float(timestamp) > 3.0
            )
            freshness[modality] = not stale
            if stale:
                st.warning(
                    f"{modality.capitalize()}: Stale result; excluded from current inference."
                )
            else:
                if modality in {"face", "behavior"}:
                    capture_status = snapshot["camera_status"]
                    capture_message = snapshot["camera_message"]
                else:
                    capture_status = snapshot["microphone_status"]
                    capture_message = snapshot["microphone_message"]
                analysis_attempts = snapshot[f"{modality}_analysis_attempts"]
                analysis_error = snapshot["last_error"].get(modality)
                results_returned = snapshot[f"{modality}_results_returned"]
                if result is not None:
                    waiting_message = ""
                elif status == "stopped" and results_returned > 0:
                    waiting_message = (
                        f"Monitoring stopped and cleared the latest result to prevent "
                        f"stale inference; {results_returned} result(s) were previously "
                        "returned during this run."
                    )
                elif status == "stopped" and analysis_attempts > 0:
                    waiting_message = (
                        "Monitoring stopped before this branch returned a result during "
                        "the current run."
                    )
                elif analysis_attempts > 0 and analysis_error:
                    waiting_message = (
                        f"Analysis was attempted "
                        f"{analysis_attempts} time(s) but returned no result: "
                        f"{analysis_error}"
                    )
                elif analysis_attempts > 0:
                    waiting_message = (
                        f"Analysis was attempted "
                        f"{analysis_attempts} time(s); no result "
                        "has been returned yet."
                    )
                elif capture_status in {"unavailable", "error"}:
                    waiting_message = (
                        f"No result was produced because {modality} input is "
                        f"{capture_status}: {capture_message}"
                    )
                else:
                    waiting_message = (
                        f"No result has been returned yet. Current worker operation: "
                        f"{snapshot['current_operation']}."
                    )
                _render_monitoring_result_block(
                    modality.capitalize(),
                    result,
                    waiting_message,
                )
        snapshot = dict(raw_snapshot)
        for modality, is_fresh in freshness.items():
            if not is_fresh:
                snapshot[f"latest_{modality}_result"] = None
    else:
        raw_snapshot = snapshot
        freshness = {"face": False, "behavior": False, "audio": False}

    st.header("Live Pose Analysis")
    st.caption(
        "Read-only view of the latest pose result produced by the monitoring worker. "
        "This screen does not capture a second camera stream or run pose inference."
    )
    behavior_result = raw_snapshot.get("latest_behavior_result")
    if behavior_result is None:
        st.info(
            "No live pose result is available. Start monitoring to populate this view."
        )
    else:
        pose_timestamp = raw_snapshot.get("last_behavior_success_timestamp")
        st.caption(
            f"Latest worker-produced pose result timestamp (monotonic): "
            f"{pose_timestamp if pose_timestamp is not None else 'unavailable'}"
        )
        st.caption(
            "Pose evidence: "
            + (
                "fresh"
                if status == "running" and freshness.get("behavior", False)
                else "stale or monitoring stopped; not used for live inference"
            )
        )
        pose_diagnostic = getattr(behavior_result, "pose_diagnostic", {})
        region_metrics = getattr(behavior_result, "region_metrics", {})
        region_validity = getattr(behavior_result, "region_feature_validity", {})
        region_reasons = getattr(behavior_result, "region_unavailable_reasons", {})
        region_tracking = pose_diagnostic.get("regions", {})
        if pose_diagnostics_enabled:
            st.caption(
                "Temporary in-memory overlay from the latest processed camera frame."
            )
            overlay = getattr(behavior_result, "debug_overlay", None)
            if overlay is not None:
                st.image(
                    cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                    caption=(
                        "MediaPipe pose landmarks and skeleton from the monitoring "
                        "worker's camera frame."
                    ),
                )
            st.json(
                {
                    "frame_received": pose_diagnostic.get("frame_received"),
                    "frame_dimensions": pose_diagnostic.get("frame_dimensions"),
                    "pose_detected": pose_diagnostic.get("pose_detected"),
                    "visibility_threshold": pose_diagnostic.get(
                        "visibility_threshold"
                    ),
                    "wrists": pose_diagnostic.get("wrists", {}),
                }
            )
        region_rows = []
        for region in (
            "head",
            "left_arm",
            "right_arm",
            "torso",
            "left_leg",
            "right_leg",
            "whole_body",
        ):
            tracking = region_tracking.get(region, {})
            metrics = region_metrics.get(region, {})
            validity = region_validity.get(region, {})
            reasons = region_reasons.get(region, {})
            region_rows.append(
                {
                    "Region": region.replace("_", " ").title(),
                    "Tracking": (
                        "Available" if tracking.get("available", False)
                        else "Unavailable"
                    ),
                    "Confidence": "; ".join(
                        f"{name}={score:.2f}"
                        for name, score in tracking.get("confidence", {}).items()
                        if score is not None
                    ),
                    "Intensity (normalized/s)": (
                        metrics.get("movement_energy")
                        if validity.get("movement_energy", False)
                        else None
                    ),
                    "Repetition (Hz)": (
                        metrics.get("repetition_rate_hz")
                        if validity.get("repetition_rate_hz", False)
                        else None
                    ),
                    "Acceleration (normalized/s²)": (
                        metrics.get("peak_acceleration")
                        if validity.get("peak_acceleration", False)
                        else None
                    ),
                    "Observations / span": (
                        f"{tracking.get('observation_count', 0)} / "
                        f"{tracking.get('span_seconds', 0.0):.2f}s"
                    ),
                    "Reason": (
                        tracking.get("reason")
                        or reasons.get("repetition_rate_hz")
                        or reasons.get("movement_energy")
                        or ""
                    ),
                }
            )
        st.table(region_rows)
        inference_snapshot = manager.snapshot() if manager is not None else {}
        latest_inference = inference_snapshot.get("latest_inference")
        inference_timestamp = inference_snapshot.get("latest_inference_timestamp")
        inference_fresh = (
            status == "running"
            and inference_timestamp is not None
            and 0.0 <= time.monotonic() - float(inference_timestamp) <= 3.0
        )
        if inference_fresh and latest_inference is not None:
            cue_indicators = getattr(latest_inference, "cue_indicators", {})
            st.caption(
                "Current configured cue states from the same fresh inference used "
                "by the live detector (not a diagnosis)."
            )
            st.table(
                [
                    {"Cue": name.replace("_", " ").title(), "State": value}
                    for name, value in cue_indicators.items()
                ]
            )
        else:
            st.info("No fresh inference is available; cue states are unavailable.")
        if not pose_diagnostics_enabled:
            st.caption(
                "Enable the pose analysis overlay and diagnostics above to view the "
                "annotated camera image and wrist confidence details."
            )
        st.caption(
            "Regional intensity, speed, and acceleration are diagnostic measurements. "
            "The active rule detector uses configured regional repetition cues plus "
            "the existing wrist-based cues; these measurements alone do not trigger alerts."
        )

    st.subheader("Phase 3 fusion evidence")
    fusion_evidence = build_fusion_evidence(snapshot)
    if not fusion_evidence.has_data:
        if status == "stopped":
            st.info(
                "Current live features were cleared when monitoring stopped; no live "
                "feature values are available now."
            )
        else:
            st.info(
                "Waiting for first multimodal evidence. No feature values are available yet."
            )
    else:
        rows = [
            {
                "Feature": name,
                "Value": (
                    f"{fusion_evidence.feature_values[name]:.6f}"
                    if fusion_evidence.feature_validity.get(name, False)
                    and fusion_evidence.feature_values.get(name) is not None
                    else "Unavailable"
                ),
                "Valid": "Yes" if fusion_evidence.feature_validity.get(name, False) else "No",
                "Reason": fusion_evidence.unavailable_reasons.get(name, ""),
            }
            for name in fusion_evidence.feature_names
        ]
        st.table(rows)
    st.caption(
        "Feature values retain their validity and unavailable state. Missing modalities "
        "are not treated as normal evidence."
    )

    feature_record = evidence_to_feature_record(
        fusion_evidence,
        provenance={"source": "live_monitoring"},
    )
    worker_snapshot = manager.snapshot() if manager is not None else {}
    alert_feedback_context = retained_alert_feedback_context(
        worker_snapshot.get("alert_feature_record")
    )
    inference = (
        worker_snapshot.get("latest_inference")
        if status == "running"
        else None
    )
    inference_timestamp = worker_snapshot.get("latest_inference_timestamp")
    inference_age = (
        time.monotonic() - float(inference_timestamp)
        if inference_timestamp is not None
        else None
    )
    inference_stale = (
        status == "running"
        and inference_age is not None
        and inference_age > 3.0
    )
    inference_error = worker_snapshot.get("latest_inference_error")
    inference_metadata = (
        getattr(inference, "metadata", None) if inference is not None else None
    )
    detection_mode = (
        inference_metadata.get("detection_mode", "model")
        if isinstance(inference_metadata, dict)
        else "model"
    )
    rule_based = detection_mode == "rule_based"

    live_results = {
        modality: raw_snapshot.get(f"latest_{modality}_result")
        for modality in ("face", "behavior", "audio")
    }
    live_modalities = live_modality_summary(
        status,
        results=live_results,
        freshness=freshness,
        available_modalities=feature_record.available_modalities,
    )
    st.subheader("Live modality evidence")
    st.table(
        [
            {"Modality": modality.capitalize(), "Live contribution": state}
            for modality, state in live_modalities.items()
        ]
    )
    sensor_now = time.monotonic()
    sensor_rows = [
        {
            "Sensor": label,
            "State": live_sensor_status(
                raw_snapshot.get(status_key, "unavailable"),
                raw_snapshot.get(timestamp_key),
                status,
                now=sensor_now,
            ),
            "Last successful sample (monotonic)": raw_snapshot.get(timestamp_key),
        }
        for label, status_key, timestamp_key in (
            ("Camera", "camera_status", "last_camera_success_timestamp"),
            ("Microphone", "microphone_status", "last_microphone_success_timestamp"),
        )
    ]
    st.caption("Sensor state")
    st.table(sensor_rows)
    from training import evidence_sufficient

    evidence_is_sufficient, _ = evidence_sufficient(feature_record)
    if inference is not None and rule_based and not inference_stale:
        evidence_is_sufficient = inference.rule_evidence_sufficient
    if manager is not None and status == "running":
        diagnostic_evidence = build_fusion_evidence(raw_snapshot)
        diagnostic_record = evidence_to_feature_record(
            diagnostic_evidence,
            provenance={"source": "monitoring_diagnostic_snapshot"},
        )
        diagnostic_sufficient, diagnostic_sufficiency_reason = evidence_sufficient(
            diagnostic_record
        )
        diagnostic_source_timestamps = {
            modality: (
                raw_snapshot.get(f"last_{modality}_success_timestamp")
                if raw_snapshot.get(f"latest_{modality}_result") is not None
                else None
            )
            for modality in ("face", "behavior", "audio")
        }
        diagnostic_snapshot = build_fusion_diagnostic_snapshot(
            diagnostic_evidence,
            diagnostic_record,
            source_timestamps=diagnostic_source_timestamps,
            sufficient=diagnostic_sufficient,
            sufficiency_reason=diagnostic_sufficiency_reason,
            runtime_diagnostics={
                "monitor_status": raw_snapshot.get("status"),
                "worker_alive": raw_snapshot.get("worker_alive"),
                "worker_exit_reason": raw_snapshot.get("worker_exit_reason"),
                "camera_status": raw_snapshot.get("camera_status"),
                "camera_capture_attempts": raw_snapshot.get("camera_capture_attempts"),
                "camera_frames_captured": raw_snapshot.get("camera_frames_captured"),
                "last_camera_opened": raw_snapshot.get("last_camera_opened"),
                "last_camera_frame_info": raw_snapshot.get("last_camera_frame_info"),
                "face_analysis_attempts": raw_snapshot.get("face_analysis_attempts"),
                "face_inference_invocations": raw_snapshot.get(
                    "face_inference_invocations"
                ),
                "face_results_returned": raw_snapshot.get("face_results_returned"),
                "face_results_published": raw_snapshot.get("face_results_published"),
                "behavior_analysis_attempts": raw_snapshot.get(
                    "behavior_analysis_attempts"
                ),
                "behavior_inference_invocations": raw_snapshot.get(
                    "behavior_inference_invocations"
                ),
                "behavior_results_returned": raw_snapshot.get(
                    "behavior_results_returned"
                ),
                "behavior_results_published": raw_snapshot.get(
                    "behavior_results_published"
                ),
                "last_face_success_timestamp": raw_snapshot.get(
                    "last_face_success_timestamp"
                ),
                "last_behavior_success_timestamp": raw_snapshot.get(
                    "last_behavior_success_timestamp"
                ),
                "diagnostics": {
                    modality: raw_snapshot.get("diagnostics", {}).get(modality)
                    for modality in ("camera", "face", "behavior", "worker")
                },
            },
        )
        manager.store_fusion_diagnostic_snapshot(diagnostic_snapshot)

    st.subheader("Historical fusion diagnostic snapshot")
    st.caption(
        "Read-only diagnostic history. This snapshot is never used for live inference "
        "or alert updates."
    )
    historical_snapshot = (
        manager.snapshot()["fusion_diagnostic_snapshot"] if manager is not None else None
    )
    if historical_snapshot is None:
        st.info("Fusion evidence was never constructed during this monitoring session.")
    else:
        construction_status = historical_snapshot["construction_status"]
        if construction_status == "constructed_with_valid_multimodal_features":
            st.success(
                "Fusion evidence was constructed with valid multimodal features."
            )
        else:
            st.warning("Fusion evidence was constructed, but was insufficient.")
        st.caption(
            f"Constructed at: {historical_snapshot['constructed_at_utc'] or 'unknown'}"
        )
        if historical_snapshot["sufficiency_reason"]:
            st.caption(historical_snapshot["sufficiency_reason"])
        modality_rows = [
            {
                "Modality": modality,
                "Status": historical_snapshot["modality_status"].get(modality, "waiting"),
                "Message": historical_snapshot["modality_messages"].get(modality, ""),
                "Source timestamp (monotonic)": historical_snapshot[
                    "source_timestamps"
                ].get(modality),
            }
            for modality in ("face", "behavior", "audio")
        ]
        st.table(modality_rows)
        feature_rows = [
            {
                "Feature": name,
                "Value": (
                    f"{historical_snapshot['feature_values'][name]:.6f}"
                    if historical_snapshot["feature_validity"].get(name, False)
                    and historical_snapshot["feature_values"].get(name) is not None
                    else "Unavailable"
                ),
                "Valid": (
                    "Yes"
                    if historical_snapshot["feature_validity"].get(name, False)
                    else "No"
                ),
                "Reason": historical_snapshot["unavailable_reasons"].get(name, ""),
            }
            for name in historical_snapshot["feature_names"]
        ]
        if feature_rows:
            st.table(feature_rows)
        st.caption(
            f"{historical_snapshot['valid_feature_count']} valid feature(s); "
            "available modalities: "
            + (
                ", ".join(
                    modality
                    for modality, available in historical_snapshot[
                        "available_modalities"
                    ].items()
                    if available
                )
                or "none"
            )
        )
        runtime = historical_snapshot["runtime_diagnostics"]
        st.caption(
            "Monitoring state when constructed — "
            f"status={runtime.get('monitor_status')}, "
            f"worker_alive={runtime.get('worker_alive')}, "
            f"camera attempts={runtime.get('camera_capture_attempts')}, "
            f"frames captured={runtime.get('camera_frames_captured')}, "
            f"face invoked/returned/published="
            f"{runtime.get('face_inference_invocations')}/"
            f"{runtime.get('face_results_returned')}/"
            f"{runtime.get('face_results_published')}, "
            f"behavior invoked/returned/published="
            f"{runtime.get('behavior_inference_invocations')}/"
            f"{runtime.get('behavior_results_returned')}/"
            f"{runtime.get('behavior_results_published')}."
        )
        st.caption(
            f"Camera opened for last capture: {runtime.get('last_camera_opened')}; "
            f"last frame metadata: {runtime.get('last_camera_frame_info')}. "
            "A single camera session is reused during monitoring and released on stop."
        )
        for modality in ("camera", "face", "behavior", "worker"):
            diagnostic = runtime.get("diagnostics", {}).get(modality)
            if diagnostic and diagnostic.get("stage") != "idle":
                st.caption(
                    f"At construction — {modality}: {diagnostic['stage']} — "
                    f"{diagnostic.get('message') or ''}"
                )

    model_available = not (
        inference is not None
        and not rule_based
        and inference.status == "unknown"
        and inference.reason
        and "no trained distress model" in inference.reason.lower()
    )
    inference_display_reason = (
        "Latest inference is stale and excluded from current display."
        if inference_stale
        else inference_error
        or (
            inference.reason
            if inference is not None and inference.status == "unknown"
            else None
        )
    )
    fusion_status, fusion_status_message = live_fusion_status(
        status,
        model_available=model_available,
        evidence_sufficient=evidence_is_sufficient,
        inference_reason=inference_display_reason,
        model_error=inference_error,
    )
    if status == "running" and rule_based:
        fusion_status = "RULE-BASED CUES"
        fusion_status_message = (
            "Fresh feature values are evaluated by configurable audio and movement "
            "rules; no fusion classifier is loaded."
        )
    if inference_stale:
        fusion_status = "STALE"
        fusion_status_message = (
            "The last inference is older than the freshness limit; its scores are hidden."
        )
    alert_manager = st.session_state.alert_manager
    alert = (
        worker_snapshot.get("latest_alert")
        if status == "running"
        else alert_manager.report_unknown(
            "Monitoring is stopped; no fresh inference is being produced."
        )
    )

    st.subheader("Alert view")
    if rule_based:
        st.caption(
            "Rule-based demonstration mode: a fresh, sufficient, unique configured cue "
            "activates immediately. No probability is produced or used. An active alert "
            "releases after "
            f"{alert_manager.policy.release_count} consecutive valid results with no "
            "configured cue, then enters a "
            f"{alert_manager.policy.cooldown_seconds:.0f}-second cooldown. "
            "These settings are tunable, not clinically validated."
        )
    else:
        st.caption(
            "Live model activation is immediate on one new, unique, fresh, sufficient "
            "result labeled potential_distress when its raw potential-distress "
            "probability is at least "
            f"{alert_manager.policy.raw_probability_threshold:.2f}. The smoothed score "
            "is display-only."
        )
    st.write(f"Fusion inference status: **{fusion_status}**")
    st.caption(fusion_status_message)
    if status == "stopped":
        st.write("Monitoring status: **STOPPED**")
        st.write("Live inference: **NOT RUNNING**")
        st.write("Live classifier probability: **Unavailable**")
        st.caption(
            "Live features were cleared when monitoring stopped. No new inference is "
            "being produced; this stopped state does not describe the previous run."
        )
        st.info("Alert state: UNKNOWN because monitoring is stopped.")
    elif inference_stale:
        st.write("Live inference: **STALE — excluded from current results**")
        st.write("Prediction: **Unavailable**")
        st.write("Predicted-class probability: **Unavailable**")
        st.write("Potential-distress probability: **Unavailable**")
        st.info("Alert state: UNKNOWN — the latest inference is stale.")
    elif inference is not None:
        st.write(f"Inference status: **{inference.status}**")
        if inference.status == "potential_distress":
            st.warning("DETECTOR RESULT: POTENTIAL DISTRESS CUE")
        elif inference.status == "no_sufficient_distress_evidence":
            st.info("DETECTOR RESULT: NO SUFFICIENT DISTRESS EVIDENCE")
        else:
            st.warning(
                "DETECTOR RESULT: UNKNOWN — evidence is unavailable, stale, or "
                "insufficient to evaluate the selected mode."
            )
        st.caption(
            "A detector result is distinct from the alert manager's lifecycle state "
            "shown below; stale evidence is excluded from both."
        )
        if inference.reason:
            st.caption(inference.reason)
        if rule_based:
            st.write(f"Rule result: **{inference.predicted_label or 'Unavailable'}**")
            combination_mode = inference.metadata.get("combination_mode", "unknown")
            st.write(
                f"Selected rule combination mode: **{combination_mode}**"
            )
            if combination_mode == "movement_only":
                st.caption(
                    "Audio cues are monitored but cannot activate alerts in "
                    "movement_only mode."
                )
            elif combination_mode == "audio_only":
                st.caption(
                    "Movement cues are monitored but cannot activate alerts in "
                    "audio_only mode."
                )
            st.write(
                "Triggered cues: **"
                + (
                    ", ".join(
                        cue
                        for cue, active in inference.cue_indicators.items()
                        if active is True
                    )
                    or "none"
                )
                + "**"
            )
            st.write(f"Cue evaluation: {inference.cue_reason}")
            for modality in ("audio", "behavior"):
                st.caption(
                    f"{modality.title()} evidence: "
                    f"{inference.modality_evidence_status[modality]} — "
                    f"{inference.modality_evidence_reasons[modality]}"
                )
            unavailable_cues = inference.metadata.get(
                "cue_unavailable_reasons", {}
            )
            for cue, cue_reason in unavailable_cues.items():
                if cue_reason:
                    st.caption(f"{cue}: unavailable — {cue_reason}")
        else:
            st.write(
                f"Prediction: **{inference.predicted_label or 'Unavailable'}**"
            )
            if inference.predicted_label == "no_sufficient_distress_evidence":
                st.caption(
                    "The model selected its no-sufficient-distress-evidence class; "
                    "this is distinct from unknown, which means no valid prediction was produced."
                )
        st.write(
            f"Evidence: {inference.valid_feature_count} valid feature(s); "
            f"available modalities: {', '.join(name for name, value in inference.available_modalities.items() if value) or 'none'}"
        )
        score_columns = st.columns(3)
        if rule_based:
            audio_cue_state = inference.modality_cues["audio"]
            movement_cue_state = inference.modality_cues["behavior"]
            with score_columns[0]:
                st.metric(
                    "Audio cue",
                    "Active"
                    if audio_cue_state is True
                    else "Inactive"
                    if audio_cue_state is False
                    else "Unavailable",
                )
            with score_columns[1]:
                st.metric(
                    "Movement cue",
                    "Active"
                    if movement_cue_state is True
                    else "Inactive"
                    if movement_cue_state is False
                    else "Unavailable",
                )
            with score_columns[2]:
                st.metric("Classifier probability", "Not used")
        else:
            with score_columns[0]:
                if inference.model_score is not None:
                    st.metric(
                        f"Probability of predicted class: {inference.predicted_label}",
                        f"{inference.model_score:.3f}",
                    )
                else:
                    st.metric("Predicted-class probability", "Unavailable")
            with score_columns[1]:
                if inference.potential_distress_score is not None:
                    st.metric(
                        "Raw potential-distress class probability (not calibrated)",
                        f"{inference.potential_distress_score:.3f}",
                    )
                else:
                    st.metric("Potential-distress probability", "Unavailable")
            with score_columns[2]:
                if alert.smoothed_score is not None:
                    st.metric("Smoothed distress score", f"{alert.smoothed_score:.3f}")
                else:
                    st.metric("Smoothed distress score", "Unavailable")
        st.caption(
            "Inference timestamp (UTC): "
            f"{worker_snapshot.get('latest_inference_updated_at_utc') or 'unavailable'}"
        )
        if rule_based:
            st.caption("No probability or smoothed score is produced by rule-based detection.")
        elif alert.smoothed_score is not None:
            st.caption(
                f"Alert-manager smoothed potential-distress score: "
                f"{alert.smoothed_score:.3f}"
            )
        if alert is not None and alert.state == "potential_distress":
            st.error(
                (
                    "**LIVE RULE-CUE ALERT — CONFIGURED CUE ACTIVE**\n\n"
                    if rule_based
                    else "**CARETAKER ALERT — POTENTIAL DISTRESS**\n\n"
                )
                + f"Issued: {alert.activated_at_utc or 'timestamp unavailable'}\n\n"
                f"Alert ID: {alert.alert_id}\n\n"
                f"{alert.reason}"
            )
        elif alert.state == "normal":
            st.info(f"Alert lifecycle: NO ACTIVE ALERT — {alert.reason}")
        elif alert.state == "cooldown":
            st.info(f"Alert state: COOLDOWN — {alert.reason}")
        else:
            st.info(f"Alert state: UNKNOWN — {alert.reason}")
    else:
        st.write(f"Monitoring status: **{status.upper()}**")
        st.write("Live inference: **NOT RUNNING**")
        st.write("Live model score: **Unavailable**")
        st.info(f"Alert state: UNKNOWN — {alert.reason}")

    if alert_feedback_context is not None:
        st.subheader("Feedback for this Alert")
        st.caption(
            f"Alert ID: {alert_feedback_context.alert_id} · "
            f"Triggered: {alert_feedback_context.alert.activated_at_utc or 'timestamp unavailable'}"
        )
        if alert_feedback_context.record is None:
            st.warning(
                "The alert's triggering feature snapshot is unavailable; "
                "confirmation and rejection are disabled."
            )
        feedback_left, feedback_right = st.columns(2)
        with feedback_left:
            if st.button(
                "Confirm",
                key=f"feedback_confirm_{alert_feedback_context.alert_id}",
                disabled=alert_feedback_context.record is None,
            ):
                try:
                    result = submit_alert_feedback(
                        st.session_state.feedback_store,
                        alert_feedback_context,
                        "confirm_alert",
                    )
                    st.session_state.feedback_message = (
                        "Confirmation saved locally."
                        if result.inserted
                        else "This confirmation was already saved."
                    )
                    st.session_state.feedback_message_kind = "alert"
                    st.session_state.feedback_message_target = (
                        f"alert:{alert_feedback_context.alert_id}"
                    )
                    st.rerun()
                except (FeedbackStorageError, ValueError) as exc:
                    st.session_state.feedback_message = (
                        f"Feedback storage failed: {exc}"
                    )
                    st.session_state.feedback_message_kind = "alert"
                    st.session_state.feedback_message_target = (
                        f"alert:{alert_feedback_context.alert_id}"
                    )
        with feedback_right:
            if st.button(
                "Reject",
                key=f"feedback_reject_{alert_feedback_context.alert_id}",
                disabled=alert_feedback_context.record is None,
            ):
                try:
                    result = submit_alert_feedback(
                        st.session_state.feedback_store,
                        alert_feedback_context,
                        "reject_alert",
                    )
                    st.session_state.feedback_message = (
                        "Rejection saved locally."
                        if result.inserted
                        else "This rejection was already saved."
                    )
                    st.session_state.feedback_message_kind = "alert"
                    st.session_state.feedback_message_target = (
                        f"alert:{alert_feedback_context.alert_id}"
                    )
                    st.rerun()
                except (FeedbackStorageError, ValueError) as exc:
                    st.session_state.feedback_message = (
                        f"Feedback storage failed: {exc}"
                    )
                    st.session_state.feedback_message_kind = "alert"
                    st.session_state.feedback_message_target = (
                        f"alert:{alert_feedback_context.alert_id}"
                    )
        if (
            st.session_state.feedback_message_kind == "alert"
            and st.session_state.feedback_message_target
            == f"alert:{alert_feedback_context.alert_id}"
        ):
            if st.session_state.feedback_message.startswith(
                "Feedback storage failed:"
            ):
                st.error(st.session_state.feedback_message)
            else:
                st.success(st.session_state.feedback_message)

    st.subheader("Report a Missed Event")
    st.caption(
        "For a caregiver who observed an event that the detector did not alert on."
    )
    latest_feedback_record = worker_snapshot.get("latest_feature_record")
    missed_snapshot_is_current = (
        latest_feedback_record is not None
        and inference_timestamp is not None
        and 0.0 <= time.monotonic() - float(inference_timestamp) <= 3.0
    )
    missed_feedback_record = latest_feedback_record or feature_record
    if st.button(
        "Report Missed Event",
        key="feedback_missed_event",
        disabled=st.session_state.missed_event_submitted,
    ):
        try:
            result = st.session_state.feedback_store.submit(
                "missed_event",
                idempotency_key=st.session_state.missed_event_idempotency_key,
                record=missed_feedback_record,
                provenance_available=missed_snapshot_is_current,
            )
            st.session_state.feedback_message = (
                "Missed event saved locally."
                if result.inserted
                else "This missed-event report was already saved."
            )
            st.session_state.feedback_message_kind = "missed"
            st.session_state.feedback_message_target = "missed"
            st.session_state.missed_event_submitted = True
            st.rerun()
        except FeedbackStorageError as exc:
            st.session_state.feedback_message = f"Feedback storage failed: {exc}"
            st.session_state.feedback_message_kind = "missed"
            st.session_state.feedback_message_target = "missed"
    if st.session_state.missed_event_submitted and st.button(
        "Start another missed-event report",
        key="feedback_new_missed_event",
    ):
        st.session_state.missed_event_idempotency_key = str(uuid.uuid4())
        st.session_state.missed_event_submitted = False
        st.session_state.feedback_message = None
        st.session_state.feedback_message_kind = None
        st.session_state.feedback_message_target = None
    if (
        st.session_state.feedback_message_kind == "missed"
        and st.session_state.feedback_message_target == "missed"
    ):
        if st.session_state.feedback_message.startswith(
            "Feedback storage failed:"
        ):
            st.error(st.session_state.feedback_message)
        else:
            st.success(st.session_state.feedback_message)

    if inference is not None:
        limitations = inference.metadata.get("limitations")
        if limitations:
            st.caption(str(limitations))

    st.subheader("Recent Events")
    st.caption(
        "This live session records rule-based results, not classifier predictions. "
        "Probability columns remain empty; use the separate alert diagnostics export "
        "for detection mode and cue details."
    )
    history = worker_snapshot.get("inference_history", [])
    if history:
        history_columns = (
            "timestamp",
            "prediction",
            "predicted_class_probability",
            "potential_distress_probability",
            "smoothed_distress_score",
            "alert_state",
            "available_modalities",
            "inference_status",
            "detection_mode",
            "cue_indicators",
            "cue_reason",
        )
        st.table(
            [
                {column: entry.get(column) for column in history_columns}
                for entry in reversed(history)
            ]
        )
        st.download_button(
            "Download inference history CSV",
            data=inference_history_csv(history),
            file_name="distress_inference_history.csv",
            mime="text/csv",
            key="download_inference_history",
        )
    else:
        st.info("No inference results have been recorded in this session.")
    st.caption(
        "History contains inference summaries only. No camera frames or audio are retained."
    )

    with st.expander("Advanced Details"):
        st.caption(
            "Separate, versioned diagnostic export for alert-policy troubleshooting. "
            "Contains timestamps, scores, evidence summaries, and state transitions only; "
            "no raw media or personal identifiers."
        )
    alert_diagnostics = worker_snapshot.get("alert_diagnostics", [])
    if alert_diagnostics:
        diagnostic_columns = (
            "result_key",
            "decision",
            "evidence_freshness",
            "evidence_sufficient",
            "smoothed_score_before",
            "smoothed_score_after",
            "potential_distress_probability",
            "raw_probability_threshold",
            "activation_qualified",
            "release_counter_before",
            "release_counter_after",
            "alert_state_before",
            "alert_state_after",
        )
        st.table(
            [
                {column: entry.get(column) for column in diagnostic_columns}
                for entry in reversed(alert_diagnostics[-10:])
            ]
        )
        st.download_button(
            "Download temporary alert diagnostics CSV",
            data=alert_diagnostics_csv(alert_diagnostics),
            file_name="live_alert_diagnostics_v3.csv",
            mime="text/csv",
            key="download_live_alert_diagnostics",
        )
    else:
        st.info("No live alert-decision diagnostics have been recorded in this session.")

    st.session_state.monitoring_panel_render_ms = (
        time.perf_counter() - render_started_at
    ) * 1000.0


def _render_simulated_caretaker_workflow_demo() -> None:
    st.subheader("Simulated caretaker workflow demo")
    st.warning(
        "SIMULATED ONLY: this control creates an immediate demo alert. It does not "
        "use sensors, change live inference, enter live history, or save feedback."
    )
    if st.button(
        "Run simulated distress alert",
        key="run_simulated_distress_alert",
    ):
        st.session_state.simulated_demo_alert_manager.activate()
    demo_manager = st.session_state.simulated_demo_alert_manager
    demo_alert = demo_manager.alert
    if demo_alert is not None:
        st.error(
            f"🚨 **SIMULATED DEMO ALERT** — caretaker workflow demonstration only. "
            f"Issued: {demo_alert.issued_at_utc}"
        )
        st.caption(f"SIMULATED demo alert ID: {demo_alert.alert_id}")
        simulated_feedback_columns = st.columns(4)
        with simulated_feedback_columns[0]:
            if st.button(
                "Confirm SIMULATED demo",
                key="confirm_simulated_demo_alert",
            ):
                demo_manager.record_feedback("confirm")
        with simulated_feedback_columns[1]:
            if st.button(
                "Reject SIMULATED demo",
                key="reject_simulated_demo_alert",
            ):
                demo_manager.record_feedback("reject")
        with simulated_feedback_columns[2]:
            if st.button(
                "Report SIMULATED missed event",
                key="report_simulated_missed_event",
            ):
                demo_manager.record_feedback("report_missed_event")
        with simulated_feedback_columns[3]:
            if st.button(
                "Dismiss / Reset SIMULATED alert",
                key="reset_simulated_demo_alert",
            ):
                demo_manager.reset()
        st.caption(
            "Confirm, Reject, and Report Missed Event are simulated acknowledgements "
            "only; they are not stored in the real feedback dataset."
        )
        for feedback in demo_manager.feedback:
            st.info(
                f"SIMULATED {feedback['action']} handled for "
                f"{feedback['alert_id']}; not persisted."
            )


_render_monitoring_controls()

st.divider()
_render_distress_score_and_threshold()

st.divider()
_render_feedback_history_and_personalization()

st.divider()
render_monitoring_panel()

st.divider()
_render_simulated_caretaker_workflow_demo()

st.divider()
st.subheader("Audio-feature test")
st.caption(
    "This branch records one short local microphone window, computes quiet/signal "
    "features, and tracks a rolling baseline. It does not classify distress or label "
    "sounds in this phase."
)

audio_test_column, audio_reset_column = st.columns(2)
with audio_test_column:
    test_audio = st.button(
        "Test audio features",
        use_container_width=True,
        disabled=device_tests_disabled,
    )
with audio_reset_column:
    reset_audio = st.button("Reset audio baseline", use_container_width=True)

if reset_audio:
    audio_branch = st.session_state.audio_branch
    if audio_branch is not None:
        audio_branch.reset()
    st.session_state.audio_result = None
    st.session_state.audio_capture_message = None

if test_audio:
    branch = get_audio_branch()
    st.session_state.audio_result = None
    try:
        from microphone import capture_audio_window

        capture = capture_audio_window()
    except Exception as exc:  # pragma: no cover - UI fallback for hardware-edge cases
        st.session_state.audio_capture_message = f"Audio capture failed: {exc}"
        capture = None
    if capture is None:
        st.session_state.audio_capture_message = (
            "Audio capture could not be completed with the current microphone configuration."
        )
    elif capture.status == "missing":
        st.session_state.audio_capture_message = capture.message
    elif capture.status == "unknown":
        st.session_state.audio_capture_message = capture.message
    elif capture.status == "silent":
        st.session_state.audio_capture_message = capture.message
    else:
        st.session_state.audio_capture_message = capture.message
        st.session_state.audio_result = branch.analyze(capture)

audio_result = st.session_state.audio_result
if st.session_state.audio_capture_message:
    if audio_result is not None and audio_result.status == "silent":
        st.warning(st.session_state.audio_capture_message)
    elif audio_result is not None and audio_result.status in {"warming_up", "ok", "partial"}:
        st.info(st.session_state.audio_capture_message)
    else:
        st.warning(st.session_state.audio_capture_message)
if audio_result is not None:
    if audio_result.status == "ok":
        st.success(audio_result.status_message)
    elif audio_result.status == "warming_up":
        st.info(audio_result.status_message)
    elif audio_result.status == "silent":
        st.warning(audio_result.status_message)
    elif audio_result.status == "partial":
        st.warning(audio_result.status_message)
    else:
        st.error(audio_result.status_message)

    st.write(
        f"Audio quality: {audio_result.quality_status} "
        f"({audio_result.quality_flag})"
    )
    st.write(
        "Baseline: "
        f"{'ready' if audio_result.baseline_ready else 'warming up'} "
        f"({audio_result.baseline_sample_count} valid samples)"
    )
    if audio_result.rms_loudness is not None:
        st.metric("RMS loudness", f"{audio_result.rms_loudness:.6f}")
    feature_rows = [
        {
            "Feature": name,
            "Value": (
                f"{audio_result.feature_values[name]:.6f}"
                if audio_result.feature_validity[name]
                else "Unavailable"
            ),
            "Valid": "Yes" if audio_result.feature_validity[name] else "No",
            "Reason": audio_result.unavailable_reasons.get(name, ""),
        }
        for name in audio_result.feature_names
    ]
    st.table(feature_rows)
    if audio_result.audio_contribution is not None:
        st.metric("Audio contribution", f"{audio_result.audio_contribution:.4f}")
    st.caption(
        "RMS loudness is the local root-mean-square amplitude of the captured window, not a dB SPL figure. "
        "Spectral centroid and bandwidth are in Hz using a Hann window and FFT. The onset score is a short-term "
        "energy change relative to the recent local baseline and is not a scream or crying detector."
    )

st.divider()
st.subheader("Facial-expression test")
st.caption(
    "This branch reports expression-model estimates as a weak feature. "
    "It does not determine distress. The camera starts only when you request a test."
)


if st.button(
    "Test face expression",
    use_container_width=True,
    disabled=device_tests_disabled,
):
    camera_result = capture_frame()
    if not camera_result.available or camera_result.frame is None:
        st.session_state.face_capture_message = (
            camera_result.error or "Could not capture a frame for the face test."
        )
        st.session_state.face_expression_result = None
        logger.warning("Face test camera capture unavailable: %s", st.session_state.face_capture_message)
    else:
        st.session_state.face_capture_message = None
        st.session_state.face_expression_result = get_face_branch().analyze(
            camera_result.frame
        )

face_result: FaceResult | None = st.session_state.face_expression_result
if st.session_state.face_capture_message:
    st.error(st.session_state.face_capture_message)
elif face_result is not None:
    if face_result.face_detected:
        st.success("Face detected.")
    elif face_result.abstention_reason == "no_face":
        st.warning("No face detected; expression is unknown.")
    else:
        st.info(f"Face status: {face_result.status_message}")

    if face_result.face_quality_status == "usable":
        st.success(f"Face quality: usable ({face_result.face_quality_score:.2f})")
    elif face_result.face_quality_status == "poor":
        st.warning(
            f"Face quality: unusable ({face_result.face_quality_score:.2f}) — "
            f"{face_result.status_message}"
        )
    else:
        st.info(f"Face quality: {face_result.face_quality_status}")

    probability_rows = [
        {
            "Class": label,
            "Probability": (
                f"{face_result.expression_probabilities[label]:.4f}"
                if face_result.expression_probabilities[label] is not None
                else "Unavailable"
            ),
            "Valid estimate": (
                "Yes" if face_result.class_validity[label] else "No"
            ),
            "Supported by model": (
                "Yes" if face_result.class_availability[label] else "No"
            ),
        }
        for label in face_result.feature_vector_order
    ]
    st.table(probability_rows)
    if face_result.negative_affect_feature is not None:
        st.metric(
            "Weak negative-affect feature",
            f"{face_result.negative_affect_feature:.4f}",
        )
        st.caption(
            "Mean of angry, fear, and sad probabilities. This feature is not a "
            "distress classification."
        )
    if face_result.prediction_confidence is not None:
        confidence = face_result.prediction_confidence
        st.caption(
            "Uncertainty summary (not correctness): "
            f"top probability {confidence.maximum_probability:.3f}; "
            f"normalized entropy {confidence.normalized_entropy:.3f}; "
            f"top-two margin {confidence.top_two_margin:.3f}."
        )
    if face_result.abstention_reason:
        st.warning(face_result.status_message)
    elif face_result.status_message:
        st.caption(face_result.status_message)

st.divider()
st.subheader("Upper-body behavior test")
st.caption(
    "This branch reports local movement and posture measurements only. "
    "It does not decide whether movement is distress or harmful."
)


behavior_test_column, behavior_reset_column = st.columns(2)
with behavior_test_column:
    test_behavior = st.button(
        "Test upper-body behavior",
        use_container_width=True,
        disabled=device_tests_disabled,
    )
with behavior_reset_column:
    reset_behavior = st.button(
        "Reset behavior baseline",
        use_container_width=True,
    )

if reset_behavior:
    behavior_branch = st.session_state.behavior_branch
    if behavior_branch is not None:
        behavior_branch.reset()
    st.session_state.behavior_result = None
    st.session_state.behavior_capture_message = None

if test_behavior:
    behavior_branch = get_behavior_branch()
    st.session_state.behavior_result = None
    if not behavior_branch.is_ready:
        st.session_state.behavior_capture_message = (
            behavior_branch.initialization_error
            or "The local Pose Landmarker is not ready."
        )
    else:
        sequence = capture_frame_sequence()
        if not sequence.available:
            st.session_state.behavior_capture_message = (
                sequence.error or "Could not capture frames for the behavior test."
            )
        else:
            st.session_state.behavior_capture_message = sequence.error
            behavior_result = None
            for frame, captured_at in sequence.frames:
                behavior_result = behavior_branch.analyze(
                    frame,
                    timestamp_seconds=captured_at,
                )
            st.session_state.behavior_result = behavior_result

behavior_result: BehaviorResult | None = st.session_state.behavior_result
if st.session_state.behavior_capture_message:
    st.warning(st.session_state.behavior_capture_message)
if behavior_result is not None:
    if behavior_result.status == "ok":
        st.success(behavior_result.status_message)
    elif behavior_result.status == "warming_up":
        st.info(behavior_result.status_message)
    elif behavior_result.status == "partial":
        st.warning(behavior_result.status_message)
    else:
        st.warning(behavior_result.status_message)

    baseline_text = (
        f"Ready ({behavior_result.baseline_sample_count} valid samples)"
        if behavior_result.baseline_ready
        else (
            f"Warming up ({behavior_result.baseline_sample_count} valid samples)"
        )
    )
    st.write(f"Landmark quality: {behavior_result.quality_score:.0%}")
    st.write(f"Movement baseline: {baseline_text}")
    feature_rows = [
        {
            "Feature": name,
            "Value": (
                f"{behavior_result.feature_values[name]:.4f}"
                if behavior_result.feature_validity[name]
                else "Unavailable"
            ),
            "Valid": "Yes" if behavior_result.feature_validity[name] else "No",
            "Reason": behavior_result.unavailable_reasons.get(name, ""),
        }
        for name in behavior_result.feature_names
    ]
    st.table(feature_rows)
    st.caption(
        "Movement energy is measured in shoulder-widths per second. The sudden-"
        "movement indicator is a motion-change rule, not a distress label. "
        "Repetitive-movement change is unavailable in this phase."
    )

st.divider()
st.info(
    "Windows may ask for camera and microphone permission. Enable access for desktop "
    "apps in Settings → Privacy & security → Camera / Microphone. Test each device "
    "separately; the microphone test lasts 0.5 seconds and is not saved. Face and "
    "behavior tests are optional and only run when clicked. Behavior testing "
    "captures a short frame burst and keeps measurements in memory only."
)
