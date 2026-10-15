import ast

from fusion.live_status import (
    live_fusion_status,
    live_modality_summary,
    live_sensor_status,
)


def test_live_fusion_status_covers_prediction_waiting_unavailable_and_stopped():
    assert live_fusion_status(
        "running",
        model_available=True,
        evidence_sufficient=True,
    ) == ("RUNNING", "Prediction produced from sufficient fresh evidence.")

    waiting = live_fusion_status(
        "running",
        model_available=True,
        evidence_sufficient=False,
        inference_reason="Only one modality has valid features.",
    )
    assert waiting == (
        "WAITING FOR SUFFICIENT FRESH EVIDENCE",
        "Only one modality has valid features.",
    )

    unavailable = live_fusion_status(
        "running",
        model_available=False,
        evidence_sufficient=True,
    )
    assert unavailable == ("UNAVAILABLE", "No trained local model is available.")

    stopped = live_fusion_status(
        "stopped",
        model_available=True,
        evidence_sufficient=True,
    )
    assert stopped == (
        "STOPPED",
        "Monitoring is stopped; live inference is not running.",
    )


def test_live_modality_summary_marks_valid_missing_stale_and_cleared():
    summary = live_modality_summary(
        "running",
        results={"face": object(), "behavior": object(), "audio": None},
        freshness={"face": True, "behavior": False, "audio": True},
        available_modalities={"face": True, "behavior": False, "audio": False},
    )
    assert summary == {
        "face": "valid fresh evidence used",
        "behavior": "stale; excluded from live inference",
        "audio": "missing",
    }

    stopped = live_modality_summary(
        "stopped",
        results={"face": object(), "behavior": object(), "audio": object()},
        freshness={"face": True, "behavior": True, "audio": True},
        available_modalities={"face": True, "behavior": True, "audio": True},
    )
    assert set(stopped.values()) == {"cleared when monitoring stopped"}


def test_sensor_display_distinguishes_running_unavailable_stale_and_stopped():
    assert live_sensor_status("running", 9.5, "running", now=10.0) == "RUNNING"
    assert live_sensor_status("unavailable", None, "running", now=10.0) == "UNAVAILABLE"
    assert live_sensor_status("running", 5.0, "running", now=10.0) == "STALE"
    assert live_sensor_status("running", 9.5, "stopped", now=10.0) == "STOPPED"
    assert live_sensor_status("running", None, "running", now=10.0) == "RUNNING"
    assert live_sensor_status("running", 5.0, "stopping", now=10.0) == "STOPPED"


def test_app_passes_current_monotonic_time_using_supported_keyword():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )
    assert "live_sensor_status(" in source
    assert "now=sensor_now" in source


def test_app_renders_rule_cues_without_loading_or_mislabeling_model_scores():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )
    assert 'if status == "running":' in source
    assert "model_score=model_score" not in source
    assert "Selected rule combination mode:" in source
    assert "movement_only mode" in source
    assert "DETECTOR RESULT: POTENTIAL DISTRESS CUE" in source
    assert "DETECTOR RESULT: NO SUFFICIENT DISTRESS EVIDENCE" in source
    assert "DETECTOR RESULT: UNKNOWN" in source
    assert "Previous monitoring-panel render time:" in source
    assert "Worker stage timings are in-memory aggregates." in source
    assert 'detection_mode == "rule_based"' in source
    assert "inference.potential_distress_score" in source
    assert "f\"Prediction: **{inference.predicted_label or 'Unavailable'}**\"" in source
    assert "alert.state == \"potential_distress\"" in source
    assert "Alert-manager smoothed potential-distress score:" in source
    assert "raw potential-distress " in source
    assert "raw_probability_threshold" in source
    assert "is display-only." in source
    assert "activation_qualified" in source
    assert "alert.smoothed_score" in source
    assert "Live classifier probability: **Unavailable**" in source
    assert "Classifier probability" in source
    assert "Fusion inference status: **{fusion_status}**" in source
    assert "live_modalities = live_modality_summary(" in source
    assert "LIVE RULE CUE ALERT — POTENTIAL DISTRESS" in source
    assert "alert.activated_at_utc" in source
    assert "@st.fragment(run_every=LIVE_DASHBOARD_REFRESH_SECONDS)" in source
    assert "1.5" in source
    assert "inference_history_csv(history)" in source
    assert "process_monitoring_inference" in source
    assert "Run simulated distress alert" in source
    assert "SIMULATED DEMO ALERT" in source
    assert "Report SIMULATED missed event" in source
    assert "Dismiss / Reset SIMULATED alert" in source


def test_monitoring_panel_initializes_inference_context_before_branch_reads():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    panel = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "render_monitoring_panel"
    )

    for variable, expected_value in (("inference", None), ("rule_based", False)):
        initialization = next(
            node
            for node in panel.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == variable
                for target in node.targets
            )
        )
        assert isinstance(initialization.value, ast.Constant)
        assert initialization.value.value is expected_value

        first_read = min(
            node.lineno
            for node in ast.walk(panel)
            if isinstance(node, ast.Name)
            and node.id == variable
            and isinstance(node.ctx, ast.Load)
        )
        assert initialization.lineno < first_read


def test_device_diagnostics_refresh_with_monitoring_fragment():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )
    fragment_start = source.index(
        "@st.fragment(run_every=LIVE_DASHBOARD_REFRESH_SECONDS)"
    )
    fragment_end = source.index("\nrender_monitoring_panel()", fragment_start)
    fragment_source = source[fragment_start:fragment_end]

    assert '"Test webcam"' in fragment_source
    assert '"Test microphone"' in fragment_source
    assert "disabled=device_tests_disabled" in fragment_source
    assert fragment_source.index("monitoring_manager_for_tests =") < fragment_source.index(
        "disabled=device_tests_disabled"
    )
    assert source.count("disabled=device_tests_disabled") >= 5
    assert source.count('st.rerun(scope="app")') == 2
