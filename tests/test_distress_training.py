import json
from types import SimpleNamespace

import pytest

from fusion import build_fusion_evidence, evidence_to_feature_record
from fusion.schema import FEATURE_SCHEMA_VERSION, FeatureRecord, feature_order
from training import (
    evidence_sufficient,
    infer_distress_status,
    load_distress_model,
    train_distress_logistic_model,
)
from training.distress_logistic import _split_records_by_group
from training.generate_acted_demo_records import (
    LABEL_PROFILES,
    generate_movement_demo_scenarios,
    generate_records,
    write_records,
)
from training.train_distress_model import _load_jsonl, main as train_cli


def _training_record(scenario, value):
    names = feature_order()
    values = {name: None for name in names}
    validities = {name: False for name in names}
    for name in names[:4]:
        values[name] = value
        validities[name] = True
    return FeatureRecord(
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=names,
        feature_values=values,
        feature_validity=validities,
        unavailable_reasons={},
        modality_status={"face": "ok", "behavior": "waiting", "audio": "waiting"},
        modality_messages={"face": "", "behavior": "", "audio": ""},
        scenario_label=scenario,
        provenance={"scenario_group": scenario},
        record_kind="labeled_demo",
    )


def _train(model_path):
    samples = [
        _training_record("distress-a", 1.0),
        _training_record("distress-b", 0.9),
        _training_record("baseline-a", 0.0),
        _training_record("baseline-b", 0.1),
    ]
    return train_distress_logistic_model(
        samples,
        label_lookup={
            "distress-a": "potential_distress",
            "distress-b": "potential_distress",
            "baseline-a": "no_sufficient_distress_evidence",
            "baseline-b": "no_sufficient_distress_evidence",
        },
        use_group_split=False,
        model_path=model_path,
    )


def test_training_requires_explicit_labels_and_rejects_synthetic_by_default(tmp_path):
    record = _training_record("distress-a", 1.0)
    with pytest.raises(ValueError, match="explicit ground-truth"):
        train_distress_logistic_model([record], use_group_split=False)

    acted_payload = record.to_dict()
    acted_payload["record_kind"] = "synthetic_acted"
    acted_payload["is_acted_synthetic"] = True
    acted = FeatureRecord.from_dict(acted_payload)
    with pytest.raises(ValueError, match="allow_acted_synthetic"):
        train_distress_logistic_model(
            [acted, _training_record("baseline-a", 0.0)],
            label_lookup={
                "distress-a": "potential_distress",
                "baseline-a": "no_sufficient_distress_evidence",
            },
            use_group_split=False,
            model_path=tmp_path / "must-not-exist.joblib",
        )


def test_train_save_and_load_validate_metadata(tmp_path):
    path = tmp_path / "pipeline.joblib"
    trained = _train(path)
    loaded = load_distress_model(path, strict=True)

    assert loaded is not None
    assert loaded.metadata["schema_version"] == FEATURE_SCHEMA_VERSION
    assert loaded.metadata["feature_order"] == list(feature_order())
    assert loaded.metadata["training_record_counts_by_class"] == {
        "potential_distress": 2,
        "no_sufficient_distress_evidence": 2,
    }
    test_record = _training_record("live-sample", 1.0)
    before = infer_distress_status(test_record, model=trained)
    after = infer_distress_status(test_record, model=loaded)
    assert before.status == after.status
    assert before.model_score == after.model_score
    assert before.potential_distress_score == after.potential_distress_score
    assert trained.target_label_mapping == {
        "potential_distress": 1,
        "no_sufficient_distress_evidence": 0,
    }


def test_model_score_is_the_probability_for_the_predicted_class(tmp_path):
    from training.distress_logistic import _record_to_feature_vector

    model = _train(tmp_path / "pipeline.joblib")
    record = _training_record("live-sample", 1.0)
    for name in ("behavior.movement_energy", "behavior.sudden_movement"):
        record.feature_values[name] = 1.0
        record.feature_validity[name] = True

    result = infer_distress_status(record, model=model)
    probabilities = model.pipeline.predict_proba(
        _record_to_feature_vector(record)
    )[0]
    predicted_index = next(
        index
        for index, label in model.target_label_mapping_inverse.items()
        if label == result.predicted_label
    )

    assert result.predicted_label is not None
    assert result.model_score == probabilities[predicted_index]
    distress_index = next(
        index
        for index, class_id in enumerate(
            model.pipeline.named_steps["classifier"].classes_
        )
        if int(class_id) == model.target_label_mapping["potential_distress"]
    )
    assert result.potential_distress_score == probabilities[distress_index]
    assert result.potential_distress_probability == probabilities[distress_index]


def test_jsonl_cli_trains_and_saves_the_default_format(tmp_path):
    records_path = tmp_path / "labeled.jsonl"
    labels = (
        ("potential_distress", "distress-a", 1.0),
        ("potential_distress", "distress-b", 0.9),
        ("no_sufficient_distress_evidence", "baseline-a", 0.0),
        ("no_sufficient_distress_evidence", "baseline-b", 0.1),
    )
    with records_path.open("w", encoding="utf-8") as target:
        for label, scenario, value in labels:
            payload = _training_record(scenario, value).to_dict()
            payload["label"] = label
            target.write(json.dumps(payload) + "\n")

    model_path = tmp_path / "pipeline.joblib"
    assert train_cli(
        ["--records", str(records_path), "--model-path", str(model_path)]
    ) == 0
    assert model_path.is_file()
    assert model_path.with_suffix(".metadata.json").is_file()
    assert load_distress_model(model_path, strict=True) is not None


def test_loader_rejects_missing_or_incompatible_metadata(tmp_path):
    import joblib
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    artifact = tmp_path / "pipeline.joblib"
    pipeline = Pipeline([("classifier", DummyClassifier(strategy="prior"))])
    pipeline.fit([[0], [1]], [0, 1])
    joblib.dump(pipeline, artifact)
    with pytest.raises(ValueError, match="metadata file is missing"):
        load_distress_model(artifact, strict=True)

    _train(artifact)
    metadata_path = artifact.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = "obsolete"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible schema version"):
        load_distress_model(artifact, strict=True)

    metadata["schema_version"] = FEATURE_SCHEMA_VERSION
    metadata["feature_order"] = list(reversed(feature_order()))
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="feature order"):
        load_distress_model(artifact, strict=True)


def test_inference_without_a_model_is_unknown(tmp_path):
    record = _training_record("live-sample", 1.0)
    from training import infer_distress_status

    result = infer_distress_status(
        record,
        model_path=tmp_path / "missing-pipeline.joblib",
        strict=False,
    )
    assert result.status == "unknown"
    assert result.predicted_label is None
    assert result.model_score is None
    assert result.potential_distress_score is None
    assert "No trained" in result.reason


def test_default_evidence_policy_rejects_single_modality_records(tmp_path):
    names = feature_order()
    values = {name: None for name in names}
    validity = {name: False for name in names}
    for name in [
        "audio.rms_loudness",
        "audio.spectral_centroid_hz",
        "audio.spectral_bandwidth_hz",
        "audio.spectral_flatness",
        "audio.onset_score",
    ]:
        values[name] = 0.25
        validity[name] = True
    record = FeatureRecord(
        schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=names,
        feature_values=values,
        feature_validity=validity,
        unavailable_reasons={name: "Unavailable" for name in names if not validity[name]},
        modality_status={"face": "missing", "behavior": "missing", "audio": "ok"},
        modality_messages={"face": "missing", "behavior": "missing", "audio": "ok"},
        scenario_label="audio-only-demo",
        provenance={"scenario_group": "audio-only-demo"},
        record_kind="labeled_demo",
    )

    sufficient, reason = evidence_sufficient(record)

    assert sufficient is False
    assert reason is not None
    assert "modality" in reason.lower()
    assert record.available_modalities == {
        "face": False,
        "behavior": False,
        "audio": True,
    }
    assert all(
        value is None and not record.feature_validity[name]
        for name, value in record.feature_values.items()
        if not name.startswith("audio.")
    )

    inference = infer_distress_status(
        record,
        model=_train(tmp_path / "pipeline.joblib"),
    )
    assert inference.status == "unknown"
    assert inference.model_score is None
    assert "modality" in inference.reason.lower()


def test_acted_demo_records_are_reproducible_valid_and_group_separated(tmp_path):
    records = generate_records(seed=91, groups_per_class=6, records_per_group=2)
    repeated = generate_records(seed=91, groups_per_class=6, records_per_group=2)
    assert records == repeated
    assert len(records) == 24
    assert all(record["is_acted_synthetic"] is True for record in records)
    assert all(record["record_kind"] == "synthetic_acted" for record in records)
    assert {record["provenance"]["seed"] for record in records} == {91}
    assert {record["label"] for record in records} == {
        "potential_distress",
        "no_sufficient_distress_evidence",
    }
    assert all(record["feature_names"] == list(feature_order()) for record in records)
    assert all(
        record["feature_values"]["behavior.repetition_change"] is None
        and record["feature_validity"]["behavior.repetition_change"] is False
        for record in records
    )
    assert all(
        record["feature_values"][name] is None
        and record["feature_validity"][name] is False
        for record in records
        for name in feature_order()
        if not record["feature_validity"][name]
    )

    path = tmp_path / "acted.jsonl"
    write_records(path, records)
    loaded_records = _load_jsonl(path)
    assert loaded_records == records
    validated = [FeatureRecord.from_dict(record) for record in loaded_records]
    train_records, test_records = _split_records_by_group(validated, random_state=42)
    train_groups = {record.provenance["scenario_group"] for record in train_records}
    test_groups = {record.provenance["scenario_group"] for record in test_records}
    assert train_groups.isdisjoint(test_groups)
    assert {record.scenario_label.rsplit(":", 1)[0] for record in train_records} == {
        record.provenance["scenario_group"] for record in train_records
    }
    assert {record.scenario_label.rsplit(":", 1)[0] for record in test_records} == {
        record.provenance["scenario_group"] for record in test_records
    }
    assert {
        record["label"] for record in records if record["provenance"]["scenario_group"] in train_groups
    } == {"potential_distress", "no_sufficient_distress_evidence"}
    assert all(
        record.scenario_label.startswith(f"{record.provenance['scenario_group']}:")
        for record in validated
    )


def test_behavior_v2_features_map_in_canonical_training_order():
    movement_names = (
        "wrist_average_speed",
        "wrist_peak_speed",
        "wrist_peak_acceleration",
        "wrist_movement_amplitude",
        "wrist_repetition_rate_hz",
        "wrist_elevated_movement_proportion",
        "left_wrist_visibility",
        "right_wrist_visibility",
    )
    behavior_values = {name: float(index + 1) for index, name in enumerate(movement_names)}
    snapshot = {
        "latest_behavior_result": SimpleNamespace(
            feature_values=behavior_values,
            feature_validity={name: True for name in movement_names},
            feature_names=movement_names,
            unavailable_reasons={},
            status="ok",
            status_message="Synthetic feature-order fixture.",
        )
    }

    record = evidence_to_feature_record(build_fusion_evidence(snapshot))

    assert FEATURE_SCHEMA_VERSION == "phase3-feature-schema-v2"
    assert record.feature_names == feature_order()
    for index, name in enumerate(movement_names):
        feature_name = f"behavior.{name}"
        assert record.feature_values[feature_name] == float(index + 1)
        assert record.feature_validity[feature_name] is True
        assert record.feature_vector[record.feature_names.index(feature_name)] == float(index + 1)


def test_labeled_movement_demo_scenarios_are_repeatable_and_not_movement_only_distress():
    scenarios = generate_movement_demo_scenarios(seed=20261015)
    repeated = generate_movement_demo_scenarios(seed=20261015)

    assert scenarios == repeated
    assert len(scenarios) == 7
    labels_by_scenario = {
        record["provenance"]["scenario_name"]: record["label"]
        for record in scenarios
    }
    assert labels_by_scenario["sustained_repetitive_hand_movement"] == (
        "no_sufficient_distress_evidence"
    )
    assert labels_by_scenario["sustained_vigorous_hand_arm_movement"] == (
        "no_sufficient_distress_evidence"
    )
    assert labels_by_scenario["high_energy_audio_little_movement"] == "potential_distress"
    assert labels_by_scenario["combined_movement_and_high_energy_audio"] == (
        "potential_distress"
    )
    assert all(record["is_acted_synthetic"] for record in scenarios)
    assert all(
        record["feature_validity"]["behavior.wrist_repetition_rate_hz"]
        for record in scenarios
    )
    for record in scenarios:
        assert record["feature_names"] == list(feature_order())
        assert record["provenance"]["label_rationale"]
        assert record["record_kind"] == "synthetic_acted"
        assert FeatureRecord.from_dict(record).feature_names == feature_order()
    combined = next(
        record
        for record in scenarios
        if record["provenance"]["scenario_name"]
        == "combined_movement_and_high_energy_audio"
    )
    assert "movement is context only" in combined["provenance"]["label_rationale"]
    assert all(
        LABEL_PROFILES["potential_distress"]["behavior"][name]
        == LABEL_PROFILES["no_sufficient_distress_evidence"]["behavior"][name]
        for name in LABEL_PROFILES["potential_distress"]["behavior"]
    )


def test_demo_training_metadata_records_v2_order_provenance_and_group_split(tmp_path):
    records = generate_records(seed=17, groups_per_class=6, records_per_group=2)
    records.extend(generate_movement_demo_scenarios(seed=17))
    records_path = tmp_path / "schema-v2-demo.jsonl"
    write_records(records_path, records)
    model_path = tmp_path / "separate-v2-demo" / "pipeline.joblib"

    assert train_cli(
        [
            "--records",
            str(records_path),
            "--model-path",
            str(model_path),
            "--allow-acted-synthetic",
        ]
    ) == 0

    metadata = json.loads(
        model_path.with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["schema_version"] == "phase3-feature-schema-v2"
    assert metadata["feature_order"] == list(feature_order())
    assert metadata["contains_acted_or_synthetic_records"] is True
    assert metadata["training_provenance"]["all_records_synthetic_or_acted"] is True
    assert metadata["training_provenance"]["scenario_groups_disjoint"] is True
    assert metadata["training_provenance"]["scenario_group_counts_by_split"]["test"] > 0
    assert "pipeline demonstration only" in metadata["limitations"]


def test_live_inference_reports_incompatible_v1_model_instead_of_generic_unknown(tmp_path):
    model_path = tmp_path / "pipeline.joblib"
    _train(model_path)
    metadata_path = model_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_version"] = "phase3-feature-schema-v1"
    metadata["feature_order"] = [
        "face.expression_angry",
        "face.expression_fear",
        "face.expression_happy",
        "face.expression_neutral",
        "face.expression_sad",
        "face.expression_surprise",
        "face.negative_affect_mean_angry_fear_sad",
        "face.face_quality_score",
        "behavior.movement_energy",
        "behavior.sudden_movement",
        "behavior.movement_change_from_baseline",
        "behavior.shoulder_alignment",
        "behavior.shoulder_asymmetry",
        "behavior.left_upper_arm_angle_deg",
        "behavior.right_upper_arm_angle_deg",
        "behavior.left_forearm_angle_deg",
        "behavior.right_forearm_angle_deg",
        "behavior.head_offset_x",
        "behavior.head_offset_y",
        "behavior.head_orientation_deg",
        "behavior.repetition_change",
        "audio.rms_loudness",
        "audio.spectral_centroid_hz",
        "audio.spectral_bandwidth_hz",
        "audio.spectral_flatness",
        "audio.onset_score",
        "audio.baseline_relative_rms",
        "audio.audio_contribution",
    ]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = infer_distress_status(
        _training_record("live-v2-record", 0.5),
        model_path=model_path,
        strict=False,
    )

    assert result.status == "unknown"
    assert result.predicted_label is None
    assert "phase3-feature-schema-v1" in result.reason
    assert "phase3-feature-schema-v2" in result.reason


def test_controlled_synthetic_training_can_use_movement_feature_in_fusion(tmp_path):
    samples = []
    for index, (label, movement_speed) in enumerate(
        (
            ("potential_distress", 0.85),
            ("potential_distress", 0.95),
            ("potential_distress", 1.05),
            ("potential_distress", 1.15),
            ("no_sufficient_distress_evidence", 0.00),
            ("no_sufficient_distress_evidence", 0.03),
            ("no_sufficient_distress_evidence", 0.06),
            ("no_sufficient_distress_evidence", 0.09),
        )
    ):
        values = {name: 0.0 for name in feature_order()}
        validity = {name: True for name in feature_order()}
        values["behavior.wrist_average_speed"] = movement_speed
        record = FeatureRecord(
            schema_version=FEATURE_SCHEMA_VERSION,
            feature_names=feature_order(),
            feature_values=values,
            feature_validity=validity,
            unavailable_reasons={},
            modality_status={"face": "synthetic", "behavior": "synthetic", "audio": "waiting"},
            modality_messages={"face": "", "behavior": "", "audio": ""},
            scenario_label=f"controlled-movement-{index}",
            provenance={"scenario_group": f"controlled-movement-{index}"},
            record_kind="synthetic_acted",
            is_acted_synthetic=True,
            notes="Test-only toy training data; no predictive validity.",
        )
        payload = record.to_dict()
        payload["label"] = label
        samples.append(payload)
    model = train_distress_logistic_model(
        samples,
        use_group_split=False,
        allow_acted_synthetic=True,
        model_path=tmp_path / "movement-model.joblib",
    )

    def movement_record(speed):
        values = {name: 0.0 for name in feature_order()}
        validity = {name: True for name in feature_order()}
        values["behavior.wrist_average_speed"] = speed
        return FeatureRecord(
            schema_version=FEATURE_SCHEMA_VERSION,
            feature_names=feature_order(),
            feature_values=values,
            feature_validity=validity,
            unavailable_reasons={},
            modality_status={"face": "ok", "behavior": "ok", "audio": "waiting"},
            modality_messages={"face": "", "behavior": "", "audio": ""},
        )

    low = infer_distress_status(movement_record(0.02), model=model)
    high = infer_distress_status(movement_record(1.0), model=model)

    assert low.status == "no_sufficient_distress_evidence"
    assert high.status == "potential_distress"
    assert high.potential_distress_probability > low.potential_distress_probability


def test_live_branch_outputs_convert_to_complete_canonical_schema():
    snapshot = {
        "latest_face_result": SimpleNamespace(
            expression_probabilities={
                "angry": 0.3,
                "disgust": None,
                "fear": 0.2,
                "happy": 0.1,
                "neutral": 0.2,
                "sad": 0.1,
                "surprise": 0.1,
            },
            class_validity={
                "angry": True,
                "disgust": False,
                "fear": True,
                "happy": True,
                "neutral": True,
                "sad": True,
                "surprise": True,
            },
            feature_vector_order=(
                "angry",
                "disgust",
                "fear",
                "happy",
                "neutral",
                "sad",
                "surprise",
            ),
            negative_affect_feature=0.2,
            face_quality_score=0.9,
            face_quality_status="usable",
            status="ok",
            status_message="Face ready",
        ),
        "latest_behavior_result": SimpleNamespace(
            feature_values={"movement_energy": 0.0},
            feature_validity={"movement_energy": True},
            feature_names=("movement_energy",),
            unavailable_reasons={},
            status="partial",
            status_message="Behavior partial",
        ),
        "latest_audio_result": SimpleNamespace(
            feature_values={"rms_loudness": 0.0},
            feature_validity={"rms_loudness": True},
            feature_names=("rms_loudness",),
            unavailable_reasons={},
            status="warming_up",
            status_message="Audio baseline is warming up.",
        ),
    }

    evidence = build_fusion_evidence(snapshot)
    record = evidence_to_feature_record(evidence)

    assert record.feature_names == feature_order()
    assert record.feature_values["face.expression_angry"] == 0.3
    assert record.feature_values["face.negative_affect_mean_angry_fear_sad"] == 0.2
    assert record.feature_values["face.face_quality_score"] == 0.9
    assert record.feature_values["behavior.movement_energy"] == 0.0
    assert record.feature_validity["behavior.movement_energy"] is True
    assert record.feature_values["audio.rms_loudness"] == 0.0
    assert record.feature_validity["audio.rms_loudness"] is True
    assert record.feature_values["audio.spectral_centroid_hz"] is None
    assert record.feature_validity["audio.spectral_centroid_hz"] is False
