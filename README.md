# Distress Companion — Phase 3 Foundation

The project combines local face, upper-body behavior, and audio feature branches
with bounded continuous monitoring. Live cues are currently evaluated by a
configurable rule-based detector over fresh audio and movement features. The
versioned v2 shared evidence schema, optional logistic-regression training and
inference utilities, engineering alert-state management, and local caretaker
feedback persistence remain available; live monitoring does not load a
classifier.

Live alerts indicate configured potential-distress cues, not a diagnosis or
definitive distress classification. Defaults are tunable demonstration settings,
not clinically validated thresholds. Missing, invalid, or stale evidence is
unavailable/unknown and is never silently treated as calm. Future federated
learning is a requirement for the overall multimodal edge system; federated
logic is not implemented.

## Project structure

```text
distress-companion/
├── app.py
├── alert_management.py
├── config.py
├── microphone.py
├── monitoring.py
├── monitoring_history.py
├── rule_based_detection.py
├── requirements.txt
├── simulated_alert_test.py
├── webcam.py
├── README.md
├── branches/
│   ├── __init__.py
│   ├── behavior_branch.py
│   ├── audio_branch.py
│   ├── face_branch.py
│   ├── face_model.py
│   └── interfaces.py
├── training/
│   ├── __init__.py
│   ├── distress_logistic.py
│   ├── generate_acted_demo_records.py
│   ├── train_distress_model.py
│   └── train_face_model.py
├── fusion/
│   ├── __init__.py
│   ├── fusion.py
│   └── schema.py
├── feedback/
│   ├── __init__.py
│   └── store.py
├── federated/
│   └── __init__.py
├── data/
│   ├── raw/
│   │   └── autism_emotion_dataset/
│   ├── acted/
│   │   └── phase3_demo_records.jsonl  # generated, synthetic demonstration only
│   └── processed/
├── models/
│   ├── pose_landmarker_lite.task
│   └── face_expression_mobilenet_v3_small/
├── tests/
│   ├── __init__.py
│   ├── test_simulated_alert_test.py
│   ├── test_monitoring_history.py
│   ├── test_face_branch.py
│   ├── test_face_training.py
│   ├── test_behavior_branch.py
│   ├── test_audio_branch.py
│   ├── test_monitoring.py
│   ├── test_rule_based_detection.py
│   ├── test_microphone.py
│   └── test_webcam.py
└── reports/
```

## Model and preprocessing

### Live rule-based cue detection

`rule_based_detection.py` evaluates fresh schema-v2 feature values and their
validity flags. It uses audio RMS/onset features and the behavior branch's
movement-energy, sudden-movement, wrist-speed, acceleration, amplitude,
repetition-rate, and elevated-movement features. Face evidence remains available
in the dashboard, history, and feedback window, but is not currently a rule cue.

The combination mode is set by `RULE_COMBINATION_MODE` in `config.py`:
`any`, `audio_only`, `movement_only`, or `audio_and_movement`. The current
project setting is `movement_only`; audio is captured and evaluated for display
but cannot trigger a live alert in that mode.
Initial demonstration settings require audio RMS >= 0.08 for 2 seconds for a
sustained audio cue; an onset score >= 0.35 or an RMS rise crossing >= 0.25
triggers a brief audio-peak cue. Repetitive movement requires repetition rate
>= 0.5 Hz, amplitude >= 0.06 shoulder widths, elevated-movement proportion
>= 0.5, and 2 seconds of persistence. Vigorous movement requires movement energy
>= 0.8 and peak wrist speed >= 1.0 shoulder widths/second (or acceleration
>= 2.0) for 1 second. A branch sudden-movement indicator >= 0.75 is an
immediate cue. The detector uses a 4-second history and gaps longer than
1.5 seconds break persistence. Values can be changed in `config.py`; these
starting settings are not calibrated or clinically validated. The dashboard
shows the individual audio/movement cue states and reasons. Configured regional
repetition cues (head, arms, torso, legs, and whole body) are also evaluated by
the live detector. Regional intensity, speed, and acceleration are displayed as
diagnostics and do not independently trigger a rule.

Repetitive movement may be self-regulatory; vigorous movement and loud sounds
also have harmless causes. No individual cue proves distress. `unknown` means
the configured mode cannot be evaluated from fresh, valid evidence; a missing
modality is never interpreted as calm. In `any` mode either available cue may
trigger, while `audio_and_movement` requires both. Single-modality modes are
explicit demonstration settings. Rule results contain no classifier probability.

### Schema, optional model utilities, alerts, and feedback

`fusion/schema.py` defines the canonical `phase3-feature-schema-v2` feature
order. Each feature has a separate validity flag; a real numeric zero remains
distinct from a missing value. `training/distress_logistic.py` and its
schema-compatible inference functions remain for offline/model experiments; the
live dashboard does not load or train that classifier. `alert_management.py`
keeps its existing freshness, duplicate, lifecycle, release, and cooldown
safeguards. The legacy model path still uses its raw-probability threshold
(default 0.75), separate 0.55 release threshold, and display-only smoothed score.
The live rule path instead requires an explicit rule trigger and a sufficient
fresh-evidence gate; it does not synthesize a probability or change model
smoothing. A rule alert activates immediately and releases after two valid,
unique results without an active configured cue, then enters the existing
30-second cooldown. Missing/stale rule evidence follows the alert manager's
existing unknown-state behavior. These are engineering settings, not clinical
criteria.

The behavior branch also retains only timestamped, shoulder-center-relative
wrist coordinates in a bounded **4-second** rolling window. It reports wrist
speed/energy, robust peak speed and acceleration, movement amplitude, an
oscillation-rate estimate, elevated-movement proportion, and wrist visibility.
Displacements smaller than the configurable 0.012 shoulder-width jitter floor
are ignored; gaps longer than 1.5 seconds reset the temporal trace. The window,
gap, jitter floor, and movement thresholds are engineering starting settings,
not clinically calibrated values. Missing or low-visibility wrists make
movement unavailable rather than being imputed as stillness. These are
behavioral measurements only: repetitive or vigorous movement does not itself
mean distress.

The same pose pass also tracks head, left/right arms, torso, left/right legs,
and whole-body movement from whichever selected landmarks pass the existing
0.50 visibility/presence confidence gate. Coordinates are normalized by shoulder
width, with hip width or shoulder-to-hip length as scale fallbacks; region
movement is relative to a body anchor, while whole-body translation is tracked
relative to the camera frame and normalized by scale. Each region reports
amplitude, movement energy, peak speed, peak acceleration, repetition rate, and
elevated-movement proportion with independent validity and warm-up reasons.
Repetition requires at least 6 observations over 1.5 seconds (configurable with
`BEHAVIOR_REGION_MIN_OBSERVATIONS` and `BEHAVIOR_REGION_MIN_SPAN_SECONDS`), and
the existing 4-second rolling window and 1.5-second gap reset still apply.
Missing wrists no longer invalidate other visible regions. Region-specific
repetition cues are enabled by `RULE_MOVEMENT_REGIONS`; they use the configured
movement repetition rate, amplitude, elevated-proportion, and persistence
settings. These remain demonstration cues, not diagnoses.

The **Live Pose Analysis** section reads only the latest pose result and
temporary overlay already published by the monitoring worker. Enable **live
pose analysis overlay and diagnostics** to see the annotated frame, selected
landmark skeleton, wrist confidence scores, and rejection reasons. The view
also reports region availability, intensity, repetition, acceleration, and
current configured cue states. It does not open another camera, run another
MediaPipe instance, or save raw media. The overlay is held only in the latest
in-memory result and is cleared by the existing monitoring lifecycle.

Multi-region measurements are passed directly from the fresh behavior result
to the rule detector. They do not alter the canonical Phase 3 schema-v2 feature
order or classifier/training records; exported feature/history formats remain
unchanged. The live rule path still requires fresh evidence and preserves the
configured combination modes, cue thresholds, alert lifecycle, and missing
evidence behavior.

Caretaker confirmations, rejections, and missed-event reports are stored in a
local SQLite database at `~/.distress_companion/feedback.sqlite3` (Windows:
`%USERPROFILE%\\.distress_companion\\feedback.sqlite3`). Feedback includes only
an alert reference when applicable and numerical feature values with validity
flags. It stores no names, personal identifiers, images, audio, or video.
Idempotency keys prevent duplicate submissions across reruns. Feedback does not
automatically retrain a model.

To train the multimodal fusion model, first prepare a UTF-8 JSONL file containing
one complete serialized `FeatureRecord` per line plus an explicit ground-truth
`label` (`potential_distress` or `no_sufficient_distress_evidence`). Include a
distinct `provenance.scenario_group` for each source scenario so related records
remain together in the leakage-safe split. `scenario_label` describes the
scenario; it is not automatically treated as ground truth. The project currently
does not include real labeled multimodal records, so it does not ship a validated
fusion model. To create the deterministic synthetic pipeline demonstration and
train it, run these commands:

```powershell
.\.venv\Scripts\python.exe -m training.generate_acted_demo_records --output data\acted\phase3_schema_v2_demo_records.jsonl --include-movement-scenarios
.\.venv\Scripts\python.exe -m training.train_distress_model --records data\acted\phase3_schema_v2_demo_records.jsonl --model-path models\distress_logistic\phase3_schema_v2_demo\pipeline.joblib --allow-acted-synthetic
```

These human-authored acted feature profiles exercise only the schema and
training/inference plumbing; they are not sensor recordings, evidence of
accuracy, or a defensible basis for selecting an alert threshold. The
repetitive-only and vigorous-only profiles are labeled
`no_sufficient_distress_evidence`, while the audio and combined profiles are
synthetically labeled `potential_distress` only to exercise the named
high-energy-audio demonstration scenarios. Their per-record provenance contains
the synthetic label rationale. Movement values in the ordinary generated
positive-class profiles overlap the baseline profile; in the combined
movement/audio scenario, audio is the explicitly designated synthetic cue and
movement is context only. These labels are author-created test fixtures, not
empirically justified distress evidence. Prefer reviewed, consented, explicitly
labeled real records for model development.

The generator writes explicitly labeled records with complete canonical feature
ordering, validity flags, source-group provenance, and the `synthetic_acted`
marker. Each independent scenario group stays entirely on one side of the
trainer's reproducible group split. The hand-authored toy profiles deliberately
exercise serialization, preprocessing, loading, and inference; they are not real
data and their model outputs or held-out results are not meaningful performance
estimates. For eventual real training, use reviewed, consented, explicitly labeled
multimodal records and omit `--allow-acted-synthetic`:

```powershell
.\.venv\Scripts\python.exe -m training.train_distress_model --records data\processed\labeled_fusion_records.jsonl
```

This writes `models\distress_logistic\pipeline.joblib` and its required companion
metadata file `pipeline.metadata.json`. These artifacts are for explicit
offline/model experiments; live monitoring uses the rule detector and does not
load them. The app never trains automatically. The command rejects acted/synthetic
records unless `--allow-acted-synthetic` is explicitly supplied, and then marks
the saved model as a pipeline demonstration rather than validated detection.
Existing schema-v1 fusion artifacts are incompatible with schema v2 and must be
retrained together with their metadata before schema-v2 offline inference can use them.
No old artifact is silently migrated or supplied movement values in an old
feature order. The two commands above save under a separate
`phase3_schema_v2_demo` directory and do not overwrite the current model. The
generated synthetic dataset is not a responsible training source for deployment;
use it only to smoke-test schema-v2 serialization, group splitting, and artifact
compatibility. Training metadata records the canonical feature order, schema
version, source counts, synthetic/acted status, and train/test scenario-group
split summary.
Run the dashboard with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Continuous monitoring

**Test webcam** and **Test microphone** remain short, one-shot diagnostics. Those
controls are disabled while monitoring owns the devices. **Start monitoring**
creates one worker, which reuses a camera session and a microphone input stream.
Microphone callbacks write only into a bounded in-memory ring buffer; short
windows are passed to the audio branch and are not recorded to disk. Face,
behavior, and audio feature extraction, fresh evidence conversion, rule
evaluation, and alert-state updates run outside Streamlit rendering. The dashboard fragment refreshes about every
1.5 seconds, independently of the worker's sensor sampling intervals. Set
`LIVE_CAMERA_INTERVAL_SECONDS`, `LIVE_AUDIO_INTERVAL_SECONDS`, and
`LIVE_DASHBOARD_REFRESH_SECONDS` in `config.py` to adjust those rates. The
monitoring panel reports aggregate capture/branch/pose-region/audio/rule timings,
observed camera inter-capture rate, and the previous dashboard-render duration.
These measurements are in-memory only and retain no media.

Sensor statuses are presented as RUNNING, UNAVAILABLE, STALE, or STOPPED. Live
inference older than the existing three-second freshness limit is not displayed
as current. The in-memory inference history survives Streamlit reruns and
monitoring stops for the lifetime of the session, and can be downloaded as CSV.
It contains timestamps, labels, probabilities, scores, states, and modality
summaries only; it stores no frames or audio.

The **Temporary alert-decision diagnostics** section separately exports
`live_alert_diagnostics_v3.csv` with the `live-alert-diagnostic-v3` schema.
It includes the deduplication key, UTC inference time, monotonic source times
and ages, per-modality freshness/availability, evidence sufficiency, detection
mode, cue indicators/reason, alert-state transitions, and update decision.
Probability and probability thresholds are blank for rule-based results so a
cue is not mislabeled as a model score. The inference-history CSV retains its existing columns. In this live app its
rows represent rule-based results (not classifier predictions); rule cue
summaries and the explicit detection mode are available in the dashboard and
the separate diagnostic CSV.
Monotonic timestamps and ages are meaningful only within the same monitoring
session. This in-memory diagnostic history remains available after Stop for the
session lifetime. It contains no raw media or personal identifiers and is separate
from the inference-history CSV; exporting or recording it does not affect alerting.

The **Run simulated distress alert** control creates an immediate
**SIMULATED DEMO ALERT** for demonstrating the caretaker workflow. Its separate
in-memory state supports Confirm, Reject, Report Missed Event, and Dismiss/Reset.
These simulated acknowledgements are not saved to the caretaker feedback
database, and the demo alert never changes live inference, alert state, history,
or diagnostics. The separate policy-exercise helper can still be used by tests
to exercise activation, release, and cooldown behavior.

For a hardware verification, first use the one-shot device tests. Then start
monitoring and confirm that camera and microphone status stay RUNNING and that
their success timestamps continue to advance. Verify fresh feature values,
individual cue indicators, and alert state; stop and ensure both sensors become
STOPPED, live values clear, and history remains available for CSV export.
Use **Download temporary alert diagnostics CSV** to inspect the policy decision
for each generated inference result, including duplicate, stale, or insufficient
evidence outcomes.
This verifies operation on the current machine only and does not establish
clinical accuracy.

For movement-feature verification, place the camera far enough away to keep both
shoulders, elbows, and wrists visible. After starting monitoring, remain still
for at least four seconds, then try ordinary gestures, repeated hand movements,
and sustained faster arm movements separately. Check that valid speed,
amplitude, repetition, and elevated-movement features update over time. Test
audio-only while sitting still, followed by combined cues. With defaults,
sustained audio takes 2 seconds, repetitive movement takes 2 seconds, vigorous
movement takes 1 second, and an onset/peak or sudden-movement cue can be
immediate. These are tunable demonstrations, not claims about real-world
accuracy. Verify missing/occluded wrists and a disconnected/stale microphone are
shown as unavailable rather than calm. Stopping monitoring should release both
devices and clear live alert state. Mocked tests do not verify the physical
camera, MediaPipe runtime, microphone, or real-world usefulness.

Run the rule-specific tests and then the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_rule_based_detection.py
.\.venv\Scripts\python.exe -m pytest -q
```

Synthetic fixtures verify software logic only. This is a prototype, not a
clinically validated distress detector, and must not replace human judgment or
care.

The branch uses Torchvision MobileNetV3-Small initialized from
`MobileNet_V3_Small_Weights.IMAGENET1K_V1` (official weight file:
`https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth`). It freezes
the feature backbone and trains the classification head only. Inference
preprocessing uses the transform supplied by that Torchvision weight enum: RGB,
resize the shorter side to 256 with bilinear interpolation, center-crop 224×224,
rescale to [0, 1], and normalize with mean [0.485, 0.456, 0.406] and standard
deviation [0.229, 0.224, 0.225]. Training augmentation (random resized crop and
horizontal flip) is used only for training images.

The dataset classes are fixed and ordered as:
`anger`, `fear`, `joy`, `Natural`, `sadness`, `surprise`.

Display naming is explicitly mapped:

| Dataset class | Display label |
| --- | --- |
| `Natural` | `neutral` |
| `joy` | `happy` |
| `sadness` | `sad` |
| `anger` | `angry` |
| `fear` | `fear` |
| `surprise` | `surprise` |

These are naming conventions only; they do not establish that the source
dataset's class definitions match another model's definitions. There is no
`disgust` training class. The seven-label interface always marks `disgust` as
unavailable and uses `None`, not a fabricated zero, wherever an estimate is
unavailable. The face branch abstains with unknown output if the face, image
quality, local model, or inference result is unusable.

Negative affect is the simple mean
`(P(angry) + P(fear) + P(sad)) / 3`. It is a weak feature, not a threshold,
alert, or proof of distress. Softmax confidence summaries are uncertainty
descriptors, not correctness estimates.

## Dataset audit

The configured dataset root is `data/raw/autism_emotion_dataset/`. The supplied
local data is nested one level further at
`Autism emotion recogition dataset/train/` and `test/`; the training loader finds
this unique sibling split without moving or renaming any files.

An audit found 1,199 readable training images and 226 readable test images:

| Class | Train | Test |
| --- | ---: | ---: |
| anger | 200 | 40 |
| fear | 200 | 36 |
| joy | 200 | 35 |
| Natural | 200 | 40 |
| sadness | 199 | 38 |
| surprise | 200 | 37 |

The inspected files were 1,420 JPEGs and 5 PNGs, with no unreadable images.
Dimensions vary: 987 images are 224×224 and the remainder use multiple sizes.
There are exact duplicate image groups in training, including some duplicated
pixels with conflicting class labels. A 64-bit difference-hash screen also
identified near-duplicate candidates within the supplied test split and across
the train/test split. Some filenames explicitly resemble augmented variants.
The training split has 45 exact duplicate groups, including 12 groups where the
same decoded pixels occur under different class labels; the cross-split screen
found 28 train/test near-duplicate candidate pairs and no exact cross-split
duplicates.
The hashes are a screening tool, not proof of shared identity, but **the supplied
test set cannot be treated as a clean independent estimate**. The test set is
left intact and is not used for model selection. Training validation keeps exact
and close perceptual-hash groups together to reduce train/validation leakage.
The training report records the observed overlap candidates and warns about
interpretation of the test metrics.

## Windows setup

Use Python 3.11. MediaPipe 1.1.0 was selected because it publishes a Windows
x64 wheel for Python 3.11 and exposes the current Tasks Pose Landmarker API.
The branch uses `mediapipe.tasks.python.vision.PoseLandmarker` in IMAGE mode;
it does not depend on the legacy `mp.solutions.pose` API. MediaPipe requires
OpenCV-contrib, so `opencv-contrib-python` replaces `opencv-python` while
retaining the existing `cv2` APIs. From PowerShell:

```powershell
cd D:\PROJECTS\Distress_detection\distress-companion
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, PyTorch may install a CUDA-enabled or CPU-only build depending on the
package index and compatible wheels. Check actual availability with:

```powershell
python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Using CPU')"
```

For this project's tested CUDA 12.8-compatible version pair on an NVIDIA Windows
system, install the official CUDA wheels after the requirements install:

```powershell
python -m pip install --upgrade "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" `
  --index-url https://download.pytorch.org/whl/cu128
```

The inspected RTX 3050 Laptop GPU has a 12.9-capable driver; CUDA 12.8 wheels are
driver-compatible. If the CUDA installation is unavailable, CPU mode remains
supported with `--device cpu`.

If PowerShell blocks virtual-environment activation, allow it for this process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Train and evaluate

Training does not run when the dashboard starts. The defaults are 12 epochs,
batch size 32, learning rate 0.001, seed 42, 20% grouped validation, and automatic
device selection. In a clean environment the initial training run downloads the
official pretrained MobileNetV3-Small weights; it errors rather than silently
falling back to a randomly initialized backbone if those weights cannot be
obtained.

```powershell
python -m training.train_face_model
```

Example overrides (all options are also shown by `--help`):

```powershell
python -m training.train_face_model `
  --epochs 12 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --seed 42 `
  --validation-fraction 0.2 `
  --device auto
```

The configured data and output paths can be overridden with `--data-root` and
`--output-dir`. Use `--device cpu` for CPU training or `--device cuda` to require
CUDA. Validation picks the best head by validation accuracy. The untouched
supplied test split is evaluated once after training with per-class precision,
recall, F1, support, and a confusion matrix (rows=true class, columns=predicted).
Use these test results cautiously because the local audit detected likely
train/test related images.

Outputs go under
`models/face_expression_mobilenet_v3_small/`:

- `model.pt` — local checkpoint and state dictionary
- `metadata.json` — architecture, class order, preprocessing, training and data
  details
- `class_to_index.json` — saved class-index mapping
- `preprocessing.json` — inference preprocessing details
- `training_results.json` — epoch history, validation score, test metrics, and
  overlap warning

Training refuses to replace existing output files. Choose a different
`--output-dir`, or explicitly pass `--overwrite` when replacing that run.

The model is not considered trained until this command completes successfully.
If no checkpoint exists, the face branch displays an actionable command and
abstains.

## Observed training run

The local run completed 12 epochs on the RTX 3050 using the frozen
MobileNetV3-Small feature backbone. Best grouped-validation accuracy was 0.5292.
The supplied test split yielded 0.3805 accuracy; because the local audit found
28 perceptual near-duplicate train/test candidates, this is **not an independent
generalization estimate and is not an accuracy claim**.

| Test class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| anger | 0.4222 | 0.4750 | 0.4471 | 40 |
| fear | 0.3333 | 0.3056 | 0.3188 | 36 |
| joy | 0.5405 | 0.5714 | 0.5556 | 35 |
| Natural | 0.3902 | 0.4000 | 0.3951 | 40 |
| sadness | 0.1739 | 0.1053 | 0.1311 | 38 |
| surprise | 0.3404 | 0.4324 | 0.3810 | 37 |

## Tests and launch

```powershell
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

Tests use mock camera, microphone, detector, and model components plus synthetic
images. They require neither a physical camera/microphone nor a GPU or network.
The app opens at `http://localhost:8501`. It retains the separate webcam and
microphone tests; click **Test face expression** to explicitly capture one frame
and run face detection/inference, or **Test upper-body behavior** for a short
frame burst. The camera is not opened automatically and frames are not persisted.
Click **Start monitoring** to initialize fresh branch instances for that monitoring
lifecycle. A branch initialization failure is reported with its available local
diagnostic; it does not substitute a prediction. **Stop monitoring** keeps the
worker and its branches intact until any in-progress capture finishes, and reports
`stopping` rather than `stopped` if shutdown takes longer than the wait timeout.
The monitoring counters describe capture attempts and returned branch results;
they do not establish that valid inference or alert evidence was produced.
The dashboard also reports worker liveness and per-stage diagnostics for sensor
capture, branch initialization/inference, result publication, and worker exit.
Historical runtime diagnostics now include camera-open/frame metadata and per-branch
invoked, returned, and published counts; frame metadata is retained without media.
Stopping clears the latest live results to keep stopped evidence out of fusion,
but retains returned-result counts and the stop reason; a stopped view therefore
does not mean that no result was published earlier in that monitoring run.
The historical fusion diagnostic preserves the latest constructed values,
validity, modality status, and source timestamps for inspection after stopping;
it is never reused for model inference or alert updates.

## Manual face checks

1. With a visible, front-facing, well-lit face and trained weights, click
   **Test face expression**. Check the face/quality status, six finite
   probabilities summing to 1, and the `disgust` row marked unavailable.
2. With no face, check that status is unknown, probabilities are unavailable,
   and neutral is not assigned.
3. Try a small, blurred, dark, or overexposed face crop and confirm abstention
   with a reason.
4. Confirm the missing/corrupt-model tests in `test_face_branch.py` report an
   actionable setup/load error and no fabricated scores.
5. Run the existing webcam/microphone tests to confirm those controls still work.

## Phase 2B — Upper-body behavior features

The optional behavior action uses the local MediaPipe Pose Landmarker Lite task
asset at `models/pose_landmarker_lite.task`. It tracks normalized image
coordinates for the nose, shoulders, elbows, and wrists (MediaPipe landmark
indices 0 and 11–16). Landmark confidence is the minimum available visibility
and presence score; values below 0.5 are marked unavailable. Frames are captured
only after clicking **Test upper-body behavior**, using one bounded camera
session for 12 frames at roughly 0.12-second intervals. Monotonic capture times
are used for rates. Captured frames are discarded after processing. The
official model asset is available at
`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`;
if it is missing, save it at `models/pose_landmarker_lite.task`.

The stable 13-value feature vector and ordering are:

1. `movement_energy` — mean displacement of shared visible landmarks divided by
   shoulder width and elapsed seconds (shoulder-widths/second).
2. `sudden_movement` — 1 when current energy exceeds the previous baseline mean
   by at least 0.75 shoulder-widths/second; otherwise 0. It is unavailable until
   the baseline has warmed up and is not a distress signal.
3. `movement_change_from_baseline` — `(current energy - previous baseline mean) /
   max(previous baseline mean, 0.05)`; dimensionless and unavailable during
   warm-up.
4. `shoulder_alignment` — signed right-minus-left shoulder vertical offset
   divided by shoulder width.
5. `shoulder_asymmetry` — absolute shoulder alignment.
6–9. `left_upper_arm_angle_deg`, `right_upper_arm_angle_deg`,
   `left_forearm_angle_deg`, `right_forearm_angle_deg` — image-plane segment
   orientations from `atan2(dy, dx)`.
10–11. `head_offset_x`, `head_offset_y` — nose offset from shoulder midpoint
   divided by shoulder width.
12. `head_orientation_deg` — image-plane angle of that nose-to-shoulder-midpoint
   offset.
13. `repetition_change` — explicitly unavailable; repetitive movement is not
   estimated in this phase.

Unavailable numeric values are `None` in the vector and have a false entry in
`feature_validity`; they are not represented as zero. Baseline history is a
bounded 30-value deque of valid movement-energy samples. At least five samples
are required before baseline comparisons are reported. The current measurement
is compared with earlier history and then added. **Reset behavior baseline**
clears prior landmarks and history. The branch retains only the previous valid
landmark coordinates and scalar movement history in memory while active; it
does not persist or transmit frames, coordinates, or histories.

Statuses distinguish `unknown` (no person, invalid input, inference failure, or
too few landmarks), `partial` (some tracked landmarks missing), `warming_up`
(valid observations but not enough baseline samples), and `ok`. Missing
landmarks do not imply stillness or calm. Angles and head offsets are image-space
geometry, not interpretations of emotional state. Repetition measurement is
left unavailable because this short feature window is not sufficient to support
a reliable repeated-movement change estimate. The branch does not identify
autism-specific stimming or determine whether movement is harmful.

To test manually, run the dashboard and click **Test upper-body behavior**
several times with a person in camera view; observe baseline warm-up and feature
validity, then click **Reset behavior baseline**. Also test with no person and
partially obscured arms. The burst lasts about 1.3 seconds; camera permissions
must be enabled in Windows privacy settings. Unit tests mock landmarks and the
camera and do not require physical hardware.

## Phase 2C — Acoustic feature extraction

The audio branch is intentionally local and privacy-preserving. It records one
short in-memory microphone window only when the user clicks **Test audio
features**, computes RMS loudness, spectral centroid/bandwidth/flatness, and a
short-term onset score, and updates a bounded rolling baseline across valid
windows. Silent windows are treated separately from unavailable devices, and the
baseline is reset with **Reset audio baseline**. Raw audio is discarded after
feature extraction and is not saved to disk, logged, or transmitted.

The branch never labels screaming, crying, shouting, or distress. The onset
score is a local signal-change metric relative to the recent session baseline,
not a distress detector, and the audio contribution is a transparent summary in
[0, 1] used for later local fusion logic. Missing values remain `None` in the
feature vector and are marked invalid rather than being fabricated.

## Limitations

- The dataset is small and contains duplicates and possible augmentation leakage;
  neither validation nor supplied test metrics establish real-world performance.
- The dataset has no disgust class. No disgust probability is produced.
- Pose measurements depend on landmark visibility and camera framing and are
  not a distress classifier. The sudden-movement indicator is a configurable
  measurement rule, not evidence that movement is abnormal.
- `Natural`/`neutral` and other display mappings do not establish label
  equivalence with external emotion taxonomies.
- OpenCV's frontal-face Haar cascade is a lightweight local detector, not a
  robust occlusion or identity detector. Size, exposure, and blur checks are
  basic quality proxies; they cannot reliably detect every partially obscured
  face, pose, or demographic failure mode.
- Facial-expression predictions do not establish internal emotional state or
  distress. Repetitive movement and facial expression must not independently
  determine distress; later phases need quality-aware multimodal fusion and an
  explicit unknown state.
- Webcam and microphone permission must be enabled in Windows **Settings →
  Privacy & security → Camera / Microphone** as applicable.
