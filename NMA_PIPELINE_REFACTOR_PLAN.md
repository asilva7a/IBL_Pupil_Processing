# NMA Pipeline Refactor Plan

**Source notebook:** `engagement_bias_states_FINAL-5.html`  
**Plan created:** 2026-07-24  
**Last validation update:** 2026-07-24  
**Refactor strategy:** one script at a time, preserving validated outputs before changing scientific logic.

## 1. Objective

Convert the current stateful analysis notebook into a reproducible, inspectable pipeline whose stages can be run and validated independently.

The first pass will follow this logical split:

1. `config.py`
2. `utils.py`
3. `01_build_trial_table.py`
4. `02_preprocess_pupil.py`
5. `03_align_glmhmm_states.py`
6. `04_fit_rl_models.py`
7. `05_build_transition_regressors.py`
8. `06_fit_transition_models.py`
9. `07_make_figures.py`

A script is considered complete only after it runs successfully, writes its expected outputs, and passes the acceptance checks listed below. The notebook remains the reference implementation until each corresponding stage is validated.

## 2. Why this order

The plan is ordered by **cross-pipeline utility**, not by the current notebook cell order.

The notebook contains 61 code cells and repeatedly imports or redefines common operations. For example, NumPy is imported in 26 cells, SciPy in 25, and pandas in 21. `rightward_choice`, robust z-scoring, state relabeling, and significance-label helpers are each defined more than once. It also contains two GLM-HMM implementations. The first refactor priority is therefore to centralize project decisions and genuinely reusable operations before moving scientific stages.

The order also respects dependencies:

```text
config.py
    ↓
utils.py
    ↓
01_build_trial_table.py
    ↓
02_preprocess_pupil.py
    ↓
03_align_glmhmm_states.py
    ↓
04_fit_rl_models.py
    ↓
05_build_transition_regressors.py
    ↓
06_fit_transition_models.py
    ↓
07_make_figures.py
```

`02_preprocess_pupil.py` and `03_align_glmhmm_states.py` both consume the trial table and may be run independently after Stage 1. `04_fit_rl_models.py` is also behavior-only and does not depend on pupil preprocessing, but it is placed after GLM-HMM alignment because the transition analyses ultimately join both model outputs.

## 3. Working principles

- **Preserve before improving.** First reproduce the current validated output; redesign models only after the stage is stable.
- **One script per iteration.** Do not begin the next script until the current one is marked complete.
- **No hidden notebook state.** Every script loads explicit inputs and writes explicit outputs.
- **No model-dependent data exclusions.** Pre-existing quality-control exclusions are configured separately from diagnostic flags produced by later models.
- **Idempotent stages.** Re-running a script with the same inputs and configuration should recreate the same outputs.
- **Deterministic model fitting.** Random seeds and initialization counts live in `config.py`.
- **Fail early.** Missing columns, invalid state probabilities, broken session ordering, and absent files should raise clear errors before fitting.
- **Separate data engineering from scientific inference.** Construction scripts create tables; modeling scripts fit models; figure scripts display saved results.
- **Prefer Parquet for trial-level tables.** CSV remains appropriate for compact summaries and human-readable coefficient tables.
- **Do not store credentials in source files.** ONE authentication must use environment variables or the user's existing local ONE configuration.

## 4. Proposed project structure

```text
nma_pipeline/
├── config.py
├── utils.py
├── 01_build_trial_table.py
├── 02_preprocess_pupil.py
├── 03_align_glmhmm_states.py
├── 04_fit_rl_models.py
├── 05_build_transition_regressors.py
├── 06_fit_transition_models.py
├── 07_make_figures.py
│
├── data/
│   ├── raw/                 # Optional local/raw source files; never overwritten
│   ├── interim/             # Stage outputs not yet analysis-ready
│   └── processed/           # Validated trial-level analysis tables
│
├── output/
│   ├── figures/
│   ├── models/
│   ├── tables/
│   └── logs/
│
├── tests/
│   ├── test_utils.py
│   ├── test_trial_table.py
│   ├── test_pupil.py
│   └── test_state_posteriors.py
│
└── archive/
    └── engagement_bias_states_FINAL-5.html
```

The first implementation can keep the scripts in the project root. Packaging them under `src/nma_pipeline/` can wait until the pipeline is stable.

---

# 5. Utility-first implementation plan

## Task 0 — Project plan

**Status:** [x] Complete  
**Output:** `NMA_PIPELINE_REFACTOR_PLAN.md`

### Subtasks

- [x] Define the logical script split.
- [x] Order work by reuse and dependency.
- [x] Define expected outputs and acceptance criteria.
- [x] Add a persistent completion log.

---

## Task 1 — Build `config.py`

**Status:** [x] Complete  
**Priority:** Highest; imported by every later stage.

### Purpose

Centralize project-wide decisions and constants. A value that changes the behavior of multiple stages should be defined once here rather than repeated in notebook cells.

### Content to migrate

#### Paths

- Project root.
- `data/raw`, `data/interim`, and `data/processed`.
- `output/models`, `output/tables`, `output/figures`, and `output/logs`.
- Canonical input and output filenames for each stage.

#### Canonical column names

- Subject, session/EID, sequence, trial index, state, and posterior columns.
- Choice and feedback columns.
- Contrast and block-prior columns.
- Stimulus, feedback, and pupil timing columns.
- Tonic and phasic pupil feature names.

#### Behavioral conventions

- IBL choice encoding: `+1 = left`, `-1 = right`.
- Signed contrast convention: right contrast minus left contrast.
- Reward encoding from `feedbackType`.
- State labels: engaged, biased-left, biased-right.

#### Session and QC settings

- Minimum DLC sessions.
- Minimum total trials.
- Whether no-go trials enter each QC calculation.
- Camera name and nominal frame rate.
- Pre-existing subject exclusions and the reason for each exclusion.

#### Pupil settings

- DLC likelihood threshold.
- Maximum accepted NaN fraction.
- Blink velocity threshold and padding.
- Tonic baseline window.
- Stimulus-locked phasic window.
- Future feedback-locked baseline and response windows.
- Robust-z clipping thresholds.

#### GLM-HMM settings

- Number of states.
- Gaussian prior variance.
- Sticky transition prior.
- Number of initializations.
- Screening iterations and retained starts.
- Full optimization limits and tolerances.
- Global random seed.

#### RL settings

- Initial action values.
- Parameter bounds.
- Number of restarts.
- Minimum valid choices.
- Diagnostic boundary tolerances.

#### Transition-analysis settings

- Future lability window.
- Burst quantile and refractory interval.
- Matching tolerances and position-bin settings.
- Minimum trial, transition, and subject counts.
- Cluster variable.

#### Figure settings

- DPI.
- Half-width and full-width presentation sizes.
- Output formats.
- Shared state and sex labels.

### Items that must **not** go in `config.py`

- DataFrames or arrays derived from the current dataset.
- Fitted parameters.
- Statistical results.
- Functions.
- ONE passwords or other credentials.
- Figure-specific annotation coordinates.

### Deliverable

- `config.py`

### Acceptance criteria

- [x] `python -m py_compile config.py` succeeds.
- [x] Importing `config.py` creates required output directories or exposes a single explicit function that does so.
- [x] Every configured path resolves from the project root rather than the current working directory.
- [x] Exclusion records include a reason, not only a subject ID.
- [x] No credentials are present.
- [x] A short smoke check prints key paths, state labels, and modeling seeds.
- [x] Full source compilation succeeds in the target Windows PowerShell environment.

---

## Task 2 — Build `utils.py`

**Status:** [x] Complete  
**Priority:** Highest reuse after configuration.

### Purpose

Provide small, tested, reusable functions that are independent of a particular scientific conclusion. Functions should accept explicit inputs, return explicit outputs, and avoid reliance on notebook globals.

### Utility groups, ordered by expected frequency

#### 2.1 Validation and filesystem utilities

- `ensure_directory`
- `require_columns`
- `require_unique_key`
- `validate_probability_matrix`
- `validate_trial_order`
- `load_table` / `save_table` with Parquet and CSV support
- Optional compact stage logging helper

#### 2.2 Trial encoding utilities

- `encode_ibl_choice`
- `encode_reward`
- `build_signed_contrast`
- `rightward_choice`
- No-go handling

#### 2.3 Grouping and sequence utilities

- Detect session/sequence boundaries from EID or timestamp resets.
- Add previous- and next-trial columns without crossing boundaries.
- Add within-sequence trial number and relative session progress.
- Safe groupwise shifts and rolling windows.

#### 2.4 Scaling and QC utilities

- `safe_zscore`
- `robust_zscore`
- Within-subject and within-session/context standardization.
- Median absolute deviation helpers.
- Missingness summaries.
- Generic outlier flags that return diagnostics without silently dropping rows.

#### 2.5 State-posterior utilities

- Sanitize and renormalize posterior matrices.
- Posterior entropy.
- Jensen–Shannon divergence.
- Hard state from posterior maximum.
- State occupancy summaries.
- State relabel mapping application.

#### 2.6 Generic statistical-output utilities

- Tidy a statsmodels coefficient table.
- Confidence intervals and odds ratios.
- Paired effect size.
- P-value formatting.
- These functions may format results but must not choose a scientific model.

#### 2.7 Pupil trace primitives

Only low-level reusable operations:

- Blink/artifact masking.
- Time-window selection.
- Baseline subtraction.
- Event-locked response extraction.

Loading IBL datasets and selecting the scientific pupil windows remain in `02_preprocess_pupil.py`.

### Notebook functions expected to migrate or consolidate

- `rightward_choice`
- `robust_zscore`, `robust_zscore_series`, and repeated local z-score functions
- `sanitize_probability_matrix`
- repeated sequence-safe shift logic
- generic coefficient-table construction
- generic p-value/significance formatting
- reusable event-window calculations

### Functions that must **not** go in `utils.py`

- `GLMHMM` or `GLMHMMWeighted`
- RL likelihoods
- model-fitting formulas
- burst-matching scientific specifications
- tonic-pupil hypothesis tests
- state-transition regressions
- complete figures or slide composition

Those belong in the corresponding analysis scripts.

### Deliverables

- `utils.py`
- `tests/test_utils.py`

### Acceptance criteria

- [x] `python -m py_compile utils.py` succeeds.
- [x] Utility functions contain docstrings and type hints for public arguments and returns.
- [x] Unit tests cover normal, missing, constant, and malformed inputs.
- [x] Choice-direction test confirms that IBL `-1` maps to rightward choice.
- [x] Sequence-shift test confirms that lag/lead values never cross a session boundary.
- [x] Probability tests confirm rows sum to one after sanitization.
- [x] JSD is zero for identical distributions and finite for valid inputs.
- [x] No utility function reads notebook globals.
- [x] Full 27-test suite passes in the target Windows environment (`27 passed in 22.61s`).

---

## Task 3 — Build `01_build_trial_table.py`

**Status:** [ ] Implemented; target-data validation pending

### Purpose

Discover eligible sessions, load behavioral trials, apply session-level QC, assign sequence and epoch labels, and write one canonical behavioral trial table.

### Notebook content to migrate

- ONE connection setup without embedded credentials.
- `find_video_sessions`
- `is_choiceworld`
- `find_pupil_sessions`
- `get_sex`
- `load_session_trials`
- subject/session discovery and QC sweep
- trial-count estimation
- `label_trial_epochs`
- signed contrast, reward, and rightward-choice encoding

### Inputs

- IBL ONE configuration.
- Optional subject allowlist from `config.py`.
- Session and behavioral QC thresholds.

### Outputs

- `data/processed/trial_table.parquet`
- `output/tables/subject_qc.csv`
- `output/tables/session_manifest.csv`
- `output/logs/01_build_trial_table.log`

### Required trial-table keys

- `subject`
- `eid`
- `sequence_id`
- `trial_index`

Together these must uniquely identify a row.

### Acceptance criteria

- [ ] No duplicate trial keys.
- [ ] Trials are ordered within every subject/session.
- [ ] Encoded choice agrees with the IBL convention.
- [ ] Signed contrast direction passes an easy-trial sanity check.
- [ ] Epoch labels reproduce the notebook's unbiased/transition/stable definition.
- [ ] The number of subjects, sessions, and trials is printed and saved.
- [ ] Running the script twice produces the same table shape and keys.

---

## Task 4 — Build `02_preprocess_pupil.py`

**Status:** [ ] Implemented; target-data validation pending

### Purpose

Load pupil traces, perform trace-level QC and blink handling, align traces to trial events, and create validated trial-level pupil features.

### Notebook content to migrate

- `remove_blinks`
- `load_pupil_diameter`
- `per_trial_pupil`
- pupil missingness gates
- tonic and stimulus-locked phasic feature construction
- within-animal robust z-scores
- pupil QC reports and known unusable-pupil exclusions

### Inputs

- `data/processed/trial_table.parquet`
- IBL camera timestamps and DLC data
- pupil configuration values

### Outputs

- `data/processed/pupil_trial_features.parquet`
- `output/tables/pupil_qc.csv`
- `output/tables/pupil_exclusion_report.csv`
- `output/logs/02_preprocess_pupil.log`

### Design requirement for future feedback-locked analyses

The stage should preserve enough trace provenance to reconstruct event-locked measures later. At minimum, each processed session should retain:

- camera sample timestamps or a documented cache reference;
- cleaned pupil samples or a documented cache reference;
- the exact preprocessing parameters used;
- trial stimulus and feedback timestamps.

This avoids the current situation in which a trial table contains `feedback_times` but only stimulus-locked pupil summaries.

### Acceptance criteria

- [ ] Pupil features join one-to-one onto trial keys.
- [ ] No trial is matched across sessions.
- [ ] Missingness and blink-removal fractions are reported by subject/session.
- [ ] Known unusable subjects are flagged using pre-existing QC rules.
- [ ] Tonic and phasic definitions are recorded in output metadata.
- [ ] A one-session diagnostic plot reproduces the expected pupil trace and windows.

---

## Task 5 — Build `03_align_glmhmm_states.py`

**Status:** [ ] Implemented; target-data validation pending

### Purpose

Fit the behavioral GLM-HMM, validate parameter recovery, relabel states consistently, and write trial-level state posteriors and subject-level model parameters.

The filename retains the earlier logical split, but this stage includes both **fitting** and **alignment/relabeling** because state alignment is inseparable from the model output in the current pipeline.

### Notebook content to migrate

- `GLMHMM`
- `fit_best`
- parameter-recovery simulation
- `build_design_matrix`
- `fit_animal_glmhmm`
- state ordering and engagement-based relabeling
- posterior extraction
- occupancy summaries
- posterior normalization checks

The experimental pupil-weighted GLM-HMM should not be merged into the primary implementation. It can be retained as an optional comparison after the standard model is validated.

### Inputs

- `data/processed/trial_table.parquet`
- GLM-HMM settings from `config.py`

### Outputs

- `data/processed/glmhmm_trial_states.parquet`
- `output/models/glmhmm_subject_parameters.npz` or one model file per subject
- `output/tables/glmhmm_fit_diagnostics.csv`
- `output/tables/state_occupancy.csv`
- `output/logs/03_align_glmhmm_states.log`

### Acceptance criteria

- [ ] Simulated parameter recovery passes before real-data fitting.
- [ ] Posterior rows are finite, nonnegative, and sum to one.
- [ ] State labels are stable under the documented relabeling rule.
- [ ] No state sequence crosses session boundaries.
- [ ] Model likelihood and convergence diagnostics are saved for every subject.
- [ ] State-conditioned psychometric curves reproduce the notebook-level sanity check.

---

## Task 6 — Build `04_fit_rl_models.py`

**Status:** [ ] Implemented; target-data validation pending

### Purpose

Fit behavior-only reinforcement models and generate trial-wise computational regressors without mixing them with transition outcomes.

### Current scope

- Preserve the hybrid sensory–Q-learning model as a baseline.
- Fit a sensory-only comparison model.
- Use held-out or cross-validated prediction to quantify the incremental value of reinforcement terms.
- Save parameter-boundary diagnostics.
- Do not automatically exclude boundary-fit animals.

### Notebook/conversation content to migrate

- RL likelihood and optimizer.
- Trial-wise Q values, expected reward, RPE, positive/negative RPE, outcome surprise, and choice entropy.
- Subject-level parameter tables.
- Boundary-fit summaries.

### Planned scientific extension

After the baseline is stable, consider a task-aligned learned-prior model in which reward updates the inferred block-side prior rather than generic left/right action values.

### Inputs

- `data/processed/trial_table.parquet`

### Outputs

- `data/processed/rl_trial_regressors.parquet`
- `output/tables/rl_subject_parameters.csv`
- `output/tables/rl_model_comparison.csv`
- `output/tables/rl_fit_diagnostics.csv`
- `output/logs/04_fit_rl_models.log`

### Acceptance criteria

- [ ] Choice and signed-contrast direction are validated before fitting.
- [ ] Full and sensory-only models are compared out of sample.
- [ ] Parameter bounds and convergence are reported.
- [ ] Trial-wise regressors reset at session boundaries.
- [ ] Negative RPE versus binary failure correlation is reported.
- [ ] Saved regressors join one-to-one onto trial keys.

---

## Task 7 — Build `05_build_transition_regressors.py`

**Status:** [ ] Implemented; target-data validation pending

### Purpose

Join behavior, pupil, GLM-HMM, and RL outputs and construct all transition predictors and outcomes without fitting inferential models.

### Notebook content to migrate

- posterior entropy
- corrected Jensen–Shannon divergence
- next-trial hard state
- future JSD and switch windows
- failure/reward histories
- block and session-progress covariates
- burst detection and isolation
- event-window tables
- matched-control candidate fields

### Inputs

- `trial_table.parquet`
- `pupil_trial_features.parquet`
- `glmhmm_trial_states.parquet`
- `rl_trial_regressors.parquet`

### Outputs

- `data/processed/transition_regressors.parquet`
- `output/tables/transition_regressor_qc.csv`
- `output/logs/05_build_transition_regressors.log`

### Acceptance criteria

- [ ] All joins are one-to-one on canonical trial keys.
- [ ] Next-trial and future-window values never cross session boundaries.
- [ ] JSD is finite and in its valid range.
- [ ] Hard-switch counts agree with direct state comparisons.
- [ ] Burst selection is deterministic for a fixed threshold and seed.
- [ ] The table records whether each pupil measure is stimulus-locked or feedback-locked.

---

## Task 8 — Build `06_fit_transition_models.py`

**Status:** [ ] Implemented; target-data validation pending

### Purpose

Fit inferential models predicting state lability and origin–destination transitions from outcomes, reinforcement variables, and pupil measures.

### Notebook content to migrate

- lability regressions
- hard-switch regressions
- origin–destination models
- clustered standard errors
- event-triggered burst analyses
- strict one-to-one pseudo-event matching
- subject-level Wilcoxon summaries
- robustness and sensitivity analyses

### Model organization

Keep each model specification named and explicit, for example:

- `failure_only`
- `failure_plus_rl`
- `failure_x_tonic`
- `feedback_phasic_error_trials`
- `origin_destination_failure`
- `burst_matched_controls`

### Inputs

- `data/processed/transition_regressors.parquet`

### Outputs

- `output/tables/transition_model_coefficients.csv`
- `output/tables/origin_destination_coefficients.csv`
- `output/tables/burst_matched_results.csv`
- `output/tables/transition_model_diagnostics.csv`
- serialized fitted models under `output/models/`
- `output/logs/06_fit_transition_models.log`

### Acceptance criteria

- [ ] Every result row records the specification name, outcome, predictors, sample size, subject count, and exclusion set.
- [ ] Clustered standard errors use the configured subject identifier.
- [ ] Matched controls are unique and remain in the same subject/session/context.
- [ ] Main effects are not interpreted without their interactions when interactions are present.
- [ ] Sensitivity specifications are saved rather than overwritten.
- [ ] Unsupported approximate soft-switch outcomes are not presented as true transition probabilities.

---

## Task 9 — Build `07_make_figures.py`

**Status:** [ ] Implemented; target-data validation pending

### Purpose

Generate publication- and presentation-ready figures entirely from saved tables and model outputs.

### Notebook content to migrate

- behavioral psychometrics
- state-conditioned psychometrics
- state occupancy figures
- tonic and phasic pupil summaries
- raw and z-scored tonic pupil by epoch
- animal-level paired plots
- sex-stratified summaries
- transition and burst-analysis figures
- 300-DPI export settings

### Rules

- No model fitting inside the figure script.
- No silent exclusions.
- Every figure caption or metadata record states its input table and exclusion set.
- Plotting transformations are limited to display reshaping and aggregation already defined by saved analysis outputs.
- Slide-composition helpers from the notebook are archived unless explicitly needed; they are not part of the scientific figure pipeline.

### Inputs

- Saved tables and model outputs from Stages 1–8.

### Outputs

- `output/figures/*.png` at 300 DPI
- optional vector `*.pdf` or `*.svg`
- `output/tables/figure_manifest.csv`
- `output/logs/07_make_figures.log`

### Acceptance criteria

- [ ] Figures regenerate from a fresh Python process.
- [ ] No notebook globals are required.
- [ ] Output dimensions and DPI are validated.
- [ ] Figure filenames are stable and descriptive.
- [ ] A figure manifest records source tables, script version, and exclusions.

---


# 5A. Implementation snapshot — 2026-07-24

All planned source files now exist:

```text
config.py
utils.py
01_build_trial_table.py
02_preprocess_pupil.py
03_align_glmhmm_states.py
04_fit_rl_models.py
05_build_transition_regressors.py
06_fit_transition_models.py
07_make_figures.py
requirements.txt
README_PIPELINE.md
tests/
```

Validation completed in the construction environment:

- all nine Python source files compile successfully;
- `config.py` smoke check resolves paths and creates configured directories;
- 27 unit/synthetic tests pass;
- an in-memory synthetic pipeline smoke run successfully exercised behavioral
  canonicalization, pupil feature QC, per-session GLM-HMM fitting, RL fitting,
  transition-regressor construction, named transition models, and figure
  helpers;
- GLM-HMM and RL histories reset at session boundaries;
- the transition stage uses exact posterior JSD and never labels an independent
  posterior product as a true transition probability.

The stage scripts are not marked complete because the full target-data
acceptance criteria require the user's ONE/IBL environment, Parquet support, and
comparison with the notebook-reference cohort outputs. Their current status is
**implemented; target-data validation pending**.

# 5B. Bonus objectives / polish

These objectives are valuable but do **not** block scientific validation of the
core seven-stage pipeline. They should be taken on only when the current stage is
stable or when they remove a clear usability bottleneck.

## Progress and operator feedback

- [x] Add detailed progress logging to Stage 1, including discovery position,
  unique subjects found, elapsed time, per-subject/session status, and cumulative
  trial counts. Validated in the target Windows environment on 2026-07-24.
- [ ] Generalize a consistent progress display across all stages, with stage name,
  completed/total units, elapsed time, processing rate, estimated time remaining,
  warnings, and skipped items.
- [ ] End every stage with a compact run summary showing inputs, outputs, retained
  subjects/sessions/trials, exclusions, failures, runtime, and peak memory when
  available.
- [ ] Collapse repetitive third-party warnings into a counted warning summary while
  retaining full details in the log file.

## Restartability and orchestration

- [ ] Add safe checkpoints for long subject- or session-level stages so an
  interrupted run can resume without repeating completed work.
- [ ] Detect valid existing outputs and skip completed stages by default, with an
  explicit `--force` option to rebuild them.
- [ ] Add a lightweight `run_pipeline.py` orchestrator supporting `--from-stage`,
  `--to-stage`, `--subjects`, `--dry-run`, `--force`, and a status-only mode.
- [ ] Handle `Ctrl+C` cleanly by flushing logs, saving checkpoint metadata, and
  reporting exactly where execution stopped.

## Reproducibility and provenance

- [ ] Save a machine-readable run manifest for every stage containing the pipeline
  version, timestamp, command line, configuration snapshot, input/output checksums,
  Python and package versions, random seeds, and source-control commit when
  available.
- [ ] Add an environment lock file after target validation so the working Windows
  environment can be recreated precisely.
- [ ] Add explicit cache metadata so cached ONE downloads and derived intermediates
  can be distinguished from newly retrieved or recomputed data.

## Performance and quality-of-life

- [ ] Record per-stage timing and throughput benchmarks on a small smoke cohort and
  the full cohort, making regressions visible after later changes.
- [ ] Add a fast smoke-test mode that processes a small named subset of subjects and
  writes outputs to a separate validation directory.
- [ ] Add optional parallel execution only for stages whose subject/session units are
  demonstrably independent and deterministic.
- [ ] Add static-analysis and formatting checks after the scientific code paths are
  stable, without allowing pandas typing noise to obscure genuine runtime issues.
- [ ] Generate a compact HTML or Markdown run report linking the figure manifest,
  QC tables, model summaries, warnings, exclusions, and runtime information.

# 6. Deferred work

These scientific or architectural changes are intentionally postponed until the
first-pass scripts reproduce the current validated pipeline. Operator-facing
enhancements such as progress displays, restartability, orchestration, optional
parallelism, and run reports are tracked separately under Bonus objectives /
polish.

- Converting the scripts into an installable Python package.
- A joint RL–GLM-HMM.
- A learned-prior reinforcement model.
- Final validation and data-driven tuning of the feedback-locked pupil windows on real cleaned traces.
- Replacing the standard GLM-HMM with the pupil-weighted experimental model.

# 7. Completion protocol

When a script is successfully completed:

1. Change its status from `[ ] Not started` to `[x] Complete`.
2. Mark each completed subtask or acceptance criterion.
3. Record the completion date.
4. Record the exact output files produced.
5. Note any deviations from this plan.
6. Add validation results and unresolved issues to the completion log.
7. Identify the next script, but do not begin it until requested.

A task may be marked **Blocked** or **Partial** when appropriate; completion should not be claimed merely because a file exists.

# 8. Completion log

| Date | Task | Status | Validation summary | Outputs | Open issues |
|---|---|---|---|---|---|
| 2026-07-24 | Project plan | Complete | Logical split, dependencies, deliverables, and acceptance criteria documented | `NMA_PIPELINE_REFACTOR_PLAN.md` | None |
| 2026-07-24 | `config.py` | Complete | Compiles locally and in target Windows PowerShell; path/directory smoke check passes; exclusions have reasons; no credentials | `config.py`, `config_smoke_check.txt` | Project values remain editable as target validation proceeds |
| 2026-07-24 | `utils.py` | Complete | Compiles locally and in target Windows PowerShell; full suite passes (`27 passed in 22.61s`) | `utils.py`, `tests/test_utils.py` | None |
| 2026-07-24 | `01_build_trial_table.py` | Implemented | Compiles; local-table canonicalization and epoch/encoding tests pass; synthetic smoke run passes; detailed progress logging validated during remote ONE discovery in Windows PowerShell | `01_build_trial_table.py`, `tests/test_trial_table.py` | Full remote run, final cohort counts, and notebook parity pending |
| 2026-07-24 | `02_preprocess_pupil.py` | Implemented | Compiles; synthetic event-window and metric-specific QC tests pass | `02_preprocess_pupil.py`, `tests/test_pupil.py` | Real DLC loading, trace QC, and diagnostic plot pending |
| 2026-07-24 | `03_align_glmhmm_states.py` | Implemented | Compiles; multi-sequence posterior tests and synthetic pipeline fit pass | `03_align_glmhmm_states.py`, `tests/test_state_posteriors.py` | Full recovery/cohort fits and notebook psychometric parity pending |
| 2026-07-24 | `04_fit_rl_models.py` | Implemented | Compiles; Q-reset and bounded-fit tests pass; synthetic pipeline fit passes | `04_fit_rl_models.py`, `tests/test_rl.py` | Full cohort CV, boundary summaries, and saved regressors pending |
| 2026-07-24 | `05_build_transition_regressors.py` | Implemented | Compiles; JSD/session-boundary and burst-selection tests pass; synthetic smoke run passes | `05_build_transition_regressors.py`, `tests/test_transition_regressors.py` | Full joined table and notebook count parity pending |
| 2026-07-24 | `06_fit_transition_models.py` | Implemented | Compiles; one-to-one matching test and synthetic named-model run pass | `06_fit_transition_models.py`, `tests/test_transition_models.py` | Full cohort cluster models and robustness outputs pending |
| 2026-07-24 | `07_make_figures.py` | Implemented | Compiles; all planned figure families and manifest generation are implemented; figure helpers exercised in synthetic smoke run | `07_make_figures.py` | Full saved-output figure regeneration, visual review, and notebook parity pending |

# 9. Immediate next task

Run **`01_build_trial_table.py`** in the target ONE/IBL environment and compare its
subject, session, trial, choice-direction, signed-contrast, and epoch counts with
the notebook reference. Once that stage passes, mark Task 3 complete before
validating the pupil stage.
