# NMA Engagement/Bias-State Pipeline

This directory contains the first-pass refactor of
`engagement_bias_states_FINAL-5.html` into explicit pipeline stages.

## Files

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
tests/
```

The scientific stage order is fixed:

```text
01_build_trial_table.py
        ↓
02_preprocess_pupil.py ─┐
03_align_glmhmm_states.py ├─→ 05_build_transition_regressors.py
04_fit_rl_models.py ─────┘                ↓
                              06_fit_transition_models.py
                                         ↓
                              07_make_figures.py
```

`02_preprocess_pupil.py`, `03_align_glmhmm_states.py`, and
`04_fit_rl_models.py` all consume the Stage 1 trial table and may be run
independently once that table exists.

## Installation

Create an environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The IBL stages require a working local ONE configuration. Credentials are not
stored in this project. The public OpenAlyx URL can be changed with the
`ONE_BASE_URL` environment variable.

## Initial validation

```bash
python config.py
python -m py_compile \
  config.py utils.py \
  01_build_trial_table.py 02_preprocess_pupil.py \
  03_align_glmhmm_states.py 04_fit_rl_models.py \
  05_build_transition_regressors.py \
  06_fit_transition_models.py 07_make_figures.py
pytest -q
```

The delivered refactor passes 27 synthetic/unit tests. Those tests establish
encoding direction, session-safe lags/leads, pupil-window extraction, posterior
normalization, GLM-HMM sequence separation, Q-value resets, JSD boundaries,
one-to-one burst matching, and figure helper execution.

They do **not** replace validation against the full IBL cohort.

## Running the pipeline

### 1. Build the behavioral trial table

Remote ONE discovery:

```bash
python 01_build_trial_table.py
```

Canonicalize an existing local table instead:

```bash
python 01_build_trial_table.py --input-table path/to/trials.csv
```

The local input must include at least:

```text
subject, eid, choice, feedbackType,
contrastLeft, contrastRight, probabilityLeft
```

Stimulus and feedback timestamps are required by the pupil stage.

### 2. Preprocess pupil traces

```bash
python 02_preprocess_pupil.py
```

The script caches cleaned session traces under
`data/interim/pupil_traces/`. Each cache contains sample timestamps, raw pupil
diameter when available, cleaned diameter, artifact mask, and preprocessing
metadata.

After caches have been created, an offline rebuild is possible:

```bash
python 02_preprocess_pupil.py --offline
```

The output contains:

- prestimulus tonic pupil;
- stimulus-locked phasic pupil;
- feedback-locked phasic pupil when feedback timestamps and trace coverage are
  available;
- independent validity flags and robust z-scores for each metric.

### 3. Fit and align GLM-HMM states

```bash
python 03_align_glmhmm_states.py
```

For a quicker first environment check:

```bash
python 03_align_glmhmm_states.py --n-initializations 2
```

Every session is treated as a separate HMM sequence. Previous-choice inputs,
initial-state probabilities, and transition counts do not cross session
boundaries. The script performs a synthetic recovery check before fitting unless
`--skip-recovery` is supplied.

### 4. Fit reinforcement-learning baselines

```bash
python 04_fit_rl_models.py
```

This fits sensory-only and hybrid sensory–Q-learning models. Q values reset at
each session. Model comparison uses held-out sessions; final trial regressors use
parameters fitted to all sessions. Boundary fits are reported but not excluded.

### 5. Build transition regressors

```bash
python 05_build_transition_regressors.py
```

This stage creates posterior entropy, exact next-trial JSD, hard switches,
three-transition future-lability outcomes, reward/failure history, block and
session position variables, and deterministic isolated stimulus-locked pupil
bursts.

### 6. Fit transition models

```bash
python 06_fit_transition_models.py
```

Named specifications include:

- `failure_only`;
- `failure_plus_rl`;
- `failure_x_tonic`;
- `lability_failure_plus_rl`;
- `feedback_phasic_error_trials`;
- six pairwise origin–destination models;
- primary and strict burst-matching sensitivity specifications.

Models use subject-clustered standard errors. Skipped or failed specifications
are retained in `transition_model_diagnostics.csv` with a reason.

### 7. Generate figures

```bash
python 07_make_figures.py
```

This stage only reads saved tables and model outputs. It does not fit models.
Missing optional inputs are recorded as skipped in `figure_manifest.csv`; use
`--strict` to fail on the first missing input.

## Output contracts

Trial-level outputs are Parquet by default:

```text
data/processed/trial_table.parquet
data/processed/pupil_trial_features.parquet
data/processed/glmhmm_trial_states.parquet
data/processed/rl_trial_regressors.parquet
data/processed/transition_regressors.parquet
```

Compact summaries, coefficients, diagnostics, and manifests are CSV files under
`output/tables/`. Serialized models are written under `output/models/`. Figures
are written under `output/figures/` in PNG and PDF formats.

## Validation status

`config.py` and `utils.py` have passed their local compile, smoke, and unit-test
checks. Stages 1–7 are implemented, compile successfully, pass synthetic tests,
and have passed an in-memory synthetic pipeline smoke run.

They remain **target-data validation pending** until each stage is run against
the user's full ONE/IBL environment and its notebook-reference outputs are
compared.
