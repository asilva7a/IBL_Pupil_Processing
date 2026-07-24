"""Central configuration for the NMA engagement/bias-state pipeline.

This module contains project-wide decisions and constants only.  It must remain
safe to import: it does not connect to ONE, load data, fit models, or mutate the
filesystem automatically.  Run ``python config.py`` for a compact smoke check
and to create the standard project directories explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DATA_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"

OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "output"
MODEL_DIR: Final[Path] = OUTPUT_DIR / "models"
TABLE_DIR: Final[Path] = OUTPUT_DIR / "tables"
FIGURE_DIR: Final[Path] = OUTPUT_DIR / "figures"
LOG_DIR: Final[Path] = OUTPUT_DIR / "logs"
ARCHIVE_DIR: Final[Path] = PROJECT_ROOT / "archive"
GLMHMM_MODEL_DIR: Final[Path] = MODEL_DIR / "glmhmm"
TRANSITION_MODEL_DIR: Final[Path] = MODEL_DIR / "transitions"
PUPIL_TRACE_CACHE_DIR: Final[Path] = INTERIM_DATA_DIR / "pupil_traces"

PROJECT_DIRECTORIES: Final[tuple[Path, ...]] = (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    LOG_DIR,
    ARCHIVE_DIR,
    GLMHMM_MODEL_DIR,
    TRANSITION_MODEL_DIR,
    PUPIL_TRACE_CACHE_DIR,
)

# Canonical stage outputs.
TRIAL_TABLE_PATH: Final[Path] = PROCESSED_DATA_DIR / "trial_table.parquet"
PUPIL_FEATURES_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "pupil_trial_features.parquet"
)
GLMHMM_STATES_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "glmhmm_trial_states.parquet"
)
RL_REGRESSORS_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "rl_trial_regressors.parquet"
)
TRANSITION_REGRESSORS_PATH: Final[Path] = (
    PROCESSED_DATA_DIR / "transition_regressors.parquet"
)

SUBJECT_QC_PATH: Final[Path] = TABLE_DIR / "subject_qc.csv"
SESSION_MANIFEST_PATH: Final[Path] = TABLE_DIR / "session_manifest.csv"
PUPIL_QC_PATH: Final[Path] = TABLE_DIR / "pupil_qc.csv"
PUPIL_EXCLUSION_REPORT_PATH: Final[Path] = (
    TABLE_DIR / "pupil_exclusion_report.csv"
)
GLMHMM_DIAGNOSTICS_PATH: Final[Path] = (
    TABLE_DIR / "glmhmm_fit_diagnostics.csv"
)
STATE_OCCUPANCY_PATH: Final[Path] = TABLE_DIR / "state_occupancy.csv"
RL_PARAMETER_PATH: Final[Path] = TABLE_DIR / "rl_subject_parameters.csv"
RL_MODEL_COMPARISON_PATH: Final[Path] = TABLE_DIR / "rl_model_comparison.csv"
RL_DIAGNOSTICS_PATH: Final[Path] = TABLE_DIR / "rl_fit_diagnostics.csv"
TRANSITION_COEFFICIENT_PATH: Final[Path] = (
    TABLE_DIR / "transition_model_coefficients.csv"
)
ORIGIN_DESTINATION_COEFFICIENT_PATH: Final[Path] = (
    TABLE_DIR / "origin_destination_coefficients.csv"
)
FIGURE_MANIFEST_PATH: Final[Path] = TABLE_DIR / "figure_manifest.csv"
PUPIL_FEATURE_METADATA_PATH: Final[Path] = TABLE_DIR / "pupil_feature_metadata.json"
GLMHMM_PSYCHOMETRIC_QC_PATH: Final[Path] = TABLE_DIR / "glmhmm_psychometric_qc.csv"
GLMHMM_RECOVERY_PATH: Final[Path] = TABLE_DIR / "glmhmm_parameter_recovery.csv"
TRANSITION_REGRESSOR_QC_PATH: Final[Path] = TABLE_DIR / "transition_regressor_qc.csv"
BURST_EVENT_WINDOW_PATH: Final[Path] = TABLE_DIR / "burst_event_windows.csv"
BURST_MATCHED_RESULTS_PATH: Final[Path] = TABLE_DIR / "burst_matched_results.csv"
TRANSITION_MODEL_DIAGNOSTICS_PATH: Final[Path] = TABLE_DIR / "transition_model_diagnostics.csv"


# ---------------------------------------------------------------------------
# ONE / IBL access
# ---------------------------------------------------------------------------

ONE_BASE_URL: Final[str] = os.environ.get(
    "ONE_BASE_URL",
    "https://openalyx.internationalbrainlab.org",
)
ONE_QUERY_TYPE: Final[str] = "remote"
ONE_HTTP_DOWNLOAD_THREADS: Final[int] = 1

# Credentials deliberately do not live in source control.  Use the user's
# existing ONE configuration or environment-specific authentication.


# ---------------------------------------------------------------------------
# Canonical columns and trial keys
# ---------------------------------------------------------------------------

SUBJECT_COLUMN: Final[str] = "subject"
SEX_COLUMN: Final[str] = "sex"
SESSION_COLUMN: Final[str] = "eid"
SEQUENCE_COLUMN: Final[str] = "sequence_id"
TRIAL_INDEX_COLUMN: Final[str] = "trial_index"
TRIAL_KEY_COLUMNS: Final[tuple[str, ...]] = (
    SUBJECT_COLUMN,
    SESSION_COLUMN,
    SEQUENCE_COLUMN,
    TRIAL_INDEX_COLUMN,
)

CHOICE_COLUMN: Final[str] = "choice"
FEEDBACK_COLUMN: Final[str] = "feedbackType"
REWARD_COLUMN: Final[str] = "reward"
ACCURACY_COLUMN: Final[str] = "accuracy"
CONTRAST_LEFT_COLUMN: Final[str] = "contrastLeft"
CONTRAST_RIGHT_COLUMN: Final[str] = "contrastRight"
SIGNED_CONTRAST_COLUMN: Final[str] = "signed_contrast"
BLOCK_PRIOR_COLUMN: Final[str] = "probabilityLeft"
EPOCH_COLUMN: Final[str] = "epoch"

STIMULUS_TIME_COLUMN: Final[str] = "stimOn_times"
FEEDBACK_TIME_COLUMN: Final[str] = "feedback_times"

STATE_COLUMN: Final[str] = "state"
STATE_LABEL_COLUMN: Final[str] = "state_label"
STATE_POSTERIOR_COLUMNS: Final[tuple[str, ...]] = (
    "p_state0",
    "p_state1",
    "p_state2",
)

PUPIL_TONIC_COLUMN: Final[str] = "pupil_tonic"
PUPIL_PHASIC_COLUMN: Final[str] = "pupil_phasic"
PUPIL_TONIC_ROBUST_Z_COLUMN: Final[str] = "pupil_tonic_rz"
PUPIL_PHASIC_ROBUST_Z_COLUMN: Final[str] = "pupil_phasic_rz"
PUPIL_FEEDBACK_PHASIC_COLUMN: Final[str] = "pupil_feedback_phasic"
PUPIL_FEEDBACK_PHASIC_ROBUST_Z_COLUMN: Final[str] = "pupil_feedback_phasic_rz"


# ---------------------------------------------------------------------------
# Behavioral conventions
# ---------------------------------------------------------------------------

# IBL raw choice coding.
IBL_LEFT_CHOICE: Final[int] = 1
IBL_RIGHT_CHOICE: Final[int] = -1
IBL_NOGO_CHOICE: Final[int] = 0

# Analysis coding.
ENCODED_LEFT_CHOICE: Final[int] = 0
ENCODED_RIGHT_CHOICE: Final[int] = 1

# Signed contrast is right minus left.  Positive values therefore favor a
# rightward response under the coding above.
SIGNED_CONTRAST_DIRECTION: Final[str] = "right_minus_left"

STATE_NAMES: Final[dict[int, str]] = {
    0: "engaged",
    1: "biased-left",
    2: "biased-right",
}
ENGAGED_STATE: Final[int] = 0
BIASED_STATES: Final[tuple[int, int]] = (1, 2)
EPOCH_ORDER: Final[tuple[str, ...]] = (
    "unbiased",
    "transition",
    "stable",
)
TRANSITION_EPOCH_TRIALS: Final[int] = 10


# ---------------------------------------------------------------------------
# Session discovery and quality control
# ---------------------------------------------------------------------------

SUBJECT_ALLOWLIST: Final[tuple[str, ...] | None] = None
MIN_DLC_SESSIONS: Final[int] = 2
MIN_TOTAL_TRIALS: Final[int] = 1_000
USE_CHOICE_TRIALS_FOR_QC: Final[bool] = True

# Keep stage-specific exclusions separate.  SH015 is a pupil-data exclusion,
# not a behavior/model exclusion.
PUPIL_SUBJECT_EXCLUSIONS: Final[dict[str, str]] = {
    "SH015": (
        "Known unusable pupil data; exclusion was established before the "
        "downstream pupil summaries."
    ),
}
BEHAVIOR_SUBJECT_EXCLUSIONS: Final[dict[str, str]] = {}
RL_SUBJECT_EXCLUSIONS: Final[dict[str, str]] = {}


# ---------------------------------------------------------------------------
# Pupil preprocessing
# ---------------------------------------------------------------------------

PUPIL_CAMERA: Final[str] = "left"
NOMINAL_CAMERA_FPS: Final[float] = 60.0
PUPIL_DLC_LIKELIHOOD_THRESHOLD: Final[float] = 0.90
PUPIL_SMOOTH_STD_THRESHOLD: Final[float] = 5.0
PUPIL_SMOOTH_NAN_THRESHOLD: Final[float] = 1.0
MAX_PUPIL_NAN_FRACTION: Final[float] = 0.30

BLINK_VELOCITY_MAD_THRESHOLD: Final[float] = 5.0
BLINK_PADDING_SECONDS: Final[float] = 0.125
FRAME_ROBUST_Z_MAX: Final[float] = 8.0

STIMULUS_BASELINE_WINDOW: Final[tuple[float, float]] = (-0.5, 0.0)
STIMULUS_PHASIC_WINDOW: Final[tuple[float, float]] = (0.0, 1.0)

# Reserved for the future feedback-locked extraction stage.  These are starting
# windows, not tuned estimates; the pupil response shape should be inspected
# before they are treated as final scientific settings.
FEEDBACK_BASELINE_WINDOW: Final[tuple[float, float]] = (-0.5, 0.0)
FEEDBACK_RESPONSE_WINDOW: Final[tuple[float, float]] = (0.5, 2.5)

MIN_EVENT_WINDOW_VALID_FRACTION: Final[float] = 0.80
TONIC_ROBUST_Z_MAX: Final[float] = 5.0
PHASIC_ROBUST_Z_MAX: Final[float] = 4.0
MIN_PUPIL_CONTEXT_TRIALS: Final[int] = 20


# ---------------------------------------------------------------------------
# GLM-HMM settings
# ---------------------------------------------------------------------------

GLMHMM_N_STATES: Final[int] = 3
GLMHMM_N_FEATURES: Final[int] = 4
GLMHMM_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "signed_contrast_z",
    "bias",
    "previous_choice",
    "previous_stimulus",
)
GLMHMM_WEIGHT_PRIOR_VARIANCE: Final[float] = 2.0
GLMHMM_STICKY_DIRICHLET_ALPHA: Final[float] = 2.0
GLMHMM_N_INITIALIZATIONS: Final[int] = 20
GLMHMM_SCREENING_ITERATIONS: Final[int] = 20
GLMHMM_RETAINED_INITIALIZATIONS: Final[int] = 4
GLMHMM_EM_MAX_ITERATIONS: Final[int] = 150
GLMHMM_EM_ABSOLUTE_TOLERANCE: Final[float] = 1e-4
GLMHMM_EM_RELATIVE_TOLERANCE: Final[float] = 1e-4
GLMHMM_EM_PATIENCE: Final[int] = 3
GLMHMM_EM_MIN_ITERATIONS: Final[int] = 10
GLMHMM_LOGISTIC_MAX_ITERATIONS: Final[int] = 150
GLMHMM_LOGISTIC_TOLERANCE: Final[float] = 1e-4
GLMHMM_TEST_SESSION_FRACTION: Final[float] = 0.25
GLMHMM_RECOVERY_TRIALS: Final[int] = 2_000
GLMHMM_RECOVERY_MIN_DECODE_ACCURACY: Final[float] = 0.75


# ---------------------------------------------------------------------------
# Reinforcement-learning baseline settings
# ---------------------------------------------------------------------------

# These values belong to the post-notebook RL extension and should be checked
# against the final Stage 4 implementation before that stage is declared stable.
RL_INITIAL_Q_LEFT: Final[float] = 0.5
RL_INITIAL_Q_RIGHT: Final[float] = 0.5
RL_MIN_VALID_CHOICES: Final[int] = 50
RL_N_RESTARTS: Final[int] = 4
RL_TEST_SESSION_FRACTION: Final[float] = 0.25
RL_OPTIMIZER_MAX_ITERATIONS: Final[int] = 1_000
RL_PROBABILITY_EPSILON: Final[float] = 1e-9
RL_RANDOM_SEED: Final[int] = 2026
RL_BOUNDARY_FRACTION_TOLERANCE: Final[float] = 0.01
RL_PARAMETER_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "alpha": (0.001, 0.999),
    "beta_value": (0.0, 20.0),
    "beta_stimulus": (-50.0, 50.0),
    "bias": (-10.0, 10.0),
    "lapse": (0.0001, 0.20),
}


# ---------------------------------------------------------------------------
# Transition and burst analyses
# ---------------------------------------------------------------------------

FUTURE_LABILITY_TRIALS: Final[int] = 3
BURST_PRE_TRIALS: Final[int] = 10
BURST_POST_TRIALS: Final[int] = 10
BURST_QUANTILE: Final[float] = 0.90
BURST_REFRACTORY_TRIALS: Final[int] = 5
MIN_VALID_BURST_OFFSETS_PER_SUBJECT: Final[int] = 8

MATCH_PRE_OFFSETS: Final[tuple[int, ...]] = (-3, -2, -1)
MATCH_POST_OFFSETS: Final[tuple[int, ...]] = (0, 1, 2)
N_POSITION_BINS: Final[int] = 10
BURST_EXCLUSION_RADIUS: Final[int] = 5
MATCH_MAX_POSITION_BIN_DIFFERENCE: Final[int] = 1
MATCH_MAX_TRIAL_FRACTION_DIFFERENCE: Final[float] = 0.10
STRICT_MATCH_MAX_TRIAL_FRACTION_DIFFERENCE: Final[float] = 0.05
MATCH_LARGE_INVALID_COST: Final[float] = 1e6
TRANSITION_RANDOM_SEED: Final[int] = 2026

MIN_TRIALS_PER_TRANSITION_MODEL: Final[int] = 250
MIN_TRANSITIONS_PER_MODEL: Final[int] = 20
MIN_SUBJECTS_PER_MODEL: Final[int] = 10
CLUSTER_COLUMN: Final[str] = SUBJECT_COLUMN


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

FIGURE_DPI: Final[int] = 300
HALF_SLIDE_SIZE_INCHES: Final[tuple[float, float]] = (6.1, 4.8)
FULL_SLIDE_SIZE_INCHES: Final[tuple[float, float]] = (12.3, 5.4)
FIGURE_OUTPUT_FORMATS: Final[tuple[str, ...]] = ("png", "pdf")
FIGURE_STATE_COLORS: Final[dict[str, str]] = {
    "engaged": "#333333",
    "biased-left": "#377eb8",
    "biased-right": "#e41a1c",
}
SEX_ORDER: Final[tuple[str, ...]] = ("M", "F")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

PIPELINE_VERSION: Final[str] = "0.1.0-refactor-2026-07-24"
GLOBAL_RANDOM_SEED: Final[int] = 2026


def ensure_project_directories() -> tuple[Path, ...]:
    """Create the configured project directories and return them."""

    for directory in PROJECT_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)
    return PROJECT_DIRECTORIES


def _smoke_check() -> None:
    """Print configuration essentials for a manual import check."""

    ensure_project_directories()
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"TRIAL_TABLE_PATH: {TRIAL_TABLE_PATH}")
    print(f"STATE_NAMES: {STATE_NAMES}")
    print(
        "Seeds: "
        f"global={GLOBAL_RANDOM_SEED}, "
        f"RL={RL_RANDOM_SEED}, "
        f"transition={TRANSITION_RANDOM_SEED}"
    )
    print("Configured directories exist:")
    for directory in PROJECT_DIRECTORIES:
        print(f"  {directory}: {directory.exists()}")


if __name__ == "__main__":
    _smoke_check()
