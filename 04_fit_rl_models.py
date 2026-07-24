"""Stage 4: fit sensory-only and hybrid sensory–Q-learning choice models.

The hybrid baseline preserves the model discussed during the notebook analysis:

    P(right_t) = lapse/2 + (1-lapse) * sigmoid(
        bias + beta_stimulus * signed_contrast_t
             + beta_value * (Q_right_t - Q_left_t)
    )

Only the chosen action value is updated and Q values reset at every session.
Session-level holdout prediction quantifies the incremental value of the Q term.
Boundary fits are reported, not automatically excluded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

import config
from utils import (
    configure_stage_logger,
    load_table,
    require_columns,
    require_unique_key,
    save_table,
)

LOG_PATH = config.LOG_DIR / "04_fit_rl_models.log"
FULL_PARAMETER_NAMES = ("alpha", "beta_value", "beta_stimulus", "bias", "lapse")
SENSORY_PARAMETER_NAMES = ("beta_stimulus", "bias", "lapse")


@dataclass(frozen=True)
class RLSequence:
    """One independent behavioral session for RL fitting."""

    choice_right: np.ndarray
    reward: np.ndarray
    signed_stimulus: np.ndarray
    keys: pd.DataFrame | None = None


@dataclass(frozen=True)
class FitResult:
    """Compact optimizer result used by both model variants."""

    parameters: dict[str, float]
    negative_log_likelihood: float
    success: bool
    message: str
    n_iterations: int


def _validate_sequence(sequence: RLSequence) -> None:
    n = len(sequence.choice_right)
    if len(sequence.reward) != n or len(sequence.signed_stimulus) != n:
        raise ValueError("choice, reward, and stimulus arrays must have equal length.")
    if not np.isin(sequence.choice_right, [0.0, 1.0]).all():
        raise ValueError("choice_right must contain only 0 and 1.")
    if not np.isin(sequence.reward, [0.0, 1.0]).all():
        raise ValueError("reward must contain only 0 and 1.")


def hybrid_sequence(
    sequence: RLSequence,
    parameters: Mapping[str, float],
    *,
    return_trial_values: bool = False,
) -> tuple[float, pd.DataFrame | None]:
    """Evaluate one session and optionally return trial-wise latent variables."""

    _validate_sequence(sequence)
    alpha = float(parameters["alpha"])
    beta_value = float(parameters["beta_value"])
    beta_stimulus = float(parameters["beta_stimulus"])
    bias = float(parameters["bias"])
    lapse = float(parameters["lapse"])
    epsilon = config.RL_PROBABILITY_EPSILON

    q_left = config.RL_INITIAL_Q_LEFT
    q_right = config.RL_INITIAL_Q_RIGHT
    nll = 0.0
    rows: list[dict[str, float]] = []
    for choice, reward, stimulus in zip(
        sequence.choice_right, sequence.reward, sequence.signed_stimulus
    ):
        q_difference = q_right - q_left
        linear = bias + beta_stimulus * stimulus + beta_value * q_difference
        probability_right = lapse / 2.0 + (1.0 - lapse) * expit(linear)
        probability_right = float(np.clip(probability_right, epsilon, 1.0 - epsilon))
        choice_probability = probability_right if choice == 1 else 1.0 - probability_right
        nll -= float(np.log(np.clip(choice_probability, epsilon, 1.0)))

        chosen_q = q_right if choice == 1 else q_left
        prediction_error = reward - chosen_q
        outcome_probability = chosen_q if reward == 1 else 1.0 - chosen_q
        outcome_surprise = -np.log(np.clip(outcome_probability, epsilon, 1.0))
        choice_entropy = -(
            probability_right * np.log(probability_right)
            + (1.0 - probability_right) * np.log(1.0 - probability_right)
        )
        rows.append(
            {
                "rl_q_left": q_left,
                "rl_q_right": q_right,
                "rl_q_difference": q_difference,
                "rl_expected_reward": chosen_q,
                "rl_rpe": prediction_error,
                "rl_rpe_positive": max(prediction_error, 0.0),
                "rl_rpe_negative": max(-prediction_error, 0.0),
                "rl_outcome_surprise": outcome_surprise,
                "rl_probability_right": probability_right,
                "rl_choice_probability": choice_probability,
                "rl_choice_entropy": choice_entropy,
            }
        )
        if choice == 1:
            q_right += alpha * prediction_error
        else:
            q_left += alpha * prediction_error

    return nll, pd.DataFrame(rows) if return_trial_values else None


def sensory_sequence(
    sequence: RLSequence,
    parameters: Mapping[str, float],
) -> float:
    """Evaluate the sensory-only lapse-logistic model on one session."""

    _validate_sequence(sequence)
    beta_stimulus = float(parameters["beta_stimulus"])
    bias = float(parameters["bias"])
    lapse = float(parameters["lapse"])
    probability_right = lapse / 2.0 + (1.0 - lapse) * expit(
        bias + beta_stimulus * sequence.signed_stimulus
    )
    probability_right = np.clip(
        probability_right,
        config.RL_PROBABILITY_EPSILON,
        1.0 - config.RL_PROBABILITY_EPSILON,
    )
    probability_choice = np.where(
        sequence.choice_right == 1, probability_right, 1.0 - probability_right
    )
    return float(-np.log(probability_choice).sum())


def _bounds(parameter_names: Sequence[str]) -> list[tuple[float, float]]:
    return [config.RL_PARAMETER_BOUNDS[name] for name in parameter_names]


def _initial_point(
    parameter_names: Sequence[str], rng: np.random.Generator, restart: int
) -> np.ndarray:
    defaults = {
        "alpha": 0.2,
        "beta_value": 1.0,
        "beta_stimulus": 5.0,
        "bias": 0.0,
        "lapse": 0.02,
    }
    if restart == 0:
        return np.asarray([defaults[name] for name in parameter_names], dtype=float)
    values = []
    for name, (lower, upper) in zip(parameter_names, _bounds(parameter_names)):
        if name == "lapse":
            values.append(float(rng.uniform(lower, min(upper, 0.08))))
        elif name == "alpha":
            values.append(float(rng.uniform(0.03, 0.8)))
        else:
            values.append(float(rng.uniform(lower, upper)))
    return np.asarray(values, dtype=float)


def fit_model(
    sequences: Sequence[RLSequence],
    *,
    model: str,
    seed: int,
    n_restarts: int = config.RL_N_RESTARTS,
) -> FitResult:
    """Fit a model with deterministic bounded multi-start optimization."""

    sequences = list(sequences)
    if not sequences:
        raise ValueError("At least one RL sequence is required.")
    for sequence in sequences:
        _validate_sequence(sequence)

    if model == "hybrid":
        names = FULL_PARAMETER_NAMES

        def objective(vector: np.ndarray) -> float:
            parameters = dict(zip(names, vector))
            return float(sum(hybrid_sequence(s, parameters)[0] for s in sequences))

    elif model == "sensory":
        names = SENSORY_PARAMETER_NAMES

        def objective(vector: np.ndarray) -> float:
            parameters = dict(zip(names, vector))
            return float(sum(sensory_sequence(s, parameters) for s in sequences))

    else:
        raise ValueError("model must be 'hybrid' or 'sensory'.")

    rng = np.random.default_rng(seed)
    best = None
    for restart in range(max(1, int(n_restarts))):
        result = minimize(
            objective,
            _initial_point(names, rng, restart),
            method="L-BFGS-B",
            bounds=_bounds(names),
            options={"maxiter": config.RL_OPTIMIZER_MAX_ITERATIONS},
        )
        if best is None or result.fun < best.fun:
            best = result
    if best is None:
        raise RuntimeError("RL optimizer produced no result.")
    return FitResult(
        parameters={name: float(value) for name, value in zip(names, best.x)},
        negative_log_likelihood=float(best.fun),
        success=bool(best.success),
        message=str(best.message),
        n_iterations=int(getattr(best, "nit", -1)),
    )


def build_subject_sequences(subject_table: pd.DataFrame) -> list[RLSequence]:
    """Build choice/reward arrays and reset Q values at each session boundary."""

    required = [
        *config.TRIAL_KEY_COLUMNS,
        "rightward_choice",
        config.REWARD_COLUMN,
        config.SIGNED_CONTRAST_COLUMN,
    ]
    require_columns(subject_table, required, table_name="subject trial table")
    sequences: list[RLSequence] = []
    for _, session in subject_table.groupby(config.SESSION_COLUMN, sort=True):
        valid = session.loc[
            session["rightward_choice"].notna() & session[config.REWARD_COLUMN].notna()
        ].sort_values(config.TRIAL_INDEX_COLUMN, kind="stable")
        if valid.empty:
            continue
        sequences.append(
            RLSequence(
                choice_right=valid["rightward_choice"].to_numpy(float),
                reward=valid[config.REWARD_COLUMN].to_numpy(float),
                signed_stimulus=pd.to_numeric(
                    valid[config.SIGNED_CONTRAST_COLUMN], errors="coerce"
                ).fillna(0.0).to_numpy(float),
                keys=valid[list(config.TRIAL_KEY_COLUMNS)].reset_index(drop=True),
            )
        )
    return sequences


def split_session_indices(n_sessions: int, seed: int) -> tuple[list[int], list[int]]:
    """Deterministically split whole sessions into train and test sets."""

    if n_sessions < 2 or config.RL_TEST_SESSION_FRACTION <= 0:
        return list(range(n_sessions)), []
    n_test = max(1, int(round(config.RL_TEST_SESSION_FRACTION * n_sessions)))
    n_test = min(n_test, n_sessions - 1)
    rng = np.random.default_rng(seed)
    test = sorted(rng.choice(n_sessions, size=n_test, replace=False).tolist())
    test_set = set(test)
    return [index for index in range(n_sessions) if index not in test_set], test


def score_model(
    sequences: Sequence[RLSequence], parameters: Mapping[str, float], model: str
) -> tuple[float, int]:
    """Return log likelihood and trial count for a frozen parameter set."""

    n_trials = sum(len(sequence.choice_right) for sequence in sequences)
    if model == "hybrid":
        nll = sum(hybrid_sequence(sequence, parameters)[0] for sequence in sequences)
    elif model == "sensory":
        nll = sum(sensory_sequence(sequence, parameters) for sequence in sequences)
    else:
        raise ValueError("Unknown model.")
    return -float(nll), n_trials


def boundary_flags(parameters: Mapping[str, float]) -> dict[str, bool]:
    """Flag estimates lying near either configured optimization boundary."""

    flags: dict[str, bool] = {}
    tolerance = config.RL_BOUNDARY_FRACTION_TOLERANCE
    for name, value in parameters.items():
        lower, upper = config.RL_PARAMETER_BOUNDS[name]
        width = upper - lower
        flags[f"{name}_low"] = value <= lower + tolerance * width
        flags[f"{name}_high"] = value >= upper - tolerance * width
    flags["any_boundary"] = any(flags.values())
    return flags


def generate_trial_regressors(
    sequences: Sequence[RLSequence], parameters: Mapping[str, float]
) -> pd.DataFrame:
    """Generate trial-wise Q values and prediction-error variables."""

    pieces = []
    for sequence in sequences:
        _, values = hybrid_sequence(sequence, parameters, return_trial_values=True)
        assert values is not None and sequence.keys is not None
        piece = pd.concat(
            [sequence.keys.reset_index(drop=True), values.reset_index(drop=True)], axis=1
        )
        piece["rl_choice_right"] = sequence.choice_right
        piece["rl_reward"] = sequence.reward
        piece["rl_signed_stimulus"] = sequence.signed_stimulus
        pieces.append(piece)
    output = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if not output.empty:
        output["rl_outcome_surprise_lag1"] = output.groupby(
            [config.SUBJECT_COLUMN, config.SESSION_COLUMN], sort=False
        )["rl_outcome_surprise"].shift(1)
        output["rl_recent_reward_ewm"] = output.groupby(
            [config.SUBJECT_COLUMN, config.SESSION_COLUMN], sort=False
        )["rl_reward"].transform(
            lambda values: values.shift(1).ewm(alpha=0.2, adjust=False).mean()
        )
    return output


def fit_subject(
    subject: str,
    subject_table: pd.DataFrame,
    *,
    seed: int,
    n_restarts: int,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object], dict[str, object]]:
    """Fit final parameters, holdout comparison, and trial regressors."""

    sequences = build_subject_sequences(subject_table)
    n_trials = sum(len(sequence.choice_right) for sequence in sequences)
    if n_trials < config.RL_MIN_VALID_CHOICES:
        raise ValueError(
            f"Only {n_trials} valid choices; minimum is {config.RL_MIN_VALID_CHOICES}."
        )
    train_indices, test_indices = split_session_indices(len(sequences), seed)
    train = [sequences[index] for index in train_indices]
    test = [sequences[index] for index in test_indices]

    hybrid_cv = fit_model(train, model="hybrid", seed=seed, n_restarts=n_restarts)
    sensory_cv = fit_model(
        train, model="sensory", seed=seed + 1_000, n_restarts=n_restarts
    )
    hybrid_test_ll, n_test = score_model(test, hybrid_cv.parameters, "hybrid")
    sensory_test_ll, _ = score_model(test, sensory_cv.parameters, "sensory")
    hybrid_train_ll, n_train = score_model(train, hybrid_cv.parameters, "hybrid")
    sensory_train_ll, _ = score_model(train, sensory_cv.parameters, "sensory")

    final_fit = fit_model(
        sequences, model="hybrid", seed=seed + 10_000, n_restarts=n_restarts
    )
    regressors = generate_trial_regressors(sequences, final_fit.parameters)
    regressors[config.SUBJECT_COLUMN] = str(subject)

    parameter_row: dict[str, object] = {
        config.SUBJECT_COLUMN: str(subject),
        **final_fit.parameters,
        "negative_log_likelihood": final_fit.negative_log_likelihood,
        "optimizer_success": final_fit.success,
        "optimizer_message": final_fit.message,
        "optimizer_iterations": final_fit.n_iterations,
        "n_trials": n_trials,
        "n_sessions": len(sequences),
        **boundary_flags(final_fit.parameters),
    }
    comparison_row: dict[str, object] = {
        config.SUBJECT_COLUMN: str(subject),
        "n_train_sessions": len(train_indices),
        "n_test_sessions": len(test_indices),
        "n_train_trials": n_train,
        "n_test_trials": n_test,
        "hybrid_train_loglik_per_trial": hybrid_train_ll / n_train if n_train else np.nan,
        "sensory_train_loglik_per_trial": sensory_train_ll / n_train if n_train else np.nan,
        "hybrid_test_loglik_per_trial": hybrid_test_ll / n_test if n_test else np.nan,
        "sensory_test_loglik_per_trial": sensory_test_ll / n_test if n_test else np.nan,
        "test_delta_loglik_per_trial": (
            (hybrid_test_ll - sensory_test_ll) / n_test if n_test else np.nan
        ),
    }
    failure = 1.0 - regressors["rl_reward"]
    correlation = regressors["rl_rpe_negative"].corr(failure)
    diagnostic_row: dict[str, object] = {
        config.SUBJECT_COLUMN: str(subject),
        "status": "success",
        "n_trials": n_trials,
        "negative_rpe_failure_correlation": correlation,
        "q_difference_sd": regressors["rl_q_difference"].std(ddof=0),
        "expected_reward_sd": regressors["rl_expected_reward"].std(ddof=0),
        "any_boundary": parameter_row["any_boundary"],
    }
    return regressors, parameter_row, comparison_row, diagnostic_row


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=config.TRIAL_TABLE_PATH)
    parser.add_argument("--output", type=Path, default=config.RL_REGRESSORS_PATH)
    parser.add_argument("--n-restarts", type=int, default=config.RL_N_RESTARTS)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config.ensure_project_directories()
    logger = configure_stage_logger("04_fit_rl_models", log_path=LOG_PATH)
    trials = load_table(args.trials)
    require_unique_key(trials, config.TRIAL_KEY_COLUMNS, table_name="trial table")
    require_columns(
        trials,
        ["rightward_choice", config.REWARD_COLUMN, config.SIGNED_CONTRAST_COLUMN],
        table_name="trial table",
    )

    if float(pd.Series([-1]).map({-1: 1}).iloc[0]) != 1.0:
        raise AssertionError("Choice-direction smoke check failed.")
    positive = trials.loc[trials[config.SIGNED_CONTRAST_COLUMN] > 0, "rightward_choice"].mean()
    negative = trials.loc[trials[config.SIGNED_CONTRAST_COLUMN] < 0, "rightward_choice"].mean()
    if np.isfinite(positive) and np.isfinite(negative) and positive <= negative:
        raise AssertionError("Signed contrast direction failed the behavioral sanity check.")

    regressor_pieces: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for subject_index, (subject, subject_table) in enumerate(
        trials.groupby(config.SUBJECT_COLUMN, sort=True)
    ):
        if str(subject) in config.RL_SUBJECT_EXCLUSIONS:
            diagnostic_rows.append(
                {
                    config.SUBJECT_COLUMN: str(subject),
                    "status": "preexisting_exclusion",
                    "reason": config.RL_SUBJECT_EXCLUSIONS[str(subject)],
                }
            )
            continue
        try:
            regressors, parameters, comparison, diagnostics = fit_subject(
                str(subject),
                subject_table,
                seed=config.RL_RANDOM_SEED + subject_index * 100,
                n_restarts=args.n_restarts,
            )
            regressor_pieces.append(regressors)
            parameter_rows.append(parameters)
            comparison_rows.append(comparison)
            diagnostic_rows.append(diagnostics)
            logger.info("Fitted RL models for %s (%d trials)", subject, len(regressors))
        except Exception as error:
            diagnostic_rows.append(
                {
                    config.SUBJECT_COLUMN: str(subject),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            logger.exception("RL fitting failed for %s", subject)

    if not regressor_pieces:
        raise RuntimeError("No subject RL model was fitted successfully.")
    regressors = pd.concat(regressor_pieces, ignore_index=True).sort_values(
        list(config.TRIAL_KEY_COLUMNS), kind="stable"
    )
    require_unique_key(regressors, config.TRIAL_KEY_COLUMNS, table_name="RL regressors")
    parameters = pd.DataFrame(parameter_rows)
    comparison = pd.DataFrame(comparison_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    overall_correlation = regressors["rl_rpe_negative"].corr(1.0 - regressors["rl_reward"])
    if not diagnostics.empty:
        diagnostics["cohort_negative_rpe_failure_correlation"] = overall_correlation

    save_table(regressors, args.output)
    save_table(parameters, config.RL_PARAMETER_PATH)
    save_table(comparison, config.RL_MODEL_COMPARISON_PATH)
    save_table(diagnostics, config.RL_DIAGNOSTICS_PATH)
    logger.info(
        "Stage 4 complete: %d subjects, %d trials, cohort corr=%.4f",
        regressors[config.SUBJECT_COLUMN].nunique(),
        len(regressors),
        overall_correlation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
