"""Stage 3: fit per-animal GLM-HMMs and align behavioral state labels.

Unlike a naïve concatenation, this implementation treats every session as an
independent HMM sequence.  Previous-choice regressors, initial-state
probabilities, and transition counts therefore reset at session boundaries.
Session-level holdout scores are computed with a frozen training model, after
which a final model is fitted to all sessions for trial-level posterior export.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from sklearn.linear_model import LogisticRegression

import config
from utils import (
    configure_stage_logger,
    load_table,
    require_columns,
    require_unique_key,
    sanitize_probability_matrix,
    save_table,
    state_occupancy_summary,
    validate_probability_matrix,
)

LOG_PATH = config.LOG_DIR / "03_align_glmhmm_states.log"


@dataclass(frozen=True)
class SequenceData:
    """One independent session sequence for the GLM-HMM."""

    X: np.ndarray
    y: np.ndarray
    keys: pd.DataFrame | None = None


class GLMHMM:
    """Bernoulli GLM-HMM fitted by MAP expectation-maximization.

    Emission weights have an L2/Gaussian prior and transition rows receive a
    sticky Dirichlet prior.  The model accepts multiple independent sequences,
    preventing transitions or initial conditions from crossing session
    boundaries.
    """

    def __init__(
        self,
        n_states: int = config.GLMHMM_N_STATES,
        n_features: int = config.GLMHMM_N_FEATURES,
        *,
        seed: int = config.GLOBAL_RANDOM_SEED,
        weight_prior_variance: float = config.GLMHMM_WEIGHT_PRIOR_VARIANCE,
        sticky_alpha: float = config.GLMHMM_STICKY_DIRICHLET_ALPHA,
    ) -> None:
        if n_states < 2 or n_features < 1:
            raise ValueError("n_states must be >=2 and n_features must be >=1.")
        self.K = int(n_states)
        self.D = int(n_features)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.weight_prior_variance = float(weight_prior_variance)
        self.sticky_alpha = float(sticky_alpha)
        self.W: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.pi: np.ndarray | None = None
        self.ll_history: list[float] = []
        self.converged_: bool = False
        self.n_iter_: int = 0
        self._logistic_cache: list[LogisticRegression | None] = [None] * self.K

    def _initialize(self) -> None:
        off_diagonal = 0.02 / max(self.K - 1, 1)
        self.A = np.full((self.K, self.K), off_diagonal, dtype=float)
        np.fill_diagonal(self.A, 0.98)
        self.pi = np.full(self.K, 1.0 / self.K, dtype=float)
        self.W = 0.2 * self.rng.standard_normal((self.K, self.D))
        self.W[0, 0] = 3.0
        self.ll_history = []
        self.converged_ = False
        self.n_iter_ = 0
        self._logistic_cache = [None] * self.K

    def _check_fitted(self) -> None:
        if self.W is None or self.A is None or self.pi is None:
            raise RuntimeError("GLMHMM has not been fitted.")

    def _log_emission(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._check_fitted()
        assert self.W is not None
        probability_right = np.clip(expit(X @ self.W.T), 1e-9, 1 - 1e-9)
        return (
            y[:, None] * np.log(probability_right)
            + (1.0 - y[:, None]) * np.log(1.0 - probability_right)
        )

    def _forward_backward(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        self._check_fitted()
        assert self.A is not None and self.pi is not None
        log_emission = self._log_emission(X, y)
        n_trials = len(y)
        if n_trials == 0:
            raise ValueError("An HMM sequence cannot be empty.")
        log_a = np.log(self.A + 1e-16)
        log_pi = np.log(self.pi + 1e-16)

        forward = np.zeros((n_trials, self.K), dtype=float)
        forward[0] = log_pi + log_emission[0]
        for trial in range(1, n_trials):
            forward[trial] = log_emission[trial] + logsumexp(
                forward[trial - 1][:, None] + log_a, axis=0
            )

        backward = np.zeros((n_trials, self.K), dtype=float)
        for trial in range(n_trials - 2, -1, -1):
            backward[trial] = logsumexp(
                log_a
                + log_emission[trial + 1][None, :]
                + backward[trial + 1][None, :],
                axis=1,
            )

        log_likelihood = float(logsumexp(forward[-1]))
        log_gamma = forward + backward
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        xi_counts = np.zeros((self.K, self.K), dtype=float)
        for trial in range(n_trials - 1):
            log_xi = (
                forward[trial][:, None]
                + log_a
                + log_emission[trial + 1][None, :]
                + backward[trial + 1][None, :]
            )
            xi_counts += np.exp(log_xi - logsumexp(log_xi))
        return gamma, xi_counts, log_likelihood

    def _e_step(
        self, sequences: Sequence[SequenceData]
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float]:
        gammas: list[np.ndarray] = []
        xi_total = np.zeros((self.K, self.K), dtype=float)
        initial_total = np.zeros(self.K, dtype=float)
        total_log_likelihood = 0.0
        for sequence in sequences:
            gamma, xi, log_likelihood = self._forward_backward(
                sequence.X, sequence.y
            )
            gammas.append(gamma)
            xi_total += xi
            initial_total += gamma[0]
            total_log_likelihood += log_likelihood
        return gammas, xi_total, initial_total, total_log_likelihood

    def _m_step_weights(
        self, sequences: Sequence[SequenceData], gammas: Sequence[np.ndarray]
    ) -> None:
        assert self.W is not None
        X = np.vstack([sequence.X for sequence in sequences])
        y = np.concatenate([sequence.y for sequence in sequences]).astype(int)
        gamma = np.vstack(gammas)
        for state in range(self.K):
            weights = gamma[:, state]
            if weights.sum() < 1e-8:
                continue
            if (weights * y).sum() < 1e-6 or (weights * (1 - y)).sum() < 1e-6:
                continue
            logistic = self._logistic_cache[state]
            if logistic is None:
                logistic = LogisticRegression(
                    C=self.weight_prior_variance,
                    fit_intercept=False,
                    solver="lbfgs",
                    warm_start=True,
                    max_iter=config.GLMHMM_LOGISTIC_MAX_ITERATIONS,
                    tol=config.GLMHMM_LOGISTIC_TOLERANCE,
                )
                self._logistic_cache[state] = logistic
            logistic.fit(X, y, sample_weight=weights)
            self.W[state] = logistic.coef_[0]

    def fit(
        self,
        sequences: Sequence[SequenceData],
        *,
        n_iter: int = config.GLMHMM_EM_MAX_ITERATIONS,
        absolute_tolerance: float = config.GLMHMM_EM_ABSOLUTE_TOLERANCE,
        relative_tolerance: float = config.GLMHMM_EM_RELATIVE_TOLERANCE,
        patience: int = config.GLMHMM_EM_PATIENCE,
        minimum_iterations: int = config.GLMHMM_EM_MIN_ITERATIONS,
        verbose: bool = False,
    ) -> "GLMHMM":
        """Fit the model to independent sequences and return ``self``."""

        sequences = list(sequences)
        if not sequences:
            raise ValueError("At least one nonempty sequence is required.")
        for sequence in sequences:
            if sequence.X.ndim != 2 or sequence.X.shape[1] != self.D:
                raise ValueError("Every X must have shape (trials, n_features).")
            if sequence.y.ndim != 1 or len(sequence.y) != len(sequence.X):
                raise ValueError("Every y must align one-to-one with X.")

        self._initialize()
        previous = -np.inf
        stall = 0
        for iteration in range(int(n_iter)):
            gammas, xi, initial, log_likelihood = self._e_step(sequences)
            self.ll_history.append(float(log_likelihood))
            assert self.A is not None
            self.pi = initial / initial.sum()
            prior_counts = np.eye(self.K) * max(self.sticky_alpha - 1.0, 0.0)
            transition_counts = xi + prior_counts
            row_sums = transition_counts.sum(axis=1, keepdims=True)
            self.A = np.divide(
                transition_counts,
                row_sums,
                out=np.full_like(transition_counts, 1.0 / self.K),
                where=row_sums > 0,
            )
            self._m_step_weights(sequences, gammas)
            self.n_iter_ = iteration + 1

            if verbose and (iteration % 25 == 0 or iteration == n_iter - 1):
                print(f"EM iter {iteration:3d}: loglik={log_likelihood:.3f}")
            if iteration >= minimum_iterations and np.isfinite(previous):
                gain = log_likelihood - previous
                relative_gain = gain / max(abs(previous), 1e-12)
                if gain < absolute_tolerance or relative_gain < relative_tolerance:
                    stall += 1
                else:
                    stall = 0
                if stall >= patience:
                    self.converged_ = True
                    break
            previous = log_likelihood
        else:
            self.converged_ = False
        return self

    def posterior(self, sequence: SequenceData) -> np.ndarray:
        """Return posterior state probabilities for one independent sequence."""

        return self._forward_backward(sequence.X, sequence.y)[0]

    def loglik(self, sequences: Sequence[SequenceData]) -> float:
        """Return total log likelihood across independent sequences."""

        return float(
            sum(self._forward_backward(sequence.X, sequence.y)[2] for sequence in sequences)
        )


def fit_best(
    sequences: Sequence[SequenceData],
    *,
    n_states: int = config.GLMHMM_N_STATES,
    n_features: int = config.GLMHMM_N_FEATURES,
    n_initializations: int = config.GLMHMM_N_INITIALIZATIONS,
    base_seed: int = config.GLOBAL_RANDOM_SEED,
    screening_iterations: int = config.GLMHMM_SCREENING_ITERATIONS,
    retained_initializations: int = config.GLMHMM_RETAINED_INITIALIZATIONS,
    full_iterations: int = config.GLMHMM_EM_MAX_ITERATIONS,
    verbose: bool = False,
) -> GLMHMM:
    """Screen random starts briefly, then fully fit the best starts."""

    sequences = list(sequences)
    n_initializations = max(1, int(n_initializations))
    retained = max(1, min(int(retained_initializations), n_initializations))
    screened: list[tuple[float, int]] = []
    for initialization in range(n_initializations):
        seed = base_seed + initialization
        model = GLMHMM(n_states, n_features, seed=seed).fit(
            sequences,
            n_iter=screening_iterations,
            absolute_tolerance=1e-3,
            minimum_iterations=min(5, screening_iterations),
        )
        score = model.ll_history[-1] if model.ll_history else -np.inf
        screened.append((score, seed))
        if verbose:
            print(f"screen {initialization + 1}/{n_initializations}: {score:.2f}")
    screened.sort(reverse=True)

    best_model: GLMHMM | None = None
    best_score = -np.inf
    for _, seed in screened[:retained]:
        model = GLMHMM(n_states, n_features, seed=seed).fit(
            sequences, n_iter=full_iterations, verbose=verbose
        )
        score = model.ll_history[-1] if model.ll_history else -np.inf
        if score > best_score:
            best_model = model
            best_score = score
    if best_model is None:
        raise RuntimeError("All GLM-HMM initializations failed.")
    return best_model


def build_design_matrix(session: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build Ashwood-style regressors for one session's valid-choice trials."""

    required = [
        *config.TRIAL_KEY_COLUMNS,
        config.CHOICE_COLUMN,
        config.SIGNED_CONTRAST_COLUMN,
        "rightward_choice",
    ]
    require_columns(session, required, table_name="behavioral session")
    valid = session.loc[session["rightward_choice"].notna()].copy()
    valid = valid.sort_values(config.TRIAL_INDEX_COLUMN, kind="stable").reset_index(drop=True)
    signed_contrast = pd.to_numeric(
        valid[config.SIGNED_CONTRAST_COLUMN], errors="coerce"
    ).fillna(0.0).to_numpy(float)
    scale = np.nanstd(signed_contrast)
    signed_contrast_z = signed_contrast / scale if scale > 0 else signed_contrast
    choice = valid["rightward_choice"].to_numpy(float)
    previous_choice = np.zeros(len(valid), dtype=float)
    previous_stimulus = np.zeros(len(valid), dtype=float)
    if len(valid) > 1:
        previous_choice[1:] = np.where(choice[:-1] == 1, 1.0, -1.0)
        previous_stimulus[1:] = np.sign(signed_contrast[:-1])
    X = np.column_stack(
        [signed_contrast_z, np.ones(len(valid)), previous_choice, previous_stimulus]
    )
    return X, choice, valid


def relabel_states_by_engagement(model: GLMHMM) -> tuple[np.ndarray, list[str]]:
    """Return engaged, biased-left, biased-right order from emission weights."""

    model._check_fitted()
    assert model.W is not None
    engaged = int(np.argmax(np.abs(model.W[:, 0])))
    remaining = [state for state in range(model.K) if state != engaged]
    remaining = sorted(remaining, key=lambda state: model.W[state, 1])
    if len(remaining) != 2:
        raise ValueError("The current relabeling rule requires exactly three states.")
    return np.asarray([engaged, remaining[0], remaining[1]]), [
        config.STATE_NAMES[index] for index in range(3)
    ]


def _split_session_indices(n_sessions: int, seed: int) -> tuple[list[int], list[int]]:
    if n_sessions < 2 or config.GLMHMM_TEST_SESSION_FRACTION <= 0:
        return list(range(n_sessions)), []
    n_test = max(1, int(round(config.GLMHMM_TEST_SESSION_FRACTION * n_sessions)))
    n_test = min(n_test, n_sessions - 1)
    rng = np.random.default_rng(seed)
    test = sorted(rng.choice(n_sessions, size=n_test, replace=False).tolist())
    train = [index for index in range(n_sessions) if index not in set(test)]
    return train, test


def _score_per_trial(model: GLMHMM, sequences: Sequence[SequenceData]) -> tuple[float, int]:
    n_trials = sum(len(sequence.y) for sequence in sequences)
    return (model.loglik(sequences) / n_trials if n_trials else np.nan), n_trials


def fit_subject(
    subject_table: pd.DataFrame,
    *,
    seed: int,
    n_initializations: int = config.GLMHMM_N_INITIALIZATIONS,
) -> tuple[GLMHMM, pd.DataFrame, dict[str, float | int | bool], np.ndarray]:
    """Fit one subject, score session holdout, and return trial posteriors."""

    sequences: list[SequenceData] = []
    valid_tables: list[pd.DataFrame] = []
    session_ids: list[str] = []
    for eid, session in subject_table.groupby(config.SESSION_COLUMN, sort=True):
        X, y, valid = build_design_matrix(session)
        if len(y) == 0:
            continue
        sequences.append(SequenceData(X=X, y=y, keys=valid[list(config.TRIAL_KEY_COLUMNS)]))
        valid_tables.append(valid)
        session_ids.append(str(eid))
    if not sequences:
        raise ValueError("Subject has no valid-choice sessions.")

    train_indices, test_indices = _split_session_indices(len(sequences), seed)
    cv_model = fit_best(
        [sequences[index] for index in train_indices],
        n_initializations=n_initializations,
        base_seed=seed,
    )
    train_ll, n_train = _score_per_trial(
        cv_model, [sequences[index] for index in train_indices]
    )
    test_ll, n_test = _score_per_trial(
        cv_model, [sequences[index] for index in test_indices]
    )

    final_model = fit_best(
        sequences,
        n_initializations=n_initializations,
        base_seed=seed + 10_000,
    )
    order, labels = relabel_states_by_engagement(final_model)
    posterior_tables: list[pd.DataFrame] = []
    for sequence, valid, eid in zip(sequences, valid_tables, session_ids):
        posterior = final_model.posterior(sequence)[:, order]
        posterior, valid_rows = sanitize_probability_matrix(posterior)
        if not valid_rows.all():
            raise ValueError(f"Invalid state posterior rows for subject session {eid}.")
        validate_probability_matrix(posterior)
        output = valid[list(config.TRIAL_KEY_COLUMNS)].copy()
        for state, column in enumerate(config.STATE_POSTERIOR_COLUMNS):
            output[column] = posterior[:, state]
        output[config.STATE_COLUMN] = np.argmax(posterior, axis=1).astype(int)
        output[config.STATE_LABEL_COLUMN] = output[config.STATE_COLUMN].map(config.STATE_NAMES)
        posterior_tables.append(output)

    diagnostics: dict[str, float | int | bool] = {
        "n_sessions": len(sequences),
        "n_train_sessions": len(train_indices),
        "n_test_sessions": len(test_indices),
        "n_train_trials": n_train,
        "n_test_trials": n_test,
        "train_loglik_per_trial": train_ll,
        "test_loglik_per_trial": test_ll,
        "final_loglik": final_model.ll_history[-1],
        "final_iterations": final_model.n_iter_,
        "converged": final_model.converged_,
    }
    return final_model, pd.concat(posterior_tables, ignore_index=True), diagnostics, order


def save_subject_model(
    subject: str, model: GLMHMM, order: np.ndarray
) -> Path:
    """Save an aligned subject model as a compressed NPZ file."""

    model._check_fitted()
    assert model.W is not None and model.A is not None and model.pi is not None
    path = config.GLMHMM_MODEL_DIR / f"{subject}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        subject=np.asarray(subject),
        W=model.W[order],
        A=model.A[np.ix_(order, order)],
        pi=model.pi[order],
        original_state_order=order,
        ll_history=np.asarray(model.ll_history),
        feature_names=np.asarray(config.GLMHMM_FEATURE_NAMES),
        state_labels=np.asarray([config.STATE_NAMES[index] for index in range(3)]),
    )
    return path


def parameter_recovery(
    *,
    n_trials: int = config.GLMHMM_RECOVERY_TRIALS,
    seed: int = config.GLOBAL_RANDOM_SEED,
    n_initializations: int = 6,
) -> pd.DataFrame:
    """Run a compact synthetic parameter/state recovery check."""

    rng = np.random.default_rng(seed)
    true_W = np.array(
        [[5.0, 0.0, 0.6, 0.2], [0.8, -2.5, 1.0, 0.1], [0.8, 2.5, 1.0, 0.1]]
    )
    true_A = np.array([[0.97, 0.015, 0.015], [0.03, 0.95, 0.02], [0.03, 0.02, 0.95]])
    true_pi = np.array([0.8, 0.1, 0.1])
    sequence_lengths = np.full(4, n_trials // 4, dtype=int)
    sequence_lengths[: n_trials % 4] += 1
    sequences: list[SequenceData] = []
    true_states: list[np.ndarray] = []
    for sequence_length in sequence_lengths:
        contrast = rng.choice(
            [-1, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1],
            size=sequence_length,
        )
        X = np.column_stack(
            [
                contrast,
                np.ones(sequence_length),
                rng.choice([-1, 1], sequence_length),
                rng.choice([-1, 1], sequence_length),
            ]
        )
        states = np.zeros(sequence_length, dtype=int)
        states[0] = rng.choice(3, p=true_pi)
        for trial in range(1, sequence_length):
            states[trial] = rng.choice(3, p=true_A[states[trial - 1]])
        probability = expit(np.sum(X * true_W[states], axis=1))
        y = (rng.random(sequence_length) < probability).astype(float)
        sequences.append(SequenceData(X, y))
        true_states.append(states)

    fitted = fit_best(
        sequences,
        n_initializations=n_initializations,
        retained_initializations=min(3, n_initializations),
        screening_iterations=15,
        full_iterations=100,
        base_seed=seed + 1,
    )
    decoded = np.concatenate(
        [np.argmax(fitted.posterior(sequence), axis=1) for sequence in sequences]
    )
    truth = np.concatenate(true_states)
    best_permutation = max(
        permutations(range(3)),
        key=lambda mapping: np.mean([mapping[state] == target for state, target in zip(decoded, truth)]),
    )
    accuracy = float(
        np.mean([best_permutation[state] == target for state, target in zip(decoded, truth)])
    )
    return pd.DataFrame(
        [
            {
                "n_trials": n_trials,
                "decode_accuracy": accuracy,
                "minimum_required": config.GLMHMM_RECOVERY_MIN_DECODE_ACCURACY,
                "passed": accuracy >= config.GLMHMM_RECOVERY_MIN_DECODE_ACCURACY,
                "fitted_loglik": fitted.ll_history[-1],
            }
        ]
    )


def psychometric_qc(
    trials: pd.DataFrame, states: pd.DataFrame
) -> pd.DataFrame:
    """Create a saved state-conditioned psychometric sanity table."""

    joined = states.merge(
        trials[
            [
                *config.TRIAL_KEY_COLUMNS,
                config.SIGNED_CONTRAST_COLUMN,
                "rightward_choice",
                config.EPOCH_COLUMN,
            ]
        ],
        on=list(config.TRIAL_KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    return (
        joined.groupby(
            [config.STATE_LABEL_COLUMN, config.EPOCH_COLUMN, config.SIGNED_CONTRAST_COLUMN],
            dropna=False,
        )
        .agg(
            probability_right=("rightward_choice", "mean"),
            n_trials=("rightward_choice", "size"),
        )
        .reset_index()
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=config.TRIAL_TABLE_PATH)
    parser.add_argument("--output", type=Path, default=config.GLMHMM_STATES_PATH)
    parser.add_argument("--skip-recovery", action="store_true")
    parser.add_argument(
        "--n-initializations", type=int, default=config.GLMHMM_N_INITIALIZATIONS
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config.ensure_project_directories()
    logger = configure_stage_logger("03_align_glmhmm_states", log_path=LOG_PATH)
    trials = load_table(args.trials)
    require_unique_key(trials, config.TRIAL_KEY_COLUMNS, table_name="trial table")
    require_columns(
        trials,
        [
            config.SUBJECT_COLUMN,
            config.SESSION_COLUMN,
            config.SIGNED_CONTRAST_COLUMN,
            "rightward_choice",
        ],
        table_name="trial table",
    )

    if not args.skip_recovery:
        recovery = parameter_recovery(
            n_initializations=min(args.n_initializations, 6)
        )
        save_table(recovery, config.GLMHMM_RECOVERY_PATH)
        if not bool(recovery["passed"].iloc[0]):
            raise RuntimeError(
                "GLM-HMM parameter recovery did not meet the configured minimum."
            )
        logger.info("Parameter recovery passed: %s", recovery.iloc[0].to_dict())

    state_tables: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, object]] = []
    for subject_index, (subject, subject_table) in enumerate(
        trials.groupby(config.SUBJECT_COLUMN, sort=True)
    ):
        try:
            model, state_table, diagnostics, order = fit_subject(
                subject_table,
                seed=config.GLOBAL_RANDOM_SEED + subject_index * 100,
                n_initializations=args.n_initializations,
            )
            model_path = save_subject_model(str(subject), model, order)
            state_tables.append(state_table)
            diagnostics_rows.append(
                {
                    config.SUBJECT_COLUMN: str(subject),
                    "status": "success",
                    "model_path": str(model_path),
                    **diagnostics,
                }
            )
            logger.info("Fitted %s (%d state trials)", subject, len(state_table))
        except Exception as error:
            diagnostics_rows.append(
                {
                    config.SUBJECT_COLUMN: str(subject),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            logger.exception("GLM-HMM failed for %s", subject)

    if not state_tables:
        raise RuntimeError("No subject GLM-HMM was fitted successfully.")
    states = pd.concat(state_tables, ignore_index=True).sort_values(
        list(config.TRIAL_KEY_COLUMNS), kind="stable"
    )
    require_unique_key(states, config.TRIAL_KEY_COLUMNS, table_name="GLM-HMM states")
    validate_probability_matrix(states[list(config.STATE_POSTERIOR_COLUMNS)].to_numpy())

    occupancy = state_occupancy_summary(
        states,
        state_column=config.STATE_COLUMN,
        group_columns=[config.SUBJECT_COLUMN],
        state_names=config.STATE_NAMES,
    )
    diagnostics = pd.DataFrame(diagnostics_rows)
    save_table(states, args.output)
    save_table(diagnostics, config.GLMHMM_DIAGNOSTICS_PATH)
    save_table(occupancy, config.STATE_OCCUPANCY_PATH)
    save_table(psychometric_qc(trials, states), config.GLMHMM_PSYCHOMETRIC_QC_PATH)
    logger.info(
        "Stage 3 complete: %d subjects, %d state trials",
        states[config.SUBJECT_COLUMN].nunique(),
        len(states),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
