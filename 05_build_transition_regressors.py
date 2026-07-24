"""Stage 5: join stage outputs and construct transition predictors/outcomes.

This script performs data engineering only.  It creates sequence-safe posterior
entropy, next-trial Jensen–Shannon divergence, hard state switches, future
lability windows, feedback histories, block/session position variables, and
deterministic isolated pupil-burst indicators.  It does not fit inferential
models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config
from utils import (
    add_future_window_summary,
    add_trial_position,
    configure_stage_logger,
    jensen_shannon_divergence,
    load_table,
    posterior_entropy,
    require_columns,
    require_unique_key,
    sanitize_probability_matrix,
    save_table,
    validate_probability_matrix,
)

LOG_PATH = config.LOG_DIR / "05_build_transition_regressors.log"
GROUP_COLUMNS = [config.SUBJECT_COLUMN, config.SESSION_COLUMN]


def _one_to_one_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    name: str,
    how: str = "left",
) -> pd.DataFrame:
    """Merge on canonical keys after validating both sides."""

    require_unique_key(left, config.TRIAL_KEY_COLUMNS, table_name="merge base")
    require_unique_key(right, config.TRIAL_KEY_COLUMNS, table_name=name)
    return left.merge(
        right,
        on=list(config.TRIAL_KEY_COLUMNS),
        how=how,
        validate="one_to_one",
        suffixes=("", f"_{name}"),
    )


def join_stage_outputs(
    trials: pd.DataFrame,
    pupil: pd.DataFrame,
    states: pd.DataFrame,
    rl: pd.DataFrame,
) -> pd.DataFrame:
    """Join all stage outputs on valid-choice GLM-HMM trial keys."""

    require_unique_key(trials, config.TRIAL_KEY_COLUMNS, table_name="trial table")
    base = trials.merge(
        states[list(config.TRIAL_KEY_COLUMNS)],
        on=list(config.TRIAL_KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    state_columns = [
        *config.TRIAL_KEY_COLUMNS,
        *config.STATE_POSTERIOR_COLUMNS,
        config.STATE_COLUMN,
        config.STATE_LABEL_COLUMN,
    ]
    base = _one_to_one_merge(base, states[state_columns], name="states", how="left")

    pupil_columns = [
        column
        for column in pupil.columns
        if column in config.TRIAL_KEY_COLUMNS
        or column.startswith("pupil_")
    ]
    base = _one_to_one_merge(base, pupil[pupil_columns], name="pupil", how="left")
    rl_columns = [
        column
        for column in rl.columns
        if column in config.TRIAL_KEY_COLUMNS or column.startswith("rl_")
    ]
    base = _one_to_one_merge(base, rl[rl_columns], name="rl", how="left")
    return base.sort_values(list(config.TRIAL_KEY_COLUMNS), kind="stable").reset_index(drop=True)


def add_state_lability(table: pd.DataFrame) -> pd.DataFrame:
    """Add posterior entropy, next-trial JSD, and future transition outcomes."""

    result = table.copy()
    probabilities, valid = sanitize_probability_matrix(
        result[list(config.STATE_POSTERIOR_COLUMNS)].to_numpy(float)
    )
    if not valid.all():
        raise ValueError(f"Found {int((~valid).sum())} invalid state-posterior rows.")
    validate_probability_matrix(probabilities)
    result.loc[:, list(config.STATE_POSTERIOR_COLUMNS)] = probabilities
    result["posterior_entropy"] = posterior_entropy(probabilities)

    for state_column in config.STATE_POSTERIOR_COLUMNS:
        result[f"{state_column}_next"] = result.groupby(
            GROUP_COLUMNS, sort=False
        )[state_column].shift(-1)
    result["state_next"] = result.groupby(GROUP_COLUMNS, sort=False)[
        config.STATE_COLUMN
    ].shift(-1)

    current = result[list(config.STATE_POSTERIOR_COLUMNS)].to_numpy(float)
    future = result[[f"{column}_next" for column in config.STATE_POSTERIOR_COLUMNS]].to_numpy(float)
    result["posterior_js_to_next"] = jensen_shannon_divergence(current, future)
    result["hard_switch"] = (
        result["state_next"].notna()
        & result["state_next"].ne(result[config.STATE_COLUMN])
    ).astype(float)
    result.loc[result["state_next"].isna(), "hard_switch"] = np.nan

    horizon = config.FUTURE_LABILITY_TRIALS
    # posterior_js_to_next at row t already describes the transition t -> t+1.
    # A horizon of three therefore uses offsets 0, 1, and 2: exactly three
    # consecutive transitions, not the current transition plus three more.
    for value_column, output_column in (
        ("posterior_js_to_next", f"future_js_max_{horizon}"),
        ("hard_switch", f"future_switch_any_{horizon}"),
    ):
        temporary = []
        grouped = result.groupby(GROUP_COLUMNS, sort=False)[value_column]
        for offset in range(horizon):
            column = f"__{value_column}_offset_{offset}"
            result[column] = grouped.shift(-offset)
            temporary.append(column)
        result[output_column] = result[temporary].max(axis=1, skipna=True)
        result = result.drop(columns=temporary)
    return result


def add_feedback_and_position(table: pd.DataFrame) -> pd.DataFrame:
    """Add failure histories, block transitions, and session progress."""

    result = add_trial_position(table, group_columns=GROUP_COLUMNS)
    reward_source = "rl_reward" if "rl_reward" in result else config.REWARD_COLUMN
    require_columns(result, [reward_source, config.BLOCK_PRIOR_COLUMN])
    result["reward"] = pd.to_numeric(result[reward_source], errors="coerce")
    result["failure"] = 1.0 - result["reward"]

    grouped_reward = result.groupby(GROUP_COLUMNS, sort=False)["reward"]
    grouped_failure = result.groupby(GROUP_COLUMNS, sort=False)["failure"]
    for lag in range(1, 6):
        result[f"reward_lag{lag}"] = grouped_reward.shift(lag)
        result[f"failure_lag{lag}"] = grouped_failure.shift(lag)
    result["recent_reward_5"] = pd.concat(
        [result[f"reward_lag{lag}"] for lag in range(1, 6)], axis=1
    ).mean(axis=1, skipna=True)
    result["recent_failure_5"] = pd.concat(
        [result[f"failure_lag{lag}"] for lag in range(1, 6)], axis=1
    ).mean(axis=1, skipna=True)

    previous_prior = result.groupby(GROUP_COLUMNS, sort=False)[
        config.BLOCK_PRIOR_COLUMN
    ].shift(1)
    result["block_changed"] = (
        previous_prior.notna()
        & ~np.isclose(
            pd.to_numeric(result[config.BLOCK_PRIOR_COLUMN], errors="coerce"),
            pd.to_numeric(previous_prior, errors="coerce"),
            equal_nan=True,
        )
    )
    result["biased_block_changed"] = result["block_changed"] & ~np.isclose(
        pd.to_numeric(result[config.BLOCK_PRIOR_COLUMN], errors="coerce"), 0.5
    )

    def trials_since_change(group: pd.DataFrame) -> pd.Series:
        change_index = np.where(group["block_changed"].to_numpy(bool), np.arange(len(group)), np.nan)
        last_change = pd.Series(change_index).ffill().to_numpy(float)
        return pd.Series(np.arange(len(group)) - last_change, index=group.index).fillna(np.nan)

    result["trials_since_block_change"] = result.groupby(
        GROUP_COLUMNS, sort=False, group_keys=False
    ).apply(trials_since_change, include_groups=False)
    result["position_bin"] = np.minimum(
        np.floor(result["trial_progress"] * config.N_POSITION_BINS),
        config.N_POSITION_BINS - 1,
    ).astype(int)
    return result


def select_isolated_bursts(
    amplitudes: np.ndarray,
    candidates: np.ndarray,
    *,
    refractory_trials: int = config.BURST_REFRACTORY_TRIALS,
) -> list[int]:
    """Greedily retain the largest candidates outside the refractory interval."""

    positions = np.flatnonzero(candidates)
    ordered = sorted(positions, key=lambda position: amplitudes[position], reverse=True)
    accepted: list[int] = []
    for position in ordered:
        if not any(abs(position - selected) <= refractory_trials for selected in accepted):
            accepted.append(int(position))
    return sorted(accepted)


def add_burst_fields(table: pd.DataFrame) -> pd.DataFrame:
    """Detect subject-quantile stimulus-locked pupil bursts deterministically."""

    result = table.copy()
    phasic_column = config.PUPIL_PHASIC_ROBUST_Z_COLUMN
    if phasic_column not in result:
        result[phasic_column] = np.nan
    result["phasic_for_burst"] = pd.to_numeric(result[phasic_column], errors="coerce")
    if "pupil_phasic_ok" in result:
        result.loc[~result["pupil_phasic_ok"].fillna(False), "phasic_for_burst"] = np.nan

    thresholds = result.groupby(config.SUBJECT_COLUMN, sort=False)["phasic_for_burst"].transform(
        lambda values: values.quantile(config.BURST_QUANTILE)
    )
    result["burst_threshold"] = thresholds
    result["burst_candidate"] = (
        result["phasic_for_burst"].notna()
        & result["burst_threshold"].notna()
        & result["phasic_for_burst"].ge(result["burst_threshold"])
    )
    result["isolated_burst"] = False
    result["burst_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["distance_to_nearest_burst"] = np.nan
    next_burst_id = 0

    for _, index in result.groupby(GROUP_COLUMNS, sort=False).groups.items():
        positions = np.asarray(list(index), dtype=int)
        sequence = result.loc[positions]
        accepted_local = select_isolated_bursts(
            sequence["phasic_for_burst"].fillna(-np.inf).to_numpy(float),
            sequence["burst_candidate"].to_numpy(bool),
        )
        if accepted_local:
            accepted_global = positions[np.asarray(accepted_local, dtype=int)]
            result.loc[accepted_global, "isolated_burst"] = True
            for global_index in accepted_global:
                result.at[global_index, "burst_id"] = next_burst_id
                next_burst_id += 1
            local_positions = np.arange(len(sequence))
            distances = np.min(
                np.abs(local_positions[:, None] - np.asarray(accepted_local)[None, :]), axis=1
            )
            result.loc[positions, "distance_to_nearest_burst"] = distances

    result["eligible_nonburst_control"] = (
        ~result["burst_candidate"]
        & result["phasic_for_burst"].notna()
        & result["distance_to_nearest_burst"].fillna(np.inf).gt(
            config.BURST_EXCLUSION_RADIUS
        )
        & result[f"future_js_max_{config.FUTURE_LABILITY_TRIALS}"].notna()
    )
    sequence_length = result.groupby(GROUP_COLUMNS, sort=False)[config.TRIAL_INDEX_COLUMN].transform("size")
    within_left = result.groupby(GROUP_COLUMNS, sort=False).cumcount()
    within_right = sequence_length - within_left - 1
    result["burst_complete_window"] = (
        within_left.ge(config.BURST_PRE_TRIALS)
        & within_right.ge(config.BURST_POST_TRIALS)
    )
    result["pupil_phasic_lock"] = "stimulus"
    if config.PUPIL_FEEDBACK_PHASIC_COLUMN in result:
        result["pupil_feedback_phasic_lock"] = "feedback"
    return result


def build_burst_event_windows(table: pd.DataFrame) -> pd.DataFrame:
    """Create a long event-window table around complete isolated bursts."""

    rows: list[dict[str, object]] = []
    offsets = range(-config.BURST_PRE_TRIALS, config.BURST_POST_TRIALS + 1)
    for (subject, eid), sequence in table.groupby(GROUP_COLUMNS, sort=False):
        sequence = sequence.reset_index(drop=True)
        for burst_position in np.flatnonzero(
            sequence["isolated_burst"].to_numpy(bool)
            & sequence["burst_complete_window"].to_numpy(bool)
        ):
            burst_id = sequence.loc[burst_position, "burst_id"]
            for offset in offsets:
                trial = sequence.iloc[burst_position + offset]
                rows.append(
                    {
                        config.SUBJECT_COLUMN: subject,
                        config.SESSION_COLUMN: eid,
                        "burst_id": burst_id,
                        "offset": offset,
                        config.TRIAL_INDEX_COLUMN: trial[config.TRIAL_INDEX_COLUMN],
                        "reward": trial.get("reward"),
                        "failure": trial.get("failure"),
                        "posterior_js_to_next": trial.get("posterior_js_to_next"),
                        "hard_switch": trial.get("hard_switch"),
                        config.STATE_LABEL_COLUMN: trial.get(config.STATE_LABEL_COLUMN),
                        "phasic_for_burst": trial.get("phasic_for_burst"),
                    }
                )
    return pd.DataFrame(rows)


def build_qc(table: pd.DataFrame, event_windows: pd.DataFrame) -> pd.DataFrame:
    """Build compact cohort- and subject-level regressor diagnostics."""

    rows = []
    for subject, group in table.groupby(config.SUBJECT_COLUMN, sort=True):
        js = group["posterior_js_to_next"]
        rows.append(
            {
                config.SUBJECT_COLUMN: subject,
                "n_trials": len(group),
                "n_sessions": group[config.SESSION_COLUMN].nunique(),
                "n_valid_js": int(js.notna().sum()),
                "js_min": js.min(),
                "js_max": js.max(),
                "n_hard_switches": int(group["hard_switch"].fillna(0).sum()),
                "n_burst_candidates": int(group["burst_candidate"].sum()),
                "n_isolated_bursts": int(group["isolated_burst"].sum()),
                "n_complete_bursts": int(
                    (group["isolated_burst"] & group["burst_complete_window"]).sum()
                ),
                "n_burst_window_rows": int(
                    (event_windows[config.SUBJECT_COLUMN] == subject).sum()
                    if not event_windows.empty
                    else 0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, default=config.TRIAL_TABLE_PATH)
    parser.add_argument("--pupil", type=Path, default=config.PUPIL_FEATURES_PATH)
    parser.add_argument("--states", type=Path, default=config.GLMHMM_STATES_PATH)
    parser.add_argument("--rl", type=Path, default=config.RL_REGRESSORS_PATH)
    parser.add_argument("--output", type=Path, default=config.TRANSITION_REGRESSORS_PATH)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config.ensure_project_directories()
    logger = configure_stage_logger("05_build_transition_regressors", log_path=LOG_PATH)
    trials = load_table(args.trials)
    pupil = load_table(args.pupil)
    states = load_table(args.states)
    rl = load_table(args.rl)
    joined = join_stage_outputs(trials, pupil, states, rl)
    joined = add_state_lability(joined)
    joined = add_feedback_and_position(joined)
    joined = add_burst_fields(joined)
    require_unique_key(joined, config.TRIAL_KEY_COLUMNS, table_name="transition regressors")

    js = joined["posterior_js_to_next"].dropna()
    if not js.between(0.0, 1.0).all():
        raise ValueError("Jensen–Shannon divergence left its valid [0, 1] range.")
    direct_switch = (
        joined["state_next"].notna()
        & joined["state_next"].ne(joined[config.STATE_COLUMN])
    )
    if not np.array_equal(
        direct_switch.loc[joined["hard_switch"].notna()].to_numpy(bool),
        joined.loc[joined["hard_switch"].notna(), "hard_switch"].to_numpy(bool),
    ):
        raise AssertionError("Hard-switch counts disagree with direct state comparisons.")

    event_windows = build_burst_event_windows(joined)
    qc = build_qc(joined, event_windows)
    save_table(joined, args.output)
    save_table(qc, config.TRANSITION_REGRESSOR_QC_PATH)
    save_table(event_windows, config.BURST_EVENT_WINDOW_PATH)
    logger.info(
        "Stage 5 complete: %d trials, %d switches, %d isolated bursts",
        len(joined),
        int(joined["hard_switch"].fillna(0).sum()),
        int(joined["isolated_burst"].sum()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
