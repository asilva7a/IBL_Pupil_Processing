"""Stage 6: fit state-lability, origin–destination, and burst-control models.

Every specification is named and saved with its outcome, predictors, sample
size, subject count, and status.  Binary outcomes use clustered logistic GLMs;
continuous Jensen–Shannon lability uses clustered OLS.  The script never treats
independent posterior products as true soft-transition probabilities.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon

import config
from utils import (
    configure_stage_logger,
    load_table,
    paired_effect_size,
    require_columns,
    save_table,
    safe_zscore,
    tidy_statsmodels_result,
)

LOG_PATH = config.LOG_DIR / "06_fit_transition_models.log"


def configured_exclusion_set() -> str:
    """Return the configured pre-existing exclusions as serialized metadata."""

    return json.dumps(
        {
            "behavior": config.BEHAVIOR_SUBJECT_EXCLUSIONS,
            "pupil": config.PUPIL_SUBJECT_EXCLUSIONS,
            "rl": config.RL_SUBJECT_EXCLUSIONS,
        },
        sort_keys=True,
    )


def add_model_covariates(table: pd.DataFrame) -> pd.DataFrame:
    """Add standardized nuisance and interaction-ready variables."""

    result = table.copy()
    result["absolute_contrast"] = pd.to_numeric(
        result[config.SIGNED_CONTRAST_COLUMN], errors="coerce"
    ).abs()
    for source, destination in (
        ("absolute_contrast", "absolute_contrast_z"),
        ("trial_progress", "trial_progress_z"),
        ("rl_rpe_negative", "negative_rpe_z"),
    ):
        if source in result:
            result[destination] = result.groupby(
                config.SUBJECT_COLUMN, sort=False
            )[source].transform(lambda values: safe_zscore(values, constant="zeros"))

    if config.PUPIL_TONIC_ROBUST_Z_COLUMN in result:
        result["tonic_z"] = pd.to_numeric(
            result[config.PUPIL_TONIC_ROBUST_Z_COLUMN], errors="coerce"
        )
    else:
        result["tonic_z"] = np.nan
    if config.PUPIL_FEEDBACK_PHASIC_ROBUST_Z_COLUMN in result:
        result["feedback_phasic_z"] = pd.to_numeric(
            result[config.PUPIL_FEEDBACK_PHASIC_ROBUST_Z_COLUMN], errors="coerce"
        )
    else:
        result["feedback_phasic_z"] = np.nan

    failure_rows = result["failure"].eq(1) & result.get("rl_rpe_negative", pd.Series(index=result.index, dtype=float)).notna()
    result["unexpected_failure_z"] = 0.0
    result.loc[failure_rows, "unexpected_failure_z"] = result.loc[failure_rows].groupby(
        config.SUBJECT_COLUMN, sort=False
    )["rl_rpe_negative"].transform(lambda values: safe_zscore(values, constant="zeros"))
    result["failure_x_tonic"] = result["failure"] * result["tonic_z"]
    result["unexpected_x_feedback_phasic"] = (
        result["unexpected_failure_z"] * result["feedback_phasic_z"]
    )
    return result


def build_design_matrix(
    table: pd.DataFrame,
    *,
    numeric_terms: Sequence[str],
    include_epoch: bool = True,
) -> pd.DataFrame:
    """Build a stable numeric design matrix with an intercept and epoch dummies."""

    require_columns(table, numeric_terms)
    design = pd.DataFrame({"intercept": 1.0}, index=table.index)
    for term in numeric_terms:
        design[term] = pd.to_numeric(table[term], errors="coerce")
    if include_epoch and config.EPOCH_COLUMN in table:
        dummies = pd.get_dummies(
            table[config.EPOCH_COLUMN].astype(str),
            prefix="epoch",
            drop_first=True,
            dtype=float,
        )
        design = pd.concat([design, dummies], axis=1)
    constant = [
        column
        for column in design.columns
        if column != "intercept" and design[column].nunique(dropna=True) <= 1
    ]
    return design.drop(columns=constant).astype(float)


def fit_clustered_model(
    table: pd.DataFrame,
    *,
    specification: str,
    outcome: str,
    numeric_terms: Sequence[str],
    family: str,
    model_directory: Path = config.TRANSITION_MODEL_DIR,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Fit one clustered model and return a tidy table plus diagnostics."""

    design = build_design_matrix(table, numeric_terms=numeric_terms)
    valid = table[outcome].notna() & design.notna().all(axis=1)
    model_table = table.loc[valid].copy()
    design = design.loc[valid]
    n_trials = len(model_table)
    n_subjects = model_table[config.SUBJECT_COLUMN].nunique()
    n_positive = int(pd.to_numeric(model_table[outcome], errors="coerce").fillna(0).sum())
    diagnostics: dict[str, Any] = {
        "specification": specification,
        "outcome": outcome,
        "predictors": json.dumps(list(design.columns)),
        "family": family,
        "n_trials": n_trials,
        "n_subjects": n_subjects,
        "n_positive": n_positive if family == "binomial" else np.nan,
        "status": "skipped",
        "message": "",
        "exclusion_set": configured_exclusion_set(),
        "pipeline_version": config.PIPELINE_VERSION,
    }
    if n_trials < config.MIN_TRIALS_PER_TRANSITION_MODEL:
        diagnostics["message"] = "too_few_trials"
        return None, diagnostics
    if n_subjects < config.MIN_SUBJECTS_PER_MODEL:
        diagnostics["message"] = "too_few_subjects"
        return None, diagnostics
    if family == "binomial":
        if n_positive < config.MIN_TRANSITIONS_PER_MODEL or n_positive >= n_trials:
            diagnostics["message"] = "insufficient_outcome_variation"
            return None, diagnostics

    try:
        outcome_values = pd.to_numeric(model_table[outcome], errors="coerce").astype(float)
        if family == "binomial":
            model = sm.GLM(outcome_values, design, family=sm.families.Binomial())
        elif family == "gaussian":
            model = sm.OLS(outcome_values, design)
        else:
            raise ValueError("family must be 'binomial' or 'gaussian'.")
        fitted = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": model_table[config.SUBJECT_COLUMN].astype(str)},
            maxiter=300,
        )
        coefficients = tidy_statsmodels_result(
            fitted,
            model_name=specification,
            exponentiate=family == "binomial",
        )
        coefficients.insert(1, "outcome", outcome)
        coefficients["family"] = family
        coefficients["n_trials"] = n_trials
        coefficients["n_subjects"] = n_subjects
        coefficients["predictors"] = json.dumps(list(design.columns))
        coefficients["exclusion_set"] = configured_exclusion_set()
        coefficients["pipeline_version"] = config.PIPELINE_VERSION
        model_directory.mkdir(parents=True, exist_ok=True)
        model_path = model_directory / f"{specification}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(fitted, handle)
        diagnostics.update(
            status="success",
            message="",
            model_path=str(model_path),
            converged=bool(getattr(fitted, "converged", True)),
            aic=float(getattr(fitted, "aic", np.nan)),
        )
        return coefficients, diagnostics
    except Exception as error:
        diagnostics.update(
            status="failed",
            message=f"{type(error).__name__}: {error}",
        )
        return None, diagnostics


def fit_named_specs(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the predeclared lability/switch specifications."""

    horizon = config.FUTURE_LABILITY_TRIALS
    switch_outcome = f"future_switch_any_{horizon}"
    js_outcome = f"future_js_max_{horizon}"
    specs = [
        {
            "name": "failure_only",
            "outcome": switch_outcome,
            "terms": ["failure", "absolute_contrast_z", "trial_progress_z"],
            "family": "binomial",
            "subset": pd.Series(True, index=table.index),
        },
        {
            "name": "failure_plus_rl",
            "outcome": switch_outcome,
            "terms": [
                "failure",
                "negative_rpe_z",
                "absolute_contrast_z",
                "trial_progress_z",
            ],
            "family": "binomial",
            "subset": pd.Series(True, index=table.index),
        },
        {
            "name": "failure_x_tonic",
            "outcome": switch_outcome,
            "terms": [
                "failure",
                "tonic_z",
                "failure_x_tonic",
                "absolute_contrast_z",
                "trial_progress_z",
            ],
            "family": "binomial",
            "subset": table["tonic_z"].notna(),
        },
        {
            "name": "lability_failure_plus_rl",
            "outcome": js_outcome,
            "terms": [
                "failure",
                "negative_rpe_z",
                "absolute_contrast_z",
                "trial_progress_z",
            ],
            "family": "gaussian",
            "subset": pd.Series(True, index=table.index),
        },
        {
            "name": "feedback_phasic_error_trials",
            "outcome": switch_outcome,
            "terms": [
                "unexpected_failure_z",
                "feedback_phasic_z",
                "unexpected_x_feedback_phasic",
                "absolute_contrast_z",
                "trial_progress_z",
            ],
            "family": "binomial",
            "subset": table["failure"].eq(1) & table["feedback_phasic_z"].notna(),
        },
    ]
    coefficient_tables: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, Any]] = []
    for specification in specs:
        coefficients, diagnostics = fit_clustered_model(
            table.loc[specification["subset"]],
            specification=specification["name"],
            outcome=specification["outcome"],
            numeric_terms=specification["terms"],
            family=specification["family"],
        )
        diagnostics_rows.append(diagnostics)
        if coefficients is not None:
            coefficient_tables.append(coefficients)
    return (
        pd.concat(coefficient_tables, ignore_index=True)
        if coefficient_tables
        else pd.DataFrame(),
        pd.DataFrame(diagnostics_rows),
    )


def fit_origin_destination(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit pairwise stay-versus-destination models for all six transitions."""

    coefficient_tables: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for origin, origin_name in config.STATE_NAMES.items():
        for destination, destination_name in config.STATE_NAMES.items():
            if origin == destination:
                continue
            subset = table.loc[
                table[config.STATE_COLUMN].eq(origin)
                & table["state_next"].isin([origin, destination])
            ].copy()
            outcome = "transition_to_destination"
            subset[outcome] = subset["state_next"].eq(destination).astype(float)
            specification = f"origin_{origin_name}_to_{destination_name}".replace("-", "_")
            coefficients, diagnostics = fit_clustered_model(
                subset,
                specification=specification,
                outcome=outcome,
                numeric_terms=[
                    "failure",
                    "negative_rpe_z",
                    "tonic_z",
                    "absolute_contrast_z",
                    "trial_progress_z",
                ],
                family="binomial",
            )
            diagnostics.update(
                origin_state=origin_name,
                destination_state=destination_name,
            )
            diagnostic_rows.append(diagnostics)
            if coefficients is not None:
                coefficients.insert(0, "origin_state", origin_name)
                coefficients.insert(1, "destination_state", destination_name)
                coefficient_tables.append(coefficients)
    return (
        pd.concat(coefficient_tables, ignore_index=True)
        if coefficient_tables
        else pd.DataFrame(),
        pd.DataFrame(diagnostic_rows),
    )


def _valid_match_matrix(
    bursts: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    max_bin_difference: int | None,
    max_fraction_difference: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    burst_bin = bursts["position_bin"].to_numpy(float)
    control_bin = controls["position_bin"].to_numpy(float)
    burst_fraction = bursts["trial_progress"].to_numpy(float)
    control_fraction = controls["trial_progress"].to_numpy(float)
    bin_difference = np.abs(burst_bin[:, None] - control_bin[None, :])
    fraction_difference = np.abs(
        burst_fraction[:, None] - control_fraction[None, :]
    )
    valid = np.ones_like(fraction_difference, dtype=bool)
    if max_bin_difference is not None:
        valid &= bin_difference <= max_bin_difference
    if max_fraction_difference is not None:
        valid &= fraction_difference <= max_fraction_difference
    cost = fraction_difference + 0.01 * bin_difference
    cost[~valid] = config.MATCH_LARGE_INVALID_COST
    return cost, bin_difference, fraction_difference


def match_bursts(
    table: pd.DataFrame,
    *,
    specification: str,
    max_bin_difference: int | None,
    max_fraction_difference: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Perform strict one-to-one matching within subject/session/state/outcome."""

    outcome = f"future_js_max_{config.FUTURE_LABILITY_TRIALS}"
    bursts = table.loc[
        table["isolated_burst"]
        & table["burst_complete_window"]
        & table[outcome].notna()
    ].copy()
    controls = table.loc[
        table["eligible_nonburst_control"] & table[outcome].notna()
    ].copy()
    strata = [
        config.SUBJECT_COLUMN,
        config.SESSION_COLUMN,
        config.STATE_COLUMN,
        "reward",
    ]
    matches: list[dict[str, Any]] = []
    for key, burst_group in bursts.groupby(strata, dropna=False, sort=False):
        mask = np.ones(len(controls), dtype=bool)
        key_tuple = key if isinstance(key, tuple) else (key,)
        for column, value in zip(strata, key_tuple):
            if pd.isna(value):
                mask &= controls[column].isna().to_numpy()
            else:
                mask &= controls[column].eq(value).to_numpy()
        control_group = controls.loc[mask]
        if burst_group.empty or control_group.empty:
            continue
        cost, bin_difference, fraction_difference = _valid_match_matrix(
            burst_group,
            control_group,
            max_bin_difference=max_bin_difference,
            max_fraction_difference=max_fraction_difference,
        )
        row_indices, column_indices = linear_sum_assignment(cost)
        for row_index, column_index in zip(row_indices, column_indices):
            if cost[row_index, column_index] >= config.MATCH_LARGE_INVALID_COST:
                continue
            burst = burst_group.iloc[row_index]
            control = control_group.iloc[column_index]
            matches.append(
                {
                    "specification": specification,
                    config.SUBJECT_COLUMN: burst[config.SUBJECT_COLUMN],
                    config.SESSION_COLUMN: burst[config.SESSION_COLUMN],
                    "reward": burst["reward"],
                    "feedback_label": (
                        "Rewarded" if burst["reward"] == 1 else "Negative feedback"
                    ),
                    "state": burst[config.STATE_COLUMN],
                    "state_label": burst[config.STATE_LABEL_COLUMN],
                    "burst_trial_index": burst[config.TRIAL_INDEX_COLUMN],
                    "control_trial_index": control[config.TRIAL_INDEX_COLUMN],
                    "burst_outcome": burst[outcome],
                    "control_outcome": control[outcome],
                    "delta_js": burst[outcome] - control[outcome],
                    "position_bin_difference": bin_difference[row_index, column_index],
                    "trial_fraction_difference": fraction_difference[row_index, column_index],
                    "exclusion_set": configured_exclusion_set(),
                    "pipeline_version": config.PIPELINE_VERSION,
                }
            )
    matched = pd.DataFrame(matches)
    diagnostics = {
        "specification": specification,
        "eligible_bursts": len(bursts),
        "eligible_controls": len(controls),
        "matched_pairs": len(matched),
        "retained_fraction": len(matched) / len(bursts) if len(bursts) else np.nan,
        "n_subjects": matched[config.SUBJECT_COLUMN].nunique() if not matched.empty else 0,
        "median_trial_fraction_difference": (
            matched["trial_fraction_difference"].median() if not matched.empty else np.nan
        ),
        "maximum_trial_fraction_difference": (
            matched["trial_fraction_difference"].max() if not matched.empty else np.nan
        ),
        "exclusion_set": configured_exclusion_set(),
        "pipeline_version": config.PIPELINE_VERSION,
    }
    return matched, diagnostics


def _signed_rank(values: pd.Series, comparison: str, specification: str) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    row: dict[str, Any] = {
        "record_type": "population_test",
        "specification": specification,
        "comparison": comparison,
        "n_subjects": len(clean),
        "mean_difference": clean.mean() if len(clean) else np.nan,
        "median_difference": clean.median() if len(clean) else np.nan,
        "cohens_dz": paired_effect_size(clean),
        "wilcoxon_statistic": np.nan,
        "p_value": np.nan,
    }
    if len(clean) >= 2 and not np.allclose(clean, 0.0):
        test = wilcoxon(clean, zero_method="wilcox", alternative="two-sided")
        row["wilcoxon_statistic"] = float(test.statistic)
        row["p_value"] = float(test.pvalue)
    return row


def summarize_matches(matches: pd.DataFrame, specification: str) -> pd.DataFrame:
    """Create subject medians and paired Wilcoxon summaries for a match set."""

    if matches.empty:
        return pd.DataFrame()
    subject = (
        matches.groupby([config.SUBJECT_COLUMN, "feedback_label"], sort=False)
        .agg(
            median_burst_outcome=("burst_outcome", "median"),
            median_control_outcome=("control_outcome", "median"),
            median_delta_js=("delta_js", "median"),
            n_pairs=("delta_js", "size"),
        )
        .reset_index()
    )
    subject["record_type"] = "subject_summary"
    subject["specification"] = specification
    wide_delta = subject.pivot(
        index=config.SUBJECT_COLUMN,
        columns="feedback_label",
        values="median_delta_js",
    )
    tests = []
    for feedback_label in ("Rewarded", "Negative feedback"):
        if feedback_label in wide_delta:
            tests.append(
                _signed_rank(
                    wide_delta[feedback_label],
                    f"{feedback_label} burst minus matched non-burst",
                    specification,
                )
            )
    if {"Rewarded", "Negative feedback"}.issubset(wide_delta.columns):
        tests.append(
            _signed_rank(
                wide_delta["Negative feedback"] - wide_delta["Rewarded"],
                "Difference-in-differences",
                specification,
            )
        )
    pair_records = matches.copy()
    pair_records["record_type"] = "matched_pair"
    return pd.concat(
        [pair_records, subject, pd.DataFrame(tests)],
        ignore_index=True,
        sort=False,
    )


def run_burst_matching(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run primary and strict matching sensitivity specifications."""

    specifications = [
        (
            "Primary bin±1 and within 10%",
            config.MATCH_MAX_POSITION_BIN_DIFFERENCE,
            config.MATCH_MAX_TRIAL_FRACTION_DIFFERENCE,
        ),
        ("Exact position bin", 0, None),
        ("Within 5% sequence position", None, config.STRICT_MATCH_MAX_TRIAL_FRACTION_DIFFERENCE),
        ("Exact bin and within 5%", 0, config.STRICT_MATCH_MAX_TRIAL_FRACTION_DIFFERENCE),
    ]
    result_tables: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for name, max_bin, max_fraction in specifications:
        matches, diagnostics = match_bursts(
            table,
            specification=name,
            max_bin_difference=max_bin,
            max_fraction_difference=max_fraction,
        )
        diagnostic_rows.append(diagnostics)
        summary = summarize_matches(matches, name)
        if not summary.empty:
            result_tables.append(summary)
    return (
        pd.concat(result_tables, ignore_index=True, sort=False)
        if result_tables
        else pd.DataFrame(),
        pd.DataFrame(diagnostic_rows),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=config.TRANSITION_REGRESSORS_PATH)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config.ensure_project_directories()
    logger = configure_stage_logger("06_fit_transition_models", log_path=LOG_PATH)
    table = add_model_covariates(load_table(args.input))
    require_columns(
        table,
        [
            config.SUBJECT_COLUMN,
            config.STATE_COLUMN,
            "state_next",
            "failure",
            f"future_switch_any_{config.FUTURE_LABILITY_TRIALS}",
            f"future_js_max_{config.FUTURE_LABILITY_TRIALS}",
        ],
        table_name="transition regressor table",
    )

    coefficients, model_diagnostics = fit_named_specs(table)
    origin_coefficients, origin_diagnostics = fit_origin_destination(table)
    burst_results, burst_diagnostics = run_burst_matching(table)
    diagnostics = pd.concat(
        [
            model_diagnostics.assign(model_group="named_specification"),
            origin_diagnostics.assign(model_group="origin_destination"),
            burst_diagnostics.assign(model_group="burst_matching"),
        ],
        ignore_index=True,
        sort=False,
    )

    save_table(coefficients, config.TRANSITION_COEFFICIENT_PATH)
    save_table(origin_coefficients, config.ORIGIN_DESTINATION_COEFFICIENT_PATH)
    save_table(burst_results, config.BURST_MATCHED_RESULTS_PATH)
    save_table(diagnostics, config.TRANSITION_MODEL_DIAGNOSTICS_PATH)
    logger.info(
        "Stage 6 complete: %d coefficient rows, %d origin rows, %d burst rows",
        len(coefficients),
        len(origin_coefficients),
        len(burst_results),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
