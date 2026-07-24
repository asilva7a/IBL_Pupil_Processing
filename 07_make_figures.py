"""Stage 7: regenerate pipeline figures from saved tables and model outputs only.

No statistical model is fitted here.  Missing optional inputs cause a named
figure to be skipped and recorded in the figure manifest rather than silently
changing the analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from utils import configure_stage_logger, ensure_directory, load_table, save_table

LOG_PATH = config.LOG_DIR / "07_make_figures.log"


def save_figure(
    figure: plt.Figure,
    *,
    stem: str,
    sources: list[Path],
    description: str,
    exclusions: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    """Save one figure in every configured format and return manifest rows."""

    ensure_directory(config.FIGURE_DIR)
    rows = []
    for extension in config.FIGURE_OUTPUT_FORMATS:
        path = config.FIGURE_DIR / f"{stem}.{extension}"
        figure.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight", facecolor="white")
        rows.append(
            {
                "figure": stem,
                "path": str(path),
                "format": extension,
                "dpi": config.FIGURE_DPI,
                "width_inches": float(figure.get_size_inches()[0]),
                "height_inches": float(figure.get_size_inches()[1]),
                "sources": json.dumps([str(source) for source in sources]),
                "description": description,
                "exclusions": json.dumps(exclusions or {}),
                "status": "created",
                "pipeline_version": config.PIPELINE_VERSION,
                "figure_script": "07_make_figures.py",
            }
        )
    plt.close(figure)
    return rows


def psychometric_by_epoch(trials: pd.DataFrame) -> plt.Figure:
    """Plot behavioral psychometrics for unbiased/transition/stable epochs."""

    figure, axis = plt.subplots(figsize=config.HALF_SLIDE_SIZE_INCHES)
    valid = trials.loc[trials["rightward_choice"].notna()].copy()
    for epoch in config.EPOCH_ORDER:
        grouped = (
            valid.loc[valid[config.EPOCH_COLUMN] == epoch]
            .groupby(config.SIGNED_CONTRAST_COLUMN)["rightward_choice"]
            .agg(["mean", "size"])
            .reset_index()
        )
        if grouped.empty:
            continue
        axis.plot(
            grouped[config.SIGNED_CONTRAST_COLUMN],
            grouped["mean"],
            marker="o",
            label=f"{epoch} (n={grouped['size'].sum():,})",
        )
    axis.axhline(0.5, linestyle=":", linewidth=1)
    axis.axvline(0.0, linestyle=":", linewidth=1)
    axis.set(
        xlabel="Signed contrast (right minus left)",
        ylabel="P(rightward choice)",
        title="Psychometric function by task epoch",
        ylim=(-0.03, 1.03),
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def state_psychometric(psychometric: pd.DataFrame) -> plt.Figure:
    """Plot saved state-conditioned psychometric summaries."""

    figure, axis = plt.subplots(figsize=config.HALF_SLIDE_SIZE_INCHES)
    aggregated = (
        psychometric.groupby([config.STATE_LABEL_COLUMN, config.SIGNED_CONTRAST_COLUMN])
        .apply(
            lambda group: pd.Series(
                {
                    "probability_right": np.average(
                        group["probability_right"], weights=group["n_trials"]
                    ),
                    "n_trials": group["n_trials"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    for state in config.STATE_NAMES.values():
        group = aggregated.loc[aggregated[config.STATE_LABEL_COLUMN] == state]
        if group.empty:
            continue
        axis.plot(
            group[config.SIGNED_CONTRAST_COLUMN],
            group["probability_right"],
            marker="o",
            label=state,
            color=config.FIGURE_STATE_COLORS.get(state),
        )
    axis.axhline(0.5, linestyle=":", linewidth=1)
    axis.axvline(0.0, linestyle=":", linewidth=1)
    axis.set(
        xlabel="Signed contrast (right minus left)",
        ylabel="P(rightward choice)",
        title="State-conditioned psychometric functions",
        ylim=(-0.03, 1.03),
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def occupancy_figure(occupancy: pd.DataFrame) -> plt.Figure:
    """Plot animal-level occupancy for the three aligned states."""

    figure, axis = plt.subplots(figsize=config.HALF_SLIDE_SIZE_INCHES)
    labels = list(config.STATE_NAMES.values())
    for position, state in enumerate(labels):
        values = occupancy.loc[occupancy["state_label"] == state, "occupancy"].dropna()
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) else np.array([])
        axis.scatter(np.full(len(values), position) + jitter, values, alpha=0.5, s=18)
        if len(values):
            axis.errorbar(
                position,
                values.mean(),
                yerr=values.sem(),
                marker="o",
                capsize=4,
                linewidth=1.5,
            )
    axis.set_xticks(range(len(labels)), labels, rotation=15)
    axis.set(ylabel="State occupancy", title="GLM-HMM state occupancy", ylim=(0, 1))
    figure.tight_layout()
    return figure


def pupil_by_state(transitions: pd.DataFrame) -> plt.Figure:
    """Plot subject-level tonic and stimulus-phasic pupil by state."""

    metrics = [
        (config.PUPIL_TONIC_ROBUST_Z_COLUMN, "Tonic pupil (robust z)"),
        (config.PUPIL_PHASIC_ROBUST_Z_COLUMN, "Stimulus-phasic pupil (robust z)"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=config.FULL_SLIDE_SIZE_INCHES)
    labels = list(config.STATE_NAMES.values())
    for axis, (column, ylabel) in zip(axes, metrics):
        if column not in transitions:
            axis.text(0.5, 0.5, f"{column} unavailable", ha="center", va="center")
            axis.set_axis_off()
            continue
        subject_summary = (
            transitions.loc[transitions[column].notna()]
            .groupby([config.SUBJECT_COLUMN, config.STATE_LABEL_COLUMN])[column]
            .mean()
            .reset_index()
        )
        for position, state in enumerate(labels):
            values = subject_summary.loc[
                subject_summary[config.STATE_LABEL_COLUMN] == state, column
            ]
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) else np.array([])
            axis.scatter(np.full(len(values), position) + jitter, values, alpha=0.45, s=17)
            if len(values):
                axis.errorbar(position, values.mean(), yerr=values.sem(), marker="o", capsize=4)
        axis.axhline(0, linestyle=":", linewidth=1)
        axis.set_xticks(range(len(labels)), labels, rotation=15)
        axis.set(ylabel=ylabel, title=ylabel + " by behavioral state")
    figure.tight_layout()
    return figure


def tonic_by_epoch_figure(transitions: pd.DataFrame) -> plt.Figure:
    """Plot raw and robust-z tonic pupil across task epochs using subject means."""

    metrics = [
        (config.PUPIL_TONIC_COLUMN, "Raw tonic pupil"),
        (config.PUPIL_TONIC_ROBUST_Z_COLUMN, "Tonic pupil (robust z)"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=config.FULL_SLIDE_SIZE_INCHES)
    for axis, (column, ylabel) in zip(axes, metrics):
        if column not in transitions:
            axis.text(0.5, 0.5, f"{column} unavailable", ha="center", va="center")
            axis.set_axis_off()
            continue
        summary = (
            transitions.loc[transitions[column].notna()]
            .groupby([config.SUBJECT_COLUMN, config.EPOCH_COLUMN], sort=False)[column]
            .mean()
            .reset_index()
        )
        wide = summary.pivot(
            index=config.SUBJECT_COLUMN,
            columns=config.EPOCH_COLUMN,
            values=column,
        ).reindex(columns=config.EPOCH_ORDER)
        for _, row in wide.iterrows():
            axis.plot(range(len(config.EPOCH_ORDER)), row.to_numpy(float), alpha=0.14, linewidth=0.7)
        means = wide.mean(axis=0)
        sems = wide.sem(axis=0)
        axis.errorbar(
            range(len(config.EPOCH_ORDER)),
            means,
            yerr=sems,
            marker="o",
            linewidth=2,
            capsize=4,
        )
        axis.set_xticks(range(len(config.EPOCH_ORDER)), config.EPOCH_ORDER, rotation=15)
        axis.set(ylabel=ylabel, title=f"{ylabel} by task epoch")
        if "robust z" in ylabel.lower():
            axis.axhline(0.0, linestyle=":", linewidth=1)
    figure.tight_layout()
    return figure


def animal_paired_state_pupil_figure(transitions: pd.DataFrame) -> plt.Figure:
    """Plot within-animal tonic-pupil means across aligned behavioral states."""

    column = config.PUPIL_TONIC_ROBUST_Z_COLUMN
    figure, axis = plt.subplots(figsize=config.HALF_SLIDE_SIZE_INCHES)
    if column not in transitions:
        axis.text(0.5, 0.5, f"{column} unavailable", ha="center", va="center")
        axis.set_axis_off()
        return figure
    state_order = list(config.STATE_NAMES.values())
    summary = (
        transitions.loc[transitions[column].notna()]
        .groupby([config.SUBJECT_COLUMN, config.STATE_LABEL_COLUMN], sort=False)[column]
        .mean()
        .reset_index()
    )
    wide = summary.pivot(
        index=config.SUBJECT_COLUMN,
        columns=config.STATE_LABEL_COLUMN,
        values=column,
    ).reindex(columns=state_order)
    for _, row in wide.iterrows():
        axis.plot(range(len(state_order)), row.to_numpy(float), alpha=0.18, linewidth=0.8)
    axis.errorbar(
        range(len(state_order)),
        wide.mean(axis=0),
        yerr=wide.sem(axis=0),
        marker="o",
        linewidth=2,
        capsize=4,
    )
    axis.axhline(0.0, linestyle=":", linewidth=1)
    axis.set_xticks(range(len(state_order)), state_order, rotation=15)
    axis.set(
        ylabel="Mean tonic pupil (robust z)",
        title="Within-animal tonic pupil across behavioral states",
    )
    figure.tight_layout()
    return figure


def sex_stratified_pupil_figure(transitions: pd.DataFrame) -> plt.Figure:
    """Plot subject-level tonic pupil by state, stratified by recorded sex."""

    column = config.PUPIL_TONIC_ROBUST_Z_COLUMN
    figure, axes = plt.subplots(1, 2, figsize=config.FULL_SLIDE_SIZE_INCHES, sharey=True)
    if column not in transitions or config.SEX_COLUMN not in transitions:
        for axis in axes:
            axis.text(0.5, 0.5, "Sex or tonic pupil unavailable", ha="center", va="center")
            axis.set_axis_off()
        return figure
    state_order = list(config.STATE_NAMES.values())
    summary = (
        transitions.loc[transitions[column].notna()]
        .groupby(
            [config.SUBJECT_COLUMN, config.SEX_COLUMN, config.STATE_LABEL_COLUMN],
            sort=False,
        )[column]
        .mean()
        .reset_index()
    )
    for axis, sex in zip(axes, config.SEX_ORDER):
        sex_table = summary.loc[summary[config.SEX_COLUMN].astype(str).str.upper() == sex]
        for position, state in enumerate(state_order):
            values = sex_table.loc[sex_table[config.STATE_LABEL_COLUMN] == state, column]
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) else np.array([])
            axis.scatter(np.full(len(values), position) + jitter, values, alpha=0.45, s=18)
            if len(values):
                axis.errorbar(position, values.mean(), yerr=values.sem(), marker="o", capsize=4)
        axis.axhline(0.0, linestyle=":", linewidth=1)
        axis.set_xticks(range(len(state_order)), state_order, rotation=15)
        axis.set(title=f"Sex: {sex}", xlabel="Behavioral state")
    axes[0].set_ylabel("Mean tonic pupil (robust z)")
    figure.suptitle("Sex-stratified tonic pupil by behavioral state")
    figure.tight_layout()
    return figure


def rl_comparison_figure(comparison: pd.DataFrame) -> plt.Figure:
    """Plot held-out hybrid-minus-sensory log-likelihood improvement."""

    figure, axis = plt.subplots(figsize=config.HALF_SLIDE_SIZE_INCHES)
    values = comparison["test_delta_loglik_per_trial"].dropna().sort_values().to_numpy()
    axis.scatter(np.arange(len(values)), values, s=22, alpha=0.65)
    axis.axhline(0.0, linestyle="--", linewidth=1)
    axis.set(
        xlabel="Animal (sorted)",
        ylabel="Held-out Δ log likelihood / trial",
        title="Incremental predictive value of Q-learning",
    )
    figure.tight_layout()
    return figure


def coefficient_forest(
    coefficients: pd.DataFrame,
    *,
    title: str,
    terms: Sequence[str] | None = None,
) -> plt.Figure:
    """Create a coefficient forest from a saved tidy coefficient table."""

    table = coefficients.copy()
    if terms is not None:
        table = table.loc[table["term"].isin(terms)]
    table = table.loc[~table["term"].eq("intercept")].copy()
    table["label"] = table["model"].astype(str) + " | " + table["term"].astype(str)
    table = table.sort_values("estimate").reset_index(drop=True)
    height = max(4.5, 0.34 * len(table) + 1.5)
    figure, axis = plt.subplots(figsize=(10, height))
    positions = np.arange(len(table))
    axis.errorbar(
        table["estimate"],
        positions,
        xerr=[table["estimate"] - table["ci_lower"], table["ci_upper"] - table["estimate"]],
        fmt="o",
        capsize=3,
    )
    axis.axvline(0.0, linestyle="--", linewidth=1)
    axis.set_yticks(positions, table["label"])
    axis.set(xlabel="Coefficient", title=title)
    figure.tight_layout()
    return figure


def burst_summary_figure(results: pd.DataFrame) -> plt.Figure:
    """Plot strict-specification subject median burst-control differences."""

    strict = results.loc[
        (results["record_type"] == "subject_summary")
        & (results["specification"] == "Exact bin and within 5%")
    ].copy()
    figure, axis = plt.subplots(figsize=config.HALF_SLIDE_SIZE_INCHES)
    labels = ["Rewarded", "Negative feedback"]
    for position, label in enumerate(labels):
        values = strict.loc[strict["feedback_label"] == label, "median_delta_js"].dropna()
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) else np.array([])
        axis.scatter(np.full(len(values), position) + jitter, values, alpha=0.55, s=20)
        if len(values):
            axis.errorbar(position, values.mean(), yerr=values.sem(), marker="o", capsize=4)
    axis.axhline(0, linestyle="--", linewidth=1)
    axis.set_xticks(range(2), labels)
    axis.set(
        ylabel="Subject median ΔJSD\n(burst − matched non-burst)",
        title="Stimulus-locked pupil bursts and subsequent state lability",
    )
    figure.tight_layout()
    return figure


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Fail when an optional figure input is missing.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config.ensure_project_directories()
    logger = configure_stage_logger("07_make_figures", log_path=LOG_PATH)
    manifest_rows: list[dict[str, object]] = []

    jobs: list[tuple[str, list[Path], Callable[..., plt.Figure], str, dict[str, str] | None]] = [
        (
            "behavior_psychometric_by_epoch",
            [config.TRIAL_TABLE_PATH],
            psychometric_by_epoch,
            "Behavioral psychometric functions separated by task epoch.",
            config.BEHAVIOR_SUBJECT_EXCLUSIONS,
        ),
        (
            "state_conditioned_psychometric",
            [config.GLMHMM_PSYCHOMETRIC_QC_PATH],
            state_psychometric,
            "Psychometric functions conditioned on aligned GLM-HMM state.",
            config.BEHAVIOR_SUBJECT_EXCLUSIONS,
        ),
        (
            "state_occupancy",
            [config.STATE_OCCUPANCY_PATH],
            occupancy_figure,
            "Animal-level occupancy of aligned GLM-HMM states.",
            config.BEHAVIOR_SUBJECT_EXCLUSIONS,
        ),
        (
            "pupil_by_behavioral_state",
            [config.TRANSITION_REGRESSORS_PATH],
            pupil_by_state,
            "Subject-level tonic and stimulus-phasic pupil summaries by state.",
            config.PUPIL_SUBJECT_EXCLUSIONS,
        ),
        (
            "tonic_pupil_by_epoch",
            [config.TRANSITION_REGRESSORS_PATH],
            tonic_by_epoch_figure,
            "Raw and robust-z tonic pupil across unbiased, transition, and stable epochs.",
            config.PUPIL_SUBJECT_EXCLUSIONS,
        ),
        (
            "animal_paired_tonic_by_state",
            [config.TRANSITION_REGRESSORS_PATH],
            animal_paired_state_pupil_figure,
            "Within-animal paired tonic-pupil means across aligned states.",
            config.PUPIL_SUBJECT_EXCLUSIONS,
        ),
        (
            "sex_stratified_tonic_by_state",
            [config.TRANSITION_REGRESSORS_PATH],
            sex_stratified_pupil_figure,
            "Subject-level tonic pupil by state, stratified by recorded sex.",
            config.PUPIL_SUBJECT_EXCLUSIONS,
        ),
        (
            "rl_heldout_model_comparison",
            [config.RL_MODEL_COMPARISON_PATH],
            rl_comparison_figure,
            "Held-out hybrid-minus-sensory log-likelihood improvement.",
            config.RL_SUBJECT_EXCLUSIONS,
        ),
        (
            "transition_model_coefficients",
            [config.TRANSITION_COEFFICIENT_PATH],
            lambda table: coefficient_forest(
                table,
                title="Feedback, reinforcement, and pupil predictors of state lability",
                terms=[
                    "failure",
                    "negative_rpe_z",
                    "tonic_z",
                    "failure_x_tonic",
                    "feedback_phasic_z",
                    "unexpected_x_feedback_phasic",
                ],
            ),
            "Clustered model coefficients for named transition specifications.",
            None,
        ),
        (
            "origin_destination_coefficients",
            [config.ORIGIN_DESTINATION_COEFFICIENT_PATH],
            lambda table: coefficient_forest(
                table,
                title="Origin–destination state-transition coefficients",
                terms=["failure", "negative_rpe_z", "tonic_z"],
            ),
            "Pairwise stay-versus-destination transition coefficients.",
            None,
        ),
        (
            "burst_matched_summary",
            [config.BURST_MATCHED_RESULTS_PATH],
            burst_summary_figure,
            "Strictly matched stimulus-locked pupil-burst summary.",
            config.PUPIL_SUBJECT_EXCLUSIONS,
        ),
    ]

    for stem, sources, function, description, exclusions in jobs:
        missing = [source for source in sources if not source.exists()]
        if missing:
            message = f"Missing inputs: {missing}"
            logger.warning("Skipping %s: %s", stem, message)
            manifest_rows.append(
                {
                    "figure": stem,
                    "path": "",
                    "sources": json.dumps([str(source) for source in sources]),
                    "description": description,
                    "exclusions": json.dumps(exclusions or {}),
                    "status": "skipped",
                    "message": message,
                    "pipeline_version": config.PIPELINE_VERSION,
                    "figure_script": "07_make_figures.py",
                }
            )
            if args.strict:
                raise FileNotFoundError(message)
            continue
        try:
            tables = [load_table(source) for source in sources]
            figure = function(*tables)
            manifest_rows.extend(
                save_figure(
                    figure,
                    stem=stem,
                    sources=sources,
                    description=description,
                    exclusions=exclusions,
                )
            )
            logger.info("Created %s", stem)
        except Exception as error:
            logger.exception("Figure failed: %s", stem)
            manifest_rows.append(
                {
                    "figure": stem,
                    "path": "",
                    "sources": json.dumps([str(source) for source in sources]),
                    "description": description,
                    "exclusions": json.dumps(exclusions or {}),
                    "status": "failed",
                    "message": f"{type(error).__name__}: {error}",
                    "pipeline_version": config.PIPELINE_VERSION,
                    "figure_script": "07_make_figures.py",
                }
            )
            if args.strict:
                raise

    save_table(pd.DataFrame(manifest_rows), config.FIGURE_MANIFEST_PATH)
    logger.info(
        "Stage 7 complete: %d created files, %d manifest rows",
        sum(row.get("status") == "created" for row in manifest_rows),
        len(manifest_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
