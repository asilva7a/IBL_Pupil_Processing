"""Reusable utilities for the NMA engagement/bias-state pipeline.

The functions in this module are deliberately small and analysis-agnostic.
They do not connect to ONE, depend on notebook globals, select scientific model
formulas, or silently remove observations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Filesystem, validation, and logging
# ---------------------------------------------------------------------------


def ensure_directory(path: str | Path) -> Path:
    """Create *path* when necessary and return it as an absolute ``Path``."""

    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def require_columns(
    table: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    table_name: str = "table",
) -> None:
    """Raise ``KeyError`` when a DataFrame lacks required columns."""

    required = tuple(required_columns)
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise KeyError(f"{table_name} is missing required columns: {missing}")


def require_unique_key(
    table: pd.DataFrame,
    key_columns: Sequence[str],
    *,
    table_name: str = "table",
) -> None:
    """Raise ``ValueError`` when a proposed key contains missing or duplicate rows."""

    require_columns(table, key_columns, table_name=table_name)

    missing_key = table.loc[:, list(key_columns)].isna().any(axis=1)
    if missing_key.any():
        examples = table.loc[missing_key, list(key_columns)].head(10)
        raise ValueError(
            f"{table_name} has {int(missing_key.sum())} rows with missing key "
            f"values. Examples:\n{examples.to_string(index=False)}"
        )

    duplicate_key = table.duplicated(list(key_columns), keep=False)
    if duplicate_key.any():
        examples = (
            table.loc[duplicate_key, list(key_columns)]
            .sort_values(list(key_columns))
            .head(10)
        )
        raise ValueError(
            f"{table_name} has {int(duplicate_key.sum())} rows participating "
            f"in duplicate keys. Examples:\n{examples.to_string(index=False)}"
        )


def validate_trial_order(
    table: pd.DataFrame,
    group_columns: Sequence[str],
    order_column: str,
    *,
    strictly_increasing: bool = True,
    table_name: str = "trial table",
) -> None:
    """Validate temporal ordering inside each subject/session/sequence group.

    The function checks the current row order.  It does not sort the table.
    """

    require_columns(
        table,
        [*group_columns, order_column],
        table_name=table_name,
    )

    if table[order_column].isna().any():
        raise ValueError(f"{table_name}.{order_column} contains missing values.")

    invalid_groups: list[tuple[Any, ...]] = []
    for group_key, group in table.groupby(list(group_columns), sort=False, dropna=False):
        values = pd.to_numeric(group[order_column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            invalid_groups.append(
                group_key if isinstance(group_key, tuple) else (group_key,)
            )
            continue
        differences = np.diff(values)
        valid = np.all(differences > 0) if strictly_increasing else np.all(differences >= 0)
        if not valid:
            invalid_groups.append(
                group_key if isinstance(group_key, tuple) else (group_key,)
            )

    if invalid_groups:
        raise ValueError(
            f"{table_name} is not correctly ordered within "
            f"{len(invalid_groups)} group(s). First groups: {invalid_groups[:10]}"
        )


def load_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV or Parquet table based on its file extension."""

    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Table not found: {source.resolve()}")

    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source, **kwargs)
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(source, **kwargs)
        except ImportError as error:
            raise ImportError(
                "Parquet support requires pyarrow or fastparquet."
            ) from error
    raise ValueError(f"Unsupported table format: {suffix!r}")


def save_table(
    table: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """Save a DataFrame as CSV or Parquet and return the resolved path."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()

    if suffix == ".csv":
        table.to_csv(destination, index=index, **kwargs)
    elif suffix in {".parquet", ".pq"}:
        try:
            table.to_parquet(destination, index=index, **kwargs)
        except ImportError as error:
            raise ImportError(
                "Parquet support requires pyarrow or fastparquet."
            ) from error
    else:
        raise ValueError(f"Unsupported table format: {suffix!r}")

    return destination


def configure_stage_logger(
    name: str,
    *,
    log_path: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a compact console logger with an optional file handler."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    if not any(getattr(handler, "_nma_console", False) for handler in logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console._nma_console = True  # type: ignore[attr-defined]
        logger.addHandler(console)

    if log_path is not None:
        destination = Path(log_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        existing_files = {
            Path(getattr(handler, "baseFilename", "")).resolve()
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        }
        if destination not in existing_files:
            file_handler = logging.FileHandler(destination)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Trial encoding
# ---------------------------------------------------------------------------


def encode_ibl_choice(values: pd.Series | Sequence[Any]) -> pd.Series:
    """Encode IBL choice as ``0=left``, ``1=right``, and ``NaN=no-go/invalid``.

    IBL's raw numeric convention is ``+1=left``, ``-1=right``, and ``0=no-go``.
    Common textual left/right labels are also accepted.
    """

    series = pd.Series(values, copy=False)
    numeric = pd.to_numeric(series, errors="coerce")
    encoded = pd.Series(np.nan, index=series.index, dtype=float)

    encoded.loc[numeric.eq(1)] = 0.0
    encoded.loc[numeric.eq(-1)] = 1.0

    text = series.astype(str).str.strip().str.lower()
    encoded.loc[text.isin({"left", "l"})] = 0.0
    encoded.loc[text.isin({"right", "r"})] = 1.0
    return encoded


def rightward_choice(
    table: pd.DataFrame,
    *,
    choice_column: str = "choice",
) -> pd.Series:
    """Return the canonical rightward-choice encoding for a trial table."""

    require_columns(table, [choice_column])
    return encode_ibl_choice(table[choice_column]).rename("rightward_choice")


def encode_reward(values: pd.Series | Sequence[Any]) -> pd.Series:
    """Encode IBL ``feedbackType`` as ``1=reward``, ``0=no reward``.

    Values other than ``+1`` and ``-1`` are treated as missing rather than
    silently recoded.
    """

    series = pd.Series(values, copy=False)
    numeric = pd.to_numeric(series, errors="coerce")
    reward = pd.Series(np.nan, index=series.index, dtype=float)
    reward.loc[numeric.eq(1)] = 1.0
    reward.loc[numeric.eq(-1)] = 0.0
    return reward


def build_signed_contrast(
    table: pd.DataFrame,
    *,
    left_column: str = "contrastLeft",
    right_column: str = "contrastRight",
) -> pd.Series:
    """Construct signed contrast as right contrast minus left contrast."""

    require_columns(table, [left_column, right_column])
    left = pd.to_numeric(table[left_column], errors="coerce").fillna(0.0)
    right = pd.to_numeric(table[right_column], errors="coerce").fillna(0.0)
    return (right - left).rename("signed_contrast")


def choice_trial_mask(values: pd.Series | Sequence[Any]) -> pd.Series:
    """Return ``True`` for valid left/right choices and ``False`` otherwise."""

    return encode_ibl_choice(values).notna()


# ---------------------------------------------------------------------------
# Sequence-safe trial operations
# ---------------------------------------------------------------------------


def detect_time_reset_sequences(
    table: pd.DataFrame,
    *,
    subject_column: str,
    time_column: str,
) -> pd.Series:
    """Create within-subject sequence IDs when timestamps reset or stop increasing."""

    require_columns(table, [subject_column, time_column])
    times = pd.to_numeric(table[time_column], errors="coerce")
    previous = times.groupby(table[subject_column], sort=False).shift(1)
    first_trial = table.groupby(subject_column, sort=False).cumcount().eq(0)
    starts = first_trial | times.isna() | previous.isna() | times.le(previous)
    return (
        starts.groupby(table[subject_column], sort=False)
        .cumsum()
        .astype(int)
        .sub(1)
        .rename("sequence_id")
    )


def add_groupwise_shift(
    table: pd.DataFrame,
    *,
    column: str,
    group_columns: Sequence[str],
    periods: int,
    output_column: str | None = None,
) -> pd.DataFrame:
    """Add a lag/lead column without allowing values to cross group boundaries."""

    require_columns(table, [column, *group_columns])
    result = table.copy()
    if output_column is None:
        direction = "lag" if periods > 0 else "lead"
        output_column = f"{column}_{direction}{abs(periods)}"
    result[output_column] = (
        result.groupby(list(group_columns), sort=False, dropna=False)[column]
        .shift(periods)
    )
    return result


def add_trial_position(
    table: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    trial_number_column: str = "trial_in_sequence",
    progress_column: str = "trial_progress",
) -> pd.DataFrame:
    """Add zero-based trial number and relative progress within each sequence."""

    require_columns(table, group_columns)
    result = table.copy()
    group = result.groupby(list(group_columns), sort=False, dropna=False)
    result[trial_number_column] = group.cumcount()
    maximum = group[trial_number_column].transform("max")
    result[progress_column] = np.where(
        maximum.gt(0),
        result[trial_number_column] / maximum,
        0.0,
    )
    return result


def add_future_window_summary(
    table: pd.DataFrame,
    *,
    value_column: str,
    group_columns: Sequence[str],
    horizon: int,
    output_column: str,
    statistic: str = "max",
    include_current: bool = False,
) -> pd.DataFrame:
    """Add a sequence-safe summary over the current/future trial window."""

    if horizon < 1:
        raise ValueError("horizon must be at least one.")
    if statistic not in {"max", "mean", "min"}:
        raise ValueError("statistic must be one of: 'max', 'mean', 'min'.")
    require_columns(table, [value_column, *group_columns])

    result = table.copy()
    offsets = range(0 if include_current else 1, horizon + 1)
    temporary: list[str] = []
    grouped = result.groupby(list(group_columns), sort=False, dropna=False)[value_column]
    for offset in offsets:
        name = f"__{value_column}_lead_{offset}"
        result[name] = grouped.shift(-offset)
        temporary.append(name)

    if statistic == "max":
        result[output_column] = result[temporary].max(axis=1, skipna=True)
    elif statistic == "min":
        result[output_column] = result[temporary].min(axis=1, skipna=True)
    else:
        result[output_column] = result[temporary].mean(axis=1, skipna=True)

    return result.drop(columns=temporary)


# ---------------------------------------------------------------------------
# Scaling, missingness, and generic QC
# ---------------------------------------------------------------------------


def median_absolute_deviation(values: pd.Series | Sequence[Any]) -> float:
    """Return the unscaled median absolute deviation, ignoring missing values."""

    series = pd.to_numeric(pd.Series(values, copy=False), errors="coerce")
    median = series.median(skipna=True)
    return float((series - median).abs().median(skipna=True))


def _constant_group_output(
    values: pd.Series,
    constant: str,
) -> pd.Series:
    if constant not in {"zeros", "nan"}:
        raise ValueError("constant must be either 'zeros' or 'nan'.")
    output = pd.Series(np.nan, index=values.index, dtype=float)
    if constant == "zeros":
        output.loc[values.notna()] = 0.0
    return output


def safe_zscore(
    values: pd.Series | Sequence[Any],
    *,
    constant: str = "zeros",
) -> pd.Series:
    """Ordinary z-score with explicit behavior for constant groups."""

    series = pd.to_numeric(pd.Series(values, copy=False), errors="coerce").astype(float)
    standard_deviation = series.std(skipna=True, ddof=0)
    if not np.isfinite(standard_deviation) or standard_deviation <= np.finfo(float).eps:
        return _constant_group_output(series, constant)
    return (series - series.mean(skipna=True)) / standard_deviation


def robust_zscore(
    values: pd.Series | Sequence[Any],
    *,
    fallback_to_sd: bool = True,
    constant: str = "zeros",
) -> pd.Series:
    """Median/MAD z-score with an optional standard-deviation fallback."""

    series = pd.to_numeric(pd.Series(values, copy=False), errors="coerce").astype(float)
    center = series.median(skipna=True)
    scale = 1.4826 * median_absolute_deviation(series)

    if (
        fallback_to_sd
        and (not np.isfinite(scale) or scale <= np.finfo(float).eps)
    ):
        scale = series.std(skipna=True, ddof=0)

    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return _constant_group_output(series, constant)
    return (series - center) / scale


def context_standardize(
    table: pd.DataFrame,
    *,
    column: str,
    context_columns: Sequence[str],
    fallback_columns: Sequence[str],
    minimum_context_size: int = 20,
    robust: bool = True,
    constant: str = "zeros",
) -> pd.Series:
    """Standardize within a narrow context, falling back for small groups."""

    require_columns(table, [column, *context_columns, *fallback_columns])
    if minimum_context_size < 1:
        raise ValueError("minimum_context_size must be positive.")

    transform = (
        (lambda x: robust_zscore(x, constant=constant))
        if robust
        else (lambda x: safe_zscore(x, constant=constant))
    )
    context_group = table.groupby(
        list(context_columns), sort=False, dropna=False
    )[column]
    fallback_group = table.groupby(
        list(fallback_columns), sort=False, dropna=False
    )[column]

    context_count = context_group.transform("count")
    context_z = context_group.transform(transform)
    fallback_z = fallback_group.transform(transform)
    return pd.Series(
        np.where(context_count >= minimum_context_size, context_z, fallback_z),
        index=table.index,
        dtype=float,
        name=f"{column}_z",
    )


def missingness_summary(
    table: pd.DataFrame,
    *,
    columns: Sequence[str],
    group_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize nonmissing counts and missing fractions globally or by group."""

    require_columns(table, [*columns, *(group_columns or [])])

    def summarize(group: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for column in columns:
            n_total = len(group)
            n_missing = int(group[column].isna().sum())
            rows.append(
                {
                    "column": column,
                    "n_total": n_total,
                    "n_nonmissing": n_total - n_missing,
                    "n_missing": n_missing,
                    "missing_fraction": n_missing / n_total if n_total else np.nan,
                }
            )
        return pd.DataFrame(rows)

    if not group_columns:
        return summarize(table)

    output = []
    for key, group in table.groupby(list(group_columns), sort=False, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        summary = summarize(group)
        for column, value in zip(group_columns, key_tuple):
            summary[column] = value
        output.append(summary)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def flag_robust_outliers(
    values: pd.Series | Sequence[Any],
    *,
    threshold: float,
) -> pd.Series:
    """Return a boolean robust-z outlier flag without dropping observations."""

    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    scores = robust_zscore(values, constant="zeros")
    return scores.abs().gt(threshold).fillna(False).rename("outlier")


# ---------------------------------------------------------------------------
# State-posterior utilities
# ---------------------------------------------------------------------------


def sanitize_probability_matrix(
    raw_probabilities: np.ndarray | Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Clip and renormalize an ``N x K`` probability matrix.

    Rows containing nonfinite values or a nonpositive total are returned as
    all-NaN and marked ``False`` in the accompanying validity mask.
    """

    raw = np.asarray(raw_probabilities, dtype=float)
    if raw.ndim != 2:
        raise ValueError("Probability matrix must be two-dimensional.")

    finite_rows = np.isfinite(raw).all(axis=1)
    cleaned = np.full_like(raw, np.nan, dtype=float)
    cleaned[finite_rows] = np.clip(raw[finite_rows], 0.0, None)
    row_sums = np.nansum(cleaned, axis=1)
    valid_rows = finite_rows & np.isfinite(row_sums) & (row_sums > 0)

    normalized = np.full_like(cleaned, np.nan, dtype=float)
    normalized[valid_rows] = cleaned[valid_rows] / row_sums[valid_rows, None]
    return normalized, np.asarray(valid_rows, dtype=bool)


def validate_probability_matrix(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-8,
    allow_nan_rows: bool = False,
) -> None:
    """Raise ``ValueError`` unless rows are valid probability vectors."""

    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Probability matrix must be two-dimensional.")

    nan_rows = np.isnan(matrix).all(axis=1)
    partial_nan_rows = np.isnan(matrix).any(axis=1) & ~nan_rows
    if partial_nan_rows.any():
        raise ValueError("Probability matrix contains partially missing rows.")
    if nan_rows.any() and not allow_nan_rows:
        raise ValueError("Probability matrix contains missing rows.")

    valid = ~nan_rows
    if not np.isfinite(matrix[valid]).all():
        raise ValueError("Probability matrix contains nonfinite values.")
    if (matrix[valid] < -tolerance).any():
        raise ValueError("Probability matrix contains negative values.")
    row_sums = matrix[valid].sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("Probability rows do not sum to one.")


def posterior_entropy(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Compute row-wise posterior entropy, optionally normalized to ``[0, 1]``."""

    matrix, valid = sanitize_probability_matrix(probabilities)
    entropy = np.full(matrix.shape[0], np.nan, dtype=float)
    p = matrix[valid]
    terms = np.zeros_like(p)
    positive = p > 0
    terms[positive] = -p[positive] * np.log2(p[positive])
    values = terms.sum(axis=1)
    if normalize and matrix.shape[1] > 1:
        values = values / np.log2(matrix.shape[1])
    entropy[valid] = np.clip(values, 0.0, 1.0 if normalize else np.inf)
    return entropy


def jensen_shannon_divergence(
    first: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
    second: np.ndarray | Sequence[Sequence[float]] | Sequence[float],
    *,
    normalize: bool = True,
) -> np.ndarray | float:
    """Compute Jensen-Shannon divergence row-wise.

    With ``normalize=True``, base-2 logarithms produce values in ``[0, 1]``.
    One-dimensional inputs return a scalar; two-dimensional inputs return an
    array.  Invalid rows return ``NaN``.
    """

    first_array = np.asarray(first, dtype=float)
    second_array = np.asarray(second, dtype=float)
    scalar_input = first_array.ndim == 1 and second_array.ndim == 1
    if first_array.ndim == 1:
        first_array = first_array[None, :]
    if second_array.ndim == 1:
        second_array = second_array[None, :]
    if first_array.shape != second_array.shape:
        raise ValueError("Probability arrays must have identical shapes.")

    p, valid_p = sanitize_probability_matrix(first_array)
    q, valid_q = sanitize_probability_matrix(second_array)
    valid = valid_p & valid_q
    output = np.full(p.shape[0], np.nan, dtype=float)

    if valid.any():
        pv = p[valid]
        qv = q[valid]
        midpoint = (pv + qv) / 2.0

        def kl_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            terms = np.zeros_like(left)
            positive = left > 0
            if normalize:
                terms[positive] = left[positive] * (
                    np.log2(left[positive]) - np.log2(right[positive])
                )
            else:
                terms[positive] = left[positive] * (
                    np.log(left[positive]) - np.log(right[positive])
                )
            return terms.sum(axis=1)

        values = 0.5 * kl_divergence(pv, midpoint) + 0.5 * kl_divergence(qv, midpoint)
        upper = 1.0 if normalize else np.log(2.0)
        output[valid] = np.clip(values, 0.0, upper)

    return float(output[0]) if scalar_input else output


def hard_state_from_posterior(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    *,
    invalid_value: int = -1,
) -> np.ndarray:
    """Return posterior argmax states, using ``invalid_value`` for bad rows."""

    matrix, valid = sanitize_probability_matrix(probabilities)
    states = np.full(matrix.shape[0], invalid_value, dtype=int)
    states[valid] = np.argmax(matrix[valid], axis=1)
    return states


def relabel_posterior_columns(
    probabilities: np.ndarray | Sequence[Sequence[float]],
    order: Sequence[int],
) -> np.ndarray:
    """Reorder posterior columns after validating a complete state permutation."""

    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Probability matrix must be two-dimensional.")
    order_array = np.asarray(order, dtype=int)
    if sorted(order_array.tolist()) != list(range(matrix.shape[1])):
        raise ValueError("order must be a complete permutation of state indices.")
    return matrix[:, order_array]


def apply_state_mapping(
    states: pd.Series | Sequence[Any],
    mapping: Mapping[int, int],
    *,
    invalid_value: int = -1,
) -> pd.Series:
    """Apply an explicit old-to-new integer state mapping."""

    numeric = pd.to_numeric(pd.Series(states, copy=False), errors="coerce")
    mapped = numeric.map(mapping)
    return mapped.fillna(invalid_value).astype(int)


def state_occupancy_summary(
    table: pd.DataFrame,
    *,
    state_column: str,
    group_columns: Sequence[str],
    state_names: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Calculate trial counts and occupancy proportions by group and state."""

    require_columns(table, [state_column, *group_columns])
    counts = (
        table.groupby([*group_columns, state_column], dropna=False, sort=False)
        .size()
        .rename("n_trials")
        .reset_index()
    )
    totals = counts.groupby(list(group_columns), dropna=False)["n_trials"].transform("sum")
    counts["occupancy"] = counts["n_trials"] / totals
    if state_names is not None:
        counts["state_label"] = counts[state_column].map(state_names)
    return counts


# ---------------------------------------------------------------------------
# Generic statistical output helpers
# ---------------------------------------------------------------------------


def tidy_statsmodels_result(
    fitted_result: Any,
    *,
    model_name: str | None = None,
    exponentiate: bool = False,
) -> pd.DataFrame:
    """Convert a statsmodels-like result object into a tidy coefficient table."""

    parameters = pd.Series(fitted_result.params)
    standard_errors = pd.Series(fitted_result.bse, index=parameters.index)
    statistics = pd.Series(fitted_result.tvalues, index=parameters.index)
    p_values = pd.Series(fitted_result.pvalues, index=parameters.index)
    confidence = pd.DataFrame(fitted_result.conf_int(), index=parameters.index)

    table = pd.DataFrame(
        {
            "term": parameters.index.astype(str),
            "estimate": parameters.to_numpy(float),
            "standard_error": standard_errors.to_numpy(float),
            "statistic": statistics.to_numpy(float),
            "p_value": p_values.to_numpy(float),
            "ci_lower": confidence.iloc[:, 0].to_numpy(float),
            "ci_upper": confidence.iloc[:, 1].to_numpy(float),
        }
    )
    if exponentiate:
        table["odds_ratio"] = np.exp(table["estimate"])
        table["odds_ratio_lower"] = np.exp(table["ci_lower"])
        table["odds_ratio_upper"] = np.exp(table["ci_upper"])
    if model_name is not None:
        table["model"] = model_name
    return table


def paired_effect_size(differences: pd.Series | Sequence[Any]) -> float:
    """Return paired-sample Cohen's ``dz`` from a vector of differences."""

    values = (
        pd.to_numeric(pd.Series(differences, copy=False), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(values) < 2:
        return np.nan
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation <= np.finfo(float).eps:
        return np.nan
    return float(values.mean() / standard_deviation)


def format_p_value(p_value: float, *, threshold: float = 0.001) -> str:
    """Format a p-value for figures and compact result tables."""

    if not np.isfinite(p_value):
        return "p = n/a"
    if p_value < threshold:
        return f"p < {threshold:.3f}".replace("0.", ".")
    return f"p = {p_value:.3f}".replace("0.", ".")


def significance_label(p_value: float) -> str:
    """Return a conventional compact significance label."""

    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Pupil trace primitives
# ---------------------------------------------------------------------------


def mask_pupil_artifacts(
    diameter: np.ndarray | Sequence[float],
    sample_times: np.ndarray | Sequence[float],
    *,
    velocity_mad_threshold: float = 5.0,
    padding_seconds: float = 0.125,
    nominal_fps: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask pupil blinks/artifacts using missing samples and robust velocity.

    Returns a cleaned copy of the diameter trace and the boolean artifact mask.
    No interpolation is performed.
    """

    diameter_array = np.asarray(diameter, dtype=float)
    time_array = np.asarray(sample_times, dtype=float)
    if diameter_array.ndim != 1 or time_array.ndim != 1:
        raise ValueError("diameter and sample_times must be one-dimensional.")
    if len(diameter_array) != len(time_array):
        raise ValueError("diameter and sample_times must have equal length.")
    if velocity_mad_threshold <= 0 or padding_seconds < 0 or nominal_fps <= 0:
        raise ValueError("Artifact parameters must be positive (padding may be zero).")

    cleaned = diameter_array.copy()
    bad = ~np.isfinite(cleaned) | ~np.isfinite(time_array)
    if len(cleaned) < 2:
        cleaned[bad] = np.nan
        return cleaned, bad

    positive_dt = np.diff(time_array)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
    fallback_dt = 1.0 / nominal_fps
    median_dt = float(np.median(positive_dt)) if positive_dt.size else fallback_dt

    dt = np.gradient(time_array)
    dt[~np.isfinite(dt) | (dt <= 0)] = median_dt
    velocity = np.abs(np.gradient(cleaned) / dt)
    finite_velocity = velocity[np.isfinite(velocity)]
    if finite_velocity.size:
        center = np.median(finite_velocity)
        scale = 1.4826 * np.median(np.abs(finite_velocity - center))
        if np.isfinite(scale) and scale > np.finfo(float).eps:
            bad |= velocity > center + velocity_mad_threshold * scale

    pad_frames = int(round(padding_seconds / median_dt))
    if pad_frames > 0 and bad.any():
        kernel = np.ones(2 * pad_frames + 1, dtype=int)
        bad = np.convolve(bad.astype(int), kernel, mode="same") > 0

    cleaned[bad] = np.nan
    return cleaned, bad


def time_window_mask(
    sample_times: np.ndarray | Sequence[float],
    *,
    event_time: float,
    window: tuple[float, float],
    include_right_endpoint: bool = False,
) -> np.ndarray:
    """Return samples falling inside a time window relative to an event."""

    times = np.asarray(sample_times, dtype=float)
    if times.ndim != 1:
        raise ValueError("sample_times must be one-dimensional.")
    start, stop = window
    if not np.isfinite(event_time) or not (np.isfinite(start) and np.isfinite(stop)):
        return np.zeros(len(times), dtype=bool)
    if stop <= start:
        raise ValueError("window stop must be greater than window start.")
    relative = times - event_time
    if include_right_endpoint:
        return (relative >= start) & (relative <= stop)
    return (relative >= start) & (relative < stop)


def extract_event_locked_response(
    sample_times: np.ndarray | Sequence[float],
    values: np.ndarray | Sequence[float],
    *,
    event_time: float,
    baseline_window: tuple[float, float],
    response_window: tuple[float, float],
    minimum_valid_fraction: float = 0.80,
) -> dict[str, float | int | bool]:
    """Extract baseline, response, and baseline-subtracted event response."""

    times = np.asarray(sample_times, dtype=float)
    signal = np.asarray(values, dtype=float)
    if times.ndim != 1 or signal.ndim != 1:
        raise ValueError("sample_times and values must be one-dimensional.")
    if len(times) != len(signal):
        raise ValueError("sample_times and values must have equal length.")
    if not 0 < minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must lie in (0, 1].")

    baseline_mask = time_window_mask(
        times, event_time=event_time, window=baseline_window
    )
    response_mask = time_window_mask(
        times, event_time=event_time, window=response_window,
        include_right_endpoint=True,
    )

    def summarize(mask: np.ndarray) -> tuple[float, int, int, float]:
        segment = signal[mask]
        n_total = int(segment.size)
        n_valid = int(np.isfinite(segment).sum())
        fraction = n_valid / n_total if n_total else 0.0
        mean = float(np.nanmean(segment)) if n_valid else np.nan
        return mean, n_valid, n_total, fraction

    baseline, baseline_valid, baseline_total, baseline_fraction = summarize(baseline_mask)
    response, response_valid, response_total, response_fraction = summarize(response_mask)
    valid = (
        baseline_total > 0
        and response_total > 0
        and baseline_fraction >= minimum_valid_fraction
        and response_fraction >= minimum_valid_fraction
        and np.isfinite(baseline)
        and np.isfinite(response)
    )

    return {
        "baseline": baseline if valid else np.nan,
        "response": response if valid else np.nan,
        "delta": response - baseline if valid else np.nan,
        "baseline_n_valid": baseline_valid,
        "baseline_n_total": baseline_total,
        "baseline_valid_fraction": baseline_fraction,
        "response_n_valid": response_valid,
        "response_n_total": response_total,
        "response_valid_fraction": response_fraction,
        "valid": bool(valid),
    }


def extract_event_locked_responses(
    sample_times: np.ndarray | Sequence[float],
    values: np.ndarray | Sequence[float],
    event_times: Sequence[float],
    *,
    baseline_window: tuple[float, float],
    response_window: tuple[float, float],
    minimum_valid_fraction: float = 0.80,
) -> pd.DataFrame:
    """Apply :func:`extract_event_locked_response` to multiple event times."""

    rows = []
    for event_index, event_time in enumerate(event_times):
        row = extract_event_locked_response(
            sample_times,
            values,
            event_time=float(event_time),
            baseline_window=baseline_window,
            response_window=response_window,
            minimum_valid_fraction=minimum_valid_fraction,
        )
        row["event_index"] = event_index
        row["event_time"] = event_time
        rows.append(row)
    return pd.DataFrame(rows)
