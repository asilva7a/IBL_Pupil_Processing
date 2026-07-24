from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from utils import (
    add_future_window_summary,
    add_groupwise_shift,
    add_trial_position,
    apply_state_mapping,
    build_signed_contrast,
    choice_trial_mask,
    context_standardize,
    detect_time_reset_sequences,
    encode_ibl_choice,
    encode_reward,
    extract_event_locked_response,
    flag_robust_outliers,
    hard_state_from_posterior,
    jensen_shannon_divergence,
    mask_pupil_artifacts,
    posterior_entropy,
    relabel_posterior_columns,
    require_columns,
    require_unique_key,
    robust_zscore,
    safe_zscore,
    sanitize_probability_matrix,
    save_table,
    load_table,
    state_occupancy_summary,
    validate_probability_matrix,
    validate_trial_order,
)


def test_config_paths_resolve_from_module_location() -> None:
    assert config.PROJECT_ROOT == Path(config.__file__).resolve().parent
    assert config.TRIAL_TABLE_PATH.is_absolute()
    assert config.STATE_NAMES[0] == "engaged"
    assert config.STATE_NAMES[1] == "biased-left"
    assert config.STATE_NAMES[2] == "biased-right"
    assert "SH015" in config.PUPIL_SUBJECT_EXCLUSIONS


def test_require_columns_and_unique_key() -> None:
    table = pd.DataFrame({"subject": ["a", "b"], "trial": [0, 0]})
    require_columns(table, ["subject", "trial"])
    with pytest.raises(KeyError):
        require_columns(table, ["missing"])
    require_unique_key(table, ["subject", "trial"])
    duplicate = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        require_unique_key(duplicate, ["subject", "trial"])


def test_ibl_choice_encoding_direction() -> None:
    raw = pd.Series([1, -1, 0, "left", "right", np.nan])
    encoded = encode_ibl_choice(raw)
    assert encoded.iloc[0] == 0.0
    assert encoded.iloc[1] == 1.0  # IBL -1 is rightward.
    assert np.isnan(encoded.iloc[2])
    assert encoded.iloc[3] == 0.0
    assert encoded.iloc[4] == 1.0
    assert choice_trial_mask(raw).tolist() == [True, True, False, True, True, False]


def test_reward_and_signed_contrast_encoding() -> None:
    reward = encode_reward(pd.Series([1, -1, 0, np.nan]))
    assert reward.iloc[:2].tolist() == [1.0, 0.0]
    assert reward.iloc[2:].isna().all()

    trials = pd.DataFrame(
        {
            "contrastLeft": [0.5, np.nan, 0.25],
            "contrastRight": [np.nan, 1.0, 0.25],
        }
    )
    signed = build_signed_contrast(trials)
    np.testing.assert_allclose(signed, [-0.5, 1.0, 0.0])


def test_sequence_shift_never_crosses_boundary() -> None:
    table = pd.DataFrame(
        {
            "subject": ["m1"] * 4,
            "sequence": [0, 0, 1, 1],
            "value": [10, 11, 20, 21],
        }
    )
    result = add_groupwise_shift(
        table,
        column="value",
        group_columns=["subject", "sequence"],
        periods=-1,
        output_column="next_value",
    )
    assert result["next_value"].iloc[0] == 11
    assert np.isnan(result["next_value"].iloc[1])
    assert result["next_value"].iloc[2] == 21
    assert np.isnan(result["next_value"].iloc[3])


def test_time_reset_sequence_detection() -> None:
    table = pd.DataFrame(
        {
            "subject": ["a", "a", "a", "b", "b"],
            "time": [1.0, 2.0, 0.5, 5.0, 6.0],
        }
    )
    sequence = detect_time_reset_sequences(
        table, subject_column="subject", time_column="time"
    )
    assert sequence.tolist() == [0, 0, 1, 0, 0]


def test_trial_position_and_future_summary_are_sequence_safe() -> None:
    table = pd.DataFrame(
        {
            "subject": ["a"] * 5,
            "sequence": [0, 0, 0, 1, 1],
            "value": [1.0, 2.0, 3.0, 100.0, 200.0],
        }
    )
    positioned = add_trial_position(
        table, group_columns=["subject", "sequence"]
    )
    assert positioned["trial_in_sequence"].tolist() == [0, 1, 2, 0, 1]
    np.testing.assert_allclose(
        positioned["trial_progress"], [0.0, 0.5, 1.0, 0.0, 1.0]
    )

    summarized = add_future_window_summary(
        table,
        value_column="value",
        group_columns=["subject", "sequence"],
        horizon=2,
        output_column="future_max",
        statistic="max",
    )
    assert summarized["future_max"].iloc[0] == 3.0
    assert summarized["future_max"].iloc[2] != 100.0
    assert summarized["future_max"].iloc[3] == 200.0


def test_zscores_handle_missing_and_constant_groups() -> None:
    values = pd.Series([1.0, 2.0, 3.0, np.nan])
    ordinary = safe_zscore(values)
    assert abs(ordinary.dropna().mean()) < 1e-12
    robust = robust_zscore(values)
    assert robust.iloc[1] == pytest.approx(0.0)

    constant = pd.Series([4.0, 4.0, np.nan])
    assert safe_zscore(constant).dropna().eq(0.0).all()
    assert robust_zscore(constant).dropna().eq(0.0).all()


def test_context_standardize_falls_back_for_small_groups() -> None:
    table = pd.DataFrame(
        {
            "subject": ["a"] * 4,
            "session": ["s1", "s1", "s2", "s2"],
            "value": [1.0, 2.0, 10.0, 20.0],
        }
    )
    scores = context_standardize(
        table,
        column="value",
        context_columns=["subject", "session"],
        fallback_columns=["subject"],
        minimum_context_size=3,
    )
    expected = robust_zscore(table["value"])
    np.testing.assert_allclose(scores, expected)


def test_probability_sanitization_and_validation() -> None:
    raw = np.array(
        [
            [0.2, 0.8, 0.0],
            [-1e-12, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [np.nan, 0.5, 0.5],
        ]
    )
    cleaned, valid = sanitize_probability_matrix(raw)
    assert valid.tolist() == [True, True, False, False]
    np.testing.assert_allclose(cleaned[valid].sum(axis=1), 1.0)
    assert np.isnan(cleaned[~valid]).all()
    validate_probability_matrix(cleaned, allow_nan_rows=True)

    with pytest.raises(ValueError):
        validate_probability_matrix(np.array([[0.2, 0.2]]))


def test_entropy_and_jsd_known_values() -> None:
    probabilities = np.array([[1.0, 0.0], [0.5, 0.5]])
    entropy = posterior_entropy(probabilities)
    np.testing.assert_allclose(entropy, [0.0, 1.0], atol=1e-12)

    identical = jensen_shannon_divergence([0.5, 0.5], [0.5, 0.5])
    disjoint = jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0])
    assert identical == pytest.approx(0.0, abs=1e-12)
    assert disjoint == pytest.approx(1.0, abs=1e-12)


def test_hard_state_relabeling_and_occupancy() -> None:
    posterior = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
    assert hard_state_from_posterior(posterior).tolist() == [1, 0]
    reordered = relabel_posterior_columns(posterior, [1, 0, 2])
    np.testing.assert_allclose(reordered[:, 0], posterior[:, 1])
    mapped = apply_state_mapping(pd.Series([0, 1, 2, np.nan]), {0: 2, 1: 0, 2: 1})
    assert mapped.tolist() == [2, 0, 1, -1]

    table = pd.DataFrame(
        {"subject": ["a", "a", "a", "b"], "state": [0, 0, 1, 2]}
    )
    occupancy = state_occupancy_summary(
        table,
        state_column="state",
        group_columns=["subject"],
        state_names=config.STATE_NAMES,
    )
    a = occupancy.loc[occupancy["subject"].eq("a")]
    assert a["occupancy"].sum() == pytest.approx(1.0)


def test_trial_order_validation() -> None:
    ordered = pd.DataFrame(
        {"subject": ["a", "a", "b", "b"], "trial": [0, 1, 0, 1]}
    )
    validate_trial_order(ordered, ["subject"], "trial")
    bad = ordered.copy()
    bad.loc[1, "trial"] = 0
    with pytest.raises(ValueError):
        validate_trial_order(bad, ["subject"], "trial")


def test_outlier_flag_does_not_drop_rows() -> None:
    values = pd.Series([0.0, 0.1, -0.1, 20.0, np.nan])
    flags = flag_robust_outliers(values, threshold=4.0)
    assert len(flags) == len(values)
    assert flags.iloc[3]
    assert not flags.iloc[4]


def test_pupil_artifact_mask_and_event_response() -> None:
    times = np.arange(0.0, 4.0, 0.1)
    diameter = np.ones_like(times)
    diameter[15] = np.nan
    cleaned, mask = mask_pupil_artifacts(
        diameter,
        times,
        velocity_mad_threshold=5.0,
        padding_seconds=0.1,
        nominal_fps=10.0,
    )
    assert mask[15]
    assert np.isnan(cleaned[15])
    assert mask[14] and mask[16]

    signal = np.zeros_like(times)
    signal[(times >= 2.5) & (times <= 3.0)] = 2.0
    response = extract_event_locked_response(
        times,
        signal,
        event_time=2.0,
        baseline_window=(-0.5, 0.0),
        response_window=(0.5, 1.0),
        minimum_valid_fraction=0.8,
    )
    assert response["valid"]
    assert response["baseline"] == pytest.approx(0.0)
    assert response["response"] == pytest.approx(2.0)
    assert response["delta"] == pytest.approx(2.0)


def test_csv_roundtrip(tmp_path: Path) -> None:
    table = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = save_table(table, tmp_path / "table.csv")
    loaded = load_table(path)
    pd.testing.assert_frame_equal(table, loaded)
