from __future__ import annotations

import numpy as np
import pandas as pd

import config
from conftest import load_stage

stage = load_stage("05_build_transition_regressors.py", "stage05")


def _base_table() -> pd.DataFrame:
    rows = []
    posteriors = [
        (0.9, 0.05, 0.05),
        (0.8, 0.1, 0.1),
        (0.1, 0.8, 0.1),
        (0.9, 0.05, 0.05),
        (0.85, 0.1, 0.05),
    ]
    sessions = ["a", "a", "a", "b", "b"]
    for index, (posterior, eid) in enumerate(zip(posteriors, sessions)):
        rows.append(
            {
                "subject": "m1",
                "eid": eid,
                "sequence_id": 0 if eid == "a" else 1,
                "trial_index": index if eid == "a" else index - 3,
                "p_state0": posterior[0],
                "p_state1": posterior[1],
                "p_state2": posterior[2],
                "state": int(np.argmax(posterior)),
            }
        )
    return pd.DataFrame(rows)


def test_lability_never_crosses_session_boundary() -> None:
    result = stage.add_state_lability(_base_table())
    last_a = result.loc[result["eid"] == "a"].iloc[-1]
    assert np.isnan(last_a["posterior_js_to_next"])
    assert np.isnan(last_a["hard_switch"])
    first_b = result.loc[result["eid"] == "b"].iloc[0]
    assert first_b["posterior_js_to_next"] >= 0


def test_isolated_burst_selection_prefers_larger_candidate() -> None:
    amplitudes = np.array([0.0, 4.0, 5.0, 0.0, 3.0])
    candidates = np.array([False, True, True, False, True])
    selected = stage.select_isolated_bursts(amplitudes, candidates, refractory_trials=1)
    assert 2 in selected
    assert 1 not in selected
