from __future__ import annotations

import numpy as np
import pandas as pd

import config
from conftest import load_stage

stage = load_stage("01_build_trial_table.py", "stage01")


def test_label_trial_epochs_reproduces_notebook_rule() -> None:
    table = pd.DataFrame(
        {
            "probabilityLeft": [0.5, 0.5, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8],
        }
    )
    labeled = stage.label_trial_epochs(table, n_transition_trials=2)
    assert labeled["epoch"].tolist() == [
        "unbiased",
        "unbiased",
        "transition",
        "transition",
        "stable",
        "transition",
        "transition",
        "stable",
    ]


def test_canonicalize_session_encodings() -> None:
    raw = pd.DataFrame(
        {
            "choice": [1, -1, 0],
            "feedbackType": [1, -1, 1],
            "contrastLeft": [0.5, np.nan, 0.0],
            "contrastRight": [np.nan, 0.5, 0.0],
            "probabilityLeft": [0.5, 0.2, 0.2],
            "stimOn_times": [1.0, 2.0, 3.0],
            "feedback_times": [1.5, 2.5, 3.5],
        }
    )
    table = stage.canonicalize_session_trials(
        raw, subject="m1", eid="e1", sequence_id=0, sex="M"
    )
    assert table["rightward_choice"].iloc[0] == 0.0
    assert table["rightward_choice"].iloc[1] == 1.0
    assert np.isnan(table["rightward_choice"].iloc[2])
    np.testing.assert_allclose(table["signed_contrast"], [-0.5, 0.5, 0.0])
    assert table[list(config.TRIAL_KEY_COLUMNS)].duplicated().sum() == 0
