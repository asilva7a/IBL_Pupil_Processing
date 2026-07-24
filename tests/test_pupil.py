from __future__ import annotations

import numpy as np
import pandas as pd

import config
from conftest import load_stage

stage = load_stage("02_preprocess_pupil.py", "stage02")


def test_extract_session_features_stimulus_and_feedback() -> None:
    times = np.arange(0.0, 8.0, 0.05)
    values = np.ones_like(times)
    values[(times >= 2.0) & (times <= 3.0)] += 1.0
    values[(times >= 4.5) & (times <= 6.5)] += 2.0
    session = pd.DataFrame(
        {
            "subject": ["m1"],
            "eid": ["e1"],
            "sequence_id": [0],
            "trial_index": [0],
            "stimOn_times": [2.0],
            "feedback_times": [4.0],
        }
    )
    trace = {
        "sample_times": times,
        "cleaned_diameter": values,
        "artifact_mask": np.zeros_like(times, dtype=bool),
    }
    features = stage.extract_session_features(session, trace)
    assert features[config.PUPIL_TONIC_COLUMN].iloc[0] == 1.0
    assert features[config.PUPIL_PHASIC_COLUMN].iloc[0] > 0.5
    assert features[config.PUPIL_FEEDBACK_PHASIC_COLUMN].iloc[0] > 1.0


def test_metric_qc_keeps_independent_flags() -> None:
    rows = []
    for trial in range(5):
        rows.append(
            {
                "subject": "m1",
                "eid": "e1",
                "sequence_id": 0,
                "trial_index": trial,
                "pupil_tonic": float(trial),
                "pupil_phasic": np.nan if trial == 2 else float(trial),
                "pupil_feedback_phasic": float(trial),
            }
        )
    result = stage.add_metric_specific_qc(pd.DataFrame(rows))
    assert result.loc[2, "pupil_tonic_ok"]
    assert not result.loc[2, "pupil_phasic_ok"]
