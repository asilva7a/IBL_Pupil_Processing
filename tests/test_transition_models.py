from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import load_stage

stage = load_stage("06_fit_transition_models.py", "stage06")


def test_match_bursts_is_one_to_one() -> None:
    rows = []
    for trial in range(12):
        rows.append(
            {
                "subject": "m1",
                "eid": "e1",
                "state": 0,
                "state_label": "engaged",
                "reward": 0.0,
                "trial_index": trial,
                "trial_progress": trial / 11,
                "position_bin": min(int((trial / 11) * 10), 9),
                "isolated_burst": trial in {3, 8},
                "burst_complete_window": True,
                "eligible_nonburst_control": trial in {2, 4, 7, 9},
                "future_js_max_3": trial / 100,
            }
        )
    matches, diagnostics = stage.match_bursts(
        pd.DataFrame(rows),
        specification="test",
        max_bin_difference=1,
        max_fraction_difference=0.2,
    )
    assert len(matches) == 2
    assert matches["control_trial_index"].is_unique
    assert diagnostics["matched_pairs"] == 2
