from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import load_stage

stage = load_stage("03_align_glmhmm_states.py", "stage03")


def test_design_matrix_resets_history_with_each_session() -> None:
    session = pd.DataFrame(
        {
            "subject": ["m1", "m1"],
            "eid": ["e1", "e1"],
            "sequence_id": [0, 0],
            "trial_index": [0, 1],
            "choice": [-1, 1],
            "signed_contrast": [0.5, -0.5],
            "rightward_choice": [1.0, 0.0],
        }
    )
    X, y, _ = stage.build_design_matrix(session)
    assert X[0, 2] == 0.0
    assert X[0, 3] == 0.0
    assert X[1, 2] == 1.0
    assert X[1, 3] == 1.0
    assert y.tolist() == [1.0, 0.0]


def test_glmhmm_multiple_sequences_return_valid_posteriors() -> None:
    rng = np.random.default_rng(4)
    sequences = []
    for _ in range(2):
        X = np.column_stack(
            [rng.normal(size=60), np.ones(60), rng.choice([-1, 1], 60), rng.choice([-1, 1], 60)]
        )
        y = (rng.random(60) < 1 / (1 + np.exp(-2 * X[:, 0]))).astype(float)
        sequences.append(stage.SequenceData(X, y))
    model = stage.GLMHMM(seed=2).fit(sequences, n_iter=15, minimum_iterations=3)
    for sequence in sequences:
        posterior = model.posterior(sequence)
        assert posterior.shape == (60, 3)
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-8)
    total = model.loglik(sequences)
    separate = sum(model.loglik([sequence]) for sequence in sequences)
    assert total == separate
