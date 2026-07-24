from __future__ import annotations

import numpy as np
import pandas as pd

import config
from conftest import load_stage

stage = load_stage("04_fit_rl_models.py", "stage04")


def test_q_values_reset_between_sessions() -> None:
    keys1 = pd.DataFrame(
        {"subject": ["m"] * 2, "eid": ["a"] * 2, "sequence_id": [0] * 2, "trial_index": [0, 1]}
    )
    keys2 = pd.DataFrame(
        {"subject": ["m"] * 2, "eid": ["b"] * 2, "sequence_id": [1] * 2, "trial_index": [0, 1]}
    )
    sequences = [
        stage.RLSequence(np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.zeros(2), keys1),
        stage.RLSequence(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.zeros(2), keys2),
    ]
    parameters = {
        "alpha": 0.5,
        "beta_value": 1.0,
        "beta_stimulus": 1.0,
        "bias": 0.0,
        "lapse": 0.01,
    }
    regressors = stage.generate_trial_regressors(sequences, parameters)
    first_second_session = regressors.loc[regressors["eid"] == "b"].iloc[0]
    assert first_second_session["rl_q_left"] == config.RL_INITIAL_Q_LEFT
    assert first_second_session["rl_q_right"] == config.RL_INITIAL_Q_RIGHT


def test_hybrid_fit_returns_bounded_parameters() -> None:
    rng = np.random.default_rng(3)
    stimulus = rng.choice([-1.0, -0.5, 0.5, 1.0], 100)
    probability = 1 / (1 + np.exp(-4 * stimulus))
    choice = (rng.random(100) < probability).astype(float)
    reward = (choice == (stimulus > 0)).astype(float)
    sequence = stage.RLSequence(choice, reward, stimulus)
    result = stage.fit_model([sequence], model="hybrid", seed=5, n_restarts=2)
    assert np.isfinite(result.negative_log_likelihood)
    for name, value in result.parameters.items():
        lower, upper = config.RL_PARAMETER_BOUNDS[name]
        assert lower <= value <= upper
