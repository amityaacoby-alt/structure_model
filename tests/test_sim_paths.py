# tests/test_sim_paths.py
from __future__ import annotations

import numpy as np
import pytest

from core.sim_paths import SimSpec, simulate_gbm_paths, make_monthly_dt


def test_shapes_and_reproducibility():
    spec = SimSpec(S0=100.0, mu=0.0, sigma=0.2, steps=12, dt=1/12, n_paths=500, seed=42)
    a = simulate_gbm_paths(spec)
    b = simulate_gbm_paths(spec)
    assert a.shape == (500, 12)
    assert np.allclose(a, b)


def test_zero_vol_is_deterministic_growth():
    S0 = 100.0
    mu = 0.12
    dt = 1/12
    steps = 12
    spec = SimSpec(S0=S0, mu=mu, sigma=0.0, steps=steps, dt=dt, n_paths=3, seed=7)
    paths = simulate_gbm_paths(spec)

    expected = np.array([S0 * np.exp(mu * dt * (k + 1)) for k in range(steps)], dtype=float)
    for i in range(paths.shape[0]):
        assert np.allclose(paths[i], expected)


def test_make_monthly_dt():
    assert make_monthly_dt(12) == pytest.approx(1/12)
    assert make_monthly_dt(4) == pytest.approx(1/4)
    with pytest.raises(ValueError):
        make_monthly_dt(0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        simulate_gbm_paths(SimSpec(S0=0.0, mu=0.0, sigma=0.2, steps=12, dt=1/12, n_paths=10))
    with pytest.raises(ValueError):
        simulate_gbm_paths(SimSpec(S0=100.0, mu=0.0, sigma=-0.1, steps=12, dt=1/12, n_paths=10))
    with pytest.raises(ValueError):
        simulate_gbm_paths(SimSpec(S0=100.0, mu=0.0, sigma=0.2, steps=0, dt=1/12, n_paths=10))
    with pytest.raises(ValueError):
        simulate_gbm_paths(SimSpec(S0=100.0, mu=0.0, sigma=0.2, steps=12, dt=0.0, n_paths=10))
    with pytest.raises(ValueError):
        simulate_gbm_paths(SimSpec(S0=100.0, mu=0.0, sigma=0.2, steps=12, dt=1/12, n_paths=0))
