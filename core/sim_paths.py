# core/sim_paths.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def make_monthly_dt(months: int) -> float:
    """
    Tests expect: make_monthly_dt(12) == 1/12
    """
    if months <= 0:
        raise ValueError("months must be > 0")
    return 1.0 / float(months)


@dataclass(frozen=True)
class SimSpec:
    # NOTE: tests expect field name S0 (capital)
    S0: float
    mu: float
    sigma: float
    steps: int
    dt: float
    n_paths: int
    seed: Optional[int] = None


def simulate_gbm_paths(spec: SimSpec) -> np.ndarray:
    """
    Single-asset GBM simulator.

    IMPORTANT (per tests):
    - Returns shape (n_paths, steps)
    - Contains S_1..S_steps (does NOT include S0 column)
    """
    if spec.steps <= 0:
        raise ValueError("steps must be > 0")
    if spec.n_paths <= 0:
        raise ValueError("n_paths must be > 0")
    if spec.dt <= 0:
        raise ValueError("dt must be > 0")
    if spec.S0 <= 0:
        raise ValueError("S0 must be > 0")
    if spec.sigma < 0:
        raise ValueError("sigma must be >= 0")

    rng = np.random.default_rng(spec.seed)

    # we simulate internally with S0, then drop it to match expected API
    paths = np.zeros((spec.n_paths, spec.steps + 1), dtype=float)
    paths[:, 0] = spec.S0

    drift = (spec.mu - 0.5 * spec.sigma**2) * spec.dt
    vol = spec.sigma * np.sqrt(spec.dt)

    for t in range(1, spec.steps + 1):
        z = rng.standard_normal(spec.n_paths)
        paths[:, t] = paths[:, t - 1] * np.exp(drift + vol * z)

    # Drop S0 column → return S1..S_steps
    return paths[:, 1:]


def simulate_correlated_gbm_paths(
    s0_vec,
    mu_vec,
    sigma_vec,
    rho: float,
    T: float,
    steps: int,
    n_paths: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Correlated GBM simulator for 2 assets.

    Returns:
        np.ndarray shape (n_paths, steps+1, 2) including S0
    """
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1, 1]")

    s0_vec = np.asarray(s0_vec, dtype=float)
    mu_vec = np.asarray(mu_vec, dtype=float)
    sigma_vec = np.asarray(sigma_vec, dtype=float)

    if s0_vec.shape != (2,) or mu_vec.shape != (2,) or sigma_vec.shape != (2,):
        raise ValueError("s0_vec, mu_vec, sigma_vec must each have length 2")
    if np.any(s0_vec <= 0):
        raise ValueError("all s0 must be > 0")
    if np.any(sigma_vec < 0):
        raise ValueError("all sigma must be >= 0")

    dt = T / steps
    rng = np.random.default_rng(seed)

    corr = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    L = np.linalg.cholesky(corr)

    paths = np.zeros((n_paths, steps + 1, 2), dtype=float)
    paths[:, 0, :] = s0_vec

    drift = (mu_vec - 0.5 * sigma_vec**2) * dt
    vol = sigma_vec * np.sqrt(dt)

    for t in range(1, steps + 1):
        z = rng.standard_normal((n_paths, 2))
        zc = z @ L.T
        paths[:, t, :] = paths[:, t - 1, :] * np.exp(drift + vol * zc)

    return paths

