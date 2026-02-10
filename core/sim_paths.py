# core/sim_paths.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SimSpec:
    """
    Simple GBM simulator spec.

    S0: initial spot
    mu: drift (annualized, continuously compounded)
    sigma: vol (annualized)
    steps: number of observation steps (length of each path)
    dt: time step in years (e.g. 1/12 for monthly)
    n_paths: number of simulated paths
    seed: RNG seed for reproducibility
    """
    S0: float
    mu: float
    sigma: float
    steps: int
    dt: float
    n_paths: int
    seed: Optional[int] = 123


def simulate_gbm_paths(spec: SimSpec) -> np.ndarray:
    """
    Simulate GBM paths with log scheme:
        S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Returns:
        ndarray shape (n_paths, steps)
        Each row is a path of observation spots (first element is after 1*dt).
    """
    if spec.S0 <= 0:
        raise ValueError("S0 must be > 0")
    if spec.sigma < 0:
        raise ValueError("sigma must be >= 0")
    if spec.steps <= 0:
        raise ValueError("steps must be > 0")
    if spec.dt <= 0:
        raise ValueError("dt must be > 0")
    if spec.n_paths <= 0:
        raise ValueError("n_paths must be > 0")

    rng = np.random.default_rng(spec.seed)
    Z = rng.standard_normal(size=(spec.n_paths, spec.steps))

    drift = (spec.mu - 0.5 * (spec.sigma ** 2)) * spec.dt
    vol_term = spec.sigma * np.sqrt(spec.dt)

    log_increments = drift + vol_term * Z
    log_paths = np.cumsum(log_increments, axis=1)
    paths = spec.S0 * np.exp(log_paths)

    return paths.astype(float)


def make_monthly_dt(months_per_year: int = 12) -> float:
    if months_per_year <= 0:
        raise ValueError("months_per_year must be > 0")
    return 1.0 / float(months_per_year)
