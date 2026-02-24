from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class WorstOfPhoenixResult:
    total_payoff: float
    coupons: List[float]
    redemption: float
    autocalled: bool
    autocall_index: Optional[int]
    ki_breached: bool
    worst_ratio_path: List[float]


def _as_2d_path(path: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Expect path as shape (steps, n_assets)
    """
    arr = np.asarray(path, dtype=float)

    if arr.ndim != 2:
        raise ValueError("path must be 2D (steps x assets)")

    if arr.shape[0] < 1:
        raise ValueError("path must have at least one observation")

    if arr.shape[1] < 2:
        raise ValueError("worst-of requires at least 2 assets")

    if np.any(~np.isfinite(arr)):
        raise ValueError("path contains invalid values")

    return arr


def worstof_phoenix_payoff(
    path: Sequence[Sequence[float]],
    *,
    S0_vec: Sequence[float],
    coupon_rate: float,
    coupon_barrier: float,
    autocall_barriers: Sequence[float],
    ki_barrier: float,
    principal: float = 100.0,
    memory: bool = True,
) -> WorstOfPhoenixResult:
    """
    Worst-Of Phoenix payoff logic.

    Rules:
    - worst_ratio[t] = min_i(S_i(t) / S0_i)
    - Coupon paid if worst_ratio >= coupon_barrier
    - If memory=True, missed coupons accumulate
    - Autocall if worst_ratio >= autocall_barriers[t]
    - KI breach is STRICT (< ki_barrier)
    """

    p = _as_2d_path(path)
    S0 = np.asarray(S0_vec, dtype=float)

    if S0.ndim != 1 or S0.shape[0] != p.shape[1]:
        raise ValueError("S0_vec length must equal number of assets")

    if np.any(S0 <= 0):
        raise ValueError("S0 values must be positive")

    T, K = p.shape

    if len(autocall_barriers) != T:
        raise ValueError("autocall_barriers must match number of steps")

    if principal <= 0:
        raise ValueError("principal must be positive")

    ratios = p / S0
    worst_ratios = np.min(ratios, axis=1)

    # KI breach (STRICT)
    ki_breached = bool(np.any(worst_ratios < ki_barrier))

    coupons = [0.0] * T
    missed = 0

    autocalled = False
    autocall_index = None
    redemption = 0.0

    for t in range(T):
        wr = float(worst_ratios[t])

        # Coupon logic
        if wr >= coupon_barrier:
            if memory:
                pay_n = missed + 1
                coupons[t] = principal * coupon_rate * pay_n
                missed = 0
            else:
                coupons[t] = principal * coupon_rate
        else:
            if memory:
                missed += 1

        # Autocall logic
        if wr >= float(autocall_barriers[t]):
            autocalled = True
            autocall_index = t
            redemption = principal

            # Stop paying future coupons
            for j in range(t + 1, T):
                coupons[j] = 0.0
            break

    # Final redemption if no autocall
    if not autocalled:
        final_wr = float(worst_ratios[-1])

        if not ki_breached:
            redemption = principal
        else:
            # proportional loss (capped at principal)
            redemption = principal * min(1.0, final_wr)

    total_payoff = float(np.sum(coupons) + redemption)

    return WorstOfPhoenixResult(
        total_payoff=total_payoff,
        coupons=coupons,
        redemption=float(redemption),
        autocalled=autocalled,
        autocall_index=autocall_index,
        ki_breached=ki_breached,
        worst_ratio_path=[float(x) for x in worst_ratios],
    )