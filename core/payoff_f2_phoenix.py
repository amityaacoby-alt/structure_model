# core/payoff_f2_phoenix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PhoenixResult:
    """
    Result of Phoenix payoff evaluation (one underlying, discrete obs).

    total_payoff: total cash paid over life (coupons + redemption), in currency units.
    coupons: list of coupon cashflows per obs date (0 if not paid).
    redemption: redemption cashflow (paid at autocall date or maturity).
    autocalled: True if autocalled early.
    autocall_index: 0-based index of the obs date where autocall happened (None if not).
    ki_breached: True if KI barrier was breached at ANY time (American KI).
    """
    total_payoff: float
    coupons: List[float]
    redemption: float
    autocalled: bool
    autocall_index: Optional[int]
    ki_breached: bool


def phoenix_payoff(
    path: List[float],
    *,
    S0: float,
    coupon_rate: float,
    coupon_barrier: float,
    autocall_barriers: List[float],
    ki_barrier: float,
    principal: float = 100.0,
    memory: bool = True,
) -> PhoenixResult:
    """
    Phoenix Note v1 (locked spec):
    - Monthly observation dates (the path is the obs spots).
    - Coupon barrier: coupon_barrier * S0 (e.g., 0.70 * S0)
    - Autocall barriers: autocall_barriers[i] * S0 (step-down list)
    - KI barrier: ki_barrier * S0 (American KI: breach anytime)
    - Coupon memory: if memory=True, missed coupons accumulate and are paid in a lump sum
      the next time coupon condition is met.
    - Autocall: if S[i] >= autocall_barriers[i]*S0, redeem early at principal (plus coupon if due).
    - Maturity redemption (if not autocalled):
        if KI not breached -> principal
        if KI breached -> principal * (S_T / S0)

    IMPORTANT (v1 convention):
    - KI breach is STRICTLY below the barrier: s < ki_level (not <=).
    """
    if S0 <= 0:
        raise ValueError("S0 must be > 0")
    if principal <= 0:
        raise ValueError("principal must be > 0")
    if coupon_rate < 0:
        raise ValueError("coupon_rate must be >= 0")
    if not path:
        raise ValueError("path must be non-empty")
    if len(autocall_barriers) != len(path):
        raise ValueError("autocall_barriers must have same length as path")

    coupon_level = coupon_barrier * S0
    autocall_levels = [b * S0 for b in autocall_barriers]
    ki_level = ki_barrier * S0

    coupons: List[float] = [0.0] * len(path)
    missed = 0  # number of missed coupons (for memory)

    # American KI (touch anytime) — STRICTLY below barrier
    ki_breached = any(s < ki_level for s in path)

    autocalled = False
    autocall_index: Optional[int] = None
    redemption = 0.0

    for i, s in enumerate(path):
        # 1) Coupon logic (memory)
        coupon_hit = s >= coupon_level
        if coupon_hit:
            pay_mult = (missed + 1) if memory else 1
            coupons[i] = principal * coupon_rate * pay_mult
            missed = 0
        else:
            coupons[i] = 0.0
            if memory:
                missed += 1

        # 2) Autocall logic (can happen on same obs as coupon)
        if s >= autocall_levels[i]:
            autocalled = True
            autocall_index = i
            redemption = principal
            break  # stop after this obs date (coupon + redemption handled)

    if not autocalled:
        # 3) Maturity redemption
        ST = path[-1]
        if not ki_breached:
            redemption = principal
        else:
            redemption = principal * (ST / S0)

    total = sum(coupons) + redemption

    return PhoenixResult(
        total_payoff=float(total),
        coupons=[float(x) for x in coupons],
        redemption=float(redemption),
        autocalled=autocalled,
        autocall_index=autocall_index,
        ki_breached=ki_breached,
    )

