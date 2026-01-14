from __future__ import annotations
from typing import Dict, Any
import numpy as np

from .structure import StructureRecord

def evaluate_f1_path(s: StructureRecord, price_path: np.ndarray, *, days_per_year: int = 252) -> Dict[str, Any]:
    """
    Evaluate a SINGLE daily-close price path for F1 payoff family (Autocall/Phoenix/Reverse Convertible).

    Conventions (LOCKED):
    - Annual observations: years 1..maturity_years at day = year*days_per_year
    - Coupon at autocall: paid
    - Barrier monitoring: American (daily close)
    - Discounting: ignored
    """
    if price_path.ndim != 1:
        raise ValueError("price_path must be 1D array of daily closes")
    if len(price_path) < 1 + s.maturity_years * days_per_year:
        raise ValueError("price_path too short for maturity")

    ki_level = s.initial_price * s.barrier_level_pct
    ko_level = s.initial_price * s.autocall_trigger_pct
    coupon_level = s.initial_price * s.coupon_barrier_pct

    knock_in_breached = bool(np.any(price_path < ki_level))

    coupons_paid_by_year: Dict[int, float] = {}
    redeemed_early = False
    redeem_year = None

    # Annual observation dates
    for year in range(1, s.maturity_years + 1):
        obs_idx = year * days_per_year
        obs_price = float(price_path[obs_idx])

        # Coupon condition
        coupon_paid = s.notional * s.coupon_rate_pa if obs_price >= coupon_level else 0.0
        coupons_paid_by_year[year] = coupon_paid

        # Autocall check (if triggers: redeem and stop)
        if obs_price >= ko_level:
            redeemed_early = True
            redeem_year = year
            total_coupons = sum(coupons_paid_by_year[y] for y in range(1, year + 1))
            principal_returned = s.notional
            capital_loss = 0.0
            return {
                "knock_in_breached": knock_in_breached,
                "redeemed_early": redeemed_early,
                "redeem_year": redeem_year,
                "coupons_paid_by_year": coupons_paid_by_year,
                "total_coupons": total_coupons,
                "principal_returned": principal_returned,
                "capital_loss": capital_loss,
                "total_payoff": principal_returned + total_coupons,
            }

    # No early redemption → maturity payoff
    total_coupons = sum(coupons_paid_by_year.values())
    final_price = float(price_path[s.maturity_years * days_per_year])

    if (not knock_in_breached) or (final_price >= s.initial_price):
        principal_returned = s.notional
        capital_loss = 0.0
    else:
        # KI breached AND final below strike: proportional loss
        principal_returned = s.notional * (final_price / s.initial_price)
        principal_returned = max(0.0, principal_returned)
        capital_loss = s.notional - principal_returned

    return {
        "knock_in_breached": knock_in_breached,
        "redeemed_early": redeemed_early,
        "redeem_year": redeem_year,
        "coupons_paid_by_year": coupons_paid_by_year,
        "total_coupons": total_coupons,
        "principal_returned": principal_returned,
        "capital_loss": capital_loss,
        "total_payoff": principal_returned + total_coupons,
    }
