from __future__ import annotations
from typing import Dict, Any, List

from .structure import StructureRecord

def aggregate_mc_results(
    s: StructureRecord,
    path_results: List[Dict[str, Any]],
    *,
    market_price: float,
) -> Dict[str, float]:
    """
    Aggregate per-path results into probabilities + EV.

    market_price: price paid today (currency units, e.g., = notional for par)
    EV_investor_pct: (E[payoff] - market_price) / notional
    """
    n = len(path_results)
    if n == 0:
        raise ValueError("No path results")

    total_payoffs = [r["total_payoff"] for r in path_results]
    exp_payoff = sum(total_payoffs) / n

    # capital loss: principal_returned < notional
    loss_count = sum(1 for r in path_results if r["principal_returned"] < s.notional - 1e-12)

    # "full redemption" here means principal returned in full (regardless of coupons)
    full_redemption_count = sum(1 for r in path_results if r["principal_returned"] >= s.notional - 1e-12)

    P_capital_loss = loss_count / n
    P_full_redemption = full_redemption_count / n

    ev_investor_pct = (exp_payoff - market_price) / s.notional

    return {
        "E_payoff": float(exp_payoff),
        "P_capital_loss": float(P_capital_loss),
        "P_full_redemption": float(P_full_redemption),
        "EV_investor_pct": float(ev_investor_pct),
    }
