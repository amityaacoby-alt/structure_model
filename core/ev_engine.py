# core/ev_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np


@dataclass(frozen=True)
class EVSummary:
    """
    Generic EV summary for any payoff function on simulated paths.

    ev: expected payoff (mean)
    std: standard deviation of payoff
    p_loss: probability(payoff < principal) if principal provided, else None
    p_gain: probability(payoff > principal) if principal provided, else None
    p_zero: probability(payoff == 0)
    q05/q50/q95: payoff quantiles
    n: number of paths evaluated
    """
    ev: float
    std: float
    q05: float
    q50: float
    q95: float
    p_zero: float
    p_loss: Optional[float]
    p_gain: Optional[float]
    n: int


def estimate_ev(
    payoff_fn: Callable[..., Any],
    paths: Sequence[Sequence[float]],
    *,
    principal: Optional[float] = None,
    return_detail: bool = False,
    **payoff_kwargs: Any,
) -> Tuple[EVSummary, Optional[np.ndarray]]:
    """
    Evaluate EV and distribution stats for a payoff function over many paths.

    payoff_fn:
        Function that accepts (path, **kwargs) and returns either:
        - float (total payoff), or
        - dataclass/object/dict with a 'total_payoff' field/key (PhoenixResult style)

    paths:
        List of simulated paths. Each path is a list/array of spot values at observation dates.

    principal:
        If provided, compute p_loss = P(payoff < principal), p_gain = P(payoff > principal)

    return_detail:
        If True, also return the raw payoff array.

    payoff_kwargs:
        Passed through to payoff_fn.
    """
    if len(paths) == 0:
        raise ValueError("paths must be non-empty")

    payoffs: List[float] = []
    for path in paths:
        out = payoff_fn(list(path), **payoff_kwargs)

        # Normalize output -> float total payoff
        if isinstance(out, (int, float, np.floating)):
            total = float(out)
        elif isinstance(out, dict) and "total_payoff" in out:
            total = float(out["total_payoff"])
        else:
            total = float(getattr(out, "total_payoff"))

        payoffs.append(total)

    arr = np.asarray(payoffs, dtype=float)

    ev = float(arr.mean())
    std = float(arr.std(ddof=0))

    q05, q50, q95 = (float(np.quantile(arr, q)) for q in (0.05, 0.50, 0.95))
    p_zero = float((arr == 0.0).mean())

    p_loss = None
    p_gain = None
    if principal is not None:
        p_loss = float((arr < principal).mean())
        p_gain = float((arr > principal).mean())

    summary = EVSummary(
        ev=ev,
        std=std,
        q05=q05,
        q50=q50,
        q95=q95,
        p_zero=p_zero,
        p_loss=p_loss,
        p_gain=p_gain,
        n=int(arr.size),
    )

    return (summary, arr) if return_detail else (summary, None)
