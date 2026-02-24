# core/ev_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EVSummary:
    """
    Generic EV summary for any payoff function on simulated paths.

    ev: expected payoff (mean)
    std: standard deviation of payoff
    p_loss: P(payoff < principal) if principal provided
    p_gain: P(payoff > principal) if principal provided
    p_zero: P(payoff == 0)
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


def _extract_total_payoff(out: Any) -> float:
    """
    Normalize payoff function output into a float.

    Supported outputs:
    - float / int
    - dict with "total_payoff"
    - object with attribute .total_payoff
    """
    if isinstance(out, (int, float, np.floating)):
        return float(out)
    if isinstance(out, dict) and "total_payoff" in out:
        return float(out["total_payoff"])
    return float(getattr(out, "total_payoff"))


def estimate_ev(
    payoff_fn: Callable[..., Any],
    paths: Any,
    *,
    principal: Optional[float] = None,
    return_detail: bool = False,
    payoff_kwargs: Optional[Dict[str, Any]] = None,
    discount_rate: float = 0.0,
    dt: Optional[float] = None,
    **payoff_kwargs_passthrough: Any,
) -> Tuple[EVSummary, Optional[np.ndarray]]:
    """
    Evaluate EV and distribution stats for a payoff function over many paths.

    IMPORTANT:
    - This function is backward compatible with tests that call:
        estimate_ev(payoff_fn, paths, ...)

    Inputs:
    - payoff_fn(path, **kwargs) -> float OR object/dict with total_payoff
    - paths:
        * list of paths
        * or np.ndarray (n_paths, steps)
        * or np.ndarray (n_paths, steps, assets)
    - payoff kwargs:
        A) payoff_kwargs={...}
        B) direct kwargs like coupon_rate=..., etc. (overrides dict)

    Discounting:
    - If discount_rate != 0, requires dt.
    - Discount factor is exp(-r * T) where T = steps * dt
    - Applied once to total payoff (PV in one shot)
    """
    if payoff_kwargs is None:
        payoff_kwargs = {}
    payoff_kwargs = {**payoff_kwargs, **payoff_kwargs_passthrough}

    arr_paths = np.asarray(paths, dtype=float)

    if arr_paths.ndim == 1:
        raise ValueError("paths must be 2D or 3D: (n_paths, steps) or (n_paths, steps, assets)")
    if arr_paths.ndim == 2:
        iterable_paths = [arr_paths[i, :].tolist() for i in range(arr_paths.shape[0])]
        steps = arr_paths.shape[1]
    elif arr_paths.ndim == 3:
        iterable_paths = [arr_paths[i, :, :].tolist() for i in range(arr_paths.shape[0])]
        steps = arr_paths.shape[1]
    else:
        raise ValueError("paths must be 2D or 3D")

    if len(iterable_paths) == 0:
        raise ValueError("paths must be non-empty")

    payoffs: List[float] = []
    for p in iterable_paths:
        out = payoff_fn(p, **payoff_kwargs)
        payoffs.append(_extract_total_payoff(out))

    pay = np.asarray(payoffs, dtype=float)

    # optional PV discount
    if discount_rate != 0.0:
        if dt is None:
            raise ValueError("dt must be provided when discount_rate != 0")
        T = float(steps) * float(dt)
        pay = pay * float(np.exp(-discount_rate * T))

    ev = float(pay.mean())
    std = float(pay.std(ddof=0))
    q05, q50, q95 = (float(np.quantile(pay, q)) for q in (0.05, 0.50, 0.95))
    p_zero = float((pay == 0.0).mean())

    p_loss = None
    p_gain = None
    if principal is not None:
        p_loss = float((pay < principal).mean())
        p_gain = float((pay > principal).mean())

    summary = EVSummary(
        ev=ev,
        std=std,
        q05=q05,
        q50=q50,
        q95=q95,
        p_zero=p_zero,
        p_loss=p_loss,
        p_gain=p_gain,
        n=int(pay.size),
    )

    return (summary, pay) if return_detail else (summary, None)