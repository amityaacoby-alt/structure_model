
# tests/test_ev_engine.py
from __future__ import annotations

import pytest

from core.ev_engine import estimate_ev
from core.payoff_f2_phoenix import phoenix_payoff


def test_estimate_ev_works_with_float_payoff():
    # payoff is just the final spot (toy)
    def payoff_final(path, **kwargs):
        return float(path[-1])

    paths = [
        [100, 110],
        [100,  90],
        [100, 100],
    ]

    summary, _ = estimate_ev(payoff_final, paths)
    assert summary.n == 3
    assert summary.ev == pytest.approx((110 + 90 + 100) / 3)


def test_estimate_ev_handles_object_with_total_payoff():
    # Use PhoenixResult output (has .total_payoff)
    S0 = 100.0
    paths = [
        [80.0, 80.0, 80.0, 80.0],     # coupons likely, no autocall
        [100.0, 100.0, 100.0, 100.0], # autocall early at t0
    ]

    def stepdown(n, start=1.00, step=0.01, floor=0.70):
        out = []
        for i in range(n):
            b = start - step * i
            if b < floor:
                b = floor
            out.append(b)
        return out

    summary, arr = estimate_ev(
        phoenix_payoff,
        paths,
        return_detail=True,
        principal=100.0,  # used ONLY for p_loss / p_gain stats
        S0=S0,
        coupon_rate=0.02,
        coupon_barrier=0.70,
        autocall_barriers=stepdown(len(paths[0])),
        ki_barrier=0.60,
        memory=True,
    )

    assert summary.n == 2
    assert arr is not None
    assert len(arr) == 2
    # Should produce reasonable payoffs >= 0
    assert float(arr.min()) >= 0.0


def test_principal_stats_p_loss_p_gain():
    def payoff(path, **kwargs):
        # return either 80 or 120
        return 80.0 if path[-1] < 100 else 120.0

    paths = [
        [100,  90],  # loss
        [100, 110],  # gain
        [100, 110],  # gain
        [100,  90],  # loss
    ]

    summary, _ = estimate_ev(payoff, paths, principal=100.0)
    assert summary.p_loss == pytest.approx(0.5)
    assert summary.p_gain == pytest.approx(0.5)


def test_empty_paths_raises():
    def payoff(path, **kwargs):
        return 0.0

    with pytest.raises(ValueError):
        estimate_ev(payoff, [])
