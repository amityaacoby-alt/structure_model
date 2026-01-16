# tests/test_f2_phoenix.py
from __future__ import annotations

from core.payoff_f2_phoenix import phoenix_payoff


# Locked v1 constants
S0 = 100.0
PR = 100.0

COUPON_RATE = 0.02          # 2% per obs
COUPON_BARRIER = 0.70       # 70% of S0
KI_BARRIER = 0.60           # 60% of S0

# Autocall step-down (example)
# i=0: 100%, i=1: 99%, i=2: 98%, i=3: 97%, ...
def stepdown(n: int, start: float = 1.00, step: float = 0.01, floor: float = 0.70):
    out = []
    for i in range(n):
        b = start - step * i
        if b < floor:
            b = floor
        out.append(b)
    return out


def test_coupon_paid_when_above_coupon_barrier():
    path = [80.0]  # >=70
    res = phoenix_payoff(
        path,
        S0=S0,
        coupon_rate=COUPON_RATE,
        coupon_barrier=COUPON_BARRIER,
        autocall_barriers=stepdown(len(path)),
        ki_barrier=KI_BARRIER,
        principal=PR,
        memory=True,
    )
    assert res.coupons == [PR * COUPON_RATE]
    assert res.redemption == PR  # maturity, no autocall
    assert res.total_payoff == PR + PR * COUPON_RATE


def test_coupon_missed_then_paid_with_memory_lump_sum():
    # miss, miss, then hit coupon barrier -> pays 3 coupons at once
    path = [60.0, 65.0, 75.0]  # <70, <70, >=70
    res = phoenix_payoff(
        path,
        S0=S0,
        coupon_rate=COUPON_RATE,
        coupon_barrier=COUPON_BARRIER,
        autocall_barriers=stepdown(len(path)),
        ki_barrier=KI_BARRIER,
        principal=PR,
        memory=True,
    )
    assert res.coupons[0] == 0.0
    assert res.coupons[1] == 0.0
    assert res.coupons[2] == PR * COUPON_RATE * 3
    assert res.autocalled is False
    assert res.redemption == PR


def test_coupon_memory_off_pays_only_single_coupon_when_hit():
    path = [60.0, 65.0, 75.0]
    res = phoenix_payoff(
        path,
        S0=S0,
        coupon_rate=COUPON_RATE,
        coupon_barrier=COUPON_BARRIER,
        autocall_barriers=stepdown(len(path)),
        ki_barrier=KI_BARRIER,
        principal=PR,
        memory=False,
    )
    assert res.coupons == [0.0, 0.0, PR * COUPON_RATE]
    assert res.redemption == PR


def test_autocall_triggers_and_stops_early():
    # Autocall barrier at i=1 is 0.99*S0 = 99
    path = [80.0, 100.0, 100.0, 100.0]
    res = phoenix_payoff(
        path,
        S0=S0,
        coupon_rate=COUPON_RATE,
        coupon_barrier=COUPON_BARRIER,
        autocall_barriers=stepdown(len(path), start=1.00, step=0.01),
        ki_barrier=KI_BARRIER,
        principal=PR,
        memory=True,
    )
    assert res.autocalled is True
    assert res.autocall_index == 1
    assert res.redemption == PR
    # Coupons after autocall should remain default 0.0 (we break loop)
    assert res.coupons[2] == 0.0
    assert res.coupons[3] == 0.0


def test_autocall_can_happen_even_if_coupon_not_paid():
    # Coupon barrier is 70, autocall barrier i=0 is 100
    # If spot >=100, coupon should still pay because also >=70.
    # To create "autocall without coupon", we would need coupon barrier > autocall barrier,
    # which is not our v1. So here we test the correct v1 invariant:
    # autocall implies coupon condition is also true.
    path = [100.0]
    res = phoenix_payoff(
        path,
        S0=S0,
        coupon_rate=COUPON_RATE,
        coupon_barrier=COUPON_BARRIER,
        autocall_barriers=[1.00],
        ki_barrier=KI_BARRIER,
        principal=PR,
        memory=True,
    )
    assert res.autocalled is True
    assert res.coupons[0] == PR * COUPON_RATE


def test_no_autocall_no_ki_returns_principal_even_if_down():
    # Never hits autocall; KI never breached; final < S0
    path = [80.0, 75.0, 71.0, 65.0]  # all > 60 (KI=60)
    res = phoenix_payoff(
        path,
        S0=S0,
        coupon_rate=COUPON_RATE,
        coupon_barrier=COUPON_BARRIER,
        autocall_barriers=stepdown(len(path)),
        ki_barrier=KI_BARRIER,
        principal=PR,
        memory=True,
    )
    assert res.ki_breached is False
    assert res.redemption == PR


def test_ki_breached_american_if_touches_any_time():
    path = [80.0, 59.0, 75.0]  # touches below 60 at i=1
    res = phoenix_payoff(
        path,
        S0=S0,
        coupon_rate=COUPON_RATE,
        coupon_barrier=COUPON_BARRIER,
        autocall_barriers=stepdown(len(path)),
        ki_barrier=KI_BARRIER,
        principal=PR,
        memory=True,
    )
    assert res.ki_breached is True


def test_ki_breached_changes_redemption_to_linear_final_ratio():
    # KI breached + final below S0 => redemption = principal * (ST/S0)
    path = [80.0, 59.0, 50.0]  # KI touched; ST=50
    res = phoenix_payoff(
        path,
        S0=S0,
        coupon_rate=0.0,  # isolate redemption
        coupon_barrier=COUPON_BARRIER,
        autocall_barriers=stepdown(len(path)),
        ki_barrier=KI_BARRIER,
        principal=PR,
        memory=True,
    )
    assert res.autocalled is False
    assert res.ki_breached is True
    assert res.redemption == PR * (50.0 / S0)


def test_never_hits_coupon_barrier_no_coupons_paid():
    path = [60.0, 69.0, 69.5, 69.9]  # always below 70
    res = phoenix_payoff(
        path,
        S0=S0,
        coupon_rate=COUPON_RATE,
        coupon_barrier=COUPON_BARRIER,
        autocall_barriers=stepdown(len(path)),
        ki_barrier=KI_BARRIER,
        principal=PR,
        memory=True,
    )
    assert all(c == 0.0 for c in res.coupons)
    # KI is not breached because all >=60
    assert res.ki_breached is False
    assert res.redemption == PR


def test_worst_case_goes_to_zero_if_final_zero_and_ki_breached():
    path = [80.0, 59.0, 0.0]  # KI touched; ST=0
    res = phoenix_payoff(
        path,
        S0=S0,
        coupon_rate=0.0,
        coupon_barrier=COUPON_BARRIER,
        autocall_barriers=stepdown(len(path)),
        ki_barrier=KI_BARRIER,
        principal=PR,
        memory=True,
    )
    assert res.ki_breached is True
    assert res.redemption == 0.0
