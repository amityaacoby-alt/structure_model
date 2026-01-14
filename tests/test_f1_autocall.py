import math
import numpy as np
import pytest

from core.structure import StructureRecord
from core.payoff_f1 import evaluate_f1_path
from core.ev import aggregate_mc_results


def _mk_structure(
    *,
    initial_price: float = 100.0,
    notional: float = 1000.0,
    maturity_years: int = 2,
    barrier_level_pct: float = 0.60,         # KI barrier
    autocall_trigger_pct: float = 1.00,      # KO trigger
    coupon_barrier_pct: float = 0.60,        # coupon condition threshold
    coupon_rate_pa: float = 0.08,            # 8% p.a.
) -> StructureRecord:
    return StructureRecord(
        structure_id="TEST_F1",
        product_type="AutocallPhoenix",
        notional=notional,
        initial_price=initial_price,
        maturity_years=maturity_years,
        barrier_level_pct=barrier_level_pct,
        autocall_trigger_pct=autocall_trigger_pct,
        coupon_barrier_pct=coupon_barrier_pct,
        coupon_rate_pa=coupon_rate_pa,
        obs_per_year=1,  # annual
    )


def _path_with_annual_obs(initial: float, year1: float, year2: float, days_per_year: int = 252) -> np.ndarray:
    """Create a daily path with exact annual observation closes at day 252 and 504."""
    total_days = 1 + days_per_year * 2
    path = np.full(total_days, initial, dtype=float)
    path[days_per_year] = year1
    path[days_per_year * 2] = year2
    return path


def test_01_no_barrier_breach_full_principal_returned():
    s = _mk_structure(barrier_level_pct=0.60)
    path = _path_with_annual_obs(100, 95, 90)
    path[:] = np.maximum(path, 70.0)

    res = evaluate_f1_path(s, path, days_per_year=252)

    assert res["knock_in_breached"] is False
    assert res["redeemed_early"] is False
    assert res["principal_returned"] == pytest.approx(s.notional)
    assert res["capital_loss"] == pytest.approx(0.0)


def test_02_barrier_breach_sets_knock_in_flag():
    s = _mk_structure(barrier_level_pct=0.60)
    path = _path_with_annual_obs(100, 95, 90)
    path[100] = 59.0

    res = evaluate_f1_path(s, path, days_per_year=252)

    assert res["knock_in_breached"] is True


def test_03_breached_and_final_below_strike_capital_loss_applies_proportional():
    s = _mk_structure(initial_price=100.0, notional=1000.0, barrier_level_pct=0.60)
    path = _path_with_annual_obs(100, 95, 80)
    path[10] = 50.0  # breach

    res = evaluate_f1_path(s, path, days_per_year=252)

    assert res["knock_in_breached"] is True
    expected_principal = s.notional * (path[-1] / s.initial_price)  # 1000 * 0.8 = 800
    assert res["principal_returned"] == pytest.approx(expected_principal)
    assert res["capital_loss"] == pytest.approx(s.notional - expected_principal)


def test_04_breached_but_final_at_or_above_strike_full_principal_returned():
    s = _mk_structure(initial_price=100.0, notional=1000.0, barrier_level_pct=0.60)
    path = _path_with_annual_obs(100, 90, 110)
    path[50] = 55.0  # breach

    res = evaluate_f1_path(s, path, days_per_year=252)

    assert res["knock_in_breached"] is True
    assert res["principal_returned"] == pytest.approx(s.notional)
    assert res["capital_loss"] == pytest.approx(0.0)


def test_05_autocall_triggers_early_redemption_and_pays_coupon():
    s = _mk_structure(
        maturity_years=2,
        autocall_trigger_pct=1.00,
        coupon_rate_pa=0.08,
        coupon_barrier_pct=0.60,
    )
    path = _path_with_annual_obs(100, 100, 10)

    res = evaluate_f1_path(s, path, days_per_year=252)

    assert res["redeemed_early"] is True
    assert res["redeem_year"] == 1
    assert res["principal_returned"] == pytest.approx(s.notional)
    assert res["total_coupons"] == pytest.approx(s.notional * s.coupon_rate_pa)


def test_06_autocall_missed_continues_to_maturity():
    s = _mk_structure(maturity_years=2, autocall_trigger_pct=1.00)
    path = _path_with_annual_obs(100, 99, 90)
    path[:] = np.maximum(path, 70.0)

    res = evaluate_f1_path(s, path, days_per_year=252)

    assert res["redeemed_early"] is False
    assert res["redeem_year"] is None
    assert res["principal_returned"] == pytest.approx(s.notional)


def test_07_coupon_paid_when_condition_met():
    s = _mk_structure(
        maturity_years=2,
        coupon_rate_pa=0.10,
        coupon_barrier_pct=0.60,
        autocall_trigger_pct=10.0,  # disable autocall
    )
    path = _path_with_annual_obs(100, 70, 70)
    path[:] = np.maximum(path, 70.0)

    res = evaluate_f1_path(s, path, days_per_year=252)

    assert res["coupons_paid_by_year"][1] == pytest.approx(s.notional * s.coupon_rate_pa)


def test_08_coupon_not_paid_when_condition_missed():
    s = _mk_structure(
        maturity_years=2,
        coupon_rate_pa=0.10,
        coupon_barrier_pct=0.60,
        autocall_trigger_pct=10.0,  # disable autocall
    )
    path = _path_with_annual_obs(100, 50, 70)
    path[:] = np.maximum(path, 70.0)
    path[252] = 50.0

    res = evaluate_f1_path(s, path, days_per_year=252)

    assert res["coupons_paid_by_year"][1] == pytest.approx(0.0)


def test_09_coupons_do_not_offset_capital_loss():
    s = _mk_structure(
        maturity_years=2,
        coupon_rate_pa=0.10,
        coupon_barrier_pct=0.60,
        autocall_trigger_pct=10.0,  # disable autocall
        barrier_level_pct=0.60,
    )
    path = _path_with_annual_obs(100, 70, 80)
    path[20] = 50.0  # KI breach

    res = evaluate_f1_path(s, path, days_per_year=252)

    assert res["principal_returned"] == pytest.approx(s.notional * 0.8)
    assert res["capital_loss"] == pytest.approx(s.notional * 0.2)
    assert res["total_coupons"] > 0


def test_10_probabilities_and_ev_sanity():
    s = _mk_structure(maturity_years=2, autocall_trigger_pct=10.0)  # disable autocall

    pA = _path_with_annual_obs(100, 90, 90); pA[:] = np.maximum(pA, 70.0)
    pB = _path_with_annual_obs(100, 90, 80); pB[10] = 50.0
    pC = _path_with_annual_obs(100, 90, 110); pC[10] = 50.0
    pD = _path_with_annual_obs(100, 100, 100); pD[:] = np.maximum(pD, 70.0)

    results = [evaluate_f1_path(s, p, days_per_year=252) for p in (pA, pB, pC, pD)]
    agg = aggregate_mc_results(s, results, market_price=s.notional)

    assert 0.0 <= agg["P_capital_loss"] <= 1.0
    assert 0.0 <= agg["P_full_redemption"] <= 1.0
    assert agg["P_capital_loss"] + agg["P_full_redemption"] <= 1.0 + 1e-12
    assert math.isfinite(agg["EV_investor_pct"])
    assert not math.isnan(agg["EV_investor_pct"])
    assert agg["P_capital_loss"] == pytest.approx(0.25)
