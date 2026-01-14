from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class StructureRecord:
    structure_id: str
    product_type: str

    notional: float
    initial_price: float

    maturity_years: int
    barrier_level_pct: float           # KI barrier (e.g., 0.60 means 60% of initial)
    autocall_trigger_pct: float        # KO trigger (e.g., 1.00 means 100% of initial)
    coupon_barrier_pct: float          # coupon condition threshold
    coupon_rate_pa: float              # annual coupon rate (e.g., 0.08 = 8% p.a.)
    obs_per_year: int = 1              # annual by default (LOCKED for v1)
