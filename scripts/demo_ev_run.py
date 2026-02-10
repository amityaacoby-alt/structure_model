# scripts/demo_ev_run.py
from __future__ import annotations

import inspect

from core.payoff_f2_phoenix import phoenix_payoff
from core import sim_paths
from core.ev_engine import estimate_ev


def _call_simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    dt: float,
    steps: int,
    n_paths: int,
    seed: int = 42,
    r: float = 0.0,
):
    """
    Adapter for core.sim_paths.simulate_gbm_paths across possible signatures.
    """
    fn = sim_paths.simulate_gbm_paths
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    # Case 1: single spec argument (SimSpec)
    if len(params) == 1:
        SimSpec = getattr(sim_paths, "SimSpec", None)
        if SimSpec is None:
            raise RuntimeError("simulate_gbm_paths expects spec, but SimSpec not found in core.sim_paths")

        spec = SimSpec(
            S0=S0,
            mu=mu,
            sigma=sigma,
            steps=steps,
            dt=dt,
            n_paths=n_paths,
            seed=seed,
        )
        return fn(spec)

    # Case 2: keyword call (various naming)
    kw = {}

    # spot naming
    if "S0" in params:
        kw["S0"] = S0
    elif "s0" in params:
        kw["s0"] = S0
    elif "spot" in params:
        kw["spot"] = S0

    if "mu" in params:
        kw["mu"] = mu
    if "sigma" in params:
        kw["sigma"] = sigma

    if "dt" in params:
        kw["dt"] = dt
    elif "T" in params:
        kw["T"] = dt * steps

    if "steps" in params:
        kw["steps"] = steps
    elif "n_steps" in params:
        kw["n_steps"] = steps

    if "n_paths" in params:
        kw["n_paths"] = n_paths

    if "r" in params:
        kw["r"] = r
    if "seed" in params:
        kw["seed"] = seed

    try:
        return fn(**kw)
    except TypeError:
        # Case 3: positional fallback (best-effort)
        positional = [S0, mu, sigma]
        if "r" in params:
            positional.append(r)
        if "dt" in params:
            positional.append(dt)
        elif "T" in params:
            positional.append(dt * steps)
        positional.append(steps)
        positional.append(n_paths)
        if "seed" in params:
            positional.append(seed)
        return fn(*positional)


def _call_estimate_ev(payoff_fn, paths, principal: float, discount_rate: float, payoff_kwargs: dict):
    """
    Adapter for core.ev_engine.estimate_ev across possible signatures.
    Key rule:
      - Do NOT pass principal/discount_rate into the payoff function kwargs.
      - Pass principal/discount_rate to estimate_ev only if it accepts them.
    """
    fn = estimate_ev
    sig = inspect.signature(fn)
    params = sig.parameters

    call_kwargs = {}

    # different repos sometimes use different names/order
    # we pass by keyword only if present
    if "payoff_fn" in params:
        call_kwargs["payoff_fn"] = payoff_fn
    if "paths" in params:
        call_kwargs["paths"] = paths

    # some versions want (paths, payoff_fn) instead
    # handle minimal fallback later

    if "principal" in params:
        call_kwargs["principal"] = principal

    # discounting field name varies
    if "discount_rate" in params:
        call_kwargs["discount_rate"] = discount_rate
    elif "r" in params:
        call_kwargs["r"] = discount_rate

    # now pass payoff kwargs ONLY if estimate_ev expects **kwargs
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        call_kwargs.update(payoff_kwargs)

    # try the call
    try:
        return fn(**call_kwargs)
    except TypeError:
        # fallback to positional styles:
        # (payoff_fn, paths, principal, discount_rate, **payoff_kwargs)
        args = []
        if "payoff_fn" not in call_kwargs:
            args.append(payoff_fn)
        if "paths" not in call_kwargs:
            args.append(paths)

        # try with principal/discount_rate positionally if signature has them
        if "principal" in params:
            args.append(principal)
        if "discount_rate" in params or "r" in params:
            args.append(discount_rate)

        if has_var_kw:
            return fn(*args, **payoff_kwargs)
        return fn(*args)


def _print_summary(summary):
    """
    summary may be:
      - dataclass/object with attributes (n, ev, std, p_loss...)
      - dict
    """
    print("\n=== Phoenix EV Demo ===")

    if isinstance(summary, dict):
        for k, v in summary.items():
            try:
                print(f"{k:12s}: {float(v):.6f}")
            except Exception:
                print(f"{k:12s}: {v}")
        return

    # object/dataclass
    fields = ["n", "ev", "std", "p_loss", "p_gain", "q05", "q50", "q95"]
    for f in fields:
        if hasattr(summary, f):
            val = getattr(summary, f)
            try:
                print(f"{f:12s}: {float(val):.6f}")
            except Exception:
                print(f"{f:12s}: {val}")


def run_demo():
    # ------------------------
    # Market / simulation
    # ------------------------
    S0 = 100.0
    mu = 0.00
    sigma = 0.20
    r = 0.00

    steps = 12
    n_paths = 10_000
    dt = 1 / 12

    # ------------------------
    # Product parameters (Phoenix)
    # ------------------------
    principal = 100.0
    coupon_rate = 0.02
    coupon_barrier = 0.70
    ki_barrier = 0.60
    autocall_barriers = [1.00 - 0.02 * i for i in range(steps)]

    # ------------------------
    # Simulate paths
    # ------------------------
    paths_np = _call_simulate_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        dt=dt,
        steps=steps,
        n_paths=n_paths,
        seed=42,
        r=r,
    )
    paths = paths_np.tolist() if hasattr(paths_np, "tolist") else paths_np

    # ------------------------
    # Payoff wrapper
    # IMPORTANT: inject principal here, NOT via payoff_kwargs
    # ------------------------
    def payoff_wrapped(path, **kwargs):
        return phoenix_payoff(
            path,
            principal=principal,
            **kwargs,
        )

    # payoff kwargs that are safe to forward into phoenix_payoff
    payoff_kwargs = dict(
        S0=S0,
        coupon_rate=coupon_rate,
        coupon_barrier=coupon_barrier,
        autocall_barriers=autocall_barriers,
        ki_barrier=ki_barrier,
        memory=True,
    )

    # ------------------------
    # EV estimation (auto-adapt)
    # ------------------------
    summary, dist = _call_estimate_ev(
        payoff_fn=payoff_wrapped,
        paths=paths,
        principal=principal,
        discount_rate=r,
        payoff_kwargs=payoff_kwargs,
    )

    _print_summary(summary)


if __name__ == "__main__":
    run_demo()

