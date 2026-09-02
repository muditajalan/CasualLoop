"""
Stage 2 - DECOMPOSITION.  No LLM anywhere in this module.

Takes one detected movement (e.g. West, 2026-03, net_revenue -19.8%) and splits it
two ways:

  (a) a PRICE / VOLUME / MIX bridge  - what kind of movement is it
  (b) a DIMENSION gap table          - which SKUs and which accounts moved

Both are pure arithmetic against the same backward-looking baseline used in
detection, so an insight's parts always reconcile to its whole.
"""

import numpy as np
import pandas as pd

from engine.detect import baseline, daily_seasonal_profile, monthly_kpi, monthly_seasonal_weight


def _expected_by(sales, msw, by, period, scope):
    """
    Actual vs baseline-expected for every cell of `by` within one period+scope.

    Critically, the panel is completed with explicit zeros. An entity that
    disappears - a distributor that stops ordering - has no rows at all in the
    target month, so a naive groupby drops it silently and the single largest
    driver of the movement becomes invisible. Zero-filling makes absence
    measurable: value 0 against a positive expectation is the churn signal.
    """
    df = sales
    for k, v in scope.items():
        df = df[df[k] == v]
    if df.empty:
        return pd.DataFrame(columns=list(by) + ["value", "expected", "gap"])

    cells = monthly_kpi(df, "net_revenue", by=by)
    dims_ = list(by)
    periods = sorted(cells["period"].unique())
    first_seen = cells.groupby(dims_, as_index=False)["period"].min().rename(
        columns={"period": "first_period"})
    full = first_seen.merge(pd.DataFrame({"period": periods}), how="cross")
    # only fill from the entity's first appearance onward - back-filling zeros
    # before a SKU launched would invent a history it never had and destroy its
    # baseline (we hit exactly that with the newly launched SKU).
    full = full[full["period"] >= full["first_period"]].drop(columns=["first_period"])
    cells = full.merge(cells, on=["period"] + dims_, how="left")
    cells["value"] = cells["value"].fillna(0.0)
    cells = cells.sort_values(["period"] + dims_).reset_index(drop=True)

    rows = []
    dims = list(by)
    for key, grp in cells.groupby(dims, dropna=False):
        b = baseline(grp, msw)
        r = b[b["period"] == period]
        if r.empty or pd.isna(r["expected"].iloc[0]):
            continue
        rec = dict(zip(dims, key if isinstance(key, tuple) else (key,)))
        rec["value"] = float(r["value"].iloc[0])
        rec["expected"] = float(r["expected"].iloc[0])
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["gap"] = out["value"] - out["expected"]
    out["gap_pct"] = 100 * out["gap"] / out["expected"]
    return out.sort_values("gap").reset_index(drop=True)


def price_volume_mix(sales, period, scope, lookback=3):
    """
    Decompose the revenue gap into price, volume and mix effects.

    Reference basket = the trailing `lookback` months immediately before `period`,
    within the same scope. Standard bridge:

        volume = (U1 - U0) * P0_blended        how much did total units move
        price  = sum_s U1_s * (P1_s - P0_s)    did per-SKU realised price move
        mix    = residual                      did the SKU composition shift

    Realised price is net revenue per unit, so discounting shows up in `price`.
    """
    df = sales.copy()
    for k, v in scope.items():
        df = df[df[k] == v]
    df["period"] = pd.to_datetime(df["order_date"]).dt.to_period("M")

    prior = [period - i for i in range(1, lookback + 1)]
    cur = df[df["period"] == period]
    ref = df[df["period"].isin(prior)]
    if cur.empty or ref.empty:
        return {}

    def by_sku(d, months):
        g = d.groupby("sku", as_index=False).agg(nr=("net_revenue", "sum"), u=("units", "sum"))
        g["u"] /= months
        g["nr"] /= months
        g["p"] = g["nr"] / g["u"]
        return g.set_index("sku")

    c = by_sku(cur, 1)
    r = by_sku(ref, lookback)
    skus = r.index.union(c.index)
    c = c.reindex(skus).fillna(0.0)
    r = r.reindex(skus).fillna(0.0)

    U1, U0 = c["u"].sum(), r["u"].sum()
    P0_blend = r["nr"].sum() / U0 if U0 else 0.0

    volume = (U1 - U0) * P0_blend
    price = float((c["u"] * (c["p"] - r["p"]).fillna(0.0)).sum())
    total = float(c["nr"].sum() - r["nr"].sum())
    mix = total - volume - price

    return {
        "reference_months": [str(p) for p in reversed(prior)],
        "reference_revenue": round(float(r["nr"].sum()), 2),
        "total_change": round(total, 2),
        "volume_effect": round(volume, 2),
        "price_effect": round(price, 2),
        "mix_effect": round(mix, 2),
        "units_current": int(U1),
        "units_reference": int(U0),
        "note": "Reference is the trailing 3-month average, so this bridge answers "
                "'what kind of movement is it', not 'how far from baseline is it'.",
    }


def _reconcile(tbl, target_gap):
    """
    Scale sub-dimension gaps so they sum to the parent movement.

    Per-cell baselines are fitted independently, so their gaps do not naturally
    add up to the parent's gap - each carries its own forecast error. We rescale
    by a single factor and report that factor openly, rather than presenting
    attributions that silently fail to reconcile.
    """
    if tbl.empty:
        return tbl, 1.0
    raw = float(tbl["gap"].sum())
    f = target_gap / raw if abs(raw) > 1e-9 else 1.0
    t = tbl.copy()
    t["gap_raw"] = t["gap"]
    t["gap"] = t["gap"] * f
    t["share_pct"] = 100 * t["gap"] / target_gap if abs(target_gap) > 1e-9 else 0.0
    return t, round(f, 4)


def decompose(sales, period, scope, target_gap=None):
    """Full decomposition package for one detected movement."""
    sf = daily_seasonal_profile(sales)
    msw = monthly_seasonal_weight(sf)

    by_sku = _expected_by(sales, msw, ("sku",), period, scope)
    by_acct = _expected_by(sales, msw, ("account_id",), period, scope)
    by_chan = _expected_by(sales, msw, ("channel",), period, scope)

    if target_gap is None:
        target_gap = float(by_sku["gap"].sum()) if not by_sku.empty else 0.0
    by_sku, f_sku = _reconcile(by_sku, target_gap)
    by_acct, f_acct = _reconcile(by_acct, target_gap)
    by_chan, f_chan = _reconcile(by_chan, target_gap)

    return {
        "period": str(period),
        "scope": scope,
        "total_gap": round(float(target_gap), 2),
        "reconciliation_factors": {"sku": f_sku, "account": f_acct, "channel": f_chan},
        "bridge": price_volume_mix(sales, period, scope),
        "by_sku": by_sku,
        "by_account": by_acct,
        "by_channel": by_chan,
        "worst_skus": by_sku.head(3).to_dict("records") if not by_sku.empty else [],
        "worst_accounts": by_acct.head(3).to_dict("records") if not by_acct.empty else [],
        "vanished_accounts": (by_acct[(by_acct["value"] == 0) & (by_acct["expected"] > 0)]
                              .to_dict("records") if not by_acct.empty else []),
    }
