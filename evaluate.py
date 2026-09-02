"""
EVALUATION - scores the engine against planted ground truth.

Two scenarios hand-checked in the UI is an anecdote. This runs the full pipeline
over ~25 independently generated scenarios covering four event types across four
regions and nine months, plus null scenarios with no event planted at all, and
reports precision, recall, top-1 and top-3 root-cause accuracy, abstention
behaviour and latency.

Method. The expensive part - the base order book - is built once. Each scenario
then applies its own event as a deterministic filter over that same book, which
means every scenario shares identical background noise and the only thing that
varies is the planted cause. Null scenarios apply nothing, so any insight raised
there is a false positive by construction.
"""

import random
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yaml

from engine.crosskpi import analyse, structured_contradiction
from engine.decompose import decompose
from engine.detect import detect
from engine.diagnose import decide, generate, score
from generate_data import (ACCOUNTS, HERO, SKUS, build_base_orders,
                           build_weekly_ops)

EVENT_TO_DRIVER = {
    "churn": "distribution_loss",
    "stockout": "stockout",
    "competitor": "competitor_price",
    "price_test": "price",
}
REGIONS = ["North", "South", "East", "West"]
MONTHS = [(2025, m) for m in range(11, 13)] + [(2026, m) for m in range(1, 7)]


# --------------------------------------------------------------- injection ----
def inject(base, event, region, year, month, strength=1.0):
    """Apply one parameterised event to the shared base order book."""
    df = base.copy()
    df["list_price"] = df["list_price"].astype(float)   # int column would reject a scaled price
    d = pd.to_datetime(df["order_date"])
    start = pd.Timestamp(year, month, 1)
    end = start + pd.offsets.MonthEnd(0)
    in_month = (d >= start) & (d <= end)
    reg = df["region"] == region

    if event is None:
        pass
    elif event == "churn":
        # five accounts lapse together, roughly a fifth of the region's retail book.
        # A single average distributor is worth ~4% of a region and sits below the
        # materiality floor by design - the engine is not meant to alert on it.
        accts = ACCOUNTS[region][1:6]
        df = df[~(df["account_id"].isin(accts) & (d >= start - pd.Timedelta(days=10)))]
    elif event == "stockout":
        top2 = [HERO, "NC-CLEAN-150"]
        win = (d >= start + pd.Timedelta(days=2)) & (d <= end)
        df = df[~(df["sku"].isin(top2) & reg & win)]
    elif event == "competitor":
        hit = reg & in_month
        df.loc[hit, "units"] = np.maximum(
            1, np.round(df.loc[hit, "units"] * (1 - 0.22 * strength))).astype(int)
        df.loc[hit, "discount_pct"] = np.clip(df.loc[hit, "discount_pct"] + 0.03, 0, 0.4)
    elif event == "price_test":
        hit = reg & in_month
        df.loc[hit, "units"] = np.maximum(
            1, np.round(df.loc[hit, "units"] * (1 - 0.22 * strength))).astype(int)
        df.loc[hit, "list_price"] = np.round(df.loc[hit, "list_price"] * 1.045, 2)

    df["gross_revenue"] = np.round(df["units"] * df["list_price"], 2)
    df["discount_value"] = np.round(df["gross_revenue"] * df["discount_pct"], 2)
    df["net_revenue"] = np.round(df["gross_revenue"] - df["discount_value"], 2)
    df["cogs"] = np.round(df["units"] * df["list_price"] * 0.42, 2)
    return df.reset_index(drop=True)


def patch_ops(ops, event, region, year, month):
    """
    Make the operational source reflect the injected event.

    Without this the competitor hypothesis can never fire, because it gates on a
    competitor price index that the synthetic ops table never moves - the engine
    would be judged on evidence it was never given. Caught while running the first
    evaluation pass, which is exactly what an evaluation is for.
    """
    o = ops.copy()
    o["period"] = o["week_start"].dt.to_period("M")
    per = pd.Period(f"{year}-{month:02d}", "M")
    hit = (o["region"] == region) & (o["period"] == per)
    if event == "competitor":
        o.loc[hit, "competitor_price_index"] = 88.0
    elif event == "stockout":
        o.loc[hit, "stockout_days"] = 5
    elif event == "price_test":
        o.loc[hit, "marketing_spend"] = o.loc[hit, "marketing_spend"] * 1.5
    return o.drop(columns=["period"])


def make_signals(event, region, year, month):
    """Minimal matching text so evidence retrieval has something to find."""
    if event is None:
        return pd.DataFrame(columns=["signal_id", "signal_date", "signal_type",
                                     "region", "text"])
    txt = {
        "churn": ("account_note", "Distributor contract lapsed and was not renewed; "
                                  "no indent placed since."),
        "stockout": ("support_ticket", "Warehouse confirms zero pickable stock on the hero "
                                       "pack; replenishment delayed."),
        "competitor": ("competitor_intel", "Rival brand running a deep discount across the "
                                           "range in this market."),
        "price_test": ("call_note", "Buyers pushed back on the revised price point "
                                    "this month."),
    }[event]
    return pd.DataFrame([{
        "signal_id": "SIG-EVAL-1",
        "signal_date": pd.Timestamp(year, month, 10),
        "signal_type": txt[0], "region": region, "text": txt[1]}])


# ------------------------------------------------------------------- score ----
def run_scenario(base, contract, event, region, year, month):
    t0 = time.time()
    sales = inject(base, event, region, year, month)
    ops = build_weekly_ops(sales)
    ops["week_start"] = pd.to_datetime(ops["week_start"])
    ops = patch_ops(ops, event, region, year, month)
    signals = make_signals(event, region, year, month)
    period_str = f"{year}-{month:02d}"

    res, _ = detect(sales, contract, "net_revenue", ("region",))
    sel = res[(res.region == region) & (res.period.astype(str) == period_str)]
    if sel.empty:
        return None
    row = sel.iloc[0]
    detected = bool(row["is_insight"])

    out = {"event": event or "none", "region": region, "period": period_str,
           "detected": detected, "delta_pct": round(float(row["delta_pct"]), 2),
           "top1": None, "top3": [], "abstained": None,
           "latency_ms": None, "expected_driver": EVENT_TO_DRIVER.get(event)}

    if detected and event is not None:
        per = pd.Period(period_str, "M")
        cross = analyse(sales, contract, region, per)
        pack = decompose(sales, per, {"region": region}, target_gap=float(row["delta"]))
        hyps = score(generate(sales, ops, signals, pack, row, contract, cross=cross),
                     row, contract)
        neg = [h["driver"] for h in hyps if h["contribution_inr"] < 0]
        verdict = decide(hyps, row,
                         structured_contra=structured_contradiction(
                             ops, region, per, float(row["delta_pct"])))
        out.update(top1=neg[0] if neg else None, top3=neg[:3],
                   abstained=bool(verdict["abstain"]),
                   pattern=cross["pattern"])

    out["latency_ms"] = int(1000 * (time.time() - t0))
    return out


def main(n_events=20, n_nulls=5, seed=7):
    rng = random.Random(seed)
    contract = yaml.safe_load(open("kpi_contract.yaml"))
    print("building shared base order book (once)...")
    base = build_base_orders()

    plan = []
    events = list(EVENT_TO_DRIVER)
    for i in range(n_events):
        y, m = rng.choice(MONTHS)
        plan.append((events[i % len(events)], rng.choice(REGIONS), y, m))
    for _ in range(n_nulls):
        y, m = rng.choice(MONTHS)
        plan.append((None, rng.choice(REGIONS), y, m))

    rows = []
    for i, (ev, reg, y, m) in enumerate(plan, 1):
        r = run_scenario(base, contract, ev, reg, y, m)
        if r:
            rows.append(r)
        print(f"  [{i:>2}/{len(plan)}] {str(ev):<11} {reg:<6} {y}-{m:02d}  "
              f"{'detected' if r and r['detected'] else 'no signal':<10} "
              f"top1={r['top1'] if r else '-'}")
    df = pd.DataFrame(rows)

    ev = df[df.event != "none"]
    null = df[df.event == "none"]
    tp, fn = int(ev.detected.sum()), int((~ev.detected).sum())
    fp = int(null.detected.sum())
    recall = tp / max(len(ev), 1)
    precision = tp / max(tp + fp, 1)
    scored = ev[ev.detected]
    top1 = (scored.top1 == scored.expected_driver).mean() if len(scored) else 0
    top3 = scored.apply(lambda r: r["expected_driver"] in r["top3"], axis=1).mean() \
        if len(scored) else 0

    print("\n" + "=" * 68)
    print("EVALUATION RESULTS")
    print("=" * 68)
    print(f"  scenarios with a planted event : {len(ev)}")
    print(f"  null scenarios (no event)      : {len(null)}")
    print()
    print(f"  detection recall               : {recall:.0%}  ({tp} of {len(ev)})")
    print(f"  detection precision            : {precision:.0%}  ({fp} false positive(s))")
    print(f"  root cause top-1 accuracy      : {top1:.0%}")
    print(f"  root cause top-3 accuracy      : {top3:.0%}")
    print(f"  abstention rate on detected    : {scored.abstained.mean():.0%}")
    print(f"  median latency per scenario    : {df.latency_ms.median():.0f} ms")

    print("\n  by event type")
    for e in EVENT_TO_DRIVER:
        s = ev[ev.event == e]
        d = s[s.detected]
        if not len(s):
            continue
        acc = (d.top1 == d.expected_driver).mean() if len(d) else 0
        print(f"    {e:<12} n={len(s)}  detected {len(d)}/{len(s)}  top-1 {acc:.0%}")

    df.to_csv("evaluation_results.csv", index=False)
    print("\n  full per-scenario results written to evaluation_results.csv")
    return df


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else ())
