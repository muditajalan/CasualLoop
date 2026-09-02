"""
NovaCare synthetic dataset generator for the CausalLoop prototype.

Design rule: all randomness is drawn ONCE for a base order book that is
independent of any planted event. Events are then applied as deterministic
row-wise filters/multipliers. This means we can toggle events off and re-derive
the exact counterfactual, which gives us TRUE contribution ground truth
(no hand-waving) to score the engine against.
"""

import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 20260326
START = date(2025, 1, 1)
END = date(2026, 6, 30)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

REGIONS = {"North": 1.00, "South": 0.85, "East": 0.62, "West": 1.15}
CHANNELS = {"Retail": 0.68, "Ecom": 0.32}

# sku -> (list_price, demand_weight, launch_date)
SKUS = {
    "NC-HYDRA-500": (349, 1.00, START),   # hero SKU
    "NC-HYDRA-200": (179, 0.72, START),
    "NC-DERM-100":  (599, 0.48, START),
    "NC-DERM-250":  (899, 0.31, START),
    "NC-SUN-50":    (449, 0.55, START),
    "NC-SUN-100":   (699, 0.27, START),
    "NC-CLEAN-150": (249, 0.63, START),
    "NC-CLEAN-300": (399, 0.40, START),
    "NC-REPAIR-30": (1199, 0.22, START),
    "NC-BOOST-01":  (799, 0.35, date(2026, 2, 1)),   # sparse-history launch
}
HERO = "NC-HYDRA-500"
NEW_SKU = "NC-BOOST-01"

ACCOUNTS = {r: [f"{r[:2].upper()}-DIST-{i:03d}" for i in range(1, 25)] for r in REGIONS}
CHURN_ACCOUNT = "WE-DIST-002"          # large West distributor
CHURN_DATE = date(2026, 2, 20)
ACCOUNT_WEIGHT = {CHURN_ACCOUNT: 2.2}   # a top-3 distributor in the region

COMPETITOR_WINDOW = (date(2026, 3, 1), date(2026, 3, 31))
STOCKOUT_WINDOW = (date(2026, 3, 5), date(2026, 3, 20))
PRICE_TEST_WINDOW = (date(2026, 5, 1), date(2026, 5, 31))

ALL_EVENTS = ("competitor", "churn", "stockout", "price_test", "launch_dip")


def _daterange():
    d, out = START, []
    while d <= END:
        out.append(d)
        d += timedelta(days=1)
    return out


def build_base_orders():
    """Order book before any event is applied. Depends only on SEED."""
    rng = np.random.default_rng(SEED)
    days = _daterange()
    rows = []

    region_names = list(REGIONS)
    channel_names = list(CHANNELS)
    sku_names = list(SKUS)

    for i, d in enumerate(days):
        # baseline shape: gentle growth + annual seasonality + weekday effect
        trend = 1.0 + 0.00035 * i
        seasonal = 1.0 + 0.11 * np.sin(2 * np.pi * (d.timetuple().tm_yday - 20) / 365.0)
        weekday = [1.06, 1.04, 1.02, 1.03, 1.10, 0.86, 0.72][d.weekday()]
        n_orders = rng.poisson(250 * trend * seasonal * weekday)

        for _ in range(int(n_orders)):
            region = rng.choice(region_names, p=_norm(list(REGIONS.values())))
            channel = rng.choice(channel_names, p=_norm(list(CHANNELS.values())))
            sku = rng.choice(sku_names, p=_norm([SKUS[s][1] for s in sku_names]))
            if d < SKUS[sku][2]:
                continue  # not launched yet
            if channel == "Retail":
                accts = ACCOUNTS[region]
                w = _norm([ACCOUNT_WEIGHT.get(a, 1.0) for a in accts])
                account = rng.choice(accts, p=w)
                units = max(1, int(rng.gamma(8.0, 1.5)))
            else:
                account = "ECOM-DIRECT"
                units = max(1, int(rng.gamma(4.0, 0.42)))
            list_price = SKUS[sku][0]
            disc_pct = float(np.clip(rng.normal(0.09 if channel == "Retail" else 0.05, 0.035), 0, 0.35))
            rows.append((d, region, channel, sku, account, units, list_price, disc_pct))

    return pd.DataFrame(rows, columns=[
        "order_date", "region", "channel", "sku", "account_id",
        "units", "list_price", "discount_pct",
    ])


def _norm(w):
    a = np.asarray(w, dtype=float)
    return a / a.sum()


def apply_events(base: pd.DataFrame, events=ALL_EVENTS) -> pd.DataFrame:
    """Deterministically apply planted events. Same input -> same output."""
    df = base.copy()
    d = df["order_date"]

    if "churn" in events:
        df = df[~((df["account_id"] == CHURN_ACCOUNT) & (d >= CHURN_DATE))]
        d = df["order_date"]

    if "stockout" in events:
        df = df[~(
            (df["sku"] == HERO)
            & (df["region"] == "West")
            & d.between(*STOCKOUT_WINDOW)
        )]
        d = df["order_date"]

    mult = pd.Series(1.0, index=df.index)
    price_mult = pd.Series(1.0, index=df.index)

    if "competitor" in events:
        # competitor promo in West: fewer units AND deeper defensive discounting
        hit = (df["region"] == "West") & d.between(*COMPETITOR_WINDOW)
        mult[hit] *= 0.955
        df.loc[hit, "discount_pct"] = np.clip(df.loc[hit, "discount_pct"] + 0.022, 0, 0.40)

    if "price_test" in events:
        # South price experiment: ASP up ~6%, demand down ~5% (near-offsetting -> ambiguous)
        hit = (df["region"] == "South") & d.between(*PRICE_TEST_WINDOW)
        mult[hit] *= 0.78
        price_mult[hit] *= 1.045

    if "launch_dip" in events:
        # new SKU loses shelf momentum in June after a strong launch fortnight
        hit = (df["sku"] == NEW_SKU) & (d >= date(2026, 6, 8))
        mult[hit] *= 0.52

    df["units"] = np.maximum(1, np.round(df["units"] * mult)).astype(int)
    df["list_price"] = np.round(df["list_price"] * price_mult, 2)

    df["gross_revenue"] = np.round(df["units"] * df["list_price"], 2)
    df["discount_value"] = np.round(df["gross_revenue"] * df["discount_pct"], 2)
    df["net_revenue"] = np.round(df["gross_revenue"] - df["discount_value"], 2)
    df["cogs"] = np.round(df["units"] * df["list_price"] * 0.42, 2)
    df["order_id"] = [f"SO-{i:07d}" for i in range(1, len(df) + 1)]

    cols = ["order_id", "order_date", "region", "channel", "sku", "account_id",
            "units", "list_price", "discount_pct", "gross_revenue",
            "discount_value", "net_revenue", "cogs"]
    return df[cols].reset_index(drop=True)


def build_weekly_ops(sales: pd.DataFrame) -> pd.DataFrame:
    """Source B: weekly grain, weekly refresh. Marketing spend, supply, competitor index."""
    rng = np.random.default_rng(SEED + 1)
    s = sales.copy()
    s["week_start"] = pd.to_datetime(s["order_date"]).dt.to_period("W-SUN").dt.start_time
    g = s.groupby(["week_start", "region", "channel"], as_index=False)["net_revenue"].sum()

    # Media budgets are PLANNED from trailing performance, not set from the week's
    # own revenue. Deriving spend from current revenue would make spend fall
    # automatically whenever revenue falls, which hides the very situation we want
    # the engine to notice: spend rising while revenue drops.
    g = g.sort_values("week_start")
    g["plan_base"] = (g.groupby(["region", "channel"])["net_revenue"]
                        .transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean()))
    g["plan_base"] = g["plan_base"].fillna(g["net_revenue"])

    rows = []
    for _, r in g.iterrows():
        wk = r["week_start"].date()
        spend = r["plan_base"] * float(np.clip(rng.normal(0.085, 0.02), 0.03, 0.2))
        comp_idx = float(np.clip(rng.normal(100, 1.6), 94, 106))
        stockout_days = 0
        campaign = "none"

        if r["region"] == "West" and COMPETITOR_WINDOW[0] <= wk <= COMPETITOR_WINDOW[1]:
            comp_idx = float(np.clip(rng.normal(88, 1.2), 84, 92))  # rival cut price ~12%
        if r["region"] == "West" and STOCKOUT_WINDOW[0] <= wk + timedelta(days=6) and wk <= STOCKOUT_WINDOW[1]:
            stockout_days = int(rng.integers(4, 7))
        if r["region"] == "South" and PRICE_TEST_WINDOW[0] <= wk <= PRICE_TEST_WINDOW[1]:
            spend *= 1.35  # spend went UP while revenue fell -> contradictory evidence
            campaign = "SOUTH_MONSOON_PUSH"

        rows.append({
            "week_start": wk,
            "region": r["region"],
            "channel": r["channel"],
            "marketing_spend": round(spend, 2),
            "campaign": campaign,
            "stockout_days": stockout_days,
            "competitor_price_index": round(comp_idx, 2),
            "otd_pct": round(float(np.clip(rng.normal(94.5, 2.2), 78, 99.5)), 2),
        })
    return pd.DataFrame(rows)


def build_signals() -> pd.DataFrame:
    """Source C: unstructured, irregular refresh. Text corroborates or muddies hypotheses."""
    rng = np.random.default_rng(SEED + 2)
    rows = []

    def add(d, kind, region, text):
        rows.append({"signal_id": f"SIG-{len(rows)+1:04d}", "signal_date": d,
                     "signal_type": kind, "region": region, "text": text})

    comp = [
        "Rival brand running a flat 15% off across their hydration range in metro stores this month.",
        "Two key retailers in Pune confirmed the competitor has funded end-cap displays through March.",
        "Customer asked why our 500ml is priced above the competing pack now on promotion.",
        "Lost a repeat order today, buyer explicitly cited the competing brand's March discount.",
    ]
    stock = [
        "Warehouse confirms zero pickable stock on the 500ml hero pack, replenishment ETA 6 days.",
        "Three retail partners raised tickets about unfulfilled 500ml line items this week.",
        "Order desk holding back-orders on the hero SKU until the Bhiwandi DC restocks.",
    ]
    churn = [
        "Distributor WE-DIST-002 has not placed an indent since mid-February, no response to follow-ups.",
        "Account review flagged WE-DIST-002 moving volume to a competing portfolio after margin dispute.",
        "Credit team notes WE-DIST-002 contract lapsed on 20 Feb and was not renewed.",
    ]
    south_pro = [
        "Monsoon push creatives went live across South, reach tracking ahead of plan.",
        "Field team reports the campaign is landing well, footfall in South stores looks normal.",
    ]
    south_con = [
        "A few South buyers pushed back on the revised price point this month.",
        "Trade partner in Kochi asked whether the new MRP is permanent before committing volume.",
    ]
    noise = [
        "Routine quarterly business review scheduled with the East distributor panel.",
        "Packaging vendor confirmed the label change rollout for the next production batch.",
        "NPS verbatim: delivery was quick, product as described.",
        "Ticket closed: invoice mismatch resolved with the finance team.",
        "Sales rep note: general market sentiment stable, no unusual activity observed.",
        "Support: customer requested a bulk-order quotation for the cleanser range.",
    ]

    for t in comp:
        add(_rand_day(rng, date(2026, 2, 26), date(2026, 3, 28)), "competitor_intel", "West", t)
    for t in stock:
        add(_rand_day(rng, date(2026, 3, 7), date(2026, 3, 18)), "support_ticket", "West", t)
    for t in churn:
        add(_rand_day(rng, date(2026, 2, 18), date(2026, 3, 20)), "account_note", "West", t)
    for t in south_pro:
        add(_rand_day(rng, date(2026, 5, 2), date(2026, 5, 28)), "campaign_note", "South", t)
    for t in south_con:
        add(_rand_day(rng, date(2026, 5, 4), date(2026, 5, 30)), "call_note", "South", t)
    add(_rand_day(rng, date(2026, 5, 12), date(2026, 5, 20)), "launch_note", "West",
        "New booster SKU launched with limited listing, only two regions have shelf presence so far.")

    for _ in range(180):
        add(_rand_day(rng, START, END), "misc_note",
            str(rng.choice(list(REGIONS))), str(rng.choice(noise)))

    return pd.DataFrame(rows).sort_values("signal_date").reset_index(drop=True)


def _rand_day(rng, a, b):
    return a + timedelta(days=int(rng.integers(0, (b - a).days + 1)))


def month_revenue(df, y, m, region=None, sku=None):
    d = pd.to_datetime(df["order_date"])
    sel = (d.dt.year == y) & (d.dt.month == m)
    if region:
        sel &= df["region"] == region
    if sku:
        sel &= df["sku"] == sku
    return float(df.loc[sel, "net_revenue"].sum())


def build_ground_truth(base):
    """True contribution of each event = revenue(all events) - revenue(all but that event)."""
    full = apply_events(base, ALL_EVENTS)
    truth = {}

    cases = {
        "WEST_MAR_2026": {"scope": {"region": "West"}, "year": 2026, "month": 3,
                          "label": "Multi-factor movement (3 real drivers)",
                          "events": ["competitor", "churn", "stockout"]},
        "SOUTH_MAY_2026": {"scope": {"region": "South"}, "year": 2026, "month": 5,
                           "label": "Ambiguous / contradictory evidence -> expect ABSTAIN",
                           "events": ["price_test"]},
        "NEWSKU_JUN_2026": {"scope": {"sku": NEW_SKU}, "year": 2026, "month": 6,
                            "label": "Sparse history (SKU launched 2026-05-15)",
                            "events": ["launch_dip"]},
    }

    baseline_df = apply_events(base, ())
    for name, c in cases.items():
        y, m, scope = c["year"], c["month"], c["scope"]
        actual = month_revenue(full, y, m, **scope)
        no_events = month_revenue(baseline_df, y, m, **scope)
        contribs = {}
        for ev in c["events"]:
            without = apply_events(base, tuple(e for e in ALL_EVENTS if e != ev))
            contribs[ev] = round(actual - month_revenue(without, y, m, **scope), 2)
        total_delta = round(actual - no_events, 2)
        truth[name] = {
            "label": c["label"], "scope": scope, "period": f"{y}-{m:02d}",
            "actual_net_revenue": round(actual, 2),
            "counterfactual_no_events": round(no_events, 2),
            "total_event_delta": total_delta,
            "delta_pct_vs_counterfactual": round(100 * total_delta / no_events, 2),
            "driver_contributions_inr": contribs,
            "driver_share_pct": {k: round(100 * v / total_delta, 1) for k, v in contribs.items()} if total_delta else {},
            "unexplained_interaction_pct": round(100 * (1 - sum(contribs.values()) / total_delta), 1) if total_delta else 0.0,
        }
    return full, truth


def main():
    os.makedirs(OUT, exist_ok=True)
    base = build_base_orders()
    sales, truth = build_ground_truth(base)
    ops = build_weekly_ops(sales)
    signals = build_signals()

    sales.to_csv(f"{OUT}/sales_orders.csv", index=False)
    ops.to_csv(f"{OUT}/weekly_ops.csv", index=False)
    signals.to_csv(f"{OUT}/signals.csv", index=False)

    manifest = {
        "sales_orders.csv": {"source_system": "OrderDB (Postgres)", "grain": "order line",
                             "refresh": "daily 02:00 IST", "last_refresh": "2026-06-30T02:00:00+05:30",
                             "rows": len(sales)},
        "weekly_ops.csv": {"source_system": "MarketingHub + WMS export", "grain": "week x region x channel",
                           "refresh": "weekly Monday 09:00 IST", "last_refresh": "2026-06-29T09:00:00+05:30",
                           "rows": len(ops)},
        "signals.csv": {"source_system": "CRM notes / Zendesk / market intel", "grain": "event",
                        "refresh": "irregular, event-driven", "last_refresh": "2026-06-27T17:40:00+05:30",
                        "rows": len(signals)},
    }
    json.dump(manifest, open(f"{OUT}/source_manifest.json", "w"), indent=2)
    json.dump(truth, open(f"{OUT}/ground_truth.json", "w"), indent=2, default=str)

    print(f"sales_orders.csv  {len(sales):>7,} rows  "
          f"{sales.order_date.min()} -> {sales.order_date.max()}")
    print(f"weekly_ops.csv    {len(ops):>7,} rows")
    print(f"signals.csv       {len(signals):>7,} rows")
    print(f"total net revenue  INR {sales.net_revenue.sum()/1e7:,.1f} Cr\n")
    print(json.dumps(truth, indent=2, default=str))


if __name__ == "__main__":
    main()
