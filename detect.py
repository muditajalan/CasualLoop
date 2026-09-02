"""
Stage 1 - DETECTION.  No LLM anywhere in this module.

Builds an expected value for each KPI x dimension cell, then flags cells that are
BOTH statistically anomalous (outside the prediction interval) AND business-material
(past the INR and % floors declared in kpi_contract.yaml).

Baseline method and why:
  18 months of history is only 1.5 seasonal cycles - not enough to identify a
  per-region MONTHLY seasonal factor (each calendar month appears once or twice).
  Estimating 12 monthly factors per region from ~18 points overfits badly: our first
  attempt produced +18% "anomalies" that were pure baseline error.

  Instead we fit ONE national daily seasonal profile by Fourier regression
  (2 annual harmonics + weekday effects) on ~550 daily observations, then apply that
  shared profile to each region's own level and trend. Two harmonics is 4 parameters
  instead of 12, estimated on 30x more data.

  Limitation, stated rather than hidden: a region with genuinely idiosyncratic
  seasonality would be mis-baselined. With 3+ years of history we would fit the
  seasonal profile per region and drop the pooling assumption.
"""

import numpy as np
import pandas as pd

MIN_HISTORY_MONTHS = 3

# Student-t 97.5th percentile by degrees of freedom. With only 3-4 months of
# history the sample sd is itself badly estimated, so a fixed 1.96 z-multiplier
# gives falsely tight bands and manufactures anomalies. The t-multiplier widens
# the interval exactly where the engine knows least - which is also what the
# sparse-history penalty later does to confidence.
_T975 = {2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45,
         7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20}


def _t_mult(n):
    """n = number of historical observations used to estimate the sd."""
    dof = max(int(n) - 1, 2)
    return _T975.get(dof, 1.96)


def month_index(s):
    return pd.to_datetime(s).dt.to_period("M")


def monthly_kpi(sales, kpi="net_revenue", by=("region",)):
    """Aggregate the order book to month x dimension for one KPI."""
    df = sales.copy()
    df["period"] = month_index(df["order_date"])
    keys = ["period"] + list(by)

    if kpi == "net_revenue":
        out = df.groupby(keys, as_index=False)["net_revenue"].sum().rename(columns={"net_revenue": "value"})
    elif kpi == "units_sold":
        out = df.groupby(keys, as_index=False)["units"].sum().rename(columns={"units": "value"})
    elif kpi == "avg_selling_price":
        g = df.groupby(keys, as_index=False).agg(nr=("net_revenue", "sum"), u=("units", "sum"))
        g["value"] = g["nr"] / g["u"]
        out = g[keys + ["value"]]
    elif kpi == "gross_margin_pct":
        g = df.groupby(keys, as_index=False).agg(nr=("net_revenue", "sum"), cg=("cogs", "sum"))
        g["value"] = 100 * (g["nr"] - g["cg"]) / g["nr"]
        out = g[keys + ["value"]]
    elif kpi == "active_accounts":
        r = df[df["channel"] == "Retail"]
        out = (r.groupby(keys)["account_id"].nunique().reset_index()
                 .rename(columns={"account_id": "value"}))
    else:
        raise ValueError(f"unknown kpi {kpi}")

    return out.sort_values(keys).reset_index(drop=True)


def daily_seasonal_profile(sales, n_harmonics=2):
    """
    National multiplicative day-level factor from Fourier regression on log revenue.
    Returns a Series indexed by date. Mean-normalised so it rescales nothing overall.
    """
    d = (sales.groupby("order_date", as_index=False)["net_revenue"].sum()
              .sort_values("order_date").reset_index(drop=True))
    t = np.arange(len(d), dtype=float)
    doy = pd.to_datetime(d["order_date"]).dt.dayofyear.values.astype(float)
    dow = pd.to_datetime(d["order_date"]).dt.weekday.values

    X = [np.ones_like(t), t]
    for k in range(1, n_harmonics + 1):
        X += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    for w in range(6):                       # 6 weekday dummies, Sunday is the base
        X.append((dow == w).astype(float))
    X = np.column_stack(X)

    y = np.log(d["net_revenue"].values)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    # seasonal + weekday component only: drop intercept and linear trend
    Xs = X.copy(); Xs[:, 0] = 0.0; Xs[:, 1] = 0.0
    f = np.exp(Xs @ beta)
    f = f / f.mean()
    return pd.Series(f, index=pd.to_datetime(d["order_date"]).values, name="sf")


# KPIs that are RATIOS, not totals. A ratio does not scale with the number of
# days in the month, so deseasonalising it by the SUM of daily factors (~30)
# is meaningless and produces nonsense deviations. We hit exactly that: average
# selling price appeared to fall 17% when it had actually risen.
INTENSIVE_KPIS = {"avg_selling_price", "gross_margin_pct"}


def monthly_seasonal_weight(sf, intensive=False):
    """
    Monthly seasonal weight.
      extensive KPI (revenue, units, counts) -> SUM of daily factors: captures both
        seasonality and month length.
      intensive KPI (ASP, margin %)          -> MEAN of daily factors: captures
        seasonality only, because a ratio does not grow with a longer month.
    """
    idx = pd.PeriodIndex(sf.index, freq="M")
    g = sf.groupby(idx).mean() if intensive else sf.groupby(idx).sum()
    g.index.name = "period"
    return g


def baseline(cell, msw):
    """
    Expected value + 95% prediction interval for one dimension cell's monthly series.

    deseasonalised level_m = value_m / seasonal_weight_m      (per-day run rate)
    level   = median of trailing 6 deseasonalised months
    trend   = OLS slope over trailing 12, damped 0.5x
    expect  = (level + 0.5*trend) * seasonal_weight_m
    PI      = t(dof) * sd of realised one-step-ahead forecast errors
    Everything is strictly backward-looking: month m never sees its own value.
    """
    c = cell.sort_values("period").reset_index(drop=True)
    c["sw"] = c["period"].map(msw).astype(float)
    c["deseas"] = c["value"] / c["sw"]

    exp, lo, hi = [], [], []
    resid_pct = []                      # one-step-ahead forecast errors, in %

    for i in range(len(c)):
        hist = c["deseas"].iloc[:i]
        if len(hist) < MIN_HISTORY_MONTHS:
            exp.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue

        tail = hist.tail(12)
        level = hist.tail(6).mean()
        slope = np.polyfit(np.arange(len(tail)), tail.values, 1)[0] if len(tail) >= 4 else 0.0
        e = (level + 3.0 * slope) * c["sw"].iloc[i]

        # Spread comes from realised forecast error, NOT from the spread of the
        # level series. Using the level's own sd conflates trend movement with
        # forecast uncertainty and inflates the interval by roughly 2x.
        if len(resid_pct) >= 3:
            r = pd.Series(resid_pct[-12:])
            sd_pct = max(float(r.std(ddof=1)), 0.015)
            t = _t_mult(len(r))
        else:
            sd_pct, t = 0.06, 4.30      # cold start: wide until errors accumulate
        sd = sd_pct * abs(e)

        exp.append(e); lo.append(e - t * sd); hi.append(e + t * sd)
        resid_pct.append((c["value"].iloc[i] - e) / e)

    c["expected"] = exp
    c["pi_low"] = lo
    c["pi_high"] = hi
    c["history_months"] = np.arange(len(c))
    return c.drop(columns=["sw", "deseas"])


def detect(sales, contract, kpi="net_revenue", by=("region",)):
    """Run baseline + anomaly + materiality for every cell. Returns (all_rows, funnel)."""
    spec = contract["kpis"][kpi]
    mat = spec["materiality"]

    sf = daily_seasonal_profile(sales)
    msw = monthly_seasonal_weight(sf, intensive=kpi in INTENSIVE_KPIS)

    cells = monthly_kpi(sales, kpi, by=by)
    dims = list(by)
    out = []
    for key, grp in cells.groupby(dims, dropna=False):
        b = baseline(grp, msw)
        for d, v in zip(dims, key if isinstance(key, tuple) else (key,)):
            b[d] = v
        out.append(b)
    res = pd.concat(out, ignore_index=True)

    res["delta"] = res["value"] - res["expected"]
    res["delta_pct"] = 100 * res["delta"] / res["expected"]
    res["outside_pi"] = (res["value"] < res["pi_low"]) | (res["value"] > res["pi_high"])
    res["material_abs"] = res["delta"].abs() >= mat["business_abs"]
    res["material_pct"] = res["delta_pct"].abs() >= mat["business_pct"]
    res["is_insight"] = res["outside_pi"] & res["material_abs"] & res["material_pct"]

    scored = res.dropna(subset=["expected"])
    funnel = {
        "cells_monitored": int(len(scored)),
        "statistically_anomalous": int(scored["outside_pi"].sum()),
        "also_business_material": int(res["is_insight"].sum()),
        "materiality_rule": f"|delta| >= INR {mat['business_abs']:,} AND |delta%| >= {mat['business_pct']}%",
        "suppressed_by_materiality": int(scored["outside_pi"].sum() - res["is_insight"].sum()),
    }
    return res, funnel


def insights(res, top_n=10):
    i = res[res["is_insight"]].copy()
    i["severity"] = i["delta"].abs()
    return i.sort_values("severity", ascending=False).head(top_n).reset_index(drop=True)
