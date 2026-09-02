"""
Stage 1b - CROSS-KPI REASONING.  No LLM anywhere in this module.

A single KPI cannot tell you what kind of problem you have. Revenue down 15% is
equally consistent with "nobody bought" and "we raised prices and demand fell".
Reading net_revenue, units_sold and avg_selling_price TOGETHER resolves that
ambiguity before any hypothesis is tested.

    units DOWN  + price UP     -> price pushback      (we caused it)
    units DOWN  + price FLAT   -> volume shock        (supply / account / competitor)
    units FLAT  + price DOWN   -> discounting         (margin, not demand)
    units DOWN  + price DOWN   -> broad weakness      (worst case)

This is a small deterministic lookup, not a model. Its job is to tell the
hypothesis stage WHERE to look first, and to make the engine's reasoning legible
to an analyst who wants to check it by eye.
"""

import pandas as pd

from engine.decompose import price_volume_mix
from engine.detect import detect

TRACKED = ("net_revenue", "units_sold", "avg_selling_price")

# thresholds in % deviation from each KPI's own baseline
UNIT_DROP = -3.0
UNIT_FLAT = -3.0
PRICE_UP = 2.0
PRICE_DOWN = -3.0


def read_kpis(sales, contract, region, period, by=("region",)):
    """Deviation from baseline for each tracked KPI, for one region-month."""
    out = {}
    for kpi in TRACKED:
        res, _ = detect(sales, contract, kpi, by)
        r = res[(res["region"] == region) & (res["period"].astype(str) == str(period))]
        if r.empty or pd.isna(r["expected"].iloc[0]):
            out[kpi] = None
            continue
        out[kpi] = {
            "value": float(r["value"].iloc[0]),
            "expected": float(r["expected"].iloc[0]),
            "delta": float(r["delta"].iloc[0]),
            "delta_pct": round(float(r["delta_pct"].iloc[0]), 2),
            "outside_pi": bool(r["outside_pi"].iloc[0]),
        }
    return out


def classify(kpis, bridge):
    """
    Turn the connected KPIs into one named pattern plus prioritised drivers.

    The price signal is the LIKE-FOR-LIKE price effect from the bridge, not the
    raw average selling price. Blended ASP mixes a genuine price change together
    with a shift in what sold, and the mix term is usually the larger of the two -
    South's real +4.2% price rise showed up as -0.8% in blended ASP. The KPI
    contract makes this explicit: never explain an ASP move without first running
    the mix-versus-price split.
    """
    rev = kpis.get("net_revenue")
    uni = kpis.get("units_sold")
    if not (rev and uni) or not bridge:
        return {"pattern": "insufficient_kpis", "interpretation":
                "Not all connected KPIs have a baseline; cross-KPI reasoning skipped.",
                "prioritised_drivers": [], "implied_elasticity": None}

    ref_rev = bridge.get("reference_revenue") or 0.0
    u = uni["delta_pct"]
    p = round(100 * bridge["price_effect"] / ref_rev, 2) if ref_rev else 0.0
    kpis["like_for_like_price"] = {"delta_pct": p,
                                   "note": "mix-adjusted price effect from the bridge"}

    if u <= UNIT_DROP and p >= PRICE_UP:
        pattern = "price_pushback"
        interp = (f"Units fell {abs(u):.1f}% while realised price rose {p:.1f}%. "
                  f"Revenue lost volume faster than price recovered it - the shape of a "
                  f"price change suppressing demand, not a demand shock on its own.")
        drivers = ["price", "discount_depth", "competitor_price"]
    elif u <= UNIT_DROP and PRICE_DOWN < p < PRICE_UP:
        pattern = "volume_shock"
        interp = (f"Units fell {abs(u):.1f}% with realised price broadly unchanged "
                  f"({p:+.1f}%). Something removed demand or supply rather than "
                  f"changing what customers paid.")
        drivers = ["stockout", "distribution_loss", "competitor_price", "marketing_spend"]
    elif u > UNIT_FLAT and p <= PRICE_DOWN:
        pattern = "discounting"
        interp = (f"Units held ({u:+.1f}%) but realised price fell {abs(p):.1f}%. "
                  f"This is a margin problem, not a demand problem.")
        drivers = ["discount_depth", "mix", "price"]
    elif u <= UNIT_DROP and p <= PRICE_DOWN:
        pattern = "broad_weakness"
        interp = (f"Units fell {abs(u):.1f}% AND realised price fell {abs(p):.1f}%. "
                  f"Both levers moved against us at once.")
        drivers = ["competitor_price", "discount_depth", "distribution_loss"]
    else:
        pattern = "mix_shift"
        interp = (f"Units {u:+.1f}% and price {p:+.1f}% are both close to baseline; the "
                  f"revenue movement is most likely composition rather than either lever.")
        drivers = ["mix", "distribution_loss"]

    # Elasticity is only meaningful when price moved AND moved OPPOSITE to units.
    # If both fell together the ratio is positive, which is not an elasticity at
    # all - it is two things going wrong at once. Reporting it would be worse than
    # reporting nothing.
    elasticity = None
    if abs(p) >= 1.0 and (u / p) < 0:
        elasticity = round(u / p, 2)

    return {"pattern": pattern, "interpretation": interp,
            "prioritised_drivers": drivers, "implied_elasticity": elasticity}


def analyse(sales, contract, region, period, by=("region",)):
    kpis = read_kpis(sales, contract, region, period, by)
    bridge = price_volume_mix(sales, period, {"region": region})
    cls = classify(kpis, bridge)
    cls["kpis"] = kpis
    cls["bridge"] = bridge
    return cls


def structured_contradiction(ops, region, period, revenue_delta_pct):
    """
    A contradiction visible in STRUCTURED data, not text: marketing spend rising
    while revenue falls undercuts any demand-side story, because the most obvious
    demand lever was pushed harder, not softer.
    """
    o = ops.copy()
    o["period"] = o["week_start"].dt.to_period("M")
    # Compare weekly AVERAGES, not monthly sums: a 5-week month beats a 4-week
    # month by 25% on volume alone and would fake a spend increase every time.
    cur = o[(o["region"] == region) & (o["period"] == period)]["marketing_spend"].mean()
    prev = o[(o["region"] == region) & (o["period"] == period - 1)]["marketing_spend"].mean()
    if not prev or not cur or prev <= 0 or cur <= 0:
        return None
    chg = 100 * (cur / prev - 1)
    # 25%, not 10%: media plans are set from trailing performance, so spend drifts
    # up by ~15% purely mechanically when revenue has been growing. Only a
    # deliberate uplift beyond that drift is evidence of anything.
    if chg > 25 and revenue_delta_pct < -3:
        return {"type": "structured", "detail":
                f"Marketing spend rose {chg:.0f}% month-on-month while revenue fell "
                f"{abs(revenue_delta_pct):.1f}%. The main demand lever was pushed harder, "
                f"not eased, which argues against a simple demand-shortfall explanation.",
                "magnitude_pct": round(chg, 1)}
    return None
