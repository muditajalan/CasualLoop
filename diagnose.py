"""
Stage 3 + 4 - HYPOTHESES and CONFIDENCE.  No LLM anywhere in this module.

Generates candidate root causes from the driver library declared in
kpi_contract.yaml, quantifies each one in rupees, retrieves corroborating
(or contradicting) text, scores confidence with the transparent weighted formula
from the contract, and decides whether to answer or abstain.

The LLM never sees this stage. It is handed the finished object afterwards and
asked only to write it up.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Retrieval queries per driver. Lexical/TF-IDF rather than dense embeddings:
# the corpus is small, it needs no model download, and - more importantly - it is
# inspectable, so an analyst can see exactly why a snippet was retrieved.
DRIVER_QUERIES = {
    "stockout": "stock out of stock replenishment warehouse unfulfilled back-order inventory",
    "distribution_loss": "distributor account no indent contract lapsed not renewed churn stopped ordering",
    "competitor_price": "competitor rival discount promotion price cut competing brand lost order",
    "marketing_spend": "campaign creatives media reach spend activation",
    "price": "price MRP revised price point pushback expensive",
    "mix": "assortment shelf listing range portfolio",
}

CONTRADICTS = {
    "competitor_price": ["campaign is landing well", "footfall", "ahead of plan", "sentiment stable"],
    "marketing_spend": ["ahead of plan", "landing well", "reach tracking"],
    "price": ["landing well", "footfall", "normal"],
}


# ---------------------------------------------------------------- retrieval ----
def retrieve(signals, driver, region, period, window_days=45, k=3):
    """Top-k text signals for a driver, restricted to the region and time window."""
    if driver not in DRIVER_QUERIES:
        return []
    end = period.to_timestamp(how="end")
    start = end - pd.Timedelta(days=window_days)
    s = signals[(signals["region"] == region)
                & (signals["signal_date"] >= start)
                & (signals["signal_date"] <= end)]
    if s.empty:
        return []

    corpus = s["text"].tolist()
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1)
        X = vec.fit_transform(corpus + [DRIVER_QUERIES[driver]])
    except ValueError:
        return []
    sim = cosine_similarity(X[-1], X[:-1]).ravel()

    out = []
    for i in np.argsort(-sim)[:k]:
        if sim[i] < 0.08:
            continue
        r = s.iloc[i]
        out.append({"signal_id": r["signal_id"], "date": str(r["signal_date"].date()),
                    "type": r["signal_type"], "text": r["text"],
                    "similarity": round(float(sim[i]), 3)})
    return out


def count_contradictions(signals, driver, region, period, window_days=45):
    if driver not in CONTRADICTS:
        return 0
    end = period.to_timestamp(how="end")
    start = end - pd.Timedelta(days=window_days)
    s = signals[(signals["region"] == region)
                & (signals["signal_date"] >= start) & (signals["signal_date"] <= end)]
    n = 0
    for t in s["text"]:
        if any(p in t.lower() for p in CONTRADICTS[driver]):
            n += 1
    return n


# --------------------------------------------------------------- hypotheses ----
def generate(sales, ops, signals, decomp, detection_row, contract, cross=None):
    """Quantified, evidence-backed candidate root causes for one movement."""
    region = decomp["scope"].get("region")
    period = pd.Period(decomp["period"], "M")
    gap = decomp["total_gap"]
    by_sku, by_acct = decomp["by_sku"], decomp["by_account"]

    ops_m = ops.copy()
    ops_m["period"] = ops_m["week_start"].dt.to_period("M")
    here = ops_m[(ops_m["region"] == region) & (ops_m["period"] == period)]
    elsewhere = ops_m[(ops_m["region"] != region) & (ops_m["period"] == period)]

    hyps = []
    cross = cross or {}
    pattern = cross.get("pattern")

    # -- price pushback: only proposed when the CONNECTED KPIs show its signature
    # (units down while mix-adjusted price is up). Revenue alone cannot distinguish
    # this from a demand shock, which is the whole reason for reading KPIs together.
    if pattern == "price_pushback":
        vol = cross.get("bridge", {}).get("volume_effect", 0.0)
        contrib = max(vol, gap) if gap < 0 else 0.0
        el = cross.get("implied_elasticity")
        hyps.append({
            "driver": "price",
            "statement": (f"Mix-adjusted price rose "
                          f"{cross['kpis']['like_for_like_price']['delta_pct']:+.1f}% while units "
                          f"fell {abs(cross['kpis']['units_sold']['delta_pct']):.1f}%"
                          + (f" (implied elasticity {el})" if el else "")),
            "contribution_inr": contrib,
            "method": "cross-KPI divergence: units vs mix-adjusted price from the "
                      "price/volume/mix bridge; volume effect attributed to the price move",
            "evidence": [{"source": "order_db", "detail":
                          f"volume effect INR {vol:,.0f}, price effect INR "
                          f"{cross['bridge']['price_effect']:,.0f}"}],
            "signals": retrieve(signals, "price", region, period),
            "fires_elsewhere": 0,
        })

    # -- distribution loss: accounts that went to zero against a live expectation
    vanished = decomp.get("vanished_accounts", [])
    if vanished:
        contrib = sum(v["gap"] for v in vanished)
        names = ", ".join(v["account_id"] for v in vanished)
        hyps.append({
            "driver": "distribution_loss",
            "statement": f"{len(vanished)} distributor account(s) stopped ordering entirely ({names})",
            "contribution_inr": contrib,
            "method": "zero-fill panel: actual 0 against a positive baseline expectation",
            "evidence": [{"source": "order_db", "detail":
                          f"{names} expected INR {abs(contrib):,.0f}, actual 0"}],
            "signals": retrieve(signals, "distribution_loss", region, period),
            "fires_elsewhere": 0,
        })

    # -- stockout: worst SKU's gap in EXCESS of the broad decline, gated on ops data
    stockout_days = float(here["stockout_days"].sum()) if not here.empty else 0.0
    if not by_sku.empty and stockout_days > 0:
        worst = by_sku.iloc[0]
        broad_pct = float(by_sku["gap_pct"].iloc[1:].median())
        excess_pct = float(worst["gap_pct"]) - broad_pct
        if excess_pct < 0:
            contrib = excess_pct / 100.0 * float(worst["expected"])
            hyps.append({
                "driver": "stockout",
                "statement": f"{worst['sku']} fell {worst['gap_pct']:.0f}% versus "
                             f"{broad_pct:.0f}% for the rest of the range",
                "contribution_inr": contrib,
                "method": "excess-over-broad-decline on the affected SKU, gated on "
                          "recorded stockout days (weekly source, so window is whole weeks)",
                "evidence": [
                    {"source": "ops_hub", "detail": f"{stockout_days:.0f} recorded stockout days"},
                    {"source": "order_db", "detail":
                     f"{worst['sku']} gap INR {abs(float(worst['gap'])):,.0f}"}],
                "signals": retrieve(signals, "stockout", region, period),
                "fires_elsewhere": int((elsewhere["stockout_days"] > 0).sum()),
            })

    # -- competitor price: index z-score vs the same month in other regions
    if not here.empty and not elsewhere.empty:
        idx_here = float(here["competitor_price_index"].mean())
        mu, sd = float(elsewhere["competitor_price_index"].mean()), \
                 float(elsewhere["competitor_price_index"].std(ddof=1) or 1.0)
        z = (idx_here - mu) / sd
        if z < -2.0:
            # Measured directly, not as a residual. Isolate the market-wide effect
            # by looking only at SKUs OTHER than the stocked-out one, then removing
            # the portion already explained by the lost distributor. What remains is
            # a decline affecting the whole range across surviving accounts, which is
            # the signature of a demand-side shock rather than a supply or account one.
            rest = by_sku.iloc[1:] if len(by_sku) > 1 else by_sku
            broad_pct = float(rest["gap_pct"].median()) if not rest.empty else 0.0
            broad_effect = broad_pct / 100.0 * float(rest["expected"].sum())
            churn = sum(h["contribution_inr"] for h in hyps
                        if h["driver"] == "distribution_loss")
            contrib = min(0.0, broad_effect - churn)
            hyps.append({
                "driver": "competitor_price",
                "statement": f"Competitor price index at {idx_here:.1f} versus {mu:.1f} "
                             f"in other regions (z = {z:.1f})",
                "contribution_inr": contrib,
                "method": "residual after named drivers, gated on a competitor price "
                          "z-score; residual attribution is weaker than direct measurement",
                "evidence": [{"source": "ops_hub",
                              "detail": f"competitor_price_index {idx_here:.1f}, z={z:.1f}"}],
                "signals": retrieve(signals, "competitor_price", region, period),
                "fires_elsewhere": 0,
            })

    # -- price / mix, read straight off the bridge
    br = decomp.get("bridge", {})
    already = {h["driver"] for h in hyps}
    for drv, key in (("price", "price_effect"), ("mix", "mix_effect")):
        v = br.get(key)
        if drv in already:          # cross-KPI already raised a better-evidenced version
            continue
        if v is not None and abs(v) > 0.10 * abs(gap):
            hyps.append({
                "driver": drv,
                "statement": f"{key.replace('_', ' ')} of INR {v:,.0f} versus the "
                             f"trailing 3-month basket",
                "contribution_inr": float(v) if v < 0 else 0.0,
                "method": "price/volume/mix bridge (deterministic arithmetic)",
                "evidence": [{"source": "order_db", "detail": f"{key} INR {v:,.0f}"}],
                "signals": retrieve(signals, drv, region, period),
                "fires_elsewhere": 0,
            })

    # Drivers are measured independently, so their contributions can legitimately
    # overlap and over-explain the detected gap (interaction effects). We rescale
    # to reconcile and report the overlap openly rather than hiding it.
    named = sum(h["contribution_inr"] for h in hyps if h["contribution_inr"] < 0)
    overlap = abs(named) / abs(gap) if gap else 1.0
    prio = cross.get("prioritised_drivers", [])
    for h in hyps:
        h["prioritised_by_cross_kpi"] = h["driver"] in prio
    for h in hyps:
        if overlap > 1.0 and h["contribution_inr"] < 0:
            h["contribution_raw_inr"] = h["contribution_inr"]
            h["contribution_inr"] = h["contribution_inr"] / overlap
        h["contradictions"] = count_contradictions(signals, h["driver"], region, period)
        h["share_of_gap_pct"] = round(100 * h["contribution_inr"] / gap, 1) if gap else 0.0
    if overlap > 1.0:
        for h in hyps:
            h["overlap_adjustment"] = round(overlap, 3)
    return hyps


# --------------------------------------------------------------- confidence ----
def score(hyps, detection_row, contract, freshness_ok=True):
    """Apply the contract's weighted confidence model. Fully transparent."""
    cm = contract["confidence_model"]
    w, pen = cm["weights"], cm["penalties"]
    hist = int(detection_row.get("history_months", 99))

    for h in hyps:
        n_src = len({e["source"] for e in h["evidence"]}) + (1 if h["signals"] else 0)
        corrob = min(n_src / 3.0, 1.0)
        magnitude = min(abs(h["contribution_inr"]) / max(abs(detection_row["delta"]), 1), 1.0)
        uniqueness = 1.0 if h["fires_elsewhere"] == 0 else max(0.0, 1 - h["fires_elsewhere"] / 3.0)
        timing = 1.0 if h["signals"] or h["evidence"] else 0.4
        dq = (1.0 if freshness_ok else 0.7) * (1.0 if hist >= 12 else 0.6)

        s = (w["evidence_corroboration"] * corrob
             + w["magnitude_sufficiency"] * magnitude
             + w["uniqueness"] * uniqueness
             + w["timing_alignment"] * timing
             + w["data_quality"] * dq)

        applied = []
        if h["contradictions"] > 0:
            s += pen["contradictory_evidence"]; applied.append("contradictory_evidence")
        if hist < 12:
            s += pen["history_lt_12_weeks"]; applied.append("history_lt_12_weeks")
        if not freshness_ok:
            s += pen["stale_source_gt_2x_cadence"]; applied.append("stale_source")

        s = float(np.clip(s, 0.0, 1.0))
        h["confidence"] = round(s, 3)
        h["confidence_band"] = "high" if s >= 0.70 else ("medium" if s >= 0.45 else "low")
        h["confidence_breakdown"] = {
            "evidence_corroboration": round(float(corrob), 2),
            "magnitude_sufficiency": round(float(magnitude), 2),
            "uniqueness": round(float(uniqueness), 2),
            "timing_alignment": round(float(timing), 2),
            "data_quality": round(float(dq), 2), "penalties_applied": applied,
        }

    return sorted(hyps, key=lambda x: -abs(x["contribution_inr"]))


def decide(hyps, detection_row, structured_contra=None):
    """Answer or abstain, per the contract's abstain_rule."""
    if not hyps:
        return {"abstain": True, "reason": "No candidate driver met its evidence test.",
                "resolve_with": ["Confirm source freshness", "Widen the analysis window"]}

    top = hyps[0]

    # Contradicted leading hypothesis. A quantitatively strong driver that the
    # qualitative record argues against is exactly the case where a confident
    # answer is most dangerous - the numbers look decisive, so nobody checks.
    # A marketing-spend contradiction only undercuts DEMAND-side explanations.
    # A distributor that cancelled its contract is unaffected by how much was spent
    # on advertising, so applying the contradiction there would be a category error.
    DEMAND_SIDE = {"price", "competitor_price", "marketing_spend", "mix", "discount_depth"}
    contra_txt = top.get("contradictions", 0)
    struct = structured_contra if top["driver"] in DEMAND_SIDE else None
    if contra_txt > 0 or struct:
        bits = []
        if contra_txt:
            bits.append(f"{contra_txt} qualitative signal(s) argue against it")
        if struct:
            bits.append(struct["detail"])
        return {"abstain": True,
                "reason": (f"Leading hypothesis ({top['driver']}, confidence "
                           f"{top['confidence']:.2f}) is contradicted by other evidence: "
                           + "; ".join(bits)),
                "leading_hypothesis": top["driver"],
                "leading_confidence": top["confidence"],
                "resolve_with": _resolve(top)}

    explained = sum(abs(h["contribution_inr"]) for h in hyps)
    residual = 1 - min(explained / max(abs(detection_row["delta"]), 1), 1.0)

    if top["confidence"] < 0.45:
        return {"abstain": True,
                "reason": f"Best hypothesis ({top['driver']}) scores {top['confidence']:.2f}, "
                          f"below the 0.45 answer threshold.",
                "resolve_with": _resolve(top)}

    if len(hyps) > 1 and abs(hyps[0]["confidence"] - hyps[1]["confidence"]) < 0.08:
        return {"abstain": True,
                "reason": f"{hyps[0]['driver']} and {hyps[1]['driver']} are within 0.08 "
                          f"confidence of each other and imply different actions.",
                "resolve_with": _resolve(top)}

    if residual > 0.40:
        return {"abstain": True,
                "reason": f"{residual:.0%} of the movement is unexplained, above the 40% ceiling.",
                "resolve_with": ["Decompose one level deeper (account x SKU)",
                                 "Check for a source that has not refreshed"]}

    return {"abstain": False, "residual_pct": round(100 * residual, 1)}


def _resolve(top):
    return [
        f"Confirm or rule out {top['driver']} with the owning team",
        "Obtain a source with matching grain (the weekly source cannot resolve daily timing)",
        "Re-run once the contradicting qualitative signals are reconciled",
    ]
