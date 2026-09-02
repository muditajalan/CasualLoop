"""
Stage 5 + 6 - ACTION CHAIN and PERSONA NARRATIVE.

This is the ONLY module that may call a language model, and even here it may not
produce a number. The action chain is assembled from the governed driver library
in kpi_contract.yaml - levers and owners are looked up, never invented - and the
expected impact is computed arithmetically from the diagnosis.

The LLM is handed a finished evidence object and asked to write it up for a
specific reader. Every number it emits is then extracted and reconciled against
that object by `numeric_guard`. A narrative containing an unverifiable figure is
rejected rather than shown.
"""

import os
import re
import time

INR = lambda v: ("-" if v < 0 else "") + "INR " + format(abs(v), ",.0f")


# ------------------------------------------------------------- action chain ----
def build_action_chain(hyps, contract, kpi="net_revenue", top_n=3):
    """
    driver -> controllable lever -> action -> expected impact -> owner
           -> confidence -> monitoring plan

    Levers and owners come from the contract. Uncontrollable drivers (seasonality)
    are explanatory only and never become recommendations.
    """
    lib = contract["driver_library"]
    chain = []
    for h in hyps[:top_n]:
        spec = lib.get(h["driver"], {})
        if not spec.get("controllable"):
            continue
        if h["contribution_inr"] >= 0:
            continue
        chain.append({
            "driver": h["driver"],
            "finding": h["statement"],
            "lever": spec.get("lever"),
            "action": _action_text(h["driver"], h),
            "expected_impact_inr": round(abs(h["contribution_inr"]), 0),
            "impact_basis": "Recovering this driver's measured contribution to the "
                            "current period movement; not a forecast.",
            "owner": spec.get("owner"),
            "confidence": h["confidence"],
            "confidence_band": h["confidence_band"],
            "monitoring": _monitoring(h["driver"]),
            "controllable": spec.get("controllable"),
        })
    return chain


def _action_text(driver, h):
    return {
        "distribution_loss": "Open a win-back conversation with the lapsed account this week; "
                             "if unrecoverable, reassign the territory before the next cycle.",
        "stockout": "Reset safety stock at the serving DC for the affected SKU and confirm "
                    "the replenishment lead time that produced the gap.",
        "competitor_price": "Approve a time-boxed tactical trade promotion in the affected "
                            "region and review pack-price architecture against the rival line.",
        "marketing_spend": "Reallocate working media into the affected region and channel.",
        "price": "Review the price change: hold, partially roll back, or offset with trade "
                 "support. Decide before the next pricing cycle closes.",
        "mix": "Revisit assortment and shelf-space plan for the affected range.",
        "discount_depth": "Tighten trade-spend guardrails on the affected accounts.",
    }.get(driver, "Review with the owning team.")


def _monitoring(driver):
    return {
        "distribution_loss": "Track active_accounts weekly; alert if the account has not "
                             "reordered within 21 days.",
        "stockout": "Track stockout_days and fill rate weekly for the affected SKU-region.",
        "competitor_price": "Track competitor_price_index weekly; re-evaluate if it recovers "
                            "above 96 for two consecutive weeks.",
        "marketing_spend": "Track spend versus plan and units weekly.",
        "price": "Track units_sold and mix-adjusted price weekly; re-evaluate elasticity "
                 "after four weeks.",
        "mix": "Track mix effect monthly against the trailing basket.",
        "discount_depth": "Track discount value as a share of gross revenue weekly.",
    }.get(driver, "Review at the next cycle.")


# ---------------------------------------------------------- numeric guard ----
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _nums(text):
    out = set()
    for m in NUM.findall(text or ""):
        try:
            out.add(round(abs(float(m.replace(",", ""))), 2))
        except ValueError:
            pass
    return out


def numeric_guard(text, evidence_numbers, tolerance=0.02):
    """
    Every number in the narrative must reconcile to a number the engine computed.

    Returns (ok, unverified). Two exemptions, both to avoid noise rather than to
    weaken the check: small integers (<=100), which are counts, percentages or
    ordinals; and four-digit years, which appear inside period labels like
    "2026-03" and are not quantitative claims.
    """
    allowed = {round(abs(float(v)), 2) for v in evidence_numbers}
    unverified = []
    for n in _nums(text):
        if n <= 100:
            continue
        if 1900 <= n <= 2100 and float(n).is_integer():
            continue                      # a year inside a date, not a claim
        if any(abs(n - a) <= tolerance * max(a, 1) for a in allowed):
            continue
        unverified.append(n)
    return (len(unverified) == 0), unverified


def evidence_numbers(row, hyps, cross, chain):
    """Every figure the engine computed, and therefore every figure it may state."""
    vals = [abs(row["delta"]), abs(row["value"]), abs(row["expected"]),
            abs(row["delta_pct"])]
    for h in hyps:
        vals += [abs(h["contribution_inr"]), abs(h["share_of_gap_pct"]), h["confidence"] * 100]
    for c in chain:
        vals.append(c["expected_impact_inr"])
    for k, v in (cross.get("kpis") or {}).items():
        if isinstance(v, dict) and "delta_pct" in v:
            vals.append(abs(v["delta_pct"]))
        if isinstance(v, dict) and "value" in v:
            vals += [abs(v["value"]), abs(v["expected"])]
    if cross.get("implied_elasticity"):
        vals.append(abs(cross["implied_elasticity"]))
    b = cross.get("bridge") or {}
    for k in ("volume_effect", "price_effect", "mix_effect", "reference_revenue"):
        if b.get(k) is not None:
            vals.append(abs(b[k]))
    return vals


# --------------------------------------------------------------- narrative ----
DISHA = """You are writing for Disha, a Regional Sales Ops Manager. She owns the number and
will be asked about it in a meeting today. Give her at most 4 sentences: what moved, the
two biggest causes with rupee figures, and the single action she can authorise herself.
No methodology, no hedging language, no bullet points."""

FARHAN = """You are writing for Farhan, a Business/Data Analyst. He will check your work.
Give him the full ranked driver list with contributions and confidence, the analytical
method used for each, which alternatives were considered and rejected and why, and what
would change the conclusion. Be precise and technical."""


def render(row, hyps, cross, chain, verdict, persona="disha", use_llm=None):
    """
    Produce the persona narrative. Returns a dict including telemetry.

    If no API key is present the deterministic template is used instead. The
    numeric guard runs identically either way, because the guard is a property of
    the system, not of the model.
    """
    t0 = time.time()
    ev = evidence_numbers(row, hyps, cross, chain)
    use_llm = (os.environ.get("ANTHROPIC_API_KEY") is not None) if use_llm is None else use_llm

    if use_llm:
        text, tel = _llm_render(row, hyps, cross, chain, verdict, persona)
    else:
        text, tel = _template_render(row, hyps, cross, chain, verdict, persona), {
            "mode": "deterministic_template", "model_calls": 0,
            "input_tokens": 0, "output_tokens": 0, "cost_inr": 0.0}

    ok, unverified = numeric_guard(text, ev)
    if not ok and use_llm:
        text = _template_render(row, hyps, cross, chain, verdict, persona)
        tel["guard_triggered"] = True
        tel["rejected_numbers"] = unverified
        ok, unverified = numeric_guard(text, ev)

    tel["latency_ms"] = int(1000 * (time.time() - t0))
    tel["numeric_guard_passed"] = ok
    tel["unverified_numbers"] = unverified
    return {"persona": persona, "text": text, "telemetry": tel}


def _llm_render(row, hyps, cross, chain, verdict, persona):
    from anthropic import Anthropic

    payload = {
        "movement": {"scope": row.get("region"), "period": str(row["period"]),
                     "delta_inr": round(float(row["delta"]), 2),
                     "delta_pct": round(float(row["delta_pct"]), 2)},
        "cross_kpi_pattern": cross.get("pattern"),
        "drivers": [{"driver": h["driver"], "contribution_inr": round(h["contribution_inr"], 2),
                     "share_pct": h["share_of_gap_pct"], "confidence": h["confidence"],
                     "method": h["method"], "statement": h["statement"]} for h in hyps],
        "verdict": verdict, "actions": chain,
    }
    sys_prompt = (DISHA if persona == "disha" else FARHAN) + (
        "\n\nCRITICAL: use ONLY numbers present in the JSON below. Do not compute, "
        "round differently, estimate or infer any figure. If a number you want is not "
        "in the JSON, describe it in words instead.")

    t = time.time()
    c = Anthropic()
    r = c.messages.create(model="claude-sonnet-4-6", max_tokens=700,
                          system=sys_prompt,
                          messages=[{"role": "user", "content": str(payload)}])
    text = "".join(b.text for b in r.content if b.type == "text")
    inp, out = r.usage.input_tokens, r.usage.output_tokens
    cost_inr = (inp / 1e6 * 3.0 + out / 1e6 * 15.0) * 88.0     # USD rates -> INR
    return text, {"mode": "llm", "model": "claude-sonnet-4-6", "model_calls": 1,
                  "input_tokens": inp, "output_tokens": out,
                  "cost_inr": round(cost_inr, 3),
                  "llm_latency_ms": int(1000 * (time.time() - t))}


def _template_render(row, hyps, cross, chain, verdict, persona):
    reg, per = row.get("region"), str(row["period"])
    d, dp = float(row["delta"]), float(row["delta_pct"])
    neg = [h for h in hyps if h["contribution_inr"] < 0]

    if persona == "disha":
        if verdict["abstain"]:
            lead = neg[0] if neg else None
            s = (f"{reg} net revenue came in {INR(d)} below expectation for {per} "
                 f"({dp:.1f}%). The most likely cause is {lead['driver'].replace('_',' ')} "
                 f"at {INR(lead['contribution_inr'])}, but the evidence disagrees with "
                 f"itself, so this is not yet a conclusion you should take into the room. "
                 if lead else
                 f"{reg} net revenue came in {INR(d)} below expectation for {per}. ")
            s += (f"{verdict['reason'].split(':')[0]}. Before acting, "
                  f"{verdict['resolve_with'][0][0].lower()}{verdict['resolve_with'][0][1:]}.")
            return s
        top = neg[:2]
        s = (f"{reg} net revenue came in {INR(d)} below expectation for {per} "
             f"({dp:.1f}%). ")
        s += "The two largest causes are " + " and ".join(
            f"{h['driver'].replace('_',' ')} at {INR(h['contribution_inr'])} "
            f"({h['share_of_gap_pct']:.0f}% of the movement)" for h in top) + ". "
        if chain:
            c0 = chain[0]
            s += (f"The action you can authorise today is: {c0['action']} "
                  f"Owner is {c0['owner']}, worth about {INR(c0['expected_impact_inr'])}.")
        return s

    lines = [f"{reg} {per} - net revenue {INR(d)} vs baseline ({dp:.1f}%).",
             f"Cross-KPI pattern: {cross.get('pattern')}. {cross.get('interpretation','')}",
             "", "Ranked drivers:"]
    for h in hyps:
        lines.append(f"  {h['driver']:<19} {INR(h['contribution_inr']):>16}  "
                     f"{h['share_of_gap_pct']:>5.1f}%  conf {h['confidence']:.2f} "
                     f"({h['confidence_band']})")
        lines.append(f"      method: {h['method']}")
    lines.append("")
    if verdict["abstain"]:
        lines += ["ABSTAINED. " + verdict["reason"], "",
                  "What would resolve it:"] + [f"  - {r}" for r in verdict["resolve_with"]]
    else:
        lines.append(f"Answered. Unexplained residual {verdict.get('residual_pct',0):.0f}%.")
    rejected = [h["driver"] for h in hyps if h["contribution_inr"] >= 0]
    if rejected:
        lines.append(f"Considered and rejected (no measurable negative contribution): "
                     f"{', '.join(rejected)}.")
    return "\n".join(lines)
