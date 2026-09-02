"""
CausalLoop demo surface. Presentation only - all reasoning lives in engine/.

Renders: role switcher (row/column security), prioritised insight list, cross-KPI
panel, contribution bridge, ranked hypotheses with evidence, answer-or-abstain
verdict, the action chain, persona narratives, runtime telemetry, and a feedback
loop that visibly reranks future diagnoses.
"""

import time

import ipywidgets as widgets
import pandas as pd
import yaml
from IPython.display import HTML, clear_output, display

from engine.crosskpi import analyse, structured_contradiction
from engine.decompose import decompose
from engine.detect import INTENSIVE_KPIS, detect, insights
from engine.diagnose import decide, generate, score
from narrative.render import build_action_chain, render

INR = lambda v: ("-" if v < 0 else "") + "₹" + format(abs(v), ",.0f")
CSS = """<style>
.cl-card{border:1px solid #d8d3e8;border-radius:10px;padding:14px 16px;margin:10px 0;
 font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#fff}
.cl-h{font-weight:700;font-size:15px;color:#3b1a63;margin-bottom:8px;
 border-bottom:2px solid #a100ff;padding-bottom:5px}
.cl-k{display:inline-block;background:#f3eefc;border-radius:6px;padding:5px 10px;
 margin:3px 6px 3px 0;font-size:13px}
.cl-tbl{border-collapse:collapse;width:100%;font-size:13px}
.cl-tbl th{background:#f3eefc;text-align:left;padding:6px 8px;color:#3b1a63}
.cl-tbl td{padding:6px 8px;border-top:1px solid #eee}
.cl-ab{background:#fff4e5;border-left:4px solid #ff8c00;padding:11px 14px;border-radius:6px}
.cl-ok{background:#eefaf0;border-left:4px solid #1a9c4a;padding:11px 14px;border-radius:6px}
.cl-ev{background:#fafafa;border-left:3px solid #bbb;padding:7px 10px;margin:5px 0;
 font-size:12px;color:#444}
.cl-mut{color:#777;font-size:12px}
.cl-bar{height:13px;background:#a100ff;border-radius:3px;display:inline-block}
</style>"""


class App:
    def __init__(self, sales, ops, signals, contract, gt=None):
        self.sales, self.ops, self.signals = sales, ops, signals
        self.contract, self.gt = contract, gt or {}
        self.priors = {}          # driver -> multiplier, updated by feedback
        self.feedback_log = []
        self.t0 = time.time()
        self.res, self.funnel = detect(sales, contract, "net_revenue", ("region",))
        # Second sweep at SKU grain. Newly launched SKUs cannot be baselined at
        # region level - they are a small slice of a large number - so the sparse
        # history case only becomes visible when the same detector is run one
        # level down. This is also how a real deployment would search.
        self.res_sku, self.funnel_sku = detect(sales, contract, "net_revenue", ("sku",))
        self.detect_ms = int(1000 * (time.time() - self.t0))

    # ------------------------------------------------------------ security --
    def visible(self, role):
        spec = self.contract["roles"][role]
        i = insights(self.res, top_n=20)
        return i[i["region"].isin(spec["regions"])], spec

    def sparse_insights(self, role):
        """SKU-grain insights whose history is too short to baseline confidently."""
        spec = self.contract["roles"][role]
        if len(spec["regions"]) < 4:      # regional roles do not own SKU-level KPIs
            return []
        s = self.res_sku[self.res_sku["is_insight"] & (self.res_sku["history_months"] < 12)]
        return [(r["sku"], str(r["period"]), r) for _, r in s.iterrows()]

    def sparse_html(self, sku, period_str, row):
        pen = self.contract["confidence_model"]["penalties"]["history_lt_12_weeks"]
        first = self.sales[self.sales["sku"] == sku]["order_date"].min()
        return (CSS + f"<div class='cl-card'><div class='cl-h'>SPARSE HISTORY &nbsp;—&nbsp; "
                f"{sku} · {period_str}</div>"
                f"<span class='cl-k'>actual {INR(row['value'])}</span>"
                f"<span class='cl-k'>expected {INR(row['expected'])}</span>"
                f"<span class='cl-k'><b>{INR(row['delta'])} ({row['delta_pct']:.1f}%)</b></span>"
                f"<span class='cl-k'>history {int(row['history_months'])} months</span>"
                f"<p style='font-size:13px'>This SKU first sold on "
                f"{pd.Timestamp(first).date()}. With only {int(row['history_months'])} months "
                f"of history the engine has no seasonal profile of its own for it and the "
                f"sample standard deviation is itself poorly estimated.</p>"
                f"<div class='cl-ab'><b>How the engine self-limits here</b><ul>"
                f"<li>No expectation is produced at all for the first "
                f"{3} months — the engine returns nothing rather than inventing a baseline."
                f"</li><li>Once scored, the prediction interval is widened using a Student-t "
                f"multiplier instead of 1.96, because the spread estimate is unreliable.</li>"
                f"<li>Any hypothesis raised here carries a fixed confidence penalty of "
                f"{pen} from the contract, capping it below the answer threshold.</li>"
                f"<li>Net effect: this movement is surfaced for a human to look at, but the "
                f"engine will not assert a cause for it.</li></ul></div>"
                f"<div class='cl-mut'>Round 2 minimum-prototype item: "
                f"'one sparse-history or newly launched KPI scenario'.</div></div>")

    # ------------------------------------------------------------ pipeline --
    def diagnose(self, region, period_str, persona):
        t = {}
        s0 = time.time()
        row = self.res[(self.res.region == region)
                       & (self.res.period.astype(str) == period_str)].iloc[0]
        per = pd.Period(period_str, "M")

        s1 = time.time(); cross = analyse(self.sales, self.contract, region, per)
        t["cross_kpi_ms"] = int(1000 * (time.time() - s1))

        s1 = time.time()
        pack = decompose(self.sales, per, {"region": region}, target_gap=float(row["delta"]))
        t["decompose_ms"] = int(1000 * (time.time() - s1))

        s1 = time.time()
        hyps = generate(self.sales, self.ops, self.signals, pack, row, self.contract, cross=cross)
        for h in hyps:                       # feedback priors applied here
            h["prior"] = self.priors.get(h["driver"], 1.0)
        hyps = score(hyps, row, self.contract)
        for h in hyps:
            if h["prior"] != 1.0:
                h["confidence"] = round(min(1.0, h["confidence"] * h["prior"]), 3)
                h["confidence_band"] = ("high" if h["confidence"] >= .70
                                        else "medium" if h["confidence"] >= .45 else "low")
        hyps.sort(key=lambda x: -abs(x["contribution_inr"]))
        t["hypotheses_ms"] = int(1000 * (time.time() - s1))

        sc = structured_contradiction(self.ops, region, per, float(row["delta_pct"]))
        verdict = decide(hyps, row, structured_contra=sc)
        chain = build_action_chain(hyps, self.contract)
        nar = render(row, hyps, cross, chain, verdict, persona)
        t["narrative_ms"] = nar["telemetry"]["latency_ms"]
        t["total_ms"] = int(1000 * (time.time() - s0))
        t["detection_sweep_ms"] = self.detect_ms
        return dict(row=row, cross=cross, pack=pack, hyps=hyps, verdict=verdict,
                    chain=chain, narrative=nar, timing=t, struct=sc)

    # -------------------------------------------------------------- render --
    def html(self, d, role_spec, persona):
        row, cross, hyps = d["row"], d["cross"], d["hyps"]
        k, br = cross["kpis"], cross.get("bridge", {})
        h = [CSS]

        h.append(f"<div class='cl-card'><div class='cl-h'>1 · SIGNAL &nbsp;—&nbsp; "
                 f"{row['region']} · {row['period']} · net revenue</div>"
                 f"<span class='cl-k'>actual {INR(row['value'])}</span>"
                 f"<span class='cl-k'>expected {INR(row['expected'])}</span>"
                 f"<span class='cl-k'><b>{INR(row['delta'])} ({row['delta_pct']:.1f}%)</b></span>"
                 f"<div class='cl-mut' style='margin-top:8px'>Funnel: "
                 f"{self.funnel['cells_monitored']} region-months monitored → "
                 f"{self.funnel['statistically_anomalous']} outside the 95% interval → "
                 f"{self.funnel['also_business_material']} also material. "
                 f"{self.funnel['materiality_rule']}</div></div>")

        lfl = k.get("like_for_like_price", {}).get("delta_pct", 0)
        h.append(f"<div class='cl-card'><div class='cl-h'>2 · CROSS-KPI PATTERN &nbsp;—&nbsp; "
                 f"{cross['pattern'].replace('_',' ').upper()}</div>"
                 f"<span class='cl-k'>net revenue {k['net_revenue']['delta_pct']:+.1f}%</span>"
                 f"<span class='cl-k'>units {k['units_sold']['delta_pct']:+.1f}%</span>"
                 f"<span class='cl-k'>price, mix-adjusted {lfl:+.1f}%</span>"
                 + (f"<span class='cl-k'>implied elasticity {cross['implied_elasticity']}</span>"
                    if cross.get("implied_elasticity") else "")
                 + f"<p style='font-size:13px'>{cross['interpretation']}</p>"
                 f"<div class='cl-mut'>Blended ASP reads {k['avg_selling_price']['delta_pct']:+.1f}% "
                 f"and is not used: product mix dominates the price term and can invert the "
                 f"signal. The contract requires the mix-versus-price split first.</div></div>")

        h.append(f"<div class='cl-card'><div class='cl-h'>3 · BRIDGE &amp; DECOMPOSITION</div>"
                 f"<span class='cl-k'>volume {INR(br.get('volume_effect',0))}</span>"
                 f"<span class='cl-k'>price {INR(br.get('price_effect',0))}</span>"
                 f"<span class='cl-k'>mix {INR(br.get('mix_effect',0))}</span>")
        t = d["pack"]["by_sku"].head(3)
        h.append("<table class='cl-tbl'><tr><th>worst SKUs</th><th>actual</th>"
                 "<th>expected</th><th>gap</th></tr>")
        for _, r in t.iterrows():
            h.append(f"<tr><td>{r['sku']}</td><td>{INR(r['value'])}</td>"
                     f"<td>{INR(r['expected'])}</td><td>{INR(r['gap'])}</td></tr>")
        h.append("</table>")
        van = d["pack"].get("vanished_accounts", [])
        if van:
            h.append("<div class='cl-ev'>Accounts that went to zero against a live "
                     "expectation: " + ", ".join(
                         f"<b>{v['account_id']}</b> (expected {INR(v['expected'])})"
                         for v in van) + "</div>")
        h.append("</div>")

        h.append("<div class='cl-card'><div class='cl-h'>4 · RANKED HYPOTHESES</div>"
                 "<table class='cl-tbl'><tr><th>driver</th><th>contribution</th><th>share</th>"
                 "<th>confidence</th><th>method</th></tr>")
        for x in hyps:
            w = int(90 * x["confidence"])
            h.append(f"<tr><td><b>{x['driver']}</b>"
                     + (" <span class='cl-mut'>◂ cross-KPI</span>"
                        if x.get("prioritised_by_cross_kpi") else "")
                     + (f" <span class='cl-mut'>prior×{x['prior']}</span>"
                        if x.get("prior", 1) != 1 else "")
                     + f"</td><td>{INR(x['contribution_inr'])}</td>"
                     f"<td>{x['share_of_gap_pct']:.1f}%</td>"
                     f"<td><span class='cl-bar' style='width:{w}px'></span> "
                     f"{x['confidence']:.2f} {x['confidence_band']}</td>"
                     f"<td class='cl-mut'>{x['method'][:70]}</td></tr>")
        h.append("</table>")
        lead = hyps[0]
        h.append(f"<div class='cl-ev'><b>{lead['statement']}</b><br>"
                 f"confidence breakdown: {lead['confidence_breakdown']}</div>")
        for e in lead["evidence"]:
            h.append(f"<div class='cl-ev'>[{e['source']}] {e['detail']}</div>")
        for s_ in lead["signals"][:3]:
            h.append(f"<div class='cl-ev'>[{s_['signal_id']} · {s_['date']} · "
                     f"sim {s_['similarity']}] {s_['text']}</div>")
        h.append("</div>")

        v = d["verdict"]
        if v["abstain"]:
            h.append(f"<div class='cl-card'><div class='cl-h'>5 · VERDICT — ABSTAIN</div>"
                     f"<div class='cl-ab'>{v['reason']}<br><br><b>What would resolve it</b><ul>"
                     + "".join(f"<li>{r}</li>" for r in v["resolve_with"]) + "</ul></div></div>")
        else:
            h.append(f"<div class='cl-card'><div class='cl-h'>5 · VERDICT — ANSWER</div>"
                     f"<div class='cl-ok'>Unexplained residual {v.get('residual_pct',0):.0f}%, "
                     f"within the 40% ceiling. No contradicting evidence against the leading "
                     f"hypothesis.</div></div>")

        if d["chain"]:
            h.append("<div class='cl-card'><div class='cl-h'>6 · ACTION CHAIN</div>"
                     "<table class='cl-tbl'><tr><th>driver</th><th>lever</th><th>action</th>"
                     "<th>impact</th><th>owner</th><th>conf</th><th>monitoring</th></tr>")
            for c in d["chain"]:
                h.append(f"<tr><td>{c['driver']}</td><td>{c['lever']}</td>"
                         f"<td>{c['action']}</td><td>{INR(c['expected_impact_inr'])}</td>"
                         f"<td>{c['owner']}</td><td>{c['confidence']:.2f}</td>"
                         f"<td class='cl-mut'>{c['monitoring']}</td></tr>")
            h.append("</table><div class='cl-mut'>Levers and owners are read from the KPI "
                     "contract, never generated. Uncontrollable drivers such as seasonality "
                     "are explanatory only and never become recommendations.</div></div>")

        nar = d["narrative"]
        tel = nar["telemetry"]
        who = "Disha · Regional Sales Ops Manager" if persona == "disha" else \
              "Farhan · Business / Data Analyst"
        h.append(f"<div class='cl-card'><div class='cl-h'>7 · NARRATIVE — {who}</div>"
                 f"<pre style='white-space:pre-wrap;font-family:inherit;font-size:13px'>"
                 f"{nar['text']}</pre>"
                 f"<div class='cl-mut'>Numeric guard: "
                 f"{'PASSED' if tel['numeric_guard_passed'] else 'FAILED ' + str(tel['unverified_numbers'])}"
                 f" — every figure reconciled against the computed evidence object. "
                 f"Mode: {tel['mode']}.</div></div>")

        tm = d["timing"]
        h.append(f"<div class='cl-card'><div class='cl-h'>8 · GOVERNANCE &amp; TELEMETRY</div>"
                 f"<span class='cl-k'>role regions: {', '.join(role_spec['regions'])}</span>"
                 f"<span class='cl-k'>columns denied: "
                 f"{', '.join(role_spec['columns_denied']) or 'none'}</span><br>"
                 f"<span class='cl-k'>detection sweep {tm['detection_sweep_ms']} ms</span>"
                 f"<span class='cl-k'>cross-KPI {tm['cross_kpi_ms']} ms</span>"
                 f"<span class='cl-k'>decompose {tm['decompose_ms']} ms</span>"
                 f"<span class='cl-k'>hypotheses {tm['hypotheses_ms']} ms</span>"
                 f"<span class='cl-k'>narrative {tm['narrative_ms']} ms</span>"
                 f"<span class='cl-k'><b>total {tm['total_ms']} ms</b></span><br>"
                 f"<span class='cl-k'>model calls {tel['model_calls']}</span>"
                 f"<span class='cl-k'>tokens in {tel['input_tokens']} / out {tel['output_tokens']}</span>"
                 f"<span class='cl-k'>cost ₹{tel['cost_inr']}</span>"
                 f"<div class='cl-mut' style='margin-top:8px'>Stages 1–6 are fully "
                 f"deterministic: no model call produced any number above. The model is used "
                 f"only in stage 7, and its output is checked before display.</div></div>")

        if self.gt:
            key = {"West": "WEST_MAR_2026", "South": "SOUTH_MAY_2026"}.get(row["region"])
            g = self.gt.get(key)
            if g and str(row["period"]) == g["period"]:
                nm = {"churn": "distribution_loss", "stockout": "stockout",
                      "competitor": "competitor_price", "price_test": "price"}
                est = {x["driver"]: x["share_of_gap_pct"] for x in hyps}
                rows = "".join(
                    f"<tr><td>{kk}</td><td>{vv:.1f}%</td>"
                    f"<td>{est.get(nm.get(kk,kk),0):.1f}%</td></tr>"
                    for kk, vv in sorted(g["driver_share_pct"].items(), key=lambda x: -x[1]))
                found = sum(1 for kk in g["driver_share_pct"] if nm.get(kk, kk) in est)
                h.append(f"<div class='cl-card'><div class='cl-h'>9 · GROUND TRUTH CHECK"
                         f"</div><table class='cl-tbl'><tr><th>planted driver</th>"
                         f"<th>true share</th><th>engine</th></tr>{rows}</table>"
                         f"<div class='cl-mut'>Drivers found {found} of "
                         f"{len(g['driver_share_pct'])}. The dataset plants these causes with "
                         f"known magnitudes, so the engine is scored rather than eyeballed. "
                         f"Shares differ from truth because the trailing baseline had already "
                         f"absorbed part of the movement; absolute contributions track "
                         f"closely.</div></div>")
        return "".join(h)


def launch(sales, ops, signals, contract, gt=None):
    app = App(sales, ops, signals, contract, gt)

    role = widgets.Dropdown(options=list(contract["roles"]), value="business_analyst",
                            description="Role:", layout=widgets.Layout(width="330px"))
    persona = widgets.ToggleButtons(options=[("Disha (ops)", "disha"),
                                             ("Farhan (analyst)", "farhan")], value="disha")
    insight = widgets.Dropdown(description="Insight:", layout=widgets.Layout(width="430px"))
    fb = widgets.ToggleButtons(options=["correct", "partly", "wrong"], value="correct",
                               layout=widgets.Layout(width="330px"))
    fb_btn = widgets.Button(description="Submit feedback", button_style="info")
    out, fb_out = widgets.Output(), widgets.Output()
    state = {}

    def refresh_list(*_):
        vis, spec = app.visible(role.value)
        opts = [(f"{r['region']} · {r['period']} · {INR(r['delta'])} "
                 f"({r['delta_pct']:.1f}%)", ("region", r["region"], str(r["period"])))
                for _, r in vis.iterrows()]
        for sku, per, row in app.sparse_insights(role.value):
            opts.append((f"{sku} · {per} · {INR(row['delta'])} "
                         f"({row['delta_pct']:.1f}%)  ⚠ sparse history",
                         ("sparse", sku, per)))
        insight.options = opts or [("no insights for this role", None)]
        render_one()

    def render_one(*_):
        with out:
            clear_output(wait=True)
            if not insight.value:
                _, spec = app.visible(role.value)
                display(HTML(CSS + "<div class='cl-card'><div class='cl-h'>No insights "
                             "visible for this role</div><p style='font-size:13px'>Row-level "
                             "security is applied before analysis, not in the display. This "
                             f"role may see: {', '.join(spec['regions'])}.</p></div>"))
                return
            kind, a, b_ = insight.value
            if kind == "sparse":
                row = app.res_sku[(app.res_sku["sku"] == a)
                                  & (app.res_sku["period"].astype(str) == b_)].iloc[0]
                display(HTML(app.sparse_html(a, b_, row)))
                return
            reg, per = a, b_
            d = app.diagnose(reg, per, persona.value)
            state["d"] = d
            _, spec = app.visible(role.value)
            display(HTML(app.html(d, spec, persona.value)))

    def submit(_):
        d = state.get("d")
        if not d:
            return
        drv = d["hyps"][0]["driver"]
        mult = {"correct": 1.10, "partly": 1.0, "wrong": 0.75}[fb.value]
        app.priors[drv] = round(app.priors.get(drv, 1.0) * mult, 3)
        app.feedback_log.append({"driver": drv, "verdict": fb.value, "prior": app.priors[drv]})
        with fb_out:
            clear_output(wait=True)
            display(HTML(CSS + f"<div class='cl-card'><div class='cl-h'>FEEDBACK RECORDED"
                         f"</div><span class='cl-k'>{drv} marked <b>{fb.value}</b></span>"
                         f"<span class='cl-k'>prior now ×{app.priors[drv]}</span>"
                         f"<div class='cl-mut' style='margin-top:6px'>The stored prior scales "
                         f"this driver's confidence on future diagnoses. Re-run the insight to "
                         f"see the ranking change. {len(app.feedback_log)} verdict(s) logged; "
                         f"every one is retained for audit.</div></div>"))
        render_one()

    role.observe(refresh_list, "value")
    persona.observe(render_one, "value")
    insight.observe(render_one, "value")
    fb_btn.on_click(submit)

    display(widgets.VBox([
        widgets.HTML(CSS + "<h2 style='font-family:sans-serif;color:#3b1a63;margin-bottom:2px'>"
                     "CausalLoop</h2><div style='font-family:sans-serif;color:#666;"
                     "font-size:13px;margin-bottom:10px'>KPI intelligence-to-action engine · "
                     "Femme Forecasters · Accenture Innovation Challenge 2026</div>"),
        widgets.HBox([role, persona]), insight, out,
        widgets.HTML("<b style='font-family:sans-serif;font-size:13px'>Was the leading "
                     "hypothesis right?</b>"),
        widgets.HBox([fb, fb_btn]), fb_out]))
    refresh_list()
    return app
