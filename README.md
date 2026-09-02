# CausalLoop

**A KPI intelligence-to-action engine.** Dashboards report what happened. CausalLoop explains
why, how confident it is, and what to do next — and refuses to answer when the evidence does
not support one.

Accenture Innovation Challenge 2026 · Round 2 · Problem Track 3: BusinessIntelligence.ai
Team **Femme Forecasters** — Mitanshi Khandelwal, Mudita Jalan (IIT Kanpur)

---

## The one design rule

> The LLM is never the source of quantitative truth.

Every number CausalLoop reports is computed by deterministic code. The model parses intent and
writes prose. It does not do arithmetic, and a numeric guard rejects any narrative containing a
figure that does not reconcile against the computed evidence object.

## LLM vs non-LLM

| Stage | Method | LLM? |
|---|---|---|
| Baseline & anomaly detection | Fourier seasonal profile + prediction interval from realised forecast error | No |
| Materiality gate | Business rules from the KPI contract (₹ and % floors) | No |
| Cross-KPI pattern reasoning | Revenue + units + mix-adjusted price read together | No |
| Contribution decomposition | Price/volume/mix bridge, zero-filled panel by SKU, account, channel | No |
| Hypothesis testing | Driver library tests: step change, z-score, excess-over-broad-decline | No |
| Evidence retrieval | TF-IDF similarity over CRM, support and market-intel text | No |
| Confidence & abstention | Transparent weighted formula defined in `kpi_contract.yaml` | No |
| Action chain | Levers and owners looked up from the contract; impact computed | No |
| Intent parsing | Natural-language question → structured query | **Yes** |
| Narrative synthesis | Evidence object → persona-appropriate prose | **Yes** |
| Numeric guard | Regex extraction + reconciliation against the evidence object | No |

Nine of eleven stages are deterministic. In the demo, the deterministic pipeline completes in
roughly 950 ms with **zero model calls**.

## Pipeline

```
3 sources (daily / weekly / irregular)
  └─ KPI contract ─────── governed definitions, thresholds, lineage, entitlements
       └─ [1] DETECT       seasonal baseline → prediction interval → materiality funnel
            └─ [1b] CROSS-KPI   revenue + units + mix-adjusted price → pattern
                 └─ [2] DECOMPOSE   bridge + gaps by SKU, account, channel
                      └─ [3] HYPOTHESISE   driver tests + retrieved evidence
                           └─ [4] SCORE   corroboration · magnitude · uniqueness · timing · quality
                                ├─ confident → action chain + persona narrative
                                └─ contradicted or unexplained → ABSTAIN + what would resolve it
                                     └─ [5] FEEDBACK   analyst verdict updates driver priors
```

## Quickstart

**Colab (no setup):** paste `causalloop_prototype.py` into one cell and run. About 90 seconds.

**Local:**
```bash
pip install -r requirements.txt
python generate_data.py     # deterministic, seeded
python evaluate.py          # 25-scenario evaluation
```

No API key is needed. The entire analytical path is deterministic; set `ANTHROPIC_API_KEY` to
switch stage 7 from the deterministic template to a real model call. The numeric guard runs
identically either way, because the guard is a property of the system, not of the model.

## The dataset

NovaCare, a fictional Indian D2C skincare brand: ₹48.4 Cr over 18 months, 140,881 order lines,
4 regions × 2 channels × 10 SKUs × 24 distributors per region.

Three sources with deliberately different grains and refresh cadences:

| Source | Grain | Refresh | Trust | Rows |
|---|---|---|---|---|
| `sales_orders` | order line | daily 02:00 | high | 140,881 |
| `weekly_ops` | week × region × channel | weekly Monday | medium | 632 |
| `signals` | event | irregular | low | 195 |

The weekly source cannot support day-level attribution, so the engine widens its evidence window
when citing it. That constraint is declared in the contract rather than left to the model.

**Synthetic but not arbitrary.** All randomness is drawn once for a base order book independent
of any planted event; events are then applied as deterministic filters. Toggling an event off
reproduces the exact counterfactual, so we know the true answer and can *score* the engine
rather than eyeball whether its output looks plausible.

## Demo scenarios

| Scenario | Movement | What it demonstrates |
|---|---|---|
| West · 2026-03 | −₹1,370,530 (−13.7%) | Multi-factor movement. True composition: churn 34.1%, stockout 30.1%, competitor 27.6%. Engine finds **3 of 3 in correct rank order** |
| South · 2026-05 | −15.1% | **Abstention.** Cross-KPI identifies price pushback (units −17.8%, mix-adjusted price +3.8%, implied elasticity −4.74) and names price as the leading cause — then declines to commit, because a call note reports price resistance while campaign notes report normal footfall and spend rose 50% |
| NC-BOOST-01 · 2026-06 | −31% | **Sparse history.** Launched 2026-02-01. No baseline produced for the first 3 months; once scored, a Student-t interval and a contract confidence penalty keep the engine from asserting a cause |
| Disha vs Farhan | — | Two personas, one evidence object, different depth and different actions |
| Role switch | — | `regional_ops_south` cannot see the West insight. Security applied before analysis, not in the display |

## Evaluation

`python evaluate.py` runs 20 scenarios each planting one known cause in a random region and
month, plus 5 null scenarios planting nothing. All share the same base order book, so the only
thing that varies is the planted cause.

| Metric | Result |
|---|---|
| Detection recall | 60% (12 of 20) |
| Detection precision | 100% (0 false positives on 5 nulls) |
| Root-cause top-1 accuracy | 100% |
| Root-cause top-3 accuracy | 100% |
| Median latency per scenario | 381 ms |

By event type: competitor 5/5 detected, stockout 4/5, price change 2/5, distributor churn 1/5 —
all at 100% top-1 accuracy when detected.

**On the 60% recall.** Detection engages at roughly −15%; smaller real movements are missed.
This is a deliberate precision-over-recall stance, because the failure mode the brief warns
about is alert fatigue. Both thresholds live in `kpi_contract.yaml` and can be loosened per KPI
without touching code. Top-1 accuracy is measured only on surfaced scenarios, which is the
honest denominator: it answers "when CausalLoop names a cause, is it right".

**A bug this evaluation caught.** The first run scored 33% top-1. The engine was not at fault —
the harness injected competitor events into sales but never updated the operational table, so
the competitor hypothesis gated on a price index that never moved. The engine was being judged
on evidence it was never given. Fixed in `patch_ops`; accuracy went 33% → 100%.

## Round 2 requirements

| Requirement | Where |
|---|---|
| 3–5 connected KPIs across 2–3 sources, different grains | Panel 2 reads 3; contract declares 5; 3 sources |
| KPI / semantic contract | `kpi_contract.yaml` |
| Two personas, different narratives | Panel 7, persona toggle |
| Multi-factor movement with known drivers | West 2026-03 |
| Low-confidence scenario with abstention | South 2026-05 |
| Sparse-history / newly launched KPI | NC-BOOST-01 2026-06 |
| Role-based security scenario | Role dropdown |
| Evidence: method, contribution, confidence, lineage | Panels 3–4 |
| LLM vs non-LLM breakdown | Panel 8 and the table above |
| Runtime telemetry | Panel 8 |

## Repo layout

```
kpi_contract.yaml           governed KPI semantics — the single source of metric truth
generate_data.py            deterministic dataset + counterfactual ground truth
engine/detect.py            baseline, prediction interval, materiality funnel
engine/crosskpi.py          cross-KPI pattern reasoning
engine/decompose.py         bridge and dimension decomposition
engine/diagnose.py          hypotheses, evidence, confidence, abstention
narrative/render.py         action chain, persona narratives, numeric guard   (LLM stage)
demo_app.py                 prototype UI
evaluate.py                 25-scenario evaluation
causalloop_prototype.py     self-contained Colab launcher
causalloop_evaluation.py    self-contained Colab evaluation run
data/ground_truth.json      planted answers the engine is scored against
```

## Limitations

Stated plainly, because a system that reports confidence should be honest about its own.

- **Detection recall is 60%.** Real movements under roughly 15% are not surfaced. Tunable, but
  currently a real gap.
- **No causal inference.** Every driver test is associational. We do not run
  difference-in-differences or any placebo test, and we do not claim causal identification —
  the engine reports contribution and evidence, not proven causation.
- **Confidence weights are hand-set and uncalibrated.** A 0.75 has not been shown to be right
  75% of the time. Calibrating against outcomes is the first thing we would do next.
- **Attribution shares differ from planted truth** even when absolute contributions are close,
  because a trailing baseline absorbs part of a movement that began before the period. The
  engine measures deviation from its own expectation, which is what any real system does.
- **Retrieval is TF-IDF, not dense embeddings.** At 195 documents this is adequate and it has
  the advantage of being inspectable, but it will not generalise to a large corpus.
- **Feedback adjusts stored driver priors; it does not retrain a model.**
- **CSV inputs stand in for warehouse tables.** The semantic contract is the seam that makes
  that swap mechanical rather than a rewrite.
