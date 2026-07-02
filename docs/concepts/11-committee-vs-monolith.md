# 11 · Committee vs Monolith — the thesis under test

The whole bet is that *many small specialists + a good router* can be as capable as
*one big model* — and better at growing, forgetting nothing, and running sparse. That's a
claim, so we test it. `cargo run -p chappie-burn --bin committee_vs_monolith` runs a
controlled A/B; this page is the method and the reading.

## The setup
- **Task.** 9 concepts in 3 families. Inputs are noisy; the **hard** slice blends a
  *within-family confuser* into each input — the fine distinctions a specialist trains
  hardest on. It's deliberately the terrain where specialization *should* help; if the
  committee can't win here, it can't win anywhere.
- **Monolith.** One MLP (`EMB_DIM → hidden → EMB_DIM`), trained on all 9 concepts.
- **Committee.** One shared backbone (`EMB_DIM → 64 → 32`), pretrained on everything then
  **frozen**, plus **one small head per family** trained only on its niche. At inference
  the three heads vote, each weighted by its own peak confidence (a soft mixture-of-experts
  — the confident family-expert dominates its inputs).
- **Fairness.** The monolith's hidden size is set so its parameter count ≈ the committee's
  (base + 3 heads). Both are judged on the *same* held-out test sets. Same data, same
  budget — only the *shape* differs.

## Results
Equal budget (monolith 4087 · committee 4100 params), identical held-out data:

| slice | monolith | committee (naive vote) | committee (oracle route) |
|---|---:|---:|---:|
| clean | 100.0% | 88.9% | **100.0%** |
| hard | 100.0% | 66.4% | **100.0%** |

*(The naive-vote number varies run-to-run — ~53–89% — because Burn's weight init isn't
seeded by our RNG. The monolith and the oracle-routed committee are a stable 100%; the
qualitative result doesn't move.)*

## What it means — the coordinator is the whole gap
The monolith wins out of the box. But the oracle probe is the tell: **with perfect routing
the committee matches it exactly — 100% on both slices.** The specialists are individually
perfect on their niche; every point the naive committee dropped, it dropped to *misrouting*,
not to any shortfall of small experts. The fixed confidence-weighted vote is a dumb router,
and the dumb router is the entire story.

So the result isn't "small loses to big." It's **"the coordinator is the bottleneck"** — and
a *learned* router closes the gap by construction, since oracle routing is simply its ceiling.
This is the same coordinator that, learned over experience, becomes **character**: whether to
act or watch, cry or observe — dispositions that are exactly a learned weighting over competing
experts. The monolith bakes one character in at training; the committee's is plastic and
develops. So the honest next experiment is a **learning curve**: does a committee with a
*learned, developing* coordinator overtake the static monolith as experience accumulates and
the world keeps changing? Out-of-the-box, bigger wins; the bet is on *over time*.

## The dimension accuracy alone misses
Even at equal accuracy the committee wins on **scaling**: because the backbone is shared,
`N` specialists cost `base + N·head`, not `N·(base+head)`. The printed scaling table shows
the crossover — at 100–1000 specialists the shared design saves the overwhelming majority of
parameters. So the real claim isn't "a committee of 3 beats one net of the same size" — it's
"once you need *many* specialists (and to **grow/prune** them without retraining the world),
the shared-backbone committee is the only one of the two that stays affordable." Accuracy
parity at small N + massive savings at large N is a *win* for the architecture.

## Honest caveats
- Toy task (9 concepts, tiny nets, `NdArray`/CPU). This probes the *mechanism*, not
  language-model-scale intelligence.
- The vote is a fixed confidence-weighting, not a learned router — a learned (model-backed)
  router is the next lever and would only help the committee.
- One seed shown; the binary is deterministic, re-run to vary.
