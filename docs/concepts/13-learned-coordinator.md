# 13 · The Learned Coordinator — character, over time

[11](11-committee-vs-monolith.md) proved the coordinator is the whole gap: the specialists
are perfect, a *dumb* router loses. This is the follow-up — make the router **learned**, and
ask the two questions that matter:

1. Does a **learned** router close the gap — walk the committee up to the oracle ceiling?
2. Can the committee win **over time**, where the static monolith can't?

`cargo run -p chappie-burn --bin learned_coordinator --release`.

## Part 1 — a learned gate, and its learning curve
The fixed confidence-vote becomes a small trainable **gate**: `Linear(percept → one weight
per expert)`, softmaxed, mixing the frozen experts' logits per-sample. Only the gate trains
(experts and backbone frozen) — so this measures *exactly* whether learning to route recovers
the accuracy the naive vote threw away.

The interesting artifact is the **learning curve**: committee accuracy as the router develops,
from near-chance to — the claim — the oracle ceiling.

### Results
The router develops from near-chance to the ceiling in ~25 steps, on both slices:

| router (gate) training | clean | hard |
|---|---:|---:|
| step 0 (untrained) | 33.7% | 33.7% |
| step 25 | **100%** | **100%** |
| step 50 → 400 | 100% | 100% |
| *reference: monolith* | 100% | 100% |
| *reference: oracle* | 100% | 100% |

The committee starts at chance (random routing over 3 experts ≈ 33%) and the learned gate
**walks it straight up to the oracle ceiling** — matching the monolith. Confirmed: the gap
was entirely the coordinator, and a learned one closes it, fast. The "dumb router loses" of
[11](11-committee-vs-monolith.md) becomes "learned router wins" here.

## Part 2 — the temporal win: growth without forgetting
Out of the box the monolith wins ([11](11-committee-vs-monolith.md)); the committee's edge is
*temporal*. The sharpest case is **continual learning**: a new family of concepts arrives.

- The **monolith**, fine-tuned on the new family, **forgets** the old ones — catastrophic
  interference, because every weight is shared.
- The **committee grows a new expert** for the new family while the old experts stay
  **frozen** — so old knowledge is retained *by construction*. Only the router has to adapt.

### Results
A new family (social) arrives after the model already knows the other two:

| system | OLD (retain) | NEW (learn) |
|---|---:|---:|
| monolith — fine-tuned on the new family | **100% → 0%** | 100% |
| committee — grew a new expert | **100%** (frozen) | 100% |

Fine-tuning the monolith on the new family **erased the old ones — 100% to 0%, total
catastrophic forgetting.** Growing a new expert left the existing ones untouched, so the
committee **retained everything** while still learning the new family. This is the edge that
only appears *over time*: the same shared representation that makes the monolith win the
snapshot is exactly what makes it forget when the world changes.

## Why this is *character*
The gate isn't just "pick the right expert." It's a learned weighting over competing
responses — and once it takes **internal state** (mood, arousal) as input, that weighting
becomes *disposition*: whether to act or watch, cry or observe, shaped by experience. Two
gates trained on two life-histories are two characters over the same experts. The monolith
bakes one character in at training; the committee's is plastic and keeps developing. That is
the whole "bigger out of the box, but the committee over time" bet, made mechanical.

## Honest scope
- Toy task (9 concepts, tiny nets, `NdArray`/CPU) — this probes the *mechanism*.
- The gate here routes on the percept; feeding it `mood`/arousal (the character step) is the
  integration into the living brain, behind the [gatekeeper](12-gatekeeper.md)'s fast lane.
- Deterministic except Burn's weight init (not seeded by our RNG), so absolute numbers shift
  run-to-run; the shape (curve climbs to the oracle; monolith forgets, committee retains) is
  the result.
