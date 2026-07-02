# 1 · Thesis — the bet

## The core wager
One large model is not the only path to capable cognition. A **population of small
specialists** — routed, made to reach consensus, and improved by a good
consolidation loop — might get somewhere interesting, *if* three things are right:

1. **Routing by relevance**, so the right specialists handle each moment.
2. **Diversity**, so their errors de-correlate (two genuinely different views beat
   N near-clones).
3. **A learning loop that consolidates**, so experience turns into lasting structure.

This is not "an ensemble." It's a brain-shaped system: senses, an attention/router,
a global workspace where specialists deliberate, memory across timescales, and a sleep
cycle that reshapes the whole thing.

## What we are *not* claiming
- Not AGI, not consciousness, not biological accuracy.
- Not that small-models-beat-big in general — only that an *orchestrated, developing*
  population is a path worth exploring on its own terms.

## What we *are* exploring
Whether cognition — attention, memory, judgment, the sense of a continuous life — can
be assembled from orchestrated small parts plus a strong learning/consolidation loop,
and what that assembly teaches us about the real thing.

## The staged method
Prove the architecture **cheaply first** (a deterministic, pure-Rust simulation with
stub "brains"), so the *shape* is right before any model is loaded. Then swap stubs for
real trainable models (Burn), then real inputs (mic, camera). Every region is a trait
seam; the cognitive loop never changes when a stub becomes a model.

The hard-won early lesson: **relevance must dominate routing.** An early bug let Hebbian
weights snowball until one cluster hijacked every stimulus — the MoE router-collapse /
load-imbalance failure mode. Bounded weights + a relevance floor fixed it. Diversity is
worthless if one voice drowns the rest.
