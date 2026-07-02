# 3 · Memory — Complementary Learning Systems

Chappie's memory follows McClelland's **Complementary Learning Systems**: a fast store
that captures specifics, and a slow store that distills regularities — with sleep as the
bridge between them.

## Four stores, four timescales
| store | timescale | form | in code |
|---|---|---|---|
| **Working** | seconds | fast, decaying heap | `WorkingMemory` (cap ~7, answers `RECALL_CUE`) |
| **Episodic** | a life | specific events | `Hippocampus` buffer + per-concept reservoir |
| **Semantic** | slow | prototypes / schemas | `Semantic` prototypes (novelty/surprise) |
| **Parametric** | slow | weights | Burn networks trained by sleep replay |

- **Non-parametric** (a "standard heap"): working + episodic + semantic. Fast to write,
  explicit, retrievable.
- **Parametric** (weights): the model that *improves over time*. Slow, distributed,
  earned by replay.

## The bridge is sleep
The fast episodic heap is **replayed during sleep to train the slow weights** — the
neocortical half of CLS. Without care this forgets: as the curriculum shifts, weights
drift off old concepts. **Interleaved replay** (a per-concept reservoir mixing old with
new) prevents that catastrophic forgetting. This is already load-bearing: it's why the
model's accuracy climbs to 100% *and holds* over a life, instead of rising then falling.

## Measuring memory
The **recall benchmark** is the instrument: show a cue, then ask "what did you just
see?". Without working memory it's chance; with it, ~100%. That single metric moved from
`0.000` to `1.000` at the exact commit working memory landed — a capability change you
can trace to its cause.

## Where this is going
Sleep is being upgraded from "replay + train" into full re-experiencing: reliving,
judging, recombining, and wandering through memory. That's [Sleep & Dreaming](04-sleep-and-dreaming.md).
