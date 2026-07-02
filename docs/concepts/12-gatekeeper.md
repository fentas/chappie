# 12 · The Gatekeeper — deep memories & the fast lane

Before the deliberative coordinator ([11](11-committee-vs-monolith.md)) gets to weigh
experts, something faster has already fired. A **gatekeeper** sits at the very front,
labelling incoming against a small set of **deep memories** — and when the match is a
threat, it *reacts before the coordinator even runs*. This is the flinch, the catch, the
freeze: LeDoux's **low road**, the coarse subcortical shortcut that beats the cortex to the
punch because being slow about a snake is fatal.

## Deep memories
Not the episodic buffer — a separate, **low-capacity, high-priority** store, content-addressed
by **fingerprint** (the same activation-signature retrieval the dream loop uses). Two ways one
forms, and you need both:

- **Slow lane — overlearning.** The same pattern, with the same good reaction, again and
  again, gradually graduates into a reflex. The well-worn path. (Catching a ball you've caught
  ten thousand times.)
- **Fast lane — trauma.** A single event of extreme arousal (`|valence|` past a high
  threshold) burns straight in, **one-shot**. No repetition needed — flashbulb encoding. (The
  one time the ball hit your face.)

Deep memories **deepen** each time they're matched (repetition strengthens) and **fade** if
never used (decay in sleep). Capacity is small on purpose — the gate must stay fast and
uncluttered.

## The first door
At the front of perception, before the O(n) coordinator runs, the gatekeeper matches the
incoming fingerprint against its deep memories:

- **Fear match** (a deep memory with strongly negative valence, above the match threshold) →
  **reflex**: the burned-in reaction fires *immediately*, deliberation is skipped. Fast,
  cheap, specific — the ball is caught before you know it.
- **Any other strong match** → **prime**: a fast, strongly-weighted prior is injected into the
  workspace, so the coordinator's vote is already tilted before the experts finish bidding.
  Deep memory colours perception.

Properties, by design: **super fast** (a handful of cosine matches, not the full population),
**never overloaded** (hard capacity cap), **specific** (high match threshold — it fires on its
fingerprints, not on everything).

One subtlety that mattered: an **aversive** trauma burns a *protective* reaction (withdraw /
Move), **not a replay of the action that caused it**. A fear reflex is a hardwired flinch, not
a re-enactment of the mistake — encoding the failed action instead just repeats the harm.

## Demonstrated
Give the world a real threat — *failing to act on danger is traumatic* (a hard −1.0) — and the
mechanism fires end to end: a danger-failure **burns a protective fast-lane reflex** one-shot,
and the gatekeeper then **withdraws from danger reflexively (≈280 times over 8k ticks),
bypassing the coordinator entirely.** Fixing the reflex to encode the protective action rather
than the mistake moved reward 0.52 → 0.69; the benchmark is unmoved (recall 1.000, auc 0.950)
because the gate runs in *life*, not during evaluation. Deep memories settle at the capacity cap
(12) — a mix of overlearned good reactions and protective fear.

## Why it's the *first* door to the coordinator
The gatekeeper is not a competitor to the learned router — it's its **fast front end**. Most
input flows through to the coordinator to be deliberated and routed; a small, sharp set of
patterns are handled (or pre-labelled) before that, on the fast lane. The slow, deliberative,
character-forming router ([11](11-committee-vs-monolith.md)) handles the rest. Two speeds, one
pipeline: reflex when it must be instant, deliberation when there's time to think.

## Honest scope
- Fingerprints here are the query/percept signature; the reflex reaction is a burned stub
  action (a real motor program is later).
- The gate is a threshold rule, not yet learned — which is the point of the *next* step: the
  deliberative coordinator behind this door becomes a learned, developing router (character).
- Deep memories are transient state today (not yet in the snapshot).
