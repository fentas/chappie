# 4 · Sleep & Dreaming

The deepest area. Sleep is not bookkeeping — it is **the same cognition as waking,
pointed at itself**.

## The unifying insight
Wake and sleep run the *same loop*. Only the **stimulus source** differs:
- **Awake:** stimuli come from the world (`WorldSource`; real senses later).
- **Asleep:** stimuli come from the brain generating its own input — a `DreamSource`
  that replays and remixes memory.

`perceive → deliberate → consensus → learn` is identical. This single abstraction
(a `StimulusSource` trait) makes everything below fall out naturally.

## Reliving → weight, dismiss, or keep-uncertain
Each replayed memory is re-run through the **current** brain, and judged three ways:
- **Endorse** — the matured brain re-reaches the rewarded action with high agreement →
  strengthen, train on it, and *settle* it (lower its future replay priority).
- **Dismiss** — confidently judged low-value or inconsistent → prune it.
- **Deep uncertainty** — the brain genuinely can't decide (low agreement) → **keep it,
  and raise its priority.** Unresolved things resurface. (Zeigarnik / high-prediction-error
  instinct: keep chewing on what isn't settled.)

The weight comes from **who the brain has become**, not from who it was when the memory
formed. Old memories are re-appraised; their value can rise or fall over time.

## The critic is the brain itself
There is no external reward in a dream. The evaluator is the current model re-deriving
the memory: its **consensus agreement** and **consistency with the original outcome** are
the signal. Self-supervised consolidation — the brain re-labels its own past. (A richer
critic later: a dedicated evaluator agent, or the LLM judging its own dream.)

## Recombination — mixing memories
The dream samples memories from different days (correlated or not), **blends** their
query embeddings, and replays the blend. Coherent blends → a cross-experience
association (schema formation / integration): today gets linked to the past. Incoherent
blends are dismissed. This is where knowledge that *neither day alone contained* gets made.

## The fingerprint — content-addressable dreaming
Every episode carries a **fingerprint**: the activation pattern over agents ("which areas
fired"). The dream keeps a *current fingerprint* and retrieves memories whose fingerprint
is **similar** — content-addressable recall, i.e. pattern completion.

The key move: **the dream wanders.** Replaying a retrieved memory (plus chaos) *shifts*
the current fingerprint, so the next retrieval comes from a slightly different region,
which shifts it again. A **random walk over associative memory** — attractor dynamics,
the way dreams meander and free-association works. Not a metaphor: it's the retrieval rule.

## Valence flips generate weight
Track **valence** (reward sign) along the wander. When it crosses `+ → −` or `− → +`
between consecutive memories, that juxtaposition is salient → **forge a new connectome
link** between the two memories' dominant agents and weight the transition. A happy dream
sliding into a bad one *is* the brain discovering a bridge between two affective clusters.
Affective reconsolidation — and a genuine source of new structure.

## The chaos monkey
The dream never replays verbatim: noise on stimuli, occasional fully-random input, random
recombination of unrelated memories. Most is nonsense the critic dismisses — but
occasionally an odd combination is one the current brain finds *useful*. That's the
creative / insight function of dreaming (and, in ML terms, generative replay + exploration
noise). Seeded, so dreams stay **reproducible** — the benchmarks remain meaningful.

## The feedback loop — the brain steers the dream
The `DreamSource` is **directed, not uniform**: sample by the brain's own signals —
surprise, uncertainty, reward magnitude, current goal, and fingerprint similarity. Replay
what matters. And the brain's reaction shapes the next stimulus (lingering on the
unresolved, moving on from the settled). The simulation — dream *or* real — is a closed
loop with the brain, not a one-way feed. (In RL: prioritized experience replay. In brains:
targeted memory reactivation.)

## Daytime recurrence
High-priority *unresolved* memories occasionally surface into the **waking** stream too —
an intrusive thought / active recall — so they can be resolved with fresh context.

## Open questions
The exact critic; verbatim vs generative replay balance; how much chaos is creative vs
destabilizing; whether the valence machinery implies anything we should care about. See
[Open Questions](07-open-questions.md).
