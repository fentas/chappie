# 6 · Development — growing up

Chappie is meant to *develop*, not just run. Capability, curriculum, and even the senses
come online over time.

## The life-cycle
Stages — **Infancy → Childhood → Adolescence → Adulthood** — widen the curriculum and
raise the bar. Progression is gated by competence (the benchmark steers development), so
the world only presents what the agent is ready for. A task focused on "language" simply
can't land on an infant — the world offers it once the agent grows into it.

## The embryo phase
The current synthetic world (`Sandbox`) is the **womb**: a simple, safe, fully-internal
environment to grow the *architecture* before exposing it to reality. It is meant to be
replaced.

## Growing senses, one at a time
`StimulusSource` is the seam for growing up. Real inputs are introduced **gradually**,
one modality at a time — first a microphone, then light/camera, then more — each a new
source feeding the same `Senses` encoders (which upgrade from stubs to real per-modality
encoders as each sense comes online). This mirrors real fetal development, where hearing
comes online before vision. Same loop, more sources.

## The examined life
An endless life is only meaningful if it leaves a trace and can be handed purpose:
- **Diary** — each sleep writes a day's entry (goal, mood, what it attended to, its
  strongest new association). The examined life, in Markdown.
- **Tasks** — a dropped goal biases attention and focus, and is recorded in the diary.
- **Snapshots** — the learnable self is serialized at sleep and resumed across restarts.
  Continuity of a life across time, not a fresh boot each run.

## Drives
Homeostasis, not a score, shapes the arc: **energy** depletes and forces sleep;
**curiosity** rises with unexplained surprise and biases exploration. Behavior emerges
from keeping internal variables in range — the primitive substrate a goal system grows on.

## Open question
What makes a resumed Chappie the *same* Chappie? The snapshot is its learned structure and
its place in a life — but identity across restarts is a real question, not a settled one.
See [Open Questions](07-open-questions.md).
