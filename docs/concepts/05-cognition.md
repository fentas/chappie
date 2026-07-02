# 5 · Cognition — decisions & dual-process

## What a decision is
The action space is small and explicit today: `Speak · Move · Manipulate · Attend ·
Rest · Noop`. Active agents each emit a `Proposal` (an action + a weight); consensus
tallies by action kind, the strongest kind wins, and the winning agent's utterance rides
along. "What to do" = which kind wins; "what to say" = the winner's utterance (a concept
label today; real text once the language layer lands).

`Attend` is the honest default for *not sure yet* — hesitation, gather more — and
curiosity biases exploration.

## Reflex is fine — until it's conflicted
Humans are largely reflexive too. So System 1 (fast, cheap consensus) is the default.
But **indecision should recruit thinking.** The signal already exists: consensus returns
`agreement` (winner weight ÷ total). Low agreement = a split coalition = the reflex is
unsure.

## Thinking = conflict-triggered escalation (System 2)
When `agreement` falls below a threshold, the brain **widens the coalition** (more
resident agents deliberate, lower participation floor) and re-votes — up to a cap. Cheap
reflex almost always; expensive thinking only when genuinely conflicted.

Two properties fall out for free:
- **Adaptive compute** — you pay to think in proportion to difficulty (inference-time
  scaling: "think longer on hard problems").
- **Thinking is tiring** — it wakes more agents, which costs more energy, so the agent
  sleeps sooner. Effort has a metabolic price, unforced.

The natural home for the *expensive reasoner* is a model-backed / language agent invoked
only on escalation — not on every reflex. See [Development](06-development.md) and the
language-layer plan.
