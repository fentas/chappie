# 2 · Architecture — the cognitive loop

## One moment of life
```
perceive → attend/route → deliberate → consensus → act
        → record episode → [tired?] sleep → consolidate
```
Wake and sleep run the *same* loop; only the source of experience differs
(see [Sleep & Dreaming](04-sleep-and-dreaming.md)).

## Global workspace
The deliberation step is **Global Workspace Theory** as engineering, not a claim about
experience: the attended specialists post proposals to a shared workspace, read each
other, and a **weighted consensus** collapses them to one action. Broadcast + vote
avoids the O(N²) explosion of everyone-messaging-everyone, and it institutionalizes
diversity.

## Attention as compute placement
Every tick a scheduler ranks all agents by **priority** = relevance (cosine to the fused
query) + coupling to whatever's currently hot (shared priority → coalitions) + learned
reliability + hysteresis. It then places the top ones **hot (GPU)**, the working set
**warm (CPU)**, and the rest **cold (unloaded)**, under separate budgets. This is how a
population of hundreds could fit: only the attended few are ever resident. It mirrors how
the brain routes metabolic resources to attended processing.

## The connectome
A weighted graph links agents that fire together and are rewarded (Hebbian). It carries
**no data** — it's an affinity structure that biases routing and forms coalitions.
Bounded on purpose (see [Thesis](01-thesis.md)).

## Hemispheres
Two institutionalized processing styles for built-in diversity: **Left** =
sequential / linguistic / analytic / exploit-known; **Right** = holistic / spatial /
novelty-seeking / explore. A "corpus callosum" hands the lead to the right on novelty,
the left on the familiar.

## Everything is a seam
`Senses`, `Agent`, `World`, `Mind`, and (now) `StimulusSource` are traits. A stub, a
Burn network, or an LLM are interchangeable behind them — the loop is unchanged. That
seam discipline is what lets the system grow from embryo to model-backed without rewrites.
