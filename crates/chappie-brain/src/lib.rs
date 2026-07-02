//! chappie-brain — the regions, wired into one global-workspace cognitive loop.
//!
//! ```text
//! perceive → attend → schedule (place agents GPU/CPU/cold) → deliberate
//!          → consensus → act → record episode → [tired?] sleep→consolidate
//! ```
//!
//! Every knob comes from [`Config`]; every region is a small struct behind a
//! clean seam. Swap `Senses` for real encoders, swap `StubAgent`s in the
//! `Cortex` for Burn/Ollama models, and this loop is unchanged.

use chappie_core::*;
use chappie_harness::{Agent, Decision, Harness, StubAgent};
use serde::{Deserialize, Serialize};

/// An episode's activation fingerprint: which agents fired, as a normalized
/// vector over the agent population. Similarity between fingerprints is how the
/// dream retrieves associated memories (content-addressable recall).
fn fingerprint(active: &[AgentId], n: usize) -> Vec<f32> {
    let mut f = vec![0.0f32; n];
    for &a in active {
        if (a as usize) < n {
            f[a as usize] = 1.0;
        }
    }
    normalize(&mut f);
    f
}

// ============================================================================
// Senses — encode raw stimuli into percepts and assign salience.
// ============================================================================

struct Senses;

impl Senses {
    fn encode(&self, stim: &Stimulus, semantic: &Semantic) -> Percept {
        let emb = stim.features.clone();
        let novelty = semantic.novelty(&emb);
        let danger = emb[concept_index("danger").unwrap()].max(0.0);
        // Salience rises with intensity, novelty, and threat — right-brain vigilance.
        let salience = (0.4 * stim.intensity + 0.4 * novelty + 0.4 * danger).min(1.0);
        Percept {
            modality: stim.modality,
            label: stim.label.clone(),
            embedding: emb,
            salience,
        }
    }
}

// ============================================================================
// Thalamus — attention gate, multimodal fusion, hemisphere arbitration.
// ============================================================================

struct Thalamus;

impl Thalamus {
    fn gate(&self, percepts: Vec<Percept>, floor: f32) -> Vec<Percept> {
        percepts.into_iter().filter(|p| p.salience >= floor).collect()
    }

    /// Salience-weighted fusion into one query embedding + its dominant concept.
    fn fuse(&self, percepts: &[Percept]) -> (Embedding, usize) {
        let mut q = vec![0.0f32; EMB_DIM];
        let mut tot = 0.0f32;
        for p in percepts {
            for i in 0..EMB_DIM {
                q[i] += p.embedding[i] * p.salience;
            }
            tot += p.salience;
        }
        if tot > 0.0 {
            for x in q.iter_mut() {
                *x /= tot;
            }
        }
        let dom = argmax(&q);
        normalize(&mut q);
        (q, dom)
    }

    /// The corpus-callosum call: novelty/curiosity hands the lead to the right
    /// (explore) hemisphere; the familiar stays with the left (exploit).
    fn lead(&self, surprise: f32, curiosity: f32, threshold: f32) -> Hemisphere {
        if surprise + curiosity > threshold {
            Hemisphere::Right
        } else {
            Hemisphere::Left
        }
    }
}

// ============================================================================
// Semantic memory — prototypes that let the brain measure surprise.
// ============================================================================

struct Semantic {
    protos: Vec<Embedding>,
}

impl Semantic {
    fn novelty(&self, e: &[f32]) -> f32 {
        if self.protos.is_empty() {
            return 1.0;
        }
        let best = self.protos.iter().map(|p| cosine(p, e)).fold(f32::MIN, f32::max);
        (1.0 - best).clamp(0.0, 1.0)
    }

    fn learn(&mut self, e: &Embedding) -> bool {
        let mut bi = usize::MAX;
        let mut bs = f32::MIN;
        for (i, p) in self.protos.iter().enumerate() {
            let s = cosine(p, e);
            if s > bs {
                bs = s;
                bi = i;
            }
        }
        if bs > 0.85 && bi != usize::MAX {
            for i in 0..EMB_DIM {
                self.protos[bi][i] = 0.9 * self.protos[bi][i] + 0.1 * e[i];
            }
            normalize(&mut self.protos[bi]);
            false
        } else if self.protos.len() < 64 {
            self.protos.push(e.clone());
            true
        } else {
            false
        }
    }
}

// ============================================================================
// Hippocampus — the rolling episodic buffer sleep replays.
// ============================================================================

struct Hippocampus {
    buf: Vec<Episode>,
    cap: usize,
}

impl Hippocampus {
    fn record(&mut self, e: Episode) {
        if self.buf.len() >= self.cap {
            self.buf.remove(0);
        }
        self.buf.push(e);
    }
}

// ============================================================================
// Vitals — homeostatic drives that shape a life.
// ============================================================================

struct Vitals {
    curiosity: f32,
    age_ticks: u64,
    /// Reconciliation pressure: rises with encoded surprise + time; sleep resets it.
    pressure: f32,
    /// Ticks since last sleep — the "day" whose length the rhythm targets.
    ticks_awake: u64,
    /// Encoding difficulty: how surprising an experience must be to be stored.
    enc_threshold: f32,
    /// Boredom: rises with monotony, falls with novelty; high → the mind wanders.
    boredom: f32,
}

impl Vitals {
    fn tick(&mut self) {
        self.age_ticks += 1;
        self.ticks_awake += 1;
    }
    fn accumulate(&mut self, amount: f32) {
        self.pressure += amount;
    }
    fn feel(&mut self, reward: f32, surprise: f32, gain: f32, decay: f32) {
        self.curiosity = (self.curiosity + gain * surprise - decay * reward.max(0.0)).clamp(0.0, 1.0);
    }
    /// Wake: measure the day just lived, adapt the encoding threshold toward the
    /// target day length (Bitcoin-style difficulty), and reset the buffer.
    fn reconcile(&mut self, target_day: f32, gain: f32) -> u64 {
        let day = self.ticks_awake;
        let err = (target_day - day as f32) / target_day.max(1.0); // >0 ⇒ day too short
        self.enc_threshold = (self.enc_threshold + gain * err).clamp(0.05, 0.9);
        self.pressure = 0.0;
        self.ticks_awake = 0;
        day
    }
}

// ============================================================================
// Working memory — the short-term, fast, *decaying* heap (non-parametric).
//
// Complements the long-term stores: where the Hippocampus is the replay buffer
// and the Cortex holds slow weights, this is seconds-scale scratch space. A
// "recall" cue is answered from here; without it, the recall benchmark is chance.
// ============================================================================

struct WorkingMemory {
    slots: Vec<(usize, f32)>, // (concept index, strength)
    cap: usize,
    decay: f32,
}

impl WorkingMemory {
    fn new() -> Self {
        WorkingMemory { slots: Vec::new(), cap: 7, decay: 0.6 }
    }
    /// Age every trace; forget the faint ones.
    fn decay(&mut self) {
        for s in self.slots.iter_mut() {
            s.1 *= self.decay;
        }
        self.slots.retain(|s| s.1 > 0.05);
    }
    /// Lay down (or refresh) a trace of a just-perceived concept.
    fn push(&mut self, concept: usize, strength: f32) {
        if let Some(slot) = self.slots.iter_mut().find(|s| s.0 == concept) {
            slot.1 = slot.1.max(strength);
        } else {
            self.slots.push((concept, strength));
        }
        if self.slots.len() > self.cap {
            self.slots
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            self.slots.truncate(self.cap);
        }
    }
    /// The freshest / strongest trace — "what did I just perceive?"
    fn recall(&self) -> Option<usize> {
        self.slots
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|s| s.0)
    }
}

// ============================================================================
// Trace — a peek at one tick, for the CLI narration.
// ============================================================================

#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub tick: u64,
    pub dominant: String,
    pub lead: &'static str,
    pub surprise: f32,
    pub salient: Vec<(String, f32)>,
    pub active: Vec<(String, &'static str)>, // (name, tier)
    pub proposals: Vec<(String, String, String, f32)>, // agent, kind, utterance, weight
    pub decision: String,
    pub agreement: f32,
}

struct Pending {
    episode: Episode,
    winners: Vec<AgentId>,
    perceptual_encode: bool,
}

// ============================================================================
// Snapshot — persistent learnable state (serialized at sleep, loaded on start).
// ============================================================================

/// A resumable slice of a life: what it learned (connectome + reliabilities +
/// semantic prototypes) plus where it is (stage, day, drives, RNG). Agent net
/// weights (Burn) are not captured yet — a follow-on.
#[derive(Serialize, Deserialize)]
pub struct Snapshot {
    pub sleeps: u64,
    pub day_start_tick: u64,
    pub thinks: u64,
    pub stage: String,
    pub goal: Option<String>,
    pub pressure: f32,
    pub ticks_awake: u64,
    pub enc_threshold: f32,
    pub curiosity: f32,
    pub age_ticks: u64,
    pub rng_state: u64,
    pub reliabilities: Vec<f32>,
    pub connectome: Vec<f32>,
    pub semantic: Vec<Vec<f32>>,
    pub working: Vec<(usize, f32)>,
    pub recent_rewards: Vec<f32>,
    pub recruited_concepts: Vec<usize>,
    pub recruited: u64,
    pub pruned: u64,
}

// ============================================================================
// Brain — the whole thing.
// ============================================================================

/// A deep memory: a fast-lane, fingerprint-addressed engram — a burned-in reaction
/// that the gatekeeper matches *before* the coordinator deliberates.
struct DeepMemory {
    fingerprint: Embedding,
    action: ActionKind,
    utterance: String,
    valence: f32,
    strength: f32,
    hits: u32,
}

pub struct Brain {
    senses: Senses,
    thalamus: Thalamus,
    cortex: Harness,
    hippocampus: Hippocampus,
    semantic: Semantic,
    vitals: Vitals,
    working: WorkingMemory,
    rng: Rng,
    cfg: Config,
    stage: String,
    recent_rewards: Vec<f32>,
    sleeps: u64,
    thinks: u64,
    day_start_tick: u64,
    goal: Option<String>,
    /// Running felt valence in [-1,1]; negative = distress (drives co-regulation).
    mood: f32,
    /// Running reward expectation, for reward-prediction-error-gated encoding.
    reward_expectation: f32,
    /// Per-concept accumulated conflict/failure — drives need-driven recruitment.
    gaps: Vec<f32>,
    recruited: u64,
    pruned: u64,
    /// Concepts that have been recruited (so a snapshot can re-grow the population).
    recruited_concepts: Vec<usize>,
    /// The gatekeeper's low-capacity deep memories, slow-lane repetition counters,
    /// and how many reflexes have fired on the fast lane.
    deep: Vec<DeepMemory>,
    reflex_count: Vec<u32>,
    reflexes: u64,
    pending: Option<Pending>,
    last_trace: Trace,
}

impl Brain {
    pub fn new(agents: Vec<Box<dyn Agent>>, cfg: Config) -> Self {
        let cortex = Harness::new(agents, &cfg);
        let rng = Rng::new(cfg.seed);
        let cap = cfg.sleep.replay_cap.max(1);
        Brain {
            senses: Senses,
            thalamus: Thalamus,
            cortex,
            hippocampus: Hippocampus { buf: Vec::new(), cap },
            semantic: Semantic { protos: Vec::new() },
            vitals: Vitals {
                curiosity: 0.3,
                age_ticks: 0,
                pressure: 0.0,
                ticks_awake: 0,
                enc_threshold: 0.15,
                boredom: 0.0,
            },
            working: WorkingMemory::new(),
            rng,
            cfg,
            stage: "infancy".to_string(),
            recent_rewards: Vec::new(),
            sleeps: 0,
            thinks: 0,
            day_start_tick: 0,
            goal: None,
            mood: 0.0,
            reward_expectation: 0.5,
            gaps: vec![0.0; EMB_DIM],
            recruited: 0,
            pruned: 0,
            recruited_concepts: Vec::new(),
            deep: Vec::new(),
            reflex_count: vec![0; EMB_DIM],
            reflexes: 0,
            pending: None,
            last_trace: Trace::default(),
        }
    }

    pub fn set_stage(&mut self, stage: &str) {
        self.stage = stage.to_string();
    }

    /// Give (or clear) the current goal/task — recorded in the diary and used to
    /// bias what the world presents.
    pub fn set_goal(&mut self, goal: Option<String>) {
        self.goal = goal;
    }

    /// Grow a new specialist agent for an under-covered, chronically-failing concept.
    fn recruit_specialist(&mut self, c: usize) {
        let concept = CONCEPTS[c];
        let id = self.cortex.len() as AgentId;
        let (hemi, kind) = match concept {
            "danger" => (Hemisphere::Right, ActionKind::Move),
            "language" | "logical" | "numeric" => (Hemisphere::Left, ActionKind::Speak),
            _ => (Hemisphere::Right, ActionKind::Speak),
        };
        let agent = StubAgent::new(
            id,
            format!("Grown-{concept}"),
            hemi,
            embed(&[(concept, 1.0)]),
            kind,
            concept,
            160.0,
        )
        .boxed();
        self.cortex.add_agent(agent);
        self.recruited += 1;
        self.recruited_concepts.push(c);
    }

    /// Fast pre-attentive match: the best deep memory for a percept (idx, similarity).
    fn gatekeeper_match(&self, q: &Embedding) -> Option<(usize, f32)> {
        let mut best: Option<usize> = None;
        let mut bs = 0.0f32;
        for (i, d) in self.deep.iter().enumerate() {
            let s = cosine(q, &d.fingerprint);
            if s > bs {
                bs = s;
                best = Some(i);
            }
        }
        best.map(|i| (i, bs))
    }

    /// Burn a deep memory (fast lane = trauma, slow lane = overlearning). Merges into a
    /// near-identical existing one; else inserts, evicting the weakest at capacity.
    fn burn_deep(&mut self, fp: &Embedding, action: ActionKind, utterance: &str, valence: f32) {
        if let Some((i, sim)) = self.gatekeeper_match(fp) {
            if sim > self.cfg.gatekeeper.match_threshold {
                self.deep[i].strength += 0.5;
                self.deep[i].valence = 0.5 * self.deep[i].valence + 0.5 * valence;
                return;
            }
        }
        if self.deep.len() >= self.cfg.gatekeeper.capacity {
            if let Some((wi, _)) = self.deep.iter().enumerate().min_by(|a, b| {
                a.1.strength.partial_cmp(&b.1.strength).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                self.deep.remove(wi);
            }
        }
        self.deep.push(DeepMemory {
            fingerprint: fp.clone(),
            action,
            utterance: utterance.to_string(),
            valence,
            strength: 1.0,
            hits: 0,
        });
    }

    /// The most deeply-unresolved memory (priority raised above 1.0 by repeated
    /// uncertainty), if any — the thing most likely to intrude on waking.
    fn most_unresolved(&self) -> Option<usize> {
        let mut best = None;
        let mut bestp = 1.0f32;
        for (i, e) in self.hippocampus.buf.iter().enumerate() {
            if e.priority > bestp {
                bestp = e.priority;
                best = Some(i);
            }
        }
        best
    }

    /// Relive one memory: re-derive a decision on its query with the *current*
    /// brain — the same schedule→deliberate→consensus loop as waking (including
    /// dual-process escalation). No perception, no energy, no new episode.
    fn dream_tick(&mut self, query: &Embedding) -> (Decision, Vec<AgentId>) {
        let dom = argmax(query);
        let surprise = self.semantic.novelty(query);
        let lead =
            self.thalamus
                .lead(surprise, self.vitals.curiosity, self.cfg.hemisphere.novelty_threshold);
        let mut active = self.cortex.schedule(query, &mut self.rng);
        let proposals =
            self.cortex
                .deliberate(query, dom, lead, self.vitals.curiosity, &mut self.rng);
        let mut decision = self.cortex.consensus(&proposals);
        let mut esc = 0;
        while decision.agreement < self.cfg.thinking.agreement_threshold
            && esc < self.cfg.thinking.max_escalations
        {
            active = self.cortex.widen_participants(
                self.cfg.thinking.widen_participants * (esc + 1),
                self.cfg.thinking.widen_floor_mult,
            );
            let wider =
                self.cortex
                    .deliberate(query, dom, lead, self.vitals.curiosity, &mut self.rng);
            decision = self.cortex.consensus(&wider);
            esc += 1;
        }
        self.cortex.end_tick();
        (decision, active)
    }

    pub fn trace(&self) -> Trace {
        self.last_trace.clone()
    }

    pub fn cortex(&self) -> &Harness {
        &self.cortex
    }

    /// Capture the resumable learnable state.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            sleeps: self.sleeps,
            day_start_tick: self.day_start_tick,
            thinks: self.thinks,
            stage: self.stage.clone(),
            goal: self.goal.clone(),
            pressure: self.vitals.pressure,
            ticks_awake: self.vitals.ticks_awake,
            enc_threshold: self.vitals.enc_threshold,
            curiosity: self.vitals.curiosity,
            age_ticks: self.vitals.age_ticks,
            rng_state: self.rng.state(),
            reliabilities: self.cortex.export_reliabilities(),
            connectome: self.cortex.connectome_weights(),
            semantic: self.semantic.protos.clone(),
            working: self.working.slots.clone(),
            recent_rewards: self.recent_rewards.clone(),
            recruited_concepts: self.recruited_concepts.clone(),
            recruited: self.recruited,
            pruned: self.pruned,
        }
    }

    /// Overlay a snapshot onto a freshly-built brain (same population).
    pub fn restore(&mut self, s: Snapshot) {
        // Re-grow the population to match the snapshot before overlaying weights.
        for &c in &s.recruited_concepts {
            self.recruit_specialist(c);
        }
        self.recruited = s.recruited;
        self.pruned = s.pruned;
        self.sleeps = s.sleeps;
        self.day_start_tick = s.day_start_tick;
        self.thinks = s.thinks;
        self.stage = s.stage;
        self.goal = s.goal;
        self.vitals.pressure = s.pressure;
        self.vitals.ticks_awake = s.ticks_awake;
        self.vitals.enc_threshold = s.enc_threshold;
        self.vitals.curiosity = s.curiosity;
        self.vitals.age_ticks = s.age_ticks;
        self.rng.set_state(s.rng_state);
        self.cortex.import_reliabilities(&s.reliabilities);
        self.cortex.set_connectome_weights(&s.connectome);
        self.semantic.protos = s.semantic;
        self.working.slots = s.working;
        self.recent_rewards = s.recent_rewards;
    }

    pub fn save_snapshot(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string(&self.snapshot())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load a snapshot if the file exists and parses; returns whether it did.
    pub fn try_load_snapshot(&mut self, path: &str) -> bool {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|txt| serde_json::from_str::<Snapshot>(&txt).ok())
            .map(|snap| self.restore(snap))
            .is_some()
    }
}

impl Mind for Brain {
    fn perceive_act(&mut self, stimuli: &[Stimulus], learn: bool) -> Action {
        // 0. A recall cue is answered straight from short-term working memory —
        //    no perception, no cortex. This is what the recall benchmark tests.
        if stimuli.iter().any(|s| s.label == RECALL_CUE) {
            self.working.decay();
            let utter = self.working.recall().map(|c| CONCEPTS[c].to_string());
            self.cortex.end_tick();
            return Action {
                kind: if utter.is_some() { ActionKind::Speak } else { ActionKind::Noop },
                utterance: utter.unwrap_or_default(),
                target: vec![0.0; EMB_DIM],
                confidence: 0.6,
            };
        }

        // Boredom → the mind wanders inward (a daydream). Internal generation: it
        // relives a memory (an unresolved one first, else free-associates) — but it
        // does NOT store a real event and does NOT take the tick's action. Reality
        // stays reality. It fires only while bored, and novelty resets boredom, so
        // real input always preempts it (no merging of the two worlds → no hallucination).
        if learn
            && self.vitals.boredom > self.cfg.vitals.bored_threshold
            && !self.hippocampus.buf.is_empty()
        {
            let idx = self
                .most_unresolved()
                .unwrap_or_else(|| self.rng.next_range(self.hippocampus.buf.len()));
            let q = self.hippocampus.buf[idx].query.clone();
            let (decision, _) = self.dream_tick(&q);
            if decision.agreement >= self.cfg.sleep.uncertain_threshold {
                self.hippocampus.buf[idx].priority *= 0.6; // settled with fresh context
            }
            self.vitals.boredom *= 0.5; // the wander scratched the itch
        }

        // 1. Senses → percepts.
        let percepts: Vec<Percept> = stimuli
            .iter()
            .map(|s| self.senses.encode(s, &self.semantic))
            .collect();

        // 2. Attention gate.
        let gated = self.thalamus.gate(percepts, self.cfg.attention.floor);
        if gated.is_empty() {
            self.cortex.end_tick();
            return Action::noop();
        }

        // 3. Fuse → query + dominant; surprise; hemisphere lead.
        let (query, dom) = self.thalamus.fuse(&gated);

        // Lay down a short-term trace of what we just perceived (ages each tick).
        self.working.decay();
        self.working.push(dom, 1.0);

        let surprise = self.semantic.novelty(&query);
        let lead = self.thalamus.lead(surprise, self.vitals.curiosity, self.cfg.hemisphere.novelty_threshold);

        // FIRST DOOR — the gatekeeper's fast lane. Match the percept against deep
        // memories *before* the coordinator runs. A fear match reacts instantly
        // (bypassing all deliberation); any other strong match primes the workspace.
        let mut prime: Option<(ActionKind, String, f32)> = None;
        if learn {
            if let Some((i, sim)) = self.gatekeeper_match(&query) {
                if sim > self.cfg.gatekeeper.match_threshold {
                    self.deep[i].hits += 1;
                    self.deep[i].strength = (self.deep[i].strength + 0.05).min(3.0);
                    if self.deep[i].valence < self.cfg.gatekeeper.fear_threshold {
                        // Reflex: the burned-in reaction fires now — no deliberation.
                        self.reflexes += 1;
                        self.vitals.tick();
                        let action = Action {
                            kind: self.deep[i].action,
                            utterance: self.deep[i].utterance.clone(),
                            target: query.clone(),
                            confidence: sim,
                        };
                        let episode = Episode {
                            tick: self.vitals.age_ticks,
                            stage: self.stage.clone(),
                            query: query.clone(),
                            dominant: CONCEPTS[dom].to_string(),
                            decision: action.clone(),
                            active_agents: Vec::new(),
                            reward: 0.0,
                            surprise,
                            priority: 1.0,
                        };
                        self.pending = Some(Pending {
                            episode,
                            winners: Vec::new(),
                            perceptual_encode: surprise > self.vitals.enc_threshold,
                        });
                        self.cortex.end_tick();
                        return action;
                    }
                    prime = Some((self.deep[i].action, self.deep[i].utterance.clone(), sim));
                }
            }
        }

        // 4. Schedule: rank by priority, place agents on GPU/CPU/cold.
        let mut active = self.cortex.schedule(&query, &mut self.rng);

        // 5. Deliberate → consensus (System 1: the fast reflex).
        let mut proposals = self
            .cortex
            .deliberate(&query, dom, lead, self.vitals.curiosity, &mut self.rng);
        // Gatekeeper prime: a deep memory speaks fast and loud in the workspace.
        if let Some((kind, utter, sim)) = prime {
            proposals.push(Proposal {
                agent: u32::MAX,
                agent_name: "Gatekeeper".to_string(),
                hemisphere: Hemisphere::Right,
                action: Action { kind, utterance: utter, target: query.clone(), confidence: sim },
                weight: sim * self.cfg.gatekeeper.prime_boost,
                rationale: "deep-memory prime".to_string(),
            });
        }
        let mut decision = self.cortex.consensus(&proposals);
        // The gatekeeper is not an agent — drop its pseudo-id from the winners.
        decision.winners.retain(|&w| (w as usize) < self.cortex.len());

        // 5b. System 2: a split coalition (low agreement) means the reflex is
        //     unsure — so THINK. Widen the coalition and re-deliberate until we're
        //     decisive or hit the escalation cap. Cheap reflex most of the time;
        //     expensive thinking only when genuinely conflicted.
        let mut escalations = 0;
        while decision.agreement < self.cfg.thinking.agreement_threshold
            && escalations < self.cfg.thinking.max_escalations
        {
            let extra = self.cfg.thinking.widen_participants * (escalations + 1);
            active = self.cortex.widen_participants(extra, self.cfg.thinking.widen_floor_mult);
            let wider = self
                .cortex
                .deliberate(&query, dom, lead, self.vitals.curiosity, &mut self.rng);
            decision = self.cortex.consensus(&wider);
            escalations += 1;
        }
        if escalations > 0 {
            self.thinks += 1;
        }

        // 6. Trace.
        self.last_trace = Trace {
            tick: self.vitals.age_ticks,
            dominant: CONCEPTS[dom].to_string(),
            lead: if lead == Hemisphere::Left { "L" } else { "R" },
            surprise,
            salient: gated.iter().map(|p| (p.label.clone(), p.salience)).collect(),
            active: active
                .iter()
                .map(|&id| (self.cortex.name(id).to_string(), self.cortex.placement_of(id).tag()))
                .collect(),
            proposals: proposals
                .iter()
                .map(|p| {
                    (
                        p.agent_name.clone(),
                        p.action.kind.name().to_string(),
                        p.action.utterance.clone(),
                        p.weight,
                    )
                })
                .collect(),
            decision: format!(
                "{}{}",
                decision.action.kind.name(),
                if decision.action.utterance.is_empty() {
                    String::new()
                } else {
                    format!(" \"{}\"", decision.action.utterance)
                }
            ),
            agreement: decision.agreement,
        };

        // 7. Learning-side effects (skipped during pure-inference eval).
        if learn {
            self.semantic.learn(&query);
            self.vitals.tick();
            self.vitals.accumulate(self.cfg.vitals.time_fatigue);
            // Encoding gate (perceptual half): is this surprising enough to matter?
            // The threshold rises with maturity, so a familiar world is mostly let
            // pass. The OUTCOME half (reward-prediction-error) is judged in reward().
            let perceptual_encode = surprise > self.vitals.enc_threshold;
            // Boredom falls when something novel grabs attention, rises with monotony.
            if perceptual_encode {
                self.vitals.boredom = (self.vitals.boredom - self.cfg.vitals.boredom_gain).max(0.0);
            } else {
                self.vitals.boredom = (self.vitals.boredom + self.cfg.vitals.boredom_gain).min(1.0);
            }
            // Build the candidate memory; whether it is actually stored is decided in
            // reward(), which also knows the outcome.
            let episode = Episode {
                tick: self.vitals.age_ticks,
                stage: self.stage.clone(),
                query,
                dominant: CONCEPTS[dom].to_string(),
                decision: decision.action.clone(),
                active_agents: active,
                reward: 0.0,
                surprise,
                priority: 1.0,
            };
            self.pending = Some(Pending {
                episode,
                winners: decision.winners.clone(),
                perceptual_encode,
            });
        }

        self.cortex.end_tick();
        decision.action
    }

    fn reward(&mut self, r: f32) {
        self.recent_rewards.push(r);
        if self.recent_rewards.len() > 512 {
            self.recent_rewards.remove(0);
        }
        if let Some(mut p) = self.pending.take() {
            let surprise = p.episode.surprise;

            // Self-soothing: a coherent/harmonic self-expression (high consensus
            // agreement) is intrinsically a little positive; a dissonant one negative.
            let self_harmony =
                self.cfg.vitals.self_soothing * (p.episode.decision.confidence - 0.5);
            // Felt valence = extrinsic (the world's response-harmony) + self-harmony,
            // then co-regulation: when distressed, positive/harmonic input soothes more.
            let mut valence = r + self_harmony;
            let distress = (-self.mood).max(0.0);
            if valence > 0.0 {
                valence *= 1.0 + self.cfg.vitals.coregulation_gain * distress;
            }
            valence = valence.clamp(-1.5, 1.5);
            self.mood = (0.9 * self.mood + 0.1 * valence).clamp(-1.0, 1.0);

            // Deposit a "gap" against the dominant concept when the moment was
            // conflicted (low self-agreement) or went badly (negative valence).
            if let Some(c) = concept_index(&p.episode.dominant) {
                let conflict =
                    (1.0 - p.episode.decision.confidence).max(0.0) + (-valence).max(0.0);
                self.gaps[c] += conflict;
            }
            for g in self.gaps.iter_mut() {
                *g *= 0.999;
            }

            // Deep-memory formation. FAST lane: a traumatic (very-high-|valence|)
            // event burns a deep memory in one shot. SLOW lane: a familiar, reliably-
            // good reaction, repeated enough, graduates into an overlearned reflex.
            if valence.abs() > self.cfg.gatekeeper.trauma_threshold {
                // An aversive trauma burns a PROTECTIVE reflex (withdraw / Move), not a
                // replay of the action that caused it; a positive one keeps what worked.
                let (kind, utter) = if valence < 0.0 {
                    (ActionKind::Move, String::new())
                } else {
                    (p.episode.decision.kind, p.episode.decision.utterance.clone())
                };
                self.burn_deep(&p.episode.query, kind, &utter, valence);
            } else if let Some(c) = concept_index(&p.episode.dominant) {
                if surprise < self.vitals.enc_threshold && valence > 0.3 {
                    self.reflex_count[c] += 1;
                    if self.reflex_count[c] >= self.cfg.gatekeeper.slow_reps {
                        self.reflex_count[c] = 0;
                        self.burn_deep(
                            &p.episode.query,
                            p.episode.decision.kind,
                            &p.episode.decision.utterance,
                            valence,
                        );
                    }
                }
            }

            // Reward-prediction-error: an unexpectedly good/bad outcome is worth
            // encoding even for a perceptually dull, familiar stimulus.
            let rpe = (r - self.reward_expectation).abs();
            self.reward_expectation = 0.95 * self.reward_expectation + 0.05 * r;

            self.vitals.feel(
                r,
                surprise,
                self.cfg.vitals.curiosity_gain,
                self.cfg.vitals.curiosity_reward_decay,
            );
            // Learn from the *felt* valence (co-regulated + self-harmony), not raw reward.
            self.cortex.reinforce(&p.winners, &p.episode.active_agents, valence);

            if p.perceptual_encode || rpe > self.cfg.sleep.rpe_threshold {
                p.episode.reward = valence;
                self.vitals
                    .accumulate(self.cfg.vitals.surprise_weight * surprise.max(rpe));
                self.hippocampus.record(p.episode);
            }
        }
    }

    fn tired(&self) -> bool {
        self.vitals.pressure > self.cfg.vitals.pressure_capacity
    }

    fn sleep(&mut self) -> DreamLog {
        let episodes = self.hippocampus.buf.clone();

        // Dream: relive sampled memories through the *current* brain, and let the
        // matured brain JUDGE each one three ways — endorse (re-affirmed →
        // strengthen + settle), dismiss (confidently low-value → prune), or
        // keep-uncertain (still can't decide → raise priority so it resurfaces).
        let dream_len = self.cfg.sleep.dream_len;
        let unc = self.cfg.sleep.uncertain_threshold;
        let n_agents = self.cortex.len().max(1);

        // The dream's current fingerprint starts as today's activation pattern
        // (which agents dominated the day) and drifts as the dream wanders.
        let mut current_fp = vec![0.0f32; n_agents];
        for e in episodes.iter().filter(|e| e.tick >= self.day_start_tick) {
            for &a in &e.active_agents {
                if (a as usize) < n_agents {
                    current_fp[a as usize] += 1.0;
                }
            }
        }
        normalize(&mut current_fp);

        let recomb = self.cfg.sleep.recombine_prob;
        let mut endorsed = 0usize;
        let mut dismissed = 0usize;
        let mut kept = 0usize;
        let mut flips = 0usize;
        let mut prev_val = 0.0f32;
        let mut prev_winners: Vec<AgentId> = Vec::new();
        for _ in 0..dream_len {
            let n = self.hippocampus.buf.len();
            if n == 0 {
                break;
            }
            // Retrieve the memory most similar to the current fingerprint
            // (× priority, + chaos) — content-addressable recall.
            let mut idx = 0usize;
            let mut best = f32::MIN;
            for (i, e) in self.hippocampus.buf.iter().enumerate() {
                let sim = cosine(&current_fp, &fingerprint(&e.active_agents, n_agents));
                let score = sim * e.priority + 0.2 * self.rng.next_gauss();
                if score > best {
                    best = score;
                    idx = i;
                }
            }
            let ep = self.hippocampus.buf[idx].clone();

            // Recombination: occasionally blend this memory with another — the
            // chaos monkey turned generative (a creative mix across experiences).
            let dream_query = if self.rng.next_f32() < recomb && n > 1 {
                let j = self.rng.next_range(n);
                let other = self.hippocampus.buf[j].query.clone();
                let mut blend = ep.query.clone();
                for k in 0..blend.len().min(other.len()) {
                    blend[k] = 0.6 * blend[k] + 0.4 * other[k] + 0.03 * self.rng.next_gauss();
                }
                normalize(&mut blend);
                blend
            } else {
                ep.query.clone()
            };

            let (decision, _active) = self.dream_tick(&dream_query);
            let matches = decision.action.kind == ep.decision.kind;
            if decision.agreement < unc {
                kept += 1;
                self.hippocampus.buf[idx].priority =
                    (self.hippocampus.buf[idx].priority * 1.5).min(4.0);
            } else if ep.reward > 0.0 && matches {
                endorsed += 1;
                self.cortex.endorse(&decision.winners, ep.reward);
                self.hippocampus.buf[idx].priority *= 0.7;
            } else if ep.reward <= 0.0 {
                dismissed += 1;
                self.hippocampus.buf[idx].priority = 0.0;
            }

            // Valence flip: this memory's sign is opposite the previous one — the
            // dream just crossed between affective clusters. Forge a link between
            // the two coalitions (a good/bad association neither day held alone).
            if prev_val * ep.reward < 0.0
                && !prev_winners.is_empty()
                && !decision.winners.is_empty()
            {
                flips += 1;
                self.cortex.associate(&prev_winners, &decision.winners, 1.0);
            }
            prev_val = ep.reward;
            prev_winners = decision.winners.clone();
            // Drift the fingerprint toward what we just relived (+ chaos) — the
            // wander: the next retrieval comes from a nearby region of memory.
            let fp = fingerprint(&ep.active_agents, n_agents);
            for k in 0..n_agents {
                current_fp[k] = 0.6 * current_fp[k] + 0.4 * fp[k] + 0.04 * self.rng.next_gauss();
            }
            normalize(&mut current_fp);
        }
        self.hippocampus.buf.retain(|e| e.priority > 0.05);

        // Offline weight training + forgetting.
        self.cortex.consolidate(&episodes, &mut self.rng);

        let mut new_protos = 0usize;
        for e in &episodes {
            if e.reward > 0.0 && self.semantic.learn(&e.query) {
                new_protos += 1;
            }
        }

        // Summarize the day just ended (episodes since the previous sleep).
        let now = self.vitals.age_ticks;
        let mut counts = [0u32; EMB_DIM];
        let mut rsum = 0.0f32;
        let mut rn = 0u32;
        for e in episodes.iter().filter(|e| e.tick >= self.day_start_tick) {
            if let Some(c) = concept_index(&e.dominant) {
                counts[c] += 1;
            }
            rsum += e.reward;
            rn += 1;
        }
        let mut concept_counts: Vec<(String, u32)> = CONCEPTS
            .iter()
            .enumerate()
            .filter(|(i, _)| counts[*i] > 0)
            .map(|(i, &name)| (name.to_string(), counts[i]))
            .collect();
        concept_counts.sort_by(|a, b| b.1.cmp(&a.1));
        let day_reward = if rn > 0 { rsum / rn as f32 } else { 0.0 };
        let day_episodes = rn as usize;
        self.day_start_tick = now;

        // Growth & pruning: recruit a specialist for the worst-covered chronic gap,
        // and cull agents that have gone unused (over-produce then prune).
        if self.cfg.growth.enabled {
            if self.cortex.live_count() < self.cfg.growth.max_agents {
                let mut best_c: Option<usize> = None;
                let mut best_gap = self.cfg.growth.recruit_gap;
                for c in 0..EMB_DIM {
                    if self.gaps[c] > best_gap
                        && self.cortex.best_coverage(&embed(&[(CONCEPTS[c], 1.0)]))
                            < self.cfg.growth.recruit_coverage
                    {
                        best_gap = self.gaps[c];
                        best_c = Some(c);
                    }
                }
                if let Some(c) = best_c {
                    self.recruit_specialist(c);
                    self.gaps[c] = 0.0;
                }
            }
            let victims = self
                .cortex
                .prune_candidates(self.cfg.growth.prune_idle, self.cfg.growth.prune_reliability);
            for id in victims {
                self.cortex.prune(id);
                self.pruned += 1;
            }
        }

        // Deep memories fade if unused (and drop when spent) — the gate stays lean.
        let gk_decay = self.cfg.gatekeeper.decay;
        for d in self.deep.iter_mut() {
            d.strength *= gk_decay;
        }
        self.deep.retain(|d| d.strength > 0.15);

        self.vitals
            .reconcile(self.cfg.vitals.target_day, self.cfg.vitals.difficulty_gain);
        self.sleeps += 1;
        let strengthened = self.cortex.top_edges(5);
        DreamLog {
            day: self.sleeps,
            replayed: episodes.len(),
            strengthened,
            new_prototypes: new_protos,
            note: format!(
                "endorsed {endorsed} · dismissed {dismissed} · kept {kept} · valence-flips {flips} ({day_episodes} today)"
            ),
            day_reward,
            concept_counts,
            goal: self.goal.clone(),
        }
    }

    fn stats(&self) -> MindStats {
        let avg = if self.recent_rewards.is_empty() {
            0.0
        } else {
            self.recent_rewards.iter().sum::<f32>() / self.recent_rewards.len() as f32
        };
        MindStats {
            tick: self.vitals.age_ticks,
            day: self.sleeps,
            stage: self.stage.clone(),
            energy: (1.0 - self.vitals.pressure / self.cfg.vitals.pressure_capacity.max(0.001))
                .clamp(0.0, 1.0),
            curiosity: self.vitals.curiosity,
            agents_total: self.cortex.live_count(),
            gpu_count: self.cortex.count_tier(Placement::Gpu),
            cpu_count: self.cortex.count_tier(Placement::Cpu),
            cold_count: self.cortex.count_tier(Placement::Cold),
            gpu_mb: self.cortex.gpu_mb(),
            cpu_mb: self.cortex.cpu_mb(),
            peak_gpu_mb: self.cortex.peak_gpu_mb(),
            peak_cpu_mb: self.cortex.peak_cpu_mb(),
            gpu_budget: self.cortex.gpu_budget(),
            cpu_budget: self.cortex.cpu_budget(),
            total_mb: self.cortex.total_mb(),
            sleeps: self.sleeps,
            avg_reward: avg,
            thinks: self.thinks,
            recruited: self.recruited,
            pruned: self.pruned,
            deep_memories: self.deep.len(),
            reflexes: self.reflexes,
        }
    }
}
