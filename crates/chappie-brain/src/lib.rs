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
use chappie_harness::{Agent, Decision, Harness};
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
    energy: f32,
    curiosity: f32,
    age_ticks: u64,
}

impl Vitals {
    fn spend(&mut self, cost: f32) {
        self.energy = (self.energy - cost).max(0.0);
        self.age_ticks += 1;
    }
    fn rest(&mut self) {
        self.energy = 1.0;
    }
    fn feel(&mut self, reward: f32, surprise: f32, gain: f32, decay: f32) {
        self.curiosity = (self.curiosity + gain * surprise - decay * reward.max(0.0)).clamp(0.0, 1.0);
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
    winners: Vec<AgentId>,
    active: Vec<AgentId>,
    surprise: f32,
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
    pub energy: f32,
    pub curiosity: f32,
    pub age_ticks: u64,
    pub rng_state: u64,
    pub reliabilities: Vec<f32>,
    pub connectome: Vec<f32>,
    pub semantic: Vec<Vec<f32>>,
    pub working: Vec<(usize, f32)>,
    pub recent_rewards: Vec<f32>,
}

// ============================================================================
// Brain — the whole thing.
// ============================================================================

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
            vitals: Vitals { energy: 1.0, curiosity: 0.3, age_ticks: 0 },
            working: WorkingMemory::new(),
            rng,
            cfg,
            stage: "infancy".to_string(),
            recent_rewards: Vec::new(),
            sleeps: 0,
            thinks: 0,
            day_start_tick: 0,
            goal: None,
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
            energy: self.vitals.energy,
            curiosity: self.vitals.curiosity,
            age_ticks: self.vitals.age_ticks,
            rng_state: self.rng.state(),
            reliabilities: self.cortex.export_reliabilities(),
            connectome: self.cortex.connectome_weights(),
            semantic: self.semantic.protos.clone(),
            working: self.working.slots.clone(),
            recent_rewards: self.recent_rewards.clone(),
        }
    }

    /// Overlay a snapshot onto a freshly-built brain (same population).
    pub fn restore(&mut self, s: Snapshot) {
        self.sleeps = s.sleeps;
        self.day_start_tick = s.day_start_tick;
        self.thinks = s.thinks;
        self.stage = s.stage;
        self.goal = s.goal;
        self.vitals.energy = s.energy;
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

        // 4. Schedule: rank by priority, place agents on GPU/CPU/cold.
        let mut active = self.cortex.schedule(&query, &mut self.rng);

        // 5. Deliberate → consensus (System 1: the fast reflex).
        let proposals = self
            .cortex
            .deliberate(&query, dom, lead, self.vitals.curiosity, &mut self.rng);
        let mut decision = self.cortex.consensus(&proposals);

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
            let cost = self.cfg.vitals.energy_cost_base
                + self.cfg.vitals.energy_cost_per_agent * active.len() as f32;
            self.vitals.spend(cost);
            self.hippocampus.record(Episode {
                tick: self.vitals.age_ticks,
                stage: self.stage.clone(),
                query,
                dominant: CONCEPTS[dom].to_string(),
                decision: decision.action.clone(),
                active_agents: active.clone(),
                reward: 0.0,
                surprise,
                priority: 1.0,
            });
            self.pending = Some(Pending {
                winners: decision.winners.clone(),
                active,
                surprise,
            });
        }

        self.cortex.end_tick();
        decision.action
    }

    fn reward(&mut self, r: f32) {
        if let Some(ep) = self.hippocampus.buf.last_mut() {
            ep.reward = r;
        }
        self.recent_rewards.push(r);
        if self.recent_rewards.len() > 512 {
            self.recent_rewards.remove(0);
        }
        if let Some(p) = self.pending.take() {
            self.vitals.feel(
                r,
                p.surprise,
                self.cfg.vitals.curiosity_gain,
                self.cfg.vitals.curiosity_reward_decay,
            );
            self.cortex.reinforce(&p.winners, &p.active, r);
        }
    }

    fn tired(&self) -> bool {
        self.vitals.energy < self.cfg.vitals.tired_threshold
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

        self.vitals.rest();
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
            energy: self.vitals.energy,
            curiosity: self.vitals.curiosity,
            agents_total: self.cortex.len(),
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
        }
    }
}
