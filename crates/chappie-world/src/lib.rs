//! chappie-world — the environment Chappie lives in.
//!
//! A [`World`] emits multimodal [`Stimulus`]es and scores [`Action`]s. The
//! bundled [`Sandbox`] grows in complexity through human-like life stages, so a
//! long run traces a developmental arc. Swap in a richer world (a game, a robot
//! sim, a chat partner) behind the same trait.

use chappie_core::*;

pub trait World {
    /// Produce the next scene. Records what a good response would be.
    fn observe(&mut self, rng: &mut Rng) -> Vec<Stimulus>;
    /// Score the agent's action against the current scene, in `[-1, 1]`.
    fn step(&mut self, action: &Action, rng: &mut Rng) -> f32;
    /// Advance the life stage if competence warrants it.
    fn advance(&mut self, competence: f32);
    fn stage(&self) -> &str;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LifeStage {
    Infancy,
    Childhood,
    Adolescence,
    Adulthood,
}

impl LifeStage {
    pub fn name(&self) -> &'static str {
        match self {
            LifeStage::Infancy => "infancy",
            LifeStage::Childhood => "childhood",
            LifeStage::Adolescence => "adolescence",
            LifeStage::Adulthood => "adulthood",
        }
    }

    /// Which concepts the world presents at this stage — the curriculum widens.
    fn palette(&self) -> &'static [&'static str] {
        match self {
            LifeStage::Infancy => &["visual", "auditory", "tactile"],
            LifeStage::Childhood => &["visual", "auditory", "tactile", "olfactory", "language"],
            LifeStage::Adolescence => &[
                "visual", "auditory", "language", "social", "danger", "spatial",
            ],
            LifeStage::Adulthood => &[
                "visual", "language", "logical", "numeric", "social", "danger",
            ],
        }
    }

    /// How many extra distractor stimuli accompany the dominant one.
    fn distractors(&self) -> usize {
        match self {
            LifeStage::Infancy => 0,
            LifeStage::Childhood => 1,
            LifeStage::Adolescence => 2,
            LifeStage::Adulthood => 3,
        }
    }

    fn next(&self) -> LifeStage {
        match self {
            LifeStage::Infancy => LifeStage::Childhood,
            LifeStage::Childhood => LifeStage::Adolescence,
            LifeStage::Adolescence => LifeStage::Adulthood,
            LifeStage::Adulthood => LifeStage::Adulthood,
        }
    }
}

fn modality_for(concept: &str) -> Modality {
    match concept {
        "visual" | "spatial" => Modality::Sight,
        "auditory" => Modality::Sound,
        "tactile" => Modality::Touch,
        "olfactory" => Modality::Smell,
        "gustatory" => Modality::Taste,
        "language" | "logical" | "numeric" | "social" => Modality::Language,
        _ => Modality::Interoception,
    }
}

/// What the world wants for a given dominant concept. Most things should be
/// named (Speak); danger should be avoided (Move) — differentiated behavior.
pub fn expected_kind(concept: &str) -> ActionKind {
    match concept {
        "danger" => ActionKind::Move,
        _ => ActionKind::Speak,
    }
}

pub struct Sandbox {
    stage: LifeStage,
    expected_concept: String,
    expected_kind: ActionKind,
    since_advance: u64,
    focus: Option<String>,
}

impl Sandbox {
    pub fn new() -> Self {
        Sandbox {
            stage: LifeStage::Infancy,
            expected_concept: "visual".to_string(),
            expected_kind: ActionKind::Speak,
            since_advance: 0,
            focus: None,
        }
    }

    pub fn life_stage(&self) -> LifeStage {
        self.stage
    }

    /// Bias what the world presents toward a concept (a task's focus), or clear it.
    pub fn set_focus(&mut self, concept: Option<String>) {
        self.focus = concept;
    }
}

impl Default for Sandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl World for Sandbox {
    fn observe(&mut self, rng: &mut Rng) -> Vec<Stimulus> {
        let palette = self.stage.palette();
        // With a task focus, present that concept ~half the time (when the stage
        // offers it); otherwise pick uniformly from the stage's palette.
        let dom_name: String = match &self.focus {
            Some(f) if rng.next_f32() < 0.5 && palette.contains(&f.as_str()) => f.clone(),
            _ => palette[rng.next_range(palette.len())].to_string(),
        };
        let dom = dom_name.as_str();
        self.expected_concept = dom_name.clone();
        self.expected_kind = expected_kind(dom);

        let mut stimuli = Vec::new();
        // Dominant stimulus: strong, clean.
        stimuli.push(Stimulus {
            modality: modality_for(dom),
            label: dom.to_string(),
            features: embed(&[(dom, 1.0)]),
            intensity: 0.8 + 0.2 * rng.next_f32(),
        });
        // Distractors: weaker, other concepts.
        for _ in 0..self.stage.distractors() {
            let d = palette[rng.next_range(palette.len())];
            stimuli.push(Stimulus {
                modality: modality_for(d),
                label: d.to_string(),
                features: embed(&[(d, 0.6), (dom, 0.2)]),
                intensity: 0.3 + 0.3 * rng.next_f32(),
            });
        }
        stimuli
    }

    fn step(&mut self, action: &Action, _rng: &mut Rng) -> f32 {
        let mut r = 0.0f32;
        // Right kind of response?
        if action.kind == self.expected_kind {
            r += 0.6;
        } else if action.kind == ActionKind::Attend || action.kind == ActionKind::Noop {
            r -= 0.1; // hesitation is mildly bad, not as bad as a wrong act
        } else {
            r -= 0.3;
        }
        // Correctly identified the thing?
        if action.utterance == self.expected_concept {
            r += 0.4;
        }
        // Reward calibrated confidence a touch.
        r += 0.1 * (action.confidence - 0.5);
        self.since_advance += 1;
        r.clamp(-1.0, 1.0)
    }

    fn advance(&mut self, competence: f32) {
        // Graduate when consistently competent and enough time has passed.
        if competence > 0.7 && self.since_advance > 300 && self.stage != LifeStage::Adulthood {
            self.stage = self.stage.next();
            self.since_advance = 0;
        }
    }

    fn stage(&self) -> &str {
        self.stage.name()
    }
}
