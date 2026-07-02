//! chappie-examiner — the benchmark battery and the feedback loop.
//!
//! The examiner periodically drives the [`Mind`] over a *fixed, held-out* battery
//! in pure-inference mode (no learning), scores it, and records the result. The
//! score stream is the learning curve; it also gates life-stage progression, so
//! the benchmark literally steers development.

use chappie_core::*;

/// One held-out probe: show a scene, expect a kind + an utterance.
pub struct Task {
    pub name: String,
    pub stimuli: Vec<Stimulus>,
    pub expect_kind: ActionKind,
    pub expect_utterance: String,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub tick: u64,
    pub stage: String,
    pub score: f32,
    pub detail: Vec<(String, f32)>,
}

pub struct Examiner {
    battery: Vec<Task>,
    pub history: Vec<Report>,
}

impl Examiner {
    /// A battery with one clean probe per nameable concept.
    pub fn standard() -> Self {
        let concepts = [
            "visual", "auditory", "tactile", "olfactory", "language", "logical", "numeric",
            "social", "danger",
        ];
        let battery = concepts
            .iter()
            .map(|&c| {
                let kind = if c == "danger" {
                    ActionKind::Move
                } else {
                    ActionKind::Speak
                };
                Task {
                    name: c.to_string(),
                    stimuli: vec![Stimulus {
                        modality: Modality::Sight,
                        label: c.to_string(),
                        features: embed(&[(c, 1.0)]),
                        intensity: 0.9,
                    }],
                    expect_kind: kind,
                    expect_utterance: c.to_string(),
                }
            })
            .collect();
        Examiner {
            battery,
            history: Vec::new(),
        }
    }

    /// Score the mind over the whole battery (no learning). Returns mean score.
    pub fn examine(&mut self, mind: &mut dyn Mind, tick: u64, stage: &str) -> f32 {
        let mut total = 0.0f32;
        let mut detail = Vec::with_capacity(self.battery.len());
        for task in &self.battery {
            let action = mind.perceive_act(&task.stimuli, false);
            let mut s = 0.0f32;
            if action.kind == task.expect_kind {
                s += 0.5;
            }
            if action.utterance == task.expect_utterance {
                s += 0.5;
            }
            total += s;
            detail.push((task.name.clone(), s));
        }
        let score = total / self.battery.len() as f32;
        self.history.push(Report {
            tick,
            stage: stage.to_string(),
            score,
            detail,
        });
        score
    }

    pub fn latest(&self) -> Option<&Report> {
        self.history.last()
    }
}
