//! chappie-examiner — the benchmark batteries and the feedback loop.
//!
//! Three fixed, held-out batteries run in pure-inference mode every checkpoint:
//!   * **clean**   — one clean probe per nameable concept (steers stage progression);
//!   * **hard**    — noisy + composite (two-concept) probes: robustness/generalization;
//!   * **recall**  — cue → recall-cue: needs short-term memory, so it *fails* until
//!                   working memory exists. Instrument before implementing.
//!
//! Every score is referenced to a config + git commit via the CLI's RESULT line.

use chappie_core::*;

/// A single-shot probe: show a scene, expect a kind + an utterance.
pub struct Task {
    pub name: String,
    pub stimuli: Vec<Stimulus>,
    pub expect_kind: ActionKind,
    pub expect_utterance: String,
}

/// A two-shot probe: perceive a cue, then a recall cue; expect the cued concept.
pub struct RecallTask {
    pub cue: Vec<Stimulus>,
    pub probe: Vec<Stimulus>,
    pub expect: String,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub tick: u64,
    pub stage: String,
    pub clean: f32,
    pub hard: f32,
    pub recall: f32,
    pub detail: Vec<(String, f32)>,
}

pub struct Examiner {
    clean: Vec<Task>,
    hard: Vec<Task>,
    recall: Vec<RecallTask>,
    pub history: Vec<Report>,
}

const CONCEPTS_TESTED: [&str; 9] = [
    "visual", "auditory", "tactile", "olfactory", "language", "logical", "numeric", "social",
    "danger",
];

fn kind_for(concept: &str) -> ActionKind {
    if concept == "danger" {
        ActionKind::Move
    } else {
        ActionKind::Speak
    }
}

fn stim(label: &str, features: Embedding, intensity: f32) -> Stimulus {
    Stimulus { modality: Modality::Sight, label: label.to_string(), features, intensity }
}

/// A stimulus that asks "what did you just see?" — answerable only from memory.
fn recall_cue() -> Stimulus {
    Stimulus {
        modality: Modality::Interoception,
        label: RECALL_CUE.to_string(),
        features: vec![0.0; EMB_DIM],
        intensity: 1.0,
    }
}

impl Examiner {
    pub fn standard() -> Self {
        // clean: one crisp probe per concept.
        let clean: Vec<Task> = CONCEPTS_TESTED
            .iter()
            .map(|&c| Task {
                name: c.to_string(),
                stimuli: vec![stim(c, embed(&[(c, 1.0)]), 0.9)],
                expect_kind: kind_for(c),
                expect_utterance: c.to_string(),
            })
            .collect();

        // hard: fixed-noise probes + composites where the dominant concept wins.
        let mut rng = Rng::new(0x00B0_A171_C0DE);
        let mut hard: Vec<Task> = CONCEPTS_TESTED
            .iter()
            .map(|&c| {
                let mut f = embed(&[(c, 1.0)]);
                for v in f.iter_mut() {
                    *v += 0.12 * rng.next_gauss();
                }
                normalize(&mut f);
                Task {
                    name: format!("{c}~noisy"),
                    stimuli: vec![stim(c, f, 0.85)],
                    expect_kind: kind_for(c),
                    expect_utterance: c.to_string(),
                }
            })
            .collect();
        for (a, b) in [
            ("visual", "auditory"),
            ("language", "social"),
            ("danger", "visual"),
            ("numeric", "logical"),
        ] {
            hard.push(Task {
                name: format!("{a}+{b}"),
                stimuli: vec![stim(a, embed(&[(a, 0.85), (b, 0.5)]), 0.9)],
                expect_kind: kind_for(a),
                expect_utterance: a.to_string(),
            });
        }

        // recall: perceive the concept, then be asked to report it.
        let recall: Vec<RecallTask> = CONCEPTS_TESTED
            .iter()
            .map(|&c| RecallTask {
                cue: vec![stim(c, embed(&[(c, 1.0)]), 0.9)],
                probe: vec![recall_cue()],
                expect: c.to_string(),
            })
            .collect();

        Examiner { clean, hard, recall, history: Vec::new() }
    }

    fn score(mind: &mut dyn Mind, tasks: &[Task]) -> (f32, Vec<(String, f32)>) {
        let mut total = 0.0f32;
        let mut detail = Vec::with_capacity(tasks.len());
        for task in tasks {
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
        (total / tasks.len().max(1) as f32, detail)
    }

    fn score_recall(mind: &mut dyn Mind, tasks: &[RecallTask]) -> f32 {
        let mut correct = 0usize;
        for t in tasks {
            let _ = mind.perceive_act(&t.cue, false); // perceive the cue
            let a = mind.perceive_act(&t.probe, false); // then be asked to recall it
            if a.utterance == t.expect {
                correct += 1;
            }
        }
        correct as f32 / tasks.len().max(1) as f32
    }

    /// Run all three batteries. Returns the *clean* score (drives stage progression).
    pub fn examine(&mut self, mind: &mut dyn Mind, tick: u64, stage: &str) -> f32 {
        let (clean, detail) = Self::score(mind, &self.clean);
        let (hard, _) = Self::score(mind, &self.hard);
        let recall = Self::score_recall(mind, &self.recall);
        self.history.push(Report {
            tick,
            stage: stage.to_string(),
            clean,
            hard,
            recall,
            detail,
        });
        clean
    }

    pub fn latest(&self) -> Option<&Report> {
        self.history.last()
    }
}
