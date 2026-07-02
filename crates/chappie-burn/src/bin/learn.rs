//! Prove the long-term parametric memory improves over time: replay rewarded
//! episodes into a BurnAgent's `consolidate` and watch its accuracy climb from
//! ~random to mastery — real backprop on the GPU, one "sleep" per day.
//!
//! Run:  WGPU_BACKEND=vulkan cargo run -p chappie-burn --bin learn

use chappie_burn::BurnAgent;
use chappie_core::*;
use chappie_harness::Agent;

fn main() {
    let concepts = [
        "visual", "auditory", "tactile", "olfactory", "language", "logical", "numeric", "social",
        "danger",
    ];
    let mut rng = Rng::new(7);
    let mut agent = BurnAgent::new(0, "Neocortex", Hemisphere::Left, embed(&[("language", 1.0)]), 320.0);

    let eval = |agent: &BurnAgent| -> f32 {
        let correct = concepts
            .iter()
            .filter(|&&c| agent.top_concept(&embed(&[(c, 1.0)])) == c)
            .count();
        correct as f32 / concepts.len() as f32
    };

    println!(
        "Long-term parametric memory — accuracy on {} concepts as sleep replays accumulate:",
        concepts.len()
    );
    println!("day  0 (untrained): {:>3.0}%", eval(&agent) * 100.0);

    for day in 1..=20 {
        // A day's worth of rewarded episodes: each concept seen several times, noisy.
        let mut episodes = Vec::new();
        for c in concepts {
            for _ in 0..4 {
                let mut q = embed(&[(c, 1.0)]);
                for v in q.iter_mut() {
                    *v += 0.05 * rng.next_gauss();
                }
                normalize(&mut q);
                episodes.push(Episode {
                    tick: 0,
                    stage: "demo".into(),
                    query: q,
                    dominant: c.to_string(),
                    decision: Action::noop(),
                    active_agents: vec![0],
                    reward: 1.0,
                    surprise: 0.0,
                });
            }
        }
        agent.consolidate(&episodes, &mut rng);
        if day % 2 == 0 {
            println!("day {day:>2}          : {:>3.0}%", eval(&agent) * 100.0);
        }
    }
}
