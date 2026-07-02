//! chappie — assemble a brain, drop it into a world, and run one deterministic life.
//!
//! Usage:
//!   chappie [--seed N] [--ticks N] [--config FILE] [--set key=value ...]
//!
//! Every tunable lives in `Config`; `--set` and `--config` are the fine-adjustment
//! surface. Each run prints its effective config and a machine-readable RESULT
//! line, so a benchmark score can always be referenced back to an exact config.

use chappie_brain::Brain;
use chappie_core::*;
use chappie_examiner::Examiner;
use chappie_harness::{Agent, StubAgent};
use chappie_world::{Sandbox, World};

const SPARK: [&str; 9] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

fn main() {
    let cfg = build_config();
    let opts = RunOpts::from_args();

    println!("┌─────────────────────────────────────────────────────────────────────┐");
    println!("│  Chappie — one life                                                   │");
    println!("└─────────────────────────────────────────────────────────────────────┘");
    print_config(&cfg);

    let agents = build_population();
    let total_agents = agents.len();

    let mut brain = Brain::new(agents, cfg.clone());
    let mut world = Sandbox::new();
    let mut exam = Examiner::standard();
    let mut wrng = Rng::new(cfg.seed ^ 0xABCD_1234);

    if opts.endless {
        run_endless(&cfg, &mut brain, &mut world, &mut exam, &mut wrng, &opts);
        return;
    }

    #[cfg(feature = "burn")]
    let neo_id = brain.cortex().id_of("Neocortex");
    #[cfg(feature = "burn")]
    let mut neo_curve: Vec<f32> = Vec::new();

    let s0 = brain.stats();
    println!(
        "\npopulation: {} agents, {:.0}MB total if all resident\n",
        total_agents, s0.total_mb
    );
    println!("── first moments of life ──────────────────────────────────────────────");

    let mut day_last_logged = 0u64;
    for t in 0..cfg.ticks {
        let stimuli = world.observe(&mut wrng);
        let action = brain.perceive_act(&stimuli, true);
        let reward = world.step(&action, &mut wrng);
        brain.reward(reward);

        if t < 5 {
            narrate_tick(t, &brain.trace(), reward);
        }

        if brain.tired() {
            let dream = brain.sleep();
            if dream.day.is_multiple_of(25) && dream.day != day_last_logged {
                day_last_logged = dream.day;
                let top = dream
                    .strengthened
                    .first()
                    .map(|(a, b, w)| format!("{a}~{b} ({w:.2})"))
                    .unwrap_or_else(|| "—".into());
                println!(
                    "  💤 day {:>3}: replayed {:>3} · +{} prototypes · strongest link {}",
                    dream.day, dream.replayed, dream.new_prototypes, top
                );
            }
        }

        if t > 0 && t.is_multiple_of(250) {
            let score = exam.examine(&mut brain, t as u64, world.stage());
            world.advance(score);
            brain.set_stage(world.stage());
            let (hard, recall) = exam.latest().map(|r| (r.hard, r.recall)).unwrap_or((0.0, 0.0));
            let st = brain.stats();
            println!(
                "  t={:>5} {:>11} │ clean {} {:.0}% │ hard {:.0}% recall {:.0}% │ rwd {:+.2} │ gpu{} cpu{} cold{}",
                t,
                st.stage,
                bar(score, 8),
                score * 100.0,
                hard * 100.0,
                recall * 100.0,
                st.avg_reward,
                st.gpu_count,
                st.cpu_count,
                st.cold_count,
            );
            #[cfg(feature = "burn")]
            if let Some(nid) = neo_id {
                neo_curve.push(neo_accuracy(&brain, nid));
            }
        }
    }

    final_report(&brain, &exam, &cfg);

    #[cfg(feature = "burn")]
    print_neo_curve(&neo_curve);
}

// ============================================================================
// Config assembly — defaults, then a config file, then --set overrides.
// ============================================================================

fn build_config() -> Config {
    let mut cfg = Config::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let next = args.get(i + 1).cloned();
        match args[i].as_str() {
            "--seed" => {
                apply(&mut cfg, "seed", next.as_deref());
                i += 2;
            }
            "--ticks" => {
                apply(&mut cfg, "ticks", next.as_deref());
                i += 2;
            }
            "--config" => {
                if let Some(path) = next.as_deref() {
                    load_config_file(&mut cfg, path);
                }
                i += 2;
            }
            "--set" => {
                if let Some(kv) = next.as_deref() {
                    if let Some((k, v)) = kv.split_once('=') {
                        if !cfg.set(k.trim(), v.trim()) {
                            eprintln!("warning: unknown/invalid setting '{kv}'");
                        }
                    }
                }
                i += 2;
            }
            "--endless" => i += 1,
            "--days" | "--diary-dir" | "--task-dir" | "--state-dir" => i += 2,
            other => {
                eprintln!("warning: ignoring unknown arg '{other}'");
                i += 1;
            }
        }
    }
    cfg
}

fn apply(cfg: &mut Config, key: &str, val: Option<&str>) {
    if let Some(v) = val {
        if !cfg.set(key, v) {
            eprintln!("warning: invalid value for '{key}': '{v}'");
        }
    }
}

fn load_config_file(cfg: &mut Config, path: &str) {
    match std::fs::read_to_string(path) {
        Ok(text) => {
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Some((k, v)) = line.split_once('=') {
                    if !cfg.set(k.trim(), v.trim()) {
                        eprintln!("warning: unknown/invalid setting in {path}: '{line}'");
                    }
                }
            }
        }
        Err(e) => eprintln!("warning: could not read config '{path}': {e}"),
    }
}

fn print_config(cfg: &Config) {
    println!("seed={}  ticks={}  propose_threshold={}", cfg.seed, cfg.ticks, cfg.propose_threshold);
    println!(
        "budget: gpu={:.0}MB cpu={:.0}MB max_participants={}",
        cfg.budget.gpu_mb, cfg.budget.cpu_mb, cfg.budget.max_participants
    );
    let p = &cfg.priority;
    println!(
        "priority: w_rel={} w_shared={} w_reliab={} hysteresis={} floor={} cpu_penalty={}",
        p.w_relevance, p.w_shared, p.w_reliability, p.hysteresis, p.floor, p.cpu_penalty
    );
    let h = &cfg.hebbian;
    println!(
        "hebbian: online={} sleep={} decay={} max_weight={}   attention.floor={}",
        h.online_rate, h.sleep_rate, h.decay, h.max_weight, cfg.attention.floor
    );
    println!("  (adjust any knob with --set <group.key>=<value>, or --config <file>)");
}

// ============================================================================
// Population — a diverse cast spanning senses, faculties, and hemispheres.
// ============================================================================

fn build_population() -> Vec<Box<dyn Agent>> {
    use ActionKind::{Attend, Move, Speak};
    use Hemisphere::{Left, Right};

    let spec: &[(&str, Hemisphere, &[(&str, f32)], ActionKind, &str, f32)] = &[
        ("Wernicke", Left, &[("language", 1.0)], Speak, "language", 220.0),
        ("Broca", Left, &[("language", 0.8), ("social", 0.3)], Speak, "language", 200.0),
        ("Logician", Left, &[("logical", 1.0)], Speak, "logical", 180.0),
        ("Numerist", Left, &[("numeric", 1.0)], Speak, "numeric", 160.0),
        ("Namer", Left, &[("visual", 0.7), ("language", 0.5)], Speak, "visual", 190.0),
        ("Grammar", Left, &[("language", 0.6), ("logical", 0.4)], Speak, "language", 150.0),
        ("Gestalt", Right, &[("visual", 1.0), ("spatial", 0.4)], Speak, "visual", 240.0),
        ("Navigator", Right, &[("spatial", 1.0), ("visual", 0.3)], Speak, "spatial", 210.0),
        ("Amygdala", Right, &[("danger", 1.0)], Move, "danger", 120.0),
        ("Empath", Right, &[("social", 1.0)], Speak, "social", 200.0),
        ("Prosody", Right, &[("auditory", 0.8), ("social", 0.4)], Speak, "auditory", 170.0),
        ("Ear", Right, &[("auditory", 1.0)], Speak, "auditory", 150.0),
        ("Nose", Right, &[("olfactory", 1.0)], Speak, "olfactory", 90.0),
        ("Tongue", Right, &[("gustatory", 1.0)], Speak, "gustatory", 90.0),
        ("Skin", Right, &[("tactile", 1.0)], Speak, "tactile", 110.0),
        ("Novelty", Right, &[("reward", 0.7), ("danger", 0.3)], Attend, "", 100.0),
    ];

    #[cfg_attr(not(feature = "burn"), allow(unused_mut))]
    let mut agents: Vec<Box<dyn Agent>> = spec
        .iter()
        .enumerate()
        .map(|(i, (name, hemi, comp, kind, utter, fp))| {
            StubAgent::new(i as AgentId, *name, *hemi, embed(comp), *kind, *utter, *fp).boxed()
        })
        .collect();

    // The long-term *parametric* memory: a real trainable net that learns the
    // query→concept mapping by replaying the episodic heap during sleep. It joins
    // the population like any other agent — the loop doesn't know it's special.
    #[cfg(feature = "burn")]
    {
        let id = agents.len() as AgentId;
        agents.push(
            chappie_burn::BurnAgent::new(
                id,
                "Neocortex",
                Hemisphere::Left,
                embed(&[("language", 0.5), ("logical", 0.5), ("numeric", 0.5), ("visual", 0.4)]),
                320.0,
            )
            .boxed(),
        );
    }

    agents
}

// ============================================================================
// Rendering
// ============================================================================

fn bar(x: f32, width: usize) -> String {
    let filled = ((x.clamp(0.0, 1.0)) * width as f32).round() as usize;
    (0..width).map(|i| if i < filled { '█' } else { '░' }).collect()
}

fn spark(scores: &[f32]) -> String {
    scores
        .iter()
        .map(|&x| {
            let lvl = ((x.clamp(0.0, 1.0)) * 8.0).round() as usize;
            SPARK[lvl.min(8)]
        })
        .collect()
}

fn narrate_tick(t: usize, tr: &chappie_brain::Trace, reward: f32) {
    let salient: Vec<String> = tr.salient.iter().map(|(l, s)| format!("{l}:{s:.2}")).collect();
    println!(
        "t={t}  see[{}]  dom={} lead={} surprise={:.2}",
        salient.join(" "),
        tr.dominant,
        tr.lead,
        tr.surprise
    );
    let active: Vec<String> = tr
        .active
        .iter()
        .map(|(name, tier)| {
            let mark = if *tier == "gpu" { "🔥" } else { "·" };
            format!("{mark}{name}")
        })
        .collect();
    println!("     placed: {}", active.join(" "));
    let mut props: Vec<&(String, String, String, f32)> = tr.proposals.iter().collect();
    props.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    for (name, kind, utter, w) in props.iter().take(4) {
        let u = if utter.is_empty() { String::new() } else { format!(" \"{utter}\"") };
        println!("       {name:<10} → {kind}{u}  (w={w:.2})");
    }
    println!(
        "     ⇒ consensus: {} (agree {:.0}%)  reward {:+.2}\n",
        tr.decision,
        tr.agreement * 100.0,
        reward
    );
}

fn final_report(brain: &Brain, exam: &Examiner, cfg: &Config) {
    let st = brain.stats();
    let scores: Vec<f32> = exam.history.iter().map(|r| r.clean).collect();
    let recall_curve: Vec<f32> = exam.history.iter().map(|r| r.recall).collect();
    let (last_hard, last_recall) = exam
        .history
        .last()
        .map(|r| (r.hard, r.recall))
        .unwrap_or((0.0, 0.0));

    println!("\n╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  LIFE REPORT                                                          ║");
    println!("╚═══════════════════════════════════════════════════════════════════════╝");
    println!(
        "lived {} ticks over {} sleep-days · final stage: {} · escalated to thinking {} times",
        st.tick, st.day, st.stage, st.thinks
    );

    let (first, last, best, auc) = if scores.is_empty() {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        let first = scores[0];
        let last = *scores.last().unwrap();
        let best = scores.iter().cloned().fold(0.0, f32::max);
        let auc = scores.iter().sum::<f32>() / scores.len() as f32;
        (first, last, best, auc)
    };

    if !scores.is_empty() {
        println!("\nbenchmark over life (clean · hard · recall):");
        println!(
            "  clean  {}  start {:.0}% → end {:.0}%  (best {:.0}%, avg {:.0}%)",
            spark(&scores),
            first * 100.0,
            last * 100.0,
            best * 100.0,
            auc * 100.0
        );
        println!(
            "  recall {}  end {:.0}%  (cue→recall; solved by short-term working memory)",
            spark(&recall_curve),
            last_recall * 100.0
        );
        println!("  hard   end {:.0}%  (noisy + composite probes)", last_hard * 100.0);
    }

    if let Some(rep) = exam.history.last() {
        println!("\nfinal report card:");
        for (name, s) in &rep.detail {
            println!("  {name:<10} {} {:.0}%", bar(*s, 8), s * 100.0);
        }
    }

    println!("\nspecialists that emerged (reliability · activations):");
    for (name, rel, acts) in brain.cortex().top_reliability(6) {
        println!("  {name:<10} {} {:.2}  ({} activations)", bar(rel / 1.5, 8), rel, acts);
    }

    println!("\nstrongest connections formed (fire together, wire together):");
    let edges = brain.cortex().top_edges(6);
    if edges.is_empty() {
        println!("  (none yet — a longer life wires more)");
    } else {
        for (a, b, w) in edges {
            println!("  {a:<10} ── {:.2} ── {b}", w);
        }
    }

    // Attention tiers as of the last tick — the priority scheduler in action.
    println!("\nattention tiers (compute hierarchy, last tick):");
    let gpu = brain.cortex().tier_names(Placement::Gpu);
    let cpu = brain.cortex().tier_names(Placement::Cpu);
    println!(
        "  🔥 gpu  (hot) : {:<40} [{:.0}/{:.0}MB, peak {:.0}]",
        if gpu.is_empty() { "—".into() } else { gpu.join(", ") },
        st.gpu_mb,
        st.gpu_budget,
        st.peak_gpu_mb
    );
    println!(
        "  ·  cpu  (warm): {:<40} [{:.0}/{:.0}MB, peak {:.0}]",
        if cpu.is_empty() { "—".into() } else { cpu.join(", ") },
        st.cpu_mb,
        st.cpu_budget,
        st.peak_cpu_mb
    );
    println!("  ·  cold       : {} agents unloaded", st.cold_count);
    println!(
        "  → peak resident {:.0}MB of {:.0}MB total ({:.0}% at once)",
        st.peak_gpu_mb + st.peak_cpu_mb,
        st.total_mb,
        if st.total_mb > 0.0 {
            100.0 * (st.peak_gpu_mb + st.peak_cpu_mb) / st.total_mb
        } else {
            0.0
        }
    );

    // Machine-readable line — a benchmark referenced to its exact config AND the
    // git commit it ran against, so benchmark movements correlate to changes.
    let flavor = if cfg!(feature = "burn") { "burn" } else { "std" };
    println!(
        "\nRESULT git={} build={} seed={} ticks={} bench_final={:.3} bench_best={:.3} bench_auc={:.3} bench_hard={:.3} bench_recall={:.3} reward={:.3} thinks={} days={} stage={} peak_gpu_mb={:.0} peak_cpu_mb={:.0} gpu_budget={:.0} cpu_budget={:.0}",
        git_hash(), flavor, cfg.seed, cfg.ticks, last, best, auc, last_hard, last_recall, st.avg_reward, st.thinks, st.day, st.stage,
        st.peak_gpu_mb, st.peak_cpu_mb, cfg.budget.gpu_mb, cfg.budget.cpu_mb
    );
}

/// Short git commit the binary is running against ("nogit" if unavailable).
fn git_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "nogit".to_string())
}

// ============================================================================
// Endless mode — live forever, keep a diary, take tasks from an inbox.
// ============================================================================

struct RunOpts {
    endless: bool,
    max_days: Option<u64>,
    diary_dir: String,
    task_dir: String,
    state_dir: String,
}

impl RunOpts {
    fn from_args() -> Self {
        let mut o = RunOpts {
            endless: false,
            max_days: None,
            diary_dir: "diary".into(),
            task_dir: "tasks".into(),
            state_dir: "state".into(),
        };
        let args: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--endless" => {
                    o.endless = true;
                    i += 1;
                }
                "--days" => {
                    o.max_days = args.get(i + 1).and_then(|s| s.parse().ok());
                    i += 2;
                }
                "--diary-dir" => {
                    if let Some(v) = args.get(i + 1) {
                        o.diary_dir = v.clone();
                    }
                    i += 2;
                }
                "--task-dir" => {
                    if let Some(v) = args.get(i + 1) {
                        o.task_dir = v.clone();
                    }
                    i += 2;
                }
                "--state-dir" => {
                    if let Some(v) = args.get(i + 1) {
                        o.state_dir = v.clone();
                    }
                    i += 2;
                }
                _ => i += 1,
            }
        }
        o
    }
}

fn run_endless(
    _cfg: &Config,
    brain: &mut Brain,
    world: &mut Sandbox,
    exam: &mut Examiner,
    wrng: &mut Rng,
    opts: &RunOpts,
) {
    let inbox = format!("{}/inbox", opts.task_dir);
    let done = format!("{}/done", opts.task_dir);
    std::fs::create_dir_all(&inbox).ok();
    std::fs::create_dir_all(&done).ok();
    std::fs::create_dir_all(&opts.diary_dir).ok();
    std::fs::create_dir_all(&opts.state_dir).ok();
    let snap_path = format!("{}/latest.json", opts.state_dir);
    if brain.try_load_snapshot(&snap_path) {
        let st = brain.stats();
        world.set_stage_by_name(&st.stage);
        println!(
            "  ⟳ resumed from snapshot — day {}, stage {}, age {} ticks",
            st.day, st.stage, st.tick
        );
    }
    println!(
        "\n── Chappie is living (endless) ─ diary: {}/ · drop tasks in {}/ · Ctrl-C to stop ──",
        opts.diary_dir, inbox
    );

    let mut t: u64 = 0;
    loop {
        // Occasionally check the task inbox; a dropped file becomes the goal.
        if t % 200 == 0 {
            if let Some((name, text)) = poll_task(&inbox, &done) {
                let focus = CONCEPTS
                    .iter()
                    .find(|&&c| text.to_lowercase().contains(c))
                    .map(|&c| c.to_string());
                world.set_focus(focus.clone());
                brain.set_goal(Some(text.trim().to_string()));
                println!(
                    "  📥 task '{}': \"{}\"  → focus: {}",
                    name,
                    text.trim(),
                    focus.as_deref().unwrap_or("(none)")
                );
            }
        }

        let stimuli = world.observe(wrng);
        let action = brain.perceive_act(&stimuli, true);
        let reward = world.step(&action, wrng);
        brain.reward(reward);

        if brain.tired() {
            let dream = brain.sleep();
            write_diary(&opts.diary_dir, &dream, &brain.stats());
            brain.save_snapshot(&snap_path).ok();
            println!(
                "  💤 day {:>4} → diary  (reward {:+.2}, {} memories consolidated)",
                dream.day, dream.day_reward, dream.replayed
            );
            if let Some(max) = opts.max_days {
                if max > 0 && dream.day >= max {
                    println!("  reached {max} days — stopping.");
                    break;
                }
            }
        }

        if t > 0 && t % 500 == 0 {
            let score = exam.examine(brain, t, world.stage());
            world.advance(score);
            brain.set_stage(world.stage());
        }
        t += 1;
    }
}

fn poll_task(inbox: &str, done: &str) -> Option<(String, String)> {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(inbox)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    let path = files.into_iter().next()?;
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let name = path.file_name()?.to_string_lossy().to_string();
    let _ = std::fs::rename(&path, format!("{done}/{name}"));
    Some((name, text))
}

fn mood(r: f32) -> &'static str {
    if r > 0.7 {
        "good"
    } else if r > 0.3 {
        "okay"
    } else if r > -0.1 {
        "mixed"
    } else {
        "hard"
    }
}

fn write_diary(dir: &str, dream: &DreamLog, st: &MindStats) {
    let mut s = String::new();
    s += &format!("# Day {} — {}\n\n", dream.day, st.stage);
    if let Some(g) = &dream.goal {
        s += &format!("**Working on:** {g}\n\n");
    }
    s += &format!(
        "Today felt {}. Reward {:+.2}; I consolidated {} memories in my sleep.\n\n",
        mood(dream.day_reward),
        dream.day_reward,
        dream.replayed
    );
    if !dream.concept_counts.is_empty() {
        s += "What I paid attention to:\n";
        for (c, n) in dream.concept_counts.iter().take(6) {
            s += &format!("- {c} ×{n}\n");
        }
        s += "\n";
    }
    if dream.new_prototypes > 0 {
        s += &format!("I recognized {} new pattern(s).\n\n", dream.new_prototypes);
    }
    if let Some((a, b, w)) = dream.strengthened.first() {
        s += &format!("My strongest association right now: **{a} ↔ {b}** ({w:.2}).\n\n");
    }
    s += &format!(
        "_age {} ticks · escalated to thinking {} times · {} agents resident_\n",
        st.tick,
        st.thinks,
        st.gpu_count + st.cpu_count
    );
    let _ = std::fs::write(format!("{dir}/day-{:04}.md", dream.day), s);
}

#[cfg(feature = "burn")]
fn neo_accuracy(brain: &Brain, nid: AgentId) -> f32 {
    let concepts = [
        "visual", "auditory", "tactile", "olfactory", "language", "logical", "numeric", "social",
        "danger",
    ];
    let correct = concepts
        .iter()
        .filter(|&&c| brain.cortex().probe(nid, &embed(&[(c, 1.0)])) == concept_index(c))
        .count();
    correct as f32 / concepts.len() as f32
}

#[cfg(feature = "burn")]
fn print_neo_curve(curve: &[f32]) {
    if curve.is_empty() {
        return;
    }
    println!("\nlong-term parametric memory (Neocortex) — solo accuracy over life:");
    println!("  {}", spark(curve));
    let first = curve.first().copied().unwrap_or(0.0);
    let last = curve.last().copied().unwrap_or(0.0);
    println!(
        "  start {:.0}%  →  end {:.0}%   — it learned the query→concept map by replaying sleep",
        first * 100.0,
        last * 100.0
    );
}
