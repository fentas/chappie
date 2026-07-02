//! measure — profile what we actually have.
//!
//! Real numbers for the current (CPU, stub-agent) system: how the cognitive loop
//! scales with population size, where per-tick time goes, real resident memory,
//! and the connectome's O(n^2) cost. Run: `cargo run --bin measure`.

use chappie_core::*;
use chappie_harness::*;
use std::time::Instant;

/// Resident set size of this process, in MB (Linux /proc).
fn rss_mb() -> f64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1).map(|x| x.to_string()))
        .and_then(|p| p.parse::<f64>().ok())
        .map(|pages| pages * 4096.0 / (1024.0 * 1024.0))
        .unwrap_or(0.0)
}

/// A population of `n` stub specialists spread over the concept vocabulary.
fn make_pop(n: usize) -> Vec<Box<dyn Agent>> {
    (0..n)
        .map(|i| {
            let c = i % CONCEPTS.len();
            let hemi = if i % 2 == 0 {
                Hemisphere::Left
            } else {
                Hemisphere::Right
            };
            StubAgent::new(
                i as AgentId,
                format!("A{i}"),
                hemi,
                embed(&[(CONCEPTS[c], 1.0)]),
                ActionKind::Speak,
                CONCEPTS[c],
                160.0,
            )
            .boxed()
        })
        .collect()
}

fn query_for(i: usize, rng: &mut Rng) -> (Embedding, usize) {
    let c = rng.next_range(CONCEPTS.len());
    let _ = i;
    (embed(&[(CONCEPTS[c], 1.0)]), c)
}

/// One full cognitive tick against the harness.
fn tick(h: &mut Harness, rng: &mut Rng) {
    let (q, dom) = query_for(0, rng);
    let active = h.schedule(&q, rng);
    let props = h.deliberate(&q, dom, Hemisphere::Left, 0.3, rng);
    let dec = h.consensus(&props);
    h.reinforce(&dec.winners, &active, 0.5);
    h.end_tick();
}

fn main() {
    let cfg = Config::default();
    println!(
        "budgets: gpu={:.0}MB cpu={:.0}MB  ·  max_participants={}  ·  160MB/agent",
        cfg.budget.gpu_mb, cfg.budget.cpu_mb, cfg.budget.max_participants
    );
    println!(
        "(so ~{} agents fit hot, ~{} warm — the rest cold regardless of population)\n",
        (cfg.budget.gpu_mb / 160.0) as usize,
        (cfg.budget.cpu_mb / 160.0) as usize
    );

    // ---- 1. Population scaling -------------------------------------------
    println!("== cognitive-loop scaling (sparse activation) ==");
    println!(
        "{:>7} {:>12} {:>12} {:>11} {:>12} {:>12}",
        "agents", "us/tick", "ticks/sec", "resident", "conctome MB", "proc RSS MB"
    );
    let base_rss = rss_mb();
    for &n in &[16usize, 64, 256, 1024, 4096] {
        let mut h = Harness::new(make_pop(n), &cfg);
        let mut rng = Rng::new(42);
        for _ in 0..200 {
            tick(&mut h, &mut rng); // warm up
        }
        let iters = 2000;
        let t = Instant::now();
        for _ in 0..iters {
            tick(&mut h, &mut rng);
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let resident = h.count_tier(Placement::Gpu) + h.count_tier(Placement::Cpu);
        let conctome_mb = (n * n * 4) as f64 / (1024.0 * 1024.0);
        println!(
            "{:>7} {:>12.2} {:>12.0} {:>11} {:>12.2} {:>12.1}",
            n,
            us,
            1e6 / us,
            resident,
            conctome_mb,
            rss_mb() - base_rss
        );
        drop(h);
    }

    // ---- 2. Where the tick goes (at n=1024) ------------------------------
    println!("\n== per-tick breakdown (n=1024) ==");
    let n = 1024;
    let mut h = Harness::new(make_pop(n), &cfg);
    let mut rng = Rng::new(7);
    for _ in 0..200 {
        tick(&mut h, &mut rng);
    }
    let iters = 3000;
    let (mut t_sched, mut t_delib, mut t_cons) = (0.0f64, 0.0f64, 0.0f64);
    for _ in 0..iters {
        let (q, dom) = query_for(0, &mut rng);
        let t = Instant::now();
        let active = h.schedule(&q, &mut rng);
        t_sched += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let props = h.deliberate(&q, dom, Hemisphere::Left, 0.3, &mut rng);
        t_delib += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let dec = h.consensus(&props);
        t_cons += t.elapsed().as_secs_f64();
        h.reinforce(&dec.winners, &active, 0.5);
        h.end_tick();
    }
    let us = |x: f64| x * 1e6 / iters as f64;
    println!(
        "schedule (rank all n): {:>7.2} us   ·  deliberate (capped participants): {:>6.2} us   ·  consensus: {:>5.2} us",
        us(t_sched),
        us(t_delib),
        us(t_cons)
    );
    println!("→ schedule is the O(n) part; deliberation (the 'inference') is O(participants), constant in n.");

    // ---- 3. Connectome growth cost ---------------------------------------
    println!("\n== connectome grow() cost (the O(n^2) resize) ==");
    for &n in &[256usize, 1024, 4096] {
        let mut c = Connectome::new(n, 1.0);
        let iters = 50;
        let t = Instant::now();
        for k in 0..iters {
            c.grow(n + k + 1);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let mb = (n * n * 4) as f64 / (1024.0 * 1024.0);
        println!("  n={:>5}: {:>7.3} ms/grow   (matrix {:>7.2} MB)", n, ms, mb);
    }
    println!("→ dense connectome is n^2: fine to ~4k agents (67MB); a sparse graph is needed beyond that.");
}
