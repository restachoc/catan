//! Throughput benchmark: full games played by the built-in bots, single-thread and all cores.
//! Run with `cargo bench -p catan-core`.

use std::time::Instant;

use catan_core::bots::{heuristic_action, random_action};
use catan_core::rng::Rng;
use catan_core::{write_obs, Config, State, OBS_SIZE};

fn play(seed: u64, heuristic: bool, with_obs: bool, obs: &mut [f32]) -> (u64, bool) {
    let cfg = Config { max_turns: 1000, ..Config::default() };
    let mut s = State::new(cfg, seed);
    let mut rng = Rng::new(seed ^ 0xABCD);
    let mut steps = 0u64;
    while !s.is_over() {
        if with_obs {
            write_obs(&s, obs);
        }
        let a = if heuristic { heuristic_action(&s, &mut rng) } else { random_action(&s, &mut rng) };
        s.step(a);
        steps += 1;
    }
    (steps, s.winner >= 0)
}

fn run(label: &str, games: u64, threads: u64, heuristic: bool, with_obs: bool) {
    let t = Instant::now();
    let per = games / threads;
    let results: Vec<(u64, u64)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|th| {
                sc.spawn(move || {
                    let mut obs = vec![0f32; OBS_SIZE];
                    let (mut steps, mut wins) = (0, 0);
                    for g in 0..per {
                        let (st, won) = play(th * 1_000_000 + g, heuristic, with_obs, &mut obs);
                        steps += st;
                        wins += won as u64;
                    }
                    (steps, wins)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let dt = t.elapsed().as_secs_f64();
    let steps: u64 = results.iter().map(|r| r.0).sum();
    let finished: u64 = results.iter().map(|r| r.1).sum();
    let g = per * threads;
    println!(
        "{label:<38} {threads:>2} thr | {:>9.0} games/s | {:>6.2} M steps/s | {:>5.0} steps/game | {:>5.1}% decided",
        g as f64 / dt,
        steps as f64 / dt / 1e6,
        steps as f64 / g as f64,
        100.0 * finished as f64 / g as f64
    );
}

fn main() {
    let cores = std::thread::available_parallelism().map(|n| n.get() as u64).unwrap_or(1);
    run("random bot", 2_000, 1, false, false);
    run("random bot + obs encoding", 2_000, 1, false, true);
    run("heuristic bot", 5_000, 1, true, false);
    run("heuristic bot + obs encoding", 5_000, 1, true, true);
    run("heuristic bot", 50_000, cores, true, false);
    run("heuristic bot + obs encoding", 50_000, cores, true, true);
}
