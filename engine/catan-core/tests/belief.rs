use catan_core::actions::*;
use catan_core::belief::{Beliefs, Hands, MAX_WORLDS};
use catan_core::bots::{heuristic_action, random_action};
use catan_core::rng::Rng;
use catan_core::state::*;

fn state(n: u8) -> State {
    State::new(Config { n_players: n, max_turns: 1000, ..Config::default() }, 1)
}

/// Step `s` to `hands` through action `a` (the change itself is set by hand) and update the beliefs.
fn fake_step(b: &mut Beliefs, s: &mut State, a: usize, hands: Hands) {
    let prev = s.hands;
    s.hands = hands;
    b.update(&prev, a, s);
}

#[test]
fn steal_resolved_by_later_payment() {
    // Seat 1 steals from seat 2 (1 wood, 1 brick); seat 0 doesn't see the card. Seat 1 then pays a wood
    // it can only have got from the steal: seat 0 deduces both hands exactly.
    let mut s = state(4);
    s.cur = 1;
    s.hands = [[0; 5]; MAX_P];
    s.hands[2] = [1, 1, 0, 0, 0];
    let mut b = Beliefs::new(&s);
    let mut h = s.hands;
    h[2] = [0, 1, 0, 0, 0];
    h[1] = [1, 0, 0, 0, 0];
    fake_step(&mut b, &mut s, STEAL + 1, h);
    assert_eq!(b.seats[0].len(), 2);
    let e = b.seats[0].expected();
    assert!((e[1][0] - 0.5).abs() < 1e-6 && (e[1][1] - 0.5).abs() < 1e-6);
    assert_eq!(b.seats[1].len(), 1, "the thief saw the card");
    assert_eq!(b.seats[2].len(), 1, "the victim saw the card");

    let mut h = s.hands;
    h[1][0] = 0; // pays a wood (any public action)
    fake_step(&mut b, &mut s, TRADE, h);
    assert_eq!(b.seats[0].len(), 1);
    let e = b.seats[0].expected();
    assert_eq!(e[2], [0.0, 1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn monopoly_reveals_counts() {
    // After an unseen steal, a Monopoly on wood shows how much wood seat 2 had.
    let mut s = state(4);
    s.cur = 1;
    s.hands = [[0; 5]; MAX_P];
    s.hands[2] = [1, 1, 0, 0, 0];
    let mut b = Beliefs::new(&s);
    let mut h = s.hands;
    h[2] = [1, 0, 0, 0, 0];
    h[1] = [0, 1, 0, 0, 0];
    fake_step(&mut b, &mut s, STEAL + 1, h);
    assert_eq!(b.seats[3].len(), 2);
    s.cur = 3;
    let mut h = s.hands;
    h[2][0] = 0;
    h[3][0] = 1; // seat 3 takes seat 2's single wood
    fake_step(&mut b, &mut s, PLAY_MONOPOLY, h);
    assert_eq!(b.seats[0].len(), 1);
    assert_eq!(b.seats[0].expected()[1], [0.0, 1.0, 0.0, 0.0, 0.0]);
}

/// Every observer's worlds stay consistent with the truth over many games.
#[test]
fn beliefs_track_true_hands() {
    let (mut steals, mut uncertain, mut obs_steps, mut max_worlds) = (0u64, 0u64, 0u64, 0usize);
    let (mut pruned, mut resets) = (0u32, 0u32);
    for g in 0..3000u64 {
        let n = 2 + (g % 3) as u8;
        let mut s = State::new(Config { n_players: n, max_turns: 1000, random_board: g % 2 == 0, ..Config::default() }, g);
        let mut b = Beliefs::new(&s);
        let mut rng = Rng::new(g ^ 7);
        while !s.is_over() {
            let a = if (s.actor() + g as usize) % 3 == 0 { random_action(&s, &mut rng) } else { heuristic_action(&s, &mut rng) };
            let prev = s.hands;
            s.step(a);
            if (MOVE_ROBBER..DISCARD).contains(&a) && s.hands != prev {
                steals += 1;
            }
            b.update(&prev, a, &s);
            for o in 0..s.n() {
                let bel = &b.seats[o];
                obs_steps += 1;
                uncertain += (bel.len() > 1) as u64;
                max_worlds = max_worlds.max(bel.len());
                assert!(bel.len() >= 1 && bel.len() <= MAX_WORLDS);
                let total: f32 = bel.worlds().map(|(_, w)| w).sum();
                assert!((total - 1.0).abs() < 1e-4, "weights sum to {total}");
                for (h, _) in bel.worlds() {
                    assert_eq!(h[o], s.hands[o], "observer {o} unsure of its own hand");
                    for q in 0..s.n() {
                        assert_eq!(h[q].iter().map(|&c| c as u32).sum::<u32>(), s.hand_total(q), "hand sizes are public");
                    }
                }
                if b.pruned == 0 {
                    assert!(bel.worlds().any(|(h, _)| *h == s.hands), "game {g}: observer {o} lost the true hands");
                }
            }
        }
        pruned += b.pruned;
        resets += b.resets;
    }
    println!(
        "steals {steals}, observer-steps with >1 world {:.2}%, max worlds {max_worlds}, pruned {pruned}, resets {resets}",
        100.0 * uncertain as f64 / obs_steps as f64
    );
    assert_eq!(resets, 0);
}
