use catan_core::actions::*;
use catan_core::board::*;
use catan_core::bots::{heuristic_action, random_action, random_any_action};
use catan_core::rng::Rng;
use catan_core::state::*;
use catan_core::topology::*;
use catan_core::{exact_hands, write_obs, OBS_SIZE};

fn cfg(n: u8) -> Config {
    Config { n_players: n, max_turns: 1000, ..Config::default() }
}

// ------------------------------------------------------------------ topology & board

#[test]
fn topology_shape() {
    let t = &*TOPO;
    for v in 0..N_VERT {
        let deg = t.vertex_edges[v].count_ones();
        assert!(deg == 2 || deg == 3, "vertex {v} degree {deg}");
        assert_eq!(t.vertex_neighbors[v].count_ones(), deg);
        let nh = t.vertex_hexes[v].count_ones();
        assert!((1..=3).contains(&nh));
    }
    // 30 coastal vertices: 18 have degree 2, 12 have degree 3.
    let deg2 = (0..N_VERT).filter(|&v| t.vertex_edges[v].count_ones() == 2).count();
    assert_eq!(deg2, 18);
    for h in 0..N_HEX {
        assert_eq!(t.hex_vmask[h].count_ones(), 6);
        let nb = t.hex_neighbors[h].count_ones();
        assert!((3..=6).contains(&nb));
    }
    // Coast edges form a closed cycle: consecutive edges share a vertex.
    for i in 0..N_COAST {
        let a = t.edge_vertices[t.coast_edges[i] as usize];
        let b = t.edge_vertices[t.coast_edges[(i + 1) % N_COAST] as usize];
        assert!(a.iter().any(|x| b.contains(x)), "coast gap at {i}");
    }
}

#[test]
fn board_composition() {
    let mut rng = Rng::new(7);
    for b in [Board::beginner(), Board::random(&mut rng), Board::random(&mut rng)] {
        let mut counts = [0; 6];
        for &r in &b.hex_res {
            counts[r as usize] += 1;
        }
        assert_eq!(counts, [4, 3, 4, 4, 3, 1]);
        assert_eq!(b.hex_num[b.desert as usize], 0);
        let mut nums = [0; 13];
        for &n in &b.hex_num {
            nums[n as usize] += 1;
        }
        assert_eq!(nums, [1, 0, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1]);
        // 9 ports on 18 distinct vertices.
        let all: u64 = b.port_mask.iter().fold(0, |m, &x| m | x);
        assert_eq!(all.count_ones(), 18);
    }
    for _ in 0..50 {
        let b = Board::random(&mut rng);
        let red = |n: u8| n == 6 || n == 8;
        for h in 0..N_HEX {
            if red(b.hex_num[h]) {
                assert!(bits32(TOPO.hex_neighbors[h]).all(|g| !red(b.hex_num[g])));
            }
        }
    }
}

#[test]
fn action_space_layout() {
    assert_eq!(N_ACTIONS, 321);
    let mut seen = std::collections::HashSet::new();
    for o in 0..60 {
        let (g, gn, r, rn) = offer_terms(o);
        assert!(g != r && (gn, rn) != (2, 2) && seen.insert((g, gn, r, rn)));
    }
    for give in 0..5 {
        for get in 0..5 {
            if give != get {
                assert_eq!(trade_pair(trade_id(give, get) - TRADE), (give, get));
            }
        }
    }
}

// ------------------------------------------------------------------ targeted rules

/// Play the setup phase with the heuristic bot.
fn after_setup(n: u8, seed: u64) -> State {
    let mut s = State::new(cfg(n), seed);
    let mut rng = Rng::new(seed);
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad) {
        s.step(heuristic_action(&s, &mut rng));
    }
    s
}

#[test]
fn setup_snake_order_and_starting_resources() {
    let mut s = State::new(cfg(4), 1);
    let mut rng = Rng::new(1);
    let mut order = vec![];
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad) {
        if s.phase == Phase::SetupSettlement {
            order.push(s.cur);
        }
        s.step(heuristic_action(&s, &mut rng));
    }
    assert_eq!(order, vec![0, 1, 2, 3, 3, 2, 1, 0]);
    assert_eq!(s.phase, Phase::Roll);
    assert_eq!(s.cur, 0);
    for p in 0..4 {
        assert_eq!(s.settlements[p].count_ones(), 2);
        assert_eq!(s.roads[p].count_ones(), 2);
        // Resources equal the non-desert hexes around the second settlement (1..=3 cards).
        let t = s.hand_total(p);
        assert!((1..=3).contains(&t), "player {p} got {t}");
    }
}

#[test]
fn distance_rule_enforced() {
    let s = after_setup(4, 3);
    let occ = s.occupied();
    for p in 0..4 {
        for v in bits64(s.settle_candidates(p, false)) {
            assert_eq!(occ >> v & 1, 0);
            assert_eq!(TOPO.vertex_neighbors[v] & occ, 0);
        }
    }
}

#[test]
fn seven_triggers_discard_then_robber() {
    // Find a seed where the first roll is a 7 and give player 2 a big hand.
    for seed in 0..500 {
        let mut s = after_setup(4, seed);
        s.hands[2] = [3, 3, 2, 1, 1];
        for r in 0..5 {
            s.bank[r] = 19 - (0..4).map(|p| s.hands[p][r]).sum::<u8>();
        }
        s.step(ROLL);
        if s.last_roll[0] + s.last_roll[1] != 7 {
            continue;
        }
        assert_eq!(s.phase, Phase::Discard);
        assert_eq!(s.actor(), 2);
        assert_eq!(s.discard_need[2], 5);
        while s.phase == Phase::Discard {
            assert_eq!(s.actor(), 2);
            let r = (0..5).find(|&r| s.hands[2][r] > 0).unwrap();
            s.step(DISCARD + r);
        }
        assert_eq!(s.hand_total(2), 5);
        assert_eq!(s.phase, Phase::MoveRobber);
        assert_eq!(s.actor(), 0);
        return;
    }
    panic!("no seed rolled a 7");
}

#[test]
fn robber_blocks_production() {
    let mut s = after_setup(2, 11);
    // Put the robber on every producing hex; no one may gain resources on that number then.
    for seed in 0..200u64 {
        let mut t = s;
        t.reseed_chance(seed);
        let before: Vec<u32> = (0..2).map(|p| t.hand_total(p)).collect();
        t.step(ROLL);
        let roll = t.last_roll[0] + t.last_roll[1];
        if roll == 7 {
            continue;
        }
        let hexes = t.board.num_hexes[roll as usize];
        let mut t2 = s;
        t2.reseed_chance(seed);
        t2.robber = hexes.trailing_zeros() as u8;
        t2.step(ROLL);
        let gained_open: u32 = (0..2).map(|p| t.hand_total(p) - before[p]).sum();
        let gained_blocked: u32 = (0..2).map(|p| t2.hand_total(p) - before[p]).sum();
        assert!(gained_blocked <= gained_open);
    }
    s.step(ROLL);
}

#[test]
fn longest_road_award_and_break() {
    let mut s = State::new(cfg(2), 0);
    // Hand-build: player 0 gets a 5-road chain along the coast.
    s.phase = Phase::Main;
    s.rolled = true;
    let chain: Vec<usize> = (0..5).map(|i| TOPO.coast_edges[i] as usize).collect();
    let start = TOPO.edge_vertices[chain[0]];
    let first_v = start.iter().copied().find(|v| !TOPO.edge_vertices[chain[1]].contains(v)).unwrap();
    s.settlements[0] = 1 << first_v;
    for &e in &chain {
        s.hands[0] = [1, 1, 0, 0, 0];
        s.bank = [18, 18, 19, 19, 19];
        s.step(ROAD + e);
    }
    assert_eq!(s.road_len[0], 5);
    assert_eq!(s.longest_road, 0);
    assert_eq!(s.vp(0), 3);
    // Player 1 settles in the middle of the chain (legal-by-construction here) and breaks it.
    let mid = TOPO.edge_vertices[chain[2]][0];
    let mid = if TOPO.edge_vertices[chain[1]].contains(&mid) { mid } else { TOPO.edge_vertices[chain[2]][1] };
    s.cur = 1;
    s.roads[1] = TOPO.vertex_edges[mid as usize] & !s.roads[0];
    s.hands[1] = [1, 1, 1, 1, 0];
    s.step(SETTLE + mid as usize);
    assert!(s.road_len[0] < 5);
    assert_eq!(s.longest_road, -1);
}

#[test]
fn largest_army() {
    let mut s = after_setup(3, 5);
    s.step(ROLL);
    while s.phase != Phase::Main && s.phase != Phase::Roll {
        let mut rng = Rng::new(0);
        s.step(heuristic_action(&s, &mut rng));
    }
    for i in 0..3 {
        s.dev_hand[0][DEV_KNIGHT] = 1;
        s.dev_played = false;
        s.phase = Phase::Main;
        s.cur = 0;
        s.rolled = true;
        s.step(PLAY_KNIGHT);
        let h = (0..N_HEX).find(|&h| h != s.robber as usize).unwrap();
        s.step(MOVE_ROBBER + h);
        if s.phase == Phase::Steal {
            let a = mask_iter(&s.legal_mask()).next().unwrap();
            s.step(a);
        }
        assert_eq!(s.largest_army, if i >= 2 { 0 } else { -1 });
    }
}

#[test]
fn dev_card_not_playable_on_purchase_turn() {
    let mut s = after_setup(2, 9);
    s.step(ROLL);
    let mut rng = Rng::new(0);
    while s.phase != Phase::Main {
        s.step(heuristic_action(&s, &mut rng));
    }
    // Rig the deck so the next card is a knight.
    s.dev_deck[s.dev_deck_len as usize - 1] = DEV_KNIGHT as u8;
    s.hands[0] = [0, 0, 1, 1, 1];
    s.step(BUY_DEV);
    assert!(!s.is_legal(PLAY_KNIGHT));
    s.step(END_TURN);
    assert_eq!(s.dev_new[0], [0; 5]);
}

// ------------------------------------------------------------------ invariants over many games

fn check_invariants(s: &State, played_dev: u32) {
    let n = s.n();
    for r in 0..5 {
        let held: u32 = (0..n).map(|p| s.hands[p][r] as u32).sum();
        assert_eq!(held + s.bank[r] as u32, 19, "resource {r} not conserved");
    }
    let devs: u32 = (0..n).map(|p| s.dev_total(p)).sum();
    assert_eq!(devs + s.dev_deck_len as u32 + played_dev, 25, "dev cards not conserved");
    let mut occ = 0u64;
    let mut roads = 0u128;
    for p in 0..n {
        assert!(s.settlements[p].count_ones() <= 5);
        assert!(s.cities[p].count_ones() <= 4);
        assert!(s.roads[p].count_ones() <= 15);
        assert_eq!(s.settlements[p] & s.cities[p], 0);
        assert_eq!(occ & s.buildings(p), 0, "two buildings on one vertex");
        assert_eq!(roads & s.roads[p], 0, "two roads on one edge");
        occ |= s.buildings(p);
        roads |= s.roads[p];
        assert_eq!(s.road_len[p], s.longest_road_of(p));
        for r in 0..5 {
            assert!(s.dev_new[p][r] <= s.dev_hand[p][r]);
        }
    }
    for v in bits64(occ) {
        assert_eq!(TOPO.vertex_neighbors[v] & occ, 0, "distance rule broken at {v}");
    }
    if s.longest_road >= 0 {
        let h = s.longest_road as usize;
        assert!(s.road_len[h] >= 5);
        assert!((0..n).all(|q| s.road_len[q] <= s.road_len[h]));
    }
    if s.largest_army >= 0 {
        let h = s.largest_army as usize;
        assert!(s.knights[h] >= 3);
        assert!((0..n).all(|q| s.knights[q] <= s.knights[h]));
    }
    assert!(s.offers_made <= MAX_OFFERS);
    if s.phase == Phase::TradeRespond {
        assert_ne!(s.responder, s.cur);
    }
    if s.is_over() {
        if s.winner >= 0 {
            assert!(s.vp(s.winner as usize) >= s.cfg.vp_target as u32);
        }
    } else {
        assert!(mask_count(&s.legal_mask()) > 0, "no legal action in {:?}", s.phase);
    }
}

/// A state in the main phase with chosen hands (seat 0 to move).
fn main_phase(n: u8, hands: &[[u8; 5]]) -> State {
    let mut s = State::new(Config { trading: true, ..cfg(n) }, 3);
    s.phase = Phase::Main;
    s.rolled = true;
    s.cur = 0;
    for (p, h) in hands.iter().enumerate() {
        for r in 0..5 {
            s.bank[r] += s.hands[p][r];
            s.bank[r] -= h[r];
        }
        s.hands[p] = *h;
    }
    s
}

#[test]
fn trade_offer_flow() {
    // Seat 0 offers 2 wood for 1 ore. Seat 1 has no ore (can only decline), seats 2 and 3 accept, seat 0 picks 3.
    let mut s = main_phase(4, &[[2, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 2]]);
    let offer = (0..60).find(|&o| offer_terms(o) == (0, 2, 4, 1)).unwrap();
    assert!(!s.is_legal(OFFER + offer), "terms come after proposing");
    s.try_step(PROPOSE_TRADE).unwrap();
    assert_eq!(s.phase, Phase::OfferTerms);
    s.try_step(OFFER + offer).unwrap();
    assert_eq!((s.phase, s.actor()), (Phase::TradeRespond, 1));
    assert!(!s.is_legal(ACCEPT_OFFER), "seat 1 cannot pay");
    s.try_step(DECLINE_OFFER).unwrap();
    s.try_step(ACCEPT_OFFER).unwrap();
    s.try_step(ACCEPT_OFFER).unwrap();
    assert_eq!((s.phase, s.actor()), (Phase::TradeChoose, 0));
    assert!(!s.is_legal(CHOOSE_PARTNER + 1) && s.is_legal(CHOOSE_PARTNER + 2) && s.is_legal(CHOOSE_PARTNER + 3));
    s.try_step(CHOOSE_PARTNER + 3).unwrap();
    assert_eq!(s.phase, Phase::Main);
    assert_eq!(s.hands[0], [0, 0, 0, 0, 1]);
    assert_eq!(s.hands[3], [2, 0, 0, 0, 1]);
    assert_eq!(s.hands[2], [0, 0, 0, 0, 1], "the other accepter keeps its cards");
    assert_eq!((s.trades[0], s.trades[3], s.trades[2]), (1, 1, 0));
}

#[test]
fn trading_off_by_default() {
    let mut s = main_phase(4, &[[2, 2, 2, 2, 2], [0; 5], [0; 5], [0; 5]]);
    assert!(s.is_legal(PROPOSE_TRADE));
    s.cfg.trading = false;
    assert!(!Config::default().trading);
    assert!(!s.is_legal(PROPOSE_TRADE));
}

#[test]
fn trade_offers_are_limited_per_turn() {
    let mut s = main_phase(3, &[[5, 0, 0, 0, 0], [0, 5, 0, 0, 0], [0, 5, 0, 0, 0]]);
    s.try_step(PROPOSE_TRADE).unwrap();
    s.try_step(CANCEL_OFFER).unwrap(); // a cancelled proposal still counts
    for _ in 1..MAX_OFFERS {
        s.try_step(PROPOSE_TRADE).unwrap();
        s.try_step(OFFER).unwrap(); // 1 wood for 1 brick
        s.try_step(DECLINE_OFFER).unwrap();
        s.try_step(DECLINE_OFFER).unwrap();
        assert_eq!(s.phase, Phase::Main, "nobody accepted");
    }
    assert!(!s.is_legal(PROPOSE_TRADE));
    s.try_step(END_TURN).unwrap();
    assert_eq!(s.offers_made, 0);
}

fn stress(games: u64, heuristic_mix: bool, random_board: bool) {
    let mut obs = vec![0f32; OBS_SIZE];
    let mut decided = 0;
    for g in 0..games {
        let n = 2 + (g % 3) as u8;
        let mut s = State::new(Config { random_board, trading: g % 2 == 1, ..cfg(n) }, g);
        let mut rng = Rng::new(g ^ 99);
        let mut played = 0;
        let mut steps = 0;
        while !s.is_over() {
            let a = if heuristic_mix && (s.actor() + g as usize) % 2 == 0 {
                heuristic_action(&s, &mut rng)
            } else if g % 2 == 1 {
                random_any_action(&s, &mut rng) // exercises player-to-player trading
            } else {
                random_action(&s, &mut rng)
            };
            assert!(s.is_legal(a));
            if a == PLAY_KNIGHT || a == PLAY_ROAD_BUILDING || (PLAY_MONOPOLY..MOVE_ROBBER).contains(&a) {
                played += 1;
            }
            s.step(a);
            if steps % 7 == 0 {
                write_obs(&s, &exact_hands(&s), &mut obs);
                assert!(obs.iter().all(|x| x.is_finite() && *x >= 0.0));
            }
            check_invariants(&s, played);
            steps += 1;
            assert!(steps < 200_000, "game did not terminate");
        }
        decided += (s.winner >= 0) as u32;
    }
    if heuristic_mix {
        assert!(decided as u64 > games * 9 / 10, "too many draws: {decided}/{games}");
    }
}

#[test]
fn stress_random_games() {
    stress(300, false, false);
}

#[test]
fn stress_mixed_games_random_boards() {
    stress(300, true, true);
}

#[test]
#[ignore = "long; run with --release -- --ignored"]
fn stress_many_games() {
    stress(100_000, true, true);
}

#[test]
fn determinism() {
    for seed in 0..20 {
        let play = || {
            let mut s = State::new(cfg(4), seed);
            let mut rng = Rng::new(seed);
            let mut actions = vec![];
            while !s.is_over() {
                let a = heuristic_action(&s, &mut rng);
                actions.push(a);
                s.step(a);
            }
            (actions, s.winner, s.turn)
        };
        assert_eq!(play(), play());
    }
}

#[test]
fn chance_streams_independent_of_play() {
    // Same seed, different players: the k-th roll and the dev deck order must not depend on the actions.
    for seed in 0..20 {
        let play = |bot_seed: u64, heuristic: bool| {
            let mut s = State::new(cfg(4), seed);
            let deck = s.dev_deck;
            let mut rng = Rng::new(bot_seed);
            let mut rolls = vec![];
            while !s.is_over() {
                let a = if heuristic { heuristic_action(&s, &mut rng) } else { random_action(&s, &mut rng) };
                s.step(a);
                if a == ROLL {
                    rolls.push(s.last_roll);
                }
            }
            (rolls, deck, s.board.hex_res)
        };
        let (r1, d1, b1) = play(1, true);
        let (r2, d2, b2) = play(2, false);
        let n = r1.len().min(r2.len());
        assert!(n > 20);
        assert_eq!(r1[..n], r2[..n]);
        assert_eq!((d1, b1), (d2, b2));
    }
}

#[test]
fn replay_from_actions() {
    let mut s = State::new(cfg(4), 42);
    let mut rng = Rng::new(1);
    let mut actions = vec![];
    while !s.is_over() {
        let a = heuristic_action(&s, &mut rng);
        actions.push(a);
        s.step(a);
    }
    let mut r = State::new(cfg(4), 42);
    for &a in &actions {
        r.try_step(a).unwrap();
    }
    assert_eq!(r.winner, s.winner);
    assert_eq!(r.hands, s.hands);
    assert_eq!(r.roads, s.roads);
}

#[test]
fn heuristic_beats_random() {
    let mut wins = 0;
    let games = 200;
    for g in 0..games {
        let mut s = State::new(cfg(4), g);
        let mut rng = Rng::new(g);
        let hero = (g % 4) as usize;
        while !s.is_over() {
            let a = if s.actor() == hero { heuristic_action(&s, &mut rng) } else { random_action(&s, &mut rng) };
            s.step(a);
        }
        wins += (s.winner == hero as i8) as u32;
    }
    assert!(wins > games as u32 * 3 / 4, "heuristic won only {wins}/{games}");
}

// ------------------------------------------------------------------ MCTS

mod mcts_tests {
    use catan_core::mcts::*;
    use catan_core::obs::OBS_SIZE;
    use catan_core::rng::Rng;
    use catan_core::state::*;
    use catan_core::{exact_hands, write_obs, N_ACTIONS};

    /// Uniform priors and a flat value: a search driven only by its own terminal backups.
    fn run(root: State, sims: u32) -> Search {
        let mut s = Search::new(root, MctsConfig { sims, ..MctsConfig::default() }, 7);
        let priors = vec![1.0f32; N_ACTIONS];
        let mut obs = vec![0f32; OBS_SIZE];
        while let Some(leaf) = s.select() {
            write_obs(leaf, &exact_hands(leaf), &mut obs); // the real driver encodes every leaf; make sure that works too
            let v = [0.25; MAX_P];
            s.expand(&priors, &v);
        }
        s
    }

    #[test]
    fn visits_sum_to_sims_and_are_legal() {
        let mut rng = Rng::new(3);
        for seed in 0..30 {
            let mut st = State::new(Config::default(), seed);
            // Advance a random amount into the game.
            for _ in 0..(seed * 13) {
                if st.is_over() {
                    break;
                }
                st.step(catan_core::bots::heuristic_action(&st, &mut rng));
            }
            if st.is_over() {
                continue;
            }
            let s = run(st, 100);
            let visits = s.visits();
            let total: u32 = visits.iter().map(|v| v.1).sum();
            assert_eq!(total, 99, "root expansion uses one simulation");
            for (a, _) in visits {
                assert!(st.is_legal(a));
            }
        }
    }

    #[test]
    fn search_finds_immediate_win() {
        // Player 0 at 9 VP can build a city for the win; everything else ends the turn.
        let mut rng = Rng::new(0);
        let mut st = State::new(Config::default(), 11);
        while st.phase != Phase::Main || st.cur != 0 {
            st.step(catan_core::bots::heuristic_action(&st, &mut rng));
        }
        st.dev_hand[0][DEV_VP] = (9 - st.vp(0)) as u8 + st.dev_hand[0][DEV_VP];
        assert_eq!(st.vp(0), 9);
        let give: [u8; 5] = [0, 0, 0, 2, 3];
        for r in 0..5 {
            let need = give[r].saturating_sub(st.hands[0][r]);
            st.hands[0][r] += need;
            st.bank[r] -= need;
        }
        let s = run(st, 400);
        let best = s.visits().into_iter().max_by_key(|v| v.1).unwrap().0;
        assert!((catan_core::actions::CITY..catan_core::actions::ROAD).contains(&best), "picked {}", catan_core::action_name(best));
        assert!(s.root_value()[0] > 0.9);
    }

    #[test]
    fn determinize_conserves_cards() {
        let mut rng = Rng::new(5);
        for seed in 0..50 {
            let mut st = State::new(Config::default(), seed);
            while !st.is_over() && st.turn < 120 {
                st.step(catan_core::bots::heuristic_action(&st, &mut rng));
            }
            let viewer = st.actor();
            let before: Vec<u32> = (0..4).map(|p| st.dev_total(p)).collect();
            let mut deck_before = [0u32; 5];
            for p in 0..4 {
                for c in 0..5 {
                    deck_before[c] += st.dev_hand[p][c] as u32;
                }
            }
            for &c in &st.dev_deck[..st.dev_deck_len as usize] {
                deck_before[c as usize] += 1;
            }
            let own = st.dev_hand[viewer];
            let mut d = st;
            determinize(&mut d, viewer, &mut rng);
            assert_eq!(d.dev_hand[viewer], own);
            for p in 0..4 {
                assert_eq!(d.dev_total(p), before[p]);
                for c in 0..5 {
                    assert!(d.dev_new[p][c] <= d.dev_hand[p][c]);
                }
            }
            let mut after = [0u32; 5];
            for p in 0..4 {
                for c in 0..5 {
                    after[c] += d.dev_hand[p][c] as u32;
                }
            }
            for &c in &d.dev_deck[..d.dev_deck_len as usize] {
                after[c as usize] += 1;
            }
            assert_eq!(after, deck_before);
        }
    }
}
