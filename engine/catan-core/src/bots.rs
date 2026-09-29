//! Baseline bots: uniform-random over legal actions, and a greedy rule-based heuristic.
//! They serve as test drivers, opponents for early training and fixed evaluation benchmarks.

use crate::actions::*;
use crate::board::*;
use crate::rng::Rng;
use crate::state::*;
use crate::topology::*;

pub fn random_action(s: &State, rng: &mut Rng) -> usize {
    let m = s.legal_mask();
    let n = mask_count(&m);
    debug_assert!(n > 0);
    let k = rng.below(n) as usize;
    mask_iter(&m).nth(k).unwrap()
}

/// Pip production per resource for player `p`, counting cities twice and skipping the robber.
fn production(s: &State, p: usize) -> [u32; 5] {
    let mut prod = [0u32; 5];
    for h in 0..N_HEX {
        let r = s.board.hex_res[h];
        if r == DESERT || h == s.robber as usize {
            continue;
        }
        let vm = TOPO.hex_vmask[h];
        let c = (s.settlements[p] & vm).count_ones() + 2 * (s.cities[p] & vm).count_ones();
        prod[r as usize] += c * pips(s.board.hex_num[h]) as u32;
    }
    prod
}

fn vertex_pips(s: &State, v: usize) -> u32 {
    bits32(TOPO.vertex_hexes[v]).map(|h| pips(s.board.hex_num[h]) as u32).sum()
}

/// Value of settling at `v` for player `p`, favouring pips and resources `p` lacks.
fn settle_value(s: &State, _p: usize, v: usize, prod: &[u32; 5]) -> f32 {
    let mut val = 0.0;
    for h in bits32(TOPO.vertex_hexes[v]) {
        let r = s.board.hex_res[h];
        if r == DESERT {
            continue;
        }
        let pip = pips(s.board.hex_num[h]) as f32;
        let scarcity = if prod[r as usize] == 0 { 1.6 } else { 1.0 };
        let weight = [1.0, 1.0, 0.8, 1.1, 1.1][r as usize];
        val += pip * scarcity * weight;
    }
    let bit = 1u64 << v;
    for r in 0..5 {
        if s.board.port_mask[r] & bit != 0 {
            val += 0.3 * prod[r] as f32;
        }
    }
    if s.board.port_mask[PORT_ANY as usize] & bit != 0 {
        val += 1.0;
    }
    val
}

fn best_by<I: Iterator<Item = usize>>(it: I, mut f: impl FnMut(usize) -> f32) -> Option<usize> {
    let mut best = None;
    let mut best_v = f32::NEG_INFINITY;
    for x in it {
        let v = f(x);
        if v > best_v {
            best_v = v;
            best = Some(x);
        }
    }
    best
}

/// Score a road by the best free settlement spot it brings within reach.
fn road_value(s: &State, p: usize, e: usize, free: u64, prod: &[u32; 5]) -> f32 {
    let [a, b] = TOPO.edge_vertices[e];
    let mut best = 0.0f32;
    for w in [a as usize, b as usize] {
        if free >> w & 1 == 1 {
            best = best.max(settle_value(s, p, w, prod));
        }
        for u in bits64(TOPO.vertex_neighbors[w] & free) {
            best = best.max(0.8 * settle_value(s, p, u, prod));
        }
    }
    best
}

fn robber_hex(s: &State, p: usize) -> usize {
    best_by((0..N_HEX).filter(|&h| h != s.robber as usize), |h| {
        let vm = TOPO.hex_vmask[h];
        if s.buildings(p) & vm != 0 {
            return -100.0;
        }
        let pip = pips(s.board.hex_num[h]) as f32;
        let mut score = 0.0;
        for q in 0..s.n() {
            if q == p {
                continue;
            }
            let c = (s.settlements[q] & vm).count_ones() + 2 * (s.cities[q] & vm).count_ones();
            score += c as f32 * pip * (1.0 + s.public_vp(q) as f32 / 4.0);
            if c > 0 && s.hand_total(q) > 0 {
                score += 0.5;
            }
        }
        score
    })
    .unwrap()
}

/// Missing cards to afford `cost`, and whether one maritime trade can cover it.
fn trade_toward(s: &State, p: usize, cost: &[u8; 5], m: &Mask) -> Option<usize> {
    let hand = s.hands[p];
    let missing: Vec<usize> = (0..5).filter(|&r| hand[r] < cost[r]).collect();
    let short: u8 = missing.iter().map(|&r| cost[r] - hand[r]).sum();
    if short != 1 {
        return None;
    }
    let need = missing[0];
    let ratios = s.trade_ratios(p);
    best_by((0..5).filter(|&g| g != need && hand[g] >= cost[g] + ratios[g]), |g| {
        (hand[g] - cost[g]) as f32 - ratios[g] as f32
    })
    .map(|g| trade_id(g, need))
    .filter(|&a| mask_has(m, a))
}

pub fn heuristic_action(s: &State, rng: &mut Rng) -> usize {
    let m = s.legal_mask();
    let p = s.actor();
    let prod = production(s, p);
    match s.phase {
        Phase::SetupSettlement => {
            best_by(mask_iter(&m), |a| settle_value(s, p, a - SETTLE, &prod) + rng.below(100) as f32 * 1e-3).unwrap()
        }
        Phase::SetupRoad | Phase::RoadBuilding => {
            let free = s.settle_candidates(p, false);
            best_by(mask_iter(&m), |a| road_value(s, p, a - ROAD, free, &prod) + rng.below(100) as f32 * 1e-3)
                .unwrap()
        }
        Phase::Roll => {
            let robbed = s.buildings(p) & TOPO.hex_vmask[s.robber as usize] != 0;
            if mask_has(&m, PLAY_KNIGHT) && robbed {
                PLAY_KNIGHT
            } else {
                ROLL
            }
        }
        Phase::Discard => {
            // Throw away whatever we hold most of.
            best_by(mask_iter(&m), |a| s.hands[p][a - DISCARD] as f32 + rng.below(10) as f32 * 0.01).unwrap()
        }
        Phase::MoveRobber => MOVE_ROBBER + robber_hex(s, p),
        Phase::Steal => {
            best_by(mask_iter(&m), |a| s.hand_total((p + a - STEAL) % s.n()) as f32).unwrap()
        }
        Phase::Main => heuristic_main(s, p, &m, &prod, rng),
        Phase::GameOver => unreachable!("no action in a finished game"),
    }
}

fn heuristic_main(s: &State, p: usize, m: &Mask, prod: &[u32; 5], rng: &mut Rng) -> usize {
    // 1. Cities on the best-producing settlement.
    if let Some(a) = best_by(mask_iter(m).filter(|&a| (CITY..ROAD).contains(&a)), |a| vertex_pips(s, a - CITY) as f32) {
        return a;
    }
    // 2. Settlements.
    if let Some(a) = best_by(mask_iter(m).filter(|&a| (SETTLE..CITY).contains(&a)), |a| settle_value(s, p, a - SETTLE, prod)) {
        return a;
    }
    // 3. Development cards that help now.
    if s.robber != s.board.desert
        && mask_has(m, PLAY_KNIGHT)
        && (s.buildings(p) & TOPO.hex_vmask[s.robber as usize] != 0 || s.knights[p] >= 2)
    {
        return PLAY_KNIGHT;
    }
    if mask_has(m, PLAY_MONOPOLY) {
        let r = (0..5)
            .max_by_key(|&r| (0..s.n()).filter(|&q| q != p).map(|q| s.hands[q][r] as u32).sum::<u32>())
            .unwrap();
        let total: u32 = (0..s.n()).filter(|&q| q != p).map(|q| s.hands[q][r] as u32).sum();
        if total >= 3 {
            return PLAY_MONOPOLY + r;
        }
    }
    let has_spot = s.settlements_left(p) > 0 && s.settle_candidates(p, true) != 0;
    let target: &[u8; 5] = if s.cities_left(p) > 0 && s.settlements[p] != 0 {
        &COST_CITY
    } else if has_spot {
        &COST_SETTLEMENT
    } else {
        &COST_DEV
    };
    if let Some(a) = mask_iter(m).filter(|&a| (PLAY_YOP..MOVE_ROBBER).contains(&a)).find(|&a| {
        let (x, y) = YOP_PAIRS[a - PLAY_YOP];
        let mut h = s.hands[p];
        h[x as usize] += 1;
        h[y as usize] += 1;
        (0..5).all(|r| h[r] >= target[r])
    }) {
        return a;
    }
    // 4. Roads toward a new settlement spot (or to contest longest road).
    let free = s.settle_candidates(p, false);
    if s.settlements_left(p) > 0 && !has_spot {
        if let Some(a) = best_by(mask_iter(m).filter(|&a| (ROAD..BUY_DEV).contains(&a)), |a| {
            road_value(s, p, a - ROAD, free, prod) + rng.below(100) as f32 * 1e-3
        }) {
            if road_value(s, p, a - ROAD, free, prod) > 0.0 {
                return a;
            }
        }
    }
    if mask_has(m, PLAY_ROAD_BUILDING) {
        return PLAY_ROAD_BUILDING;
    }
    // 5. Maritime trades that complete a build.
    for cost in [&COST_CITY, &COST_SETTLEMENT, &COST_DEV] {
        if std::ptr::eq(cost, &COST_SETTLEMENT) && !has_spot {
            continue;
        }
        if let Some(a) = trade_toward(s, p, cost, m) {
            return a;
        }
    }
    // 6. Development cards with spare resources.
    if mask_has(m, BUY_DEV) {
        return BUY_DEV;
    }
    // 7. Extend roads when holding lots of wood/brick.
    if s.hands[p][WOOD as usize] >= 2 && s.hands[p][BRICK as usize] >= 2 {
        if let Some(a) = best_by(mask_iter(m).filter(|&a| (ROAD..BUY_DEV).contains(&a)), |a| {
            road_value(s, p, a - ROAD, free, prod) + rng.below(100) as f32 * 1e-3
        }) {
            return a;
        }
    }
    END_TURN
}
