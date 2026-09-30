//! AlphaZero-style MCTS (PUCT) with an external, batched evaluator.
//!
//! The search is driven in two halves so many games can share one network batch:
//! `select()` walks the tree to a leaf that needs an evaluation and returns it; the caller evaluates
//! the leaf (priors over the legal actions + per-seat win probabilities) and hands the result to
//! `expand()`, which expands the leaf and backs the value up. Terminal leaves are backed up inside
//! `select()` without an evaluation.
//!
//! Chance (dice, dev-card draws, steals) is handled by sampling: every traversal of an edge replays
//! the action on a copy of the parent state with a fresh RNG seed, and each distinct resulting state
//! gets its own child node. Outcomes are therefore visited in proportion to their probability,
//! and edge statistics average over them.
//!
//! Hidden information: before searching, `determinize` reshuffles what the searching player cannot
//! know (opponents' unplayed dev cards and the deck order). Resource hands are treated as known.

use crate::actions::*;
use crate::rng::Rng;
use crate::state::*;

#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub sims: u32,
    pub c_puct: f32,
    /// Unvisited edges start at the node's mean value minus this.
    pub fpu_reduction: f32,
    pub dirichlet_alpha: f32,
    /// Weight of Dirichlet noise mixed into the root priors (0 disables it).
    pub noise_frac: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig { sims: 64, c_puct: 1.5, fpu_reduction: 0.1, dirichlet_alpha: 0.3, noise_frac: 0.25 }
    }
}

struct Edge {
    action: u16,
    prior: f32,
    n: u32,
    w: [f32; MAX_P],
    /// Children by outcome key (one for deterministic actions, several after chance).
    children: Vec<(u64, u32)>,
}

struct Node {
    state: State,
    expanded: bool,
    n: u32,
    w: [f32; MAX_P],
    edges: Vec<Edge>,
}

impl Node {
    fn new(state: State) -> Self {
        Node { state, expanded: false, n: 0, w: [0.0; MAX_P], edges: Vec::new() }
    }
}

pub struct Search {
    nodes: Vec<Node>,
    path: Vec<(u32, u32)>,
    leaf: Option<u32>,
    pub sims_done: u32,
    rng: Rng,
    cfg: MctsConfig,
}

impl Search {
    /// Search from `root` on behalf of its actor (the state should already be determinized).
    pub fn new(root: State, cfg: MctsConfig, seed: u64) -> Self {
        Search { nodes: vec![Node::new(root)], path: Vec::new(), leaf: None, sims_done: 0, rng: Rng::new(seed), cfg }
    }

    pub fn done(&self) -> bool {
        self.sims_done >= self.cfg.sims
    }

    pub fn root_state(&self) -> &State {
        &self.nodes[0].state
    }

    /// Run traversals until one reaches a leaf that needs evaluation (returned) or the simulation
    /// budget is used up (None).
    pub fn select(&mut self) -> Option<&State> {
        debug_assert!(self.leaf.is_none(), "select() called twice without expand()");
        while !self.done() {
            self.path.clear();
            let mut node = 0u32;
            loop {
                let nd = &self.nodes[node as usize];
                if nd.state.is_over() {
                    let v = terminal_value(&nd.state);
                    self.backup(node, &v);
                    break;
                }
                if !nd.expanded {
                    self.leaf = Some(node);
                    return Some(&self.nodes[node as usize].state);
                }
                let e = self.pick_edge(node);
                node = self.child(node, e);
            }
        }
        None
    }

    /// Expand the pending leaf with `priors` (indexed by action id, only legal entries are read) and
    /// back up `value` (win probability per absolute seat).
    pub fn expand(&mut self, priors: &[f32], value: &[f32; MAX_P]) {
        let leaf = self.leaf.take().expect("expand() without a pending leaf");
        let is_root = leaf == 0;
        let nd = &mut self.nodes[leaf as usize];
        let mask = nd.state.legal_mask();
        let mut edges: Vec<Edge> = mask_iter(&mask)
            .map(|a| Edge { action: a as u16, prior: priors[a].max(0.0), n: 0, w: [0.0; MAX_P], children: Vec::new() })
            .collect();
        let total: f32 = edges.iter().map(|e| e.prior).sum();
        for e in edges.iter_mut() {
            e.prior = if total > 0.0 { e.prior / total } else { 1.0 / mask_count(&mask) as f32 };
        }
        if is_root && self.cfg.noise_frac > 0.0 && edges.len() > 1 {
            let noise = dirichlet(&mut self.rng, self.cfg.dirichlet_alpha, edges.len());
            for (e, x) in edges.iter_mut().zip(noise) {
                e.prior = (1.0 - self.cfg.noise_frac) * e.prior + self.cfg.noise_frac * x;
            }
        }
        nd.edges = edges;
        nd.expanded = true;
        self.backup(leaf, value);
    }

    /// Root visit counts per action.
    pub fn visits(&self) -> Vec<(usize, u32)> {
        self.nodes[0].edges.iter().map(|e| (e.action as usize, e.n)).collect()
    }

    /// Mean backed-up value of the root per absolute seat.
    pub fn root_value(&self) -> [f32; MAX_P] {
        let r = &self.nodes[0];
        let n = r.n.max(1) as f32;
        r.w.map(|w| w / n)
    }

    fn pick_edge(&self, node: u32) -> u32 {
        let nd = &self.nodes[node as usize];
        let actor = nd.state.actor();
        let sqrt_n = (nd.n as f32).sqrt();
        let fpu = if nd.n > 0 { nd.w[actor] / nd.n as f32 - self.cfg.fpu_reduction } else { 0.0 };
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (i, e) in nd.edges.iter().enumerate() {
            let q = if e.n > 0 { e.w[actor] / e.n as f32 } else { fpu };
            let score = q + self.cfg.c_puct * e.prior * sqrt_n / (1.0 + e.n as f32);
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best as u32
    }

    /// Follow edge `e` of `node`: replay the action with a fresh seed and find or create the child.
    fn child(&mut self, node: u32, e: u32) -> u32 {
        let mut s = self.nodes[node as usize].state;
        s.rng = Rng::new(self.rng.next_u64());
        let action = self.nodes[node as usize].edges[e as usize].action;
        s.step(action as usize);
        let key = state_key(&s);
        self.path.push((node, e));
        let edge = &self.nodes[node as usize].edges[e as usize];
        if let Some(&(_, c)) = edge.children.iter().find(|(k, _)| *k == key) {
            return c;
        }
        let c = self.nodes.len() as u32;
        self.nodes.push(Node::new(s));
        self.nodes[node as usize].edges[e as usize].children.push((key, c));
        c
    }

    fn backup(&mut self, leaf: u32, v: &[f32; MAX_P]) {
        let add = |w: &mut [f32; MAX_P]| {
            for (a, b) in w.iter_mut().zip(v) {
                *a += b;
            }
        };
        let l = &mut self.nodes[leaf as usize];
        l.n += 1;
        add(&mut l.w);
        for &(node, e) in &self.path {
            let nd = &mut self.nodes[node as usize];
            nd.n += 1;
            add(&mut nd.w);
            let ed = &mut nd.edges[e as usize];
            ed.n += 1;
            add(&mut ed.w);
        }
        self.sims_done += 1;
    }
}

/// Win probability per seat of a finished game (a draw splits evenly).
pub fn terminal_value(s: &State) -> [f32; MAX_P] {
    let mut v = [0.0; MAX_P];
    if s.winner >= 0 {
        v[s.winner as usize] = 1.0;
    } else {
        for x in v.iter_mut().take(s.n()) {
            *x = 1.0 / s.n() as f32;
        }
    }
    v
}

/// Hash of everything that distinguishes game positions (excludes the RNG state).
pub fn state_key(s: &State) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0000_0100_0000_01b3).rotate_left(29);
    };
    for p in 0..s.n() {
        mix(s.settlements[p]);
        mix(s.cities[p]);
        mix(s.roads[p] as u64);
        mix((s.roads[p] >> 64) as u64);
        mix(u64::from_le_bytes([s.hands[p][0], s.hands[p][1], s.hands[p][2], s.hands[p][3], s.hands[p][4], s.knights[p], s.discard_need[p], 0]));
        mix(u64::from_le_bytes([s.dev_hand[p][0], s.dev_hand[p][1], s.dev_hand[p][2], s.dev_hand[p][3], s.dev_hand[p][4], s.dev_new[p][0], s.dev_new[p][2], s.dev_new[p][3]]));
    }
    mix(u64::from_le_bytes([
        s.robber,
        s.cur,
        s.phase as u8,
        s.rolled as u8 | (s.dev_played as u8) << 1,
        s.free_roads,
        s.dev_deck_len,
        s.last_roll[0] + s.last_roll[1],
        s.winner as u8,
    ]));
    mix(s.turn as u64 | (s.discarder as u64) << 16 | (s.longest_road as u8 as u64) << 24 | (s.largest_army as u8 as u64) << 32);
    h
}

/// Resample what `viewer` cannot know: opponents' unplayed, not-just-bought dev cards and the
/// order of the deck are shuffled together and dealt back with the same counts.
pub fn determinize(s: &mut State, viewer: usize, rng: &mut Rng) {
    let n = s.n();
    let mut pool: Vec<u8> = s.dev_deck[..s.dev_deck_len as usize].to_vec();
    let mut counts = [0u32; MAX_P];
    for q in (0..n).filter(|&q| q != viewer) {
        for c in 0..5 {
            let hidden = s.dev_hand[q][c] - s.dev_new[q][c];
            counts[q] += hidden as u32;
            s.dev_hand[q][c] -= hidden;
            pool.extend(std::iter::repeat_n(c as u8, hidden as usize));
        }
    }
    rng.shuffle(&mut pool);
    let mut it = pool.into_iter();
    for q in (0..n).filter(|&q| q != viewer) {
        for _ in 0..counts[q] {
            let c = it.next().unwrap() as usize;
            s.dev_hand[q][c] += 1;
        }
    }
    let rest: Vec<u8> = it.collect();
    s.dev_deck[..rest.len()].copy_from_slice(&rest);
    debug_assert_eq!(rest.len(), s.dev_deck_len as usize);
    s.rng = Rng::new(rng.next_u64());
}

fn gamma(rng: &mut Rng, alpha: f32) -> f32 {
    // Marsaglia-Tsang, with the alpha < 1 boost.
    if alpha < 1.0 {
        let u = uniform(rng);
        return gamma(rng, alpha + 1.0) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = uniform(rng);
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

fn uniform(rng: &mut Rng) -> f32 {
    ((rng.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
}

fn normal(rng: &mut Rng) -> f32 {
    let (u1, u2) = (uniform(rng), uniform(rng));
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn dirichlet(rng: &mut Rng, alpha: f32, k: usize) -> Vec<f32> {
    let g: Vec<f32> = (0..k).map(|_| gamma(rng, alpha)).collect();
    let s: f32 = g.iter().sum::<f32>().max(1e-12);
    g.into_iter().map(|x| x / s).collect()
}
