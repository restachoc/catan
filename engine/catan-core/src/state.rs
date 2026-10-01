//! Game state and rules. `State` is `Copy` (a few hundred bytes, no heap), so cloning for search
//! or snapshots is essentially free.

use crate::actions::*;
use crate::board::*;
use crate::rng::Rng;
use crate::topology::*;

pub const MAX_P: usize = 4;

pub const COST_ROAD: [u8; 5] = [1, 1, 0, 0, 0];
pub const COST_SETTLEMENT: [u8; 5] = [1, 1, 1, 1, 0];
pub const COST_CITY: [u8; 5] = [0, 0, 0, 2, 3];
pub const COST_DEV: [u8; 5] = [0, 0, 1, 1, 1];

pub const DEV_KNIGHT: usize = 0;
pub const DEV_VP: usize = 1;
pub const DEV_ROAD_BUILDING: usize = 2;
pub const DEV_YOP: usize = 3;
pub const DEV_MONOPOLY: usize = 4;

pub const MAX_SETTLEMENTS: u32 = 5;
pub const MAX_CITIES: u32 = 4;
pub const MAX_ROADS: u32 = 15;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub n_players: u8,
    pub vp_target: u8,
    pub random_board: bool,
    /// Turn cap; reaching it ends the game as a draw (winner = -1).
    pub max_turns: u16,
}

impl Default for Config {
    fn default() -> Self {
        Config { n_players: 4, vp_target: 10, random_board: false, max_turns: 500 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Phase {
    SetupSettlement,
    SetupRoad,
    Roll,
    Main,
    Discard,
    MoveRobber,
    Steal,
    RoadBuilding,
    GameOver,
}
pub const N_PHASES: usize = 9;

#[derive(Clone, Copy, Debug)]
pub struct State {
    pub cfg: Config,
    pub board: Board,
    /// Chance streams, separate so that one source's draws never shift another's: the k-th roll of a game
    /// is the same whatever the players do (common random numbers for evaluations). Board and dev deck
    /// come from a third, setup-only stream in `new`.
    pub dice: Rng,
    pub steal_rng: Rng,
    pub settlements: [u64; MAX_P],
    pub cities: [u64; MAX_P],
    pub roads: [u128; MAX_P],
    pub hands: [[u8; 5]; MAX_P],
    pub bank: [u8; 5],
    pub dev_hand: [[u8; 5]; MAX_P],
    /// Dev cards bought this turn (not yet playable).
    pub dev_new: [[u8; 5]; MAX_P],
    pub dev_deck: [u8; 25],
    pub dev_deck_len: u8,
    pub knights: [u8; MAX_P],
    pub road_len: [u8; MAX_P],
    pub longest_road: i8,
    pub largest_army: i8,
    pub robber: u8,
    pub cur: u8,
    pub phase: Phase,
    pub turn: u16,
    pub rolled: bool,
    pub dev_played: bool,
    pub setup_step: u8,
    pub last_settlement: u8,
    pub free_roads: u8,
    pub discard_need: [u8; MAX_P],
    pub discarder: u8,
    pub last_roll: [u8; 2],
    /// Bookkeeping for strategy analysis (not part of the observation).
    pub dev_bought: [u8; MAX_P],
    /// Pips per resource of each player's buildings right after setup.
    pub opening_pips: [[u8; 5]; MAX_P],
    pub winner: i8,
}

const DICE_STREAM: u64 = 0xD1CE_5EED_0000_0001;
const STEAL_STREAM: u64 = 0x57EA_15EE_D000_0002;

impl State {
    /// Replace all in-game chance streams (dice, steals) with ones derived from `seed`. Used by search to
    /// sample a different outcome per edge traversal.
    pub fn reseed_chance(&mut self, seed: u64) {
        self.dice = Rng::new(seed ^ DICE_STREAM);
        self.steal_rng = Rng::new(seed ^ STEAL_STREAM);
    }

    pub fn new(cfg: Config, seed: u64) -> Self {
        assert!((2..=4).contains(&cfg.n_players));
        let mut rng = Rng::new(seed);
        let board = if cfg.random_board { Board::random(&mut rng) } else { Board::beginner() };
        let mut dev_deck = [0u8; 25];
        let counts = [(DEV_KNIGHT, 14), (DEV_VP, 5), (DEV_ROAD_BUILDING, 2), (DEV_YOP, 2), (DEV_MONOPOLY, 2)];
        let mut i = 0;
        for (card, n) in counts {
            for _ in 0..n {
                dev_deck[i] = card as u8;
                i += 1;
            }
        }
        rng.shuffle(&mut dev_deck);
        State {
            cfg,
            board,
            dice: Rng::new(seed ^ DICE_STREAM),
            steal_rng: Rng::new(seed ^ STEAL_STREAM),
            settlements: [0; MAX_P],
            cities: [0; MAX_P],
            roads: [0; MAX_P],
            hands: [[0; 5]; MAX_P],
            bank: [19; 5],
            dev_hand: [[0; 5]; MAX_P],
            dev_new: [[0; 5]; MAX_P],
            dev_deck,
            dev_deck_len: 25,
            knights: [0; MAX_P],
            road_len: [0; MAX_P],
            longest_road: -1,
            largest_army: -1,
            robber: board.desert,
            cur: 0,
            phase: Phase::SetupSettlement,
            turn: 0,
            rolled: false,
            dev_played: false,
            setup_step: 0,
            last_settlement: 0,
            free_roads: 0,
            discard_need: [0; MAX_P],
            discarder: 0,
            last_roll: [0, 0],
            dev_bought: [0; MAX_P],
            opening_pips: [[0; 5]; MAX_P],
            winner: -1,
        }
    }

    // ---------------------------------------------------------------- queries

    #[inline]
    pub fn n(&self) -> usize {
        self.cfg.n_players as usize
    }

    /// The player who must act now (differs from `cur` only while discarding).
    #[inline]
    pub fn actor(&self) -> usize {
        if self.phase == Phase::Discard {
            self.discarder as usize
        } else {
            self.cur as usize
        }
    }

    #[inline]
    pub fn is_over(&self) -> bool {
        self.phase == Phase::GameOver
    }

    #[inline]
    pub fn buildings(&self, p: usize) -> u64 {
        self.settlements[p] | self.cities[p]
    }

    #[inline]
    pub fn occupied(&self) -> u64 {
        (0..self.n()).fold(0, |m, p| m | self.buildings(p))
    }

    #[inline]
    pub fn all_roads(&self) -> u128 {
        (0..self.n()).fold(0, |m, p| m | self.roads[p])
    }

    pub fn hand_total(&self, p: usize) -> u32 {
        self.hands[p].iter().map(|&c| c as u32).sum()
    }

    pub fn dev_total(&self, p: usize) -> u32 {
        self.dev_hand[p].iter().map(|&c| c as u32).sum()
    }

    pub fn settlements_left(&self, p: usize) -> u32 {
        MAX_SETTLEMENTS - self.settlements[p].count_ones()
    }
    pub fn cities_left(&self, p: usize) -> u32 {
        MAX_CITIES - self.cities[p].count_ones()
    }
    pub fn roads_left(&self, p: usize) -> u32 {
        MAX_ROADS - self.roads[p].count_ones()
    }

    /// Victory points visible to everyone (excludes hidden VP cards).
    pub fn public_vp(&self, p: usize) -> u32 {
        self.settlements[p].count_ones()
            + 2 * self.cities[p].count_ones()
            + if self.longest_road == p as i8 { 2 } else { 0 }
            + if self.largest_army == p as i8 { 2 } else { 0 }
    }

    pub fn vp(&self, p: usize) -> u32 {
        self.public_vp(p) + self.dev_hand[p][DEV_VP] as u32
    }

    #[inline]
    pub fn can_afford(&self, p: usize, cost: &[u8; 5]) -> bool {
        (0..5).all(|r| self.hands[p][r] >= cost[r])
    }

    pub fn trade_ratios(&self, p: usize) -> [u8; 5] {
        let b = self.buildings(p);
        let base = if b & self.board.port_mask[PORT_ANY as usize] != 0 { 3 } else { 4 };
        let mut r = [base; 5];
        for (res, ratio) in r.iter_mut().enumerate() {
            if b & self.board.port_mask[res] != 0 {
                *ratio = 2;
            }
        }
        r
    }

    /// Pips per resource produced by `p`'s buildings (cities count twice; robber ignored).
    pub fn production_pips(&self, p: usize) -> [u8; 5] {
        let mut out = [0u8; 5];
        for h in 0..N_HEX {
            let r = self.board.hex_res[h];
            if r == DESERT {
                continue;
            }
            let vm = TOPO.hex_vmask[h];
            let c = (self.settlements[p] & vm).count_ones() + 2 * (self.cities[p] & vm).count_ones();
            out[r as usize] += (c * pips(self.board.hex_num[h]) as u32) as u8;
        }
        out
    }

    fn road_vertices(&self, p: usize) -> u64 {
        bits128(self.roads[p]).fold(0, |m, e| {
            let [a, b] = TOPO.edge_vertices[e];
            m | (1 << a) | (1 << b)
        })
    }

    /// Vertices where `p` may place a settlement (distance rule; optionally road-connected).
    pub fn settle_candidates(&self, p: usize, need_road: bool) -> u64 {
        let occ = self.occupied();
        let blocked = bits64(occ).fold(occ, |m, v| m | TOPO.vertex_neighbors[v]);
        let mut cand = !blocked & ((1u64 << N_VERT) - 1);
        if need_road {
            cand &= self.road_vertices(p);
        }
        cand
    }

    /// Empty edges where `p` may place a road.
    pub fn road_candidates(&self, p: usize) -> u128 {
        let own = self.buildings(p);
        let opp = self.occupied() & !own;
        let frontier = own | (self.road_vertices(p) & !opp);
        bits64(frontier).fold(0u128, |m, v| m | TOPO.vertex_edges[v]) & !self.all_roads()
    }

    fn is_steal_victim(&self, v: usize) -> bool {
        v != self.cur as usize
            && self.hand_total(v) > 0
            && self.buildings(v) & TOPO.hex_vmask[self.robber as usize] != 0
    }

    fn setup_player(&self, step: u8) -> u8 {
        let n = self.cfg.n_players;
        if step < n {
            step
        } else {
            2 * n - 1 - step
        }
    }

    // ---------------------------------------------------------------- legality

    pub fn legal_mask(&self) -> Mask {
        let mut m: Mask = [0; 4];
        let p = self.actor();
        match self.phase {
            Phase::SetupSettlement => {
                for v in bits64(self.settle_candidates(p, false)) {
                    mask_set(&mut m, SETTLE + v);
                }
            }
            Phase::SetupRoad => {
                let edges = TOPO.vertex_edges[self.last_settlement as usize] & !self.all_roads();
                for e in bits128(edges) {
                    mask_set(&mut m, ROAD + e);
                }
            }
            Phase::Roll => {
                mask_set(&mut m, ROLL);
                self.dev_mask(p, &mut m);
            }
            Phase::Main => {
                mask_set(&mut m, END_TURN);
                if self.settlements_left(p) > 0 && self.can_afford(p, &COST_SETTLEMENT) {
                    for v in bits64(self.settle_candidates(p, true)) {
                        mask_set(&mut m, SETTLE + v);
                    }
                }
                if self.cities_left(p) > 0 && self.can_afford(p, &COST_CITY) {
                    for v in bits64(self.settlements[p]) {
                        mask_set(&mut m, CITY + v);
                    }
                }
                if self.roads_left(p) > 0 && self.can_afford(p, &COST_ROAD) {
                    for e in bits128(self.road_candidates(p)) {
                        mask_set(&mut m, ROAD + e);
                    }
                }
                if self.dev_deck_len > 0 && self.can_afford(p, &COST_DEV) {
                    mask_set(&mut m, BUY_DEV);
                }
                self.dev_mask(p, &mut m);
                let ratios = self.trade_ratios(p);
                for give in 0..5 {
                    if self.hands[p][give] >= ratios[give] {
                        for get in 0..5 {
                            if get != give && self.bank[get] > 0 {
                                mask_set(&mut m, trade_id(give, get));
                            }
                        }
                    }
                }
            }
            Phase::Discard => {
                for r in 0..5 {
                    if self.hands[p][r] > 0 {
                        mask_set(&mut m, DISCARD + r);
                    }
                }
            }
            Phase::MoveRobber => {
                for h in 0..N_HEX {
                    if h != self.robber as usize {
                        mask_set(&mut m, MOVE_ROBBER + h);
                    }
                }
            }
            Phase::Steal => {
                for k in 1..self.n() {
                    if self.is_steal_victim((p + k) % self.n()) {
                        mask_set(&mut m, STEAL + k);
                    }
                }
            }
            Phase::RoadBuilding => {
                for e in bits128(self.road_candidates(p)) {
                    mask_set(&mut m, ROAD + e);
                }
            }
            Phase::GameOver => {}
        }
        m
    }

    fn dev_mask(&self, p: usize, m: &mut Mask) {
        if self.dev_played {
            return;
        }
        let playable = |c: usize| self.dev_hand[p][c] > self.dev_new[p][c];
        if playable(DEV_KNIGHT) {
            mask_set(m, PLAY_KNIGHT);
        }
        if playable(DEV_ROAD_BUILDING) && self.roads_left(p) > 0 && self.road_candidates(p) != 0 {
            mask_set(m, PLAY_ROAD_BUILDING);
        }
        if playable(DEV_MONOPOLY) {
            for r in 0..5 {
                mask_set(m, PLAY_MONOPOLY + r);
            }
        }
        if playable(DEV_YOP) {
            for (i, &(a, b)) in YOP_PAIRS.iter().enumerate() {
                let ok = if a == b { self.bank[a as usize] >= 2 } else { self.bank[a as usize] >= 1 && self.bank[b as usize] >= 1 };
                if ok {
                    mask_set(m, PLAY_YOP + i);
                }
            }
        }
    }

    pub fn is_legal(&self, a: usize) -> bool {
        a < N_ACTIONS && mask_has(&self.legal_mask(), a)
    }

    // ---------------------------------------------------------------- transitions

    /// Apply a legal action. Legality is only checked in debug builds; use `try_step` for input
    /// from untrusted sources (the UI).
    pub fn step(&mut self, a: usize) {
        debug_assert!(self.is_legal(a), "illegal action {} in {:?}", action_name(a), self.phase);
        let p = self.actor();
        match a {
            ROLL => self.roll(),
            END_TURN => self.end_turn(),
            a if a < CITY => self.build_settlement(p, a - SETTLE),
            a if a < ROAD => {
                self.pay(p, &COST_CITY);
                let bit = 1u64 << (a - CITY);
                self.settlements[p] &= !bit;
                self.cities[p] |= bit;
            }
            a if a < BUY_DEV => self.build_road(p, a - ROAD),
            BUY_DEV => {
                self.pay(p, &COST_DEV);
                self.dev_bought[p] += 1;
                self.dev_deck_len -= 1;
                let c = self.dev_deck[self.dev_deck_len as usize] as usize;
                self.dev_hand[p][c] += 1;
                self.dev_new[p][c] += 1;
            }
            PLAY_KNIGHT => {
                self.use_dev(p, DEV_KNIGHT);
                self.knights[p] += 1;
                let la = self.largest_army;
                if self.knights[p] >= 3 && (la < 0 || self.knights[p] > self.knights[la as usize]) {
                    self.largest_army = p as i8;
                }
                self.phase = Phase::MoveRobber;
            }
            PLAY_ROAD_BUILDING => {
                self.use_dev(p, DEV_ROAD_BUILDING);
                self.free_roads = self.roads_left(p).min(2) as u8;
                self.phase = Phase::RoadBuilding;
            }
            a if a < PLAY_YOP => {
                self.use_dev(p, DEV_MONOPOLY);
                let r = a - PLAY_MONOPOLY;
                for q in 0..self.n() {
                    if q != p {
                        self.hands[p][r] += self.hands[q][r];
                        self.hands[q][r] = 0;
                    }
                }
            }
            a if a < MOVE_ROBBER => {
                self.use_dev(p, DEV_YOP);
                let (x, y) = YOP_PAIRS[a - PLAY_YOP];
                for r in [x as usize, y as usize] {
                    self.bank[r] -= 1;
                    self.hands[p][r] += 1;
                }
            }
            a if a < STEAL => self.move_robber(a - MOVE_ROBBER),
            a if a < DISCARD => {
                let victim = (p + a - STEAL) % self.n();
                self.steal(victim);
                self.phase = self.return_phase();
            }
            a if a < TRADE => self.discard(p, a - DISCARD),
            a if a < N_ACTIONS => {
                let (give, get) = trade_pair(a - TRADE);
                let ratio = self.trade_ratios(p)[give];
                self.hands[p][give] -= ratio;
                self.bank[give] += ratio;
                self.hands[p][get] += 1;
                self.bank[get] -= 1;
            }
            _ => unreachable!(),
        }
        if self.phase != Phase::GameOver {
            let c = self.cur as usize;
            if self.phase != Phase::SetupSettlement
                && self.phase != Phase::SetupRoad
                && self.vp(c) >= self.cfg.vp_target as u32
            {
                self.winner = c as i8;
                self.phase = Phase::GameOver;
            }
        }
    }

    pub fn try_step(&mut self, a: usize) -> Result<(), String> {
        if !self.is_legal(a) {
            return Err(format!("illegal action {} ({}) in phase {:?}", a, action_name(a), self.phase));
        }
        self.step(a);
        Ok(())
    }

    fn pay(&mut self, p: usize, cost: &[u8; 5]) {
        for r in 0..5 {
            self.hands[p][r] -= cost[r];
            self.bank[r] += cost[r];
        }
    }

    fn use_dev(&mut self, p: usize, c: usize) {
        self.dev_hand[p][c] -= 1;
        self.dev_played = true;
    }

    #[inline]
    fn return_phase(&self) -> Phase {
        if self.rolled {
            Phase::Main
        } else {
            Phase::Roll
        }
    }

    fn build_settlement(&mut self, p: usize, v: usize) {
        if self.phase == Phase::SetupSettlement {
            self.settlements[p] |= 1 << v;
            self.last_settlement = v as u8;
            if self.setup_step >= self.cfg.n_players {
                for h in bits32(TOPO.vertex_hexes[v]) {
                    let r = self.board.hex_res[h];
                    if r != DESERT {
                        self.hands[p][r as usize] += 1;
                        self.bank[r as usize] -= 1;
                    }
                }
            }
            self.phase = Phase::SetupRoad;
            return;
        }
        self.pay(p, &COST_SETTLEMENT);
        self.settlements[p] |= 1 << v;
        // A new settlement can cut an opponent's road.
        let mut changed = false;
        for q in 0..self.n() {
            if q != p && self.roads[q] & TOPO.vertex_edges[v] != 0 {
                self.road_len[q] = self.longest_road_of(q);
                changed = true;
            }
        }
        if changed {
            self.update_longest_road();
        }
    }

    fn build_road(&mut self, p: usize, e: usize) {
        match self.phase {
            Phase::SetupRoad => {
                self.roads[p] |= 1 << e;
                self.road_len[p] = self.longest_road_of(p);
                self.setup_step += 1;
                if self.setup_step == 2 * self.cfg.n_players {
                    for q in 0..self.n() {
                        self.opening_pips[q] = self.production_pips(q);
                    }
                    self.cur = 0;
                    self.phase = Phase::Roll;
                } else {
                    self.cur = self.setup_player(self.setup_step);
                    self.phase = Phase::SetupSettlement;
                }
                return;
            }
            Phase::RoadBuilding => {
                self.roads[p] |= 1 << e;
                self.free_roads -= 1;
                if self.free_roads == 0 || self.roads_left(p) == 0 || self.road_candidates(p) == 0 {
                    self.free_roads = 0;
                    self.phase = self.return_phase();
                }
            }
            _ => {
                self.pay(p, &COST_ROAD);
                self.roads[p] |= 1 << e;
            }
        }
        self.road_len[p] = self.longest_road_of(p);
        self.update_longest_road();
    }

    fn roll(&mut self) {
        let d1 = 1 + self.dice.below(6) as u8;
        let d2 = 1 + self.dice.below(6) as u8;
        self.last_roll = [d1, d2];
        self.rolled = true;
        let sum = d1 + d2;
        if sum == 7 {
            let mut any = false;
            for p in 0..self.n() {
                let t = self.hand_total(p);
                self.discard_need[p] = if t > 7 { (t / 2) as u8 } else { 0 };
                any |= t > 7;
            }
            if any {
                self.phase = Phase::Discard;
                self.discarder = self.next_discarder();
            } else {
                self.phase = Phase::MoveRobber;
            }
        } else {
            self.produce(sum);
            self.phase = Phase::Main;
        }
    }

    fn produce(&mut self, num: u8) {
        let n = self.n();
        let mut demand = [[0u8; 5]; MAX_P];
        let mut total = [0u8; 5];
        for h in bits32(self.board.num_hexes[num as usize]) {
            if h == self.robber as usize {
                continue;
            }
            let r = self.board.hex_res[h] as usize;
            let vm = TOPO.hex_vmask[h];
            for p in 0..n {
                let c = (self.settlements[p] & vm).count_ones() + 2 * (self.cities[p] & vm).count_ones();
                demand[p][r] += c as u8;
                total[r] += c as u8;
            }
        }
        for r in 0..5 {
            if total[r] == 0 {
                continue;
            }
            if total[r] <= self.bank[r] {
                for p in 0..n {
                    self.hands[p][r] += demand[p][r];
                }
                self.bank[r] -= total[r];
            } else {
                // Bank short: if only one player is owed, they get what is left; otherwise nobody.
                let mut owed = (0..n).filter(|&p| demand[p][r] > 0);
                if let (Some(p), None) = (owed.next(), owed.next()) {
                    self.hands[p][r] += self.bank[r];
                    self.bank[r] = 0;
                }
            }
        }
    }

    fn next_discarder(&self) -> u8 {
        let n = self.n();
        (0..n).map(|k| (self.cur as usize + k) % n).find(|&q| self.discard_need[q] > 0).unwrap() as u8
    }

    fn discard(&mut self, p: usize, r: usize) {
        self.hands[p][r] -= 1;
        self.bank[r] += 1;
        self.discard_need[p] -= 1;
        if self.discard_need[p] == 0 {
            if self.discard_need.iter().any(|&d| d > 0) {
                self.discarder = self.next_discarder();
            } else {
                self.phase = Phase::MoveRobber;
            }
        }
    }

    fn move_robber(&mut self, h: usize) {
        self.robber = h as u8;
        let n = self.n();
        let cur = self.cur as usize;
        let mut victims = (1..n).map(|k| (cur + k) % n).filter(|&v| self.is_steal_victim(v));
        match (victims.next(), victims.next()) {
            (None, _) => self.phase = self.return_phase(),
            (Some(v), None) => {
                self.steal(v);
                self.phase = self.return_phase();
            }
            _ => self.phase = Phase::Steal,
        }
    }

    fn steal(&mut self, victim: usize) {
        let total = self.hand_total(victim);
        if total == 0 {
            return;
        }
        let mut k = self.steal_rng.below(total);
        for r in 0..5 {
            let c = self.hands[victim][r] as u32;
            if k < c {
                self.hands[victim][r] -= 1;
                self.hands[self.cur as usize][r] += 1;
                return;
            }
            k -= c;
        }
    }

    fn end_turn(&mut self) {
        let c = self.cur as usize;
        self.dev_new[c] = [0; 5];
        self.dev_played = false;
        self.rolled = false;
        self.cur = ((c + 1) % self.n()) as u8;
        self.turn += 1;
        self.phase = Phase::Roll;
        if self.turn >= self.cfg.max_turns {
            self.phase = Phase::GameOver;
            self.winner = -1;
        }
    }

    // ---------------------------------------------------------------- longest road

    pub fn longest_road_of(&self, p: usize) -> u8 {
        let roads = self.roads[p];
        if roads == 0 {
            return 0;
        }
        let blocked = self.occupied() & !self.buildings(p);
        let mut best = 0;
        for v in bits64(self.road_vertices(p)) {
            best = best.max(road_dfs(v, 0, roads, blocked));
        }
        best
    }

    fn update_longest_road(&mut self) {
        let n = self.n();
        let lens = &self.road_len[..n];
        let holder = self.longest_road;
        if holder >= 0 {
            let h = holder as usize;
            if lens[h] >= 5 && lens.iter().all(|&l| l <= lens[h]) {
                return;
            }
        }
        let max = *lens.iter().max().unwrap();
        let count = lens.iter().filter(|&&l| l == max).count();
        self.longest_road = if max >= 5 && count == 1 {
            lens.iter().position(|&l| l == max).unwrap() as i8
        } else {
            -1
        };
    }
}

fn road_dfs(v: usize, used: u128, roads: u128, blocked: u64) -> u8 {
    let mut best = 0;
    for e in bits128(TOPO.vertex_edges[v] & roads & !used) {
        let [a, b] = TOPO.edge_vertices[e];
        let w = if a as usize == v { b } else { a } as usize;
        let len = 1 + if blocked >> w & 1 == 1 { 0 } else { road_dfs(w, used | 1 << e, roads, blocked) };
        best = best.max(len);
    }
    best
}
