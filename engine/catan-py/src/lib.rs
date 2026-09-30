//! Python bindings: a single `Game` for the UI and a batched, multi-threaded `VecEnv` for RL.

use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use catan_core::actions::{action_name, mask_iter};
use catan_core::bots::{heuristic_action, random_action};
use catan_core::mcts::{determinize, MctsConfig, Search};
use catan_core::rng::Rng;
use catan_core::stats::{player_stats, N_STATS, STAT_NAMES};
use catan_core::view::{board_view, state_view};
use catan_core::{write_obs, Config, State, N_ACTIONS, OBS_SIZE};

/// Who controls a seat inside a VecEnv.
const SEAT_EXTERNAL: u8 = 0;
const SEAT_RANDOM: u8 = 1;
const SEAT_HEURISTIC: u8 = 2;

fn make_config(n_players: u8, vp_target: u8, random_board: bool, max_turns: u16) -> PyResult<Config> {
    if !(2..=4).contains(&n_players) {
        return Err(PyValueError::new_err("n_players must be 2..=4"));
    }
    Ok(Config { n_players, vp_target, random_board, max_turns })
}

fn bot_action(kind: u8, s: &State, rng: &mut Rng) -> usize {
    match kind {
        SEAT_RANDOM => random_action(s, rng),
        _ => heuristic_action(s, rng),
    }
}

// ---------------------------------------------------------------- Game

#[pyclass(module = "catan_rl._engine", skip_from_py_object)]
#[derive(Clone)]
struct Game {
    state: State,
    seed: u64,
    history: Vec<u16>,
    bot_rng: Rng,
}

#[pymethods]
impl Game {
    #[new]
    #[pyo3(signature = (seed=0, n_players=4, vp_target=10, random_board=false, max_turns=500))]
    fn new(seed: u64, n_players: u8, vp_target: u8, random_board: bool, max_turns: u16) -> PyResult<Self> {
        let cfg = make_config(n_players, vp_target, random_board, max_turns)?;
        Ok(Game { state: State::new(cfg, seed), seed, history: vec![], bot_rng: Rng::new(seed ^ 0x5EED) })
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.seed
    }
    #[getter]
    fn actor(&self) -> usize {
        self.state.actor()
    }
    #[getter]
    fn current_player(&self) -> u8 {
        self.state.cur
    }
    #[getter]
    fn is_over(&self) -> bool {
        self.state.is_over()
    }
    #[getter]
    fn winner(&self) -> i8 {
        self.state.winner
    }
    #[getter]
    fn phase(&self) -> String {
        format!("{:?}", self.state.phase)
    }
    #[getter]
    fn history(&self) -> Vec<u16> {
        self.history.clone()
    }
    #[getter]
    fn n_players(&self) -> u8 {
        self.state.cfg.n_players
    }

    fn vp(&self, p: usize) -> u32 {
        self.state.vp(p)
    }

    fn legal_actions(&self) -> Vec<usize> {
        mask_iter(&self.state.legal_mask()).collect()
    }

    fn step(&mut self, action: usize) -> PyResult<()> {
        self.state.try_step(action).map_err(PyValueError::new_err)?;
        self.history.push(action as u16);
        Ok(())
    }

    /// Pick an action with a built-in bot ("heuristic" or "random") without applying it.
    #[pyo3(signature = (kind="heuristic"))]
    fn bot_action(&mut self, kind: &str) -> PyResult<usize> {
        if self.state.is_over() {
            return Err(PyValueError::new_err("game is over"));
        }
        let k = match kind {
            "random" => SEAT_RANDOM,
            "heuristic" => SEAT_HEURISTIC,
            _ => return Err(PyValueError::new_err("unknown bot kind")),
        };
        Ok(bot_action(k, &self.state, &mut self.bot_rng))
    }

    fn observation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mut obs = vec![0f32; OBS_SIZE];
        write_obs(&self.state, &mut obs);
        PyArray1::from_vec(py, obs)
    }

    fn action_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let mut m = vec![false; N_ACTIONS];
        for a in mask_iter(&self.state.legal_mask()) {
            m[a] = true;
        }
        PyArray1::from_vec(py, m)
    }

    fn board_json(&self) -> String {
        serde_json::to_string(&board_view(&self.state)).unwrap()
    }

    /// State as seen by `viewer` (None = omniscient).
    #[pyo3(signature = (viewer=None))]
    fn state_json(&self, viewer: Option<usize>) -> String {
        serde_json::to_string(&state_view(&self.state, viewer)).unwrap()
    }

    fn copy(&self) -> Self {
        self.clone()
    }
}

// ---------------------------------------------------------------- VecEnv

struct Slot {
    state: State,
    bot_rng: Rng,
    seats: [u8; 4],
    steps: u32,
    /// Per-seat end-of-game statistics of the most recently finished game.
    last_stats: [[f32; N_STATS]; 4],
}

/// Batched environments stepped in parallel. Seats marked as bots are played inside Rust, so
/// Python only ever sees states where an external (policy-controlled) seat must act.
///
/// All outputs are written into caller-owned numpy buffers:
///   obs [N, OBS_SIZE] f32, mask [N, N_ACTIONS] bool, actor [N] i64,
///   done [N] bool, winner [N] i64 (-1 draw / not done), length [N] i64 (steps of finished game),
///   final_vp [N, 4] i64 (victory points per seat of the finished game, incl. hidden cards).
#[pyclass(module = "catan_rl._engine")]
struct VecEnv {
    slots: Vec<Slot>,
    cfg: Config,
    next_seed: u64,
}

impl VecEnv {
    fn reset_slot(slot: &mut Slot, cfg: Config, seed: u64) {
        slot.state = State::new(cfg, seed);
        slot.bot_rng = Rng::new(seed ^ 0xB07);
        slot.steps = 0;
    }

    /// Let bot seats move until an external seat must act or the game ends.
    fn advance_bots(slot: &mut Slot) {
        while !slot.state.is_over() {
            let kind = slot.seats[slot.state.actor()];
            if kind == SEAT_EXTERNAL {
                break;
            }
            let a = bot_action(kind, &slot.state, &mut slot.bot_rng);
            slot.state.step(a);
            slot.steps += 1;
        }
    }

    fn write(slot: &Slot, obs: &mut [f32], mask: &mut [bool], actor: &mut i64) {
        write_obs(&slot.state, obs);
        mask.fill(false);
        for a in mask_iter(&slot.state.legal_mask()) {
            mask[a] = true;
        }
        *actor = slot.state.actor() as i64;
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (num_envs, seed=0, n_players=4, vp_target=10, random_board=false, max_turns=500))]
    fn new(num_envs: usize, seed: u64, n_players: u8, vp_target: u8, random_board: bool, max_turns: u16) -> PyResult<Self> {
        let cfg = make_config(n_players, vp_target, random_board, max_turns)?;
        let slots = (0..num_envs)
            .map(|i| Slot {
                state: State::new(cfg, seed + i as u64),
                bot_rng: Rng::new(seed + i as u64),
                seats: [SEAT_EXTERNAL; 4],
                steps: 0,
                last_stats: [[0.0; N_STATS]; 4],
            })
            .collect();
        Ok(VecEnv { slots, cfg, next_seed: seed + num_envs as u64 })
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.slots.len()
    }
    #[getter]
    fn n_players(&self) -> u8 {
        self.cfg.n_players
    }
    #[classattr]
    fn obs_size() -> usize {
        OBS_SIZE
    }
    #[classattr]
    fn n_actions() -> usize {
        N_ACTIONS
    }

    /// Seat controllers for one env: list of "external" / "random" / "heuristic".
    /// Takes effect immediately (bots will move at the next step/reset).
    fn set_seats(&mut self, env: usize, seats: Vec<String>) -> PyResult<()> {
        let slot = self.slots.get_mut(env).ok_or_else(|| PyValueError::new_err("env out of range"))?;
        for (i, s) in seats.iter().enumerate().take(4) {
            slot.seats[i] = match s.as_str() {
                "external" => SEAT_EXTERNAL,
                "random" => SEAT_RANDOM,
                "heuristic" => SEAT_HEURISTIC,
                _ => return Err(PyValueError::new_err(format!("unknown seat kind {s}"))),
            };
        }
        Ok(())
    }

    /// Reset every env (fresh seeds) and fill the buffers.
    fn reset(
        &mut self,
        py: Python<'_>,
        obs: &Bound<'_, PyArray2<f32>>,
        mask: &Bound<'_, PyArray2<bool>>,
        actor: &Bound<'_, PyArray1<i64>>,
    ) -> PyResult<()> {
        let mut obs = obs.readwrite();
        let mut mask = mask.readwrite();
        let mut actor = actor.readwrite();
        let obs = obs.as_slice_mut()?;
        let mask = mask.as_slice_mut()?;
        let actor = actor.as_slice_mut()?;
        let cfg = self.cfg;
        let base = self.next_seed;
        self.next_seed += self.slots.len() as u64;
        let slots = &mut self.slots;
        py.detach(|| {
            slots
                .par_iter_mut()
                .zip(obs.par_chunks_mut(OBS_SIZE))
                .zip(mask.par_chunks_mut(N_ACTIONS))
                .zip(actor.par_iter_mut())
                .enumerate()
                .for_each(|(i, (((slot, o), m), a))| {
                    Self::reset_slot(slot, cfg, base + i as u64);
                    Self::advance_bots(slot);
                    Self::write(slot, o, m, a);
                });
        });
        Ok(())
    }

    /// Apply one action per env. Finished games are recorded in done/winner/length and
    /// auto-reset; obs/mask/actor then describe the new game.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray1<'_, i64>,
        obs: &Bound<'_, PyArray2<f32>>,
        mask: &Bound<'_, PyArray2<bool>>,
        actor: &Bound<'_, PyArray1<i64>>,
        done: &Bound<'_, PyArray1<bool>>,
        winner: &Bound<'_, PyArray1<i64>>,
        length: &Bound<'_, PyArray1<i64>>,
        final_vp: &Bound<'_, PyArray2<i64>>,
    ) -> PyResult<()> {
        let actions = actions.as_slice()?;
        if actions.len() != self.slots.len() {
            return Err(PyValueError::new_err("actions length != num_envs"));
        }
        for (i, (&a, slot)) in actions.iter().zip(&self.slots).enumerate() {
            if a < 0 || !slot.state.is_legal(a as usize) {
                return Err(PyValueError::new_err(format!(
                    "env {i}: illegal action {a} ({}) in {:?}",
                    action_name(a.max(0) as usize),
                    slot.state.phase
                )));
            }
        }
        let (mut obs, mut mask, mut actor) = (obs.readwrite(), mask.readwrite(), actor.readwrite());
        let (mut done, mut winner, mut length) = (done.readwrite(), winner.readwrite(), length.readwrite());
        let obs = obs.as_slice_mut()?;
        let mask = mask.as_slice_mut()?;
        let actor = actor.as_slice_mut()?;
        let done = done.as_slice_mut()?;
        let winner = winner.as_slice_mut()?;
        let length = length.as_slice_mut()?;
        let mut final_vp = final_vp.readwrite();
        let final_vp = final_vp.as_slice_mut()?;
        let cfg = self.cfg;
        let base = self.next_seed;
        self.next_seed += self.slots.len() as u64;
        let slots = &mut self.slots;
        py.detach(|| {
            slots
                .par_iter_mut()
                .zip(obs.par_chunks_mut(OBS_SIZE))
                .zip(mask.par_chunks_mut(N_ACTIONS))
                .zip(actor.par_iter_mut())
                .zip(done.par_iter_mut())
                .zip(winner.par_iter_mut())
                .zip(length.par_iter_mut())
                .zip(final_vp.par_chunks_mut(4))
                .zip(actions.par_iter())
                .enumerate()
                .with_min_len(16)
                .for_each(|(i, ((((((((slot, o), m), a), d), w), l), fv), &act))| {
                    slot.state.step(act as usize);
                    slot.steps += 1;
                    Self::advance_bots(slot);
                    if slot.state.is_over() {
                        *d = true;
                        *w = slot.state.winner as i64;
                        *l = slot.steps as i64;
                        for (p, v) in fv.iter_mut().enumerate() {
                            *v = if p < slot.state.n() { slot.state.vp(p) as i64 } else { 0 };
                        }
                        for p in 0..slot.state.n() {
                            slot.last_stats[p] = player_stats(&slot.state, p);
                        }
                        Self::reset_slot(slot, cfg, base + i as u64);
                        Self::advance_bots(slot);
                    } else {
                        *d = false;
                        *w = -1;
                        *l = 0;
                    }
                    Self::write(slot, o, m, a);
                });
        });
        Ok(())
    }

    /// End-of-game statistics [N, 4, len(STAT_NAMES)] of each env's most recently finished game
    /// (read right after a step that reported `done`).
    fn last_game_stats<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        let n = self.slots.len();
        let flat: Vec<f32> = self.slots.iter().flat_map(|s| s.last_stats.iter().flatten().copied()).collect();
        PyArray1::from_vec(py, flat).reshape([n, 4, N_STATS]).unwrap()
    }

    /// Victory points (including hidden cards) per seat for one env.
    fn vps(&self, env: usize) -> Vec<u32> {
        let s = &self.slots[env].state;
        (0..s.n()).map(|p| s.vp(p)).collect()
    }
}

// ---------------------------------------------------------------- AzPool (AlphaZero self-play)

struct Sample {
    obs: Vec<f32>,
    mask: catan_core::Mask,
    pi: Vec<(u16, f32)>,
    actor: u8,
}

struct AzGame {
    state: State,
    search: Option<Search>,
    seats: [u8; 4],
    rng: Rng,
    searched_moves: u32,
    samples: Vec<Sample>,
    leaf_actor: u8,
    leaf_obs: Vec<f32>,
    leaf_mask: Vec<bool>,
    has_leaf: bool,
}

/// Many games searched in lockstep. Each round, `select` returns one leaf per game that is
/// searching; the caller evaluates them in one network batch and passes the result to `expand`.
/// When every search has used its budget, `advance` plays the chosen moves (and any forced moves
/// or bot moves) until each game needs a new search.
///
/// Seats: "search" (MCTS with the network; recorded as training data when `record` is on),
/// "heuristic" or "random" (built-in bots).
#[pyclass(module = "catan_rl._engine")]
struct AzPool {
    games: Vec<AzGame>,
    cfg: Config,
    mcts: MctsConfig,
    temp_moves: u32,
    record: bool,
    next_seed: u64,
    pending: Vec<usize>,
    out_obs: Vec<f32>,
    out_pi: Vec<f32>,
    out_z: Vec<f32>,
    out_mask: Vec<bool>,
    /// (winner, vp per seat, seat kinds) of finished games since the last `take_results`.
    results: Vec<(i8, [u32; 4], [u8; 4])>,
}

const SEAT_SEARCH: u8 = 0;

impl AzPool {
    fn new_game(cfg: Config, seed: u64, seats: [u8; 4]) -> AzGame {
        AzGame {
            state: State::new(cfg, seed),
            search: None,
            seats,
            rng: Rng::new(seed ^ 0xA2),
            searched_moves: 0,
            samples: Vec::new(),
            leaf_actor: 0,
            leaf_obs: vec![0.0; OBS_SIZE],
            leaf_mask: vec![false; N_ACTIONS],
            has_leaf: false,
        }
    }

    /// Play bot and forced moves until the game needs a search (or is over).
    fn prepare(g: &mut AzGame, mcts: MctsConfig) {
        while !g.state.is_over() {
            let p = g.state.actor();
            let kind = g.seats[p];
            if kind != SEAT_SEARCH {
                let a = bot_action(kind, &g.state, &mut g.rng);
                g.state.step(a);
                continue;
            }
            let m = g.state.legal_mask();
            if catan_core::actions::mask_count(&m) == 1 {
                g.state.step(mask_iter(&m).next().unwrap());
                continue;
            }
            let mut root = g.state;
            determinize(&mut root, p, &mut g.rng);
            g.search = Some(Search::new(root, mcts, g.rng.next_u64()));
            return;
        }
    }
}

#[pymethods]
impl AzPool {
    #[new]
    #[pyo3(signature = (num_games, seed=0, n_players=4, random_board=false, max_turns=500, sims=64, c_puct=1.5,
                        dirichlet_alpha=0.3, noise_frac=0.25, temp_moves=30, record=true))]
    #[allow(clippy::too_many_arguments)]
    fn new(num_games: usize, seed: u64, n_players: u8, random_board: bool, max_turns: u16, sims: u32, c_puct: f32,
           dirichlet_alpha: f32, noise_frac: f32, temp_moves: u32, record: bool) -> PyResult<Self> {
        let cfg = make_config(n_players, 10, random_board, max_turns)?;
        let mcts = MctsConfig { sims, c_puct, dirichlet_alpha, noise_frac, ..MctsConfig::default() };
        let mut games: Vec<AzGame> =
            (0..num_games).map(|i| Self::new_game(cfg, seed + i as u64, [SEAT_SEARCH; 4])).collect();
        for g in games.iter_mut() {
            Self::prepare(g, mcts);
        }
        Ok(AzPool {
            games,
            cfg,
            mcts,
            temp_moves,
            record,
            next_seed: seed + num_games as u64,
            pending: Vec::new(),
            out_obs: Vec::new(),
            out_pi: Vec::new(),
            out_z: Vec::new(),
            out_mask: Vec::new(),
            results: Vec::new(),
        })
    }

    /// Seat controllers for one game ("search" / "heuristic" / "random"); restarts that game.
    fn set_seats(&mut self, game: usize, seats: Vec<String>) -> PyResult<()> {
        let mut kinds = [SEAT_SEARCH; 4];
        for (i, s) in seats.iter().enumerate().take(4) {
            kinds[i] = match s.as_str() {
                "search" => SEAT_SEARCH,
                "random" => SEAT_RANDOM,
                "heuristic" => SEAT_HEURISTIC,
                _ => return Err(PyValueError::new_err(format!("unknown seat kind {s}"))),
            };
        }
        let seed = self.next_seed;
        self.next_seed += 1;
        let g = self.games.get_mut(game).ok_or_else(|| PyValueError::new_err("game out of range"))?;
        *g = Self::new_game(self.cfg, seed, kinds);
        Self::prepare(g, self.mcts);
        Ok(())
    }

    /// Collect one leaf per searching game into obs [N, OBS_SIZE] / mask [N, N_ACTIONS];
    /// returns how many rows were written (0 = all searches done, call `advance`).
    fn select(&mut self, py: Python<'_>, obs: &Bound<'_, PyArray2<f32>>, mask: &Bound<'_, PyArray2<bool>>) -> PyResult<usize> {
        let games = &mut self.games;
        py.detach(|| {
            games.par_iter_mut().for_each(|g| {
                g.has_leaf = false;
                if let Some(search) = g.search.as_mut() {
                    if let Some(leaf) = search.select() {
                        write_obs(leaf, &mut g.leaf_obs);
                        g.leaf_mask.fill(false);
                        for a in mask_iter(&leaf.legal_mask()) {
                            g.leaf_mask[a] = true;
                        }
                        g.leaf_actor = leaf.actor() as u8;
                        g.has_leaf = true;
                    }
                }
            })
        });
        let mut obs = obs.readwrite();
        let mut mask = mask.readwrite();
        let obs = obs.as_slice_mut()?;
        let mask = mask.as_slice_mut()?;
        self.pending.clear();
        for (i, g) in self.games.iter().enumerate() {
            if g.has_leaf {
                let k = self.pending.len();
                obs[k * OBS_SIZE..(k + 1) * OBS_SIZE].copy_from_slice(&g.leaf_obs);
                mask[k * N_ACTIONS..(k + 1) * N_ACTIONS].copy_from_slice(&g.leaf_mask);
                self.pending.push(i);
            }
        }
        Ok(self.pending.len())
    }

    /// Evaluations for the rows returned by the last `select`: priors [n, N_ACTIONS] (probabilities)
    /// and values [n, 4] = win probability per seat relative to the leaf's actor.
    fn expand(&mut self, py: Python<'_>, priors: PyReadonlyArray2<'_, f32>, values: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let priors = priors.as_slice()?;
        let values = values.as_slice()?;
        if priors.len() != self.pending.len() * N_ACTIONS || values.len() != self.pending.len() * 4 {
            return Err(PyValueError::new_err("expand: shapes do not match the last select()"));
        }
        let n = self.cfg.n_players as usize;
        let mut work: Vec<(&mut AzGame, usize)> = Vec::with_capacity(self.pending.len());
        let mut it = self.games.iter_mut().enumerate();
        for (k, &gi) in self.pending.iter().enumerate() {
            let g = loop {
                let (i, g) = it.next().unwrap();
                if i == gi {
                    break g;
                }
            };
            work.push((g, k));
        }
        py.detach(|| {
            work.into_par_iter().for_each(|(g, k)| {
                let rel = &values[k * 4..k * 4 + 4];
                let total: f32 = rel[..n].iter().sum::<f32>().max(1e-6);
                let mut v = [0f32; 4];
                for j in 0..n {
                    v[(g.leaf_actor as usize + j) % n] = rel[j] / total;
                }
                g.search.as_mut().unwrap().expand(&priors[k * N_ACTIONS..(k + 1) * N_ACTIONS], &v);
            })
        });
        self.pending.clear();
        Ok(())
    }

    /// Play the searched move in every game whose search is finished, then run bot/forced moves until
    /// each game needs a new search. Finished games are recorded and restarted.
    fn advance(&mut self, py: Python<'_>) {
        let (cfg, mcts, temp_moves, record) = (self.cfg, self.mcts, self.temp_moves, self.record);
        let base = self.next_seed;
        self.next_seed += self.games.len() as u64;
        let finished: Vec<Option<(Vec<Sample>, i8, [u32; 4], [u8; 4])>> = py.detach(|| {
            self.games
                .par_iter_mut()
                .enumerate()
                .map(|(i, g)| {
                    let Some(search) = g.search.as_ref() else { return None };
                    if !search.done() {
                        return None;
                    }
                    let visits = search.visits();
                    let total: u32 = visits.iter().map(|v| v.1).sum::<u32>().max(1);
                    let action = if g.searched_moves < temp_moves {
                        let mut k = g.rng.below(total);
                        visits.iter().find(|v| { if k < v.1 { true } else { k -= v.1; false } }).map(|v| v.0)
                            .unwrap_or(visits[0].0)
                    } else {
                        visits.iter().max_by_key(|v| v.1).unwrap().0
                    };
                    if record {
                        let mut obs = vec![0f32; OBS_SIZE];
                        write_obs(&g.state, &mut obs);
                        let pi = visits.iter().filter(|v| v.1 > 0).map(|v| (v.0 as u16, v.1 as f32 / total as f32)).collect();
                        g.samples.push(Sample { obs, mask: g.state.legal_mask(), pi, actor: g.state.actor() as u8 });
                    }
                    g.searched_moves += 1;
                    g.search = None;
                    g.state.step(action);
                    Self::prepare(g, mcts);
                    if !g.state.is_over() {
                        return None;
                    }
                    let s = &g.state;
                    let vps = [0, 1, 2, 3].map(|p| if p < s.n() { s.vp(p) } else { 0 });
                    let out = (std::mem::take(&mut g.samples), s.winner, vps, g.seats);
                    let seats = g.seats;
                    *g = Self::new_game(cfg, base + i as u64, seats);
                    Self::prepare(g, mcts);
                    Some(out)
                })
                .collect()
        });
        let n = self.cfg.n_players as usize;
        for (samples, winner, vps, seats) in finished.into_iter().flatten() {
            for smp in samples {
                self.out_obs.extend_from_slice(&smp.obs);
                let mut pi = [0f32; N_ACTIONS];
                for (a, p) in smp.pi {
                    pi[a as usize] = p;
                }
                self.out_pi.extend_from_slice(&pi);
                let mut legal = [false; N_ACTIONS];
                for a in mask_iter(&smp.mask) {
                    legal[a] = true;
                }
                self.out_mask.extend_from_slice(&legal);
                let mut z = [0f32; 4];
                for j in 0..n {
                    let seat = (smp.actor as usize + j) % n;
                    z[j] = if winner < 0 { 1.0 / n as f32 } else { (winner as usize == seat) as u8 as f32 };
                }
                self.out_z.extend_from_slice(&z);
            }
            self.results.push((winner, vps, seats));
        }
    }

    /// Training samples of finished games since the last call: (obs [M, OBS], legal mask [M, N_ACTIONS],
    /// pi [M, N_ACTIONS] root visit distribution, z [M, 4] winner one-hot relative to the sample's actor).
    #[allow(clippy::type_complexity)]
    fn take_samples<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<bool>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
        let m = self.out_z.len() / 4;
        let obs = PyArray1::from_vec(py, std::mem::take(&mut self.out_obs)).reshape([m, OBS_SIZE]).unwrap();
        let mask = PyArray1::from_vec(py, std::mem::take(&mut self.out_mask)).reshape([m, N_ACTIONS]).unwrap();
        let pi = PyArray1::from_vec(py, std::mem::take(&mut self.out_pi)).reshape([m, N_ACTIONS]).unwrap();
        let z = PyArray1::from_vec(py, std::mem::take(&mut self.out_z)).reshape([m, 4]).unwrap();
        (obs, mask, pi, z)
    }

    /// Finished games since the last call: list of (winner, vp per seat, seat kinds as strings).
    fn take_results(&mut self) -> Vec<(i8, Vec<u32>, Vec<String>)> {
        let n = self.cfg.n_players as usize;
        let name = |k: u8| match k { SEAT_SEARCH => "search", SEAT_RANDOM => "random", _ => "heuristic" }.to_string();
        std::mem::take(&mut self.results)
            .into_iter()
            .map(|(w, vps, seats)| (w, vps[..n].to_vec(), seats[..n].iter().map(|&k| name(k)).collect()))
            .collect()
    }
}

// ---------------------------------------------------------------- module

#[pyfunction]
fn action_names() -> Vec<String> {
    (0..N_ACTIONS).map(action_name).collect()
}

/// Play `games` full games with built-in bots per seat; returns win counts per seat (+ draws last).
#[pyfunction]
#[pyo3(signature = (seats, games, seed=0, random_board=false))]
fn bot_tournament(py: Python<'_>, seats: Vec<String>, games: u64, seed: u64, random_board: bool) -> PyResult<Vec<u64>> {
    let n = seats.len();
    let cfg = make_config(n as u8, 10, random_board, 500)?;
    let kinds: Vec<u8> = seats
        .iter()
        .map(|s| if s == "random" { SEAT_RANDOM } else { SEAT_HEURISTIC })
        .collect();
    Ok(py.detach(|| {
        let res: Vec<i8> = (0..games)
            .into_par_iter()
            .map(|g| {
                let mut s = State::new(cfg, seed + g);
                let mut rng = Rng::new(seed + g);
                while !s.is_over() {
                    let a = bot_action(kinds[s.actor()], &s, &mut rng);
                    s.step(a);
                }
                s.winner
            })
            .collect();
        let mut counts = vec![0u64; n + 1];
        for w in res {
            counts[if w < 0 { n } else { w as usize }] += 1;
        }
        counts
    }))
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add_class::<VecEnv>()?;
    m.add_class::<AzPool>()?;
    m.add_function(wrap_pyfunction!(action_names, m)?)?;
    m.add_function(wrap_pyfunction!(bot_tournament, m)?)?;
    m.add("OBS_SIZE", OBS_SIZE)?;
    m.add("N_ACTIONS", N_ACTIONS)?;
    m.add("STAT_NAMES", STAT_NAMES.to_vec())?;
    Ok(())
}
