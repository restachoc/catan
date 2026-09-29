//! Python bindings: a single `Game` for the UI and a batched, multi-threaded `VecEnv` for RL.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use catan_core::actions::{action_name, mask_iter};
use catan_core::bots::{heuristic_action, random_action};
use catan_core::rng::Rng;
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
}

/// Batched environments stepped in parallel. Seats marked as bots are played inside Rust, so
/// Python only ever sees states where an external (policy-controlled) seat must act.
///
/// All outputs are written into caller-owned numpy buffers:
///   obs [N, OBS_SIZE] f32, mask [N, N_ACTIONS] bool, actor [N] i64,
///   done [N] bool, winner [N] i64 (-1 draw / not done), length [N] i64 (steps of finished game).
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
                .zip(actions.par_iter())
                .enumerate()
                .with_min_len(16)
                .for_each(|(i, (((((((slot, o), m), a), d), w), l), &act))| {
                    slot.state.step(act as usize);
                    slot.steps += 1;
                    Self::advance_bots(slot);
                    if slot.state.is_over() {
                        *d = true;
                        *w = slot.state.winner as i64;
                        *l = slot.steps as i64;
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

    /// Victory points (including hidden cards) per seat for one env.
    fn vps(&self, env: usize) -> Vec<u32> {
        let s = &self.slots[env].state;
        (0..s.n()).map(|p| s.vp(p)).collect()
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
    m.add_function(wrap_pyfunction!(action_names, m)?)?;
    m.add_function(wrap_pyfunction!(bot_tournament, m)?)?;
    m.add("OBS_SIZE", OBS_SIZE)?;
    m.add("N_ACTIONS", N_ACTIONS)?;
    Ok(())
}
