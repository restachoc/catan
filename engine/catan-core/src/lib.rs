//! Fast Settlers of Catan rules engine (base game, no player-to-player trading yet).

pub mod actions;
pub mod belief;
pub mod board;
pub mod bots;
pub mod mcts;
pub mod obs;
pub mod rng;
pub mod state;
pub mod stats;
pub mod topology;
pub mod view;

pub use actions::{action_name, Mask, N_ACTIONS};
pub use obs::{exact_hands, write_obs, OBS_SIZE};
pub use state::{Config, Phase, State};
