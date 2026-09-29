//! Serializable snapshots for the UI: static board geometry and per-viewer game state.

use serde::Serialize;

use crate::actions::*;
use crate::state::*;
use crate::topology::*;

#[derive(Serialize)]
pub struct BoardView {
    /// Lattice coords: real x = X * sqrt(3)/2, real y = Y / 2 (hex size 1).
    pub hex_center: Vec<(i16, i16)>,
    pub hex_vertices: Vec<[u8; 6]>,
    pub hex_res: Vec<u8>,
    pub hex_num: Vec<u8>,
    pub vertex_xy: Vec<(i16, i16)>,
    pub edge_vertices: Vec<[u8; 2]>,
    pub port_edges: Vec<u8>,
    pub port_types: Vec<u8>,
}

#[derive(Serialize)]
pub struct PlayerView {
    pub settlements: Vec<u8>,
    pub cities: Vec<u8>,
    pub roads: Vec<u8>,
    pub public_vp: u32,
    pub cards: u32,
    pub dev_cards: u32,
    pub knights: u8,
    pub road_len: u8,
    /// Only filled in for the viewer (or everyone once the game is over).
    pub hand: Option<[u8; 5]>,
    pub dev_hand: Option<[u8; 5]>,
    pub dev_new: Option<[u8; 5]>,
    pub vp: Option<u32>,
}

#[derive(Serialize)]
pub struct StateView {
    pub n_players: u8,
    pub phase: String,
    pub cur: u8,
    pub actor: u8,
    pub turn: u16,
    pub robber: u8,
    pub last_roll: [u8; 2],
    pub rolled: bool,
    pub dev_played: bool,
    pub free_roads: u8,
    pub discard_need: Vec<u8>,
    pub bank: [u8; 5],
    pub dev_deck: u8,
    pub longest_road: i8,
    pub largest_army: i8,
    pub winner: i8,
    pub vp_target: u8,
    pub players: Vec<PlayerView>,
    /// Legal actions for the viewer, if it is their move.
    pub legal: Vec<u16>,
    pub trade_ratios: Option<[u8; 5]>,
}

pub fn board_view(s: &State) -> BoardView {
    BoardView {
        hex_center: TOPO.hex_center.to_vec(),
        hex_vertices: TOPO.hex_vertices.to_vec(),
        hex_res: s.board.hex_res.to_vec(),
        hex_num: s.board.hex_num.to_vec(),
        vertex_xy: TOPO.vertex_xy.to_vec(),
        edge_vertices: TOPO.edge_vertices.to_vec(),
        port_edges: s.board.port_edges.to_vec(),
        port_types: s.board.port_types.to_vec(),
    }
}

/// State as seen by `viewer` (None = omniscient spectator).
pub fn state_view(s: &State, viewer: Option<usize>) -> StateView {
    let over = s.is_over();
    let players = (0..s.n())
        .map(|p| {
            let show = over || viewer.is_none() || viewer == Some(p);
            PlayerView {
                settlements: bits64(s.settlements[p]).map(|v| v as u8).collect(),
                cities: bits64(s.cities[p]).map(|v| v as u8).collect(),
                roads: bits128(s.roads[p]).map(|e| e as u8).collect(),
                public_vp: s.public_vp(p),
                cards: s.hand_total(p),
                dev_cards: s.dev_total(p),
                knights: s.knights[p],
                road_len: s.road_len[p],
                hand: show.then_some(s.hands[p]),
                dev_hand: show.then_some(s.dev_hand[p]),
                dev_new: show.then_some(s.dev_new[p]),
                vp: show.then_some(s.vp(p)),
            }
        })
        .collect();
    let my_move = viewer.is_none_or(|v| v == s.actor());
    StateView {
        n_players: s.cfg.n_players,
        phase: format!("{:?}", s.phase),
        cur: s.cur,
        actor: s.actor() as u8,
        turn: s.turn,
        robber: s.robber,
        last_roll: s.last_roll,
        rolled: s.rolled,
        dev_played: s.dev_played,
        free_roads: s.free_roads,
        discard_need: s.discard_need[..s.n()].to_vec(),
        bank: s.bank,
        dev_deck: s.dev_deck_len,
        longest_road: s.longest_road,
        largest_army: s.largest_army,
        winner: s.winner,
        vp_target: s.cfg.vp_target,
        players,
        legal: if my_move { mask_iter(&s.legal_mask()).map(|a| a as u16).collect() } else { vec![] },
        trade_ratios: viewer.map(|v| s.trade_ratios(v)),
    }
}
