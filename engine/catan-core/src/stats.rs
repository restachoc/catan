//! End-of-game per-player statistics, used to characterise the strategy a player followed.

use crate::state::*;

pub const STAT_NAMES: [&str; 23] = [
    "won",
    "vp",
    "roads",
    "settlements",
    "cities",
    "dev_bought",
    "knights",
    "longest_road",
    "largest_army",
    "vp_cards",
    "road_len",
    "open_wood",
    "open_brick",
    "open_wool",
    "open_grain",
    "open_ore",
    "final_wood",
    "final_brick",
    "final_wool",
    "final_grain",
    "final_ore",
    "offers",
    "trades",
];
pub const N_STATS: usize = STAT_NAMES.len();

pub fn player_stats(s: &State, p: usize) -> [f32; N_STATS] {
    let mut out = [0f32; N_STATS];
    out[0] = (s.winner == p as i8) as u8 as f32;
    out[1] = s.vp(p) as f32;
    out[2] = s.roads[p].count_ones() as f32;
    out[3] = s.settlements[p].count_ones() as f32;
    out[4] = s.cities[p].count_ones() as f32;
    out[5] = s.dev_bought[p] as f32;
    out[6] = s.knights[p] as f32;
    out[7] = (s.longest_road == p as i8) as u8 as f32;
    out[8] = (s.largest_army == p as i8) as u8 as f32;
    out[9] = s.dev_hand[p][DEV_VP] as f32;
    out[10] = s.road_len[p] as f32;
    let fin = s.production_pips(p);
    for r in 0..5 {
        out[11 + r] = s.opening_pips[p][r] as f32;
        out[16 + r] = fin[r] as f32;
    }
    out[21] = s.offers[p] as f32;
    out[22] = s.trades[p] as f32;
    out
}
