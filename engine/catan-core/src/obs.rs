//! Flat, actor-relative observation vector. Seat 0 in every per-player block is the acting player;
//! opponents follow in turn order. Only public information about opponents is encoded.

use crate::board::*;
use crate::state::*;
use crate::topology::*;

const HEX_F: usize = 8; // resource one-hot (6), pips/5, robber
const VERT_F: usize = 2 * MAX_P + 6; // settlement/city per seat, port one-hot (5 resources + 3:1)
const EDGE_F: usize = MAX_P;
const PLAYER_F: usize = 11;
const SELF_F: usize = 21;
const GLOBAL_F: usize = N_PHASES + 4 + 11 + 5 + 2;

pub const OBS_SIZE: usize =
    N_HEX * HEX_F + N_VERT * VERT_F + N_EDGE * EDGE_F + MAX_P * PLAYER_F + SELF_F + GLOBAL_F;

pub fn write_obs(s: &State, out: &mut [f32]) {
    debug_assert_eq!(out.len(), OBS_SIZE);
    out.fill(0.0);
    let n = s.n();
    let me = s.actor();
    let seat = |q: usize| (q + n - me) % n;
    let mut o = 0;

    for h in 0..N_HEX {
        out[o + s.board.hex_res[h] as usize] = 1.0;
        out[o + 6] = pips(s.board.hex_num[h]) as f32 / 5.0;
        out[o + 7] = (s.robber as usize == h) as u8 as f32;
        o += HEX_F;
    }

    for v in 0..N_VERT {
        let bit = 1u64 << v;
        for q in 0..n {
            let k = seat(q);
            if s.settlements[q] & bit != 0 {
                out[o + 2 * k] = 1.0;
            }
            if s.cities[q] & bit != 0 {
                out[o + 2 * k + 1] = 1.0;
            }
        }
        for t in 0..6 {
            if s.board.port_mask[t] & bit != 0 {
                out[o + 2 * MAX_P + t] = 1.0;
            }
        }
        o += VERT_F;
    }

    for e in 0..N_EDGE {
        let bit = 1u128 << e;
        for q in 0..n {
            if s.roads[q] & bit != 0 {
                out[o + seat(q)] = 1.0;
            }
        }
        o += EDGE_F;
    }

    for k in 0..MAX_P {
        if k < n {
            let q = (me + k) % n;
            let f = &mut out[o..o + PLAYER_F];
            f[0] = 1.0; // seat present
            f[1] = s.public_vp(q) as f32 / 10.0;
            f[2] = s.hand_total(q) as f32 / 10.0;
            f[3] = s.dev_total(q) as f32 / 5.0;
            f[4] = s.knights[q] as f32 / 5.0;
            f[5] = s.road_len[q] as f32 / 10.0;
            f[6] = (s.longest_road == q as i8) as u8 as f32;
            f[7] = (s.largest_army == q as i8) as u8 as f32;
            f[8] = s.settlements_left(q) as f32 / 5.0;
            f[9] = s.cities_left(q) as f32 / 4.0;
            f[10] = s.roads_left(q) as f32 / 15.0;
        }
        o += PLAYER_F;
    }

    {
        let f = &mut out[o..o + SELF_F];
        let ratios = s.trade_ratios(me);
        for r in 0..5 {
            f[r] = s.hands[me][r] as f32 / 5.0;
            f[5 + r] = (s.dev_hand[me][r] - s.dev_new[me][r]) as f32 / 2.0;
            f[10 + r] = s.dev_new[me][r] as f32;
            f[15 + r] = ratios[r] as f32 / 4.0;
        }
        f[20] = s.vp(me) as f32 / 10.0;
        o += SELF_F;
    }

    let f = &mut out[o..o + GLOBAL_F];
    f[s.phase as usize] = 1.0;
    let mut g = N_PHASES;
    f[g] = s.rolled as u8 as f32;
    f[g + 1] = s.dev_played as u8 as f32;
    f[g + 2] = s.free_roads as f32 / 2.0;
    f[g + 3] = s.discard_need[me] as f32 / 5.0;
    g += 4;
    let roll = (s.last_roll[0] + s.last_roll[1]) as usize;
    if roll >= 2 {
        f[g + roll - 2] = 1.0;
    }
    g += 11;
    for r in 0..5 {
        f[g + r] = s.bank[r] as f32 / 19.0;
    }
    g += 5;
    f[g] = s.dev_deck_len as f32 / 25.0;
    f[g + 1] = s.turn as f32 / 200.0;
}
