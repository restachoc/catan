//! Flat action space. Every decision in the game is one integer id in [0, N_ACTIONS).

use crate::topology::{N_EDGE, N_HEX, N_VERT};

pub const ROLL: usize = 0;
pub const END_TURN: usize = 1;
pub const SETTLE: usize = 2;
pub const CITY: usize = SETTLE + N_VERT;
pub const ROAD: usize = CITY + N_VERT;
pub const BUY_DEV: usize = ROAD + N_EDGE;
pub const PLAY_KNIGHT: usize = BUY_DEV + 1;
pub const PLAY_ROAD_BUILDING: usize = PLAY_KNIGHT + 1;
pub const PLAY_MONOPOLY: usize = PLAY_ROAD_BUILDING + 1;
pub const PLAY_YOP: usize = PLAY_MONOPOLY + 5;
pub const MOVE_ROBBER: usize = PLAY_YOP + 15;
/// Steal from the player `k` seats after the current player (k = 1..n-1; slot 0 unused).
pub const STEAL: usize = MOVE_ROBBER + N_HEX;
pub const DISCARD: usize = STEAL + 4;
/// Maritime (bank / port) trade: give resource `t / 4`, get the `t % 4`-th other resource.
pub const TRADE: usize = DISCARD + 5;
/// Terms of an offer to all opponents, chosen after `PROPOSE_TRADE`: `kind * 20 + pair`, with `pair` as in `TRADE`
/// (give resource, get resource) and kind 0 = give 1 get 1, 1 = give 2 get 1, 2 = give 1 get 2.
pub const OFFER: usize = TRADE + 20;
/// An opponent's answer to the open offer (accept only if it can pay).
pub const ACCEPT_OFFER: usize = OFFER + 60;
pub const DECLINE_OFFER: usize = ACCEPT_OFFER + 1;
/// The proposer trades with the accepter `k` seats after it (k = 1..n-1; slot 0 unused), or cancels (also while
/// choosing the terms).
pub const CHOOSE_PARTNER: usize = DECLINE_OFFER + 1;
pub const CANCEL_OFFER: usize = CHOOSE_PARTNER + 4;
/// Start an offer (after rolling, at most `MAX_OFFERS` per turn): one main-phase choice, so that the 60 sets of
/// terms don't crowd out everything else under an untrained policy.
pub const PROPOSE_TRADE: usize = CANCEL_OFFER + 1;
pub const N_ACTIONS: usize = PROPOSE_TRADE + 1;
pub const MAX_OFFERS: u8 = 3;

/// Legal-action bitset.
pub const MASK_WORDS: usize = N_ACTIONS.div_ceil(64);
pub type Mask = [u64; MASK_WORDS];

#[inline]
pub fn mask_set(m: &mut Mask, a: usize) {
    m[a >> 6] |= 1 << (a & 63);
}

#[inline]
pub fn mask_has(m: &Mask, a: usize) -> bool {
    m[a >> 6] >> (a & 63) & 1 == 1
}

pub fn mask_iter(m: &Mask) -> impl Iterator<Item = usize> {
    let m = *m;
    (0..m.len()).flat_map(move |w| crate::topology::bits64(m[w]).map(move |b| w * 64 + b))
}

pub fn mask_count(m: &Mask) -> u32 {
    m.iter().map(|w| w.count_ones()).sum()
}

/// Year-of-plenty resource pairs (unordered, with repetition).
pub const YOP_PAIRS: [(u8, u8); 15] = {
    let mut out = [(0u8, 0u8); 15];
    let mut i = 0;
    let mut a = 0;
    while a < 5 {
        let mut b = a;
        while b < 5 {
            out[i] = (a, b);
            i += 1;
            b += 1;
        }
        a += 1;
    }
    out
};

#[inline]
pub fn trade_pair(t: usize) -> (usize, usize) {
    let give = t / 4;
    let g = t % 4;
    (give, if g >= give { g + 1 } else { g })
}

/// Terms of offer `o` (0..60): (give resource, give count, get resource, get count).
#[inline]
pub fn offer_terms(o: usize) -> (usize, u8, usize, u8) {
    let (give, get) = trade_pair(o % 20);
    let (gn, rn) = [(1, 1), (2, 1), (1, 2)][o / 20];
    (give, gn, get, rn)
}

#[inline]
pub fn trade_id(give: usize, get: usize) -> usize {
    TRADE + give * 4 + if get > give { get - 1 } else { get }
}

/// Human-readable action name, for logs and debugging.
pub fn action_name(a: usize) -> String {
    const R: [&str; 5] = ["wood", "brick", "wool", "grain", "ore"];
    match a {
        ROLL => "roll".into(),
        END_TURN => "end_turn".into(),
        a if a < CITY => format!("settlement@v{}", a - SETTLE),
        a if a < ROAD => format!("city@v{}", a - CITY),
        a if a < BUY_DEV => format!("road@e{}", a - ROAD),
        BUY_DEV => "buy_dev".into(),
        PLAY_KNIGHT => "play_knight".into(),
        PLAY_ROAD_BUILDING => "play_road_building".into(),
        a if a < PLAY_YOP => format!("monopoly:{}", R[a - PLAY_MONOPOLY]),
        a if a < MOVE_ROBBER => {
            let (x, y) = YOP_PAIRS[a - PLAY_YOP];
            format!("year_of_plenty:{}+{}", R[x as usize], R[y as usize])
        }
        a if a < STEAL => format!("robber@h{}", a - MOVE_ROBBER),
        a if a < DISCARD => format!("steal:+{}", a - STEAL),
        a if a < TRADE => format!("discard:{}", R[a - DISCARD]),
        a if a < OFFER => {
            let (g, r) = trade_pair(a - TRADE);
            format!("trade:{}->{}", R[g], R[r])
        }
        a if a < ACCEPT_OFFER => {
            let (g, gn, r, rn) = offer_terms(a - OFFER);
            format!("offer:{gn}{}->{rn}{}", R[g], R[r])
        }
        ACCEPT_OFFER => "accept_offer".into(),
        DECLINE_OFFER => "decline_offer".into(),
        a if a < CANCEL_OFFER => format!("choose_partner:+{}", a - CHOOSE_PARTNER),
        CANCEL_OFFER => "cancel_offer".into(),
        PROPOSE_TRADE => "propose_trade".into(),
        _ => format!("invalid({a})"),
    }
}
