//! Card counting: what each seat can deduce about every player's resource hand from public information.
//!
//! Every resource movement is public except a robber steal, which only the thief and the victim see. Each
//! observer keeps a small set of possible worlds (every player's exact hand) with probabilities. A steal it
//! doesn't take part in splits each world by the card the victim may have lost, weighted by the victim's hand
//! in that world. Every other change applies to all worlds, and worlds that would need a negative count, or
//! that contradict the amounts a Monopoly revealed, are dropped. That is exact Bayesian filtering: a later
//! payment that only one world can afford resolves the steal for the thief and the victim at once.
//!
//! The true hands are always among the surviving worlds, unless the cap of `MAX_WORLDS` forced dropping the
//! least likely ones (`pruned`); if that ever leaves no world, the observer restarts from the true hands
//! (`resets`, a small information leak that the stress test checks stays negligible).

use crate::actions::*;
use crate::state::{State, MAX_P};

pub const MAX_WORLDS: usize = 128;
pub type Hands = [[u8; 5]; MAX_P];

/// One observer's belief: up to `MAX_WORLDS` candidate hand tables with probabilities summing to 1.
#[derive(Clone, Copy, Debug)]
pub struct Belief {
    worlds: [Hands; MAX_WORLDS],
    w: [f32; MAX_WORLDS],
    n: usize,
}

impl Belief {
    fn certain(h: &Hands) -> Self {
        let mut b = Belief { worlds: [[[0; 5]; MAX_P]; MAX_WORLDS], w: [0.0; MAX_WORLDS], n: 1 };
        b.worlds[0] = *h;
        b.w[0] = 1.0;
        b
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn worlds(&self) -> impl Iterator<Item = (&Hands, f32)> {
        self.worlds[..self.n].iter().zip(self.w[..self.n].iter().copied())
    }

    /// Expected count of each resource in each player's hand.
    pub fn expected(&self) -> [[f32; 5]; MAX_P] {
        let mut e = [[0.0; 5]; MAX_P];
        for (h, w) in self.worlds() {
            for q in 0..MAX_P {
                for r in 0..5 {
                    e[q][r] += w * h[q][r] as f32;
                }
            }
        }
        e
    }

    /// Apply a public hand change to every world; drop worlds it contradicts. `mono` = (resource, thief) of
    /// a Monopoly, which reveals each victim's exact count of that resource.
    fn apply(&mut self, delta: &[[i16; 5]; MAX_P], mono: Option<(usize, usize)>, n: usize) {
        let mut k = 0;
        'worlds: for i in 0..self.n {
            let mut h = self.worlds[i];
            for q in 0..n {
                for r in 0..5 {
                    if let Some((m, thief)) = mono {
                        if q != thief && r == m && h[q][r] as i16 != -delta[q][r] {
                            continue 'worlds;
                        }
                    }
                    let v = h[q][r] as i16 + delta[q][r];
                    if v < 0 {
                        continue 'worlds;
                    }
                    h[q][r] = v as u8;
                }
            }
            self.worlds[k] = h;
            self.w[k] = self.w[i];
            k += 1;
        }
        self.n = k;
        self.normalize();
    }

    /// A card of unknown type moved from `victim` to `thief`. Returns true if worlds had to be pruned.
    fn branch_steal(&mut self, victim: usize, thief: usize) -> bool {
        let mut nw = [[[0u8; 5]; MAX_P]; MAX_WORLDS * 5];
        let mut nwt = [0f32; MAX_WORLDS * 5];
        let mut m = 0;
        for i in 0..self.n {
            let h = self.worlds[i];
            let total: u32 = h[victim].iter().map(|&c| c as u32).sum();
            for r in 0..5 {
                if h[victim][r] == 0 {
                    continue;
                }
                let mut g = h;
                g[victim][r] -= 1;
                g[thief][r] += 1;
                let wt = self.w[i] * h[victim][r] as f32 / total as f32;
                match (0..m).find(|&j| nw[j] == g) {
                    Some(j) => nwt[j] += wt,
                    None => {
                        nw[m] = g;
                        nwt[m] = wt;
                        m += 1;
                    }
                }
            }
        }
        let pruned = m > MAX_WORLDS;
        if pruned {
            let mut idx: [usize; MAX_WORLDS * 5] = std::array::from_fn(|i| i);
            idx[..m].sort_unstable_by(|&a, &b| nwt[b].total_cmp(&nwt[a]));
            for (k, &j) in idx[..MAX_WORLDS].iter().enumerate() {
                self.worlds[k] = nw[j];
                self.w[k] = nwt[j];
            }
            self.n = MAX_WORLDS;
        } else {
            self.worlds[..m].copy_from_slice(&nw[..m]);
            self.w[..m].copy_from_slice(&nwt[..m]);
            self.n = m;
        }
        self.normalize();
        pruned
    }

    fn normalize(&mut self) {
        let s: f32 = self.w[..self.n].iter().sum();
        if s > 0.0 {
            self.w[..self.n].iter_mut().for_each(|w| *w /= s);
        }
    }
}

/// Beliefs of every seat about every hand, updated after each action.
#[derive(Clone, Copy, Debug)]
pub struct Beliefs {
    pub seats: [Belief; MAX_P],
    /// Steals that overflowed `MAX_WORLDS` (least likely worlds dropped).
    pub pruned: u32,
    /// Observers restarted from the true hands because pruning had dropped every consistent world.
    pub resets: u32,
}

impl Beliefs {
    pub fn new(s: &State) -> Self {
        Beliefs { seats: [Belief::certain(&s.hands); MAX_P], pruned: 0, resets: 0 }
    }

    /// Update after action `a` took the game from hands `prev` to state `s`.
    pub fn update(&mut self, prev: &Hands, a: usize, s: &State) {
        let n = s.n();
        let mut delta = [[0i16; 5]; MAX_P];
        let mut moved = false;
        for q in 0..n {
            for r in 0..5 {
                delta[q][r] = s.hands[q][r] as i16 - prev[q][r] as i16;
                moved |= delta[q][r] != 0;
            }
        }
        if !moved {
            return;
        }
        let thief = s.cur as usize;
        let victim = if (MOVE_ROBBER..DISCARD).contains(&a) {
            (0..n).find(|&q| q != thief && delta[q].iter().sum::<i16>() < 0)
        } else {
            None
        };
        let mono = (PLAY_MONOPOLY..PLAY_YOP).contains(&a).then(|| (a - PLAY_MONOPOLY, thief));
        for (o, b) in self.seats[..n].iter_mut().enumerate() {
            match victim {
                Some(v) if o != thief && o != v => self.pruned += b.branch_steal(v, thief) as u32,
                _ => b.apply(&delta, mono, n),
            }
            if b.n == 0 {
                *b = Belief::certain(&s.hands);
                self.resets += 1;
            }
        }
    }
}
