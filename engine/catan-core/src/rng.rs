/// Small, fast, deterministic PRNG (wyrand). Games are replayable from (seed, actions).
#[derive(Clone, Copy, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng(seed ^ 0x9E37_79B9_7F4A_7C15);
        r.next_u64();
        r
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0xa076_1d64_78bd_642f);
        let t = (self.0 as u128).wrapping_mul((self.0 ^ 0xe703_7ed1_a0b4_28db) as u128);
        ((t >> 64) as u64) ^ (t as u64)
    }

    /// Uniform integer in [0, n).
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        (((self.next_u64() >> 32) * n as u64) >> 32) as u32
    }

    pub fn shuffle<T>(&mut self, s: &mut [T]) {
        for i in (1..s.len()).rev() {
            let j = self.below(i as u32 + 1) as usize;
            s.swap(i, j);
        }
    }
}
