//! Per-game board layout: tile resources, number tokens and ports.

use crate::rng::Rng;
use crate::topology::{bits32, N_HEX, TOPO};

pub const WOOD: u8 = 0;
pub const BRICK: u8 = 1;
pub const WOOL: u8 = 2;
pub const GRAIN: u8 = 3;
pub const ORE: u8 = 4;
pub const DESERT: u8 = 5;
/// Port type index for the generic 3:1 port (0..5 are 2:1 resource ports).
pub const PORT_ANY: u8 = 5;
pub const N_PORTS: usize = 9;

/// Coastal-edge slots (indices into `TOPO.coast_edges`) that carry a port.
const PORT_SLOTS: [usize; N_PORTS] = [0, 3, 6, 10, 13, 16, 20, 23, 26];

const BEGINNER_RES: [u8; N_HEX] = [
    ORE, WOOL, WOOD, //
    GRAIN, BRICK, WOOL, BRICK, //
    GRAIN, WOOD, DESERT, WOOD, ORE, //
    WOOD, ORE, GRAIN, WOOL, //
    BRICK, GRAIN, WOOL,
];
const BEGINNER_NUM: [u8; N_HEX] = [
    10, 2, 9, //
    12, 6, 4, 10, //
    9, 11, 0, 3, 8, //
    8, 3, 4, 5, //
    5, 6, 11,
];
const BEGINNER_PORTS: [u8; N_PORTS] = [PORT_ANY, GRAIN, ORE, PORT_ANY, WOOL, PORT_ANY, PORT_ANY, BRICK, WOOD];

#[derive(Clone, Copy, Debug)]
pub struct Board {
    pub hex_res: [u8; N_HEX],
    pub hex_num: [u8; N_HEX],
    pub port_edges: [u8; N_PORTS],
    pub port_types: [u8; N_PORTS],
    /// Vertices with access to each port type (0..5 resource 2:1, 5 = 3:1).
    pub port_mask: [u64; 6],
    /// Hexes carrying each dice number.
    pub num_hexes: [u32; 13],
    pub desert: u8,
}

#[inline]
pub fn pips(num: u8) -> u8 {
    if num == 0 {
        0
    } else {
        6 - (7i8 - num as i8).unsigned_abs()
    }
}

impl Board {
    pub fn beginner() -> Self {
        Self::build(BEGINNER_RES, BEGINNER_NUM, BEGINNER_PORTS)
    }

    /// Random tiles, numbers (no 6/8 adjacent to another 6/8) and port types.
    pub fn random(rng: &mut Rng) -> Self {
        let mut res = BEGINNER_RES;
        rng.shuffle(&mut res);
        let mut nums: [u8; 18] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];
        let hex_num = loop {
            rng.shuffle(&mut nums);
            let mut hn = [0u8; N_HEX];
            let mut it = nums.iter();
            for h in 0..N_HEX {
                if res[h] != DESERT {
                    hn[h] = *it.next().unwrap();
                }
            }
            let red = |n: u8| n == 6 || n == 8;
            let ok = (0..N_HEX)
                .all(|h| !red(hn[h]) || bits32(TOPO.hex_neighbors[h]).all(|g| !red(hn[g])));
            if ok {
                break hn;
            }
        };
        let mut ports = BEGINNER_PORTS;
        rng.shuffle(&mut ports);
        Self::build(res, hex_num, ports)
    }

    fn build(hex_res: [u8; N_HEX], hex_num: [u8; N_HEX], port_types: [u8; N_PORTS]) -> Self {
        let mut port_edges = [0u8; N_PORTS];
        let mut port_mask = [0u64; 6];
        for (i, &slot) in PORT_SLOTS.iter().enumerate() {
            let e = TOPO.coast_edges[slot];
            port_edges[i] = e;
            let [a, b] = TOPO.edge_vertices[e as usize];
            port_mask[port_types[i] as usize] |= (1 << a) | (1 << b);
        }
        let mut num_hexes = [0u32; 13];
        let mut desert = 0;
        for h in 0..N_HEX {
            num_hexes[hex_num[h] as usize] |= 1 << h;
            if hex_res[h] == DESERT {
                desert = h as u8;
            }
        }
        num_hexes[0] = 0;
        Board { hex_res, hex_num, port_edges, port_types, port_mask, num_hexes, desert }
    }
}
