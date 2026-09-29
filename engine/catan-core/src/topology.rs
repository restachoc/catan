//! Static board graph: 19 hexes, 54 vertices, 72 edges. Built once, shared by every game.
//!
//! Hexes are pointy-top, in axial coords (q, r), ordered row by row (r = -2..=2, q ascending),
//! i.e. the reading order of the physical board. Vertex coordinates live on an integer lattice:
//! real x = X * sqrt(3)/2, real y = Y / 2 (unit hex size). Vertices are sorted by (Y, X) and edges
//! by midpoint, so ids are stable and read top-left to bottom-right.

use std::sync::LazyLock;

pub const N_HEX: usize = 19;
pub const N_VERT: usize = 54;
pub const N_EDGE: usize = 72;
pub const N_COAST: usize = 30;

pub struct Topology {
    pub hex_axial: [(i8, i8); N_HEX],
    pub hex_center: [(i16, i16); N_HEX],
    pub hex_vertices: [[u8; 6]; N_HEX],
    pub hex_vmask: [u64; N_HEX],
    /// Bitmask of hexes sharing an edge with this hex.
    pub hex_neighbors: [u32; N_HEX],
    pub vertex_xy: [(i16, i16); N_VERT],
    /// Bitmask of hexes touching this vertex.
    pub vertex_hexes: [u32; N_VERT],
    pub vertex_neighbors: [u64; N_VERT],
    pub vertex_edges: [u128; N_VERT],
    pub edge_vertices: [[u8; 2]; N_EDGE],
    /// Coastal edges, ordered clockwise around the island.
    pub coast_edges: [u8; N_COAST],
}

pub static TOPO: LazyLock<Topology> = LazyLock::new(build);

const CORNER_OFFSETS: [(i16, i16); 6] = [(0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1)];

fn build() -> Topology {
    let mut axial = Vec::new();
    for r in -2i8..=2 {
        for q in -2i8..=2 {
            if (q + r).abs() <= 2 {
                axial.push((q, r));
            }
        }
    }
    assert_eq!(axial.len(), N_HEX);
    let centers: Vec<(i16, i16)> =
        axial.iter().map(|&(q, r)| (2 * q as i16 + r as i16, 3 * r as i16)).collect();

    let mut pts: Vec<(i16, i16)> = centers
        .iter()
        .flat_map(|c| CORNER_OFFSETS.iter().map(move |o| (c.0 + o.0, c.1 + o.1)))
        .collect();
    pts.sort_by_key(|&(x, y)| (y, x));
    pts.dedup();
    assert_eq!(pts.len(), N_VERT);
    let vid = |p: (i16, i16)| pts.iter().position(|&q| q == p).unwrap() as u8;

    let mut hex_vertices = [[0u8; 6]; N_HEX];
    let mut hex_vmask = [0u64; N_HEX];
    let mut vertex_hexes = [0u32; N_VERT];
    for (h, c) in centers.iter().enumerate() {
        for (k, o) in CORNER_OFFSETS.iter().enumerate() {
            let v = vid((c.0 + o.0, c.1 + o.1));
            hex_vertices[h][k] = v;
            hex_vmask[h] |= 1 << v;
            vertex_hexes[v as usize] |= 1 << h;
        }
    }

    let mut edges: Vec<[u8; 2]> = Vec::new();
    for hv in &hex_vertices {
        for k in 0..6 {
            let (a, b) = (hv[k], hv[(k + 1) % 6]);
            edges.push([a.min(b), a.max(b)]);
        }
    }
    let mid = |e: &[u8; 2]| {
        let (a, b) = (pts[e[0] as usize], pts[e[1] as usize]);
        (a.1 + b.1, a.0 + b.0)
    };
    edges.sort_by_key(mid);
    edges.dedup();
    assert_eq!(edges.len(), N_EDGE);

    let mut edge_vertices = [[0u8; 2]; N_EDGE];
    let mut vertex_edges = [0u128; N_VERT];
    let mut vertex_neighbors = [0u64; N_VERT];
    for (i, e) in edges.iter().enumerate() {
        edge_vertices[i] = *e;
        let (a, b) = (e[0] as usize, e[1] as usize);
        vertex_edges[a] |= 1 << i;
        vertex_edges[b] |= 1 << i;
        vertex_neighbors[a] |= 1 << b;
        vertex_neighbors[b] |= 1 << a;
    }

    let mut hex_neighbors = [0u32; N_HEX];
    for a in 0..N_HEX {
        for b in 0..N_HEX {
            if a != b && (hex_vmask[a] & hex_vmask[b]).count_ones() == 2 {
                hex_neighbors[a] |= 1 << b;
            }
        }
    }

    // Coastal edges belong to exactly one hex; order them by angle around the centre.
    let mut coast: Vec<(f64, u8)> = edges
        .iter()
        .enumerate()
        .filter(|(_, e)| (vertex_hexes[e[0] as usize] & vertex_hexes[e[1] as usize]).count_ones() == 1)
        .map(|(i, e)| {
            let (a, b) = (pts[e[0] as usize], pts[e[1] as usize]);
            let x = (a.0 + b.0) as f64 * 3f64.sqrt() / 2.0;
            let y = (a.1 + b.1) as f64 / 2.0;
            (y.atan2(x), i as u8)
        })
        .collect();
    assert_eq!(coast.len(), N_COAST);
    coast.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut coast_edges = [0u8; N_COAST];
    for (i, c) in coast.iter().enumerate() {
        coast_edges[i] = c.1;
    }

    Topology {
        hex_axial: axial.try_into().unwrap(),
        hex_center: centers.try_into().unwrap(),
        hex_vertices,
        hex_vmask,
        hex_neighbors,
        vertex_xy: pts.try_into().unwrap(),
        vertex_hexes,
        vertex_neighbors,
        vertex_edges,
        edge_vertices,
        coast_edges,
    }
}

/// Iterate set bits of a u64 / u128 / u32 mask.
#[inline]
pub fn bits64(mut m: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if m == 0 {
            None
        } else {
            let i = m.trailing_zeros() as usize;
            m &= m - 1;
            Some(i)
        }
    })
}

#[inline]
pub fn bits128(mut m: u128) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if m == 0 {
            None
        } else {
            let i = m.trailing_zeros() as usize;
            m &= m - 1;
            Some(i)
        }
    })
}

#[inline]
pub fn bits32(m: u32) -> impl Iterator<Item = usize> {
    bits64(m as u64)
}
