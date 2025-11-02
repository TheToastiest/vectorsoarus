//! soarus-nn — neighbor indices for 3D (exact grid to start; kd/LBVH later).

use hashbrown::HashMap;
use smallvec::SmallVec;
use soarus_core::{CloudView};
use hashbrown::hash_map::Entry;
#[derive(Copy, Clone, Debug)]
pub struct Neighbor { pub idx: usize, pub dist2: f32 }

/// Trait for geometry-first neighborhood queries.
pub trait NeighborIndex3 {
    fn knn(&self, i: usize, k: usize) -> SmallVec<[Neighbor; 64]>;
    fn radius(&self, i: usize, r: f32) -> SmallVec<[Neighbor; 128]>;
}

/// Uniform grid hash, cell size = r (good for radius queries / voxel ops).
pub struct GridIndex<'a> {
    pts: CloudView<'a>,
    cell: f32,
    buckets: HashMap<[i32;3], Vec<usize>>,
}

impl<'a> GridIndex<'a> {
    pub fn build(pts: CloudView<'a>, cell: f32) -> Self {
        let mut buckets: HashMap<[i32;3], Vec<usize>> = HashMap::new();
        let inv = 1.0 / cell.max(1e-12);
        for i in 0..pts.x.len() {
            let key = [
                (pts.x[i] * inv).floor() as i32,
                (pts.y[i] * inv).floor() as i32,
                (pts.z[i] * inv).floor() as i32,
            ];
            match buckets.entry(key) {
                Entry::Vacant(v) => { v.insert(vec![i]); }
                Entry::Occupied(mut o) => o.get_mut().push(i),
            }
        }
        Self { pts, cell, buckets }
    }

    fn neighbor_cells(&self, key: [i32;3]) -> impl Iterator<Item=[i32;3]> {
        (-1..=1).flat_map(move |dx|
            (-1..=1).flat_map(move |dy|
                (-1..=1).map(move |dz| [key[0]+dx, key[1]+dy, key[2]+dz])))
    }

    fn key_of(&self, i: usize) -> [i32;3] {
        let inv = 1.0 / self.cell;
        [
            (self.pts.x[i]*inv).floor() as i32,
            (self.pts.y[i]*inv).floor() as i32,
            (self.pts.z[i]*inv).floor() as i32,
        ]
    }
}

impl<'a> NeighborIndex3 for GridIndex<'a> {
    fn knn(&self, i: usize, k: usize) -> SmallVec<[Neighbor; 64]> {
        // crude knn via expanding cubes; adequate for bootstrap.
        let mut out: SmallVec<[Neighbor; 64]> = SmallVec::new();
        let p = [self.pts.x[i], self.pts.y[i], self.pts.z[i]];
        let mut layer = 0;
        while out.len() < k && layer < 4 { // expand up to 4 layers for now
            let base = self.key_of(i);
            for dx in -layer..=layer {
                for dy in -layer..=layer {
                    for dz in -layer..=layer {
                        let key = [base[0]+dx, base[1]+dy, base[2]+dz];
                        if let Some(bin) = self.buckets.get(&key) {
                            for &j in bin {
                                if j == i { continue; }
                                let d2 = (self.pts.x[j]-p[0]).powi(2)
                                    + (self.pts.y[j]-p[1]).powi(2)
                                    + (self.pts.z[j]-p[2]).powi(2);
                                out.push(Neighbor{ idx:j, dist2:d2 });
                            }
                        }
                    }
                }
            }
            layer += 1;
        }
        out.sort_by(|a,b| a.dist2.total_cmp(&b.dist2));
        out.truncate(k);
        out
    }

    fn radius(&self, i: usize, r: f32) -> SmallVec<[Neighbor; 128]> {
        let mut out = SmallVec::<[Neighbor;128]>::new();
        let p = [self.pts.x[i], self.pts.y[i], self.pts.z[i]];
        let key = self.key_of(i);
        let r2 = r*r;
        for key in self.neighbor_cells(key) {
            if let Some(bin) = self.buckets.get(&key) {
                for &j in bin {
                    if j == i { continue; }
                    let d2 = (self.pts.x[j]-p[0]).powi(2)
                        + (self.pts.y[j]-p[1]).powi(2)
                        + (self.pts.z[j]-p[2]).powi(2);
                    if d2 <= r2 { out.push(Neighbor{ idx:j, dist2:d2 }); }
                }
            }
        }
        out
    }
}
