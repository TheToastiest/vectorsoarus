//! soarus-reg — ICP (point-to-plane) scaffold (implementation: next passes).

use anyhow::Result;
use nalgebra::{Isometry3, Matrix3, Vector3};
use rayon::prelude::*;
use soarus_core::Cloud;
use soarus_nn::{GridIndex, NeighborIndex3};

pub struct IcpPt2Plane {
    pub max_iters: usize,
    pub radius: f32,
}

impl Default for IcpPt2Plane {
    fn default() -> Self { Self { max_iters: 20, radius: 0.2 } }
}

impl IcpPt2Plane {
    pub fn align(&self, src: &Cloud, tgt: &Cloud) -> Result<Isometry3<f32>> {
        // Placeholder: returns identity. (Full solve to be added.)
        let _index = GridIndex::build(tgt.into(), self.radius);
        let t = Isometry3::identity();
        Ok(t)
    }
}
