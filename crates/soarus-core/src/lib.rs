//! soarus-core — core data model and shared math/types.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Structure-of-Arrays point cloud.
/// Keep hot columns (x,y,z) tight; put optional columns in a name→column map.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Cloud {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,

    /// Optional attributes (same length as x/y/z).
    /// Common keys: "nx","ny","nz","r","g","b","intensity","class".
    pub attrs_f32: HashMap<String, Vec<f32>>,
}

impl Cloud {
    pub fn len(&self) -> usize { self.x.len() }
    pub fn is_empty(&self) -> bool { self.x.is_empty() }
    pub fn push(&mut self, px: f32, py: f32, pz: f32) {
        self.x.push(px); self.y.push(py); self.z.push(pz);
    }
    pub fn reserve(&mut self, n: usize) {
        self.x.reserve(n); self.y.reserve(n); self.z.reserve(n);
        for v in self.attrs_f32.values_mut() { v.reserve(n); }
    }
}

/// Zero-copy view into a Cloud (slice-of-SoA).
#[derive(Copy, Clone)]
pub struct CloudView<'a> {
    pub x: &'a [f32],
    pub y: &'a [f32],
    pub z: &'a [f32],
}

impl<'a> From<&'a Cloud> for CloudView<'a> {
    fn from(c: &'a Cloud) -> Self { Self { x: &c.x, y: &c.y, z: &c.z } }
}

/// Simple AABB
#[derive(Copy, Clone, Debug)]
pub struct Aabb { pub min: [f32;3], pub max: [f32;3] }
impl Aabb {
    pub fn contains(&self, p: [f32;3]) -> bool {
        (0..3).all(|i| p[i] >= self.min[i] && p[i] <= self.max[i])
    }
}
