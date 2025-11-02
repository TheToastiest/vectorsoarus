//! soarus-filters — voxel grid, pass-through, etc.

use anyhow::Result;
use hashbrown::HashMap;
use rayon::prelude::*;
use soarus_core::Cloud;

/// Voxel grid downsample (centroid). Grid size in same units as input.
pub fn voxel_downsample(input: &Cloud, voxel: f32) -> Result<Cloud> {
    let inv = 1.0 / voxel.max(1e-12);

    // Key = (ix,iy,iz) packed into i64 for hashmap friendliness
    fn key(ix:i32,iy:i32,iz:i32) -> i64 {
        ((ix as i64) & 0x1fffff) << 42 |
            ((iy as i64) & 0x1fffff) << 21 |
            ((iz as i64) & 0x1fffff)
    }

    let mut bins: HashMap<i64, (f64,f64,f64,u32)> = HashMap::new();
    for i in 0..input.len() {
        let ix = (input.x[i]*inv).floor() as i32;
        let iy = (input.y[i]*inv).floor() as i32;
        let iz = (input.z[i]*inv).floor() as i32;
        let k = key(ix,iy,iz);
        let e = bins.entry(k).or_insert((0.0,0.0,0.0,0));
        e.0 += input.x[i] as f64;
        e.1 += input.y[i] as f64;
        e.2 += input.z[i] as f64;
        e.3 += 1;
    }

    let mut out = Cloud::default();
    out.reserve(bins.len());
    for (_k,(sx,sy,sz,cnt)) in bins.into_iter() {
        let invc = 1.0 / (cnt as f64);
        out.push((sx*invc) as f32, (sy*invc) as f32, (sz*invc) as f32);
    }
    Ok(out)
}
