//! soarus-filters — voxel grid, pass-through, etc.

use anyhow::Result;
use hashbrown::HashMap;
use rayon::prelude::*;
use soarus_nn::{GridIndex, NeighborIndex3};
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
/// Keep points that have at least `min_pts` neighbors within radius `r`.
pub fn radius_outlier(input: &Cloud, r: f32, min_pts: usize) -> Result<Cloud> {
    let view = input.into();
    let index = GridIndex::build(view, r);
    let keep: Vec<bool> = (0..input.len())
        .into_par_iter()
        .map(|i| index.radius(i, r).len() >= min_pts)
        .collect();

    let mut out = Cloud::default();
    out.reserve(keep.iter().filter(|&&k| k).count());

    // for attrs, only copy those with correct length; preserve alignment
    let mut attrs_keys: Vec<String> = input
        .attrs_f32
        .iter()
        .filter_map(|(k,v)| if v.len()==input.len(){ Some(k.clone()) } else { None })
        .collect();

    let mut attrs_out: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
    for k in &attrs_keys {
        attrs_out.insert(k.clone(), Vec::new());
    }

    for i in 0..input.len() {
        if keep[i] {
            out.push(input.x[i], input.y[i], input.z[i]);
            for k in &attrs_keys {
                attrs_out.get_mut(k).unwrap().push(input.attrs_f32[k][i]);
            }
        }
    }
    out.attrs_f32 = attrs_out;
    Ok(out)
}
/// Statistical Outlier Removal (kNN µ±σ)
pub fn statistical_outlier(input: &Cloud, k: usize, stddev_mul: f32) -> Result<Cloud> {
    if input.len() == 0 { return Ok(input.clone()); }

    // crude bucket size guess; fine for now
    let r = estimate_radius(input).max(0.1);
    let index = GridIndex::build(input.into(), r);

    let means: Vec<f32> = (0..input.len())
        .into_par_iter()
        .map(|i| {
            let neigh = index.knn(i, k);
            if neigh.is_empty() { return f32::INFINITY; }
            let sum = neigh.iter().map(|n| n.dist2.sqrt()).sum::<f32>();
            sum / (neigh.len() as f32)
        })
        .collect();

    let vals: Vec<f32> = means.into_iter().filter(|v| v.is_finite()).collect();
    if vals.is_empty() { return Ok(input.clone()); }

    let m  = vals.iter().copied().sum::<f32>() / (vals.len() as f32);
    let var= vals.iter().map(|v| (v - m)*(v - m)).sum::<f32>() / (vals.len() as f32);
    let sd = var.sqrt();
    let thresh = m + stddev_mul * sd;

    let mut out = Cloud::default();
    out.reserve(input.len());

    // copy only attrs that match length
    let keys: Vec<String> = input.attrs_f32
        .iter().filter_map(|(k,v)| if v.len()==input.len(){Some(k.clone())} else {None})
        .collect();
    let mut attrs_out = std::collections::HashMap::<String, Vec<f32>>::new();
    for k in &keys { attrs_out.insert(k.clone(), Vec::new()); }

    for i in 0..input.len() {
        // use original per-point mean; treat missing as outlier
        let keep = i < vals.len() && vals[i] <= thresh;
        if keep {
            out.push(input.x[i], input.y[i], input.z[i]);
            for k in &keys { attrs_out.get_mut(k).unwrap().push(input.attrs_f32[k][i]); }
        }
    }
    out.attrs_f32 = attrs_out;
    Ok(out)
}

fn estimate_radius(c: &Cloud) -> f32 {
    if c.len() < 2 { return 0.1; }
    let (mut minx,mut miny,mut minz) = (f32::INFINITY,f32::INFINITY,f32::INFINITY);
    let (mut maxx,mut maxy,mut maxz) = (f32::NEG_INFINITY,f32::NEG_INFINITY,f32::NEG_INFINITY);
    for i in 0..c.len() {
        minx = minx.min(c.x[i]); miny = miny.min(c.y[i]); minz = minz.min(c.z[i]);
        maxx = maxx.max(c.x[i]); maxy = maxy.max(c.y[i]); maxz = maxz.max(c.z[i]);
    }
    let dx=maxx-minx; let dy=maxy-miny; let dz=maxz-minz;
    let diag = (dx*dx + dy*dy + dz*dz).sqrt();
    (diag / (c.len() as f32).cbrt()).max(1e-3)
}