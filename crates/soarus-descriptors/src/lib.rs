use anyhow::Result;
use rayon::prelude::*;
use soarus_core::Cloud;
use soarus_nn::{GridIndex, NeighborIndex3};

#[derive(Clone, Copy)]
pub struct FpfhCfg {
    /// neighborhood radius (meters)
    pub radius: f32,
    /// bins per SPFH channel (alpha, phi, theta); final dim = 3*bins
    pub bins: usize,
}

impl Default for FpfhCfg {
    fn default() -> Self { Self { radius: 0.05, bins: 11 } }
}

/// Compute FPFH (33-D by default). Requires target cloud to have normals nx,ny,nz.
/// If missing, compute normals before calling this.
pub fn fpfh(cloud: &Cloud, cfg: FpfhCfg) -> Result<Vec<Vec<f32>>> {
    let n = cloud.len();
    if n == 0 { return Ok(vec![]); }
    // normals
    let nx = cloud.attrs_f32.get("nx").ok_or_else(|| anyhow::anyhow!("missing nx"))?;
    let ny = cloud.attrs_f32.get("ny").ok_or_else(|| anyhow::anyhow!("missing ny"))?;
    let nz = cloud.attrs_f32.get("nz").ok_or_else(|| anyhow::anyhow!("missing nz"))?;
    if nx.len()!=n || ny.len()!=n || nz.len()!=n {
        anyhow::bail!("nx/ny/nz length mismatch");
    }

    let index = GridIndex::build(cloud.into(), cfg.radius);
    let bins = cfg.bins.max(3);
    let dim  = 3*bins;

    // 1) SPFH per point (3 hist channels)
    let spfh: Vec<Vec<f32>> = (0..n).into_par_iter().map(|i| {
        let mut h = vec![0.0f32; dim];
        let p  = [cloud.x[i], cloud.y[i], cloud.z[i]];
        let ni = [nx[i], ny[i], nz[i]];
        let neigh = index.radius(i, cfg.radius);
        if neigh.is_empty() { return h; }

        for nb in &neigh {
            let j = nb.idx;
            let q  = [cloud.x[j], cloud.y[j], cloud.z[j]];
            let nj = [nx[j], ny[j], nz[j]];

            // Darboux frame at p (ni as z-axis); build u,v,w
            let mut u = cross(ni, sub(q,p)); norm_inplace(&mut u);
            if length(u) < 1e-8 {
                // fallback: perpendicular to ni
                u = ortho(ni);
            }
            let w = ni;
            let mut v = cross(w, u); norm_inplace(&mut v);

            // angles per PCL paper (alpha = v·nj, phi = u·nj, theta = atan2( (w×ni)·nj , w·nj ) but simplified)
            let alpha = dot(v, nj);
            let phi   = dot(u, nj);
            let theta = dot(w, nj);

            bin3(&mut h, alpha, phi, theta, bins);
        }
        // L1 norm
        let s = h.iter().sum::<f32>().max(1e-6);
        for x in &mut h { *x /= s; }
        h
    }).collect();

    // 2) FPFH: weighted sum of neighbor SPFHs + self SPFH
    let out: Vec<Vec<f32>> = (0..n).into_par_iter().map(|i| {
        let mut f = spfh[i].clone();
        let neigh = index.radius(i, cfg.radius);
        if neigh.is_empty() { return f; }
        let mut acc = vec![0.0f32; f.len()];
        let mut wsum = 0.0f32;
        for nb in &neigh {
            let j = nb.idx;
            let d = nb.dist2.max(1e-12).sqrt();
            let w = 1.0 / d;
            wsum += w;
            for k in 0..f.len() { acc[k] += spfh[j][k] * w; }
        }
        if wsum > 0.0 {
            for k in 0..f.len() { f[k] = 0.5*f[k] + 0.5*(acc[k]/wsum); }
        }
        f
    }).collect();

    Ok(out)
}

// -------- small helpers --------
#[inline] fn dot(a:[f32;3], b:[f32;3])->f32 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
#[inline] fn sub(a:[f32;3], b:[f32;3])->[f32;3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
#[inline] fn cross(a:[f32;3], b:[f32;3])->[f32;3] {
    [ a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0] ]
}
#[inline] fn length(a:[f32;3])->f32 { (a[0]*a[0]+a[1]*a[1]+a[2]*a[2]).sqrt() }
#[inline] fn norm_inplace(a:&mut [f32;3]) {
    let l = length(*a); if l>1e-12 { a[0]/=l; a[1]/=l; a[2]/=l; }
}
#[inline] fn ortho(n:[f32;3])->[f32;3] {
    let t = if n[0].abs()<0.9 { [1.0,0.0,0.0] } else { [0.0,1.0,0.0] };
    let mut u = cross(n, t); norm_inplace(&mut u); u
}

fn bin3(h:&mut [f32], alpha:f32, phi:f32, theta:f32, bins:usize) {
    let mut put = |val:f32, off:usize| {
        // clamp to [-1,1], map to [0,1], then bin
        let v = ((val.max(-1.0).min(1.0))+1.0)*0.5;
        let bi = ((v*(bins as f32)) as usize).min(bins-1);
        h[off+bi] += 1.0;
    };
    put(alpha, 0);
    put(phi,   bins);
    put(theta, 2*bins);
}
