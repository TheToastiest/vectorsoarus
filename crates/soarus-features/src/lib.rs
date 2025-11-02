use anyhow::Result;
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use soarus_core::Cloud;
use soarus_nn::{GridIndex, NeighborIndex3};

/// Estimate normals using PCA over neighbors within radius `r`.
/// Writes "nx","ny","nz" into attrs_f32.
pub fn estimate_normals_radius(mut cloud: Cloud, r: f32) -> Result<Cloud> {
    let view = (&cloud).into();
    let index = GridIndex::build(view, r);

    // Parallel: compute one normal per point, collect, then split into columns.
    let normals: Vec<[f32; 3]> = (0..cloud.len())
        .into_par_iter()
        .map(|i| {
            let p = Vector3::new(cloud.x[i], cloud.y[i], cloud.z[i]);
            let neigh = index.radius(i, r);
            if neigh.is_empty() {
                return [0.0, 0.0, 0.0];
            }

            // mean
            let mut mean = Vector3::zeros();
            for n in &neigh {
                mean.x += cloud.x[n.idx];
                mean.y += cloud.y[n.idx];
                mean.z += cloud.z[n.idx];
            }
            mean /= neigh.len() as f32;

            // covariance
            let mut c = Matrix3::<f32>::zeros();
            for n in &neigh {
                let v = Vector3::new(cloud.x[n.idx], cloud.y[n.idx], cloud.z[n.idx]) - mean;
                c[(0,0)] += v.x*v.x; c[(0,1)] += v.x*v.y; c[(0,2)] += v.x*v.z;
                c[(1,0)] += v.y*v.x; c[(1,1)] += v.y*v.y; c[(1,2)] += v.y*v.z;
                c[(2,0)] += v.z*v.x; c[(2,1)] += v.z*v.y; c[(2,2)] += v.z*v.z;
            }

            // smallest eigenvector ~ normal
            let eig = c.symmetric_eigen();
            let (mut min_i, mut min_val) = (0, eig.eigenvalues[0]);
            for k in 1..3 {
                if eig.eigenvalues[k] < min_val { min_i = k; min_val = eig.eigenvalues[k]; }
            }
            let n = eig.eigenvectors.column(min_i);
            [n[0], n[1], n[2]]
        })
        .collect();

    // Split into columns
    let mut nx = Vec::with_capacity(normals.len());
    let mut ny = Vec::with_capacity(normals.len());
    let mut nz = Vec::with_capacity(normals.len());
    for v in normals {
        nx.push(v[0]); ny.push(v[1]); nz.push(v[2]);
    }

    cloud.attrs_f32.insert("nx".into(), nx);
    cloud.attrs_f32.insert("ny".into(), ny);
    cloud.attrs_f32.insert("nz".into(), nz);
    Ok(cloud)
}
