use anyhow::Result;
use nalgebra::{Isometry3, Point3};
use serde::{Deserialize, Serialize};
use soarus_core::{Cloud, CloudView};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum IcpMode { PointToPoint, PointToPlane }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IcpConfig {
    pub mode: IcpMode,
    pub max_corr: f32,
    pub iters: usize,
    pub trim: f32,
    pub pyramid: Option<Vec<f32>>,     // e.g., vec![0.01,0.005,0.002]
    pub normals_r_scale: f32,          // pt2plane: r = scale * voxel
    pub max_plane_dist: f32,           // pt2plane cutoff (|n·(ps-q)|)
    pub huber_delta: f32,              // pt2plane robust delta
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IcpMetrics {
    pub inliers: usize,
    pub rmse: f32,
    pub p50: f32, pub p90: f32, pub p95: f32, pub p99: f32, pub max: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IcpResult { pub pose: Isometry3<f32>, pub metrics: IcpMetrics }

fn residuals_after_align(src: &Cloud, tgt: &Cloud, T: &Isometry3<f32>, max_corr: f32) -> Vec<f32> {
    let idx = soarus_n::GridIndex::build(CloudView::from(tgt), max_corr.max(1e-3));
    let mut r = Vec::with_capacity(src.len());
    for i in 0..src.len() {
        let p = T.transform_point(&Point3::new(src.x[i], src.y[i], src.z[i]));
        if let Some((_j, d2)) = idx.nearest_in_radius([p.x, p.y, p.z], max_corr) { r.push(d2.sqrt()); }
    }
    r
}
fn stats(mut r: Vec<f32>) -> IcpMetrics {
    if r.is_empty() { return IcpMetrics{inliers:0,rmse:f32::NAN,p50:f32::NAN,p90:f32::NAN,p95:f32::NAN,p99:f32::NAN,max:f32::NAN}; }
    r.sort_by(|a,b| a.total_cmp(b));
    let n = r.len();
    let rmse = (r.iter().map(|x| x*x).sum::<f32>()/n as f32).sqrt();
    let pct = |p:f32| -> f32 { r[((p*(n as f32-1.0)).round() as usize).min(n-1)] };
    IcpMetrics{ inliers:n, rmse, p50:pct(0.50), p90:pct(0.90), p95:pct(0.95), p99:pct(0.99), max:*r.last().unwrap() }
}

/// One-shot registration. If `pyramid` is Some(levels), runs coarse→fine ICP.
/// Returns src->tgt pose and final residual statistics on full-res inputs.
pub fn icp_register(src_in: &Cloud, tgt_in: &Cloud, cfg: IcpConfig) -> Result<IcpResult> {
    let mut T = Isometry3::identity();
    let levels = cfg.pyramid.clone().unwrap_or_else(|| vec![0.0]);

    for vox in levels {
        let (src, mut tgt) = if vox > 0.0 {
            (soarus_filters::voxel_downsample(src_in, vox)?,
             soarus_filters::voxel_downsample(tgt_in, vox)?)
        } else { (src_in.clone(), tgt_in.clone()) };

        let max_corr = if vox > 0.0 { cfg.max_corr.max(vox*0.3) } else { cfg.max_corr };

        if matches!(cfg.mode, IcpMode::PointToPlane) {
            let nr = cfg.normals_r_scale.max(1e-3) * if vox>0.0 { vox } else { cfg.max_corr };
            tgt = soarus_features::estimate_normals_radius(tgt, nr)?;
        }

        // apply current pose to src and solve for incremental step
        let mut srcT = src.clone();
        for i in 0..srcT.len() {
            let p = T.transform_point(&Point3::new(srcT.x[i], srcT.y[i], srcT.z[i]));
            srcT.x[i]=p.x; srcT.y[i]=p.y; srcT.z[i]=p.z;
        }
        let step = match cfg.mode {
            IcpMode::PointToPoint =>
                soarus_reg::icp_point_to_point(&srcT, &tgt,
                                               soarus_reg::IcpP2P{ max_iters: cfg.iters, max_corr_dist: max_corr, trim_ratio: cfg.trim })?,
            IcpMode::PointToPlane =>
                soarus_reg::icp_point_to_plane(&srcT, &tgt,
                                               soarus_reg::IcpPt2Plane{ max_iters: cfg.iters, max_corr_dist: max_corr, trim_ratio: cfg.trim,
                                                   max_plane_dist: cfg.max_plane_dist, huber_delta: cfg.huber_delta })?,
        };
        T = step * T;
    }

    let res = residuals_after_align(src_in, tgt_in, &T, cfg.max_corr);
    Ok(IcpResult{ pose: T, metrics: stats(res) })
}
