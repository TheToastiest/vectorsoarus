//! soarus-reg — ICP (point-to-point & point-to-plane)

use anyhow::{anyhow, Result};
use nalgebra::{
    Isometry3, Matrix3, Matrix6, Point3, Rotation3, Translation3, UnitQuaternion, Vector3, SVector,
};
use soarus_core::Cloud;
use soarus_nn::GridIndex;

/// -------------------------
/// Point-to-Point ICP (Umeyama)
/// -------------------------
#[derive(Clone, Copy)]
pub struct IcpP2P {
    pub max_iters: usize,
    pub max_corr_dist: f32,
    /// Trim top P percent largest residuals each iter (0.0..0.5 typically)
    pub trim_ratio: f32,
}
impl Default for IcpP2P {
    fn default() -> Self {
        Self { max_iters: 20, max_corr_dist: 0.1, trim_ratio: 0.1 }
    }
}

pub fn icp_point_to_point(src: &Cloud, tgt: &Cloud, cfg: IcpP2P) -> Result<Isometry3<f32>> {
    let index = GridIndex::build(soarus_core::CloudView::from(tgt), cfg.max_corr_dist);    let mut T = Isometry3::identity();

    for _ in 0..cfg.max_iters {
        // 1) gather correspondences
        let mut pairs: Vec<(Vector3<f32>, Vector3<f32>)> = Vec::with_capacity(src.len());
        for i in 0..src.len() {
            // transform source point
            let ps = T.transform_point(&Point3::new(src.x[i], src.y[i], src.z[i]));
            if let Some((j, _d2)) = index.nearest_in_radius([ps.x, ps.y, ps.z], cfg.max_corr_dist) {
                let pt = Point3::new(tgt.x[j], tgt.y[j], tgt.z[j]);
                pairs.push((ps.coords, pt.coords));
            }
        }
        if pairs.len() < 6 { break; }

        // 2) optional trimming (robust)
        if cfg.trim_ratio > 0.0 {
            pairs.sort_unstable_by(|(a,b), (c,d)| {
                let ra = (a - *b).norm_squared();
                let rb = (c - *d).norm_squared();
                ra.total_cmp(&rb)
            });
            let keep = ((pairs.len() as f32) * (1.0 - cfg.trim_ratio)).max(6.0) as usize;
            pairs.truncate(keep);
        }

        // 3) centroids
        let n = pairs.len() as f32;
        let mut cs = Vector3::zeros();
        let mut ct = Vector3::zeros();
        for (a,b) in &pairs { cs += *a; ct += *b; }
        cs /= n; ct /= n;

        // 4) cross-covariance
        let mut H = Matrix3::<f32>::zeros();
        for (a,b) in &pairs {
            let xa = *a - cs; let xb = *b - ct;
            H += xa * xb.transpose();
        }

        // 5) rotation via SVD (Umeyama)
        let svd = H.svd(true, true);
        let mut R = svd.v_t.unwrap().transpose() * svd.u.unwrap().transpose();
        if R.determinant() < 0.0 {
            // enforce right-handed
            let mut v = svd.v_t.unwrap().transpose();
            v.column_mut(2).neg_mut();
            R = v * svd.u.unwrap().transpose();
        }

        let t = ct - R * cs;

        // Convert Matrix3 -> Rotation3 -> UnitQuaternion
        let rot3 = Rotation3::from_matrix_unchecked(R);
        let uq   = UnitQuaternion::from_rotation_matrix(&rot3);
        let dT   = Isometry3::from_parts(Translation3::from(t), uq);

        // 6) compose and check convergence
        let prev = T;
        T = dT * T;
        let rot_delta = (prev.rotation.inverse() * T.rotation).angle();
        let trans_delta = (T.translation.vector - prev.translation.vector).norm();
        if rot_delta < 1e-5 && trans_delta < 1e-5 { break; }
    }
    Ok(T)
}

/// -------------------------
/// Point-to-Plane ICP (Gauss–Newton)
/// Requires target normals "nx,ny,nz"
/// -------------------------
#[derive(Clone, Copy)]
pub struct IcpPt2Plane {
    pub max_iters: usize,
    pub max_corr_dist: f32,
    /// Trim top P% largest absolute residuals each iter (0.0..0.5)
    pub trim_ratio: f32,
}
impl Default for IcpPt2Plane {
    fn default() -> Self {
        Self { max_iters: 30, max_corr_dist: 0.05, trim_ratio: 0.1 }
    }
}

pub fn icp_point_to_plane(src: &Cloud, tgt: &Cloud, cfg: IcpPt2Plane) -> Result<Isometry3<f32>> {
    // Target normals
    let (nx, ny, nz) = (
        tgt.attrs_f32.get("nx").ok_or_else(|| anyhow!("target missing nx"))?,
        tgt.attrs_f32.get("ny").ok_or_else(|| anyhow!("target missing ny"))?,
        tgt.attrs_f32.get("nz").ok_or_else(|| anyhow!("target missing nz"))?,
    );
    if nx.len()!=tgt.len() || ny.len()!=tgt.len() || nz.len()!=tgt.len() {
        return Err(anyhow!("target normal columns not same length as points"));
    }

    let index = GridIndex::build(soarus_core::CloudView::from(tgt), cfg.max_corr_dist);    let mut T = Isometry3::identity();

    for _ in 0..cfg.max_iters {
        // Gather correspondences with residuals
        let mut rows: Vec<([f32;6], f32)> = Vec::with_capacity(src.len());

        for i in 0..src.len() {
            // transform source
            let ps = T.transform_point(&Point3::new(src.x[i], src.y[i], src.z[i]));
            if let Some((j, _d2)) = index.nearest_in_radius([ps.x, ps.y, ps.z], cfg.max_corr_dist) {
                let q = Point3::new(tgt.x[j], tgt.y[j], tgt.z[j]);
                // plane normal at q
                let n = Vector3::new(nx[j], ny[j], nz[j]);
                let nn = n.norm();
                if nn < 1e-8 { continue; }
                let n = n / nn;

                // residual r = n · (ps - q)
                let r = (ps - q).dot(&n);
                // Jacobian J = [ n × ps , n ]  (sign picked for downhill update)
                let cross = ps.coords.cross(&n); // ps × n
                let jrow = [
                    -cross.x, -cross.y, -cross.z,  // -(ps × n) == (n × ps)
                    n.x,      n.y,      n.z,
                ];
                rows.push((jrow, r));
            }
        }
        if rows.len() < 12 { break; }

        // Robust trimming on |r|
        if cfg.trim_ratio > 0.0 {
            rows.sort_unstable_by(|a,b| a.1.abs().total_cmp(&b.1.abs()));
            let keep = ((rows.len() as f32)*(1.0 - cfg.trim_ratio)).max(12.0) as usize;
            rows.truncate(keep);
        }

        // Build normal equations JTJ (6x6) and JTr (6)
        let mut jtj = [[0f32;6];6];
        let mut jtr = [0f32;6];
        for (j, r) in &rows {
            for a in 0..6 {
                jtr[a] += j[a] * (-*r); // negative gradient
                for b in a..6 {
                    jtj[a][b] += j[a] * j[b];
                }
            }
        }
        // symmetric fill
        for a in 0..6 { for b in 0..a { jtj[a][b] = jtj[b][a]; } }

        // Solve 6x6 (Cholesky). If singular, stop.
        let dx = solve_sym6(&jtj, &jtr);
        if dx.is_none() { break; }
        let dx = dx.unwrap();

        // Update transform with small twist (rx,ry,rz, tx,ty,tz)
        let (rx, ry, rz, tx, ty, tz) = (dx[0], dx[1], dx[2], dx[3], dx[4], dx[5]);
        let rot = UnitQuaternion::from_euler_angles(rx, ry, rz);
        let dT  = Isometry3::from_parts(Translation3::new(tx,ty,tz), rot);

        let prev = T;
        T = dT * T;

        // stop if small update
        let rot_delta = (prev.rotation.inverse() * T.rotation).angle();
        let trans_delta = (T.translation.vector - prev.translation.vector).norm();
        if rot_delta < 1e-6 && trans_delta < 1e-6 { break; }
    }
    Ok(T)
}

/// tiny symmetric 6x6 solver (no pivot), returns None if numerically bad
fn solve_sym6(A: &[[f32;6];6], b: &[f32;6]) -> Option<[f32;6]> {
    let mut m = Matrix6::<f32>::zeros();
    for i in 0..6 { for j in 0..6 { m[(i,j)] = A[i][j]; } }
    let L = m.cholesky()?;
    let x = L.solve(&SVector::<f32,6>::from_row_slice(b));
    Some([x[0],x[1],x[2],x[3],x[4],x[5]])
}
