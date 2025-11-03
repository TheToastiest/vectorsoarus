use anyhow::Result;
use clap::{Parser, Subcommand};
use nalgebra::{Isometry3, Point3};
use soarus_core::{Cloud, CloudView};
use soarus_descriptors::*;
use soarus_ann::*;

// ---------- helpers (metrics / utils) ----------

fn t0() -> std::time::Instant { std::time::Instant::now() }
fn lap(t: std::time::Instant, label: &str) {
    let ms = t.elapsed().as_secs_f64()*1000.0;
    println!("[{label}] {ms:.1} ms");
}

/// Compute nearest-neighbor residuals (meters) of T(src) vs tgt within max_corr.
fn residuals_after_align(src: &Cloud, tgt: &Cloud, T: &Isometry3<f32>, max_corr: f32) -> Vec<f32> {
    let idx = soarus_nn::GridIndex::build(CloudView::from(tgt), max_corr.max(1e-3));
    let mut res = Vec::<f32>::with_capacity(src.len());
    for i in 0..src.len() {
        let p = T.transform_point(&Point3::new(src.x[i], src.y[i], src.z[i]));
        if let Some((_j, d2)) = idx.nearest_in_radius([p.x, p.y, p.z], max_corr) {
            res.push(d2.sqrt());
        }
    }
    res
}

/// Print RMSE and percentiles (p50/p90/p95/p99/max) for a residual vector (meters).
fn print_residual_stats(mut r: Vec<f32>, label: &str) {
    if r.is_empty() { println!("{label}: inliers=0"); return; }
    r.sort_by(|a,b| a.total_cmp(b));
    let n = r.len();
    let rmse = (r.iter().map(|x| x*x).sum::<f32>() / n as f32).sqrt();
    let pct = |p: f32| -> f32 {
        let i = ((p * (n as f32 - 1.0)).round() as usize).min(n-1);
        r[i]
    };
    println!("{label}: inliers={}  RMSE={:.6}  p50={:.6}  p90={:.6}  p95={:.6}  p99={:.6}  max={:.6}",
             n, rmse, pct(0.50), pct(0.90), pct(0.95), pct(0.99), *r.last().unwrap());
}

/// Apply an isometry to a cloud (returns a copy).
fn apply_iso(c: &Cloud, T: &Isometry3<f32>) -> Cloud {
    let mut o = c.clone();
    for i in 0..o.len() {
        let p = T.transform_point(&Point3::new(o.x[i], o.y[i], o.z[i]));
        o.x[i]=p.x; o.y[i]=p.y; o.z[i]=p.z;
    }
    o
}

fn parse_levels(csv: &str) -> Vec<f32> {
    csv.split(',').filter_map(|s| s.trim().parse::<f32>().ok()).filter(|v| *v > 0.0).collect()
}

// ---------- CLI ----------

#[derive(Parser)]
#[command(name="soarus", version, about="VectorSoarus — 3D point-cloud tools")]
struct Args { #[command(subcommand)] cmd: Cmd }

#[derive(Subcommand)]
enum Cmd {
    /// Print basic info about a file (PLY / KITTI .bin / LAS if enabled)
    Info { input: String },

    /// Voxel downsample
    Voxel {
        input: String, output: String,
        #[arg(short, long, default_value_t=0.05)] size: f32,
    },

    /// Estimate normals using a radius
    Normals {
        input: String, output: String,
        #[arg(short, long, default_value_t=0.2)] radius: f32,
    },

    /// Remove points with too few neighbors in a radius
    RadiusOutlier {
        input: String, output: String,
        #[arg(short, long, default_value_t=0.1)] radius: f32,
        #[arg(short='n', long, default_value_t=5)] min_pts: usize,
    },

    /// Statistical Outlier Removal (kNN µ±σ)
    Sor {
        input: String, output: String,
        #[arg(short='k', long, default_value_t=16)] k: usize,
        #[arg(short='s', long, default_value_t=1.0)] stddev_mul: f32,
    },

    /// Quick view (placeholder)
    View { input: String },

    /// Make a transformed copy (ground-truth pose)
    GenTransformed {
        input: String, output: String,
        #[arg(long, default_value_t=0.0)] tx: f32,
        #[arg(long, default_value_t=0.0)] ty: f32,
        #[arg(long, default_value_t=0.0)] tz: f32,
        #[arg(long, default_value_t=0.0)] rx_deg: f32,
        #[arg(long, default_value_t=0.0)] ry_deg: f32,
        #[arg(long, default_value_t=10.0)] rz_deg: f32,
    },

    /// ICP point-to-point: align src to tgt
    IcpP2p {
        src: String, tgt: String, output: String,
        #[arg(long, default_value_t=0.1)] max_corr: f32,
        #[arg(long, default_value_t=30)]  iters: usize,
        #[arg(long, default_value_t=0.1)] trim: f32,
    },

    /// ICP point-to-plane: align src to tgt (tgt must have nx,ny,nz)
    #[command(alias = "icp-pt2-plane")] // allow both spellings
    IcpPt2Plane {
        src: String, tgt: String, output: String,
        #[arg(long, default_value_t=0.05)] max_corr: f32,
        #[arg(long, default_value_t=40)]  iters: usize,
        #[arg(long, default_value_t=0.2)] trim: f32,
    },

    /// Multi-resolution ICP (coarse→fine). Voxel sizes CSV, e.g. "0.01,0.005,0.002"
    IcpPyramid {
        src: String, tgt: String, output: String,
        #[arg(long, default_value_t=String::from("0.01,0.005,0.002"))] levels: String,
        /// use point-to-plane (target normals computed per level)
        #[arg(long, default_value_t=true, action=clap::ArgAction::Set)] pt2plane: bool,
        #[arg(long, default_value_t=0.6)]  max_corr_scale: f32,
        #[arg(long, default_value_t=20)]   iters_per_level: usize,
        #[arg(long, default_value_t=0.2)]  trim: f32,
        /// normals radius as multiple of voxel size
        #[arg(long, default_value_t=1.5)]  normals_r_scale: f32,
    },

    /// Compute FPFH descriptors and write JSON (one row per point)
    Fpfh {
        input: String, output: String,
        #[arg(long, default_value_t=0.05)] radius: f32,
        #[arg(long, default_value_t=11)]  bins: usize,
        /// if set, estimate normals first with this radius (otherwise require nx/ny/nz)
        #[arg(long)] normals_radius: Option<f32>,
    },

    /// Match FPFH between two clouds; prints top-k stats
    MatchFpfh {
        src: String, tgt: String,
        #[arg(long, default_value_t=0.05)] radius: f32,
        #[arg(long, default_value_t=11)]  bins: usize,
        #[arg(long, default_value_t=1)]   k: usize,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Info { input } => cmd_info(&input),
        Cmd::Voxel { input, output, size } => cmd_voxel(&input, &output, size),
        Cmd::Normals { input, output, radius } => cmd_normals(&input, &output, radius),
        Cmd::RadiusOutlier { input, output, radius, min_pts } =>
            cmd_radius_outlier(&input, &output, radius, min_pts),
        Cmd::Sor { input, output, k, stddev_mul } =>
            cmd_sor(&input, &output, k, stddev_mul),
        Cmd::View { input } => cmd_view(&input),
        Cmd::GenTransformed { input, output, tx,ty,tz, rx_deg,ry_deg,rz_deg } =>
            cmd_gen_transformed(&input, &output, tx,ty,tz, rx_deg,ry_deg,rz_deg),
        Cmd::IcpP2p { src, tgt, output, max_corr, iters, trim } =>
            cmd_icp_p2p(&src, &tgt, &output, max_corr, iters, trim),
        Cmd::IcpPt2Plane { src, tgt, output, max_corr, iters, trim } =>
            cmd_icp_pt2plane(&src, &tgt, &output, max_corr, iters, trim),
        Cmd::IcpPyramid { src, tgt, output, levels, pt2plane, max_corr_scale, iters_per_level, trim, normals_r_scale } =>
            cmd_icp_pyramid(&src, &tgt, &output, &levels, pt2plane, max_corr_scale, iters_per_level, trim, normals_r_scale),
        Cmd::Fpfh { input, output, radius, bins, normals_radius } =>
            cmd_fpfh(&input, &output, radius, bins, normals_radius),
        Cmd::MatchFpfh { src, tgt, radius, bins, k } =>
            cmd_match_fpfh(&src, &tgt, radius, bins, k),
    }
}

// ---------- commands ----------

fn cmd_info(path: &str) -> Result<()> {
    let cloud = soarus_io::read_auto(path)?;
    println!("points: {}", cloud.len());
    Ok(())
}

fn cmd_voxel(input: &str, output: &str, size: f32) -> Result<()> {
    let cloud = soarus_io::read_auto(input)?;
    let out = soarus_filters::voxel_downsample(&cloud, size)?;
    soarus_io::write_ply_ascii(output, &out)?;
    println!("downsampled: {} -> {}", cloud.len(), out.len());
    Ok(())
}

fn cmd_normals(input: &str, output: &str, r: f32) -> Result<()> {
    let cloud = soarus_io::read_auto(input)?;
    let out = soarus_features::estimate_normals_radius(cloud, r)?;
    soarus_io::write_ply_ascii(output, &out)?;
    println!("normals estimated at r={}", r);
    Ok(())
}

fn cmd_radius_outlier(input: &str, output: &str, r: f32, min_pts: usize) -> Result<()> {
    let cloud = soarus_io::read_auto(input)?;
    let out = soarus_filters::radius_outlier(&cloud, r, min_pts)?;
    soarus_io::write_ply_ascii(output, &out)?;
    println!("radius_outlier: {} -> {} (r={}, min_pts={})", cloud.len(), out.len(), r, min_pts);
    Ok(())
}

fn cmd_sor(input: &str, output: &str, k: usize, stddev_mul: f32) -> Result<()> {
    let cloud = soarus_io::read_auto(input)?;
    let out = soarus_filters::statistical_outlier(&cloud, k, stddev_mul)?;
    soarus_io::write_ply_ascii(output, &out)?;
    println!("sor: {} -> {} (k={}, stddev_mul={})", cloud.len(), out.len(), k, stddev_mul);
    Ok(())
}

fn cmd_view(input: &str) -> Result<()> {
    let cloud = soarus_io::read_auto(input)?;
    soarus_viewer::show("VectorSoarus", &cloud)?;
    Ok(())
}

fn cmd_gen_transformed(input:&str, output:&str, tx:f32,ty:f32,tz:f32, rx_deg:f32,ry_deg:f32,rz_deg:f32) -> Result<()> {
    let mut c = soarus_io::read_auto(input)?;
    let rx = rx_deg.to_radians(); let ry = ry_deg.to_radians(); let rz = rz_deg.to_radians();
    let rot = nalgebra::UnitQuaternion::from_euler_angles(rx, ry, rz);
    let iso = nalgebra::Isometry3::from_parts(nalgebra::Translation3::new(tx,ty,tz), rot);
    for i in 0..c.len() {
        let p = iso.transform_point(&Point3::new(c.x[i], c.y[i], c.z[i]));
        c.x[i]=p.x; c.y[i]=p.y; c.z[i]=p.z;
    }
    soarus_io::write_ply_ascii(output, &c)?;
    println!("gen-transformed: wrote {}", output);
    Ok(())
}

fn cmd_icp_p2p(src:&str, tgt:&str, output:&str, max_corr:f32, iters:usize, trim:f32) -> Result<()> {
    let src_c = soarus_io::read_auto(src)?;
    let tgt_c = soarus_io::read_auto(tgt)?;
    let T = soarus_reg::icp_point_to_point(&src_c, &tgt_c, soarus_reg::IcpP2P{
        max_corr_dist: max_corr, max_iters: iters, trim_ratio: trim
    })?;

    println!("ICP result:");
    println!("  R =\n{}", T.rotation.to_rotation_matrix());
    println!("  t = {:?}", T.translation.vector);

    let mut out = src_c.clone();
    for i in 0..out.len() {
        let p = T.transform_point(&Point3::new(out.x[i], out.y[i], out.z[i]));
        out.x[i]=p.x; out.y[i]=p.y; out.z[i]=p.z;
    }

    let euler = T.rotation.euler_angles();
    let to_deg = |r: f32| r * 180.0 / std::f32::consts::PI;
    println!("Angles (deg): roll={:.3}, pitch={:.3}, yaw={:.3}",
             to_deg(euler.0), to_deg(euler.1), to_deg(euler.2));
    let res = residuals_after_align(&src_c, &tgt_c, &T, max_corr);
    print_residual_stats(res, "Residuals");

    soarus_io::write_ply_ascii(output, &out)?;
    println!("wrote aligned -> {}", output);
    Ok(())
}

fn cmd_icp_pt2plane(src:&str, tgt:&str, output:&str, max_corr:f32, iters:usize, trim:f32) -> Result<()> {
    let src_c = soarus_io::read_auto(src)?;
    let tgt_c = soarus_io::read_auto(tgt)?;
    let T = soarus_reg::icp_point_to_plane(&src_c, &tgt_c, soarus_reg::IcpPt2Plane{
        max_corr_dist: max_corr, max_iters: iters, trim_ratio: trim, ..Default::default()
    })?;

    println!("ICP pt2plane:");
    println!("  R =\n{}", T.rotation.to_rotation_matrix());
    println!("  t = {:?}", T.translation.vector);
    let e = T.rotation.euler_angles();
    let to_deg = |r: f32| r * 180.0 / std::f32::consts::PI;
    println!("Angles (deg): roll={:.3}, pitch={:.3}, yaw={:.3}", to_deg(e.0), to_deg(e.1), to_deg(e.2));
    let res = residuals_after_align(&src_c, &tgt_c, &T, max_corr);
    print_residual_stats(res, "Residuals");

    let mut out = src_c.clone();
    for i in 0..out.len() {
        let p = T.transform_point(&Point3::new(out.x[i], out.y[i], out.z[i]));
        out.x[i]=p.x; out.y[i]=p.y; out.z[i]=p.z;
    }
    soarus_io::write_ply_ascii(output, &out)?;
    println!("wrote aligned -> {}", output);
    Ok(())
}

fn cmd_icp_pyramid(
    src: &str, tgt: &str, output: &str, levels_csv: &str,
    use_pt2plane: bool, max_corr_scale: f32, iters: usize, trim: f32, normals_r_scale: f32
) -> Result<()> {
    let src_full = soarus_io::read_auto(src)?;
    let tgt_full = soarus_io::read_auto(tgt)?;

    let mut T = Isometry3::identity();
    let levels = parse_levels(levels_csv);
    if levels.is_empty() { anyhow::bail!("no valid levels; example: --levels 0.01,0.005,0.002"); }

    for (li, vox) in levels.iter().enumerate() {
        let src_ds = soarus_filters::voxel_downsample(&src_full, *vox)?;
        let mut tgt_ds = soarus_filters::voxel_downsample(&tgt_full, *vox)?;
        let max_corr = max_corr_scale * *vox;

        if use_pt2plane {
            tgt_ds = soarus_features::estimate_normals_radius(tgt_ds, normals_r_scale * *vox)?;
            let step = soarus_reg::icp_point_to_plane(
                &apply_iso(&src_ds, &T),
                &tgt_ds,
                soarus_reg::IcpPt2Plane { max_iters: iters, max_corr_dist: max_corr, trim_ratio: trim, ..Default::default() }
            )?;
            T = step * T;
        } else {
            let step = soarus_reg::icp_point_to_point(
                &apply_iso(&src_ds, &T),
                &tgt_ds,
                soarus_reg::IcpP2P { max_iters: iters, max_corr_dist: max_corr, trim_ratio: trim }
            )?;
            T = step * T;
        }

        let res = residuals_after_align(&src_ds, &tgt_ds, &T, max_corr);
        let rmse = if res.is_empty() { f32::NAN } else { (res.iter().map(|x| x*x).sum::<f32>()/res.len() as f32).sqrt() };
        let p99 = if res.is_empty() { f32::NAN } else {
            let mut r = res.clone(); r.sort_by(|a,b| a.total_cmp(b));
            r[((0.99 * (r.len() as f32 - 1.0)).round() as usize).min(r.len()-1)]
        };
        println!("Level {} voxel={:.6}  inliers={}  RMSE={:.6}  p99={:.6}", li, vox, res.len(), rmse, p99);
    }

    let mut out = src_full.clone();
    for i in 0..out.len() {
        let p = T.transform_point(&Point3::new(out.x[i], out.y[i], out.z[i]));
        out.x[i]=p.x; out.y[i]=p.y; out.z[i]=p.z;
    }
    soarus_io::write_ply_ascii(output, &out)?;
    println!("pyramid ICP: wrote {}", output);
    Ok(())
}

fn cmd_fpfh(input:&str, output:&str, r:f32, bins:usize, normals_r:Option<f32>) -> Result<()> {
    let t_read = t0();
    let mut cloud = soarus_io::read_auto(input)?;
    lap(t_read, "read");

    let t_norm = t0();
    if let Some(nr) = normals_r {
        cloud = soarus_features::estimate_normals_radius(cloud, nr)?;
        lap(t_norm, "normals");
    }

    let t_fpfh = t0();
    let desc = soarus_descriptors::fpfh(&cloud, soarus_descriptors::FpfhCfg{ radius:r, bins })?;
    lap(t_fpfh, "fpfh");

    let t_write = t0();
    serde_json::to_writer(std::fs::File::create(output)?, &desc)?;
    lap(t_write, "write");

    println!("fpfh: {} pts → {} (dim={})", cloud.len(), output, 3*bins);
    Ok(())
}

fn cmd_match_fpfh(src:&str, tgt:&str, r:f32, bins:usize, k:usize) -> Result<()> {
    let mut cs = soarus_io::read_auto(src)?;
    let mut ct = soarus_io::read_auto(tgt)?;

    // ensure normals (compute if missing)
    for (c,name) in [(&mut cs,"src"), (&mut ct,"tgt")] {
        if !c.attrs_f32.contains_key("nx") {
            let rad = r.max(1e-3);
            *c = soarus_features::estimate_normals_radius(std::mem::take(c), rad)?;
            println!("{name}: computed normals (r={})", rad);
        }
    }

    let t_comp = t0();
    let ds = soarus_descriptors::fpfh(&cs, soarus_descriptors::FpfhCfg{ radius:r, bins })?;
    let dt = soarus_descriptors::fpfh(&ct, soarus_descriptors::FpfhCfg{ radius:r, bins })?;
    lap(t_comp, "compute fpfh (src+tgt)");

    let dim = 3*bins;
    anyhow::ensure!(!ds.is_empty() && !dt.is_empty(), "empty descriptors");

    let t_build = t0();
    let mut ann = soarus_ann::BruteL2::new(dim);
    for (i, v) in dt.iter().enumerate() { ann.add(i, v)?; }
    lap(t_build, "ANN build");

    let t_query = t0();
    let mut total=0usize; let mut sumd=0.0f32;
    for v in &ds { for (_id, d) in ann.search(v, k)? { sumd += d; total += 1; } }
    lap(t_query, "ANN queries");

    let avg = if total>0 { sumd / total as f32 } else { f32::NAN };
    println!("match-fpfh: src_pts={} tgt_pts={} k={}  avg_L2={:.6}", ds.len(), dt.len(), k, avg);
    Ok(())
}
