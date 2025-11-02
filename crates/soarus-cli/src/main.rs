use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name="soarus", version, about="VectorSoarus CLI")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd
}

#[derive(Subcommand)]
enum Cmd {
    /// Print basic info
    Info { input: String },
    /// Voxel downsample
    Voxel { input: String, output: String, #[arg(short, long, default_value_t=0.05)] size: f32 },
    /// Estimate normals (radius)
    Normals { input: String, output: String, #[arg(short, long, default_value_t=0.2)] radius: f32 },
    /// Quick view (no-op for now)
    View { input: String }
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Info { input } => cmd_info(&input),
        Cmd::Voxel { input, output, size } => cmd_voxel(&input, &output, size),
        Cmd::Normals { input, output, radius } => cmd_normals(&input, &output, radius),
        Cmd::View { input } => cmd_view(&input),
    }
}

fn cmd_info(path: &str) -> Result<()> {
    let cloud = soarus_io::read_ply_ascii(path)?;
    println!("points: {}", cloud.len());
    Ok(())
}

fn cmd_voxel(input: &str, output: &str, size: f32) -> Result<()> {
    let cloud = soarus_io::read_ply_ascii(input)?;
    let out = soarus_filters::voxel_downsample(&cloud, size)?;
    soarus_io::write_ply_ascii(output, &out)?;
    println!("downsampled: {} -> {}", cloud.len(), out.len());
    Ok(())
}

fn cmd_normals(input: &str, output: &str, r: f32) -> Result<()> {
    let cloud = soarus_io::read_ply_ascii(input)?;
    let out = soarus_features::estimate_normals_radius(cloud, r)?;
    soarus_io::write_ply_ascii(output, &out)?;
    println!("normals estimated at r={}", r);
    Ok(())
}

fn cmd_view(input: &str) -> Result<()> {
    let cloud = soarus_io::read_ply_ascii(input)?;
    soarus_viewer::show("VectorSoarus", &cloud)?;
    Ok(())
}
