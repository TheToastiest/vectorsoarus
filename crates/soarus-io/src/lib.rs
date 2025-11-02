//! soarus-io — tiny PLY reader/writer (ASCII) as a starting point.
//! Extend with LAS/LAZ/COPC later.

use anyhow::{bail, Context, Result};
use ply_rs::parser::Parser;
use ply_rs::ply::{Addable, DefaultElement, ElementDef, Ply};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use soarus_core::Cloud;


pub fn read_las(path: &str) -> Result<Cloud> {
    let mut r = las::Reader::from_path(path).with_context(|| format!("open {}", path))?;
    let hdr = r.header().clone();

    let mut c = Cloud::default();
    c.reserve(hdr.number_of_points() as usize);

    for rec in r.points() {
        let p = rec?; // las::Point
        // x/y/z are f64 with scale/offset already applied by Reader
        c.push(p.x as f32, p.y as f32, p.z as f32);

        // intensity is ALWAYS present as u16 in LAS point formats
        c.attrs_f32.entry("intensity".into())
            .or_default()
            .push(p.intensity as f32);

        // optional extras
        if let Some(color) = p.color {
            c.attrs_f32.entry("red".into()).or_default().push(color.red as f32);
            c.attrs_f32.entry("green".into()).or_default().push(color.green as f32);
            c.attrs_f32.entry("blue".into()).or_default().push(color.blue as f32);
        } else {
            // keep columns aligned if they already exist
            for k in ["red","green","blue"] {
                if let Some(col) = c.attrs_f32.get_mut(k) { col.push(0.0); }
            }
        }

        // classification (enum) -> u8 code via From<Classification> for u8
        let class_code: u8 = u8::from(p.classification);
        c.attrs_f32.entry("class".into())
            .or_default()
            .push(class_code as f32);

    }
    Ok(c)
}


pub fn read_ply_ascii(path: &str) -> Result<Cloud> {
    let f = File::open(path).with_context(|| format!("open {}", path))?;
    let mut reader = BufReader::new(f);
    let parser = Parser::<DefaultElement>::new();
    let ply: Ply<DefaultElement> = parser.read_ply(&mut reader)?;

    // Expect "vertex" element with float x,y,z.
    let vertex = ply.payload.get("vertex")
        .ok_or_else(|| anyhow::anyhow!("PLY missing 'vertex' element"))?;

    let mut c = Cloud::default();
    c.reserve(vertex.len());

    for el in vertex {
        let x = get_f32(el, "x")?;
        let y = get_f32(el, "y")?;
        let z = get_f32(el, "z")?;
        c.push(x, y, z);
        // after: c.push(x, y, z);
        let idx = c.len() - 1; // position of the just-pushed point

        for k in ["red","green","blue","intensity","nx","ny","nz"] {
            let val = get_f32(el, k).unwrap_or(0.0);

            // Create the column if missing without capturing `c` in a closure.
            let col = c.attrs_f32.entry(k.to_string()).or_insert_with(Vec::new);

            // Ensure the column is aligned up to the previous index.
            if col.len() < idx {
                col.resize(idx, 0.0);
            }

            // Push this point's value (or 0.0 if the property was missing).
            col.push(val);
        }

    }
    Ok(c)
}

fn get_f32(el: &DefaultElement, key: &str) -> Result<f32> {
    match el.get(key) {
        Some(ply_rs::ply::Property::Float(v)) => Ok(*v),
        Some(ply_rs::ply::Property::Double(v)) => Ok(*v as f32),
        Some(ply_rs::ply::Property::UChar(v)) => Ok(*v as f32),
        Some(_) => bail!("property '{}' not float-like", key),
        None => bail!("missing property '{}'", key),
    }
}

pub fn write_ply_ascii(path: &str, cloud: &Cloud) -> Result<()> {
    let n = cloud.len();
    let mut w = BufWriter::new(File::create(path)?);

    // Collect float attributes that match length
    let mut keys: Vec<&str> = cloud
        .attrs_f32
        .iter()
        .filter_map(|(k, v)| if v.len() == n { Some(k.as_str()) } else { None })
        .collect();
    // stable, nice order: x y z then nx ny nz then the rest alpha-sorted
    keys.sort();
    let mut ordered: Vec<&str> = Vec::new();
    for k in ["nx","ny","nz"] {
        if keys.binary_search(&k).is_ok() { ordered.push(k); }
    }
    for k in keys {
        if !ordered.iter().any(|&ok| ok == k) {
            ordered.push(k);
        }
    }

    // Header
    writeln!(w, "ply")?;
    writeln!(w, "format ascii 1.0")?;
    writeln!(w, "element vertex {}", n)?;
    writeln!(w, "property float x")?;
    writeln!(w, "property float y")?;
    writeln!(w, "property float z")?;
    for k in &ordered {
        writeln!(w, "property float {}", k)?;
    }
    writeln!(w, "end_header")?;

    // Body
    for i in 0..n {
        write!(w, "{} {} {}", cloud.x[i], cloud.y[i], cloud.z[i])?;
        for k in &ordered {
            let col = &cloud.attrs_f32[*k];
            write!(w, " {}", col[i])?;
        }
        writeln!(w)?;
    }
    Ok(())
}
pub fn read_auto(path: &str) -> Result<Cloud> {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".ply") {
        return read_ply_ascii(path);
    }
    // If/when you add KITTI or LAS, wire them here, e.g.:
    // if lower.ends_with(".bin") { return read_kitti_bin(path); }
    // #[cfg(feature = "lasio")]
    // if lower.ends_with(".las") || lower.ends_with(".laz") { return read_las(path); }

    // fallback: try PLY and report clearly
    match read_ply_ascii(path) {
        Ok(c) => Ok(c),
        Err(e) => Err(anyhow::anyhow!("Unsupported file (expected .ply). Root error: {e}")),
    }
}