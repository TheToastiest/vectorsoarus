//! soarus-io — tiny PLY reader/writer (ASCII) as a starting point.
//! Extend with LAS/LAZ/COPC later.

use anyhow::{bail, Context, Result};
use ply_rs::parser::Parser;
use ply_rs::ply::{Addable, DefaultElement, ElementDef, Ply};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use soarus_core::Cloud;

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
    // Minimal ASCII writer (header+lines)
    let mut w = BufWriter::new(File::create(path)?);
    writeln!(w, "ply")?;
    writeln!(w, "format ascii 1.0")?;
    writeln!(w, "element vertex {}", cloud.len())?;
    writeln!(w, "property float x")?;
    writeln!(w, "property float y")?;
    writeln!(w, "property float z")?;
    writeln!(w, "end_header")?;
    for i in 0..cloud.len() {
        writeln!(w, "{} {} {}", cloud.x[i], cloud.y[i], cloud.z[i])?;
    }
    Ok(())
}
