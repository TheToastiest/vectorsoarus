use anyhow::Result;
use soarus_core::Cloud;

/// Placeholder viewer until we wire egui/wgpu or Rerun.
pub fn show(_title: &str, _cloud: &Cloud) -> Result<()> {
    // no-op
    Ok(())
}
