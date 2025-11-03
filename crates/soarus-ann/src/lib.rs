use anyhow::Result;

pub trait AnnIndex {
    fn add(&mut self, key: usize, vec: &[f32]) -> Result<()>;
    /// returns k nearest ids with L2 distance
    fn search(&self, q: &[f32], k: usize) -> Result<Vec<(usize, f32)>>;
}

/// Simple linear ANN (L2). Replace with raggedy_anndy backend later.
pub struct BruteL2 {
    dim: usize,
    data: Vec<(usize, Vec<f32>)>,
}

impl BruteL2 {
    pub fn new(dim: usize) -> Self { Self { dim, data: Vec::new() } }
}

impl AnnIndex for BruteL2 {
    fn add(&mut self, key: usize, vec: &[f32]) -> Result<()> {
        if self.data.is_empty() { self.dim = vec.len(); }
        anyhow::ensure!(vec.len()==self.dim, "dim mismatch");
        self.data.push((key, vec.to_vec()));
        Ok(())
    }
    fn search(&self, q: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        anyhow::ensure!(q.len()==self.dim, "query dim mismatch");
        let mut hits: Vec<(usize,f32)> = self.data.iter().map(|(id,v)| {
            let d = l2(q, v);
            (*id, d)
        }).collect();
        hits.sort_by(|a,b| a.1.total_cmp(&b.1));
        hits.truncate(k);
        Ok(hits)
    }
}

#[inline] fn l2(a:&[f32], b:&[f32])->f32 {
    let mut s=0.0; for i in 0..a.len(){ let d=a[i]-b[i]; s+=d*d; } s.sqrt()
}
