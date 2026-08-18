//! Dedup-ratio benchmark: chunk a base file and an edited version, and
//! measure how many bytes of the edited version deduplicate against the
//! base's chunk set. This is the metric chunking exists to maximize —
//! throughput tells you what a backup run costs, this tells you what it
//! stores.
//!
//! Run: `cargo run --release --example dedup_ratio`

use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};

use xg16::Chunker;

fn fill_random(buf: &mut [u8], mut seed: u64) {
    let mut i = 0;
    while i + 8 <= buf.len() {
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        buf[i..i + 8].copy_from_slice(&z.to_le_bytes());
        i += 8;
    }
}

/// Cheap deterministic PRNG for edit placement.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn chunk_hashes(c: &Chunker, data: &[u8]) -> Vec<(u64, usize)> {
    c.chunks(data)
        .map(|ch| {
            let mut h = DefaultHasher::new();
            ch.hash(&mut h);
            (h.finish(), ch.len())
        })
        .collect()
}

fn scenario(name: &str, c: &Chunker, base: &[u8], edited: &[u8]) {
    let base_set: HashSet<u64> = chunk_hashes(c, base).into_iter().map(|(h, _)| h).collect();
    let v2 = chunk_hashes(c, edited);
    let total: usize = v2.iter().map(|&(_, l)| l).sum();
    let new_bytes: usize = v2
        .iter()
        .filter(|(h, _)| !base_set.contains(h))
        .map(|&(_, l)| l)
        .sum();
    let dedup_pct = 100.0 * (total - new_bytes) as f64 / total as f64;
    println!(
        "{name:>28}  chunks: {:>5}  new data stored: {:>9} B  deduped: {dedup_pct:6.2}%",
        v2.len(),
        new_bytes,
    );
}

fn main() {
    const LEN: usize = 32 << 20;
    let mut base = vec![0u8; LEN];
    fill_random(&mut base, 42);
    let c = Chunker::with_default_sizes();
    let mut rng = Rng(7);

    println!("base: {} MiB, chunker 2K/8K/64K\n", LEN >> 20);

    // Identity: everything should dedup.
    scenario("identical", &c, &base, &base.clone());

    // One small odd-length insert mid-file.
    {
        let at = LEN / 2;
        let mut v = base[..at].to_vec();
        v.extend_from_slice(b"odd");
        v.extend_from_slice(&base[at..]);
        scenario("1 insert of 3 B", &c, &base, &v);
    }

    // One 1 KiB insert.
    {
        let at = LEN / 3;
        let mut ins = vec![0u8; 1024];
        fill_random(&mut ins, 99);
        let mut v = base[..at].to_vec();
        v.extend_from_slice(&ins);
        v.extend_from_slice(&base[at..]);
        scenario("1 insert of 1 KiB", &c, &base, &v);
    }

    // 20 scattered small inserts of random odd sizes (the edit-heavy
    // text-file model; each edit is a fresh resync event).
    {
        let mut v = base.clone();
        let mut offsets: Vec<usize> = (0..20).map(|_| rng.below(v.len())).collect();
        offsets.sort_unstable_by(|a, b| b.cmp(a)); // back-to-front keeps offsets valid
        for at in offsets {
            let len = 1 + rng.below(64);
            let ins: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            v.splice(at..at, ins);
        }
        scenario("20 scattered inserts <64 B", &c, &base, &v);
    }

    // 100-byte deletion.
    {
        let at = LEN / 4;
        let mut v = base.clone();
        v.drain(at..at + 100);
        scenario("1 delete of 100 B", &c, &base, &v);
    }

    // In-place 4 KiB rewrite at a 4K-aligned offset (VM-image model:
    // no content shift at all).
    {
        let at = (LEN / 2) & !4095;
        let mut v = base.clone();
        let mut blk = vec![0u8; 4096];
        fill_random(&mut blk, 1234);
        v[at..at + 4096].copy_from_slice(&blk);
        scenario("4 KiB aligned overwrite", &c, &base, &v);
    }

    // 1 MiB append.
    {
        let mut tail = vec![0u8; 1 << 20];
        fill_random(&mut tail, 555);
        let mut v = base.clone();
        v.extend_from_slice(&tail);
        scenario("1 MiB append", &c, &base, &v);
    }
}
