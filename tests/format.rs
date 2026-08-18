use std::io::Read;

use xg16::{Config, StreamChunker, Xg16, scan};

const MIN: usize = 2048;
const AVG: usize = 8192;
const MAX: usize = 65536;

fn cut_points(data: &[u8]) -> Vec<usize> {
    Xg16::new(data, MIN, AVG, MAX)
        .map(|c| c.offset + c.length)
        .collect()
}

/// Deterministic test data (splitmix64 stream).
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
    for b in &mut buf[i..] {
        seed = seed.wrapping_add(1);
        *b = seed as u8;
    }
}

/// Cut end-offsets AND cut-state hashes, via a specific kernel.
fn cuts_with(
    c: &Config,
    data: &[u8],
    scan: impl Fn(&[u8], u64, u64) -> (Option<usize>, u64) + Copy,
) -> Vec<(usize, u64)> {
    let mut cuts = Vec::new();
    let mut off = 0;
    while off < data.len() {
        let (len, hash) = c.cut_with(&data[off..], scan);
        assert!(len > 0);
        off += len;
        cuts.push((off, hash));
    }
    cuts
}

#[test]
fn avx2_matches_reference() {
    let c = Config::new(MIN, AVG, MAX);
    for seed in 0..8u64 {
        let len = 1_000_003 + (seed as usize) * 77_777 + (seed as usize % 13);
        let mut data = vec![0u8; len];
        fill_random(&mut data, seed);
        let r = cuts_with(&c, &data, scan::scan_ref);
        // Every kernel available on this machine must match the
        // reference cut-for-cut — new arch kernels are covered by
        // registering in scan::kernels().
        for (name, k) in scan::kernels() {
            assert_eq!(r, cuts_with(&c, &data, k), "{name} diverged (seed {seed})");
        }
        assert_eq!(
            r,
            cuts_with(&c, &data, scan::scan),
            "dispatched kernel diverged (seed {seed})"
        );
        // The public iterator agrees with the reference too.
        let via_iter: Vec<(usize, u64)> = Xg16::new(&data, MIN, AVG, MAX)
            .map(|c| (c.offset + c.length, c.hash))
            .collect();
        assert_eq!(r, via_iter, "public iterator diverged (seed {seed})");
    }
}

#[test]
fn chunks_partition_input() {
    let mut data = vec![0u8; 16 << 20];
    fill_random(&mut data, 3);
    let chunks: Vec<xg16::Chunk> = Xg16::new(&data, MIN, AVG, MAX).collect();
    let mut expect_offset = 0;
    for (i, ch) in chunks.iter().enumerate() {
        assert_eq!(ch.offset, expect_offset, "chunks must tile the input");
        expect_offset += ch.length;
        assert!(ch.length <= MAX);
        assert!(ch.length > 0);
        if i + 1 < chunks.len() {
            assert!(ch.length > MIN);
        }
    }
    assert_eq!(expect_offset, data.len());
    let avg = data.len() / chunks.len();
    assert!(
        avg > AVG * 6 / 10 && avg < AVG * 14 / 10,
        "avg chunk {avg} vs target {AVG}"
    );
}

#[test]
fn structured_data_distribution() {
    // Incrementing u64 counters.
    let mut counters = Vec::with_capacity(8 << 20);
    for i in 0..(1u64 << 20) {
        counters.extend_from_slice(&i.to_le_bytes());
    }
    // Log-like ASCII text.
    let mut text = Vec::with_capacity(8 << 20);
    let mut i = 0u64;
    while text.len() < (8 << 20) {
        text.extend_from_slice(
            format!("the quick brown fox {i} jumps over the lazy dog\n").as_bytes(),
        );
        i += 1;
    }
    for (kind, data) in [("counters", &counters), ("text", &text)] {
        let cuts = cut_points(data);
        let avg = data.len() / cuts.len();
        assert!(
            avg > AVG / 3 && avg < AVG * 3,
            "{kind}: avg chunk {avg} vs target {AVG}"
        );
    }
}

#[test]
fn degenerate_inputs() {
    assert_eq!(Xg16::new(&[], MIN, AVG, MAX).count(), 0);
    let small = vec![7u8; 100];
    assert_eq!(cut_points(&small), vec![100]);

    // Constant data: state is eventually constant, so cuts are periodic
    // or forced at max — never pathological minimum-size chunks.
    for byte in [0u8, 0xAA, 0xFF] {
        let flat = vec![byte; 1 << 20];
        let cuts = cut_points(&flat);
        let first = cuts[0];
        assert!(first > MIN, "constant 0x{byte:02x} produced tiny chunks");
        for w in cuts.windows(2).take(cuts.len().saturating_sub(2)) {
            assert_eq!(w[1] - w[0], first, "constant 0x{byte:02x} not periodic");
        }
    }
}

/// The property this format exists for: byte-granular cuts resync within
/// ~a chunk after inserts of ANY length.
#[test]
fn resync_after_any_insert_length() {
    let mut data = vec![0u8; 8 << 20];
    fill_random(&mut data, 7);
    let edit_at = 1 << 20;
    let before = cut_points(&data);
    for ins_len in [1usize, 3, 7, 13, 31, 33, 4096] {
        let ins: Vec<u8> = (0..ins_len).map(|i| i as u8 ^ 0x5A).collect();
        let mut edited = data[..edit_at].to_vec();
        edited.extend_from_slice(&ins);
        edited.extend_from_slice(&data[edit_at..]);
        let after = cut_points(&edited);
        let shifted: std::collections::HashSet<usize> =
            before.iter().map(|&p| p + ins_len).collect();
        let resync = after
            .iter()
            .copied()
            .find(|p| *p > edit_at && shifted.contains(p))
            .unwrap_or(usize::MAX);
        let dist = resync.saturating_sub(edit_at);
        assert!(
            dist < 4 * MAX,
            "insert of {ins_len}: resync took {dist} bytes"
        );
    }
}

/// A reader that returns deliberately awkward read sizes.
struct WeirdReader<'a> {
    data: &'a [u8],
    pos: usize,
    seed: u64,
}

impl Read for WeirdReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.data.len() {
            return Ok(0);
        }
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let n = match self.seed >> 60 {
            0 => 1,
            1 => 7,
            2..=8 => (self.seed % 4096) as usize + 1,
            _ => (self.seed % 30_000) as usize + 1,
        };
        let n = n.min(buf.len()).min(self.data.len() - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// Streaming must produce identical chunks (offsets, lengths, hashes,
/// bytes) to slice chunking, regardless of how reads are sized.
#[test]
fn stream_matches_slice_chunking() {
    let mut data = vec![0u8; 4 << 20];
    fill_random(&mut data, 11);
    let expected: Vec<xg16::Chunk> = Xg16::new(&data, MIN, AVG, MAX).collect();

    let reader = WeirdReader {
        data: &data,
        pos: 0,
        seed: 99,
    };
    let got: Vec<xg16::ChunkData> = StreamChunker::new(reader, MIN, AVG, MAX)
        .map(|r| r.unwrap())
        .collect();

    assert_eq!(got.len(), expected.len());
    for (g, e) in got.iter().zip(&expected) {
        assert_eq!(g.offset as usize, e.offset);
        assert_eq!(g.length, e.length);
        assert_eq!(g.hash, e.hash);
        assert_eq!(g.data, &data[e.offset..e.offset + e.length]);
    }
}
