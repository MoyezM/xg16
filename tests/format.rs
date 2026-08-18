use xg16::{Chunker, scan};

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

fn cuts_with(
    c: &Chunker,
    data: &[u8],
    scan: impl Fn(&[u8], u64, u64) -> (Option<usize>, u64) + Copy,
) -> Vec<usize> {
    let mut cuts = Vec::new();
    let mut off = 0;
    while off < data.len() {
        let len = c.next_cut_with(&data[off..], scan);
        assert!(len > 0);
        off += len;
        cuts.push(off);
    }
    cuts
}

#[test]
fn avx2_matches_reference() {
    let c = Chunker::with_default_sizes();
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
    }
}

#[test]
fn chunks_partition_input() {
    let c = Chunker::with_default_sizes();
    let mut data = vec![0u8; 16 << 20];
    fill_random(&mut data, 3);
    let chunks: Vec<&[u8]> = c.chunks(&data).collect();
    let total: usize = chunks.iter().map(|s| s.len()).sum();
    assert_eq!(total, data.len());
    for (i, ch) in chunks.iter().enumerate() {
        assert!(ch.len() <= c.max_size());
        assert!(!ch.is_empty());
        if i + 1 < chunks.len() {
            assert!(ch.len() > c.min_size());
        }
    }
    let avg = data.len() / chunks.len();
    assert!(
        avg > c.avg_size() * 6 / 10 && avg < c.avg_size() * 14 / 10,
        "avg chunk {avg} vs target {}",
        c.avg_size()
    );
}

#[test]
fn structured_data_distribution() {
    let c = Chunker::with_default_sizes();
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
        let cuts = c.cut_points(data);
        let avg = data.len() / cuts.len();
        assert!(
            avg > c.avg_size() / 3 && avg < c.avg_size() * 3,
            "{kind}: avg chunk {avg} vs target {}",
            c.avg_size()
        );
    }
}

#[test]
fn degenerate_inputs() {
    let c = Chunker::with_default_sizes();
    assert_eq!(c.chunks(&[]).count(), 0);
    let small = vec![7u8; 100];
    assert_eq!(c.cut_points(&small), vec![100]);

    // Constant data: state is eventually constant, so cuts are periodic
    // or forced at max — never pathological minimum-size chunks.
    for byte in [0u8, 0xAA, 0xFF] {
        let flat = vec![byte; 1 << 20];
        let cuts = c.cut_points(&flat);
        let first = cuts[0];
        assert!(
            first > c.min_size(),
            "constant 0x{byte:02x} produced tiny chunks"
        );
        for w in cuts.windows(2).take(cuts.len().saturating_sub(2)) {
            assert_eq!(w[1] - w[0], first, "constant 0x{byte:02x} not periodic");
        }
    }
}

/// The property this format exists for: byte-granular cuts resync within
/// ~a chunk after inserts of ANY length.
#[test]
fn resync_after_any_insert_length() {
    let c = Chunker::with_default_sizes();
    let mut data = vec![0u8; 8 << 20];
    fill_random(&mut data, 7);
    let edit_at = 1 << 20;
    let before = c.cut_points(&data);
    for ins_len in [1usize, 3, 7, 13, 31, 33, 4096] {
        let ins: Vec<u8> = (0..ins_len).map(|i| i as u8 ^ 0x5A).collect();
        let mut edited = data[..edit_at].to_vec();
        edited.extend_from_slice(&ins);
        edited.extend_from_slice(&data[edit_at..]);
        let after = c.cut_points(&edited);
        let shifted: std::collections::HashSet<usize> =
            before.iter().map(|&p| p + ins_len).collect();
        let resync = after
            .iter()
            .copied()
            .find(|p| *p > edit_at && shifted.contains(p))
            .unwrap_or(usize::MAX);
        let dist = resync.saturating_sub(edit_at);
        assert!(
            dist < 4 * c.max_size(),
            "insert of {ins_len}: resync took {dist} bytes"
        );
    }
}

/// Streaming must produce byte-identical chunks to slice chunking,
/// regardless of how the input is split into pushes.
#[test]
fn stream_matches_slice_chunking() {
    let c = Chunker::with_default_sizes();
    let mut data = vec![0u8; 4 << 20];
    fill_random(&mut data, 11);
    let expected: Vec<Vec<u8>> = c.chunks(&data).map(|s| s.to_vec()).collect();

    // Deterministic pseudo-random push sizes, including tiny and huge.
    let mut seed = 99u64;
    let mut sizes = Vec::new();
    let mut total = 0usize;
    while total < data.len() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let s = match seed >> 60 {
            0 => 1,
            1 => 7,
            2..=8 => (seed % 8192) as usize + 1,
            _ => (seed % 200_000) as usize + 1,
        };
        let s = s.min(data.len() - total);
        sizes.push(s);
        total += s;
    }

    let mut got: Vec<Vec<u8>> = Vec::new();
    let mut stream = xg16::StreamChunker::new(c);
    let mut off = 0;
    for s in sizes {
        stream.push(&data[off..off + s], |ch| got.push(ch.to_vec()));
        off += s;
    }
    stream.finish(|ch| got.push(ch.to_vec()));
    assert_eq!(got.len(), expected.len());
    assert_eq!(got, expected);
}
