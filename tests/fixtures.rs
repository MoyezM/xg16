//! Golden cut fixtures: the format contract made executable.
//!
//! `tests/data/corpus.bin` is a committed byte corpus (bytes, not a
//! generator — generated code can drift and silently regenerate fixtures
//! around the drift). For each parameter set, `tests/data/cuts_*.txt`
//! pins every chunk's `offset length hash`. Hashes are pinned too:
//! a kernel that produces correct cuts with a wrong carried state would
//! corrupt phase-2 masking later rather than failing now.
//!
//! Every kernel available on the running machine is checked against the
//! fixtures, so cross-architecture drift (x86 CI vs arm CI) fails loudly.
//!
//! To regenerate after an INTENTIONAL format change (which must also bump
//! `xg16::FORMAT_ID`):
//!
//! ```text
//! XG16_BLESS=1 cargo test --release --test fixtures -- --ignored bless
//! ```

use std::path::Path;

use xg16::{Config, Xg16, scan};

/// (name, min, avg, max). vold params per its context doc; avg = 16 KiB
/// exercises the design envelope's edge (hard mask = all 16 state bits).
const PARAM_SETS: &[(&str, usize, usize, usize)] = &[
    ("default", 2048, 8192, 65536),
    ("vold8", 4096, 8192, 65536),
    ("vold16", 4096, 16384, 65536),
];

fn data_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

fn corpus_path() -> std::path::PathBuf {
    data_dir().join("corpus.bin")
}

fn cuts_path(name: &str) -> std::path::PathBuf {
    data_dir().join(format!("cuts_{name}.txt"))
}

fn splitmix_fill(buf: &mut Vec<u8>, len: usize, mut seed: u64) {
    let mut out = vec![0u8; len];
    let mut i = 0;
    while i + 8 <= len {
        seed = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        out[i..i + 8].copy_from_slice(&z.to_le_bytes());
        i += 8;
    }
    for b in &mut out[i..] {
        seed = seed.wrapping_add(1);
        *b = seed as u8;
    }
    buf.extend_from_slice(&out);
}

/// The corpus layout, used ONLY when blessing. The committed bytes are
/// authoritative; this function documents their provenance.
fn generate_corpus() -> Vec<u8> {
    let mut v = Vec::new();
    splitmix_fill(&mut v, 512 * 1024, 0xC0A7);
    // text-like section
    let mut i = 0u64;
    let text_start = v.len();
    while v.len() - text_start < 256 * 1024 {
        v.extend_from_slice(
            format!(
                "[fixture] line {i} of structured ascii content, id={:08x}\n",
                i * 2654435761
            )
            .as_bytes(),
        );
        i += 1;
    }
    v.extend_from_slice(&vec![0u8; 96 * 1024]); // long zero run
    splitmix_fill(&mut v, 300 * 1024, 0xB1);
    v.extend_from_slice(&vec![0xAAu8; 40 * 1024]); // constant run
    splitmix_fill(&mut v, 200 * 1024, 0xB2);
    splitmix_fill(&mut v, 12_345, 0xB3); // awkward tail
    v
}

fn expected_lines(data: &[u8], min: usize, avg: usize, max: usize) -> Vec<String> {
    Xg16::new(data, min, avg, max)
        .map(|c| format!("{} {} {:04x}", c.offset, c.length, c.hash))
        .collect()
}

#[test]
fn fixtures_match_all_kernels() {
    let corpus =
        std::fs::read(corpus_path()).expect("tests/data/corpus.bin missing — run the bless test");
    for &(name, min, avg, max) in PARAM_SETS {
        let expected = std::fs::read_to_string(cuts_path(name))
            .unwrap_or_else(|_| panic!("cuts_{name}.txt missing — run the bless test"));
        let expected: Vec<&str> = expected.lines().collect();

        // Public iterator (dispatched kernel).
        let got = expected_lines(&corpus, min, avg, max);
        assert_eq!(
            got, expected,
            "public iterator diverged from fixture ({name})"
        );

        // Every kernel on this machine, explicitly.
        let config = Config::new(min, avg, max);
        for (kname, k) in scan::kernels() {
            let mut got = Vec::new();
            let mut off = 0;
            while off < corpus.len() {
                let (len, hash) = config.cut_with(&corpus[off..], k);
                got.push(format!("{off} {len} {hash:04x}"));
                off += len;
            }
            assert_eq!(
                got, expected,
                "kernel {kname} diverged from fixture ({name})"
            );
        }
    }
}

/// Regenerates the corpus and fixtures. Only runs when explicitly asked;
/// requires `XG16_BLESS=1` as a second guard.
#[test]
#[ignore = "regenerates fixtures; run only for intentional format changes"]
fn bless() {
    assert_eq!(
        std::env::var("XG16_BLESS").as_deref(),
        Ok("1"),
        "set XG16_BLESS=1 to confirm fixture regeneration"
    );
    std::fs::create_dir_all(data_dir()).unwrap();
    let corpus = if corpus_path().exists() {
        std::fs::read(corpus_path()).unwrap()
    } else {
        let c = generate_corpus();
        std::fs::write(corpus_path(), &c).unwrap();
        c
    };
    for &(name, min, avg, max) in PARAM_SETS {
        let lines = expected_lines(&corpus, min, avg, max);
        std::fs::write(cuts_path(name), lines.join("\n") + "\n").unwrap();
        println!("blessed {name}: {} chunks", lines.len());
    }
}
