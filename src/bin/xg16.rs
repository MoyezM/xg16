//! xg16 CLI: chunk real files and report what the chunker did.
//!
//!   xg16 [--min N] [--avg N] [--max N] <path>...   stats per file + aggregate
//!   xg16 --compare <old> <new>                     dedup between two versions
//!
//! Sizes accept k/m suffixes (e.g. --avg 8k).

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Instant;

use xg16::Xg16;

const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
const RESET: &str = "\x1b[0m";

fn parse_size(s: &str) -> Option<usize> {
    let s = s.trim();
    let (num, mult) = match s.chars().last()? {
        'k' | 'K' => (&s[..s.len() - 1], 1024),
        'm' | 'M' => (&s[..s.len() - 1], 1024 * 1024),
        _ => (s, 1),
    };
    num.parse::<usize>().ok().map(|n| n * mult)
}

fn human(bytes: f64) -> String {
    if bytes >= 1e9 {
        format!("{:.2} GiB", bytes / (1u64 << 30) as f64)
    } else if bytes >= 1e6 {
        format!("{:.2} MiB", bytes / (1 << 20) as f64)
    } else if bytes >= 1e3 {
        format!("{:.1} KiB", bytes / 1024.0)
    } else {
        format!("{bytes:.0} B")
    }
}

fn collect_files(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_symlink() {
        return;
    }
    if path.is_file() {
        out.push(path.to_path_buf());
    } else if path.is_dir()
        && let Ok(rd) = std::fs::read_dir(path)
    {
        let mut entries: Vec<_> = rd.flatten().map(|e| e.path()).collect();
        entries.sort();
        for e in entries {
            collect_files(&e, out);
        }
    }
}

fn chunk_hash(c: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    c.hash(&mut h);
    h.finish()
}

struct FileStats {
    bytes: usize,
    lens: Vec<usize>,
    hashes: Vec<u64>,
    secs: f64,
}

fn chunk_file(sizes: (usize, usize, usize), data: &[u8]) -> FileStats {
    let (min, avg, max) = sizes;
    let t0 = Instant::now();
    let lens: Vec<usize> = Xg16::new(data, min, avg, max).map(|c| c.length).collect();
    let secs = t0.elapsed().as_secs_f64();
    let hashes: Vec<u64> = Xg16::new(data, min, avg, max)
        .map(|c| chunk_hash(&data[c.offset..c.offset + c.length]))
        .collect();
    FileStats {
        bytes: data.len(),
        lens,
        hashes,
        secs,
    }
}

fn bar(frac: f64, width: usize) -> String {
    let cells = frac * width as f64;
    let full = cells.floor() as usize;
    let partials = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉'];
    let rem = ((cells - full as f64) * 8.0) as usize;
    let mut s = "█".repeat(full);
    if full < width {
        s.push(partials[rem.min(7)]);
    }
    s
}

fn print_histogram(lens: &[usize], min: usize, max: usize) {
    // log2 buckets from min to max, plus "< min" (EOF tails) and "= max"
    // (forced cuts) called out separately.
    let mut buckets: Vec<(String, usize)> = Vec::new();
    buckets.push((format!("<{}", human(min as f64)), 0));
    let mut lo = min;
    while lo < max {
        let hi = (lo * 2).min(max);
        buckets.push((format!("{}–{}", human(lo as f64), human(hi as f64)), 0));
        lo = hi;
    }
    let mut forced = 0usize;
    for &l in lens {
        if l == max {
            forced += 1;
        }
        let idx = if l <= min {
            0
        } else {
            let mut idx = 1;
            let mut lo = min;
            while lo * 2 < l.min(max) {
                lo *= 2;
                idx += 1;
            }
            idx.min(buckets.len() - 1)
        };
        buckets[idx].1 += 1;
    }
    let total = lens.len().max(1);
    let peak = buckets.iter().map(|b| b.1).max().unwrap_or(1).max(1);
    for (label, n) in &buckets {
        if *n == 0 {
            continue;
        }
        let pct = 100.0 * *n as f64 / total as f64;
        println!(
            "    {label:>18}  {CYAN}{}{RESET} {pct:.1}%",
            bar(*n as f64 / peak as f64, 30)
        );
    }
    if forced > 0 {
        println!(
            "    {DIM}{forced} forced cut(s) at max size — low-entropy or unlucky regions{RESET}"
        );
    }
}

fn report_file(name: &str, st: &FileStats, sizes: (usize, usize, usize)) {
    let n = st.lens.len().max(1);
    let mean = st.bytes as f64 / n as f64;
    let lmin = *st.lens.iter().min().unwrap_or(&0);
    let lmax = *st.lens.iter().max().unwrap_or(&0);
    let gibs = st.bytes as f64 / st.secs / (1u64 << 30) as f64;

    println!(
        "\n{BOLD}{name}{RESET}  {DIM}({}){RESET}",
        human(st.bytes as f64)
    );
    println!(
        "  chunked in {:.1} ms  {GREEN}{gibs:.2} GiB/s{RESET}",
        st.secs * 1e3
    );
    println!(
        "  {} chunks   mean {}   min {}   max {}",
        st.lens.len(),
        human(mean),
        human(lmin as f64),
        human(lmax as f64),
    );
    print_histogram(&st.lens, sizes.0, sizes.2);

    // Self-dedup: repeated chunks within the file.
    let mut seen: HashMap<u64, usize> = HashMap::new();
    let mut dup_bytes = 0usize;
    for (h, l) in st.hashes.iter().zip(&st.lens) {
        if seen.insert(*h, *l).is_some() {
            dup_bytes += l;
        }
    }
    if dup_bytes > 0 {
        println!(
            "  self-dedup: {GREEN}{}{RESET} of repeated chunks ({:.2}% of the file)",
            human(dup_bytes as f64),
            100.0 * dup_bytes as f64 / st.bytes.max(1) as f64
        );
    }
}

fn compare(sizes: (usize, usize, usize), old_path: &Path, new_path: &Path) {
    let (min, avg, max) = sizes;
    let old = std::fs::read(old_path).expect("read old");
    let new = std::fs::read(new_path).expect("read new");
    let old_set: std::collections::HashSet<u64> = Xg16::new(&old, min, avg, max)
        .map(|c| chunk_hash(&old[c.offset..c.offset + c.length]))
        .collect();
    let mut shared = 0usize;
    let mut fresh = 0usize;
    let mut fresh_chunks = 0usize;
    let mut total_chunks = 0usize;
    for c in Xg16::new(&new, min, avg, max) {
        total_chunks += 1;
        if old_set.contains(&chunk_hash(&new[c.offset..c.offset + c.length])) {
            shared += c.length;
        } else {
            fresh += c.length;
            fresh_chunks += 1;
        }
    }
    println!(
        "\n{BOLD}compare{RESET}  {} ({})  →  {} ({})",
        old_path.display(),
        human(old.len() as f64),
        new_path.display(),
        human(new.len() as f64),
    );
    let pct = 100.0 * shared as f64 / new.len().max(1) as f64;
    println!(
        "  new version: {total_chunks} chunks, {fresh_chunks} new — store {GREEN}{}{RESET}, dedup {GREEN}{pct:.2}%{RESET}",
        human(fresh as f64)
    );
    println!(
        "  {DIM}a naive full copy would store {}; chunking saved {}{RESET}",
        human(new.len() as f64),
        human(shared as f64)
    );
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut min = 2 * 1024;
    let mut avg = 8 * 1024;
    let mut max = 64 * 1024;
    let mut do_compare = false;
    let mut paths: Vec<PathBuf> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--min" | "--avg" | "--max" => {
                let v = parse_size(args.get(i + 1).expect("missing size")).expect("bad size");
                match args[i].as_str() {
                    "--min" => min = v,
                    "--avg" => avg = v,
                    _ => max = v,
                }
                args.drain(i..=i + 1);
            }
            "--compare" => {
                do_compare = true;
                args.remove(i);
            }
            "-h" | "--help" => {
                println!("usage: xg16 [--min N] [--avg N] [--max N] [--compare] <path>...");
                return;
            }
            _ => {
                paths.push(PathBuf::from(&args[i]));
                i += 1;
            }
        }
    }
    if paths.is_empty() {
        eprintln!("usage: xg16 [--min N] [--avg N] [--max N] [--compare] <path>...");
        std::process::exit(2);
    }
    let sizes = (min, avg, max);
    let _ = Xg16::new(&[], min, avg, max); // validate sizes up front
    println!(
        "{DIM}xg16 · min {} / avg {} / max {}{RESET}",
        human(min as f64),
        human(avg as f64),
        human(max as f64)
    );

    if do_compare {
        assert!(paths.len() == 2, "--compare needs exactly two files");
        compare(sizes, &paths[0], &paths[1]);
        return;
    }

    let mut files = Vec::new();
    for p in &paths {
        // Follow symlinks the user named explicitly; skip them in walks.
        let p = p.canonicalize().unwrap_or_else(|_| p.clone());
        collect_files(&p, &mut files);
    }
    if files.is_empty() {
        eprintln!("no files found");
        std::process::exit(1);
    }

    let mut corpus: HashMap<u64, usize> = HashMap::new();
    let mut total_bytes = 0usize;
    let mut total_secs = 0f64;
    let mut total_chunks = 0usize;
    let mut stored_bytes = 0usize;

    for f in &files {
        let data = match std::fs::read(f) {
            Ok(d) if !d.is_empty() => d,
            _ => continue,
        };
        let st = chunk_file(sizes, &data);
        report_file(&f.display().to_string(), &st, sizes);
        total_bytes += st.bytes;
        total_secs += st.secs;
        total_chunks += st.lens.len();
        for (h, l) in st.hashes.iter().zip(&st.lens) {
            if corpus.insert(*h, *l).is_none() {
                stored_bytes += l;
            }
        }
    }

    if files.len() > 1 {
        let deduped = total_bytes - stored_bytes;
        println!(
            "\n{BOLD}corpus total{RESET}  {} in {} files",
            human(total_bytes as f64),
            files.len()
        );
        println!(
            "  {total_chunks} chunks, {} unique — store {GREEN}{}{RESET}, cross-file dedup {GREEN}{:.2}%{RESET}",
            corpus.len(),
            human(stored_bytes as f64),
            100.0 * deduped as f64 / total_bytes.max(1) as f64
        );
        println!(
            "  {DIM}aggregate chunking throughput {:.2} GiB/s{RESET}",
            total_bytes as f64 / total_secs / (1u64 << 30) as f64
        );
    }
}
