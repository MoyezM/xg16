//! Chunking throughput: reference vs AVX2 kernel, random and text-like
//! data. Single core, 16 MiB buffers, default 2K/8K/64K sizing.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use xg16::{Chunker, scan};

const BUF_LEN: usize = 16 << 20;

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

fn text_like(len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len + 64);
    let mut i = 0u64;
    while v.len() < len {
        v.extend_from_slice(
            format!(
                "[2026-08-18T01:02:{:02}] request id={i} path=/api/v1/items status=200\n",
                i % 60
            )
            .as_bytes(),
        );
        i += 1;
    }
    v.truncate(len);
    v
}

fn bench(cr: &mut Criterion) {
    let mut random = vec![0u8; BUF_LEN];
    fill_random(&mut random, 0xC0FFEE);
    let text = text_like(BUF_LEN);
    let c = Chunker::with_default_sizes();

    let kernels = scan::kernels();

    for (data_name, data) in [("random", &random), ("text", &text)] {
        let mut g = cr.benchmark_group(format!("chunk_{data_name}"));
        g.throughput(Throughput::Bytes(BUF_LEN as u64));
        g.sample_size(20);
        for (name, scan) in &kernels {
            g.bench_with_input(BenchmarkId::from_parameter(name), data, |b, data| {
                b.iter(|| {
                    let mut off = 0;
                    while off < data.len() {
                        off += c.next_cut_with(&data[off..], scan);
                    }
                    black_box(off)
                })
            });
        }
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
