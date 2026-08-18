# xg16

Content-defined chunking with byte-granular cuts and a vectorized scan.

xg16 is a chunker in the FastCDC family: it evaluates a rolling hash at
every byte position and cuts where the hash satisfies a mask condition,
so chunk boundaries are content-derived and realign within roughly one
chunk after an insert or delete of any length. Unlike gear-based
chunkers, the hash is designed so that sixteen positions can be
evaluated per vector instruction batch, which makes the scan about
twice as fast as a scalar gear loop on AVX2 hardware.

## Design

The rolling state is 16 bits, updated per byte:

    h = ((h << 1) ^ T[b]) & 0xFFFF
    cut after this byte  iff  h & mask == 0

where `T[b] = A[b & 15] ^ B[b >> 4]` for two fixed 16-entry tables.
Sizing follows FastCDC's normalized chunking: no cuts before `min_size`
(those bytes are not hashed), a mask two bits stricter than the average
requires up to `avg_size`, two bits looser from there to `max_size`, and
a forced cut at `max_size` or end of input.

Three properties make the scan vectorizable, and each is a deliberate
trade against classic gear:

- XOR instead of ADD. Without carries, state bits never influence
  higher bits, so a truncated state is exact and lanes stay narrow.
- A 16-bit state. Sixteen positions fit in one AVX2 register as u16
  lanes. The cost is a ~16-byte hash window (gear's is ~64), and a cap
  of `avg_size <= 16 KiB`.
- The nibble-split table. `vpshufb` performs sixteen 16-entry lookups
  in one instruction; a 256-entry table would need gathers.

The 16-bit width also removes the loop-carried dependency: the state
after a full 16-byte block equals that block's own prefix (`h << 16` is
zero), so consecutive blocks are independent work and the stream path
takes one branch per 32 bytes.

## Why the intrinsics kernel exists

`scan::scan_portable` is the same block algorithm written as safe,
data-parallel array loops through the same shared driver as the SIMD
kernel — the most vectorizer-friendly formulation we could construct. Compiled with `-C target-cpu=native` on Zen 2 it runs
at 1.3 GiB/s against the plain scalar loop's 2.5 and the intrinsics
kernel's 5.0. Inspection of the generated code shows why: LLVM
vectorizes the arithmetic but performs each table lookup as a scalar
load plus a `vpinsrw` lane insertion, and never synthesizes the
`vpshufb` lookup. The portable kernel remains in the benchmark suite so
a future compiler that learns the transform will show up in the numbers.

## Performance

Single Zen 2 core, 2K/8K/64K configuration, 16 MiB buffers:

| kernel | random | text-like |
|---|---|---|
| scalar reference | 2.4 GiB/s | 2.2 GiB/s |
| AVX2 | 4.7 GiB/s | 4.6 GiB/s |

Dedup retention against a 32 MiB base (`cargo run --release --example
dedup_ratio`):

| edit | new data stored |
|---|---|
| 3-byte insert | 11.9 KB |
| 1 KiB insert | 12.4 KB |
| 20 scattered inserts under 64 B | 198 KB |
| 100 B delete | 17.8 KB |
| 4 KiB aligned overwrite | 23.0 KB |
| 1 MiB append | 1.05 MB |

Each edit costs one to two chunks of re-stored data, which is the
behavior per-byte CDC exists to provide. Measured resync distance after
an odd-length insert is about 6.4 KB.

## Usage

```rust
use xg16::{Chunker, StreamChunker};

let chunker = Chunker::new(2 * 1024, 8 * 1024, 64 * 1024);

// Slice input:
for chunk in chunker.chunks(&data) {
    // hash/store chunk
}

// Streaming input (bounded memory, identical cuts):
let mut s = StreamChunker::new(chunker);
s.push(&piece, |chunk| { /* ... */ });
s.finish(|chunk| { /* ... */ });
```

The public API is `Chunker`, `Chunks`, and `StreamChunker`. The `scan`
module exposes the kernels for tests and benchmarks; `scan::scan_ref`
is the format definition, and the test suite pins every other kernel to
it cut-for-cut.

### CLI

`cargo install --path .` installs an `xg16` binary:

    xg16 <files/dirs...>          per-file stats: throughput, chunk-size
                                  histogram, forced cuts, dedup within
                                  and across files
    xg16 --compare <old> <new>    bytes a store would need for the new
                                  version given the old one
    xg16 --min 4k --avg 16k ...   custom sizing

## Format stability

Cut positions are an on-disk format for anything built on top of this
crate. The table seed, the 16-bit width, the update rule, and the mask
construction must not change once data has been chunked. Sizes are
powers of two with `min < avg < max` and `avg_size <= 16 KiB`.

## Development

    cargo test                                   format and kernel tests
    cargo bench                                  throughput (ref / portable / avx2)
    cargo run --release --example dedup_ratio    retained-dedup table
    nix develop                                  perf, hyperfine, valgrind

The tests cover kernel equivalence on awkward-length inputs, chunk-size
distribution on random and structured data, degenerate inputs (empty,
sub-minimum, constant bytes), resync distance for insert lengths from 1
to 4096 bytes, and stream/slice invariance under adversarial push
sizes.
