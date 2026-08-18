//! # xg16 — vectorizable content-defined chunking with byte-granular cuts
//!
//! A rolling-hash chunker in the FastCDC tradition (boundary check at
//! every byte position, so cuts land on the byte grid and streams resync
//! within ~a chunk after arbitrary-length edits), redesigned so the
//! per-byte scan vectorizes.
//!
//! ## The format
//!
//! Per input byte `b`:
//!
//! ```text
//! h = ((h << 1) ^ T[b]) & 0xFFFF          // 16-bit xor-gear
//! boundary at this byte  ⟺  h & mask == 0  // mask = top bits of 16
//! ```
//!
//! with `T[b] = A[b & 15] ^ B[b >> 4]` — two 16-entry u16 tables
//! (split tabulation). Chunk sizing is FastCDC-style normalized: no cuts
//! before `min_size` (bytes there are never hashed), a hard mask (+2
//! bits) up to `avg_size`, an easy mask (−2 bits) to `max_size`, forced
//! cut at `max_size`/EOF.
//!
//! Three deliberate deviations from classic gear, each required for SIMD:
//!
//! - **XOR, not ADD**: no carries, so state bits never interact — a
//!   truncated state is exact, and lanes stay narrow.
//! - **16-bit state**: 16 positions per AVX2 register as u16 lanes. The
//!   hash window becomes ~16 bytes (a contribution shifts out after 16
//!   steps). Caps configurations at `avg_size ≤ 16 KiB`.
//! - **Nibble-split table**: lookups become `vpshufb` (16-entry shuffles);
//!   256 effective entries derive from 32 random words.
//!
//! Because the state after a full 16-byte block equals that block's own
//! prefix (`h << 16 = 0` in u16), the vector kernel has **no loop-carried
//! dependency** — the stream path is one branch per 32 bytes.
//!
//! Measured on a Zen 2 core: ~5.0 GiB/s chunking (2.5× scalar
//! FastCDC), resync after odd-length inserts ~6.4 KB (better than
//! classic FastCDC — the short window re-locks faster).
//!
//! FORMAT-CRITICAL: the table seed, 16-bit width, and update rule define
//! cut positions forever. Optimized kernels must match [`scan::scan_ref`]
//! cut-for-cut (enforced by tests).

//! ## Quick start
//!
//! The interface follows `fastcdc-rs`: [`Xg16`] iterates over
//! [`Chunk`]s of an in-memory slice, [`StreamChunker`] wraps any
//! [`std::io::Read`] and yields owned [`ChunkData`].
//!
//! ```
//! use xg16::Xg16;
//!
//! let data: Vec<u8> = (0..100_000u32).flat_map(|i| i.to_le_bytes()).collect();
//! for chunk in Xg16::new(&data, 2048, 8192, 65536) {
//!     let bytes = &data[chunk.offset..chunk.offset + chunk.length];
//!     assert!(!bytes.is_empty());
//! }
//! ```

mod chunker;
mod stream;
mod table;

pub mod scan;

pub use chunker::{Chunk, Xg16};
pub use stream::{ChunkData, Error, StreamChunker};

#[doc(hidden)]
pub use chunker::Config;
