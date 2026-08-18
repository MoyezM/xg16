//! Slice chunking: configuration, the [`Xg16`] iterator, and [`Chunk`].

use crate::scan::{Scan, scan};

/// A chunk boundary decision for in-memory data.
///
/// `hash` is the rolling-hash state at the cut position (zero-extended
/// from 16 bits). Like the gear hash in FastCDC implementations it is a
/// property of the bytes near the boundary, NOT a fingerprint of the
/// chunk's content — use a real content hash (BLAKE3, SHA-256, ...) for
/// dedup identity. For chunks that end without a boundary condition
/// (forced cuts at `max_size`, end of input) it is the state where
/// scanning stopped, and 0 if nothing was hashed (inputs no longer than
/// `min_size`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunk {
    /// Rolling-hash state at the cut (see type docs for caveats).
    pub hash: u64,
    /// Byte offset of the chunk from the start of the source.
    pub offset: usize,
    /// Length of the chunk in bytes.
    pub length: usize,
}

/// Validated sizing configuration and mask derivation. Public only for
/// the crate's own tests and benchmarks.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub min_size: usize,
    pub avg_size: usize,
    pub max_size: usize,
    mask_hard: u64,
    mask_easy: u64,
}

/// Top-`bits` of the 16-bit state.
const fn top16(bits: u32) -> u64 {
    (0xFFFFu64 >> (16 - bits)) << (16 - bits)
}

impl Config {
    /// Panics unless sizes are powers of two, `min < avg < max`, and
    /// `avg_size <= 16 KiB` (the 16-bit state supports at most a 16-bit
    /// mask and the hard mask uses `log2(avg) + 2` bits).
    pub fn new(min_size: usize, avg_size: usize, max_size: usize) -> Self {
        assert!(
            min_size.is_power_of_two() && avg_size.is_power_of_two() && max_size.is_power_of_two(),
            "chunk sizes must be powers of two"
        );
        assert!(
            min_size < avg_size && avg_size < max_size,
            "need min < avg < max"
        );
        let bits = avg_size.trailing_zeros();
        assert!(
            bits + 2 <= 16,
            "avg_size too large for the 16-bit state (max 16 KiB)"
        );
        Config {
            min_size,
            avg_size,
            max_size,
            mask_hard: top16(bits + 2),
            mask_easy: top16(bits - 2),
        }
    }

    /// Length of the next chunk starting at `data[0]` and the state at
    /// the cut, using the supplied kernel (tests pin kernels with this).
    #[inline]
    pub fn cut_with(&self, data: &[u8], scan: impl Fn(&[u8], u64, u64) -> Scan) -> (usize, u64) {
        let n = data.len();
        if n <= self.min_size {
            return (n, 0);
        }
        let normal_end = self.avg_size.min(n);
        let (found, h) = scan(&data[self.min_size..normal_end], 0, self.mask_hard);
        if let Some(len) = found {
            return (self.min_size + len, h);
        }
        let end = self.max_size.min(n);
        let (found, h2) = scan(&data[normal_end..end], h, self.mask_easy);
        if let Some(len) = found {
            return (normal_end + len, h2);
        }
        (end, h2)
    }

    /// Length of the next chunk and state at the cut, best kernel.
    #[inline]
    pub fn cut(&self, data: &[u8]) -> (usize, u64) {
        self.cut_with(data, scan)
    }
}

/// Content-defined chunker over an in-memory byte slice; iterates over
/// [`Chunk`]s. The interface follows `fastcdc-rs`.
///
/// ```
/// use xg16::Xg16;
///
/// let data: Vec<u8> = (0..100_000u32).flat_map(|i| i.to_le_bytes()).collect();
/// let mut total = 0;
/// for chunk in Xg16::new(&data, 2048, 8192, 65536) {
///     assert_eq!(chunk.offset, total);
///     total += chunk.length;
/// }
/// assert_eq!(total, data.len());
/// ```
///
/// The final chunk may be shorter than `min_size` (it ends at end of
/// input, not at a boundary).
pub struct Xg16<'a> {
    config: Config,
    source: &'a [u8],
    offset: usize,
}

impl<'a> Xg16<'a> {
    /// Create a chunker over `source`.
    ///
    /// # Panics
    ///
    /// If sizes are not powers of two with `min < avg < max`, or
    /// `avg_size > 16 KiB`.
    pub fn new(source: &'a [u8], min_size: usize, avg_size: usize, max_size: usize) -> Self {
        Xg16 {
            config: Config::new(min_size, avg_size, max_size),
            source,
            offset: 0,
        }
    }
}

impl<'a> Iterator for Xg16<'a> {
    type Item = Chunk;

    fn next(&mut self) -> Option<Chunk> {
        let rest = &self.source[self.offset..];
        if rest.is_empty() {
            return None;
        }
        let (length, hash) = self.config.cut(rest);
        let offset = self.offset;
        self.offset += length;
        Some(Chunk {
            hash,
            offset,
            length,
        })
    }
}
