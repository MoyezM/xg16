//! The chunker: FastCDC-style normalized sizing over the xg16 scan.

use crate::scan::{Scan, scan};

/// Content-defined chunker. Cuts are byte-granular.
#[derive(Debug, Clone, Copy)]
pub struct Chunker {
    min_size: usize,
    avg_size: usize,
    max_size: usize,
    mask_hard: u64,
    mask_easy: u64,
}

/// Top-`bits` of the 16-bit state.
const fn top16(bits: u32) -> u64 {
    (0xFFFFu64 >> (16 - bits)) << (16 - bits)
}

impl Chunker {
    /// Sizes must be powers of two with `min < avg < max`, and
    /// `avg_size ≤ 16 KiB` (the 16-bit state supports at most a 16-bit
    /// mask, and the hard mask uses `log2(avg) + 2` bits).
    pub fn new(min_size: usize, avg_size: usize, max_size: usize) -> Self {
        assert!(
            min_size.is_power_of_two() && avg_size.is_power_of_two() && max_size.is_power_of_two()
        );
        assert!(min_size < avg_size && avg_size < max_size);
        let bits = avg_size.trailing_zeros();
        assert!(bits + 2 <= 16, "avg_size too large for the 16-bit state");
        Chunker {
            min_size,
            avg_size,
            max_size,
            mask_hard: top16(bits + 2),
            mask_easy: top16(bits - 2),
        }
    }

    /// 2 KiB / 8 KiB / 64 KiB.
    pub fn with_default_sizes() -> Self {
        Chunker::new(2 * 1024, 8 * 1024, 64 * 1024)
    }

    pub fn min_size(&self) -> usize {
        self.min_size
    }

    pub fn avg_size(&self) -> usize {
        self.avg_size
    }

    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Length of the next chunk starting at `data[0]`, with the supplied
    /// scan kernel (used by tests to pin kernels to the reference).
    #[inline]
    pub fn next_cut_with(&self, data: &[u8], scan: impl Fn(&[u8], u64, u64) -> Scan) -> usize {
        let n = data.len();
        if n <= self.min_size {
            return n;
        }
        let normal_end = self.avg_size.min(n);
        let (found, h) = scan(&data[self.min_size..normal_end], 0, self.mask_hard);
        if let Some(len) = found {
            return self.min_size + len;
        }
        let end = self.max_size.min(n);
        let (found, _) = scan(&data[normal_end..end], h, self.mask_easy);
        if let Some(len) = found {
            return normal_end + len;
        }
        end
    }

    /// Length of the next chunk, using the best kernel for this machine.
    #[inline]
    pub fn next_cut(&self, data: &[u8]) -> usize {
        self.next_cut_with(data, scan)
    }

    /// Iterate over the chunks of `data`.
    pub fn chunks<'a>(&self, data: &'a [u8]) -> Chunks<'a> {
        Chunks {
            chunker: *self,
            rest: data,
        }
    }

    /// Absolute end offsets of every chunk.
    pub fn cut_points(&self, data: &[u8]) -> Vec<usize> {
        let mut cuts = Vec::new();
        let mut off = 0;
        for c in self.chunks(data) {
            off += c.len();
            cuts.push(off);
        }
        cuts
    }
}

/// Iterator over the chunks of a slice.
pub struct Chunks<'a> {
    chunker: Chunker,
    rest: &'a [u8],
}

impl<'a> Iterator for Chunks<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        if self.rest.is_empty() {
            return None;
        }
        let cut = self.chunker.next_cut(self.rest);
        let (chunk, rest) = self.rest.split_at(cut);
        self.rest = rest;
        Some(chunk)
    }
}
