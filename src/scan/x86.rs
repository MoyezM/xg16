//! x86-64 AVX2 kernel: an implementation of [`Kernel`] — the per-block
//! math only. The loop, unroll, cold path, and tails live in the shared
//! driver.

use std::arch::x86_64::*;

use super::{Kernel, Scan, drive, scan_ref};
use crate::table::{A_HI, A_LO, B_HI, B_LO};

/// Per-lane multipliers 2^(j+1) for the carried-state term (lane 15's
/// 2^16 ≡ 0 mod 2^16 — exactly `h << 16`).
static MULT_H: [u16; 16] = {
    let mut m = [0u16; 16];
    let mut j = 0;
    while j < 16 {
        m[j] = ((1u32 << (j + 1)) & 0xFFFF) as u16;
        j += 1;
    }
    m
};

/// Stitch multipliers: high half receives the low half's lane-7 scan
/// value times 2^(j-7); low half zero.
static MULT_S: [u16; 16] = {
    let mut m = [0u16; 16];
    let mut j = 8;
    while j < 16 {
        m[j] = 1u16 << (j - 7);
        j += 1;
    }
    m
};

pub(super) struct Avx2;

impl Kernel for Avx2 {
    type Carry = __m256i;
    type Lanes = __m256i;
    type Hits = __m256i;

    #[inline(always)]
    fn init(h: u16) -> __m256i {
        // SAFETY: only reachable through the avx2-gated wrapper below.
        unsafe { _mm256_set1_epi16(h as i16) }
    }

    #[inline(always)]
    fn carry_value(c: __m256i) -> u16 {
        unsafe { _mm256_extract_epi16::<0>(c) as u16 }
    }

    #[inline(always)]
    fn block(blk: &[u8; 16], carry: __m256i, mask: u16) -> (__m256i, __m256i, __m256i) {
        unsafe {
            let a_lo =
                _mm256_broadcastsi128_si256(_mm_loadu_si128(A_LO.as_ptr() as *const __m128i));
            let a_hi =
                _mm256_broadcastsi128_si256(_mm_loadu_si128(A_HI.as_ptr() as *const __m128i));
            let b_lo =
                _mm256_broadcastsi128_si256(_mm_loadu_si128(B_LO.as_ptr() as *const __m128i));
            let b_hi =
                _mm256_broadcastsi128_si256(_mm_loadu_si128(B_HI.as_ptr() as *const __m128i));
            let nib_mask = _mm256_set1_epi16(0x000F);
            // 0x80 in the high byte of each u16 lane: vpshufb zeroes them.
            let idx_or = _mm256_set1_epi16(0x8000u16 as i16);
            let mult_h = _mm256_loadu_si256(MULT_H.as_ptr() as *const __m256i);
            let mult_s = _mm256_loadu_si256(MULT_S.as_ptr() as *const __m256i);
            // vpshufb pattern broadcasting u16 lane 7 (bytes 14,15).
            let bcast7 = _mm256_set1_epi16(0x0F0Eu16 as i16);

            let x = _mm_loadu_si128(blk.as_ptr() as *const __m128i);
            let v = _mm256_cvtepu8_epi16(x); // 16 u16 lanes = the bytes
            // Table lookup g_j = T[b_j] via nibble shufb.
            let lo = _mm256_and_si256(v, nib_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), nib_mask);
            let idx_l = _mm256_or_si256(lo, idx_or);
            let idx_h = _mm256_or_si256(hi, idx_or);
            let ga = _mm256_or_si256(
                _mm256_shuffle_epi8(a_lo, idx_l),
                _mm256_slli_epi16::<8>(_mm256_shuffle_epi8(a_hi, idx_l)),
            );
            let gb = _mm256_or_si256(
                _mm256_shuffle_epi8(b_lo, idx_h),
                _mm256_slli_epi16::<8>(_mm256_shuffle_epi8(b_hi, idx_h)),
            );
            let g = _mm256_xor_si256(ga, gb);
            // Shift-XOR scan per 128-lane: s_j = XOR_{k<=j} g_k << (j-k).
            let mut s = g;
            s = _mm256_xor_si256(s, _mm256_slli_epi16::<1>(_mm256_slli_si256::<2>(s)));
            s = _mm256_xor_si256(s, _mm256_slli_epi16::<2>(_mm256_slli_si256::<4>(s)));
            s = _mm256_xor_si256(s, _mm256_slli_epi16::<4>(_mm256_slli_si256::<8>(s)));
            // Stitch the low half's lane-7 prefix into the high half.
            let slo = _mm256_permute2x128_si256::<0x00>(s, s);
            let s7 = _mm256_shuffle_epi8(slo, bcast7);
            let stitch = _mm256_mullo_epi16(s7, mult_s);
            let pfull = _mm256_xor_si256(s, stitch); // full 16-byte prefix
            // Check values: prefix ^ (carry << (j+1)).
            let hterm = _mm256_mullo_epi16(carry, mult_h);
            let hv = _mm256_xor_si256(pfull, hterm);
            // Next carry = prefix lane 15, broadcast in-domain: the carry
            // depends only on this block's bytes (h << 16 = 0 in u16).
            let phi = _mm256_permute2x128_si256::<0x11>(pfull, pfull);
            let next = _mm256_shuffle_epi8(phi, bcast7);
            let maskv = _mm256_set1_epi16(mask as i16);
            let hits = _mm256_cmpeq_epi16(_mm256_and_si256(hv, maskv), _mm256_setzero_si256());
            (hv, hits, next)
        }
    }

    #[inline(always)]
    fn any(a: __m256i, b: __m256i) -> bool {
        unsafe { _mm256_movemask_epi8(_mm256_or_si256(a, b)) != 0 }
    }

    #[inline(always)]
    fn first(h: __m256i) -> Option<usize> {
        let mm = unsafe { _mm256_movemask_epi8(h) } as u32;
        if mm == 0 {
            None
        } else {
            Some((mm.trailing_zeros() / 2) as usize)
        }
    }

    #[inline(always)]
    fn lane(l: __m256i, i: usize) -> u16 {
        let mut hs = [0u16; 16];
        unsafe { _mm256_storeu_si256(hs.as_mut_ptr() as *mut __m256i, l) };
        hs[i]
    }
}

/// AVX2 kernel through the shared driver. Safe: falls back to the
/// reference until CPU support is confirmed at runtime.
pub(super) fn scan_avx2(data: &[u8], h: u64, mask: u64) -> Scan {
    if data.len() >= 64 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 presence checked.
        unsafe { scan_avx2_inner(data, h, mask) }
    } else {
        scan_ref(data, h, mask)
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scan_avx2_inner(data: &[u8], h: u64, mask: u64) -> Scan {
    drive::<Avx2>(data, h, mask)
}
