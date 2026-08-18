//! The format's tables: two 16-entry u16 nibble tables (split
//! tabulation), stored byte-planar so `vpshufb` can look them up.
//!
//! FORMAT-CRITICAL: the seed and derivation define the format.

const fn gen_tables() -> ([u8; 16], [u8; 16], [u8; 16], [u8; 16]) {
    let mut a_lo = [0u8; 16];
    let mut a_hi = [0u8; 16];
    let mut b_lo = [0u8; 16];
    let mut b_hi = [0u8; 16];
    let mut s: u64 = 0x7A31_9C55_D02B_66E1;
    let mut i = 0;
    while i < 16 {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        a_lo[i] = z as u8;
        a_hi[i] = (z >> 8) as u8;
        b_lo[i] = (z >> 16) as u8;
        b_hi[i] = (z >> 24) as u8;
        i += 1;
    }
    (a_lo, a_hi, b_lo, b_hi)
}

const TABLES: ([u8; 16], [u8; 16], [u8; 16], [u8; 16]) = gen_tables();

/// Low bytes of the low-nibble table entries.
pub(crate) const A_LO: [u8; 16] = TABLES.0;
/// High bytes of the low-nibble table entries.
pub(crate) const A_HI: [u8; 16] = TABLES.1;
/// Low bytes of the high-nibble table entries.
pub(crate) const B_LO: [u8; 16] = TABLES.2;
/// High bytes of the high-nibble table entries.
pub(crate) const B_HI: [u8; 16] = TABLES.3;

/// The materialized 256-entry table, `T[b] = A[b & 15] ^ B[b >> 4]`,
/// used by the scalar kernels.
const fn gen_t() -> [u16; 256] {
    let mut t = [0u16; 256];
    let mut b = 0;
    while b < 256 {
        let lo = b & 15;
        let hi = b >> 4;
        let a = A_LO[lo] as u16 | ((A_HI[lo] as u16) << 8);
        let bb = B_LO[hi] as u16 | ((B_HI[hi] as u16) << 8);
        t[b] = a ^ bb;
        b += 1;
    }
    t
}

pub(crate) static T: [u16; 256] = gen_t();
