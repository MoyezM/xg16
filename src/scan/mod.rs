//! Boundary-search kernels. `scan_ref` is the format definition; every
//! other kernel must match it cut-for-cut.
//!
//! ## Structure
//!
//! All block-based kernels share one generic driver, [`drive`], which
//! owns the algorithm: the block loop and two-block unroll, carry
//! plumbing, the cold fire-resolution path, and scalar head/tail
//! handling. What varies per architecture is only the [`Kernel`] trait —
//! five primitives around "evaluate one 16-byte block". The portable
//! implementation ([`Portable`]) is plain safe Rust; `x86.rs` implements
//! the same trait with AVX2 intrinsics.
//!
//! ## Adding an architecture
//!
//! 1. Implement [`Kernel`] in a new `#[cfg(target_arch = "...")]` module
//!    (see `x86.rs`: a `block` computation plus four one-liner
//!    primitives), and expose a safe wrapper that runtime-checks CPU
//!    features before entering a `#[target_feature]` function that calls
//!    `drive::<YourKernel>`.
//! 2. Register the wrapper in [`scan`] (dispatch) and [`kernels`]
//!    (coverage). The equivalence tests and benchmarks iterate
//!    `kernels()`, so nothing else is needed.
//!
//! ## Why hand-written intrinsics
//!
//! [`Portable`] through the same driver is the best auto-vectorization
//! candidate we could construct. Measured on Zen 2 with
//! `-C target-cpu=native` (16 MiB random data): scan_ref ~2.5 GiB/s,
//! portable ~1.3 GiB/s, AVX2 ~5.0 GiB/s. LLVM vectorizes the arithmetic
//! but performs each table lookup as a scalar load plus a `vpinsrw` lane
//! insert; it cannot synthesize the `vpshufb` nibble lookup. The
//! portable kernel stays benchmarked as a sentinel in case a future
//! compiler learns the transform.
//!
//! Contract: scan `data` from carried state `h` (low 16 bits used); on
//! the first byte position `i` where the updated state ANDed with `mask`
//! is zero, return `(Some(i + 1), state)` — the cut falls after that
//! byte. Otherwise `(None, state)` with the state carried out.

use crate::table::T;

pub type Scan = (Option<usize>, u64);

/// A boundary-search kernel.
pub type ScanFn = fn(&[u8], u64, u64) -> Scan;

/// Reference kernel — the format definition.
#[inline]
pub fn scan_ref(data: &[u8], h: u64, mask: u64) -> Scan {
    let mut h = h as u16;
    let m = mask as u16;
    for (i, &b) in data.iter().enumerate() {
        h = (h << 1) ^ T[b as usize];
        if h & m == 0 {
            return (Some(i + 1), h as u64);
        }
    }
    (None, h as u64)
}

/// Best kernel available on this machine.
///
/// Dispatch is a plain branch, not a cached function pointer:
/// `is_x86_feature_detected!` already caches CPUID results in an atomic
/// (each call is a load + bit test), and it const-folds to `true` when
/// the feature is enabled at compile time — so under
/// `-C target-cpu=native` this reduces to a direct call. A `cfg`-only
/// dispatch would be wrong for generic builds: the architecture is known
/// at compile time, but the running CPU's features are not.
#[inline]
pub fn scan(data: &[u8], h: u64, mask: u64) -> Scan {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return x86::scan_avx2(data, h, mask);
    }
    scan_portable(data, h, mask)
}

/// Every kernel available on this machine, named — the reference first.
/// Tests equivalence-check and benches measure ALL entries, so a new
/// architecture kernel is covered by registering it here.
pub fn kernels() -> Vec<(&'static str, ScanFn)> {
    #[allow(unused_mut)]
    let mut ks: Vec<(&'static str, ScanFn)> = vec![("ref", scan_ref), ("portable", scan_portable)];
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        ks.push(("avx2", x86::scan_avx2));
    }
    ks
}

#[cfg(target_arch = "x86_64")]
mod x86;

/// One 16-byte block evaluated in a single step: the only part of the
/// block algorithm that differs per architecture.
///
/// A kernel's `block` receives the block bytes and the carried state
/// (in an arch-native broadcast form) and returns the 16 check values,
/// a fire indicator, and the next carry. The carry after a full block is
/// a pure function of the block's bytes (`h << 16 = 0` in u16), so
/// implementations have no loop-carried dependency to worry about.
///
/// All methods must be `#[inline(always)]`: the driver is monomorphized
/// inside a `#[target_feature]` wrapper, and full inlining is what lets
/// loop-invariant setup (table broadcasts, multiplier vectors) hoist out
/// of the block loop.
pub(super) trait Kernel {
    /// Carried state in arch-native form (e.g. a lane broadcast).
    type Carry: Copy;
    /// The 16 per-position check values, unextracted.
    type Lanes: Copy;
    /// Fire indicator for one block, in arch-native encoding.
    type Hits: Copy;

    fn init(h: u16) -> Self::Carry;
    fn carry_value(c: Self::Carry) -> u16;
    fn block(
        blk: &[u8; 16],
        carry: Self::Carry,
        mask: u16,
    ) -> (Self::Lanes, Self::Hits, Self::Carry);
    /// Did either of two blocks fire? (Lets the driver take one branch
    /// per 32 bytes; implementations can OR before reducing.)
    fn any(a: Self::Hits, b: Self::Hits) -> bool;
    /// Index of the first firing lane, if any.
    fn first(h: Self::Hits) -> Option<usize>;
    /// Extract one check value (cold path only).
    fn lane(l: Self::Lanes, i: usize) -> u16;
}

/// The shared algorithm: block loop with two-block unroll, carry
/// plumbing, cold-path resolution, scalar tail. Everything except the
/// per-block math lives here, once.
#[inline(always)]
pub(super) fn drive<K: Kernel>(data: &[u8], h0: u64, mask: u64) -> Scan {
    let m = mask as u16;

    #[inline(always)]
    fn resolve<K: Kernel>(i: usize, lanes: K::Lanes, hits: K::Hits) -> Option<(usize, u64)> {
        K::first(hits).map(|lane| (i + lane + 1, K::lane(lanes, lane) as u64))
    }

    let mut carry = K::init(h0 as u16);
    let mut i = 0usize;
    while i + 32 <= data.len() {
        let b1: &[u8; 16] = data[i..i + 16].try_into().unwrap();
        let b2: &[u8; 16] = data[i + 16..i + 32].try_into().unwrap();
        let (l1, f1, c1) = K::block(b1, carry, m);
        let (l2, f2, c2) = K::block(b2, c1, m);
        if K::any(f1, f2) {
            if let Some((pos, h)) = resolve::<K>(i, l1, f1) {
                return (Some(pos), h);
            }
            let (pos, h) = resolve::<K>(i + 16, l2, f2).expect("any() implies a fire");
            return (Some(pos), h);
        }
        carry = c2;
        i += 32;
    }
    if i + 16 <= data.len() {
        let b: &[u8; 16] = data[i..i + 16].try_into().unwrap();
        let (l, f, c) = K::block(b, carry, m);
        if let Some((pos, h)) = resolve::<K>(i, l, f) {
            return (Some(pos), h);
        }
        carry = c;
        i += 16;
    }
    let (r, hh) = scan_ref(&data[i..], K::carry_value(carry) as u64, mask);
    (r.map(|x| i + x), hh)
}

/// Portable block kernel: safe Rust, the same algorithm shape as the
/// SIMD kernels (nibble-free here — it uses the materialized table).
/// Doubles as the auto-vectorization sentinel; see the module docs.
pub(super) struct Portable;

impl Kernel for Portable {
    type Carry = u16;
    type Lanes = [u16; 16];
    type Hits = u16;

    #[inline(always)]
    fn init(h: u16) -> u16 {
        h
    }

    #[inline(always)]
    fn carry_value(c: u16) -> u16 {
        c
    }

    #[inline(always)]
    fn block(blk: &[u8; 16], carry: u16, mask: u16) -> ([u16; 16], u16, u16) {
        let mut g = [0u16; 16];
        for j in 0..16 {
            g[j] = T[blk[j] as usize];
        }
        // Parallel prefix: s[j] = XOR_{k<=j} g[k] << (j-k).
        let mut s = g;
        for step in [1usize, 2, 4, 8] {
            let mut t = [0u16; 16];
            for j in 0..16 {
                t[j] = s[j] ^ if j >= step { s[j - step] << step } else { 0 };
            }
            s = t;
        }
        let mut hv = [0u16; 16];
        let mut hits = 0u16;
        for j in 0..16 {
            // Widen before shifting: `h << 16` on u16 wraps the shift
            // amount; in u32 it correctly produces 0 after truncation.
            hv[j] = s[j] ^ (((carry as u32) << (j + 1)) as u16);
            hits |= u16::from(hv[j] & mask == 0) << j;
        }
        (hv, hits, hv[15])
    }

    #[inline(always)]
    fn any(a: u16, b: u16) -> bool {
        a | b != 0
    }

    #[inline(always)]
    fn first(h: u16) -> Option<usize> {
        if h == 0 {
            None
        } else {
            Some(h.trailing_zeros() as usize)
        }
    }

    #[inline(always)]
    fn lane(l: [u16; 16], i: usize) -> u16 {
        l[i]
    }
}

/// Portable kernel through the shared driver.
pub fn scan_portable(data: &[u8], h: u64, mask: u64) -> Scan {
    drive::<Portable>(data, h, mask)
}
