//! Interrupt domain management — SIDN allocation, MSDEI decoding.
//!
//! Provides:
//! - [`SidnAllocator`]: bitmap allocator for up to 64 Supervisor Interrupt
//!   Domain Numbers (SIDNs).
//! - [`switch_interrupt_domain`]: write the active SIDN into `msdcfg.SIDN`.
//! - [`MsdeiTrap`]: decode a Machine Supervisor Domain External Interrupt
//!   (MSDEI) trap from `mcause` and enumerate pending supervisor domains.
//! - [`PendingSidIter`]: iterator over pending SID bits in a mask.
#![cfg_attr(test, allow(unused_extern_crates))]
#[cfg(test)]
extern crate std;

use core::fmt;

// ── SidnAllocator (task 6.1) ─────────────────────────────────────────────

/// Allocator for Supervisor Interrupt Domain Numbers (SIDNs).
///
/// Uses a 64-bit bitmap; bit *i* = 1 means SIDN *i* is allocated.
/// Supports a maximum of 64 SIDNs (0–63).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SidnAllocator {
    bitmap: u64,
}

impl Default for SidnAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl SidnAllocator {
    /// Create a new allocator with all SIDNs free.
    pub const fn new() -> Self {
        Self { bitmap: 0 }
    }

    /// Allocate the lowest available SIDN.
    ///
    /// Returns `Some(index)` on success, or `None` if all 64 SIDNs are
    /// already allocated.
    pub fn alloc(&mut self) -> Option<usize> {
        let idx = (!self.bitmap).trailing_zeros() as usize;
        if idx == 64 {
            None
        } else {
            self.bitmap |= 1 << idx;
            Some(idx)
        }
    }

    /// Free a previously allocated SIDN.
    ///
    /// # Panics
    ///
    /// Does nothing if `sidn >= 64`.
    pub fn free(&mut self, sidn: usize) {
        if sidn < 64 {
            self.bitmap &= !(1 << sidn);
        }
    }

    /// Returns `true` if the given SIDN is currently allocated.
    pub fn is_allocated(&self, sidn: usize) -> bool {
        if sidn < 64 {
            (self.bitmap >> sidn) & 1 == 1
        } else {
            false
        }
    }

    /// Returns the number of allocated SIDNs.
    pub fn allocated_count(&self) -> usize {
        self.bitmap.count_ones() as usize
    }
}

// ── switch_interrupt_domain (task 6.2) ───────────────────────────────────

/// Switch the active interrupt domain by writing `sidn` into `msdcfg.SIDN`.
///
/// On RISC-V this reads the `msdcfg` CSR, sets the SIDN field (bits 5:0),
/// and writes it back. On other targets this is a no-op.
#[cfg(target_arch = "riscv64")]
pub fn switch_interrupt_domain(sidn: usize) {
    use crate::csr::Msdcfg;
    let mut cfg = Msdcfg::read();
    cfg.set_sidn(sidn);
    cfg.write();
}

/// Host-safe stub (no-op).
#[cfg(not(target_arch = "riscv64"))]
pub fn switch_interrupt_domain(_sidn: usize) {}

// ── MsdeiTrap (task 6.3) ─────────────────────────────────────────────────

/// Decoded Machine Supervisor Domain External Interrupt (MSDEI) trap.
///
/// MSDEI is the interrupt class signaled via `mip` bit 14 (interrupt code
/// 14 in `mcause`).  `MsdeiTrap` records which supervisor domains have
/// pending *and* enabled interrupts by AND-ing `msideip` (pending) with
/// `msideie` (enable).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MsdeiTrap {
    pending_mask: usize,
}

impl MsdeiTrap {
    /// MSDEI interrupt code in `mcause` (corresponds to `mip` bit 14).
    pub const MSDEI_CODE: usize = 14;

    /// Decode `mcause` to check if this is an MSDEI interrupt.
    ///
    /// Returns `Some(MsdeiTrap)` if `mcause` indicates an MSDEI interrupt
    /// (interrupt bit 63 = 1, code = 14), with the pending mask already
    /// loaded from `msideip & msideie`.  Returns `None` otherwise.
    pub fn from_trap(mcause: usize) -> Option<Self> {
        let is_interrupt = (mcause >> 63) & 1 == 1;
        let code = mcause & 0x7FFF_FFFF_FFFF_FFFF;
        if is_interrupt && code == Self::MSDEI_CODE {
            let pending = Self::read_pending_mask();
            Some(Self {
                pending_mask: pending,
            })
        } else {
            None
        }
    }

    /// Decode the MSDEI state as of *now*, straight from
    /// `msideip & msideie`, without an `mcause` to check first.  Intended
    /// for handlers already entered because of an MSDEI (the cause was
    /// reported by the trap machinery itself).
    pub fn current() -> Self {
        Self {
            pending_mask: Self::read_pending_mask(),
        }
    }

    /// Read the pending SID mask: `msideip & msideie`.
    #[cfg(target_arch = "riscv64")]
    fn read_pending_mask() -> usize {
        crate::csr::read_msideip() & crate::csr::read_msideie()
    }

    #[cfg(not(target_arch = "riscv64"))]
    fn read_pending_mask() -> usize {
        0
    }

    /// Returns the raw pending mask (bit *i* = SID *i* has a pending
    /// and enabled external interrupt).
    pub fn pending_mask(&self) -> usize {
        self.pending_mask
    }

    /// Returns an iterator over the pending SID numbers whose bits are
    /// set in the pending mask.
    pub fn pending_sids(&self) -> PendingSidIter {
        PendingSidIter {
            mask: self.pending_mask,
        }
    }
}

// ── PendingSidIter ───────────────────────────────────────────────────────

/// Iterator over pending SID numbers from an MSDEI trap.
///
/// Yields the index of each set bit in the pending mask, from least
/// significant to most significant.
#[derive(Copy, Clone)]
pub struct PendingSidIter {
    mask: usize,
}

impl Iterator for PendingSidIter {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.mask == 0 {
            return None;
        }
        let sid = self.mask.trailing_zeros() as usize;
        self.mask &= !(1 << sid);
        Some(sid)
    }
}

impl fmt::Debug for PendingSidIter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PendingSidIter")
            .field("mask", &format_args!("{:#b}", self.mask))
            .finish()
    }
}

// ── Unit tests (task 6.4) ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    // ── SidnAllocator tests ──────────────────────────────────────────

    #[test]
    fn sidn_alloc_first_returns_zero() {
        let mut alloc = SidnAllocator::new();
        assert_eq!(alloc.alloc(), Some(0));
    }

    #[test]
    fn sidn_alloc_sequential() {
        let mut alloc = SidnAllocator::new();
        for i in 0..5 {
            assert_eq!(alloc.alloc(), Some(i));
        }
    }

    #[test]
    fn sidn_free_and_realloc() {
        let mut alloc = SidnAllocator::new();
        assert_eq!(alloc.alloc(), Some(0));
        assert_eq!(alloc.alloc(), Some(1));
        alloc.free(0);
        assert_eq!(alloc.alloc(), Some(0)); // reused
        assert_eq!(alloc.alloc(), Some(2));
    }

    #[test]
    fn sidn_exhaustion() {
        let mut alloc = SidnAllocator::new();
        for i in 0..64 {
            assert_eq!(alloc.alloc(), Some(i));
        }
        assert_eq!(alloc.alloc(), None);
    }

    #[test]
    fn sidn_free_invalid() {
        let mut alloc = SidnAllocator::new();
        // Freeing out-of-range SIDN should be a no-op (no panic).
        alloc.free(64);
        alloc.free(usize::MAX);
        // Allocator should still work normally.
        assert_eq!(alloc.alloc(), Some(0));
    }

    #[test]
    fn sidn_is_allocated() {
        let mut alloc = SidnAllocator::new();
        assert!(!alloc.is_allocated(0));
        alloc.alloc();
        assert!(alloc.is_allocated(0));
        alloc.free(0);
        assert!(!alloc.is_allocated(0));
    }

    #[test]
    fn sidn_allocated_count() {
        let mut alloc = SidnAllocator::new();
        assert_eq!(alloc.allocated_count(), 0);
        alloc.alloc();
        assert_eq!(alloc.allocated_count(), 1);
        alloc.alloc();
        assert_eq!(alloc.allocated_count(), 2);
        alloc.free(0);
        assert_eq!(alloc.allocated_count(), 1);
    }

    // ── MsdeiTrap tests ─────────────────────────────────────────────

    #[test]
    fn msdei_from_trap_valid() {
        // mcause = (1 << 63) | 14  →  interrupt with code 14 (MSDEI)
        let mcause = (1usize << 63) | 14;
        let trap = MsdeiTrap::from_trap(mcause);
        assert!(trap.is_some());
        // On the host read_pending_mask returns 0, so pending_mask == 0.
        assert_eq!(trap.unwrap().pending_mask(), 0);
    }

    #[test]
    fn msdei_from_trap_invalid_code() {
        // code 15 is not MSDEI
        let mcause = (1usize << 63) | 15;
        assert!(MsdeiTrap::from_trap(mcause).is_none());
    }

    #[test]
    fn msdei_from_trap_exception() {
        // mcause = 14 (exception, not interrupt → bit 63 = 0)
        let mcause = 14;
        assert!(MsdeiTrap::from_trap(mcause).is_none());
    }

    #[test]
    fn msdei_current_host_reads_zero() {
        // On the host the pending mask is a stub returning 0; `current()`
        // must never touch `mcause` (it has none to read).
        assert_eq!(MsdeiTrap::current().pending_mask(), 0);
    }

    #[test]
    fn pending_sids_iterator() {
        let trap = MsdeiTrap {
            pending_mask: 0b1010,
        };
        let sids: std::vec::Vec<usize> = trap.pending_sids().collect();
        assert_eq!(sids, std::vec![1, 3]);
    }

    #[test]
    fn pending_sids_empty() {
        let trap = MsdeiTrap { pending_mask: 0 };
        assert_eq!(trap.pending_sids().count(), 0);
    }

    #[test]
    fn pending_sids_all_bits() {
        let trap = MsdeiTrap {
            pending_mask: 0xFFFF_FFFF_FFFF_FFFF,
        };
        let sids: std::vec::Vec<usize> = trap.pending_sids().collect();
        assert_eq!(sids.len(), 64);
        // Verify first few and last few
        assert_eq!(sids[0], 0);
        assert_eq!(sids[1], 1);
        assert_eq!(sids[63], 63);
    }
}
