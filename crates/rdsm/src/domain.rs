//! Supervisor Domain management — SDID allocation and domain activation.
//!
//! Provides:
//!
//! - **[`SdidAllocator`]** — A bitmap-based allocator for up to 64 supervisor
//!   domain IDs (SDIDMAX = 6 → 2⁶ = 64).
//! - **[`SupervisorDomain`]** — Metadata struct holding SDID, SIDN, and
//!   domain state.
//! - **[`activate_domain`]** — Programs the `mmpt` CSR and issues
//!   `MFENCE.PA` to activate a domain (RISC-V only; stub on other targets).

use crate::mpt::MptTree;

// ── SdidAllocator ──────────────────────────────────────────────────────

/// A bitmap-based SDID allocator supporting up to 64 supervisor domain IDs.
///
/// SDIDMAX = 6 (per smmtt), yielding a maximum of 2⁶ = 64 domains. The
/// allocator tracks which SDIDs are in use with a 64-bit bitmap.
///
/// # Example
///
/// ```
/// use rdsm::domain::SdidAllocator;
///
/// let mut alloc = SdidAllocator::new();
/// assert_eq!(alloc.alloc(), Some(0));
/// assert_eq!(alloc.alloc(), Some(1));
/// alloc.free(0);
/// assert_eq!(alloc.alloc(), Some(0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SdidAllocator {
    bitmap: u64,
}

impl SdidAllocator {
    /// Create a new empty allocator (all SDIDs free).
    pub const fn new() -> Self {
        Self { bitmap: 0 }
    }

    /// Allocate the first available SDID.
    ///
    /// Returns `Some(sdid)` on success, or `None` if all 64 SDIDs are
    /// exhausted.
    pub fn alloc(&mut self) -> Option<usize> {
        if self.bitmap == !0u64 {
            return None;
        }
        let idx = (!self.bitmap).trailing_zeros() as usize;
        self.bitmap |= 1u64 << idx;
        Some(idx)
    }

    /// Free a previously allocated SDID.
    ///
    /// Silently ignores SDIDs ≥ 64 (out of range).
    pub fn free(&mut self, sdid: usize) {
        if sdid < 64 {
            self.bitmap &= !(1u64 << sdid);
        }
    }
}

impl Default for SdidAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ── DomainState ────────────────────────────────────────────────────────

/// Operational state of a supervisor domain.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DomainState {
    /// Domain is initialised but not yet active.
    Inactive,
    /// Domain is active (mmpt programmed for this SDID).
    Active,
    /// Domain has been torn down.
    Destroyed,
}

// ── SupervisorDomain ───────────────────────────────────────────────────

/// Metadata for a supervisor domain.
///
/// The associated MPT tree is managed separately and passed to
/// [`activate_domain`] directly.
#[derive(Copy, Clone, Debug)]
pub struct SupervisorDomain {
    /// Supervisor Domain ID (0..63).
    pub sdid: usize,
    /// Supervisor Interrupt Domain Number.
    pub sidn: usize,
    /// Current domain state.
    pub state: DomainState,
}

// ── activate_domain ────────────────────────────────────────────────────

/// Activate a supervisor domain by programming the `mmpt` CSR.
///
/// Constructs the `mmpt` value from the tree's mode, the given SDID, and the
/// tree's root PPN, writes it to the CSR, and issues `MFENCE.PA` to
/// synchronise the MPT cache.
///
/// On non-RISC-V targets this function is a no-op stub.
#[cfg(target_arch = "riscv64")]
pub fn activate_domain(sdid: usize, tree: &MptTree) {
    use crate::csr::Mmpt;
    use crate::fence::mfence_pa;

    let mmpt = Mmpt::from_parts(tree.mode(), sdid, tree.root_ppn());
    mmpt.write();
    mfence_pa(0, sdid);
}

/// Stub for non-RISC-V targets.
#[cfg(not(target_arch = "riscv64"))]
pub fn activate_domain(_sdid: usize, _tree: &MptTree) {}

// ── Unit tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_first_returns_zero() {
        let mut alloc = SdidAllocator::new();
        assert_eq!(alloc.alloc(), Some(0));
    }

    #[test]
    fn alloc_sequential() {
        let mut alloc = SdidAllocator::new();
        assert_eq!(alloc.alloc(), Some(0));
        assert_eq!(alloc.alloc(), Some(1));
        assert_eq!(alloc.alloc(), Some(2));
    }

    #[test]
    fn free_and_realloc() {
        let mut alloc = SdidAllocator::new();
        assert_eq!(alloc.alloc(), Some(0));
        assert_eq!(alloc.alloc(), Some(1));
        alloc.free(0);
        assert_eq!(alloc.alloc(), Some(0));
        // SDID 1 is still allocated.
        alloc.free(1);
    }

    #[test]
    fn exhaustion() {
        let mut alloc = SdidAllocator::new();
        for i in 0..64 {
            assert_eq!(alloc.alloc(), Some(i), "allocation {} should succeed", i);
        }
        // 65th allocation must fail.
        assert_eq!(alloc.alloc(), None);
    }

    #[test]
    fn free_invalid_sdid() {
        let mut alloc = SdidAllocator::new();
        // These should not panic.
        alloc.free(64);
        alloc.free(100);
        alloc.free(usize::MAX);
        // Allocator should still function normally.
        assert_eq!(alloc.alloc(), Some(0));
    }
}
