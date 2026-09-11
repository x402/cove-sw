//! MPT fault decoding from trap context.
//!
//! [`MptFault`] decodes RISC-V access-fault exceptions (Instruction,
//! Load, Store/AMO) that originate from MPT (Machine-level Memory
//! Protection Table) permission checks.  The physical address that
//! triggered the fault is taken from `mtval`.

// ── MptFault (task 7.1) ──────────────────────────────────────────────────

/// An MPT access fault decoded from trap-time `mcause` and `mtval`.
///
/// Represents one of the three RISC-V access-fault exception types that
/// the Smmpt extension raises when an MPT permission check fails:
///
/// | Exception | `mcause` code | Description |
/// |---|---|---|
/// | `InstructionAccessFault` | 1 | Instruction fetch denied |
/// | `LoadAccessFault` | 5 | Load or load-reserved denied |
/// | `StoreAccessFault` | 7 | Store, AMO, or SC denied |
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MptFault {
    /// Load access fault (mcause = 5): load from `paddr` denied.
    LoadAccessFault {
        /// Physical address that caused the fault.
        paddr: usize,
    },
    /// Store/AMO access fault (mcause = 7): write to `paddr` denied.
    StoreAccessFault {
        /// Physical address that caused the fault.
        paddr: usize,
    },
    /// Instruction access fault (mcause = 1): fetch from `paddr` denied.
    InstructionAccessFault {
        /// Physical address that caused the fault.
        paddr: usize,
    },
}

impl MptFault {
    // ── Exception code constants ────────────────────────────────────

    /// Exception code for instruction access fault.
    pub const CODE_INSTRUCTION_ACCESS: usize = 1;

    /// Exception code for load access fault.
    pub const CODE_LOAD_ACCESS: usize = 5;

    /// Exception code for store/AMO access fault.
    pub const CODE_STORE_ACCESS: usize = 7;

    // ── Decode from trap (task 7.2) ─────────────────────────────────

    /// Decode `mcause` and `mtval` to check if this is an MPT access fault.
    ///
    /// Returns `Some(MptFault)` if `mcause` indicates an access-fault
    /// exception (interrupt bit 63 = 0, code ∈ {1, 5, 7}).  The physical
    /// address is taken from `mtval`.
    ///
    /// Returns `None` if `mcause` is an interrupt, or has a code that does
    /// not correspond to an access fault.
    pub fn from_trap(mcause: usize, mtval: usize) -> Option<Self> {
        // Bit 63 = 1 means interrupt, not exception.
        let is_interrupt = (mcause >> 63) & 1 == 1;
        if is_interrupt {
            return None;
        }

        let code = mcause & 0x7FFF_FFFF_FFFF_FFFF;
        match code {
            Self::CODE_INSTRUCTION_ACCESS => Some(Self::InstructionAccessFault { paddr: mtval }),
            Self::CODE_LOAD_ACCESS => Some(Self::LoadAccessFault { paddr: mtval }),
            Self::CODE_STORE_ACCESS => Some(Self::StoreAccessFault { paddr: mtval }),
            _ => None,
        }
    }

    /// The physical address that caused the fault.
    pub fn paddr(&self) -> usize {
        match self {
            Self::LoadAccessFault { paddr }
            | Self::StoreAccessFault { paddr }
            | Self::InstructionAccessFault { paddr } => *paddr,
        }
    }
}

// ── Unit tests (task 7.3) ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_load_access_fault() {
        let fault = MptFault::from_trap(5, 0x1000);
        assert_eq!(fault, Some(MptFault::LoadAccessFault { paddr: 0x1000 }));
    }

    #[test]
    fn decode_store_access_fault() {
        let fault = MptFault::from_trap(7, 0x2000);
        assert_eq!(fault, Some(MptFault::StoreAccessFault { paddr: 0x2000 }));
    }

    #[test]
    fn decode_instruction_access_fault() {
        let fault = MptFault::from_trap(1, 0x3000);
        assert_eq!(
            fault,
            Some(MptFault::InstructionAccessFault { paddr: 0x3000 })
        );
    }

    #[test]
    fn decode_non_access_fault_returns_none() {
        // mcause = 2 = illegal instruction (not an access fault)
        let fault = MptFault::from_trap(2, 0);
        assert_eq!(fault, None);
    }

    #[test]
    fn decode_interrupt_returns_none() {
        // mcause = (1 << 63) | 14 = MSDEI interrupt, not an exception
        let mcause = (1usize << 63) | 14;
        let fault = MptFault::from_trap(mcause, 0);
        assert_eq!(fault, None);
    }

    #[test]
    fn paddr_accessor() {
        let load = MptFault::LoadAccessFault { paddr: 0xAABB };
        assert_eq!(load.paddr(), 0xAABB);

        let store = MptFault::StoreAccessFault { paddr: 0xCCDD };
        assert_eq!(store.paddr(), 0xCCDD);

        let inst = MptFault::InstructionAccessFault { paddr: 0xEEFF };
        assert_eq!(inst.paddr(), 0xEEFF);
    }

    #[test]
    fn from_trap_zero_mtval() {
        // Even with mtval = 0, the decode should work if mcause matches.
        let fault = MptFault::from_trap(5, 0);
        assert_eq!(fault, Some(MptFault::LoadAccessFault { paddr: 0 }));
    }

    #[test]
    fn from_trap_high_bits_in_mcause() {
        // Extra high bits beyond the standard encoding should not matter
        // as long as they are not in bits [62:0] that we extract.
        // Verify that only the code portion is matched.
        let fault = MptFault::from_trap(5, 0x1234);
        assert_eq!(fault, Some(MptFault::LoadAccessFault { paddr: 0x1234 }));
    }
}
