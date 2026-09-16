//! Private ABI definitions shared between the RDSM firmware layer (M-mode)
//! and its HS-mode callers (TSM and host) in the CoVE software stack.
//!
//! RDSM and TSM form one system spanning two privilege levels; the SBI
//! extension and payload format defined here are their private contract.
//! Both sides must include this crate instead of re-declaring the layouts.
//!
//! The crate is `no_std` and has no external dependencies.

#![no_std]

// ── Private SBI extension constants for RDSM ───────────────────────────

/// Extension ID for private RDSM SBI extension: 0x5244534D ("RDSM").
pub const EID_RDSM: usize = 0x5244534D;

/// Function ID for RDSM_GET_INFO: 0.
pub const FID_RDSM_GET_INFO: usize = 0;

/// Function ID for RDSM_MPT_SET: 1.
pub const FID_RDSM_MPT_SET: usize = 1;

/// Function ID for RDSM_MFENCE_PA: 2.
pub const FID_RDSM_MFENCE_PA: usize = 2;

/// Function ID for RDSM_TEERET: 3.
pub const FID_RDSM_TEERET: usize = 3;

// ── TEERET Reason constants ────────────────────────────────────────────

/// Parameter a0 value for NORMAL_RETURN: 0.
pub const NORMAL_RETURN: usize = 0;

/// Parameter a0 value for TVM_EXIT: 1.
pub const TVM_EXIT: usize = 1;

/// Parameter a0 value for TSM_READY: 2.
pub const TSM_READY: usize = 2;

// ── Standard CoVE extension IDs (SUPD / COVH / COVI) are deliberately
// NOT defined here: their single source is the upstream `riscv-cove`
// crate, re-exported through rdsm-policy. This crate only owns the private
// RDSM↔TSM contract. ────────────────────────────────────────────────────

// ── CoVE payload image format ──────────────────────────────────────────

/// Magic number for CoVE payload header: "COVE" (0x434F5645).
pub const COVE_PAYLOAD_MAGIC: u32 = 0x434F5645;

/// Current version of CoVE payload header.
pub const COVE_PAYLOAD_VERSION: u32 = 1;

/// CoVE Payload Header.
///
/// This is the leading 72 bytes of the payload image; packers may pad the
/// first page out to a full 4 KiB after this header.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PayloadHeader {
    /// Magic number (0x434F5645 "COVE").
    pub magic: u32,
    /// Header version (1).
    pub version: u32,
    /// TSM binary offset from payload base.
    pub tsm_offset: u64,
    /// TSM binary size in bytes.
    pub tsm_size: u64,
    /// TSM physical load address.
    pub tsm_load_paddr: u64,
    /// TSM physical entry point address.
    pub tsm_entry_paddr: u64,
    /// Host binary offset from payload base.
    pub host_offset: u64,
    /// Host binary size in bytes.
    pub host_size: u64,
    /// Host physical load address.
    pub host_load_paddr: u64,
    /// Host physical entry point address.
    pub host_entry_paddr: u64,
}

impl PayloadHeader {
    /// Validate the magic and version of this header.
    pub const fn is_valid(&self) -> bool {
        self.magic == COVE_PAYLOAD_MAGIC && self.version == COVE_PAYLOAD_VERSION
    }
}

// ── RDSM_GET_INFO platform layout ──────────────────────────────────────

/// Platform memory layout handed to the TSM through `RDSM_GET_INFO`
/// (a2 = confidential-domain buffer).
///
/// The host-allocatable whitelist derivable from this structure is:
/// `[tsm_region_end, mpt_pool_start)` ∪ `[mpt_pool_end, ram_end)` —
/// everything else (firmware gap, TSM image, MPT page pool) is reserved.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RdsmPlatformInfo {
    pub ram_start: usize,
    pub ram_end: usize,
    /// `[tsm_region_start, tsm_region_end)` = the TSM image region.
    /// `[ram_start, tsm_region_start)` is firmware / reserved gap.
    pub tsm_region_start: usize,
    pub tsm_region_end: usize,
    pub mpt_pool_start: usize,
    pub mpt_pool_end: usize,
}

const _: () = assert!(core::mem::size_of::<RdsmPlatformInfo>() == 48);

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, offset_of, size_of};

    #[test]
    fn test_payload_header_layout() {
        assert_eq!(size_of::<PayloadHeader>(), 72);
        assert_eq!(align_of::<PayloadHeader>(), 8);

        assert_eq!(offset_of!(PayloadHeader, magic), 0);
        assert_eq!(offset_of!(PayloadHeader, version), 4);
        assert_eq!(offset_of!(PayloadHeader, tsm_offset), 8);
        assert_eq!(offset_of!(PayloadHeader, tsm_size), 16);
        assert_eq!(offset_of!(PayloadHeader, tsm_load_paddr), 24);
        assert_eq!(offset_of!(PayloadHeader, tsm_entry_paddr), 32);
        assert_eq!(offset_of!(PayloadHeader, host_offset), 40);
        assert_eq!(offset_of!(PayloadHeader, host_size), 48);
        assert_eq!(offset_of!(PayloadHeader, host_load_paddr), 56);
        assert_eq!(offset_of!(PayloadHeader, host_entry_paddr), 64);
    }

    #[test]
    fn test_payload_header_validation() {
        let mut header = PayloadHeader {
            magic: COVE_PAYLOAD_MAGIC,
            version: COVE_PAYLOAD_VERSION,
            tsm_offset: 0x1000,
            tsm_size: 0x400000,
            tsm_load_paddr: 0x80400000,
            tsm_entry_paddr: 0x80400000,
            host_offset: 0x401000,
            host_size: 0x800000,
            host_load_paddr: 0x80800000,
            host_entry_paddr: 0x80800000,
        };
        assert!(header.is_valid());

        header.magic = 0x12345678;
        assert!(!header.is_valid());

        header.magic = COVE_PAYLOAD_MAGIC;
        header.version = 2;
        assert!(!header.is_valid());
    }
}
