//! Host-side (HS-mode) call wrappers for the private RDSM SBI extension.
//!
//! The ABI constants and shared structures live in the `rdsm-abi` crate
//! (the single source of truth shared with the RDSM firmware layer); this
//! module only provides the `ecall` convenience wrappers used by the TSM.

use core::arch::asm;

use rdsm_abi::{
    EID_RDSM, FID_RDSM_GET_INFO, FID_RDSM_MFENCE_PA, FID_RDSM_MPT_SET, FID_RDSM_TEERET,
};

pub use rdsm_abi::{NORMAL_RETURN, RdsmPlatformInfo, TSM_READY, TVM_EXIT};

/// Query RDSM for the platform layout. Returns `None` when the extension
/// fails or reports a nonsensical memory map (callers must then fail
/// closed and reject all host-supplied addresses).
pub fn rdsm_get_platform_info() -> Option<RdsmPlatformInfo> {
    let mut info = RdsmPlatformInfo {
        ram_start: 0,
        ram_end: 0,
        tsm_region_start: 0,
        tsm_region_end: 0,
        mpt_pool_start: 0,
        mpt_pool_end: 0,
    };
    let buf = &mut info as *mut RdsmPlatformInfo as usize;
    let mut err: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") EID_RDSM,
            in("a6") FID_RDSM_GET_INFO,
            in("a2") buf,
            lateout("a0") err,
            lateout("a1") _,
        );
    }
    if err != 0 || info.ram_end <= info.ram_start {
        return None;
    }
    Some(info)
}

pub fn rdsm_teeret(reason: usize, a1: usize, a2: usize) -> ! {
    unsafe {
        asm!(
            "ecall",
            in("a7") EID_RDSM,
            in("a6") FID_RDSM_TEERET,
            in("a0") reason,
            in("a1") a1,
            in("a2") a2,
            options(noreturn)
        );
    }
}

pub fn rdsm_mpt_set(target_sdid: usize, paddr: usize, len: usize, perm: u8) -> (usize, usize) {
    let mut err: usize;
    let mut val: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") EID_RDSM,
            in("a6") FID_RDSM_MPT_SET,
            inout("a0") target_sdid => err,
            inout("a1") paddr => val,
            in("a2") len,
            in("a3") perm as usize,
        );
    }
    (err, val)
}

pub fn rdsm_mfence_pa(paddr: usize, sdid: usize) -> (usize, usize) {
    let mut err: usize;
    let mut val: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") EID_RDSM,
            in("a6") FID_RDSM_MFENCE_PA,
            inout("a0") paddr => err,
            inout("a1") sdid => val,
        );
    }
    (err, val)
}
