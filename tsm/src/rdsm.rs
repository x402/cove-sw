use core::arch::asm;

pub const EID_RDSM: usize = 0x5244534D; // "RDSM"
pub const FID_RDSM_GET_INFO: usize = 0;
pub const FID_RDSM_MPT_SET: usize = 1;
pub const FID_RDSM_MFENCE_PA: usize = 2;
pub const FID_RDSM_TEERET: usize = 3;

/// Platform memory layout returned by RDSM_GET_INFO (a2 = buffer).
/// Host-allocatable whitelist: [tsm_region_end, mpt_pool_start) ∪
/// [mpt_pool_end, ram_end); everything else is reserved.
/// Must stay layout-compatible with the RDSM-side definition in
/// `rustsbi/prototyper/prototyper/src/sbi/rdsm.rs`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RdsmPlatformInfo {
    pub ram_start: usize,
    pub ram_end: usize,
    pub tsm_region_start: usize,
    pub tsm_region_end: usize,
    pub mpt_pool_start: usize,
    pub mpt_pool_end: usize,
}

const _: () = assert!(core::mem::size_of::<RdsmPlatformInfo>() == 48);

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


pub const NORMAL_RETURN: usize = 0;
pub const TVM_EXIT: usize = 1;
pub const TSM_READY: usize = 2;

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
