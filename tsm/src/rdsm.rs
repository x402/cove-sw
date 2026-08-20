use core::arch::asm;

pub const EID_RDSM: usize = 0x5244534D; // "RDSM"
pub const FID_RDSM_GET_INFO: usize = 0;
pub const FID_RDSM_MPT_SET: usize = 1;
pub const FID_RDSM_MFENCE_PA: usize = 2;
pub const FID_RDSM_TEERET: usize = 3;

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
