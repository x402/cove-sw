#![no_std]
#![no_main]

use core::arch::{asm, naked_asm};
use core::fmt::{self, Write};
use core::panic::PanicInfo;

static GUEST_BIN: &[u8] =
    include_bytes!("../../target/riscv64gc-unknown-none-elf/release/test-guest.bin");

struct SbiConsole;

impl Write for SbiConsole {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for b in s.bytes() {
            let _ = sbi_rt::console_write_byte(b);
        }
        Ok(())
    }
}

pub fn print_str(s: &str) {
    let _ = SbiConsole.write_str(s);
}

macro_rules! print {
    ($($arg:tt)*) => {
        let _ = core::fmt::write(&mut SbiConsole, format_args!($($arg)*));
    };
}

macro_rules! println {
    () => {
        print!("
");
    };
    ($($arg:tt)*) => {
        let _ = core::fmt::write(&mut SbiConsole, format_args!($($arg)*));
        print!("
");
    };
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.entry")]
pub unsafe extern "C" fn _start() -> ! {
    naked_asm!(
        // Set stack pointer
        "la sp, boot_stack_top",
        // Clear BSS
        "la t0, sbss",
        "la t1, ebss",
        "1:",
        "bgeu t0, t1, 2f",
        "sd zero, 0(t0)",
        "addi t0, t0, 8",
        "j 1b",
        "2:",
        // Call host_main(hart_id, fdt_paddr)
        "call host_main",
        // Should not return:
        "3:",
        "wfi",
        "j 3b"
    )
}

pub fn sbi_probe_extension(extension_id: usize) -> usize {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") 0x10, // Base EID
            in("a6") 3,    // Probe extension FID
            inout("a0") extension_id => error,
            lateout("a1") value,
        );
    }
    if error == 0 { value } else { 0 }
}

pub fn sbi_supd_get_active_domains() -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::supd::EID_SUPD,
            in("a6") riscv_cove::supd::GET_ACTIVE_DOMAINS,
            lateout("a0") error,
            lateout("a1") value,
        );
    }
    (error, value)
}

pub fn sbi_covh_get_tsm_info(info: &mut riscv_cove::host::TsmInfo) -> (usize, usize) {
    let paddr = info as *mut _ as usize;
    let len = core::mem::size_of::<riscv_cove::host::TsmInfo>();
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::GET_TSM_INFO,
            inout("a0") paddr => error,
            inout("a1") len => value,
        );
    }
    (error, value)
}

pub fn sbi_covh_convert_pages(base_paddr: usize, num_pages: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::CONVERT_PAGES,
            inout("a0") base_paddr => error,
            inout("a1") num_pages => value,
        );
    }
    (error, value)
}

pub fn sbi_covh_reclaim_pages(base_paddr: usize, num_pages: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::RECLAIM_PAGES,
            inout("a0") base_paddr => error,
            inout("a1") num_pages => value,
        );
    }
    (error, value)
}

pub fn sbi_covh_global_fence() -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::GLOBAL_FENCE,
            lateout("a0") error,
            lateout("a1") value,
        );
    }
    (error, value)
}

pub fn sbi_covh_local_fence() -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::LOCAL_FENCE,
            lateout("a0") error,
            lateout("a1") value,
        );
    }
    (error, value)
}

pub fn sbi_covh_create_tvm(params: &riscv_cove::host::TvmCreateParams) -> (usize, usize) {
    let paddr = params as *const _ as usize;
    let len = core::mem::size_of::<riscv_cove::host::TvmCreateParams>();
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::CREATE_TVM,
            inout("a0") paddr => error,
            inout("a1") len => value,
        );
    }
    (error, value)
}

pub fn sbi_covh_add_tvm_memory_region(tvm_id: usize, gpa: usize, len: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::ADD_TVM_MEMORY_REGION,
            inout("a0") tvm_id => error,
            inout("a1") gpa => value,
            in("a2") len,
        );
    }
    (error, value)
}

pub fn sbi_covh_add_tvm_page_table_pages(
    tvm_id: usize,
    base_paddr: usize,
    num_pages: usize,
) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::ADD_TVM_PAGE_TABLE_PAGES,
            inout("a0") tvm_id => error,
            inout("a1") base_paddr => value,
            in("a2") num_pages,
        );
    }
    (error, value)
}

pub fn sbi_covh_add_tvm_measured_pages(
    tvm_id: usize,
    src_paddr: usize,
    dst_paddr: usize,
    page_type: usize,
    num_pages: usize,
    gpa: usize,
) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::ADD_TVM_MEASURED_PAGES,
            inout("a0") tvm_id => error,
            inout("a1") src_paddr => value,
            in("a2") dst_paddr,
            in("a3") page_type,
            in("a4") num_pages,
            in("a5") gpa,
        );
    }
    (error, value)
}

pub fn sbi_covh_create_tvm_vcpu(
    tvm_id: usize,
    vcpu_id: usize,
    state_paddr: usize,
) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::CREATE_TVM_VCPU,
            inout("a0") tvm_id => error,
            inout("a1") vcpu_id => value,
            in("a2") state_paddr,
        );
    }
    (error, value)
}

pub fn sbi_covh_finalize_tvm(
    tvm_id: usize,
    entry_sepc: usize,
    entry_arg: usize,
    identity_addr: usize,
) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::FINALIZE_TVM,
            inout("a0") tvm_id => error,
            inout("a1") entry_sepc => value,
            in("a2") entry_arg,
            in("a3") identity_addr,
        );
    }
    (error, value)
}

pub fn sbi_covh_run_tvm_vcpu(tvm_id: usize, vcpu_id: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::RUN_TVM_VCPU,
            inout("a0") tvm_id => error,
            inout("a1") vcpu_id => value,
        );
    }
    (error, value)
}

pub fn sbi_covh_add_tvm_zero_pages(
    tvm_id: usize,
    dst_paddr: usize,
    page_type: usize,
    num_pages: usize,
    gpa: usize,
) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::ADD_TVM_ZERO_PAGES,
            inout("a0") tvm_id => error,
            inout("a1") dst_paddr => value,
            in("a2") page_type,
            in("a3") num_pages,
            in("a4") gpa,
        );
    }
    (error, value)
}

pub fn sbi_covh_add_tvm_shared_pages(
    tvm_id: usize,
    src_paddr: usize,
    dst_paddr: usize,
    num_pages: usize,
    gpa: usize,
) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::ADD_TVM_SHARED_PAGES,
            inout("a0") tvm_id => error,
            inout("a1") src_paddr => value,
            in("a2") dst_paddr,
            in("a3") num_pages,
            in("a4") gpa,
        );
    }
    (error, value)
}

pub fn sbi_covh_tvm_fence(tvm_id: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::TVM_FENCE,
            inout("a0") tvm_id => error,
            lateout("a1") value,
        );
    }
    (error, value)
}

pub fn sbi_covh_tvm_invalidate_pages(tvm_id: usize, gpa: usize, length: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::TVM_INVALIDATE_PAGES,
            inout("a0") tvm_id => error,
            inout("a1") gpa => value,
            in("a2") length,
        );
    }
    (error, value)
}

pub fn sbi_covh_tvm_validate_pages(tvm_id: usize, gpa: usize, length: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::TVM_VALIDATE_PAGES,
            inout("a0") tvm_id => error,
            inout("a1") gpa => value,
            in("a2") length,
        );
    }
    (error, value)
}

pub fn sbi_covh_tvm_remove_pages(tvm_id: usize, gpa: usize, length: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::TVM_REMOVE_PAGES,
            inout("a0") tvm_id => error,
            inout("a1") gpa => value,
            in("a2") length,
        );
    }
    (error, value)
}

pub fn sbi_covh_destroy_tvm(tvm_id: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::DESTROY_TVM,
            inout("a0") tvm_id => error,
            lateout("a1") value,
        );
    }
    (error, value)
}

/// Raw RDSM private-extension MPT_SET, issued from the host domain.
/// Must be rejected with SBI_ERR_DENIED by the sender check.
pub fn sbi_rdsm_mpt_set_from_host(
    target_sdid: usize,
    paddr: usize,
    len: usize,
    perm: usize,
) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") rdsm_abi::EID_RDSM,
            in("a6") rdsm_abi::FID_RDSM_MPT_SET,
            inout("a0") target_sdid => error,
            inout("a1") paddr => value,
            in("a2") len,
            in("a3") perm,
        );
    }
    (error, value)
}

/// Raw RDSM private-extension SET_SIDN, issued from the host domain.
/// Must be rejected with SBI_ERR_DENIED by the sender check.
pub fn sbi_rdsm_set_sidn_from_host(target_sdid: usize, sidn: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") rdsm_abi::EID_RDSM,
            in("a6") rdsm_abi::FID_RDSM_SET_SIDN,
            inout("a0") target_sdid => error,
            inout("a1") sidn => value,
            in("a2") 0,
        );
    }
    (error, value)
}

/// Raw COVH get_tsm_info with an arbitrary (possibly hostile) buffer address.
pub fn sbi_covh_get_tsm_info_raw(paddr: usize, len: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::GET_TSM_INFO,
            inout("a0") paddr => error,
            inout("a1") len => value,
        );
    }
    (error, value)
}

/// Raw COVH ecall with an arbitrary (possibly hostile) function id.
pub fn sbi_covh_raw_fid(fid: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") fid,
            lateout("a0") error,
            lateout("a1") value,
        );
    }
    (error, value)
}

pub fn sbi_covi_convert_aia_imsic(imsic_page_addr: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::interrupt::EID_COVI,
            in("a6") riscv_cove::interrupt::CONVERT_AIA_IMSIC,
            inout("a0") imsic_page_addr => error,
            lateout("a1") value,
        );
    }
    (error, value)
}

pub fn sbi_covi_reclaim_tvm_aia_imsic(imsic_page_addr: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::interrupt::EID_COVI,
            in("a6") riscv_cove::interrupt::RECLAIM_TVM_AIA_IMSIC,
            inout("a0") imsic_page_addr => error,
            lateout("a1") value,
        );
    }
    (error, value)
}

/// COVI FID 0: `init_tvm_aia` with the params block at `params_paddr`.
pub fn sbi_covi_init_tvm_aia(
    tvm_id: usize,
    params_paddr: usize,
    params_len: usize,
) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::interrupt::EID_COVI,
            in("a6") riscv_cove::interrupt::INIT_TVM_AIA,
            in("a0") tvm_id,
            in("a1") params_paddr,
            in("a2") params_len,
            lateout("a0") error,
            lateout("a1") value,
        );
    }
    (error, value)
}

/// COVI FID 1: `set_tvm_aia_cpu_imsic_addr`.
pub fn sbi_covi_set_tvm_aia_cpu_imsic_addr(
    tvm_id: usize,
    vcpu_id: usize,
    gpa: usize,
) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::interrupt::EID_COVI,
            in("a6") riscv_cove::interrupt::SET_TVM_AIA_CPU_IMSIC_ADDR,
            in("a0") tvm_id,
            in("a1") vcpu_id,
            in("a2") gpa,
            lateout("a0") error,
            lateout("a1") value,
        );
    }
    (error, value)
}

/// COVI FID 4: `bind_aia_imsic`.
pub fn sbi_covi_bind_aia_imsic(
    tvm_id: usize,
    vcpu_id: usize,
    imsic_mask: usize,
) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::interrupt::EID_COVI,
            in("a6") riscv_cove::interrupt::BIND_AIA_IMSIC,
            in("a0") tvm_id,
            in("a1") vcpu_id,
            in("a2") imsic_mask,
            lateout("a0") error,
            lateout("a1") value,
        );
    }
    (error, value)
}

/// COVI FID 7: `inject_tvm_cpu`.
pub fn sbi_covi_inject_tvm_cpu(
    tvm_id: usize,
    vcpu_id: usize,
    interrupt_id: usize,
) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::interrupt::EID_COVI,
            in("a6") riscv_cove::interrupt::INJECT_TVM_CPU,
            in("a0") tvm_id,
            in("a1") vcpu_id,
            in("a2") interrupt_id,
            lateout("a0") error,
            lateout("a1") value,
        );
    }
    (error, value)
}

/// Raw COVI ecall with an arbitrary (possibly hostile) function id.
pub fn sbi_covi_raw_fid(fid: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::interrupt::EID_COVI,
            in("a6") fid,
            lateout("a0") error,
            lateout("a1") value,
        );
    }
    (error, value)
}

/// Raw COVH create_tvm with an arbitrary (possibly hostile) params address.
pub fn sbi_covh_create_tvm_at(params_paddr: usize, params_len: usize) -> (usize, usize) {
    let mut error: usize;
    let mut value: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") riscv_cove::host::EID_COVH,
            in("a6") riscv_cove::host::CREATE_TVM,
            inout("a0") params_paddr => error,
            inout("a1") params_len => value,
        );
    }
    (error, value)
}

const NUM_PAGES: usize = 26;

/// COVI FID 0 parameter block (CoVE §11.1), handed at a host-allocatable
/// physical address.
#[repr(align(16))]
struct AiaParamsBuffer([u8; 64]);

static mut AIA_PARAMS_BUF: AiaParamsBuffer = AiaParamsBuffer([0; 64]);

/// The confidential domain's hart0 guest interrupt file (SmmTT §6.1:
/// 0x3000_8000 + 1·2^14 + 0·2^13 + 1·2^12), converted and bound to the
/// e2e TVM's vCPU.
const CONF_GUEST_FILE: usize = 0x3000_D000;
/// Base of the e2e TVM's virtual IMSIC array and the vIMSIC GPA presented
/// to the guest (inside its 1 MiB memory region, on a page nothing else
/// maps — bind adds the G-stage device mapping). Under the AIA block used
/// below this address is hart 0 / guest 0. Deliberately not the guest's
/// demand-zero page (0x8002_0000): a device leaf there would swallow the
/// guest's demand-zero writes and turn PHASE4's DEMAND_ZERO_OK into an MMIO
/// pass-through readback.
const VCPU_IMSIC_GPA: usize = 0x8003_0000;
/// A second page-aligned GPA inside the TVM's memory region: hart 1 /
/// guest 0 of the virtual array based at [`VCPU_IMSIC_GPA`]. Used where the
/// guard under test is the vCPU id, the TVM state or the TVM id, so a
/// leaked success cannot be mistaken for a duplicate-GPA rejection (the
/// e2e TVM has only vCPU 0, which may re-record its own GPA).
const VCPU_IMSIC_GPA_ALT: usize = 0x8003_1000;
/// TVM and vCPU ids that no live TVM can hold: only vCPU 0 exists and the
/// e2e TVM is the first one created. Every per-TVM/-vCPU handler must reject
/// them.
const UNKNOWN_TVM_ID: usize = 0xDEAD;
const UNKNOWN_VCPU_ID: usize = 1;

#[repr(align(16384))]
struct AlignedPages([u8; NUM_PAGES * 4096]);

static mut HOST_PAGES: AlignedPages = AlignedPages([0; NUM_PAGES * 4096]);

#[repr(align(4096))]
struct GuestSrcBuffer([u8; 4 * 4096]);

static mut GUEST_SRC_BUFFER: GuestSrcBuffer = GuestSrcBuffer([0; 4 * 4096]);

#[unsafe(no_mangle)]
pub extern "C" fn host_main(_hart_id: usize, _fdt_paddr: usize) -> ! {
    println!("[HOST] Booting... HOST_STARTED");

    // 1. Check SUPD & COVH
    let supd_probed = sbi_probe_extension(riscv_cove::supd::EID_SUPD);
    assert_eq!(supd_probed, 1, "EXT_SUPD must be discovered via base probe");
    let covh_probed = sbi_probe_extension(riscv_cove::host::EID_COVH);
    assert_eq!(covh_probed, 1, "EXT_COVH must be discovered via base probe");
    println!("[MARKER 04] HOST: Discovery passed (EXT_SUPD & EXT_COVH found).");

    let (err, active_domains) = sbi_supd_get_active_domains();
    assert_eq!(err, 0, "SUPD get_active_domains must succeed");
    assert_eq!(
        active_domains & 0b11,
        0b11,
        "Active domains must contain 0b11"
    );

    let mut tsm_info = core::mem::MaybeUninit::<riscv_cove::host::TsmInfo>::zeroed();
    let tsm_info_ref = unsafe { &mut *tsm_info.as_mut_ptr() };
    let (err, _) = sbi_covh_get_tsm_info(tsm_info_ref);
    assert_eq!(err, 0, "get_tsm_info must succeed");
    let info = unsafe { tsm_info.assume_init() };
    assert_eq!(info.tsm_state, 2, "TSM state must be READY");
    println!("[MARKER 05] HOST: TSM capability probed, state=TSM_READY.");

    // 2. Prepare Guest Source Buffer in Host memory
    println!(
        "[HOST] Preparing guest payload ({} bytes)...",
        GUEST_BIN.len()
    );
    let src_ptr = core::ptr::addr_of_mut!(GUEST_SRC_BUFFER) as *mut u8;
    unsafe {
        core::ptr::write_bytes(src_ptr, 0, 4 * 4096);
        core::ptr::copy_nonoverlapping(GUEST_BIN.as_ptr(), src_ptr, GUEST_BIN.len());
    }
    let guest_src_paddr = src_ptr as usize;

    // 3. Memory layout of the 26 converted pages:
    // [0x0000..0x4000)  (4 pages): TVM Root Page Table (16KB aligned)
    // [0x4000..0x8000)  (4 pages): TVM State (16KB)
    // [0x8000..0xA000)  (2 pages): vCPU 0 State (8KB)
    // [0xA000..0xE000)  (4 pages): TVM Page Table Pool (16KB)
    // [0xE000..0x12000) (4 pages): TVM Measured Guest Payload (16KB)
    // [0x12000..0x14000)(2 pages): TVM Shared Memory Zero Pages (8KB)
    // Remaining 6 pages: free pool for demand-zero
    let base_paddr = core::ptr::addr_of_mut!(HOST_PAGES) as usize;
    assert_eq!(base_paddr % 16384, 0, "Base paddr must be 16KB aligned");

    let root_pt_paddr = base_paddr;
    let tvm_state_paddr = base_paddr + 0x4000;
    let vcpu_state_paddr = base_paddr + 0x8000;
    let pt_pool_paddr = base_paddr + 0xA000;
    let guest_dst_paddr = base_paddr + 0xE000;
    let shared_paddr = base_paddr + 0x12000;

    println!(
        "[HOST] Converting {} pages at 0x{:x}...",
        NUM_PAGES, base_paddr
    );
    let (err, _) = sbi_covh_convert_pages(base_paddr, NUM_PAGES);
    assert_eq!(err, 0, "convert_pages must succeed");

    let (err, _) = sbi_covh_global_fence();
    assert_eq!(err, 0, "global_fence must succeed");

    let (err, _) = sbi_covh_local_fence();
    assert_eq!(err, 0, "local_fence must succeed");
    println!(
        "[MARKER 06] HOST: Converted {} physical pages to confidential memory.",
        NUM_PAGES
    );

    // 4. Create TVM
    println!("[HOST] Creating TVM...");
    let params = riscv_cove::host::TvmCreateParams {
        tvm_page_directory_addr: root_pt_paddr,
        tvm_state_addr: tvm_state_paddr,
    };
    let (err, tvm_id) = sbi_covh_create_tvm(&params);
    assert_eq!(err, 0, "create_tvm must succeed");
    println!("[HOST] TVM created with ID {}", tvm_id);

    // 5. Add Memory Region (1MB at GPA 0x8000_0000)
    println!("[HOST] Adding memory region [0x80000000, 0x80100000)...");
    let (err, _) = sbi_covh_add_tvm_memory_region(tvm_id, 0x8000_0000, 0x10_0000);
    assert_eq!(err, 0, "add_tvm_memory_region must succeed");

    // 6. Add Page Table Pages (4 pages)
    println!(
        "[HOST] Adding 4 page-table pages at 0x{:x}...",
        pt_pool_paddr
    );
    let (err, _) = sbi_covh_add_tvm_page_table_pages(tvm_id, pt_pool_paddr, 4);
    assert_eq!(err, 0, "add_tvm_page_table_pages must succeed");

    // 7. Add Measured Pages (Guest binary, 4 pages at GPA 0x8000_0000)
    println!(
        "[HOST] Adding 4 measured pages (GPA 0x80000000 -> SPA 0x{:x})...",
        guest_dst_paddr
    );
    let (err, _) = sbi_covh_add_tvm_measured_pages(
        tvm_id,
        guest_src_paddr,
        guest_dst_paddr,
        0, // 4KB page type
        4,
        0x8000_0000,
    );
    assert_eq!(err, 0, "add_tvm_measured_pages must succeed");

    // 7b. Add Zero Pages for shared memory (2 pages at GPA 0x80010000)
    println!(
        "[HOST] Adding 2 zero pages (GPA 0x80010000 -> SPA 0x{:x})...",
        shared_paddr
    );
    let (err, _) = sbi_covh_add_tvm_zero_pages(tvm_id, shared_paddr, 0, 2, 0x8001_0000);
    assert_eq!(err, 0, "add_tvm_zero_pages must succeed");
    println!(
        "[MARKER 07] HOST: TVM #{} memory regions & measured pages populated.",
        tvm_id
    );

    // 8b. COVI: convert the confidential domain's guest interrupt file,
    // then arm the TVM's AIA configuration (FID 0) and record the vCPU's
    // vIMSIC GPA (FID 1). All before finalize (CoVE §11.1/§11.2).
    println!(
        "[HOST] Converting confidential guest file at 0x{:x}...",
        CONF_GUEST_FILE
    );
    let (err, _) = sbi_covi_convert_aia_imsic(CONF_GUEST_FILE);
    assert_eq!(err, 0, "guest-file convert must succeed");
    let (err, _) = sbi_covh_global_fence();
    assert_eq!(err, 0, "post-convert global fence must succeed");
    let (err, _) = sbi_covh_local_fence();
    assert_eq!(err, 0, "post-convert local fence must succeed");

    // 9. Create vCPU 0 before FID 0: the FID 0 window below needs a vCPU to
    // observe the re-initialisation rule and FID 1 before FID 0.
    println!("[HOST] Creating vCPU 0 at 0x{:x}...", vcpu_state_paddr);
    let (err, _) = sbi_covh_create_tvm_vcpu(tvm_id, 0, vcpu_state_paddr);
    assert_eq!(err, 0, "create_tvm_vcpu must succeed");

    // N24: FID 7 before FID 0. The AIA configuration is not armed yet, so an
    // unarmed TVM state must refuse the injection. Table 40 names no
    // uninitialized-AIA error and does not list not_supported (-2); the same
    // state is invalid_param for FID 1 and FID 4, so the project maps it to
    // invalid_param. This case pins the result but cannot by itself isolate
    // the AIA-enabled guard: with that guard deleted the TSM handler's
    // default-deny rule (identity 1 has not been allowed by the guest) would
    // answer invalid_param too.
    let (err, _) = sbi_covi_inject_tvm_cpu(tvm_id, 0, 1);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 7 before FID 0 must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N24 OK: FID 7 before FID 0 rejected");

    // ── COVI negative paths: FID 0 parameter validation ──
    // Every negative call below must be refused by the TSM's COVI dispatch
    // input validation and must leave state the positive path depends on
    // intact. The valid block is staged first so every variant differs from
    // it only in the field (or length/address) under test. Under this block
    // 0x8003_0000 is hart 0 / guest 0, 0x8003_1000 (VCPU_IMSIC_GPA_ALT) is
    // hart 1 / guest 0, and 0x8003_2000 falls outside the virtual array.
    let params = riscv_cove::interrupt::TvmAiaParams {
        imsic_base_addr: VCPU_IMSIC_GPA,
        group_index_bits: 0,
        group_index_shift: 24,
        hart_index_bits: 1,
        guest_index_bits: 0,
        guests_per_hart: 0,
    };
    let aia_params_len = core::mem::size_of::<riscv_cove::interrupt::TvmAiaParams>();
    let aia_params_ptr = core::ptr::addr_of_mut!(AIA_PARAMS_BUF) as *mut u8;
    let stage_aia_params = |block: &riscv_cove::interrupt::TvmAiaParams| unsafe {
        core::ptr::write_bytes(aia_params_ptr, 0, 64);
        core::ptr::copy_nonoverlapping(
            (block as *const riscv_cove::interrupt::TvmAiaParams).cast::<u8>(),
            aia_params_ptr,
            aia_params_len,
        );
    };
    stage_aia_params(&params);

    // N01: FID 0 parameter-block syntax and the nested-IMSIC limit. (a) a
    // short block and (b) a null block address are invalid parameters. (c)
    // and (d) are invalid addresses: without that guard the block would be
    // read at +4, where group_index_bits sees the staged group_index_shift
    // (24), or from the TSM image, whose first eight bytes are not a 4 KiB
    // aligned base, so both would return -3 instead of success. (e) a block
    // whose 32-byte end overflows lies outside host memory, because
    // is_host_range computes the end with checked_add. (f) the valid block
    // with guests_per_hart = 1 and guest_index_bits = 1 meets the spec's
    // guest-bit minimum (log2(1 + 1) = 1) and is refused only by the
    // implementation's nested-IMSIC limit; deleting that limit accepts it.
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len - 1);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with params_len below sizeof(TvmAiaParams) must be SBI_ERR_INVALID_PARAM"
    );
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, 0, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with a null params address must be SBI_ERR_INVALID_PARAM"
    );
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize + 4, aia_params_len);
    assert_eq!(
        err,
        (-5isize) as usize,
        "FID 0 with a misaligned params address must be SBI_ERR_INVALID_ADDRESS"
    );
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, 0x8040_0000, aia_params_len);
    assert_eq!(
        err,
        (-5isize) as usize,
        "FID 0 with a params block in TSM memory must be SBI_ERR_INVALID_ADDRESS"
    );
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, 0xffff_ffff_ffff_ffe0, aia_params_len);
    assert_eq!(
        err,
        (-5isize) as usize,
        "FID 0 with a params block whose end overflows must be SBI_ERR_INVALID_ADDRESS"
    );
    let mut nested_params = params;
    nested_params.guests_per_hart = 1;
    nested_params.guest_index_bits = 1;
    stage_aia_params(&nested_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with guests_per_hart = 1 must be SBI_ERR_INVALID_PARAM"
    );
    println!(
        "[HOST] COVI-N01 OK: FID 0 short/null/misaligned/foreign/overflowing/nested params rejected"
    );

    stage_aia_params(&params);

    // N02a: FID 0 with an unknown TVM id. The buffer holds the same valid
    // parameter block the positive call below arms, and the TVM is still
    // Initializing, so only the id match can reject it: without that guard
    // the call would arm the AIA state and return success.
    let (err, _) = sbi_covi_init_tvm_aia(UNKNOWN_TVM_ID, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with an unknown TVM id must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N02a OK: FID 0 unknown TVM id rejected");

    // N13: imsic_base_addr is the address with group, hart and guest index
    // all zero, and it must be representable in the TVM's G-stage. The base
    // 8 bytes off page alignment is refused only by the 4 KiB alignment
    // rule: FID 0 does not run the per-vCPU membership predicate when no GPA
    // is recorded, and it never compares the base with a platform array. The
    // two page-aligned 0x8003_1000 variants are refused by the "declared
    // index bits must be zero in the base" rule. The 2^41 base has zero
    // index bits but lies outside the Sv39x4 G-stage aperture (TSM rule,
    // same bound as the page-table code). Deleting any one of these three
    // guards changes the corresponding call to success.
    let mut base_params = params;
    base_params.imsic_base_addr = VCPU_IMSIC_GPA + 8;
    stage_aia_params(&base_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with a misaligned base address must be SBI_ERR_INVALID_PARAM"
    );
    let mut base_params = params;
    base_params.imsic_base_addr = VCPU_IMSIC_GPA_ALT;
    stage_aia_params(&base_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with the hart index bit set in the base must be SBI_ERR_INVALID_PARAM"
    );
    let mut base_params = params;
    base_params.imsic_base_addr = VCPU_IMSIC_GPA_ALT;
    base_params.hart_index_bits = 0;
    base_params.guest_index_bits = 1;
    stage_aia_params(&base_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with the guest index bit set in the base must be SBI_ERR_INVALID_PARAM"
    );
    let mut base_params = params;
    base_params.imsic_base_addr = 1usize << 41;
    stage_aia_params(&base_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with a base outside the G-stage aperture must be SBI_ERR_INVALID_PARAM"
    );
    println!(
        "[HOST] COVI-N13 OK: FID 0 base outside the G-stage aperture or with index bits set rejected"
    );

    // N14: group index. This implementation supports a single group, so any
    // nonzero group field is rejected; group_index_shift must be >= 24
    // (§11.1) and fit in the address. Every other field holds the valid
    // value, so deleting the guard under test changes the result to
    // success.
    let mut group_params = params;
    group_params.group_index_bits = 1;
    stage_aia_params(&group_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with group_index_bits = 1 must be SBI_ERR_INVALID_PARAM"
    );
    let mut group_params = params;
    group_params.group_index_shift = 23;
    stage_aia_params(&group_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with group_index_shift = 23 must be SBI_ERR_INVALID_PARAM"
    );
    let mut group_params = params;
    group_params.group_index_shift = 64;
    stage_aia_params(&group_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with group_index_shift = 64 must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N14 OK: FID 0 group index / shift rejected");

    // N15: hart and guest index widths must fit above the 12 low zero bits.
    // guest_index_bits = 53 exceeds the 52-bit field above bit 12; that
    // bound is what keeps `PAGE_SHIFT + guest_index_bits` from overflowing,
    // and the call is invalid_param. hart_index_bits = 53 and
    // guest_index_bits = 52 with hart_index_bits 1 leave no room for the
    // hart field, so the mask derivation reports the field as
    // unrepresentable; that mask-fit rule is the only guard for those two
    // calls.
    let mut width_params = params;
    width_params.hart_index_bits = 53;
    stage_aia_params(&width_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with hart_index_bits = 53 must be SBI_ERR_INVALID_PARAM"
    );
    let mut width_params = params;
    width_params.hart_index_bits = 0;
    width_params.guest_index_bits = 53;
    stage_aia_params(&width_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with guest_index_bits = 53 must be SBI_ERR_INVALID_PARAM"
    );
    let mut width_params = params;
    width_params.guest_index_bits = 52;
    stage_aia_params(&width_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with guest_index_bits = 52 and hart_index_bits = 1 must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N15 OK: FID 0 out-of-range index widths rejected");

    // N16: §11.1 requires guest_index_bits >= log2(guests_per_hart + 1).
    // With guests_per_hart = 1 and guest_index_bits = 0 the block is below
    // that minimum, and the unsupported-nested-guests rule would also
    // reject it; deleting either rule alone leaves the call rejected by the
    // other, so this case pins the combination rather than one guard.
    let mut guests_params = params;
    guests_params.guests_per_hart = 1;
    stage_aia_params(&guests_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 with guests_per_hart = 1 and guest_index_bits = 0 must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N16 OK: FID 0 guests_per_hart below the guest-bit minimum rejected");

    // N17: FID 1 is only legal after sbi_covi_init_tvm_aia() (§11.2). The
    // vCPU exists and the GPA is hart 0 / guest 0 of the block about to be
    // armed, so deleting the AIA-enabled guard changes the result from
    // invalid_param to invalid_address (the empty default layout accepts no
    // nonzero GPA).
    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(tvm_id, 0, VCPU_IMSIC_GPA);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 1 before FID 0 must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N17 OK: FID 1 before FID 0 rejected");

    stage_aia_params(&params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(err, 0, "init_tvm_aia must succeed");

    // P01: §11.1's only lifecycle limit is "cannot be called after
    // sbi_covh_finalize_tvm()", so a second FID 0 before finalize is legal.
    // Treating a second call as an error would change this result to -3.
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(err, 0, "repeat init_tvm_aia before finalize must succeed");
    println!("[HOST] COVI-P01 OK: repeat FID 0 before finalize accepted");

    // P02: the block describes the TVM's virtual IMSIC, not the platform's
    // interrupt-file array, so a base outside the physical geometry is
    // legal. No vCPU GPA is recorded yet, so no recorded-GPA constraint
    // applies either. Restoring the old "base must equal the platform
    // array" check would change this result to -3.
    let mut wide_params = params;
    wide_params.imsic_base_addr = 0x9000_0000;
    stage_aia_params(&wide_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(err, 0, "init_tvm_aia with a non-platform base must succeed");
    println!(
        "[HOST] COVI-P02 OK: FID 0 with a non-platform base accepted before any GPA is recorded"
    );

    // N18: 0x9000_0000 is hart 0 / guest 0 of the array P02 armed, but no
    // registered memory region contains it. Deleting the region-containment
    // rule would change this result to success; virtual-array membership
    // alone accepts it.
    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(tvm_id, 0, 0x9000_0000);
    assert_eq!(
        err,
        (-5isize) as usize,
        "FID 1 with a GPA outside every memory region must be SBI_ERR_INVALID_ADDRESS"
    );
    println!(
        "[HOST] COVI-N18 OK: FID 1 GPA in the virtual array but outside every region rejected"
    );

    // The G-stage aperture is exactly 2^41 bytes wide (Sv39x4), so register
    // two pages above it before exercising FID 1's aperture rule. The region
    // stays registered for the rest of the run: no handler removes regions.
    // It is harmless because the other readers — the overlap check when a
    // region is added, measured-page GPA containment, demand-zero fault
    // containment and FID 1 page containment — are only ever queried with
    // addresses from the low region, and this TVM gets no further region.
    let (err, _) = sbi_covh_add_tvm_memory_region(tvm_id, 0x200_0000_0000, 0x2000);
    assert_eq!(err, 0, "adding the region above 2^41 must succeed");

    // N19: FID 1 with a GPA at the first address outside the Sv39x4
    // G-stage aperture. The block armed here has base 0, guest index bits 0
    // and hart index bits 30, so 0x200_0000_0000 is the array entry whose
    // bit 41 (inside the hart field) is set. It is page-aligned, a member
    // of the virtual array, and inside the region registered above, so only
    // the aperture rule can reject it; deleting that rule would return 0.
    let mut aperture_params = params;
    aperture_params.imsic_base_addr = 0;
    aperture_params.hart_index_bits = 30;
    stage_aia_params(&aperture_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(err, 0, "arming the wide-hart-field block must succeed");
    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(tvm_id, 0, 1usize << 41);
    assert_eq!(
        err,
        (-5isize) as usize,
        "FID 1 with a GPA at 2^41 must be SBI_ERR_INVALID_ADDRESS"
    );
    println!("[HOST] COVI-N19 OK: FID 1 GPA outside the G-stage aperture rejected");

    stage_aia_params(&params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(err, 0, "restoring the valid AIA block must succeed");

    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(tvm_id, 0, VCPU_IMSIC_GPA);
    assert_eq!(err, 0, "set_tvm_aia_cpu_imsic_addr must succeed");
    println!("[MARKER 08] HOST: TVM #{} created with 1 vCPU.", tvm_id);

    // ── COVI negative paths: unknown TVM id, vCPU id, GPA (pre-finalize) ──
    // The TVM is still Initializing and its AIA configuration is armed, so
    // only the guards named per case can produce the result.
    // N02b: FID 1 with an unknown TVM id. The TVM is Initializing, the AIA
    // state is armed, vCPU 0 exists, and the GPA is a valid per-vCPU base
    // inside the memory region and not the recorded one, so without the id
    // match the call would record it and return success.
    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(UNKNOWN_TVM_ID, 0, VCPU_IMSIC_GPA_ALT);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 1 with an unknown TVM id must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N02b OK: FID 1 unknown TVM id rejected");

    // N03a: FID 1 with an unknown vCPU id. The vCPU lookup runs before the
    // GPA is examined, so the id match produces the result even though the
    // GPA is a valid per-vCPU base inside the memory region.
    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(tvm_id, UNKNOWN_VCPU_ID, VCPU_IMSIC_GPA_ALT);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 1 with an unknown vCPU id must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N03a OK: FID 1 unknown vCPU id rejected");

    // N04: FID 1 address validation. The misaligned GPA is refused by the
    // page-alignment clause (virtual-array membership would also reject it,
    // since its low bits differ from the base). GPA 0 is outside the
    // virtual array based at 0x8003_0000 and outside every memory region;
    // there is no standalone zero-GPA rule, so deleting either of those
    // guards still leaves zero rejected by the other. 0x9000_0000 is
    // likewise outside both. 0x8003_2000 is inside the TVM's memory region
    // but outside the virtual array, so only the membership rule rejects
    // it: deleting that rule would change this result to success.
    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(tvm_id, 0, VCPU_IMSIC_GPA + 8);
    assert_eq!(
        err,
        (-5isize) as usize,
        "FID 1 with a misaligned GPA must be SBI_ERR_INVALID_ADDRESS"
    );
    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(tvm_id, 0, 0);
    assert_eq!(
        err,
        (-5isize) as usize,
        "FID 1 with a zero GPA must be SBI_ERR_INVALID_ADDRESS"
    );
    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(tvm_id, 0, 0x9000_0000);
    assert_eq!(
        err,
        (-5isize) as usize,
        "FID 1 with a GPA outside every memory region must be SBI_ERR_INVALID_ADDRESS"
    );
    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(tvm_id, 0, VCPU_IMSIC_GPA + 0x2000);
    assert_eq!(
        err,
        (-5isize) as usize,
        "FID 1 with a GPA outside the virtual array must be SBI_ERR_INVALID_ADDRESS"
    );
    println!("[HOST] COVI-N04 OK: FID 1 misaligned/zero/out-of-region/out-of-array GPA rejected");

    // P03: §11.2 forbids two vCPUs sharing a vIMSIC GPA, not one vCPU
    // re-recording its own. The TVM has a single vCPU, so re-recording the
    // GPA it already owns is accepted; treating a same-vCPU re-record as a
    // duplicate would change this result to -3.
    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(tvm_id, 0, VCPU_IMSIC_GPA);
    assert_eq!(
        err, 0,
        "FID 1 re-recording the same vCPU's vIMSIC GPA must succeed"
    );
    println!("[HOST] COVI-P03 OK: FID 1 re-recording the same vCPU's GPA accepted");

    // N20: a re-initialisation must keep every recorded vIMSIC GPA inside
    // the new virtual array. Base 0x8003_2000 makes 0x8003_0000 fall
    // outside it, so the call is refused and the previous configuration
    // stays armed. Without the recorded-GPA check the new block would be
    // armed and this call would return success.
    let mut moved_params = params;
    moved_params.imsic_base_addr = VCPU_IMSIC_GPA + 0x2000;
    stage_aia_params(&moved_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 that would invalidate the recorded GPA must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N20 OK: FID 0 that would invalidate the recorded GPA rejected");

    // P04: the rejected re-initialisation in N20 left the previous
    // configuration armed, so re-recording the same GPA still succeeds. If
    // N20 had armed its block, this GPA would be outside the new virtual
    // array and the call would return -5.
    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(tvm_id, 0, VCPU_IMSIC_GPA);
    assert_eq!(
        err, 0,
        "FID 1 after a rejected re-initialisation must still succeed"
    );
    println!("[HOST] COVI-P04 OK: previous AIA configuration survived a rejected FID 0");

    // P05: guest_index_bits = 1 adds a guest index field, but with
    // guests_per_hart = 0 guest index 0 is the only valid per-vCPU base, so
    // the recorded GPA 0x8003_0000 stays valid and the block is accepted.
    // Requiring guest_index_bits = 0 in the layout would change this result
    // to -3.
    let mut guest_bits_params = params;
    guest_bits_params.guest_index_bits = 1;
    stage_aia_params(&guest_bits_params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err, 0,
        "FID 0 with guest index bits while the recorded GPA is guest 0 must succeed"
    );
    println!(
        "[HOST] COVI-P05 OK: FID 0 with guest index bits accepted while the recorded GPA is guest 0"
    );

    // N21: under P05's layout 0x8003_1000 is guest index 1, not a per-vCPU
    // base (guests_per_hart = 0 means only guest index 0 is valid).
    // Deleting the "guest index must be 0" half of the membership predicate
    // would change this result to success.
    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(tvm_id, 0, VCPU_IMSIC_GPA_ALT);
    assert_eq!(
        err,
        (-5isize) as usize,
        "FID 1 with a guest-index GPA must be SBI_ERR_INVALID_ADDRESS"
    );
    println!("[HOST] COVI-N21 OK: FID 1 guest-index GPA rejected");

    stage_aia_params(&params);
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(err, 0, "restoring the valid AIA block must succeed");

    // N23: FID 4 before finalize. A bind is what makes a vCPU eligible to
    // run, so it requires a finalized TVM. Here the TVM is still
    // Initializing while every other bind input is valid: the AIA
    // configuration is armed, vCPU 0 carries its vIMSIC GPA, and the
    // converted guest file is free. Without the TVM-state guard the call
    // binds and returns 0; Table 37 names no state error, so invalid_param
    // is the project mapping. Deleting that guard changes this result to 0.
    let (err, _) = sbi_covi_bind_aia_imsic(tvm_id, 0, 0b10);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 4 before finalize must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N23 OK: FID 4 before finalize rejected");

    // N25: finalize refuses an unusable identity address before it changes
    // any state. 0x8040_0000 is 64-byte aligned but lies in TSM memory, and
    // base_paddr + 8 lies in host memory but is not 64-byte aligned; Table 19
    // names SBI_ERR_INVALID_PARAM for an invalid tvm_identity_addr, so both
    // are invalid parameters. The case still discriminates: without the
    // validation the first call would succeed and finalize the TVM, so the
    // valid finalize below could not run. The host cannot observe whether a
    // rejected call mutated the vCPU, because a successful finalize rewrites
    // the same fields; what this case pins is that a refused finalize leaves
    // the TVM usable.
    let (err, _) = sbi_covh_finalize_tvm(tvm_id, 0x8000_0000, 0, 0x8040_0000);
    assert_eq!(
        err,
        (-3isize) as usize,
        "finalize with an identity in TSM memory must be SBI_ERR_INVALID_PARAM"
    );
    let (err, _) = sbi_covh_finalize_tvm(tvm_id, 0x8000_0000, 0, base_paddr + 8);
    assert_eq!(
        err,
        (-3isize) as usize,
        "finalize with a misaligned host identity must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N25 OK: finalize with an unusable identity address rejected");

    // 10. Finalize TVM
    println!("[HOST] Finalizing TVM with entry PC 0x80000000...");
    let (err, _) = sbi_covh_finalize_tvm(tvm_id, 0x8000_0000, 0, 0);
    assert_eq!(err, 0, "finalize_tvm must succeed");
    println!(
        "[MARKER 09] HOST: TVM #{} finalized, launching vCPU #0...",
        tvm_id
    );

    // ── COVI negative paths: finalized TVM, unknown TVM id, unbound vCPU ──
    // N22: §11.1 forbids FID 0 after sbi_covh_finalize_tvm(). The block is
    // valid and the recorded GPA stays inside the candidate layout, so
    // without the state guard the call would return 0.
    let (err, _) = sbi_covi_init_tvm_aia(tvm_id, aia_params_ptr as usize, aia_params_len);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 0 after finalize must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N22 OK: FID 0 after finalize rejected");

    // N06: once finalized the vIMSIC GPA can no longer be set. The
    // alternative valid per-vCPU GPA makes a leaked success show up as
    // success, so the assertion can only pass if the state guard fires.
    let (err, _) = sbi_covi_set_tvm_aia_cpu_imsic_addr(tvm_id, 0, VCPU_IMSIC_GPA_ALT);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 1 after finalize must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N06 OK: FID 1 after finalize rejected");

    // N02c: FID 4 with an unknown TVM id. The TVM is finalized with its AIA
    // state armed, vCPU 0 exists, is unbound and has a vIMSIC GPA, and the
    // converted file is free, so without the id match the call would bind
    // and return success.
    let (err, _) = sbi_covi_bind_aia_imsic(UNKNOWN_TVM_ID, 0, 0b10);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 4 with an unknown TVM id must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N02c OK: FID 4 unknown TVM id rejected");

    // N03b: FID 4 with an unknown vCPU id. The finalized TVM is a state bind
    // accepts, its AIA state is armed, the mask is legal and vCPU 0 is
    // unbound with a GPA, so only the vCPU id match can reject the call.
    let (err, _) = sbi_covi_bind_aia_imsic(tvm_id, UNKNOWN_VCPU_ID, 0b10);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 4 with an unknown vCPU id must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N03b OK: FID 4 unknown vCPU id rejected");

    // N07: with guests_per_hart = 0 the only legal hgeie mask is bit 1
    // (G0, VGEIN 1); anything else is refused before the vCPU is touched.
    for mask in [0b00usize, 0b01, 0b11] {
        let (err, _) = sbi_covi_bind_aia_imsic(tvm_id, 0, mask);
        assert_eq!(
            err,
            (-3isize) as usize,
            "FID 4 with imsic_mask {:#04b} must be SBI_ERR_INVALID_PARAM",
            mask
        );
    }
    println!("[HOST] COVI-N07 OK: FID 4 illegal imsic_mask rejected");

    // N08: §7.4.2 — an AIA-enabled vCPU may not run before it is bound, even
    // though the TVM is Runnable and the vCPU id is valid. Table 28 (§10.17)
    // does not name a code for an unbound run, so invalid_param is the
    // project mapping for this case.
    let (err, _) = sbi_covh_run_tvm_vcpu(tvm_id, 0);
    assert_eq!(
        err,
        (-3isize) as usize,
        "run of an unbound AIA vCPU must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N08 OK: run before bind rejected");

    // N09: §11.8 identity bounds: identity 0 is reserved (AIA §3.1) and the
    // modeled file carries one 64-bit eip/eie word, so identities run
    // 1..=63. Both are refused before any pending bit is recorded. The
    // allow-bit guard would refuse both with the same invalid_param (bit 0
    // is never set, and the release build masks the shift), so this case
    // proves the rejection; the range guard is the one that fires only by
    // inspection.
    let (err, _) = sbi_covi_inject_tvm_cpu(tvm_id, 0, 0);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 7 with iid 0 must be SBI_ERR_INVALID_PARAM"
    );
    let (err, _) = sbi_covi_inject_tvm_cpu(tvm_id, 0, 64);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 7 with iid 64 (one above EXT_ID_MAX = 63) must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N09 OK: FID 7 out-of-range identity rejected");

    // 10a-prec. COVI bind: the vCPU binds to the hart's guest interrupt
    // file (mask bit 1 = G0, VGEIN 1) and becomes runnable (§11.5).
    let (err, _) = sbi_covi_bind_aia_imsic(tvm_id, 0, 0b10);
    assert_eq!(err, 0, "bind_aia_imsic must succeed");
    println!("[MARKER 18] HOST: COVI AIA bound (vgein=1).");

    // ── COVI negative paths: bound vCPU ──
    // N10: a bound vCPU cannot be bound again (§11.5: "The vCPU must
    // currently be unbound"), so the same legal mask must be refused while
    // the binding is held. The table does not name a code for a repeat
    // bind, so invalid_param is the project mapping for this case. The page
    // tracker's owner check in `bind_imsic_page` rejects the already-owned
    // file with the same invalid_param, so this case proves the rejection
    // but cannot, on its own, tell which of the two guards fired.
    let (err, _) = sbi_covi_bind_aia_imsic(tvm_id, 0, 0b10);
    assert_eq!(
        err,
        (-3isize) as usize,
        "repeat bind of a bound vCPU must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N10 OK: repeat bind rejected");

    // N11: the guest interrupt file stays owned by the TVM until destroy
    // releases it, so COVI FID 3 must refuse it while it is bound (§11.4).
    let (err, _) = sbi_covi_reclaim_tvm_aia_imsic(CONF_GUEST_FILE);
    assert_eq!(
        err,
        (-3isize) as usize,
        "reclaim of a bound guest interrupt file must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N11 OK: reclaim of a bound guest interrupt file rejected");

    // 10a. Run TVM vCPU 0 (first run: guest shares memory, exits)
    println!("[HOST] Running TVM vCPU 0 (first run: COVG share)...");
    let (err, val) = sbi_covh_run_tvm_vcpu(tvm_id, 0);
    println!("[HOST] TVM vCPU 0 exited with err={}, val={}", err, val);
    assert_eq!(err, 0, "run_tvm_vcpu must return success");
    assert_eq!(val, 1, "First exit must be EXIT_COVG_SHARE (=1)");

    // 10b. Host writes MAGIC to the shared SPA (Host has MPT access after COVG share)
    println!(
        "[HOST] Writing MAGIC 0xCAFEF00D to shared SPA 0x{:x}...",
        shared_paddr
    );
    unsafe {
        core::ptr::write_volatile(shared_paddr as *mut u64, 0xCAFE_F00D);
    }

    // 10c. Re-add shared pages to restore G-stage mapping for guest
    println!(
        "[HOST] Re-adding shared pages (GPA 0x80010000 -> SPA 0x{:x})...",
        shared_paddr
    );
    let (err, _) =
        sbi_covh_add_tvm_shared_pages(tvm_id, shared_paddr, shared_paddr, 2, 0x8001_0000);
    assert_eq!(err, 0, "add_tvm_shared_pages must succeed");

    // Exercise the complete fence sequence with byte lengths, as specified by COVH.
    println!("[HOST] Invalidate + fence + validate shared pages...");
    let (err, _) = sbi_covh_tvm_invalidate_pages(tvm_id, 0x8001_0000, 0x2000);
    assert_eq!(err, 0, "tvm_invalidate_pages must succeed");
    let (err, _) = sbi_covh_tvm_fence(tvm_id);
    assert_eq!(err, 0, "tvm_fence must succeed");
    let (err, _) = sbi_covh_tvm_validate_pages(tvm_id, 0x8001_0000, 0x2000);
    assert_eq!(err, 0, "tvm_validate_pages must succeed");

    // 10d. Run TVM vCPU 0 again (verify shared memory, demand-zero, unshare)
    println!("[HOST] Running TVM vCPU 0 (second run: verify + demand-zero + unshare)...");
    let (err, val) = sbi_covh_run_tvm_vcpu(tvm_id, 0);
    println!("[HOST] TVM vCPU 0 exited with err={}, val={}", err, val);
    assert_eq!(err, 0, "second run_tvm_vcpu must return success");
    assert_eq!(val, 2, "Second exit must be EXIT_COVG_UNSHARE (=2)");

    // 10e. Third run: the guest arms VS delivery and allows identity 1
    // (COVG FID 4), which exits back to the host (§12.5 implies exit +
    // resume).
    println!("[HOST] Running TVM vCPU 0 (third run: COVG allow external)...");
    let (err, val) = sbi_covh_run_tvm_vcpu(tvm_id, 0);
    println!("[HOST] TVM vCPU 0 exited with err={}, val={}", err, val);
    assert_eq!(err, 0, "third run_tvm_vcpu must return success");
    assert_eq!(val, 6, "allow exit must be EXIT_COVG_ALLOW_EXT (=6)");

    // 10f. Inject identity 1 through the guest interrupt file (§11.8):
    // the pending MSI is taken as soon as the vCPU re-enters VS.
    let (err, _) = sbi_covi_inject_tvm_cpu(tvm_id, 0, 1);
    assert_eq!(err, 0, "inject of an allowed identity must succeed");

    println!("[HOST] Running TVM vCPU 0 (fourth run: VS handler + allow identity 2)...");
    let (err, val) = sbi_covh_run_tvm_vcpu(tvm_id, 0);
    println!("[HOST] TVM vCPU 0 exited with err={}, val={}", err, val);
    assert_eq!(err, 0, "fourth run_tvm_vcpu must return success");
    assert_eq!(val, 6, "allow-2 exit must be EXIT_COVG_ALLOW_EXT (=6)");

    // 10f-2. Park a pending MSI the guest never enables or claims:
    // identity 2 is allowed (COVG FID 4 above) but not in the guest's
    // eie, so the pending bit stays in the file across every later run
    // and destroy's guest-file scrub must claim it.
    let (err, _) = sbi_covi_inject_tvm_cpu(tvm_id, 0, 2);
    assert_eq!(err, 0, "inject of the never-enabled identity must succeed");

    // N02d: FID 7 with an unknown TVM id. The TVM is armed and bound, vCPU 0
    // is valid, and identity 2 is guest-allowed and already pending, so
    // without the id match the call would re-write identity 2 into the file
    // and return success; the pending bit is already set, so a leaked bypass
    // would still leave the scrub claiming exactly one identity.
    let (err, _) = sbi_covi_inject_tvm_cpu(UNKNOWN_TVM_ID, 0, 2);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 7 with an unknown TVM id must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N02d OK: FID 7 unknown TVM id rejected");

    // N03c: FID 7 with an unknown vCPU id, now that the guest has allowed
    // identity 2 and it is already pending: the vCPU id match is the only
    // guard left that can reject this call. A leaked bypass would re-set
    // the already-pending identity-2 bit, so the guest-file scrub would
    // still claim exactly one identity.
    let (err, _) = sbi_covi_inject_tvm_cpu(tvm_id, UNKNOWN_VCPU_ID, 2);
    assert_eq!(
        err,
        (-3isize) as usize,
        "FID 7 with an unknown vCPU id must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N03c OK: FID 7 unknown vCPU id rejected");

    println!("[HOST] Running TVM vCPU 0 (fifth run: deny identity 1)...");
    let (err, val) = sbi_covh_run_tvm_vcpu(tvm_id, 0);
    println!("[HOST] TVM vCPU 0 exited with err={}, val={}", err, val);
    assert_eq!(err, 0, "fifth run_tvm_vcpu must return success");
    assert_eq!(val, 7, "deny exit must be EXIT_COVG_DENY_EXT (=7)");

    // 10g. After the guest denied identity 1 (COVG FID 5) the same
    // injection must be refused (project mapping: denied identity ->
    // invalid input).
    let (err, _) = sbi_covi_inject_tvm_cpu(tvm_id, 0, 1);
    assert_eq!(
        err,
        (-3isize) as usize,
        "inject of a denied identity must be SBI_ERR_INVALID_PARAM"
    );
    // N12: by now the guest has allowed identity 2 and denied identity 1;
    // every other identity stays denied (§12.5 default deny) and must be
    // rejected without touching the guest interrupt file.
    let (err, _) = sbi_covi_inject_tvm_cpu(tvm_id, 0, 5);
    assert_eq!(
        err,
        (-3isize) as usize,
        "inject of a non-allowed identity must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] COVI-N12 OK: inject of a non-allowed identity rejected");

    println!("[MARKER 19] HOST: COVI inject gating validated.");
    println!("[MARKER 20] HOST: COVI init/addr/bind/inject/run/reclaim negative paths PASSED.");

    // 10h. Run once more: the guest exits clean via SRST.
    println!("[HOST] Running TVM vCPU 0 (sixth run: clean exit)...");
    let (err, val) = sbi_covh_run_tvm_vcpu(tvm_id, 0);
    println!("[HOST] TVM vCPU 0 exited with err={}, val={}", err, val);
    assert_eq!(err, 0, "sixth run_tvm_vcpu must return success");
    assert_eq!(val, 0, "Final exit must be EXIT_CLEAN (=0)");
    println!(
        "[MARKER 14] HOST: Received TVM exit, tearing down TVM #{}.",
        tvm_id
    );

    // 11. Remove the shared mappings, then destroy the TVM and reclaim pages.
    println!("[HOST] Removing shared pages...");
    let (err, _) = sbi_covh_tvm_invalidate_pages(tvm_id, 0x8001_0000, 0x2000);
    assert_eq!(err, 0, "pre-remove invalidate must succeed");
    let (err, _) = sbi_covh_tvm_fence(tvm_id);
    assert_eq!(err, 0, "pre-remove fence must succeed");
    let (err, _) = sbi_covh_tvm_remove_pages(tvm_id, 0x8001_0000, 0x2000);
    assert_eq!(err, 0, "tvm_remove_pages must succeed");

    // 12. Destroy TVM
    println!("[HOST] Destroying TVM {}...", tvm_id);
    let (err, _) = sbi_covh_destroy_tvm(tvm_id);
    assert_eq!(err, 0, "destroy_tvm must succeed");

    // 13. Reclaim the guest interrupt file, then the RAM pages. Destroy
    // released the bound file back to the unassigned pool (CoVE 11.4).
    let (err, _) = sbi_covi_reclaim_tvm_aia_imsic(CONF_GUEST_FILE);
    assert_eq!(err, 0, "reclaim of the released guest file must succeed");
    println!(
        "[HOST] Reclaiming {} pages at 0x{:x}...",
        NUM_PAGES, base_paddr
    );
    let (err, _) = sbi_covh_reclaim_pages(base_paddr, NUM_PAGES);
    assert_eq!(err, 0, "reclaim_pages must succeed");

    // 13. Verify Host memory access after reclaim
    unsafe {
        core::ptr::write_volatile(base_paddr as *mut u64, 0x1234_5678_9ABC_DEF0);
        let read_val = core::ptr::read_volatile(base_paddr as *const u64);
        assert_eq!(
            read_val, 0x1234_5678_9ABC_DEF0,
            "Host memory access after reclaim must succeed"
        );
    }
    println!("[MARKER 15] HOST: All confidential pages reclaimed successfully.");

    println!("[HOST] PHASE 4 PASS: FENCE_EXIT_OK");

    // ── Hostile-host validation ──────────────────────────────
    // Every call below must be rejected by the TSM/RDSM input validation.
    println!("[HOST] PHASE 5.5: hostile-host validation tests...");

    // N1: direct RDSM MPT_SET from the host domain must be denied.
    let (err, _) = sbi_rdsm_mpt_set_from_host(1, 0x8040_0000, 0x1000, 7);
    assert_eq!(
        err,
        (-8isize) as usize,
        "host-domain MPT_SET must be SBI_ERR_DENIED"
    );
    println!("[HOST] P55-N1 OK: host-domain MPT_SET denied");

    // N2: get_tsm_info into confidential (TSM image) memory.
    let (err, _) = sbi_covh_get_tsm_info_raw(0x8040_0000, 0x40);
    assert_ne!(err, 0, "get_tsm_info into TSM memory must fail");
    println!("[HOST] P55-N2 OK: get_tsm_info bad buffer rejected");

    // N3: converting firmware / TSM image / MPT pool pages must fail.
    let (err, _) = sbi_covh_convert_pages(0x8040_0000, 1);
    assert_ne!(err, 0, "converting TSM image must fail");
    let (err, _) = sbi_covh_convert_pages(0x8090_0000, 1);
    assert_ne!(err, 0, "converting MPT page pool must fail");
    let (err, _) = sbi_covh_convert_pages(0x8000_0000, 1);
    assert_ne!(err, 0, "converting firmware gap must fail");
    println!("[HOST] P55-N3 OK: reserved-region converts rejected");

    // N4: create_tvm with the parameter block in the TSM image region
    // (never-written hostile address, page- and 8-byte-aligned).
    let (err, _) = sbi_covh_create_tvm_at(0x8040_0100, 0x40);
    assert_ne!(err, 0, "create_tvm with TSM-region params must fail");
    println!("[HOST] P55-N4 OK: create_tvm hostile params rejected");

    // N5: a second TVM with hostile measured-source / shared-target pages.
    let (err, _) = sbi_covh_convert_pages(base_paddr, NUM_PAGES);
    assert_eq!(err, 0, "reconvert for phase 5.5 TVM must succeed");
    let (err, _) = sbi_covh_global_fence();
    assert_eq!(err, 0);
    let (err, _) = sbi_covh_local_fence();
    assert_eq!(err, 0);
    let params2 = riscv_cove::host::TvmCreateParams {
        tvm_page_directory_addr: root_pt_paddr,
        tvm_state_addr: tvm_state_paddr,
    };
    let (err, tvm2) = sbi_covh_create_tvm(&params2);
    assert_eq!(err, 0, "phase 5.5 TVM creation must succeed");
    let (err, _) = sbi_covh_add_tvm_memory_region(tvm2, 0x8000_0000, 0x10_0000);
    assert_eq!(err, 0);
    let (err, _) = sbi_covh_add_tvm_page_table_pages(tvm2, pt_pool_paddr, 4);
    assert_eq!(err, 0);
    // N5a: measured pages sourced from the TSM image must be rejected.
    let (err, _) = sbi_covh_add_tvm_measured_pages(
        tvm2,
        0x8040_0000,
        guest_dst_paddr,
        0,
        1,
        0x8000_0000,
    );
    assert_ne!(err, 0, "measured pages from TSM memory must fail");
    // N5b: shared pages targeting TSM image memory must be rejected.
    let (err, _) =
        sbi_covh_add_tvm_shared_pages(tvm2, 0x8040_0000, 0x8040_0000, 1, 0x8001_1000);
    assert_ne!(err, 0, "shared pages from TSM memory must fail");
    println!("[HOST] P55-N5 OK: hostile measured/shared pages rejected");

    // N8: running a vCPU of a not-yet-finalized TVM must be rejected
    // (tvm2 is still in the Initializing state at this point).
    let (err, _) = sbi_covh_run_tvm_vcpu(tvm2, 0);
    assert_eq!(
        err,
        (-3isize) as usize,
        "run_tvm_vcpu before finalize must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] P55-N8 OK: run of unfinalized TVM rejected");

    // Give tvm2 a vCPU and finalize it once so that the repeat-finalize
    // rejection (N6) can be exercised against a legally finalized TVM.
    let (err, _) = sbi_covh_create_tvm_vcpu(tvm2, 0, vcpu_state_paddr);
    assert_eq!(err, 0, "phase 5.5 vcpu creation must succeed");
    let (err, _) = sbi_covh_finalize_tvm(tvm2, 0x8000_0000, 0, 0);
    assert_eq!(err, 0, "phase 5.5 TVM finalize must succeed");

    // N6: finalizing an already-finalized TVM must be rejected.
    let (err, _) = sbi_covh_finalize_tvm(tvm2, 0x8000_0000, 0, 0);
    assert_eq!(
        err,
        (-3isize) as usize,
        "repeat finalize_tvm must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] P55-N6 OK: repeat finalize rejected");

    // N7: destroying a TVM id that does not exist must be rejected.
    let (err, _) = sbi_covh_destroy_tvm(0xDEAD);
    assert_eq!(
        err,
        (-3isize) as usize,
        "destroy_tvm of unknown id must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] P55-N7 OK: destroy of unknown TVM id rejected");

    // N9: converting a page already owned by the live TVM (part of its page
    // table pool) must be rejected by the page-state machine.
    let (err, _) = sbi_covh_convert_pages(pt_pool_paddr, 1);
    assert_eq!(
        err,
        (-3isize) as usize,
        "convert of TVM-owned page must be SBI_ERR_INVALID_PARAM"
    );
    println!("[HOST] P55-N9 OK: convert of TVM-owned page rejected");

    // N10: an unknown COVH function id must be reported as not supported.
    let (err, _) = sbi_covh_raw_fid(0xFF);
    assert_eq!(
        err,
        (-2isize) as usize,
        "unknown COVH FID must be SBI_ERR_NOT_SUPPORTED"
    );
    println!("[HOST] P55-N10 OK: unknown COVH FID rejected");

    // N11: direct RDSM SET_SIDN from the host domain must be denied — the
    // interrupt-domain assignment is confidential-domain-only.
    let (err, _) = sbi_rdsm_set_sidn_from_host(1, 1);
    assert_eq!(
        err,
        (-8isize) as usize,
        "host-domain SET_SIDN must be SBI_ERR_DENIED"
    );
    println!("[HOST] P55-N11 OK: host-domain SET_SIDN denied");

    let (err, _) = sbi_covh_destroy_tvm(tvm2);
    assert_eq!(err, 0);
    let (err, _) = sbi_covh_reclaim_pages(base_paddr, NUM_PAGES);
    assert_eq!(err, 0);

    println!("[HOST] PHASE 5.5 PASS: HOST_VALIDATION_OK");

    // ── Phase 7: COVI convert/reclaim + bind/inject gating ──
    println!("[HOST] PHASE 7: COVI convert/reclaim tests...");

    // The extension must be advertised (probe = TSM ready + Smsdia +
    // supervisor IMSIC geometry).
    let covi_probed = sbi_probe_extension(riscv_cove::interrupt::EID_COVI);
    assert_eq!(covi_probed, 1, "EXT_COVI must be discovered via base probe");

    // NEMU's DTB layout (SmmTT v0.49 section 6.1, B + n*2^14 + h*2^13 +
    // g*2^12 at 0x3000_8000): SIDN0/1 each cover hart0/hart1 with page
    // order S,G0. The confidential domain (SIDN1) hart0 guest file is the
    // bind target; machine files and supervisor files are never
    // COVI-convertible.
    const CONF_GUEST_FILE: usize = 0x3000_D000;
    const SUPERVISOR_FILE_SIDN0: usize = 0x3000_8000;
    const MACHINE_HART1_FILE: usize = 0x3000_1000;

    // C1: addresses outside the IMSIC whitelist are rejected outright.
    let (err, _) = sbi_covi_convert_aia_imsic(0x9000_0000);
    assert_eq!(
        err,
        (-5isize) as usize,
        "non-IMSIC convert must be SBI_ERR_INVALID_ADDRESS"
    );
    // C2: misaligned IMSIC addresses are rejected.
    let (err, _) = sbi_covi_convert_aia_imsic(CONF_GUEST_FILE + 8);
    assert_eq!(
        err,
        (-5isize) as usize,
        "misaligned convert must be SBI_ERR_INVALID_ADDRESS"
    );
    // C2b: machine files live outside the supervisor whitelist, and
    // supervisor-level files (guest index 0) are not COVI-convertible.
    let (err, _) = sbi_covi_convert_aia_imsic(MACHINE_HART1_FILE);
    assert_eq!(
        err,
        (-5isize) as usize,
        "machine-file convert must be SBI_ERR_INVALID_ADDRESS"
    );
    let (err, _) = sbi_covi_convert_aia_imsic(SUPERVISOR_FILE_SIDN0);
    assert_eq!(
        err,
        (-5isize) as usize,
        "supervisor-file convert must be SBI_ERR_INVALID_ADDRESS"
    );

    // C3: the positive path — convert, fence, reclaim (CoVE 11.3 order),
    // with the tracker-state violations interleaved.
    let (err, _) = sbi_covi_convert_aia_imsic(CONF_GUEST_FILE);
    assert_eq!(err, 0, "convert_aia_imsic must succeed");
    let (err, _) = sbi_covi_convert_aia_imsic(CONF_GUEST_FILE);
    assert_eq!(
        err,
        (-3isize) as usize,
        "double convert must be SBI_ERR_INVALID_PARAM"
    );
    let (err, _) = sbi_covh_global_fence();
    assert_eq!(err, 0, "post-convert global fence must succeed");
    let (err, _) = sbi_covh_local_fence();
    assert_eq!(err, 0, "post-convert local fence must succeed");
    let (err, _) = sbi_covi_reclaim_tvm_aia_imsic(CONF_GUEST_FILE + 0x1000);
    assert_eq!(
        err,
        (-3isize) as usize,
        "reclaim of an untracked page must be SBI_ERR_INVALID_PARAM"
    );
    let (err, _) = sbi_covi_reclaim_tvm_aia_imsic(CONF_GUEST_FILE);
    assert_eq!(err, 0, "reclaim_tvm_aia_imsic must succeed");
    let (err, _) = sbi_covi_reclaim_tvm_aia_imsic(CONF_GUEST_FILE);
    assert_eq!(
        err,
        (-3isize) as usize,
        "double reclaim must be SBI_ERR_INVALID_PARAM"
    );

    // C4: COVI FIDs outside the implemented set report not-supported.
    let (err, _) = sbi_covi_raw_fid(riscv_cove::interrupt::UNBIND_AIA_IMSIC_BEGIN);
    assert_eq!(
        err,
        (-2isize) as usize,
        "unimplemented COVI FID must be SBI_ERR_NOT_SUPPORTED"
    );

    println!("[MARKER 17] HOST: COVI convert/reclaim and negative paths PASSED.");

    println!("[MARKER 16] HOST: ALL COVE E2E TESTS PASSED!");
    loop {
        unsafe {
            asm!("wfi");
        }
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    println!("[HOST PANIC] {}", info);
    loop {
        unsafe {
            asm!("wfi");
        }
    }
}
