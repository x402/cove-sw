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
            #[allow(deprecated)]
            let _ = sbi_rt::legacy::console_putchar(b as usize);
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

const NUM_PAGES: usize = 18;

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

    // 3. Memory layout of the 18 converted pages:
    // [0x0000..0x4000) (4 pages): TVM Root Page Table (16KB aligned)
    // [0x4000..0x8000) (4 pages): TVM State (16KB)
    // [0x8000..0xA000) (2 pages): vCPU 0 State (8KB)
    // [0xA000..0xE000) (4 pages): TVM Page Table Pool (16KB)
    // [0xE000..0x12000)(4 pages): TVM Measured Guest Payload (16KB)
    let base_paddr = core::ptr::addr_of_mut!(HOST_PAGES) as usize;
    assert_eq!(base_paddr % 16384, 0, "Base paddr must be 16KB aligned");

    let root_pt_paddr = base_paddr;
    let tvm_state_paddr = base_paddr + 0x4000;
    let vcpu_state_paddr = base_paddr + 0x8000;
    let pt_pool_paddr = base_paddr + 0xA000;
    let guest_dst_paddr = base_paddr + 0xE000;

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

    // 8. Create vCPU 0
    println!("[HOST] Creating vCPU 0 at 0x{:x}...", vcpu_state_paddr);
    let (err, _) = sbi_covh_create_tvm_vcpu(tvm_id, 0, vcpu_state_paddr);
    assert_eq!(err, 0, "create_tvm_vcpu must succeed");

    // 9. Finalize TVM
    println!("[HOST] Finalizing TVM with entry PC 0x80000000...");
    let (err, _) = sbi_covh_finalize_tvm(tvm_id, 0x8000_0000, 0, 0);
    assert_eq!(err, 0, "finalize_tvm must succeed");

    // 10. Run TVM vCPU 0
    println!("[HOST] Running TVM vCPU 0...");
    let (err, val) = sbi_covh_run_tvm_vcpu(tvm_id, 0);
    println!("[HOST] TVM vCPU 0 exited with err={}, val={}", err, val);
    assert_eq!(err, 0, "run_tvm_vcpu must return success");

    // 11. Destroy TVM
    println!("[HOST] Destroying TVM {}...", tvm_id);
    let (err, _) = sbi_covh_destroy_tvm(tvm_id);
    assert_eq!(err, 0, "destroy_tvm must succeed");

    // 12. Reclaim Pages
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

    println!("[HOST] PHASE 3 PASS: TVM_LIFECYCLE_OK");

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
