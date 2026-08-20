#![no_std]
#![no_main]

use core::arch::{asm, naked_asm};
use core::fmt::{self, Write};
use core::panic::PanicInfo;

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
        print!("\n");
    };
    ($($arg:tt)*) => {
        let _ = core::fmt::write(&mut SbiConsole, format_args!($($arg)*));
        print!("\n");
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

#[unsafe(no_mangle)]
pub extern "C" fn host_main(_hart_id: usize, _fdt_paddr: usize) -> ! {
    println!("[HOST] Booting... HOST_STARTED");

    // 1. Probe EXT_SUPD
    let probe_supd = sbi_probe_extension(riscv_cove::supd::EID_SUPD);
    println!(
        "[HOST] Probing EXT_SUPD (0x{:x}): result = {}",
        riscv_cove::supd::EID_SUPD,
        probe_supd
    );
    assert!(probe_supd != 0, "EXT_SUPD must be supported");

    // 2. Query active domains
    let (err, active_domains) = sbi_supd_get_active_domains();
    println!(
        "[HOST] SUPD get_active_domains: err = {}, active_domains = 0b{:b}",
        err, active_domains
    );
    assert_eq!(err, 0, "get_active_domains must succeed");
    assert_eq!(
        active_domains & 0b11,
        0b11,
        "Active domains must contain bit 0 (Host) and bit 1 (Confidential/TSM)"
    );

    // 3. Probe EXT_COVH
    let probe_covh = sbi_probe_extension(riscv_cove::host::EID_COVH);
    println!(
        "[HOST] Probing EXT_COVH (0x{:x}): result = {}",
        riscv_cove::host::EID_COVH,
        probe_covh
    );
    assert!(probe_covh != 0, "EXT_COVH must be supported");

    // 4. Call sbi_covh_get_tsm_info
    let mut tsm_info = core::mem::MaybeUninit::<riscv_cove::host::TsmInfo>::zeroed();
    let tsm_info_ref = unsafe { &mut *tsm_info.as_mut_ptr() };
    let (err, val) = sbi_covh_get_tsm_info(tsm_info_ref);
    println!("[HOST] COVH get_tsm_info: err = {}, val = {}", err, val);
    assert_eq!(err, 0, "get_tsm_info must succeed");

    let info = unsafe { tsm_info.assume_init() };
    println!(
        "[HOST] TSM Status: {}, Impl: 0x{:x}, Version: {}, Capabilities: 0x{:x}, StatePages: {}, MaxVcpus: {}, VcpuStatePages: {}",
        if info.tsm_state == 2 {
            "READY"
        } else {
            "UNKNOWN"
        },
        info.tsm_impl_id,
        info.tsm_version,
        info.tsm_capabilities,
        info.tvm_state_pages,
        info.tvm_max_vcpus,
        info.tvm_vcpu_state_pages
    );

    assert_eq!(info.tsm_state, 2, "TSM state must be READY");
    assert_eq!(
        info.tsm_impl_id, 0x54534D31,
        "TSM impl ID must match 0x54534D31"
    );
    assert_eq!(info.tsm_version, 1, "TSM version must be 1");
    assert_eq!(info.tvm_state_pages, 4, "TVM state pages must be 4");
    assert_eq!(info.tvm_max_vcpus, 1, "TVM max vcpus must be 1");
    assert_eq!(
        info.tvm_vcpu_state_pages, 2,
        "TVM vcpu state pages must be 2"
    );

    println!("[HOST] PHASE 2 PASS: GET_TSM_INFO_OK");

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
