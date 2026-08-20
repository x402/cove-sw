#![no_std]
#![no_main]

use core::arch::{asm, naked_asm};
use core::fmt::{self, Write};
use core::panic::PanicInfo;

pub const EID_RDSM: usize = 0x5244534D; // "RDSM"
pub const FID_RDSM_GET_INFO: usize = 0;
pub const FID_RDSM_MPT_SET: usize = 1;
pub const FID_RDSM_MFENCE_PA: usize = 2;
pub const FID_RDSM_TEERET: usize = 3;

pub const NORMAL_RETURN: usize = 0;
pub const TVM_EXIT: usize = 1;
pub const TSM_READY: usize = 2;

pub const TSM_IMPL_CUSTOM: u32 = 0x54534D31; // "TSM1"
pub const TSM_VERSION: u32 = 1;

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
        // Call tsm_main(hart_id, fdt_paddr)
        "call tsm_main",
        // Should not return, but if it does:
        "3:",
        "wfi",
        "j 3b"
    )
}

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsm_dispatch_entry() -> ! {
    naked_asm!(
        "la sp, boot_stack_top",
        "call tsm_dispatch",
        // tsm_dispatch returns (error, value) in a0, a1
        // Forward back to RDSM: rdsm_teeret(NORMAL_RETURN, error, value)
        "mv a2, a1",
        "mv a1, a0",
        "li a0, 0",          // NORMAL_RETURN
        "li a7, 0x5244534D", // EID_RDSM
        "li a6, 3",          // FID_RDSM_TEERET
        "ecall",
        "1: wfi",
        "j 1b"
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn tsm_main(_hart_id: usize, _fdt_paddr: usize) -> ! {
    println!("[TSM] Booting... TSM_READY");

    // Inform RDSM that TSM is ready and provide dispatch entry point
    rdsm_teeret(TSM_READY, tsm_dispatch_entry as *const () as usize, 0);
}

#[repr(C)]
pub struct SbiRet {
    pub error: usize,
    pub value: usize,
}

impl SbiRet {
    pub const fn success(value: usize) -> Self {
        Self { error: 0, value }
    }
    pub const fn not_supported() -> Self {
        Self {
            error: (-1isize) as usize,
            value: 0,
        }
    }
    pub const fn invalid_param() -> Self {
        Self {
            error: (-3isize) as usize,
            value: 0,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tsm_dispatch(
    a0: usize,
    a1: usize,
    _a2: usize,
    _a3: usize,
    _a4: usize,
    _a5: usize,
    fid: usize,
    eid: usize,
) -> SbiRet {
    match eid {
        riscv_cove::host::EID_COVH => match fid {
            riscv_cove::host::GET_TSM_INFO => handle_get_tsm_info(a0, a1),
            _ => SbiRet::not_supported(),
        },
        _ => SbiRet::not_supported(),
    }
}

fn handle_get_tsm_info(buf_paddr: usize, buf_len: usize) -> SbiRet {
    if buf_paddr == 0 || buf_len < core::mem::size_of::<riscv_cove::host::TsmInfo>() {
        return SbiRet::invalid_param();
    }
    if buf_paddr % core::mem::align_of::<riscv_cove::host::TsmInfo>() != 0 {
        return SbiRet::invalid_param();
    }

    let tsm_info = riscv_cove::host::TsmInfo {
        tsm_state: riscv_cove::host::TsmState::Ready as u32,
        tsm_impl_id: TSM_IMPL_CUSTOM,
        tsm_version: TSM_VERSION,
        tsm_capabilities: 1 << riscv_cove::host::COVE_TSM_CAP_MEMORY_ALLOCATION,
        tvm_state_pages: 4,
        tvm_max_vcpus: 1,
        tvm_vcpu_state_pages: 2,
    };

    unsafe {
        core::ptr::write_volatile(buf_paddr as *mut riscv_cove::host::TsmInfo, tsm_info);
    }

    SbiRet::success(0)
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

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    println!("[TSM PANIC] {}", info);
    loop {
        unsafe {
            asm!("wfi");
        }
    }
}
