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

pub const TSM_READY: usize = 2;

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

#[unsafe(no_mangle)]
pub extern "C" fn tsm_main(_hart_id: usize, _fdt_paddr: usize) -> ! {
    println!("[TSM] Booting... TSM_READY");

    // Inform RDSM that TSM is ready and hand over control to Host
    rdsm_teeret(TSM_READY, 0, 0);
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
