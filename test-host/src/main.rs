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

#[unsafe(no_mangle)]
pub extern "C" fn host_main(_hart_id: usize, _fdt_paddr: usize) -> ! {
    println!("[HOST] Booting... HOST_STARTED");

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
