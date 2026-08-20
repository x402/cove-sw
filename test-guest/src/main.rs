#![no_std]
#![no_main]

use core::arch::{asm, naked_asm};
use core::fmt::{self, Write};
use core::panic::PanicInfo;

struct GuestConsole;

impl Write for GuestConsole {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for b in s.bytes() {
            unsafe {
                asm!(
                    "li a7, 1",
                    "mv a0, {ch}",
                    "ecall",
                    ch = in(reg) b as usize,
                    lateout("a0") _,
                    lateout("a7") _,
                );
            }
        }
        Ok(())
    }
}

macro_rules! print {
    ($($arg:tt)*) => {
        let _ = core::fmt::write(&mut GuestConsole, format_args!($($arg)*));
    };
}

macro_rules! println {
    () => {
        print!("
");
    };
    ($($arg:tt)*) => {
        let _ = core::fmt::write(&mut GuestConsole, format_args!($($arg)*));
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
        // Call guest_main(hart_id, fdt_paddr)
        "call guest_main",
        // Should not return:
        "3:",
        "wfi",
        "j 3b"
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn guest_main(_hart_id: usize, _fdt_paddr: usize) -> ! {
    println!("[GUEST] Hello from Confidential TVM!");

    // Exit TVM via SRST (System Reset)
    unsafe {
        asm!(
            "li a7, 0x53525354",
            "li a6, 0",
            "li a0, 0",
            "li a1, 0",
            "ecall",
            options(noreturn)
        );
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    println!("[GUEST PANIC] {}", info);
    loop {
        unsafe {
            asm!("wfi");
        }
    }
}
