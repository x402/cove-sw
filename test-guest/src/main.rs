#![no_std]
#![no_main]

use core::arch::{asm, naked_asm};
use core::panic::PanicInfo;

#[unsafe(naked)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.entry")]
pub unsafe extern "C" fn _start() -> ! {
    naked_asm!(
        "la sp, boot_stack_top",
        "la t0, sbss",
        "la t1, ebss",
        "1:",
        "bgeu t0, t1, 2f",
        "sd zero, 0(t0)",
        "addi t0, t0, 8",
        "j 1b",
        "2:",
        "call guest_main",
        "3:",
        "wfi",
        "j 3b"
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn guest_main() -> ! {
    loop {
        unsafe {
            asm!("wfi");
        }
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {
        unsafe {
            asm!("wfi");
        }
    }
}
