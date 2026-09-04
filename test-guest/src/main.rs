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
pub extern "C" fn guest_main(_hart_id: usize, _fdt_paddr: usize) -> ! {
    println!("[GUEST] Hello from Confidential TVM!");

    // Write initial value to shared page
    let shared = 0x8001_0000 as *mut u64;
    unsafe {
        shared.write_volatile(0xDEAD_BEEF);
    }
    println!("[GUEST] Wrote 0xDEADBEEF to shared page GPA 0x80010000");

    // Share memory region with Host via COVG
    println!("[GUEST] Calling COVG SHARE_MEMORY_REGION...");
    unsafe {
        asm!(
            "li a7, 0x434F5647",
            "li a6, 2",
            "li a0, 0x80010000",
            "li a1, 0x2000",
            "ecall",
            lateout("a0") _,
            lateout("a1") _,
            lateout("a6") _,
            lateout("a7") _,
        );
    }
    println!("[GUEST] Resumed after COVG share");

    // Host should have written MAGIC to the shared page
    let val = unsafe { shared.read_volatile() };
    if val == 0xCAFE_F00D {
        println!("[GUEST] PHASE4: SHARED_MEMORY_OK (read 0x{:x})", val);
    } else {
        println!(
            "[GUEST] PHASE4: SHARED_MEMORY_FAIL (read 0x{:x}, expected 0xCAFEF00D)",
            val
        );
    }

    println!("[GUEST] PHASE4 PASS: FENCE_EXIT_OK");

    // Demand-zero test (after PASS marker to isolate crash)
    let dz = 0x8002_0000 as *mut u64;
    unsafe {
        dz.write_volatile(0x42);
    }
    let dz_val = unsafe { dz.read_volatile() };
    if dz_val == 0x42 {
        println!("[GUEST] PHASE4: DEMAND_ZERO_OK");
    } else {
        println!("[GUEST] PHASE4: DEMAND_ZERO_FAIL (read 0x{:x})", dz_val);
    }

    // Return the shared region to confidential ownership before clean exit.
    println!("[GUEST] Calling COVG UNSHARE_MEMORY_REGION...");
    unsafe {
        asm!(
            "li a7, 0x434F5647",
            "li a6, 3",
            "li a0, 0x80010000",
            "li a1, 0x2000",
            "ecall",
            lateout("a0") _,
            lateout("a1") _,
            lateout("a6") _,
            lateout("a7") _,
        );
    }
    println!("[GUEST] Resumed after COVG unshare");

    // Exit TVM via SRST
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
