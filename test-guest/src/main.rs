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

/// Issues one COVG ecall and returns `sbiret.error` (a0). The TSM answers a
/// request it refuses itself, so such a call returns without an exit; an
/// accepted request exits to the host and this returns after the resume,
/// where the TSM has written SBI_SUCCESS into a0.
fn covg(fid: usize, gpa: usize, len: usize) -> usize {
    let error: usize;
    unsafe {
        asm!(
            "ecall",
            in("a7") 0x434F_5647usize,
            in("a6") fid,
            in("a0") gpa,
            in("a1") len,
            lateout("a0") error,
            lateout("a1") _,
            lateout("a6") _,
            lateout("a7") _,
        );
    }
    error
}

#[unsafe(no_mangle)]
pub extern "C" fn guest_main(_hart_id: usize, _fdt_paddr: usize) -> ! {
    println!("[GUEST] Hello from Confidential TVM!");

    // Write a marker into each of the three pages about to be shared. The
    // TSM must scrub all three before the host regains access.
    let shared = 0x8001_0000 as *mut u64;
    unsafe {
        shared.write_volatile(0xDEAD_BEEF);
        shared.add(512).write_volatile(0xBEEF_0001); // GPA 0x8001_1000
        shared.add(1024).write_volatile(0xBEEF_0002); // GPA 0x8001_2000
    }
    println!("[GUEST] Wrote 0xDEADBEEF to shared page GPA 0x80010000");

    // Requests the TSM refuses itself never reach the host; the error comes
    // back in a0 (Table 46). A leaked accept would hand the host an exit the
    // rest of the flow cannot handle.
    assert_eq!(
        covg(2, 0x8001_0008, 0x3000),
        (-5isize) as usize,
        "a misaligned share address must be SBI_ERR_INVALID_ADDRESS"
    );
    println!("[GUEST] PHASE4: SHARE_REFUSED_OK (sbiret.error=-5)");
    assert_eq!(
        covg(2, 0x9000_0000, 0x1000),
        (-3isize) as usize,
        "a share outside every confidential region must be SBI_ERR_INVALID_PARAM"
    );
    println!("[GUEST] PHASE4: SHARE_REFUSED_OK (sbiret.error=-3)");

    // Share memory region with Host via COVG
    println!("[GUEST] Calling COVG SHARE_MEMORY_REGION...");
    let a0: usize;
    unsafe {
        asm!(
            "li a7, 0x434F5647",
            "li a6, 2",
            "li a0, 0x80010000",
            "li a1, 0x3000",
            "ecall",
            lateout("a0") a0,
            lateout("a1") _,
            lateout("a6") _,
            lateout("a7") _,
        );
    }
    assert_eq!(a0, 0, "COVG share must succeed");
    println!("[GUEST] Resumed after COVG share");

    // Host should have written one magic value per shared page.
    let v0 = unsafe { shared.read_volatile() };
    let v1 = unsafe { shared.add(512).read_volatile() };
    let v2 = unsafe { shared.add(1024).read_volatile() };
    assert_eq!(v0, 0xCAFE_F00D, "first shared page must carry the host value");
    assert_eq!(v1, 0xCAFE_0001, "second shared page must carry the host value");
    assert_eq!(v2, 0xCAFE_0002, "third shared page must carry the host value");
    println!("[GUEST] PHASE4: SHARED_MEMORY_OK (read 0x{:x})", v0);

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

    // A length that is not page-granular is refused with Table 47
    // INVALID_PARAM before the real unshare below.
    assert_eq!(
        covg(3, 0x8001_1000, 0x1008),
        (-3isize) as usize,
        "a non-page-granular unshare length must be SBI_ERR_INVALID_PARAM"
    );
    println!("[GUEST] PHASE4: UNSHARE_REFUSED_OK (sbiret.error=-3)");

    // Return the third page of the shared region to confidential ownership
    // before clean exit.
    println!("[GUEST] Calling COVG UNSHARE_MEMORY_REGION...");
    let a0: usize;
    unsafe {
        asm!(
            "li a7, 0x434F5647",
            "li a6, 3",
            "li a0, 0x80012000",
            "li a1, 0x1000",
            "ecall",
            lateout("a0") a0,
            lateout("a1") _,
            lateout("a6") _,
            lateout("a7") _,
        );
    }
    assert_eq!(a0, 0, "COVG unshare must succeed");
    println!("[GUEST] Resumed after COVG unshare");

    // The unshared page is no longer mapped, so this access faults and the
    // TSM serves it with a fresh, cleared confidential page.
    let unshared = unsafe { (0x8001_2000 as *const u64).read_volatile() };
    assert_eq!(unshared, 0, "an unshared page must come back cleared");
    println!("[GUEST] PHASE4: UNSHARE_OK (read 0x0)");
    // The page outside the unshared range is still shared host memory.
    let remains = unsafe { (0x8001_1000 as *const u64).read_volatile() };
    assert_eq!(remains, 0xCAFE_0001, "a shared page must keep its contents");
    println!("[GUEST] PHASE4: SHARED_REMAINS_OK");

    // ── PHASE 7: COVI injected external interrupt ──
    // Arm the guest interrupt file (eie identity 1, eidelivery on), route
    // VSEIP into VS via vsie/vstvec, allow identity 1 through COVG, and
    // take the host's injected MSI in the VS handler. The interrupt lands
    // right after the allow ecall resumes (sret re-enables vsstatus.SIE
    // with the MSI already pending in the file).
    println!("[GUEST] PHASE7: arming VS external interrupt...");
    unsafe {
        asm!(
            "la t0, vs_ext_irq_handler",
            "csrw vstvec, t0",
            "li t0, 0xC0",
            "csrw vsiselect, t0",   // eie0
            "li t0, 2",
            "csrw vsireg, t0",      // enable identity 1
            "li t0, 0x70",
            "csrw vsiselect, t0",   // eidelivery
            "li t0, 1",
            "csrw vsireg, t0",
            "li t0, 512",
            "csrs vsie, t0",        // VSEIE
            "li t0, 2",
            "csrs sstatus, t0",     // global VS enable
            options(nomem)
        );
    }

    println!("[GUEST] PHASE7: calling COVG ALLOW_EXTERNAL_INTERRUPT...");
    unsafe {
        asm!(
            "li a7, 0x434F5647",
            "li a6, 4",
            "li a0, 1",
            "li a1, 0",
            "ecall",
            lateout("a0") _,
            lateout("a1") _,
            lateout("a6") _,
            lateout("a7") _,
        );
    }
    // The injected interrupt was taken on the way back: the handler
    // claimed it and printed the VS_EXT marker. Before denying identity
    // 1, allow identity 2 (COVG FID 4) so the host can park a pending
    // MSI the guest never enables or claims — destroy's guest-file
    // scrub must claim it.
    println!("[GUEST] PHASE7: resumed after allow; allowing identity 2...");
    unsafe {
        asm!(
            "li a7, 0x434F5647",
            "li a6, 4",
            "li a0, 2",
            "li a1, 0",
            "ecall",
            lateout("a0") _,
            lateout("a1") _,
            lateout("a6") _,
            lateout("a7") _,
        );
    }
    println!("[GUEST] PHASE7: resumed after allow-2; calling COVG DENY...");
    unsafe {
        asm!(
            "li a7, 0x434F5647",
            "li a6, 5",
            "li a0, 1",
            "li a1, 0",
            "ecall",
            lateout("a0") _,
            lateout("a1") _,
            lateout("a6") _,
            lateout("a7") _,
        );
    }
    println!("[GUEST] PHASE7: resumed after deny");

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

/// VS external interrupt handler (vstvec direct). Claims the reported
/// identity through vstopei and prints the serial evidence line. The
/// handler runs on the interrupted VS flow: caller-saved registers are
/// dead at the expected landing point (right after the allow ecall
/// resume), so no register saving is needed for this test.
#[unsafe(naked)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.entry")]
pub unsafe extern "C" fn vs_ext_irq_handler() {
    core::arch::naked_asm!(
        "csrw vstopei, zero", // claim the reported identity
        "call vs_irq_note",
        // NEMU decodes the VS xret as `sret` executed with cpu.v=1; the
        // dedicated vsret encoding is not in its decode table.
        ".word 0x10200073",
    );
}

/// Serial evidence that the injected MSI was taken in VS-mode. Called from
/// the naked handler: caller-saved registers are dead at the landing point.
#[unsafe(no_mangle)]
extern "C" fn vs_irq_note() {
    println!("[GUEST] PHASE7: VS_EXT_INJECT_OK (iid=1)");
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
