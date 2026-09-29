#![no_std]
#![no_main]

extern crate alloc;

pub mod mm;
pub mod rdsm_shim;
pub mod tvm;

use core::arch::{asm, naked_asm};
use core::fmt::{self, Write};
use core::panic::PanicInfo;
use core::sync::atomic::{AtomicUsize, Ordering};
use mm::page_tracker;
use rdsm_shim::{NORMAL_RETURN, TSM_READY, rdsm_mfence_pa, rdsm_teeret};
use tvm::{run_tvm_vcpu, tvm_manager};

pub const TSM_IMPL_CUSTOM: u32 = 0x54534D31; // "TSM1"
pub const TSM_VERSION: u32 = 1;

pub struct SbiConsole;

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

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {
        let _ = core::fmt::write(&mut $crate::SbiConsole, format_args!($($arg)*));
    };
}

#[macro_export]
macro_rules! println {
    () => {
        $crate::print!("
");
    };
    ($($arg:tt)*) => {
        let _ = core::fmt::write(&mut $crate::SbiConsole, format_args!($($arg)*));
        $crate::print!("
");
    };
}

struct BumpAlloc {
    heap: [u8; 64 * 1024],
    next: AtomicUsize,
}

unsafe impl core::alloc::GlobalAlloc for BumpAlloc {
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        let align = layout.align();
        let size = layout.size();
        // CAS loop: a plain load/store pair would let two harts both
        // succeed on the same bump offset.
        let mut current = self.next.load(Ordering::Relaxed);
        loop {
            let aligned = (current + align - 1) & !(align - 1);
            if aligned + size > self.heap.len() {
                return core::ptr::null_mut();
            }
            match self.next.compare_exchange_weak(
                current,
                aligned + size,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return unsafe { self.heap.as_ptr().add(aligned) as *mut u8 },
                Err(next) => current = next,
            }
        }
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: core::alloc::Layout) {}
}

#[global_allocator]
static ALLOCATOR: BumpAlloc = BumpAlloc {
    heap: [0; 64 * 1024],
    next: AtomicUsize::new(0),
};

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

/// Maximum harts the TSM keeps dispatch stacks for (matches rdsm-mech's
/// `NUM_HARTS_MAX` convention).
pub const NUM_HARTS_MAX: usize = 8;

/// 64 KiB per hart, same size as the boot stack.
const DISPATCH_STACK_SIZE: usize = 0x10000;

/// Per-hart dispatch stacks: concurrent harts must not share
/// one stack, since a dispatch may span the whole vCPU run loop.
#[repr(C, align(16))]
struct DispatchStacks([u8; DISPATCH_STACK_SIZE * NUM_HARTS_MAX]);

#[unsafe(no_mangle)]
static mut DISPATCH_STACKS: DispatchStacks =
    DispatchStacks([0; DISPATCH_STACK_SIZE * NUM_HARTS_MAX]);

#[unsafe(naked)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tsm_dispatch_entry() -> ! {
    naked_asm!(
        // Stack indexed by the hart id kept in `scontext` (0x5a8) since
        // TSM boot; `mhartid` is not readable in HS-mode.
        "csrr t0, 0x5a8",
        "la t1, DISPATCH_STACKS",
        "slli t0, t0, 16",
        "add sp, t1, t0",
        "call tsm_dispatch",
        // tsm_dispatch returns (error, value) in a0, a1
        // Forward back to RDSM: rdsm_teeret(NORMAL_RETURN, error, value)
        "mv a2, a1",
        "mv a1, a0",
        "li a0, 0",          // NORMAL_RETURN
        "li a7, 0x5244534D", // EID_RDSM
        "li a6, 3",          // FID_RDSM_TEERET
        "ecall",
        "j tsm_dispatch_entry"
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn tsm_main(hart_id: usize, _fdt_paddr: usize) -> ! {
    // Keep the hart id in `scontext` (0x5a8), the privileged spec's
    // recommended hart-id scratch. RDSM saves/restores S-mode CSRs across
    // domain switches, so the value survives TEECALL/TEERET and gives the
    // HS-mode dispatch entry its per-hart stack index (`mhartid` itself is
    // not readable in HS-mode).
    unsafe {
        asm!("csrw 0x5a8, {}", in(reg) hart_id);
    }

    // Obtain the platform reserved-region layout from RDSM and
    // build the host-allocatable whitelist. Fail closed on error: without
    // a known memory map, every host-supplied address would be rejected.
    match rdsm_shim::rdsm_get_platform_info() {
        Some(info) => mm::init_host_regions(&info),
        None => {
            println!("[TSM PANIC] RDSM_GET_INFO failed, refusing to start");
            loop {
                unsafe {
                    asm!("wfi");
                }
            }
        }
    }

    println!("[TSM] Booting... TSM_READY");
    println!("[MARKER 02] TSM: Initialization complete, state=TSM_READY.");

    // Inform RDSM that TSM is ready and provide dispatch entry point.
    // The MARKER 03 line is printed here (just before the TEERET that makes
    // RDSM switch to the host domain): emitting UART output from inside the
    // RDSM entire handler mid-domain-switch proved unstable, so the TSM
    // emits the line on RDSM's behalf. Chronology for the E2E script is
    // identical: marker 02 < marker 03 < host's marker 04.
    println!("[MARKER 03] RDSM: Switching context to Host Domain (SDID=0).");
    let (err, _val) = rdsm_teeret(TSM_READY, tsm_dispatch_entry as *const () as usize, 0);
    // INTERIM (upstream rustsbi#286 pending): a returning TEERET means the
    // retentive switch was rejected; park this hart instead of falling off
    // the end of the ecall shim.
    println!(
        "[RDSM-INTERIM] TEERET rejected (err={:#x}); parking hart (rustsbi#286 pending).",
        err
    );
    loop {
        // SAFETY: `wfi` is a privileged hint instruction with no memory effect.
        unsafe { core::arch::asm!("wfi") };
    }
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
    pub const fn failed() -> Self {
        Self {
            error: (-1isize) as usize,
            value: 0,
        }
    }
    pub const fn not_supported() -> Self {
        Self {
            error: (-2isize) as usize,
            value: 0,
        }
    }
    pub const fn invalid_param() -> Self {
        Self {
            error: (-3isize) as usize,
            value: 0,
        }
    }
    pub const fn invalid_address() -> Self {
        Self {
            error: (-5isize) as usize,
            value: 0,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tsm_dispatch(
    a0: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
    fid: usize,
    eid: usize,
) -> SbiRet {
    match eid {
        riscv_cove::host::EID_COVH => match fid {
            riscv_cove::host::GET_TSM_INFO => handle_get_tsm_info(a0, a1),
            riscv_cove::host::CONVERT_PAGES => page_tracker().convert_pages(a0, a1),
            riscv_cove::host::RECLAIM_PAGES => page_tracker().reclaim_pages(a0, a1),
            riscv_cove::host::GLOBAL_FENCE => {
                let _ = rdsm_mfence_pa(0, 0);
                SbiRet::success(0)
            }
            riscv_cove::host::LOCAL_FENCE => {
                unsafe {
                    asm!("sfence.vma", "hfence.gvma");
                }
                SbiRet::success(0)
            }
            riscv_cove::host::CREATE_TVM => tvm_manager().create_tvm(a0, a1),
            riscv_cove::host::FINALIZE_TVM => tvm_manager().finalize_tvm(a0, a1, a2, a3),
            riscv_cove::host::DESTROY_TVM => tvm_manager().destroy_tvm(a0),
            riscv_cove::host::ADD_TVM_MEMORY_REGION => tvm_manager().add_memory_region(a0, a1, a2),
            riscv_cove::host::ADD_TVM_PAGE_TABLE_PAGES => {
                tvm_manager().add_page_table_pages(a0, a1, a2)
            }
            riscv_cove::host::ADD_TVM_MEASURED_PAGES => {
                tvm_manager().add_measured_pages(a0, a1, a2, a3, a4, a5)
            }
            riscv_cove::host::ADD_TVM_ZERO_PAGES => {
                tvm_manager().add_zero_pages(a0, a1, a2, a3, a4)
            }
            riscv_cove::host::CREATE_TVM_VCPU => tvm_manager().create_tvm_vcpu(a0, a1, a2),
            riscv_cove::host::RUN_TVM_VCPU => run_tvm_vcpu(a0, a1),
            riscv_cove::host::ADD_TVM_SHARED_PAGES => {
                tvm_manager().add_shared_pages(a0, a1, a2, a3, a4)
            }
            riscv_cove::host::TVM_FENCE => tvm_manager().tvm_fence(a0),
            riscv_cove::host::TVM_INVALIDATE_PAGES => {
                tvm_manager().tvm_invalidate_pages(a0, a1, a2)
            }
            riscv_cove::host::TVM_VALIDATE_PAGES => tvm_manager().tvm_validate_pages(a0, a1, a2),
            riscv_cove::host::TVM_REMOVE_PAGES => tvm_manager().tvm_remove_pages(a0, a1, a2),
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
        return SbiRet::invalid_address();
    }
    // The result buffer must live in host-allocatable memory.
    if !mm::is_host_range(
        buf_paddr,
        core::mem::size_of::<riscv_cove::host::TsmInfo>(),
    ) {
        return SbiRet::invalid_address();
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

    SbiRet::success(core::mem::size_of::<riscv_cove::host::TsmInfo>())
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
