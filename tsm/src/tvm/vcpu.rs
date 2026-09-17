use core::arch::global_asm;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ExitInfo {
    pub reason: usize,
    pub gpa: usize,
    pub len: usize,
}

impl ExitInfo {
    pub const fn new() -> Self {
        Self {
            reason: 0,
            gpa: 0,
            len: 0,
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct GuestContext {
    pub gprs: [usize; 32],
    pub sepc: usize,
    pub sstatus: usize,
    pub hstatus: usize,
    pub scause: usize,
    pub stval: usize,
    pub htval: usize,
    pub htinst: usize,
    pub hgatp: usize,
    /// HS-mode stack pointer saved across the guest run (offset 320 in the
    /// asm below). Per-hart state kept with the vCPU context instead of a
    /// shared global: each hart runs at most one vCPU at a time.
    pub host_sp: usize,
    /// Exit reason/state of the most recent guest exit (offsets 328..352 in
    /// the asm below). Written by the guest trap handler on the hart that
    /// ran the vCPU and read back by its run loop.
    pub last_exit: ExitInfo,
}

impl GuestContext {
    pub const fn new() -> Self {
        Self {
            gprs: [0; 32],
            sepc: 0,
            sstatus: 0x120, // SPP=1, SPIE=1
            hstatus: 0x180, // SPV=1, SPVP=1
            scause: 0,
            stval: 0,
            htval: 0,
            htinst: 0,
            hgatp: 0,
            host_sp: 0,
            last_exit: ExitInfo::new(),
        }
    }
}

pub struct Vcpu {
    pub id: usize,
    pub state_paddr: usize,
    pub ctx: GuestContext,
}

impl Vcpu {
    pub fn new(id: usize, state_paddr: usize) -> Self {
        Self {
            id,
            state_paddr,
            ctx: GuestContext::new(),
        }
    }

    pub fn init_boot(&mut self, entry_sepc: usize, entry_arg: usize, hgatp: usize) {
        self.ctx.sepc = entry_sepc;
        self.ctx.gprs[10] = self.id; // a0 = hart_id / vcpu_id
        self.ctx.gprs[11] = entry_arg; // a1 = fdt / arg
        self.ctx.sstatus = 0x120; // SPP=1, SPIE=1
        self.ctx.hstatus = 0x180; // SPV=1, SPVP=1
        self.ctx.hgatp = hgatp;
    }

    pub fn run(&mut self) -> usize {
        unsafe { tsm_enter_guest(&mut self.ctx as *mut GuestContext) }
    }
}

unsafe extern "C" {
    pub fn tsm_enter_guest(ctx: *mut GuestContext) -> usize;
}

pub const TRAP_ACTION_RESUME: usize = 0;
pub const TRAP_ACTION_EXIT: usize = 1;

// Exit reasons reported to Host via sbi_covh_run_tvm_vcpu return value
pub const EXIT_CLEAN: usize = 0;
pub const EXIT_COVG_SHARE: usize = 1;
pub const EXIT_COVG_UNSHARE: usize = 2;
pub const EXIT_COVG_ADD_MMIO: usize = 3;
pub const EXIT_COVG_REMOVE_MMIO: usize = 4;
pub const EXIT_UNEXPECTED_TRAP: usize = 5;

const EID_COVG: usize = 0x434F5647;

#[unsafe(no_mangle)]
pub extern "C" fn handle_guest_trap(ctx: *mut GuestContext) -> usize {
    let ctx_ref = unsafe { &mut *ctx };
    let scause = ctx_ref.scause;

    if scause == 10 {
        // Virtual supervisor ecall
        let eid = ctx_ref.gprs[17]; // a7
        let fid = ctx_ref.gprs[16]; // a6

        if eid == 1 {
            // SBI legacy console_putchar
            let ch = ctx_ref.gprs[10] as u8; // a0
            #[allow(deprecated)]
            let _ = sbi_rt::console_write_byte(ch);
            ctx_ref.sepc += 4;
            return TRAP_ACTION_RESUME;
        }

        if eid == 0x53525354 || eid == 0x08 {
            // Guest clean exit via SRST / shutdown (NOT COVG - handled below)
            ctx_ref.sepc += 4;
            ctx_ref.last_exit.reason = EXIT_CLEAN;
            return TRAP_ACTION_EXIT;
        }

        if eid == EID_COVG {
            let gpa = ctx_ref.gprs[10]; // a0
            let len = ctx_ref.gprs[11]; // a1

            match fid {
                0 => {
                    // ADD_MMIO_REGION
                    crate::tvm::tvm_manager().add_mmio_region_internal(gpa, len);
                    ctx_ref.sepc += 4;
                    let e = &mut ctx_ref.last_exit;
                    e.reason = EXIT_COVG_ADD_MMIO;
                    e.gpa = gpa;
                    e.len = len;
                    return TRAP_ACTION_EXIT;
                }
                1 => {
                    // REMOVE_MMIO_REGION
                    crate::tvm::tvm_manager().remove_mmio_region_internal(gpa, len);
                    ctx_ref.sepc += 4;
                    let e = &mut ctx_ref.last_exit;
                    e.reason = EXIT_COVG_REMOVE_MMIO;
                    e.gpa = gpa;
                    e.len = len;
                    return TRAP_ACTION_EXIT;
                }
                2 => {
                    // SHARE_MEMORY_REGION: unmap G-stage, grant Host MPT access
                    let ok = crate::tvm::tvm_manager().share_memory_internal(gpa, len);
                    ctx_ref.sepc += 4;
                    let e = &mut ctx_ref.last_exit;
                    e.reason = EXIT_COVG_SHARE;
                    e.gpa = gpa;
                    e.len = len;
                    let _ = ok; // Even if partial, still exit so host can inspect
                    return TRAP_ACTION_EXIT;
                }
                3 => {
                    // UNSHARE_MEMORY_REGION
                    crate::tvm::tvm_manager().unshare_memory_internal(gpa, len);
                    ctx_ref.sepc += 4;
                    let e = &mut ctx_ref.last_exit;
                    e.reason = EXIT_COVG_UNSHARE;
                    e.gpa = gpa;
                    e.len = len;
                    return TRAP_ACTION_EXIT;
                }
                _ => {
                    crate::println!("[TSM] Unhandled COVG FID={}", fid);
                    ctx_ref.sepc += 4;
                    ctx_ref.last_exit.reason = EXIT_UNEXPECTED_TRAP;
                    return TRAP_ACTION_EXIT;
                }
            }
        }

        crate::println!(
            "[TSM] Unhandled guest ECALL: eid=0x{:x}, fid=0x{:x}",
            eid,
            fid
        );
        ctx_ref.sepc += 4;
        ctx_ref.last_exit.reason = EXIT_UNEXPECTED_TRAP;
        return TRAP_ACTION_EXIT;
    }

    // Guest page/access faults:
    // scause 13: Guest load page fault
    // scause 15: Guest store/AMO page fault
    // scause 22: Guest load access fault (MPT denied at G-stage)
    // scause 23: Guest store/AMO access fault (MPT denied at G-stage)
    if scause == 13 || scause == 15 || scause == 22 || scause == 23 {
        let fault_gpa = ctx_ref.htval << 2;
        let handled = crate::tvm::tvm_manager().handle_demand_zero(fault_gpa);
        if handled {
            return TRAP_ACTION_RESUME;
        }
        crate::println!(
            "[TSM] Unhandled guest page fault: gpa=0x{:x}, scause={}",
            fault_gpa,
            scause
        );
        ctx_ref.last_exit.reason = EXIT_UNEXPECTED_TRAP;
        return TRAP_ACTION_EXIT;
    }

    crate::println!(
        "[TSM] Guest trap: scause=0x{:x}, sepc=0x{:x}, stval=0x{:x}, htval=0x{:x}",
        scause,
        ctx_ref.sepc,
        ctx_ref.stval,
        ctx_ref.htval
    );
    ctx_ref.last_exit.reason = EXIT_UNEXPECTED_TRAP;
    return TRAP_ACTION_EXIT;
}

global_asm!(
    r#"
.section .text
.global tsm_enter_guest
.global tsm_guest_trap_vector
.global tsm_exit_guest_restore

tsm_enter_guest:
    addi sp, sp, -128
    sd ra, 0(sp)
    sd s0, 8(sp)
    sd s1, 16(sp)
    sd s2, 24(sp)
    sd s3, 32(sp)
    sd s4, 40(sp)
    sd s5, 48(sp)
    sd s6, 56(sp)
    sd s7, 64(sp)
    sd s8, 72(sp)
    sd s9, 80(sp)
    sd s10, 88(sp)
    sd s11, 96(sp)
    
    csrr t0, stvec
    sd t0, 104(sp)
    csrr t1, sstatus
    sd t1, 112(sp)
    csrr t2, hstatus
    sd t2, 120(sp)

    sd sp, 320(a0)

    ld t0, 312(a0)
    csrw hgatp, t0
    hfence.gvma

    la t0, tsm_guest_trap_vector
    csrw stvec, t0

    csrw sscratch, a0

    ld t0, 272(a0)
    csrw hstatus, t0

    ld t0, 264(a0)
    csrw sstatus, t0

    ld t0, 256(a0)
    csrw sepc, t0

    ld ra, 8(a0)
    ld sp, 16(a0)
    ld gp, 24(a0)
    ld tp, 32(a0)
    ld t0, 40(a0)
    ld t1, 48(a0)
    ld t2, 56(a0)
    ld s0, 64(a0)
    ld s1, 72(a0)
    ld a1, 88(a0)
    ld a2, 96(a0)
    ld a3, 104(a0)
    ld a4, 112(a0)
    ld a5, 120(a0)
    ld a6, 128(a0)
    ld a7, 136(a0)
    ld s2, 144(a0)
    ld s3, 152(a0)
    ld s4, 160(a0)
    ld s5, 168(a0)
    ld s6, 176(a0)
    ld s7, 184(a0)
    ld s8, 192(a0)
    ld s9, 200(a0)
    ld s10, 208(a0)
    ld s11, 216(a0)
    ld t3, 224(a0)
    ld t4, 232(a0)
    ld t5, 240(a0)
    ld t6, 248(a0)
    ld a0, 80(a0)

    sret

.align 4
tsm_guest_trap_vector:
    csrrw a0, sscratch, a0

    sd ra, 8(a0)
    sd sp, 16(a0)
    sd gp, 24(a0)
    sd tp, 32(a0)
    sd t0, 40(a0)
    sd t1, 48(a0)
    sd t2, 56(a0)
    sd s0, 64(a0)
    sd s1, 72(a0)

    csrr t0, sscratch
    sd t0, 80(a0)

    sd a1, 88(a0)
    sd a2, 96(a0)
    sd a3, 104(a0)
    sd a4, 112(a0)
    sd a5, 120(a0)
    sd a6, 128(a0)
    sd a7, 136(a0)
    sd s2, 144(a0)
    sd s3, 152(a0)
    sd s4, 160(a0)
    sd s5, 168(a0)
    sd s6, 176(a0)
    sd s7, 184(a0)
    sd s8, 192(a0)
    sd s9, 200(a0)
    sd s10, 208(a0)
    sd s11, 216(a0)
    ld t3, 224(a0)
    ld t4, 232(a0)
    ld t5, 240(a0)
    ld t6, 248(a0)

    csrr t0, sepc
    sd t0, 256(a0)
    csrr t1, sstatus
    sd t1, 264(a0)
    csrr t2, hstatus
    sd t2, 272(a0)
    csrr t3, scause
    sd t3, 280(a0)
    csrr t4, stval
    sd t4, 288(a0)
    csrr t5, htval
    sd t5, 296(a0)
    csrr t6, htinst
    sd t6, 304(a0)

    ld sp, 320(a0)

    csrw sscratch, a0

    call handle_guest_trap

    bnez a0, tsm_exit_guest_restore

    csrr a0, sscratch

    ld t0, 256(a0)
    csrw sepc, t0
    ld t1, 264(a0)
    csrw sstatus, t1
    ld t2, 272(a0)
    csrw hstatus, t2

    ld ra, 8(a0)
    ld sp, 16(a0)
    ld gp, 24(a0)
    ld tp, 32(a0)
    ld t0, 40(a0)
    ld t1, 48(a0)
    ld t2, 56(a0)
    ld s0, 64(a0)
    ld s1, 72(a0)
    ld a1, 88(a0)
    ld a2, 96(a0)
    ld a3, 104(a0)
    ld a4, 112(a0)
    ld a5, 120(a0)
    ld a6, 128(a0)
    ld a7, 136(a0)
    ld s2, 144(a0)
    ld s3, 152(a0)
    ld s4, 160(a0)
    ld s5, 168(a0)
    ld s6, 176(a0)
    ld s7, 184(a0)
    ld s8, 192(a0)
    ld s9, 200(a0)
    ld s10, 208(a0)
    ld s11, 216(a0)
    ld t3, 224(a0)
    ld t4, 232(a0)
    ld t5, 240(a0)
    ld t6, 248(a0)
    ld a0, 80(a0)

    sret

tsm_exit_guest_restore:
    csrr t0, sscratch
    ld sp, 320(t0)

    csrw hgatp, zero
    hfence.gvma

    ld t0, 104(sp)
    csrw stvec, t0
    ld t1, 112(sp)
    csrw sstatus, t1
    ld t2, 120(sp)
    csrw hstatus, t2
    csrw sscratch, zero

    ld ra, 0(sp)
    ld s0, 8(sp)
    ld s1, 16(sp)
    ld s2, 24(sp)
    ld s3, 32(sp)
    ld s4, 40(sp)
    ld s5, 48(sp)
    ld s6, 56(sp)
    ld s7, 64(sp)
    ld s8, 72(sp)
    ld s9, 80(sp)
    ld s10, 88(sp)
    ld s11, 96(sp)
    addi sp, sp, 128

    li a0, 0
    ret
"#
);
