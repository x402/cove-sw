//! RDSM mechanism layer.
//!
//! Unsafe operations of the M-mode RDSM firmware live here: raw-asm CSR
//! save/restore of supervisor-domain state, `mmpt`/interrupt-domain
//! programming, physical image loading, and the per-hart storage primitives.
//! Every public function is safe; the unsafe is confined inside this crate
//! with its precondition documented at each site. Policy and sequencing
//! belong to `rdsm-policy`, which is the only intended consumer.
//!
//! This crate never encodes SBI results and never touches the embedding
//! Runtime's private trap state (`mscratch`, the trap frame): the GPR, resume
//! PC and `satp` halves of a domain switch are carried by the Runtime's
//! declarative execution contexts (`rustsbi-firmware-runtime::context`);
//! only the CSR half ([`DomainCsrs`]) is saved/restored here.

#![no_std]

use core::cell::UnsafeCell;

use rdsm::mpt::MptPageAlloc;
use rdsm_abi::{COVE_PAYLOAD_MAGIC, PayloadHeader, RdsmPlatformInfo};

// ── Per-hart storage ───────────────────────────────────────────────────

/// Maximum number of harts supported by the per-hart RDSM state.
///
/// Mirrors the RustSBI Runtime's `cfg::NUM_HART_MAX` convention
/// (platform configs currently allow up to 8).
pub const NUM_HARTS_MAX: usize = 8;

/// Hardware hart ID of the calling hart (M-mode only; the per-hart slots
/// are only accessed from M-mode firmware code).
#[inline]
pub fn current_hart_id() -> usize {
    #[cfg(target_arch = "riscv64")]
    {
        riscv::register::mhartid::read()
    }
    #[cfg(not(target_arch = "riscv64"))]
    {
        0
    }
}

/// Per-hart storage cell.
///
/// Each hart only ever touches its own slot (indexed by [`current_hart_id`]);
/// concurrent access to one slot from multiple harts is a contract violation
/// and a data race, exactly as for the per-hart arrays this primitive
/// replaces. This is the standard "indexed by hart id, no locking" scheme
/// used by the RustSBI Runtime itself.
pub struct PerHart<T: Copy> {
    slots: UnsafeCell<[T; NUM_HARTS_MAX]>,
}

// SAFETY: slots are accessed only through `get`/`get_mut`, which hand out
// references tied to the calling hart's slot. Distinct harts use distinct
// slots; same-slot aliasing across harts is excluded by the contract above.
unsafe impl<T: Copy> Sync for PerHart<T> {}

impl<T: Copy> PerHart<T> {
    /// Creates a cell whose every slot holds `value`.
    pub const fn new(value: T) -> Self {
        Self {
            slots: UnsafeCell::new([value; NUM_HARTS_MAX]),
        }
    }

    /// Returns the calling hart's slot.
    ///
    /// # Panics
    ///
    /// Panics if the hart id is out of range (`>= [`NUM_HARTS_MAX`]`).
    #[inline]
    pub fn get(&self) -> &T {
        let hart = current_hart_id();
        assert!(hart < NUM_HARTS_MAX, "hart id out of range");
        unsafe { &(*self.slots.get())[hart] }
    }

    /// Returns the calling hart's slot for mutation.
    ///
    /// # Panics
    ///
    /// Panics if the hart id is out of range (`>= [`NUM_HARTS_MAX`]`).
    #[inline]
    pub fn get_mut(&self) -> &mut T {
        let hart = current_hart_id();
        assert!(hart < NUM_HARTS_MAX, "hart id out of range");
        unsafe { &mut (*self.slots.get())[hart] }
    }
}

/// A const constructor for [`PerHartCell`] slot values.
///
/// `Default::default` cannot seed a `static` (trait methods are not
/// `const`), so a cell value declares its initial value as an associated
/// constant instead.
pub trait ConstInit {
    /// The value every slot starts with.
    const INIT: Self;
}

/// Per-hart storage cell for values that are not [`Copy`](core::marker::Copy)
/// (for example the Runtime's execution contexts).
///
/// Contract identical to [`PerHart`]: each hart only ever touches its own
/// slot (indexed by [`current_hart_id`]); same-slot access from multiple
/// harts is a contract violation and a data race.
pub struct PerHartCell<T: ConstInit> {
    slots: UnsafeCell<[T; NUM_HARTS_MAX]>,
}

// SAFETY: slots are accessed only through `get`/`get_mut`, which hand out
// references tied to the calling hart's slot. Distinct harts use distinct
// slots; same-slot aliasing across harts is excluded by the contract above.
unsafe impl<T: ConstInit> Sync for PerHartCell<T> {}

impl<T: ConstInit> PerHartCell<T> {
    /// Creates a cell whose every slot holds [`ConstInit::INIT`].
    pub const fn new() -> Self {
        Self {
            slots: UnsafeCell::new([T::INIT; NUM_HARTS_MAX]),
        }
    }

    /// Returns the calling hart's slot.
    ///
    /// # Panics
    ///
    /// Panics if the hart id is out of range (`>= [`NUM_HARTS_MAX`]`).
    #[inline]
    pub fn get(&self) -> &T {
        let hart = current_hart_id();
        assert!(hart < NUM_HARTS_MAX, "hart id out of range");
        unsafe { &(*self.slots.get())[hart] }
    }

    /// Returns the calling hart's slot for mutation.
    ///
    /// # Panics
    ///
    /// Panics if the hart id is out of range (`>= [`NUM_HARTS_MAX`]`).
    #[inline]
    pub fn get_mut(&self) -> &mut T {
        let hart = current_hart_id();
        assert!(hart < NUM_HARTS_MAX, "hart id out of range");
        unsafe { &mut (*self.slots.get())[hart] }
    }
}

// ── Domain contexts ────────────────────────────────────────────────────

/// The S-mode, HS-mode and VS-mode CSR half of a supervisor domain.
///
/// Plain data saved/restored by the monitor adjacent to a retentive domain
/// switch. GPRs, the resume PC and `satp` are deliberately absent: they are
/// carried by the Runtime's declarative execution contexts, whose transfer
/// ceremony reads the outgoing `satp` before its own entry reset and
/// installs the incoming one — a `satp` field here would record the
/// incoming domain's value, not the outgoing one.
#[derive(Clone, Copy, Debug)]
pub struct DomainCsrs {
    // S-mode CSRs
    pub sstatus: usize,
    pub stvec: usize,
    pub sip: usize,
    pub sie: usize,
    pub scounteren: usize,
    pub sscratch: usize,
    pub senvcfg: usize,
    pub scontext: usize,

    // HS-mode (hypervisor) CSRs
    pub hstatus: usize,
    pub hgatp: usize,
    pub hedeleg: usize,
    pub hideleg: usize,
    pub hvip: usize,
    pub henvcfg: usize,
    pub hcounteren: usize,

    // VS-mode CSRs
    pub vsstatus: usize,
    pub vsie: usize,
    pub vstvec: usize,
    pub vsscratch: usize,
    pub vsepc: usize,
    pub vscause: usize,
    pub vstval: usize,
    pub vsip: usize,
    pub vsatp: usize,
}

impl DomainCsrs {
    pub const fn new() -> Self {
        Self {
            sstatus: 0,
            stvec: 0,
            sip: 0,
            sie: 0,
            scounteren: 0,
            sscratch: 0,
            senvcfg: 0,
            scontext: 0,
            hstatus: 0,
            hgatp: 0,
            hedeleg: 0,
            hideleg: 0,
            hvip: 0,
            henvcfg: 0,
            hcounteren: 0,
            vsstatus: 0,
            vsie: 0,
            vstvec: 0,
            vsscratch: 0,
            vsepc: 0,
            vscause: 0,
            vstval: 0,
            vsip: 0,
            vsatp: 0,
        }
    }
}

/// Thread / Hart Context Structure (THCS): the per-hart CSR half of both
/// supervisor domains plus the TSM readiness flag. The register halves of
/// the two domains live in the Runtime execution contexts owned by
/// `rdsm-policy`.
#[derive(Clone, Copy, Debug)]
pub struct Thcs {
    pub host_csrs: DomainCsrs,
    pub tsm_csrs: DomainCsrs,
    pub tsm_ready: bool,
}

impl Thcs {
    pub const fn new() -> Self {
        Self {
            host_csrs: DomainCsrs::new(),
            tsm_csrs: DomainCsrs::new(),
            tsm_ready: false,
        }
    }
}

/// Save the S-mode, HS-mode and VS-mode CSR half of the current domain
/// into `ctx` (`satp` excluded; it belongs to the Runtime's execution
/// context).
pub fn save_domain_csrs(ctx: &mut DomainCsrs) {
    #[cfg(target_arch = "riscv64")]
    unsafe {
        // S-mode CSRs
        core::arch::asm!(
            "csrr {sstatus}, sstatus",
            "csrr {stvec}, stvec",
            "csrr {sip}, sip",
            "csrr {sie}, sie",
            "csrr {scounteren}, scounteren",
            "csrr {sscratch}, sscratch",
            "csrr {senvcfg}, 0x10a",
            "csrr {scontext}, 0x5a8",
            sstatus = out(reg) ctx.sstatus,
            stvec = out(reg) ctx.stvec,
            sip = out(reg) ctx.sip,
            sie = out(reg) ctx.sie,
            scounteren = out(reg) ctx.scounteren,
            sscratch = out(reg) ctx.sscratch,
            senvcfg = out(reg) ctx.senvcfg,
            scontext = out(reg) ctx.scontext,
            options(nomem)
        );
        // HS-mode CSRs
        core::arch::asm!(
            "csrr {hstatus}, 0x600",
            "csrr {hgatp}, 0x680",
            "csrr {hedeleg}, 0x602",
            "csrr {hideleg}, 0x603",
            "csrr {hvip}, 0x645",
            "csrr {henvcfg}, 0x60a",
            "csrr {hcounteren}, 0x606",
            hstatus = out(reg) ctx.hstatus,
            hgatp = out(reg) ctx.hgatp,
            hedeleg = out(reg) ctx.hedeleg,
            hideleg = out(reg) ctx.hideleg,
            hvip = out(reg) ctx.hvip,
            henvcfg = out(reg) ctx.henvcfg,
            hcounteren = out(reg) ctx.hcounteren,
            options(nomem)
        );
        // VS-mode CSRs
        core::arch::asm!(
            "csrr {vsstatus}, 0x200",
            "csrr {vsie}, 0x204",
            "csrr {vstvec}, 0x205",
            "csrr {vsscratch}, 0x240",
            "csrr {vsepc}, 0x241",
            "csrr {vscause}, 0x242",
            "csrr {vstval}, 0x243",
            "csrr {vsip}, 0x244",
            "csrr {vsatp}, 0x280",
            vsstatus = out(reg) ctx.vsstatus,
            vsie = out(reg) ctx.vsie,
            vstvec = out(reg) ctx.vstvec,
            vsscratch = out(reg) ctx.vsscratch,
            vsepc = out(reg) ctx.vsepc,
            vscause = out(reg) ctx.vscause,
            vstval = out(reg) ctx.vstval,
            vsip = out(reg) ctx.vsip,
            vsatp = out(reg) ctx.vsatp,
            options(nomem)
        );
    }
}

/// Restore the S-mode, HS-mode and VS-mode CSR half of a domain from `ctx`
/// (`satp` excluded; it is installed by the Runtime's transfer ceremony).
pub fn restore_domain_csrs(ctx: &DomainCsrs) {
    #[cfg(target_arch = "riscv64")]
    unsafe {
        // S-mode CSRs
        core::arch::asm!(
            "csrw sstatus, {sstatus}",
            "csrw stvec, {stvec}",
            "csrw sip, {sip}",
            "csrw sie, {sie}",
            "csrw scounteren, {scounteren}",
            "csrw sscratch, {sscratch}",
            "csrw 0x10a, {senvcfg}",
            "csrw 0x5a8, {scontext}",
            sstatus = in(reg) ctx.sstatus,
            stvec = in(reg) ctx.stvec,
            sip = in(reg) ctx.sip,
            sie = in(reg) ctx.sie,
            scounteren = in(reg) ctx.scounteren,
            sscratch = in(reg) ctx.sscratch,
            senvcfg = in(reg) ctx.senvcfg,
            scontext = in(reg) ctx.scontext,
            options(nomem)
        );
        // HS-mode CSRs
        core::arch::asm!(
            "csrw 0x600, {hstatus}",
            "csrw 0x680, {hgatp}",
            "csrw 0x602, {hedeleg}",
            "csrw 0x603, {hideleg}",
            "csrw 0x645, {hvip}",
            "csrw 0x60a, {henvcfg}",
            "csrw 0x606, {hcounteren}",
            hstatus = in(reg) ctx.hstatus,
            hgatp = in(reg) ctx.hgatp,
            hedeleg = in(reg) ctx.hedeleg,
            hideleg = in(reg) ctx.hideleg,
            hvip = in(reg) ctx.hvip,
            henvcfg = in(reg) ctx.henvcfg,
            hcounteren = in(reg) ctx.hcounteren,
            options(nomem)
        );
        // VS-mode CSRs
        core::arch::asm!(
            "csrw 0x200, {vsstatus}",
            "csrw 0x204, {vsie}",
            "csrw 0x205, {vstvec}",
            "csrw 0x240, {vsscratch}",
            "csrw 0x241, {vsepc}",
            "csrw 0x242, {vscause}",
            "csrw 0x243, {vstval}",
            "csrw 0x244, {vsip}",
            "csrw 0x280, {vsatp}",
            vsstatus = in(reg) ctx.vsstatus,
            vsie = in(reg) ctx.vsie,
            vstvec = in(reg) ctx.vstvec,
            vsscratch = in(reg) ctx.vsscratch,
            vsepc = in(reg) ctx.vsepc,
            vscause = in(reg) ctx.vscause,
            vstval = in(reg) ctx.vstval,
            vsip = in(reg) ctx.vsip,
            vsatp = in(reg) ctx.vsatp,
            options(nomem)
        );
    }
}

// ── Domain-switch staging primitives ───────────────────────────────────

/// Reprograms the MPT view of the calling hart to `(sdid, root_ppn)` and
/// switches to interrupt domain `sidn`, making the new domain's memory
/// protection effective (order: `mmpt` write, `MFENCE.PA`, `msdcfg.SIDN`).
pub fn program_domain(mpt_mode: rdsm::csr::MptMode, sdid: usize, root_ppn: usize, sidn: usize) {
    #[cfg(target_arch = "riscv64")]
    {
        let mmpt = rdsm::csr::Mmpt::from_parts(mpt_mode, sdid, root_ppn);
        mmpt.write();
        rdsm::fence::mfence_pa(0, 0);
        rdsm::interrupt::switch_interrupt_domain(sidn);
    }
}
// ── Physical image loading ─────────────────────────────────────────────

/// Volatile-reads the CoVE payload header at `payload_base`, returning it
/// only when the `COVE` magic matches.
pub fn read_payload_header(payload_base: usize) -> Option<PayloadHeader> {
    if payload_base == 0 {
        return None;
    }
    let magic = unsafe { core::ptr::read_volatile(payload_base as *const u32) };
    if magic != COVE_PAYLOAD_MAGIC {
        return None;
    }
    // SAFETY: the payload image is firmware-owned, mapped memory whose first
    // word (checked above) identifies the header layout of `rdsm-abi`.
    Some(unsafe { core::ptr::read(payload_base as *const PayloadHeader) })
}

/// Copies `len` bytes of a loaded payload image from `src` to `dst`
/// (physical addresses).
///
/// # Safety contract
///
/// The caller has validated `[dst, dst + len)` against the platform RAM
/// range; a malformed payload must not place binaries over MMIO, firmware
/// memory or beyond physical RAM.
pub fn load_image_region(src: usize, dst: usize, len: usize) {
    // SAFETY: firmware-owned mapped memory; ranges validated by the caller
    // against the platform RAM range before this call.
    unsafe {
        core::ptr::copy(src as *const u8, dst as *mut u8, len);
    }
}

/// Issues the instruction and data fences required after payload images
/// have been copied, so the next stage executes freshly written memory.
pub fn post_load_fences() {
    #[cfg(target_arch = "riscv64")]
    unsafe {
        core::arch::asm!("fence.i", options(nostack));
        core::arch::asm!("fence rw, rw", options(nostack));
    }
}

/// Volatile-writes the platform info block for the confidential domain at
/// `dst` (a caller-validated conf-domain buffer).
pub fn write_platform_info(dst: usize, info: &RdsmPlatformInfo) {
    // SAFETY: `dst` is a conf-domain-provided buffer for exactly this
    // structure; M-mode writes are the defined delivery channel.
    unsafe {
        core::ptr::write_volatile(dst as *mut RdsmPlatformInfo, *info);
    }
}

// ── MPT page pool ──────────────────────────────────────────────────────

/// Size of the MPT page pool (4 MiB = 1024 pages of 4 KiB).
pub const MPT_PAGE_POOL_SIZE: usize = 4 * 1024 * 1024;

/// Base physical address of the MPT page pool in reserved memory (0x8090_0000).
pub const MPT_PAGE_POOL_PADDR: usize = 0x8090_0000;

/// A simple bump-pointer allocator that hands out 4 KiB pages from a
/// contiguous physical memory region.
///
/// Pages are never reclaimed (`free_page` is a no-op).  This is sufficient
/// for the current phases where MPT trees are built once and kept for the
/// firmware lifetime.
pub struct MptBumpAlloc {
    base_paddr: usize,
    size: usize,
    next_offset: usize,
}

impl MptBumpAlloc {
    /// Create a new bump allocator over the region `[base_paddr, base_paddr + size)`.
    ///
    /// The region must be 4 KiB aligned and reserved (not used by anything else).
    #[allow(dead_code)]
    pub const fn new(base_paddr: usize, size: usize) -> Self {
        Self {
            base_paddr,
            size,
            next_offset: 0,
        }
    }

    /// Create a bump allocator backed by the reserved MPT page pool.
    pub fn from_pool() -> Self {
        Self::new(MPT_PAGE_POOL_PADDR, MPT_PAGE_POOL_SIZE)
    }

    /// Allocate `num_pages` contiguous 4 KiB pages and return the PPN of
    /// the first page, or `None` if the pool is exhausted.
    #[allow(dead_code)]
    pub fn alloc_contiguous(&mut self, num_pages: usize) -> Option<usize> {
        let bytes = num_pages * 4096;
        if self.next_offset + bytes > self.size {
            return None;
        }
        let ppn = (self.base_paddr + self.next_offset) >> 12;
        self.next_offset += bytes;
        Some(ppn)
    }
}

impl MptPageAlloc for MptBumpAlloc {
    fn alloc_page(&mut self) -> Option<usize> {
        if self.next_offset + 4096 > self.size {
            return None;
        }
        let ppn = (self.base_paddr + self.next_offset) >> 12;
        self.next_offset += 4096;
        Some(ppn)
    }

    fn free_page(&mut self, _ppn: usize) {
        // Bump allocator: no-op.  Pages are not reclaimed.
    }
}
