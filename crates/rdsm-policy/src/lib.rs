//! RDSM (Root Domain Security Manager) policy layer.
//!
//! This crate sequences the M-mode RDSM: it initializes the Supervisor
//! Domain substrate (Smsdid / Smmpt / Smsdia) during firmware boot,
//! dispatches the private RDSM SBI extension (EID 0x5244534D) and the CoVE
//! forwarding path (SUPD / COVH / COVI), and sequences the host↔confidential
//! domain switches (TEECALL / TEERET) on top of the [`rdsm_mech`]
//! primitives.
//!
//! The crate contains no `unsafe` (enforced at compile time); everything
//! that touches raw assembly, volatile physical memory, or the `mmpt`/
//! `msdcfg` hardware lives in [`rdsm_mech`]. The embedding SBI firmware
//! (RustSBI Prototyper, behind its `rdsm` cargo feature) injects the
//! firmware-dependent inputs once through [`init`] via [`InitEnv`], and
//! mounts the `handle_*` entry points as custom SBI extensions.
//!
//! # Boot Flow Ordering
//!
//! `init()` must be called AFTER the firmware has configured PMP because:
//!
//! 1. PMP protects M-mode firmware (including RDSM code and data).
//! 2. MPT adds per-SD isolation on top of PMP.
//! 3. The spec requires "MPT and e(PMP) are always active" when Smsdid is
//!    implemented.
//! 4. Access check order: page table -> PMP -> MPT -> all must pass.

#![no_std]
#![forbid(unsafe_code)]
#![cfg_attr(test, allow(unused_extern_crates))]
#[cfg(test)]
extern crate std;

mod machine_irq;

use log::{error, info, warn};
use spin::Mutex;

use rdsm::domain::SdidAllocator;
use rdsm::interrupt::SidnAllocator;
use rdsm_mech::{PerHart, Thcs};

// Standard CoVE extension IDs: single source is the upstream riscv-cove
// crate; re-exported here so embedders keep one import site.
pub use riscv_cove::host::EID_COVH;
pub use riscv_cove::interrupt::EID_COVI;
pub use riscv_cove::supd::EID_SUPD;
// Private RDSM↔TSM contract: single source is rdsm-abi.
pub use rdsm_abi::{
    EID_RDSM, FID_RDSM_GET_INFO, FID_RDSM_MFENCE_PA, FID_RDSM_MPT_SET,
    FID_RDSM_TEERET, NORMAL_RETURN, TSM_READY, TVM_EXIT,
};

// The substrate, re-exported so embedders keep one dependency surface
// (probe trait, CSR wrappers) without depending on `rdsm` directly.
pub use rdsm;

// Machine-interrupt claim policy (MSDEI), installed by the embedding
// firmware during boot.
pub use machine_irq::{RdsmMachineIrq, RDSM_MACHINE_IRQ};

use rdsm_abi::{COVE_PAYLOAD_VERSION, RdsmPlatformInfo};
use sbi_spec::binary::SbiRet;

// Private error codes: only non-zero-ness is contractual across the private
// interface (the standard SBI DENIED is -4 and
// NOT_SUPPORTED is -2, which these are deliberately not). Encoded in the
// register representation (`usize` two's complement), as the caller writes
// them straight into the trap frame.
const ERR_DENIED: usize = (-8isize) as usize;
const ERR_FAILED: usize = (-1isize) as usize;
const ERR_INVALID_PARAM: usize = (-3isize) as usize;

const SBI_SUCCESS: usize = 0;

/// Everything the embedding firmware must provide once, at boot.
pub struct InitEnv<'a, P: rdsm::probe::TrapSafeCsr> {
    /// Trap-safe CSR probe (e.g. guarded-CSR-based probe of the embedding
    /// Runtime).
    pub probe: &'a mut P,
    /// Physical address of the embedded payload image.
    pub payload_base: usize,
    /// Platform RAM range `(start, end)`, `None` when unknown.
    pub ram_range: Option<(usize, usize)>,
    /// Hart ID of the boot hart running `init`.
    pub hart_id: usize,
    /// Whether the device tree advertises the `smsdid` extension.
    pub dt_smsdid: bool,
}

// ── Global RDSM context ────────────────────────────────────────────────

/// Global RDSM context recording dual-MPT state, payload header info, and FDT address.
#[derive(Clone, Copy, Debug)]
pub struct RdsmContext {
    /// Set once [`init`] completed successfully; guards the ecall paths
    /// against touching `mmpt`/`msdcfg` on substrates that were never
    /// brought up (no `smsdid` in the device tree, hardware absent).
    pub substrate_active: bool,
    pub is_cove: bool,
    pub mpt_mode: Option<rdsm::csr::MptMode>,
    pub host_sdid: usize,
    pub conf_sdid: usize,
    pub host_sidn: usize,
    pub conf_sidn: usize,
    pub host_root_ppn: usize,
    pub conf_root_ppn: usize,
    pub tsm_load_paddr: usize,
    pub tsm_entry_paddr: usize,
    pub tsm_size: usize,
    pub host_load_paddr: usize,
    pub host_entry_paddr: usize,
    pub host_size: usize,
    pub fdt_address: usize,
    /// Platform RAM range `(start, end)` captured from [`InitEnv`].
    pub ram_range: Option<(usize, usize)>,
    /// Boot hart ID captured from [`InitEnv`].
    pub hart_id: usize,
}

impl RdsmContext {
    pub const fn new() -> Self {
        Self {
            substrate_active: false,
            is_cove: false,
            mpt_mode: None,
            host_sdid: 0,
            conf_sdid: 1,
            host_sidn: 0,
            conf_sidn: 1,
            host_root_ppn: 0,
            conf_root_ppn: 0,
            tsm_load_paddr: 0,
            tsm_entry_paddr: 0,
            tsm_size: 0,
            host_load_paddr: 0,
            host_entry_paddr: 0,
            host_size: 0,
            fdt_address: 0,
            ram_range: None,
            hart_id: 0,
        }
    }
}

/// Per-hart RDSM context (dual-MPT state, payload header info, FDT address).
///
/// Each hart that participates in domain switching keeps its own copy: the
/// boot hart populates it during [`init`], and a secondary hart's copy is
/// populated by its own RDSM bring-up (pending NEMU multi-hart execution
/// — `TODO(multi-hart)`).
static RDSM_CONTEXT: PerHart<RdsmContext> = PerHart::new(RdsmContext::new());

/// Per-hart domain save areas (`hssa`/`tssa`/`tsm_ready`).
static THCS: PerHart<Thcs> = PerHart::new(Thcs::new());

/// Returns the calling hart's RDSM context.
pub fn context() -> &'static RdsmContext {
    RDSM_CONTEXT.get()
}

/// Returns the calling hart's RDSM context for mutation.
fn context_mut() -> &'static mut RdsmContext {
    RDSM_CONTEXT.get_mut()
}

#[inline]
pub fn is_cove_payload() -> bool {
    context().is_cove
}

#[inline]
pub fn get_tsm_entry() -> usize {
    context().tsm_entry_paddr
}

#[inline]
#[allow(dead_code)]
pub fn get_host_entry() -> usize {
    context().host_entry_paddr
}

#[inline]
pub fn set_fdt_address(fdt: usize) {
    context_mut().fdt_address = fdt;
}

#[inline]
#[allow(dead_code)]
pub fn get_fdt_address() -> usize {
    context().fdt_address
}

#[inline]
pub fn is_tsm_ready() -> bool {
    THCS.get().tsm_ready
}

/// BASE probe value for the private RDSM extension: nonzero once the
/// substrate is up. The dispatcher derive gates `handle` on this, so the
/// extension must advertise itself here (unlike the legacy fast_handler
/// probe semantics where RDSM reported zero).
pub fn rdsm_probe() -> usize {
    context().substrate_active as usize
}

/// SDID of the supervisor domain the calling hart currently executes in.
#[inline]
pub fn sender_sdid() -> usize {
    #[cfg(target_arch = "riscv64")]
    {
        rdsm::csr::Mmpt::read().sdid()
    }
    #[cfg(not(target_arch = "riscv64"))]
    {
        0
    }
}

// ── CoVE Payload parsing & loading ─────────────────────────────────────

/// Loads the CoVE payload images from the embedded payload at
/// `payload_base` if one is present.
///
/// Returns `true` if a valid CoVE payload was loaded, `false` otherwise.
/// Load regions are validated against the platform RAM range recorded in
/// the context before any bytes are copied.
fn load_cove_payload(payload_base: usize) -> bool {
    let header = match rdsm_mech::read_payload_header(payload_base) {
        Some(header) => header,
        None => return false,
    };

    if header.version != COVE_PAYLOAD_VERSION {
        warn!(
            "RDSM: Found CoVE payload magic but unsupported version {}",
            header.version
        );
        return false;
    }

    // Reject load regions that fall outside the platform memory range or
    // overflow: a malformed payload must not be able to place binaries
    // over MMIO, firmware memory or beyond physical RAM.
    let (ram_lo, ram_hi) = match context().ram_range {
        Some(range) => range,
        None => return false,
    };
    let region_ok = |start: u64, size: u64| -> bool {
        let s = start as usize;
        let e = match s.checked_add(size as usize) {
            Some(e) => e,
            None => return false,
        };
        s >= ram_lo && e <= ram_hi
    };
    if !region_ok(header.tsm_load_paddr, header.tsm_size)
        || !region_ok(header.host_load_paddr, header.host_size)
    {
        warn!("RDSM: CoVE payload load region outside platform memory");
        return false;
    }

    info!(
        "RDSM: CoVE payload detected (TSM: offset=0x{:x}, size=0x{:x}, load=0x{:x}, entry=0x{:x}; Host: offset=0x{:x}, size=0x{:x}, load=0x{:x}, entry=0x{:x})",
        header.tsm_offset,
        header.tsm_size,
        header.tsm_load_paddr,
        header.tsm_entry_paddr,
        header.host_offset,
        header.host_size,
        header.host_load_paddr,
        header.host_entry_paddr
    );

    rdsm_mech::load_image_region(
        (payload_base as u64 + header.tsm_offset) as usize,
        header.tsm_load_paddr as usize,
        header.tsm_size as usize,
    );
    rdsm_mech::load_image_region(
        (payload_base as u64 + header.host_offset) as usize,
        header.host_load_paddr as usize,
        header.host_size as usize,
    );
    rdsm_mech::post_load_fences();

    // Record in the global RDSM context
    let r_ctx = context_mut();
    r_ctx.is_cove = true;
    r_ctx.tsm_load_paddr = header.tsm_load_paddr as usize;
    r_ctx.tsm_entry_paddr = header.tsm_entry_paddr as usize;
    r_ctx.tsm_size = header.tsm_size as usize;
    r_ctx.host_load_paddr = header.host_load_paddr as usize;
    r_ctx.host_entry_paddr = header.host_entry_paddr as usize;
    r_ctx.host_size = header.host_size as usize;

    true
}

// ── Global domain ID allocators ────────────────────────────────────────

/// Global SDID allocator, initialized once on the boot hart.
///
/// SDID allocation is a low-frequency configuration operation; the mutex
/// keeps short critical sections (id alloc/free only).
static SDID_ALLOCATOR: Mutex<SdidAllocator> = Mutex::new(SdidAllocator::new());

/// Global SIDN allocator, initialized once on the boot hart.
static SIDN_ALLOCATOR: Mutex<SidnAllocator> = Mutex::new(SidnAllocator::new());

/// Returns a handle to the global SDID allocator.
///
/// The returned guard holds the allocator lock; keep critical sections
/// short (id allocation only, no nested locks).
pub fn rdsm_sdid_allocator() -> spin::MutexGuard<'static, SdidAllocator> {
    SDID_ALLOCATOR.lock()
}

/// Returns a handle to the global SIDN allocator.
///
/// The returned guard holds the allocator lock; keep critical sections
/// short (id allocation only, no nested locks).
pub fn rdsm_sidn_allocator() -> spin::MutexGuard<'static, SidnAllocator> {
    SIDN_ALLOCATOR.lock()
}

/// Serializes MPT tree updates through the shared page pool.
///
/// [`rdsm_mech::MptBumpAlloc`] instances handed to `MptTree::set_perm` carve
/// pages from the reserved pool; two concurrent updates (e.g. MPT_SET from
/// two harts) must not interleave their allocations.
static MPT_UPDATE_LOCK: Mutex<()> = Mutex::new(());

// ── RDSM initialization ────────────────────────────────────────────────

/// Initialize the Supervisor Domain substrate on the boot hart.
///
/// This function:
///
/// 1. Probes for Smsdid / Smmpt / Smsdia hardware support.
/// 2. Allocates SDID 0 (host supervisor domain) and SIDN 0 (host interrupt
///    domain).
/// 3. Builds an MPT tree with permissive (RWX) permissions covering the
///    platform's main memory range.
/// 4. Activates the host domain by programming the `mmpt` CSR and issuing
///    `MFENCE.PA`.
/// 5. Switches to the host interrupt domain by setting `msdcfg.SIDN = 0`.
///
/// The MPT tree is intentionally leaked - it must persist for the
/// firmware's lifetime.
///
/// # Boot Flow Ordering
///
/// Must be called after the firmware has set up PMP on the boot hart.
/// Non-boot harts will have their `mmpt` CSRs left in Bare mode (reset
/// state) and must be programmed with the host MPT root before MPT
/// restrictions are enforced in future phases.
pub fn init<P: rdsm::probe::TrapSafeCsr>(env: InitEnv<'_, P>) {
    use rdsm::{
        domain::activate_domain,
        interrupt::switch_interrupt_domain,
        mpt::{MptPerm, MptTree},
        probe::{probe_sdid_len, probe_smmpt, probe_smsdia, probe_smsdid},
    };

    // Capture the environment before anything else can rely on it.
    {
        let r_ctx = context_mut();
        r_ctx.hart_id = env.hart_id;
        r_ctx.ram_range = env.ram_range;
    }

    // Check and load CoVE payload if present
    let is_cove = load_cove_payload(env.payload_base);

    // Check if RDSM extensions are detected via device tree.
    if !env.dt_smsdid {
        info!("RDSM: Smsdid not detected, skipping RDSM initialization");
        return;
    }

    info!("RDSM: Initializing Supervisor Domain substrate");

    let probe = env.probe;

    // Enable Smstateen CTX (mstateen0 bit 57) so that the HS-mode TSM can
    // keep its per-hart id in `scontext` (0x5a8): mstateen-gated state is
    // denied to modes below M while the bit is zero, and a supervisor
    // domain cannot set mstateen itself. Probed trap-safely: skipped
    // silently on platforms without mstateen0.
    if let Some(v) = probe.read_csr(0x30c) {
        probe.write_csr(0x30c, v | (1usize << 57));
    }

    // Probe hardware support via CSR access.
    if !probe_smsdid(probe) {
        warn!("RDSM: mmpt CSR not readable, Smsdid not implemented in hardware");
        return;
    }

    let mpt_mode = match probe_smmpt(&mut *probe) {
        Some(mode) => mode,
        None => {
            warn!("RDSM: No Smmpt mode supported, skipping MPT initialization");
            return;
        }
    };

    let sdid_len = probe_sdid_len(&mut *probe);
    info!(
        "RDSM: Smsdid detected, Smmpt mode={:?}, SDID width={}",
        mpt_mode, sdid_len
    );

    if !probe_smsdia(probe) {
        warn!("RDSM: Smsdia not implemented, interrupt domain features limited");
    }

    // Allocate SDID 0 for the host supervisor domain using the global
    // allocator so the state persists after init returns.
    let host_sdid = SDID_ALLOCATOR.lock().alloc().expect("SDID allocation for host failed");
    assert_eq!(host_sdid, 0, "Host SDID must be 0");

    // Allocate SDID 1 for the confidential supervisor domain.
    let conf_sdid = SDID_ALLOCATOR.lock().alloc().expect("SDID allocation for conf failed");
    assert_eq!(conf_sdid, 1, "Confidential SDID must be 1");

    // Allocate SIDN 0 for the host interrupt domain using the global
    // allocator so the state persists after init returns.
    let host_sidn = SIDN_ALLOCATOR.lock().alloc().expect("SIDN allocation for host failed");
    assert_eq!(host_sidn, 0, "Host SIDN must be 0");

    // Allocate SIDN 1 for the confidential interrupt domain.
    let conf_sidn = SIDN_ALLOCATOR.lock().alloc().expect("SIDN allocation for conf failed");
    assert_eq!(conf_sidn, 1, "Confidential SIDN must be 1");

    // Create MPT page allocator from the reserved pool.
    let mut mpt_alloc = rdsm_mech::MptBumpAlloc::from_pool();

    // Build the confidential MPT tree (MPT_CONF, SDID=1).
    let mut conf_tree = match MptTree::new(mpt_mode, &mut mpt_alloc) {
        Some(tree) => tree,
        None => {
            error!("RDSM: Failed to allocate MPT_CONF root page");
            return;
        }
    };

    // Build the host MPT tree (MPT_HOST, SDID=0).
    let mut host_tree = match MptTree::new(mpt_mode, &mut mpt_alloc) {
        Some(tree) => tree,
        None => {
            error!("RDSM: Failed to allocate MPT_HOST root page");
            return;
        }
    };

    // Set permissive permissions for the platform address range.
    //
    // Only the platform's memory region is covered to keep page-pool
    // usage bounded.  Full address-space coverage (0 ... usize::MAX) requires
    // NAPOT support and is deferred.
    let (ram_start, ram_end) = match env.ram_range {
        Some(range) => range,
        None => {
            error!("RDSM: Platform memory range not initialized");
            return;
        }
    };

    info!(
        "RDSM: Setting MPT permissions for range 0x{:x} - 0x{:x}",
        ram_start, ram_end
    );

    // Low memory (MMIO, firmware data below main RAM, e.g. 0..0x80000000).
    if ram_start > 0 {
        conf_tree.set_perm(0, ram_start, MptPerm::RWX, &mut mpt_alloc);
        host_tree.set_perm(0, ram_start, MptPerm::RWX, &mut mpt_alloc);
    }

    // MPT_CONF (SDID=1): Full platform RAM range with RWX permissions.
    conf_tree.set_perm(
        ram_start,
        ram_end.saturating_sub(ram_start),
        MptPerm::RWX,
        &mut mpt_alloc,
    );

    // MPT_HOST (SDID=0):
    if is_cove {
        let r_ctx = context();
        let tsm_load_paddr = r_ctx.tsm_load_paddr;
        let host_load_paddr = r_ctx.host_load_paddr;

        // Platform RAM before TSM range: RWX
        if tsm_load_paddr > ram_start {
            host_tree.set_perm(
                ram_start,
                tsm_load_paddr.saturating_sub(ram_start),
                MptPerm::RWX,
                &mut mpt_alloc,
            );
        }

        // TSM memory range [tsm_load_paddr, host_load_paddr): NONE (inaccessible)
        if host_load_paddr > tsm_load_paddr {
            host_tree.set_perm(
                tsm_load_paddr,
                host_load_paddr.saturating_sub(tsm_load_paddr),
                MptPerm::NONE,
                &mut mpt_alloc,
            );
        }

        // Platform RAM from host_load_paddr onwards: RWX
        if ram_end > host_load_paddr {
            host_tree.set_perm(
                host_load_paddr,
                ram_end.saturating_sub(host_load_paddr),
                MptPerm::RWX,
                &mut mpt_alloc,
            );
        }
    } else {
        // Not a CoVE payload: full platform RAM with RWX
        host_tree.set_perm(
            ram_start,
            ram_end.saturating_sub(ram_start),
            MptPerm::RWX,
            &mut mpt_alloc,
        );
    }

    info!(
        "RDSM: MPT trees built: Host root_ppn=0x{:x}, Conf root_ppn=0x{:x}, mode={:?}",
        host_tree.root_ppn(),
        conf_tree.root_ppn(),
        mpt_mode
    );

    // Save both trees/root PPNs and mode globally in RDSM context.
    {
        let r_ctx = context_mut();
        r_ctx.mpt_mode = Some(mpt_mode);
        r_ctx.host_sdid = host_sdid;
        r_ctx.conf_sdid = conf_sdid;
        r_ctx.host_sidn = host_sidn;
        r_ctx.conf_sidn = conf_sidn;
        r_ctx.host_root_ppn = host_tree.root_ppn();
        r_ctx.conf_root_ppn = conf_tree.root_ppn();
    }

    if is_cove {
        info!("[RDSM] Booting...");

        // Activate Confidential domain (SDID=1, MPT_CONF)
        activate_domain(conf_sdid, &conf_tree);
        switch_interrupt_domain(conf_sidn);
        info!("RDSM: Confidential supervisor domain (SDID=1) activated, SIDN=1 set");
    } else {
        // Activate Host domain (SDID=0, MPT_HOST)
        activate_domain(host_sdid, &host_tree);
        switch_interrupt_domain(host_sidn);
        info!("RDSM: Host supervisor domain (SDID=0) activated, SIDN=0 set");
    }

    // Intentional leak: the MPT trees must persist for the firmware's lifetime.
    core::mem::forget(host_tree);
    core::mem::forget(conf_tree);

    context_mut().substrate_active = true;
    info!("RDSM: Initialization complete");
}

// ── SBI extension handlers ─────────────────────────────────────────────

/// Handles the private RDSM extension (EID 0x5244534D) from the embedding
/// firmware's SBI dispatch.
///
/// Sender validation: the private RDSM extension is only callable from the
/// confidential domain. A host-domain ecall must never be able to program
/// MPT entries or drive the TEERET machinery directly.
pub fn handle_rdsm(function: usize, args: [usize; 6]) -> SbiRet {
    // Guard before any `mmpt` access: an uninitialized substrate must reject
    // instead of trapping on the CSR read.
    if !context().substrate_active {
        return SbiRet { error: ERR_DENIED, value: 0 };
    }

    #[cfg(target_arch = "riscv64")]
    if sender_sdid() != context().conf_sdid {
        return SbiRet { error: ERR_DENIED, value: 0 };
    }

    match function {
        FID_RDSM_GET_INFO => get_info(args[2]),
        FID_RDSM_MPT_SET => mpt_set(args),
        FID_RDSM_MFENCE_PA => {
            #[cfg(target_arch = "riscv64")]
            rdsm::fence::mfence_pa(args[0], args[1]);
            SbiRet { error: SBI_SUCCESS, value: 0 }
        }
        FID_RDSM_TEERET => {
            // Pending upstream rustsbi#286: the retentive TSM→host switch
            // stages its effect through the Runtime trap frame, which custom
            // extension dispatch cannot reach. The sequenced implementation
            // is kept in [`switch`].
            error!("RDSM: TEERET requires runtime frame access (rustsbi#286); rejecting");
            SbiRet { error: ERR_FAILED, value: 0 }
        }
        _ => SbiRet { error: ERR_FAILED, value: 0 },
    }
}

/// FID 0 GET_INFO: reports the MPT mode and, when `info_buf` is nonzero,
/// writes the platform reserved-region layout to the confidential-domain
/// buffer used for input validation.
fn get_info(info_buf: usize) -> SbiRet {
    let r_ctx = context();
    let mode_val = r_ctx.mpt_mode.map_or(0, |m| m as usize);
    if info_buf != 0 {
        let (ram_start, ram_end) = r_ctx.ram_range.unwrap_or((0, 0));
        rdsm_mech::write_platform_info(
            info_buf,
            &RdsmPlatformInfo {
                ram_start,
                ram_end,
                tsm_region_start: r_ctx.tsm_load_paddr,
                tsm_region_end: r_ctx.host_load_paddr,
                mpt_pool_start: rdsm_mech::MPT_PAGE_POOL_PADDR,
                mpt_pool_end: rdsm_mech::MPT_PAGE_POOL_PADDR + rdsm_mech::MPT_PAGE_POOL_SIZE,
            },
        );
    }
    SbiRet { error: SBI_SUCCESS, value: mode_val }
}

/// FID 1 MPT_SET: validates and applies a permission update to the target
/// domain's MPT tree.
fn mpt_set(args: [usize; 6]) -> SbiRet {
    let target_sdid = args[0];
    let paddr = args[1];
    let len = args[2];
    let perm_bits = args[3] as u8;

    let perm = match rdsm::mpt::MptPerm::from_bits(perm_bits) {
        Some(p) => p,
        None => return SbiRet { error: ERR_INVALID_PARAM, value: 0 },
    };

    let r_ctx = context();
    let mode = match r_ctx.mpt_mode {
        Some(m) => m,
        None => return SbiRet { error: ERR_FAILED, value: 0 },
    };

    let root_ppn = if target_sdid == r_ctx.host_sdid {
        r_ctx.host_root_ppn
    } else if target_sdid == r_ctx.conf_sdid {
        r_ctx.conf_root_ppn
    } else {
        return SbiRet { error: ERR_INVALID_PARAM, value: 0 };
    };

    // Alignment + reserved-region validation:
    // - paddr / len must be 4 KiB aligned, len nonzero, range in RAM
    // - host-domain updates must stay inside host-allocatable memory
    //   ([host_load, mpt_pool) ∪ [mpt_pool_end, ram_end))
    // - conf-domain updates must not cover the firmware gap
    //   ([ram_start, tsm_load)) or the MPT page pool
    let range_ok = {
        let end = paddr.checked_add(len);
        let (ram_start, ram_end) = r_ctx.ram_range.unwrap_or((0, 0));
        match end {
            Some(e) if e <= ram_end && paddr >= ram_start => {
                let overlaps_pool = !(e <= rdsm_mech::MPT_PAGE_POOL_PADDR
                    || paddr >= rdsm_mech::MPT_PAGE_POOL_PADDR + rdsm_mech::MPT_PAGE_POOL_SIZE);
                if target_sdid == r_ctx.host_sdid {
                    paddr >= r_ctx.host_load_paddr && !overlaps_pool
                } else {
                    paddr >= r_ctx.tsm_load_paddr && !overlaps_pool
                }
            }
            _ => false,
        }
    };
    if paddr % 4096 != 0 || len % 4096 != 0 || len == 0 || !range_ok {
        return SbiRet { error: ERR_INVALID_PARAM, value: 0 };
    }

    // The tree update draws pages from the shared MPT pool; hold the update
    // lock so concurrent harts cannot interleave their pool allocations.
    let _mpt_guard = MPT_UPDATE_LOCK.lock();
    let mut tree = rdsm::mpt::MptTree::from_root(mode, root_ppn);
    let mut alloc = rdsm_mech::MptBumpAlloc::from_pool();
    tree.set_perm(paddr, len, perm, &mut alloc);
    core::mem::forget(tree);

    SbiRet { error: SBI_SUCCESS, value: 0 }
}

/// Handles the SUPD discovery extension (EID 0x53555044): FID 0 reports the
/// supported domain kinds (host bit 0 + confidential bit 1).
pub fn handle_supd(function: usize) -> SbiRet {
    if function == 0 {
        SbiRet { error: SBI_SUCCESS, value: 0b11 }
    } else {
        SbiRet { error: ERR_FAILED, value: 0 }
    }
}

/// BASE probe value for SUPD: the extension is always advertised.
pub const SUPD_PROBE: usize = 1;

/// BASE probe value for COVH: advertised once the TSM has announced
/// readiness via TEERET/TSM_READY.
pub fn covh_probe() -> usize {
    is_tsm_ready() as usize
}

/// BASE probe value for COVI: interrupt virtualization is not yet offered.
pub const COVI_PROBE: usize = 0;

/// Handles a COVH / COVI call (TEECALL) from the host domain.
///
/// Sender validation mirrors the RDSM extension: only the host domain may
/// call, and only after the TSM has announced readiness. The retentive
/// host→TSM switch itself is pending upstream rustsbi#286.
pub fn handle_teecall(function: usize, args: [usize; 6]) -> SbiRet {
    // Guard before any `mmpt` access (see [`handle_rdsm`]).
    if !context().substrate_active {
        return SbiRet { error: ERR_DENIED, value: 0 };
    }
    if sender_sdid() != context().host_sdid {
        return SbiRet { error: ERR_DENIED, value: 0 };
    }
    if !is_tsm_ready() {
        return SbiRet { error: ERR_FAILED, value: 0 };
    }
    // Pending upstream rustsbi#286: the sequenced implementation is kept in
    // [`switch::teecall_enter`].
    error!(
        "RDSM: TEECALL (EID function {function}) requires runtime frame access (rustsbi#286); rejecting"
    );
    let _ = (function, args);
    SbiRet { error: ERR_FAILED, value: 0 }
}

// ── Retentive domain switching (pending upstream rustsbi#286) ──────────

/// Retentive domain switching, sequenced end-to-end on top of the
/// [`rdsm_mech`] primitives but not yet wired: reaching these functions
/// requires reading and rewriting the Runtime's trap frame, which is
/// `pub(crate)` upstream (proposal: rustsbi#286). Once that lands, the
/// embedding firmware bridges its frame to [`FrameRegs`] and calls into
/// this module from its dispatch path.
///
/// Unlike the fast-trap era these functions never touch `mscratch` (the
/// Runtime's sentinel is private) and never advance `mepc` by instruction
/// length (ecalls are always 4 bytes; the resume `pc` values here are the
/// other domain's saved or entry PCs).
#[allow(dead_code)] // pending upstream rustsbi#286
pub mod switch {
    use log::info;

    use rdsm_mech::{
        FrameRegs, Thcs, program_domain, restore_domain_csrs, save_domain_csrs,
        stage_domain_return, stage_first_domain_entry,
    };
    use rdsm_abi::{NORMAL_RETURN, TSM_READY};

    use super::{THCS, context};

    /// Host→TSM TEECALL: saves the host context into `hssa`, switches the
    /// hart to the confidential domain, and restores the TSM's saved state
    /// into `frame`. The host call arguments arrive in `frame.a` and are
    /// passed through to the TSM.
    pub fn teecall_enter(frame: &mut FrameRegs) {
        let epc = riscv::register::mepc::read();
        let host_a = frame.a;

        // 1. Save Host context to THCS.hssa
        {
            let th = THCS.get_mut();
            let h = &mut th.hssa;
            h.ra = frame.ra;
            h.sp = frame.sp;
            h.gp = frame.gp;
            h.tp = frame.tp;
            h.t = frame.t;
            h.s = frame.s;
            h.a = host_a;
            h.pc = epc + 4;
            save_domain_csrs(h);
        }

        // 2. Switch the MPT view to the confidential domain
        let r_ctx = context();
        let mpt_mode = r_ctx.mpt_mode.unwrap_or(rdsm::csr::MptMode::Bare);
        program_domain(mpt_mode, r_ctx.conf_sdid, r_ctx.conf_root_ppn, r_ctx.conf_sidn);

        // 3. Restore the TSM's saved CSRs and registers; the host call
        //    arguments ride through in a0..a7
        let th: &Thcs = THCS.get();
        restore_domain_csrs(&th.tssa);

        let tsm_pc = th.tssa.pc;
        frame.a = host_a;
        frame.ra = th.tssa.ra;
        frame.t = th.tssa.t;
        frame.s = th.tssa.s;
        frame.sp = th.tssa.sp;
        frame.gp = th.tssa.gp;
        frame.tp = th.tssa.tp;
        stage_domain_return(tsm_pc);
    }

    /// TSM→host TEERET. `Ok(())` means the switch staged successfully and
    /// the hart must resume into the host domain (no SBI return reaches the
    /// TSM); `Err(reason)` reports an unsupported TEERET reason for the
    /// caller to encode.
    pub fn teeret(frame: &mut FrameRegs, reason: usize) -> Result<(), usize> {
        match reason {
            TSM_READY => {
                let requested_entry = frame.a[1];
                let r_ctx = context();
                let tsm_entry = if requested_entry != 0 {
                    requested_entry
                } else {
                    r_ctx.tsm_entry_paddr
                };

                // 1. Save the TSM's first-boot context into tssa and mark
                //    the TSM ready
                {
                    let th = THCS.get_mut();
                    let t = &mut th.tssa;
                    t.pc = tsm_entry;
                    t.sp = frame.sp;
                    t.gp = frame.gp;
                    t.tp = frame.tp;
                    t.ra = frame.ra;
                    t.t = frame.t;
                    t.s = frame.s;
                    t.a = frame.a;
                    save_domain_csrs(t);
                    th.tsm_ready = true;
                }

                info!("[RDSM] Switching to Host Domain (SDID=0)...");

                // 2. Switch the MPT view to the host domain
                let mpt_mode = r_ctx.mpt_mode.unwrap_or(rdsm::csr::MptMode::Bare);
                program_domain(mpt_mode, r_ctx.host_sdid, r_ctx.host_root_ppn, r_ctx.host_sidn);

                // 3. Fabricate the first (non-retentive) host entry state
                let host_entry_paddr = r_ctx.host_entry_paddr;
                let fdt_address = r_ctx.fdt_address;
                stage_first_domain_entry(host_entry_paddr);

                frame.a[0] = r_ctx.hart_id;
                frame.a[1] = fdt_address;
                Ok(())
            }
            NORMAL_RETURN => {
                let epc = riscv::register::mepc::read();
                let tsm_err = frame.a[1];
                let tsm_val = frame.a[2];

                // 1. Save TSM context to THCS.tssa
                {
                    let th = THCS.get_mut();
                    let t = &mut th.tssa;
                    t.ra = frame.ra;
                    t.sp = frame.sp;
                    t.gp = frame.gp;
                    t.tp = frame.tp;
                    t.t = frame.t;
                    t.s = frame.s;
                    t.a = frame.a;
                    t.pc = epc + 4;
                    save_domain_csrs(t);
                }

                // 2. Switch the MPT view to the host domain
                let r_ctx = context();
                let mpt_mode = r_ctx.mpt_mode.unwrap_or(rdsm::csr::MptMode::Bare);
                program_domain(mpt_mode, r_ctx.host_sdid, r_ctx.host_root_ppn, r_ctx.host_sidn);

                // 3. Restore Host saved CSRs and registers; the TSM's
                //    returned error/value land in host a0/a1
                let th: &Thcs = THCS.get();
                restore_domain_csrs(&th.hssa);

                let h = &th.hssa;
                frame.ra = h.ra;
                frame.t = h.t;
                frame.s = h.s;
                frame.a[0] = tsm_err;
                frame.a[1] = tsm_val;
                frame.a[2] = h.a[2];
                frame.a[3] = h.a[3];
                frame.a[4] = h.a[4];
                frame.a[5] = h.a[5];
                frame.a[6] = h.a[6];
                frame.a[7] = h.a[7];
                frame.gp = h.gp;
                frame.tp = h.tp;
                frame.sp = h.sp;
                stage_domain_return(h.pc);
                Ok(())
            }
            other => Err(other),
        }
    }
}
