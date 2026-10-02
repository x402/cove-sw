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
use rdsm_mech::{PerHart, PerHartCell, Thcs};

// Standard CoVE extension IDs: single source is the upstream riscv-cove
// crate; re-exported here so embedders keep one import site.
pub use riscv_cove::host::EID_COVH;
pub use riscv_cove::interrupt::EID_COVI;
pub use riscv_cove::supd::EID_SUPD;
// Private RDSM↔TSM contract: single source is rdsm-abi.
pub use rdsm_abi::{
    EID_RDSM, FID_RDSM_GET_INFO, FID_RDSM_MFENCE_PA, FID_RDSM_MPT_SET, FID_RDSM_TEERET,
    NORMAL_RETURN, TSM_READY, TVM_EXIT,
};

// The substrate, re-exported so embedders keep one dependency surface
// (probe trait, CSR wrappers) without depending on `rdsm` directly.
pub use rdsm;

// Machine-interrupt claim policy (MSDEI), installed by the embedding
// firmware during boot.
pub use machine_irq::{RDSM_MACHINE_IRQ, RdsmMachineIrq};

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
    /// Dispatch entry the TSM registered with its TSM_READY TEERET, consumed
    /// by the first host TEECALL (later TEECALLs resume at the TSM's parked
    /// breakpoint instead).
    pub dispatch_entry: Option<usize>,
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
            dispatch_entry: None,
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

/// Per-hart domain save areas (`host_csrs`/`tsm_csrs`/`tsm_ready`).
static THCS: PerHart<Thcs> = PerHart::new(Thcs::new());

/// Per-hart pair of Runtime execution contexts for retentive domain
/// switching: `host` snapshots the host (VMM) execution, `tsm` the TSM's.
/// GPRs, resume PC and `satp` live here and are exchanged by the Runtime's
/// transfer ceremony; only the CSR half is handled through [`rdsm_mech`].
struct HartCtx {
    host: runtime::context::ExecutionContext,
    tsm: runtime::context::ExecutionContext,
}

impl HartCtx {
    const fn new() -> Self {
        Self {
            host: runtime::context::ExecutionContext::new(),
            tsm: runtime::context::ExecutionContext::new(),
        }
    }
}

impl rdsm_mech::ConstInit for HartCtx {
    const INIT: HartCtx = HartCtx::new();
}

static CONTEXTS: PerHartCell<HartCtx> = PerHartCell::new();

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
    let host_sdid = SDID_ALLOCATOR
        .lock()
        .alloc()
        .expect("SDID allocation for host failed");
    assert_eq!(host_sdid, 0, "Host SDID must be 0");

    // Allocate SDID 1 for the confidential supervisor domain.
    let conf_sdid = SDID_ALLOCATOR
        .lock()
        .alloc()
        .expect("SDID allocation for conf failed");
    assert_eq!(conf_sdid, 1, "Confidential SDID must be 1");

    // Allocate SIDN 0 for the host interrupt domain using the global
    // allocator so the state persists after init returns.
    let host_sidn = SIDN_ALLOCATOR
        .lock()
        .alloc()
        .expect("SIDN allocation for host failed");
    assert_eq!(host_sidn, 0, "Host SIDN must be 0");

    // Allocate SIDN 1 for the confidential interrupt domain.
    let conf_sidn = SIDN_ALLOCATOR
        .lock()
        .alloc()
        .expect("SIDN allocation for conf failed");
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
        return SbiRet {
            error: ERR_DENIED,
            value: 0,
        };
    }

    #[cfg(target_arch = "riscv64")]
    if sender_sdid() != context().conf_sdid {
        return SbiRet {
            error: ERR_DENIED,
            value: 0,
        };
    }

    match function {
        FID_RDSM_GET_INFO => get_info(args[2]),
        FID_RDSM_MPT_SET => mpt_set(args),
        FID_RDSM_MFENCE_PA => {
            #[cfg(target_arch = "riscv64")]
            rdsm::fence::mfence_pa(args[0], args[1]);
            SbiRet {
                error: SBI_SUCCESS,
                value: 0,
            }
        }
        FID_RDSM_TEERET => switch::teeret(args),
        _ => SbiRet {
            error: ERR_FAILED,
            value: 0,
        },
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
    SbiRet {
        error: SBI_SUCCESS,
        value: mode_val,
    }
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
        None => {
            return SbiRet {
                error: ERR_INVALID_PARAM,
                value: 0,
            };
        }
    };

    let r_ctx = context();
    let mode = match r_ctx.mpt_mode {
        Some(m) => m,
        None => {
            return SbiRet {
                error: ERR_FAILED,
                value: 0,
            };
        }
    };

    let root_ppn = if target_sdid == r_ctx.host_sdid {
        r_ctx.host_root_ppn
    } else if target_sdid == r_ctx.conf_sdid {
        r_ctx.conf_root_ppn
    } else {
        return SbiRet {
            error: ERR_INVALID_PARAM,
            value: 0,
        };
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
        return SbiRet {
            error: ERR_INVALID_PARAM,
            value: 0,
        };
    }

    // The tree update draws pages from the shared MPT pool; hold the update
    // lock so concurrent harts cannot interleave their pool allocations.
    let _mpt_guard = MPT_UPDATE_LOCK.lock();
    let mut tree = rdsm::mpt::MptTree::from_root(mode, root_ppn);
    let mut alloc = rdsm_mech::MptBumpAlloc::from_pool();
    tree.set_perm(paddr, len, perm, &mut alloc);
    core::mem::forget(tree);

    SbiRet {
        error: SBI_SUCCESS,
        value: 0,
    }
}

/// Handles the SUPD discovery extension (EID 0x53555044): FID 0 reports the
/// supported domain kinds (host bit 0 + confidential bit 1).
pub fn handle_supd(function: usize) -> SbiRet {
    if function == 0 {
        SbiRet {
            error: SBI_SUCCESS,
            value: 0b11,
        }
    } else {
        SbiRet {
            error: ERR_FAILED,
            value: 0,
        }
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
/// call, and only after the TSM has announced readiness. On success the
/// retentive host→TSM switch never returns to the host here — the hart
/// resumes inside the TSM and the call result is delivered by the TSM's
/// TEERET (NORMal_RETURN injects it into the host's parked snapshot).
pub fn handle_teecall(eid: usize, function: usize, args: [usize; 6]) -> SbiRet {
    // Guard before any `mmpt` access (see [`handle_rdsm`]).
    if !context().substrate_active {
        return SbiRet {
            error: ERR_DENIED,
            value: 0,
        };
    }
    if sender_sdid() != context().host_sdid {
        return SbiRet {
            error: ERR_DENIED,
            value: 0,
        };
    }
    if !is_tsm_ready() {
        return SbiRet {
            error: ERR_FAILED,
            value: 0,
        };
    }
    switch::teecall(eid, function, args)
}

// ── Retentive domain switching ─────────────────────────────────────────

/// Retentive host↔TSM domain switching, sequenced end-to-end on top of the
/// [`rdsm_mech`] primitives and the Runtime's declarative execution
/// contexts held in the per-hart `CONTEXTS` slots.
///
/// Every entry follows the same shape (CoVE ABI v0.7 §5.3):
///
/// 1. Guards — every predictable failure is checked before any state
///    changes, because a failed retentive transfer does not unwind the
///    adjacent ceremony that has already run (the transfer's own failure
///    boundary sits at staging time).
/// 2. Snapshot patch — the resume source's saved GPR snapshot is updated
///    through the context data API (call arguments in, return values out).
/// 3. Adjacent ceremony — save the outgoing CSR half, restore the incoming
///    CSR half, program the incoming domain (`mmpt` + `MFENCE.PA` + SIDN).
/// 4. Stage — hand both contexts to the Runtime; its ceremony on this
///    hart's ecall return path saves the outgoing GPR/PC/`satp` state,
///    installs the incoming one, and `mret`s. A successful stage never
///    returns to the caller.
///
/// On non-RISC-V targets (host unit tests) the CSR half and the domain
/// programming are no-ops and the staging step is compiled out, so the
/// guard and snapshot data flow is exercised without hardware.
pub mod switch {
    use log::{error, info};

    use rdsm_abi::{NORMAL_RETURN, TSM_READY};
    use rdsm_mech::{DomainCsrs, program_domain, restore_domain_csrs, save_domain_csrs};
    use runtime::context::{ContextState, Satp};
    use sbi_spec::binary::SbiRet;

    use super::{CONTEXTS, THCS, context, context_mut};

    /// Maps a context/transfer failure onto the private error channel. The
    /// error value's only contractual property is non-zero-ness (TSM and
    /// host park or report on any failure).
    fn err_after(reason: &str, e: runtime::context::TransferError) -> SbiRet {
        error!("RDSM: retentive transfer failed ({reason}): {e:?}");
        SbiRet {
            error: super::ERR_FAILED,
            value: 0,
        }
    }

    fn failed(reason: &str) -> SbiRet {
        error!("RDSM: retentive transfer rejected: {reason}");
        SbiRet {
            error: super::ERR_FAILED,
            value: 0,
        }
    }

    /// FID3 TEERET from the confidential domain. `args[0]` is the reason,
    /// the remaining parameters are reason-specific.
    pub fn teeret(args: [usize; 6]) -> SbiRet {
        match args[0] {
            TSM_READY => teeret_tsm_ready(args[1]),
            NORMAL_RETURN => teeret_normal_return(args[1], args[2]),
            other => {
                error!("RDSM: TEERET reason {other:#x} not supported");
                SbiRet {
                    error: super::ERR_FAILED,
                    value: 0,
                }
            }
        }
    }

    /// TEERET(TSM_READY): the TSM announces readiness and hands the hart to
    /// the host. The host context is fabricated fresh (the "initialized by
    /// the TSM-driver" case of the ABI's THCS model), the running TSM
    /// execution is adopted by its context at staging time.
    fn teeret_tsm_ready(requested_entry: usize) -> SbiRet {
        let hart = CONTEXTS.get();
        // Only the first TSM_READY may enter: afterwards the host context
        // holds either a parked snapshot or the running execution.
        if hart.host.state() != ContextState::Empty {
            return failed("TSM_READY but the host context is not fresh");
        }

        let r_ctx = context();
        let entry = if requested_entry != 0 {
            requested_entry
        } else {
            r_ctx.tsm_entry_paddr
        };
        let (hart_id, fdt_address, host_entry_paddr) =
            (r_ctx.hart_id, r_ctx.fdt_address, r_ctx.host_entry_paddr);
        let (mpt_mode, host_sdid, host_root_ppn, host_sidn) = (
            r_ctx.mpt_mode.unwrap_or(rdsm::csr::MptMode::Bare),
            r_ctx.host_sdid,
            r_ctx.host_root_ppn,
            r_ctx.host_sidn,
        );

        // Fabricate the first host entry snapshot with the non-retentive
        // next-stage convention: a0 = hart id, a1 = FDT address, everything
        // else zero, bare translation.
        let mut x1_x31 = [0usize; 31];
        x1_x31[9] = hart_id;
        x1_x31[10] = fdt_address;
        if let Err(e) = hart.host.fill(x1_x31, host_entry_paddr, Satp::from_bits(0)) {
            return err_after("fill of the fresh host context", e);
        }
        if let Err(e) = hart.host.park() {
            return err_after("park of the fresh host context", e);
        }

        let th = THCS.get_mut();
        th.tsm_ready = true;
        // Preserve the TSM's boot-time CSR half (notably `scontext`, which
        // carries the per-hart dispatch-stack index) for the first TEECALL.
        save_domain_csrs(&mut th.tsm_csrs);
        context_mut().dispatch_entry = Some(entry);

        info!("[RDSM] Switching to Host Domain (SDID=0)...");

        // Adjacent ceremony: the host has no CSR history to restore, so its
        // half comes up zeroed (fresh-boot semantics); then activate the
        // host domain and stage the transfer.
        restore_domain_csrs(&DomainCsrs::new());
        program_domain(mpt_mode, host_sdid, host_root_ppn, host_sidn);
        #[cfg(target_arch = "riscv64")]
        if let Err(e) = runtime::trap::stage_retentive_transfer(&hart.tsm, &hart.host) {
            return err_after("staging the transfer into the host", e);
        }
        // Unreachable on the hart: the staged transfer rewrites this call's
        // trap frame and the hart resumes inside the host domain. On host
        // tests the success value reports the staged data flow.
        SbiRet {
            error: super::SBI_SUCCESS,
            value: 0,
        }
    }

    /// TEERET(NORMAL_RETURN): the TSM hands the hart back to the host,
    /// delivering `(error, value)` in the host's a0/a1 (the ABI requires
    /// the TSM to always set both).
    fn teeret_normal_return(err: usize, val: usize) -> SbiRet {
        let hart = CONTEXTS.get();
        // The host snapshot must be parked mid-ecall from its TEECALL.
        if hart.host.state() != ContextState::Suspended {
            return failed("NORMAL_RETURN but the host context holds no parked snapshot");
        }

        // Patch the return values into the suspended host snapshot.
        let mut snap = hart.host.snapshot();
        snap.x1_x31[9] = err;
        snap.x1_x31[10] = val;
        if let Err(e) = hart.host.fill(snap.x1_x31, snap.pc, snap.satp) {
            return err_after("patching the host snapshot", e);
        }

        // Adjacent ceremony: swap the CSR half back to the host and activate
        // the host domain.
        let th = THCS.get_mut();
        save_domain_csrs(&mut th.tsm_csrs);
        restore_domain_csrs(&th.host_csrs);
        let r_ctx = context();
        program_domain(
            r_ctx.mpt_mode.unwrap_or(rdsm::csr::MptMode::Bare),
            r_ctx.host_sdid,
            r_ctx.host_root_ppn,
            r_ctx.host_sidn,
        );
        #[cfg(target_arch = "riscv64")]
        if let Err(e) = runtime::trap::stage_retentive_transfer(&hart.tsm, &hart.host) {
            return err_after("staging the transfer into the host", e);
        }
        SbiRet {
            error: super::SBI_SUCCESS,
            value: 0,
        }
    }

    /// TEECALL from the host domain (COVH / COVI): parks the host mid-ecall
    /// and resumes the TSM at its dispatch entry (first call) or parked
    /// breakpoint (subsequent calls), with the call packed into a0–a7.
    pub fn teecall(eid: usize, function: usize, args: [usize; 6]) -> SbiRet {
        let hart = CONTEXTS.get();
        // The TSM snapshot is suspended after its TSM_READY transfer; an
        // empty snapshot is admissible as a cold dispatch (its data is fully
        // rewritten below). Any other state means the TSM is mid-transfer.
        match hart.tsm.state() {
            ContextState::Suspended | ContextState::Empty => {}
            _ => return failed("TEECALL but the TSM context is not resumable"),
        }

        // Patch the call into the TSM snapshot: a0–a5 = parameters,
        // a6 = function, a7 = extension id; resume at the registered
        // dispatch entry the first time and at the parked breakpoint after.
        let mut snap = hart.tsm.snapshot();
        snap.x1_x31[9..15].copy_from_slice(&args);
        snap.x1_x31[15] = function;
        snap.x1_x31[16] = eid;
        let resume_pc = context_mut().dispatch_entry.take().unwrap_or(snap.pc);
        if let Err(e) = hart.tsm.fill(snap.x1_x31, resume_pc, snap.satp) {
            return err_after("patching the TSM snapshot", e);
        }

        // Adjacent ceremony: park the host's CSR half, restore the TSM's,
        // and activate the confidential domain.
        let th = THCS.get_mut();
        save_domain_csrs(&mut th.host_csrs);
        restore_domain_csrs(&th.tsm_csrs);
        let r_ctx = context();
        program_domain(
            r_ctx.mpt_mode.unwrap_or(rdsm::csr::MptMode::Bare),
            r_ctx.conf_sdid,
            r_ctx.conf_root_ppn,
            r_ctx.conf_sidn,
        );
        #[cfg(target_arch = "riscv64")]
        if let Err(e) = runtime::trap::stage_retentive_transfer(&hart.host, &hart.tsm) {
            return err_after("staging the transfer into the TSM", e);
        }
        SbiRet {
            error: super::SBI_SUCCESS,
            value: 0,
        }
    }

    #[cfg(test)]
    mod tests {
        use runtime::context::{ContextState, Satp};

        use super::CONTEXTS;
        use crate::{
            EID_COVH, ERR_DENIED, ERR_FAILED, FID_RDSM_TEERET, NORMAL_RETURN, SBI_SUCCESS,
            TSM_READY, context, context_mut, handle_rdsm, handle_teecall, is_tsm_ready,
        };
        use riscv_cove::host::CONVERT_PAGES;

        // The switch state lives in process-global per-hart slots and cannot
        // be reset from outside a transfer, so the whole lifecycle runs as
        // one ordered scenario (host tests execute on slot 0).
        #[test]
        fn retentive_transfer_lifecycle_and_guards() {
            // Guards with the substrate down: both handlers deny and leave
            // every context untouched.
            assert_eq!(
                handle_rdsm(FID_RDSM_TEERET, [TSM_READY, 0, 0, 0, 0, 0]).error,
                ERR_DENIED
            );
            assert_eq!(handle_teecall(EID_COVH, 0, [0; 6]).error, ERR_DENIED);
            assert_eq!(CONTEXTS.get().host.state(), ContextState::Empty);
            assert_eq!(CONTEXTS.get().tsm.state(), ContextState::Empty);

            // Simulated bring-up on the fake hart.
            {
                let ctx = context_mut();
                ctx.substrate_active = true;
                ctx.hart_id = 3;
                ctx.fdt_address = 0xf000_0000;
                ctx.host_entry_paddr = 0x8080_0100;
                ctx.tsm_entry_paddr = 0x8040_0000;
            }

            // A TEECALL before TSM_READY is rejected without side effects.
            assert_eq!(handle_teecall(EID_COVH, 0, [0; 6]).error, ERR_FAILED);
            assert_eq!(CONTEXTS.get().tsm.state(), ContextState::Empty);
            assert_eq!(context().dispatch_entry, None);

            // TSM_READY fabricates and parks the first host snapshot.
            let ret = handle_rdsm(FID_RDSM_TEERET, [TSM_READY, 0x8040_2000, 0, 0, 0, 0]);
            assert_eq!(ret.error, SBI_SUCCESS);
            assert_eq!(CONTEXTS.get().host.state(), ContextState::Suspended);
            let snap = CONTEXTS.get().host.snapshot();
            assert_eq!(snap.x1_x31[9], 3, "a0 carries the hart id");
            assert_eq!(snap.x1_x31[10], 0xf000_0000, "a1 carries the FDT address");
            assert_eq!(snap.pc, 0x8080_0100, "resume at the host entry");
            assert_eq!(snap.satp, Satp::from_bits(0));
            assert_eq!(context().dispatch_entry, Some(0x8040_2000));
            assert!(is_tsm_ready());

            // A second TSM_READY is refused: the host context is no longer
            // fresh (host snapshot parked).
            assert_eq!(
                handle_rdsm(FID_RDSM_TEERET, [TSM_READY, 0x8040_2000, 0, 0, 0, 0]).error,
                ERR_FAILED
            );

            // First TEECALL: dispatch entry consumed, call packed into a0-a7.
            let ret = handle_teecall(EID_COVH, CONVERT_PAGES, [0x8000_1000, 0x2000, 0, 0, 0, 0]);
            assert_eq!(ret.error, SBI_SUCCESS);
            let snap = CONTEXTS.get().tsm.snapshot();
            assert_eq!(&snap.x1_x31[9..15], &[0x8000_1000, 0x2000, 0, 0, 0, 0]);
            assert_eq!(snap.x1_x31[15], CONVERT_PAGES, "a6 carries the function id");
            assert_eq!(snap.x1_x31[16], EID_COVH, "a7 carries the extension id");
            assert_eq!(
                snap.pc, 0x8040_2000,
                "first dispatch uses the registered entry"
            );
            assert_eq!(context().dispatch_entry, None);

            // Second TEECALL resumes at the parked snapshot's breakpoint.
            let ret = handle_teecall(EID_COVH, CONVERT_PAGES, [0x8000_3000, 0x1000, 0, 0, 0, 0]);
            assert_eq!(ret.error, SBI_SUCCESS);
            let snap = CONTEXTS.get().tsm.snapshot();
            assert_eq!(&snap.x1_x31[9..15], &[0x8000_3000, 0x1000, 0, 0, 0, 0]);
            assert_eq!(snap.pc, 0x8040_2000, "resume at the parked breakpoint");

            // NORMAL_RETURN injects the TSM's result into the host's a0/a1
            // and keeps the host's own resume point.
            let ret = handle_rdsm(FID_RDSM_TEERET, [NORMAL_RETURN, 5, 0xdead_beef, 0, 0, 0]);
            assert_eq!(ret.error, SBI_SUCCESS);
            let snap = CONTEXTS.get().host.snapshot();
            assert_eq!(snap.x1_x31[9], 5, "a0 carries the TSM error");
            assert_eq!(snap.x1_x31[10], 0xdead_beef, "a1 carries the TSM value");
            assert_eq!(snap.pc, 0x8080_0100, "host resume point preserved");

            // Leave the substrate down for the rest of the test binary.
            context_mut().substrate_active = false;
        }
    }
}
