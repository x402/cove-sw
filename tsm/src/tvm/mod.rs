pub mod aia;
pub mod page_table;
pub mod vcpu;

use crate::SbiRet;
use crate::mm::{imsic_geometry, is_host_range, PAGE_SIZE, PageState, page_tracker};
use page_table::GStagePageTable;
use riscv_cove::host::TvmCreateParams;
use riscv_cove::interrupt::TvmAiaParams;
use spin::{Mutex, MutexGuard};
use vcpu::Vcpu;

/// COVI AIA configuration of one TVM (CoVE v0.7 §11.1–§11.2).
#[derive(Clone, Copy)]
pub struct TvmAiaState {
    pub enabled: bool,
    /// Host-supplied parameter block; meaningful only while `enabled`, in
    /// which case it has passed [`aia::validate`].
    pub params: TvmAiaParams,
    /// Index masks derived from `params`; meaningful only while `enabled`.
    pub layout: aia::AiaLayout,
}

impl TvmAiaState {
    pub const fn new() -> Self {
        Self {
            enabled: false,
            params: TvmAiaParams {
                imsic_base_addr: 0,
                group_index_bits: 0,
                group_index_shift: 0,
                hart_index_bits: 0,
                guest_index_bits: 0,
                guests_per_hart: 0,
            },
            layout: aia::AiaLayout::EMPTY,
        }
    }
}

/// Physical guest interrupt file a vCPU is bound to (CoVE §11.5): the
/// (domain, hart, guest) file page chosen from the platform geometry at
/// bind time, on the hart the bind ran on.
#[derive(Clone, Copy, Debug)]
pub struct ImsicBinding {
    pub file_paddr: usize,
    pub gpa: usize,
    pub vgein: usize,
}

/// External interrupt identity bounds: identity 0 is reserved (AIA §3.1)
/// and the model carries a single 64-bit eip/eie array per file.
pub const EXT_ID_MAX: usize = 63;

/// Hstatus.VGEIN field position/width (privileged spec).
const HSTATUS_VGEIN_SHIFT: usize = 12;
const HSTATUS_VGEIN_MASK: usize = 0x3F << HSTATUS_VGEIN_SHIFT;

/// Indirect-window selectors of one interrupt file (AIA v1.0 §3.7).
const ISELECT_EIDELIVERY: usize = 0x70;
const ISELECT_EITHRESHOLD: usize = 0x72;
const ISELECT_EIE0: usize = 0xC0;

/// Scrubs a guest interrupt file (CoVE v0.7 §5.2.4: the TSM must perform
/// scrubbing of confidential memory before returning control of the
/// memory to the host or assigning to another TVM) and claims its pending
/// identities (AIA v1.0 §3.9: writing xtopei claims the reported
/// identity). Runs in the TSM (confidential SIDN active) with
/// `hstatus.VGEIN` temporarily pointed at the file.
///
/// Order matters: a pending identity is only claimable while enabled (the
/// eie array gates topei, AIA v1.0 §3.7; eidelivery only gates the signal
/// line, §3.9), so eie is set to all-ones before the claim loop.
/// eithreshold is cleared first because a stale threshold would otherwise
/// keep high identities unclaimable. After claiming, the file is restored
/// to its reset state (eie = 0, eidelivery = 0) so a later bind cannot
/// inherit the previous tenant's interrupt state.
///
/// One 64-bit eie word at select 0xC0 covers every modeled identity
/// (EXT_ID_MAX = 63; the platform model collapses the real hardware's
/// 32-bit-per-word eip/eie array into a single word, so selects 0xC1+
/// are outside the modeled file window and are not touched here).
///
/// Returns the number of identities claimed from the file.
fn scrub_guest_file(vgein: usize) -> usize {
    let mut claimed = 0usize;
    unsafe {
        let mut hstatus: usize;
        core::arch::asm!("csrr {}, hstatus", out(reg) hstatus);
        let saved = hstatus & HSTATUS_VGEIN_MASK;
        hstatus = (hstatus & !HSTATUS_VGEIN_MASK) | (vgein << HSTATUS_VGEIN_SHIFT);
        core::arch::asm!("csrw hstatus, {}", in(reg) hstatus);

        core::arch::asm!("csrw vsiselect, {}", in(reg) ISELECT_EITHRESHOLD);
        core::arch::asm!("csrw vsireg, {}", in(reg) 0usize);
        core::arch::asm!("csrw vsiselect, {}", in(reg) ISELECT_EIE0);
        core::arch::asm!("csrw vsireg, {}", in(reg) usize::MAX);
        for _ in 0..=EXT_ID_MAX {
            let topei: usize;
            core::arch::asm!("csrr {}, vstopei", out(reg) topei);
            if topei == 0 {
                break;
            }
            core::arch::asm!("csrw vstopei, zero");
            claimed += 1;
        }
        core::arch::asm!("csrw vsiselect, {}", in(reg) ISELECT_EIE0);
        core::arch::asm!("csrw vsireg, {}", in(reg) 0usize);
        core::arch::asm!("csrw vsiselect, {}", in(reg) ISELECT_EIDELIVERY);
        core::arch::asm!("csrw vsireg, {}", in(reg) 0usize);

        core::arch::asm!("csrw hstatus, {}", in(reg) saved);
    }
    claimed
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TvmState {
    Initializing,
    Runnable,
    Destroyed,
}

#[derive(Clone, Copy, Debug)]
pub struct TvmMemoryRegion {
    pub gpa_base: usize,
    pub len: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct MmioRegion {
    pub gpa_base: usize,
    pub len: usize,
}

#[derive(Clone, Copy)]
pub struct SharedRegion {
    pub gpa_base: usize,
    pub len: usize,
}

pub struct Tvm {
    pub id: usize,
    pub state: TvmState,
    pub page_directory_addr: usize,
    pub state_addr: usize,
    pub vmid: usize,
    pub page_table: GStagePageTable,
    pub memory_regions: [Option<TvmMemoryRegion>; 8],
    pub mmio_regions: [Option<MmioRegion>; 8],
    pub shared_regions: [Option<SharedRegion>; 8],
    pub pt_pages: [usize; 64],
    pub pt_page_count: usize,
    pub pt_page_used: usize,
    pub vcpu: Option<Vcpu>,
    pub entry_sepc: usize,
    pub entry_arg: usize,
    pub identity: [u8; 64],
    /// COVI AIA configuration; inert until `init_tvm_aia` (FID 0).
    pub aia: TvmAiaState,
}

impl Tvm {
    pub fn new(id: usize, params: &TvmCreateParams) -> Self {
        let page_table = GStagePageTable::new(params.tvm_page_directory_addr);
        Self {
            id,
            state: TvmState::Initializing,
            page_directory_addr: params.tvm_page_directory_addr,
            state_addr: params.tvm_state_addr,
            vmid: id,
            page_table,
            memory_regions: [None; 8],
            mmio_regions: [None; 8],
            shared_regions: [None; 8],
            pt_pages: [0; 64],
            pt_page_count: 0,
            pt_page_used: 0,
            vcpu: None,
            entry_sepc: 0,
            entry_arg: 0,
            identity: [0; 64],
            aia: TvmAiaState::new(),
        }
    }
}

pub struct TvmManager {
    pub active_tvm: Option<Tvm>,
    pub next_tvm_id: usize,
}

impl TvmManager {
    pub const fn new() -> Self {
        Self {
            active_tvm: None,
            next_tvm_id: 1,
        }
    }

    pub fn create_tvm(&mut self, params_paddr: usize, params_len: usize) -> SbiRet {
        if self.active_tvm.is_some() {
            return SbiRet::failed(); // A TVM is already active; destroy it first
        }
        if params_paddr == 0 || params_len < core::mem::size_of::<TvmCreateParams>() {
            return SbiRet::invalid_param();
        }
        if params_paddr % core::mem::align_of::<TvmCreateParams>() != 0 {
            return SbiRet::invalid_address();
        }
        // The parameter block must live in host-allocatable memory.
        if !is_host_range(params_paddr, core::mem::size_of::<TvmCreateParams>()) {
            return SbiRet::invalid_address();
        }

        let params = unsafe { (*(params_paddr as *const TvmCreateParams)).clone() };

        if params.tvm_page_directory_addr % 16384 != 0 {
            return SbiRet::invalid_address();
        }
        if params.tvm_state_addr % PAGE_SIZE != 0 {
            return SbiRet::invalid_address();
        }

        let mut tracker = page_tracker();
        if !tracker.is_converted_and_free(params.tvm_page_directory_addr, 4) {
            return SbiRet::invalid_param();
        }
        if !tracker.is_converted_and_free(params.tvm_state_addr, 4) {
            return SbiRet::invalid_param();
        }

        let tvm_id = self.next_tvm_id;
        self.next_tvm_id += 1;

        tracker.assign_range(
            params.tvm_page_directory_addr,
            4,
            PageState::AssignedTvmRoot,
            tvm_id,
        );
        tracker.assign_range(
            params.tvm_state_addr,
            4,
            PageState::AssignedTvmState,
            tvm_id,
        );

        let tvm = Tvm::new(tvm_id, &params);
        self.active_tvm = Some(tvm);

        SbiRet::success(tvm_id)
    }

    pub fn add_memory_region(&mut self, tvm_id: usize, gpa: usize, len: usize) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };

        if tvm.state != TvmState::Initializing {
            return SbiRet::invalid_param();
        }
        if gpa % PAGE_SIZE != 0 || len % PAGE_SIZE != 0 || len == 0 {
            return SbiRet::invalid_param();
        }

        let end = match gpa.checked_add(len) {
            Some(e) => e,
            None => return SbiRet::invalid_param(),
        };

        for slot in tvm.memory_regions.iter() {
            if let Some(r) = slot {
                let r_end = r.gpa_base + r.len;
                if !(end <= r.gpa_base || gpa >= r_end) {
                    return SbiRet::invalid_param(); // Overlap
                }
            }
        }

        for slot in tvm.memory_regions.iter_mut() {
            if slot.is_none() {
                *slot = Some(TvmMemoryRegion { gpa_base: gpa, len });
                return SbiRet::success(0);
            }
        }

        SbiRet::failed()
    }

    pub fn add_page_table_pages(
        &mut self,
        tvm_id: usize,
        base_paddr: usize,
        num_pages: usize,
    ) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };

        if base_paddr % PAGE_SIZE != 0 || num_pages == 0 {
            return SbiRet::invalid_address();
        }

        let mut tracker = page_tracker();
        if !tracker.is_converted_and_free(base_paddr, num_pages) {
            return SbiRet::invalid_param();
        }

        if tvm.pt_page_count + num_pages > tvm.pt_pages.len() {
            return SbiRet::failed();
        }

        tracker.assign_range(base_paddr, num_pages, PageState::AssignedPageTable, tvm_id);

        for i in 0..num_pages {
            let paddr = base_paddr + i * PAGE_SIZE;
            unsafe {
                core::ptr::write_bytes(paddr as *mut u8, 0, PAGE_SIZE);
            }
            tvm.pt_pages[tvm.pt_page_count] = paddr;
            tvm.pt_page_count += 1;
        }

        SbiRet::success(0)
    }

    pub fn add_measured_pages(
        &mut self,
        tvm_id: usize,
        src_paddr: usize,
        dst_paddr: usize,
        page_type: usize,
        num_pages: usize,
        gpa: usize,
    ) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };

        if tvm.state != TvmState::Initializing {
            return SbiRet::invalid_param();
        }
        if page_type != 0 {
            return SbiRet::not_supported();
        }
        if src_paddr % PAGE_SIZE != 0
            || dst_paddr % PAGE_SIZE != 0
            || gpa % PAGE_SIZE != 0
            || num_pages == 0
        {
            return SbiRet::invalid_address();
        }
        // The source must be host-allocatable memory. Copying
        // from TSM-private or MPT-pool pages would leak confidential data
        // into guest-readable measured pages.
        let copy_len = match num_pages.checked_mul(PAGE_SIZE) {
            Some(l) => l,
            None => return SbiRet::invalid_param(),
        };
        if !is_host_range(src_paddr, copy_len) {
            return SbiRet::invalid_address();
        }

        // Verify GPA is in registered region
        let gpa_end = match gpa.checked_add(num_pages * PAGE_SIZE) {
            Some(e) => e,
            None => return SbiRet::invalid_param(),
        };

        let mut in_region = false;
        for slot in tvm.memory_regions.iter().flatten() {
            let r_end = slot.gpa_base + slot.len;
            if gpa >= slot.gpa_base && gpa_end <= r_end {
                in_region = true;
                break;
            }
        }
        if !in_region {
            return SbiRet::invalid_param();
        }

        let mut tracker = page_tracker();
        if !tracker.is_converted_and_free(dst_paddr, num_pages) {
            return SbiRet::invalid_param();
        }

        tracker.assign_range(dst_paddr, num_pages, PageState::AssignedPayload, tvm_id);

        let page_table = &tvm.page_table;
        let pt_pages = &tvm.pt_pages;
        let pt_page_used = &mut tvm.pt_page_used;
        let pt_page_count = tvm.pt_page_count;

        for i in 0..num_pages {
            let s_addr = src_paddr + i * PAGE_SIZE;
            let d_addr = dst_paddr + i * PAGE_SIZE;
            let cur_gpa = gpa + i * PAGE_SIZE;

            // Copy contents
            unsafe {
                core::ptr::copy_nonoverlapping(s_addr as *const u8, d_addr as *mut u8, PAGE_SIZE);
            }

            // Map into GStagePageTable
            let mut alloc_fn = || {
                if *pt_page_used < pt_page_count {
                    let p = pt_pages[*pt_page_used];
                    *pt_page_used += 1;
                    Some(p)
                } else {
                    None
                }
            };
            if page_table.map_4k(cur_gpa, d_addr, &mut alloc_fn).is_err() {
                return SbiRet::failed();
            }
        }

        SbiRet::success(0)
    }

    pub fn add_zero_pages(
        &mut self,
        tvm_id: usize,
        dst_paddr: usize,
        page_type: usize,
        num_pages: usize,
        gpa: usize,
    ) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };

        if tvm.state != TvmState::Initializing {
            return SbiRet::invalid_param();
        }
        if page_type != 0 {
            return SbiRet::not_supported();
        }
        if dst_paddr % PAGE_SIZE != 0 || gpa % PAGE_SIZE != 0 || num_pages == 0 {
            return SbiRet::invalid_address();
        }

        let mut tracker = page_tracker();
        if !tracker.is_converted_and_free(dst_paddr, num_pages) {
            return SbiRet::invalid_param();
        }

        tracker.assign_range(dst_paddr, num_pages, PageState::AssignedPayload, tvm_id);

        let page_table = &tvm.page_table;
        let pt_pages = &tvm.pt_pages;
        let pt_page_used = &mut tvm.pt_page_used;
        let pt_page_count = tvm.pt_page_count;

        for i in 0..num_pages {
            let d_addr = dst_paddr + i * PAGE_SIZE;
            let cur_gpa = gpa + i * PAGE_SIZE;

            unsafe {
                core::ptr::write_bytes(d_addr as *mut u8, 0, PAGE_SIZE);
            }

            let mut alloc_fn = || {
                if *pt_page_used < pt_page_count {
                    let p = pt_pages[*pt_page_used];
                    *pt_page_used += 1;
                    Some(p)
                } else {
                    None
                }
            };
            if page_table.map_4k(cur_gpa, d_addr, &mut alloc_fn).is_err() {
                return SbiRet::failed();
            }
        }

        SbiRet::success(0)
    }

    // ---- COVI handlers (CoVE v0.7 chapter 11) ----

    /// FID 0 `init_tvm_aia`: validates the host-supplied block as the
    /// TVM's virtual IMSIC layout (§11.1) and arms the TVM's AIA
    /// configuration. Only while Initializing. Before finalize the call may
    /// be repeated, but a new block must keep every recorded vIMSIC GPA
    /// inside its virtual array.
    pub fn init_tvm_aia(&mut self, tvm_id: usize, params_paddr: usize, params_len: usize) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };
        if tvm.state != TvmState::Initializing {
            return SbiRet::invalid_param();
        }
        if params_paddr == 0 || params_len < core::mem::size_of::<TvmAiaParams>() {
            return SbiRet::invalid_param();
        }
        if params_paddr % core::mem::align_of::<TvmAiaParams>() != 0
            || !is_host_range(params_paddr, core::mem::size_of::<TvmAiaParams>())
        {
            return SbiRet::invalid_address();
        }

        // AIA virtualization needs a platform IMSIC to supply physical
        // guest interrupt files. The block itself describes the TVM's own
        // virtual address space and is validated independently below.
        if imsic_geometry().is_none() {
            return SbiRet::not_supported();
        }
        let params = unsafe { (*(params_paddr as *const TvmAiaParams)).clone() };

        // Table 33 names no code for malformed fields inside the block, so
        // every violation is the project's invalid_param mapping.
        let layout = match aia::validate(&params) {
            Ok(layout) => layout,
            Err(_) => return SbiRet::invalid_param(),
        };
        // TSM rule, not a CoVE requirement: the TVM's vIMSIC address must be
        // representable in the Sv39x4 G-stage (the same aperture the page
        // table enforces), so a base at or above 2^41 could never be bound.
        if params.imsic_base_addr >= (1usize << 41) {
            return SbiRet::invalid_param();
        }

        // §11.1 only forbids calls after sbi_covh_finalize_tvm(), so a
        // second call while Initializing is legal. It must not orphan a
        // recorded vIMSIC GPA: the candidate array has to contain it. The
        // manager holds a single vCPU slot today; with several vCPUs this
        // becomes an iteration over all of them.
        if let Some(gpa) = tvm.vcpu.as_ref().and_then(|v| v.imsic_gpa) {
            if !layout.contains_gpa(params.imsic_base_addr, gpa) {
                return SbiRet::invalid_param();
            }
        }

        tvm.aia = TvmAiaState {
            enabled: true,
            params,
            layout,
        };
        SbiRet::success(0)
    }

    /// FID 1 `set_tvm_aia_cpu_imsic_addr`: records the vIMSIC GPA for one
    /// vCPU. Only after FID 0 and while Initializing. The GPA must be a
    /// valid per-vCPU base of the configured virtual array, a page inside a
    /// registered memory region, and not owned by another vCPU (§11.2).
    pub fn set_tvm_aia_cpu_imsic_addr(
        &mut self,
        tvm_id: usize,
        vcpu_id: usize,
        gpa: usize,
    ) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };
        if tvm.state != TvmState::Initializing || !tvm.aia.enabled {
            return SbiRet::invalid_param();
        }
        // Resolve the vCPU before validating the GPA, so an unknown id is
        // reported as such rather than as an address error.
        if !matches!(tvm.vcpu.as_ref(), Some(v) if v.id == vcpu_id) {
            return SbiRet::invalid_param();
        }

        // GPA syntax: a page inside the Sv39x4 G-stage aperture. This bound
        // makes the page end below 2^41, so the addition cannot overflow.
        if gpa % PAGE_SIZE != 0 || gpa >= (1usize << 41) {
            return SbiRet::invalid_address();
        }
        let end = gpa + PAGE_SIZE;
        // §11.2: the GPA must be valid for the AIA configuration armed by
        // FID 0, i.e. one (hart, guest) entry of its virtual array.
        if !tvm
            .aia
            .layout
            .contains_gpa(tvm.aia.params.imsic_base_addr, gpa)
        {
            return SbiRet::invalid_address();
        }
        // TSM rule, not a CoVE requirement: the whole page must already be
        // backed by one registered TVM memory region before it can be
        // presented to the guest as a device page.
        if !tvm
            .memory_regions
            .iter()
            .flatten()
            .any(|region| gpa >= region.gpa_base && end <= region.gpa_base + region.len)
        {
            return SbiRet::invalid_address();
        }
        // §11.2: no two vCPUs may share the same vIMSIC GPA. Table 34 names
        // no code for a duplicate GPA; invalid_param is the project mapping.
        // The manager holds a single vCPU slot today; with several vCPUs this
        // becomes an iteration over all of them.
        if tvm
            .vcpu
            .as_ref()
            .is_some_and(|v| v.id != vcpu_id && v.imsic_gpa == Some(gpa))
        {
            return SbiRet::invalid_param();
        }

        let vcpu = match tvm.vcpu.as_mut() {
            Some(v) if v.id == vcpu_id => v,
            _ => return SbiRet::invalid_param(),
        };
        vcpu.imsic_gpa = Some(gpa);
        SbiRet::success(0)
    }

    /// Physical address of guest interrupt file `vgein` (≥ 1) of `hart` in
    /// the confidential domain's supervisor interrupt domain (SmmTT §6.1:
    /// `B + n·2^I + h·2^hart_shift + vgein·2^12`). `None` when the geometry
    /// cannot cover the request.
    fn guest_file_paddr(geometry: crate::rdsm_shim::RdsmImsicGeometry, hart: usize, vgein: usize) -> Option<usize> {
        let guests = 1usize << geometry.guest_index_bits;
        if vgein == 0 || vgein >= guests {
            return None;
        }
        let harts = 1usize << geometry.hart_index_bits;
        if hart >= harts {
            return None;
        }
        let hart_shift = 12 + geometry.guest_index_bits;
        let domains = (geometry.range_end - geometry.range_start)
            .checked_shr(geometry.domain_index_shift as u32)?;
        if domains == 0 {
            return None;
        }
        // Single-SIDN pairing: init registers sidn0 -> host, sidn1 ->
        // confidential; the confidential domain owns domain block 1. A
        // per-domain allocator would replace the constant.
        const CONF_SIDN: usize = 1;
        if CONF_SIDN + 1 > domains {
            return None;
        }
        Some(
            geometry.range_start
                + CONF_SIDN * (1 << geometry.domain_index_shift)
                + hart * (1 << hart_shift)
                + vgein * PAGE_SIZE,
        )
    }

    /// FID 4 `bind_aia_imsic`: binds an unbound vCPU, running on the
    /// calling hart, to the confidential domain's guest interrupt file for
    /// that hart (§11.5). `imsic_mask` follows the hgeie format: bit 0
    /// always 0 and exactly `1 + guests_per_hart` bits set — with the single
    /// guest interrupt file per vCPU this implementation supports
    /// (`guests_per_hart == 0`) the only legal mask is bit 1 (G0, VGEIN 1).
    pub fn bind_aia_imsic(&mut self, tvm_id: usize, vcpu_id: usize, imsic_mask: usize) -> SbiRet {
        let geometry = match imsic_geometry() {
            Some(g) => g,
            None => return SbiRet::not_supported(),
        };
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };
        if !tvm.aia.enabled {
            return SbiRet::invalid_param();
        }
        if imsic_mask != 1 << 1 {
            // §11.5: popcount must equal 1 + guests_per_hart; bit 0 is 0.
            return SbiRet::invalid_param();
        }
        let vcpu = match tvm.vcpu.as_mut() {
            Some(v) if v.id == vcpu_id => v,
            _ => return SbiRet::invalid_param(),
        };
        if vcpu.imsic_binding.is_some() {
            // §11.5: the vCPU must currently be unbound.
            return SbiRet::invalid_param();
        }
        let gpa = match vcpu.imsic_gpa {
            Some(g) => g,
            None => return SbiRet::invalid_param(),
        };

        let hart = crate::current_hart_id();
        let file_paddr = match Self::guest_file_paddr(geometry, hart, 1) {
            Some(p) => p,
            None => return SbiRet::invalid_param(),
        };

        // The file must have been converted (COVI FID 2) and fenced, and
        // not owned by any TVM. Lock order: TVM_MANAGER -> PAGE_TRACKER.
        let mut tracker = page_tracker();
        if let SbiRet { error, .. } = tracker.bind_imsic_page(file_paddr, tvm_id, vcpu_id, gpa) {
            if error != 0 {
                return SbiRet::invalid_param();
            }
        }

        // Map the vIMSIC GPA onto the file in the G-stage: device leaf,
        // never executable. The PT pool belongs to this TVM.
        let mut pt_used = tvm.pt_page_used;
        let pt_count = tvm.pt_page_count;
        let mut alloc = || {
            if pt_used < pt_count {
                let p = tvm.pt_pages[pt_used];
                pt_used += 1;
                Some(p)
            } else {
                None
            }
        };
        if tvm.page_table.map_4k_io(gpa, file_paddr, &mut alloc).is_err() {
            let _ = tracker.release_imsic_file(file_paddr, tvm_id);
            return SbiRet::failed();
        }
        tvm.pt_page_used = pt_used;

        // VGEIN selects the guest file inside the active supervisor
        // interrupt domain (SmmTT §6.3.1): the confidential domain is paired
        // with VGEIN 1.
        let vgein = 1;
        vcpu.ctx.hstatus = (vcpu.ctx.hstatus & !HSTATUS_VGEIN_MASK) | (vgein << HSTATUS_VGEIN_SHIFT);
        // Enable the file's identity line at the hart: hgeie is one
        // element of the per-SIDN array and the TSM runs inside the
        // confidential domain (sidn1), so this write is scoped to it.
        // Delegate the VS external interrupt line into VS and open the
        // Smstateen gates the guest's indirect-window accesses pass
        // through (CSRIND selects, AIA stopi, IMSIC topei).
        unsafe {
            core::arch::asm!("csrs hgeie, {}", in(reg) 1usize << vgein);
            core::arch::asm!("csrs hideleg, {}", in(reg) 1usize << 10); // VSEIP
        }
        // Software pending file (§11.8 unbound record) is replayed into
        // the hardware file on bind: seteipnum_le takes one identity per
        // write.
        let pending = vcpu.ctx.ext_int_pending;
        if pending != 0 {
            for iid in 1..=EXT_ID_MAX {
                if pending & (1 << iid) != 0 {
                    unsafe {
                        core::ptr::write_volatile(file_paddr as *mut u32, iid as u32);
                    }
                }
            }
            vcpu.ctx.ext_int_pending = 0;
        }

        vcpu.imsic_binding = Some(ImsicBinding {
            file_paddr,
            gpa,
            vgein,
        });
        SbiRet::success(0)
    }

    /// FID 7 `inject_tvm_cpu`: injects external interrupt `iid` into one
    /// vCPU (§11.8). The identity must be guest-allowed (COVG FID 4;
    /// default deny, §12.5) — the project maps a denied identity to
    /// `invalid_param`, the table's invalid-input code for §11.8. A bound
    /// vCPU receives the MSI through its hardware file; an unbound one
    /// records it in the software pending file until bind.
    pub fn inject_tvm_cpu(&mut self, tvm_id: usize, vcpu_id: usize, iid: usize) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };
        if !tvm.aia.enabled {
            return SbiRet::not_supported();
        }
        let vcpu = match tvm.vcpu.as_mut() {
            Some(v) if v.id == vcpu_id => v,
            _ => return SbiRet::invalid_param(),
        };
        if iid == 0 || iid > EXT_ID_MAX {
            return SbiRet::invalid_param();
        }
        if vcpu.ctx.ext_int_allowed & (1 << iid) == 0 {
            return SbiRet::invalid_param();
        }
        match vcpu.imsic_binding {
            Some(binding) => {
                // §11.8: immediately injected by writing the interrupt file.
                unsafe {
                    core::ptr::write_volatile(binding.file_paddr as *mut u32, iid as u32);
                }
            }
            None => {
                // §11.8: recorded in software until the vCPU binds.
                vcpu.ctx.ext_int_pending |= 1 << iid;
            }
        }
        SbiRet::success(0)
    }

    pub fn create_tvm_vcpu(&mut self, tvm_id: usize, vcpu_id: usize, state_paddr: usize) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };

        if tvm.state != TvmState::Initializing {
            return SbiRet::invalid_param();
        }
        if vcpu_id != 0 {
            return SbiRet::invalid_param();
        }
        if state_paddr % PAGE_SIZE != 0 {
            return SbiRet::invalid_address();
        }

        let mut tracker = page_tracker();
        if !tracker.is_converted_and_free(state_paddr, 2) {
            return SbiRet::invalid_param();
        }

        tracker.assign_range(state_paddr, 2, PageState::AssignedVcpuState, tvm_id);

        unsafe {
            core::ptr::write_bytes(state_paddr as *mut u8, 0, 2 * PAGE_SIZE);
        }

        tvm.vcpu = Some(Vcpu::new(vcpu_id, state_paddr));

        SbiRet::success(0)
    }

    pub fn finalize_tvm(
        &mut self,
        tvm_id: usize,
        entry_sepc: usize,
        entry_arg: usize,
        identity_addr: usize,
    ) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };

        if tvm.state != TvmState::Initializing {
            return SbiRet::invalid_param();
        }

        let hgatp = tvm.page_table.make_hgatp(tvm.vmid);

        let vcpu = match tvm.vcpu.as_mut() {
            Some(v) => v,
            None => return SbiRet::failed(),
        };

        // §11.2: every vCPU of an AIA-enabled TVM must have its vIMSIC
        // GPA configured before finalize.
        if tvm.aia.enabled && vcpu.imsic_gpa.is_none() {
            return SbiRet::failed();
        }

        vcpu.init_boot(entry_sepc, entry_arg, hgatp);

        tvm.entry_sepc = entry_sepc;
        tvm.entry_arg = entry_arg;

        if identity_addr != 0 {
            if identity_addr % 64 != 0 {
                return SbiRet::invalid_address();
            }
            // The identity block must be host-allocatable memory.
            if !is_host_range(identity_addr, 64) {
                return SbiRet::invalid_address();
            }
            unsafe {
                core::ptr::copy_nonoverlapping(
                    identity_addr as *const u8,
                    tvm.identity.as_mut_ptr(),
                    64,
                );
            }
        }

        tvm.state = TvmState::Runnable;

        SbiRet::success(0)
    }

    // ---- COVG internal handlers (called from vcpu trap handler) ----

    pub fn add_mmio_region_internal(&mut self, gpa: usize, len: usize) -> bool {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) => t,
            None => return false,
        };
        if gpa % PAGE_SIZE != 0 || len % PAGE_SIZE != 0 || len == 0 {
            return false;
        }
        for slot in tvm.mmio_regions.iter_mut() {
            if slot.is_none() {
                *slot = Some(MmioRegion { gpa_base: gpa, len });
                return true;
            }
        }
        false
    }

    pub fn remove_mmio_region_internal(&mut self, gpa: usize, len: usize) -> bool {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) => t,
            None => return false,
        };
        let end = match gpa.checked_add(len) {
            Some(e) => e,
            None => return false,
        };
        let mut removed = false;
        for slot in tvm.mmio_regions.iter_mut() {
            if let Some(r) = slot {
                let r_end = r.gpa_base + r.len;
                if !(end <= r.gpa_base || gpa >= r_end) {
                    *slot = None;
                    removed = true;
                }
            }
        }
        removed
    }

    pub fn share_memory_internal(&mut self, gpa: usize, len: usize) -> bool {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) => t,
            None => return false,
        };
        if gpa % PAGE_SIZE != 0 || len % PAGE_SIZE != 0 || len == 0 {
            return false;
        }
        let num = len / PAGE_SIZE;
        let shared_slot = match tvm.shared_regions.iter_mut().find(|slot| slot.is_none()) {
            Some(slot) => slot,
            None => return false,
        };
        let mut all_ok = true;
        for i in 0..num {
            let cur_gpa = gpa + i * PAGE_SIZE;
            // Look up SPA from G-stage
            let spa = match tvm.page_table.lookup_4k(cur_gpa) {
                Ok(s) => s,
                Err(_) => {
                    all_ok = false;
                    continue;
                }
            };
            // Preserve ownership and make the yielded page visible to the tracker.
            page_tracker().set_shared(spa, cur_gpa, tvm.id);
            // Unmap from G-stage (guest loses access)
            let _ = tvm.page_table.unmap_4k(cur_gpa);
            // Grant Host (SDID=0) RW access to the SPA
            let (err, _) = crate::rdsm_shim::rdsm_mpt_set(0, spa, PAGE_SIZE, 3);
            if err != 0 {
                all_ok = false;
            }
        }
        if all_ok {
            *shared_slot = Some(SharedRegion { gpa_base: gpa, len });
        }
        all_ok
    }

    pub fn unshare_memory_internal(&mut self, gpa: usize, len: usize) -> bool {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) => t,
            None => return false,
        };
        if gpa % PAGE_SIZE != 0 || len % PAGE_SIZE != 0 || len == 0 {
            return false;
        }
        if gpa % PAGE_SIZE != 0 || len == 0 || len % PAGE_SIZE != 0 {
            return false;
        }
        let slot_index = tvm
            .shared_regions
            .iter()
            .position(|region| matches!(region, Some(r) if r.gpa_base == gpa && r.len == len));
        let slot_index = match slot_index {
            Some(index) => index,
            None => return false,
        };
        let region = tvm.shared_regions[slot_index].unwrap();
        let num_pages = len / PAGE_SIZE;
        let pt_pages = tvm.pt_pages;
        let pt_page_count = tvm.pt_page_count;
        let pt_page_used = &mut tvm.pt_page_used;
        let page_table = &tvm.page_table;

        for i in 0..num_pages {
            let cur_gpa = region.gpa_base + i * PAGE_SIZE;
            let spa = match page_tracker().find_shared_by_gpa(cur_gpa, tvm.id) {
                Some(spa) => spa,
                None => return false,
            };
            if crate::rdsm_shim::rdsm_mpt_set(0, spa, PAGE_SIZE, 0).0 != 0 {
                return false;
            }
            if !page_tracker().clear_shared(spa, tvm.id) {
                return false;
            }
            let mut alloc_fn = || {
                if *pt_page_used < pt_page_count {
                    let page = pt_pages[*pt_page_used];
                    *pt_page_used += 1;
                    Some(page)
                } else {
                    None
                }
            };
            if page_table.map_4k(cur_gpa, spa, &mut alloc_fn).is_err() {
                return false;
            }
        }

        let _ = crate::rdsm_shim::rdsm_mfence_pa(0, 0);
        unsafe {
            core::arch::asm!("hfence.gvma");
        }
        tvm.shared_regions[slot_index] = None;
        true
    }

    pub fn handle_demand_zero(&mut self, fault_gpa: usize) -> bool {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) => t,
            None => return false,
        };

        // Verify fault GPA is within a registered memory region
        let in_region = tvm
            .memory_regions
            .iter()
            .flatten()
            .any(|r| fault_gpa >= r.gpa_base && fault_gpa < r.gpa_base + r.len);
        if !in_region {
            return false;
        }

        // Find a free page from the tracker
        let mut tracker = page_tracker();
        let free_page = match tracker.find_free_page() {
            Some(p) => p,
            None => return false,
        };

        // Assign it to this TVM
        if !tracker.assign_page(free_page, PageState::AssignedPayload, tvm.id) {
            return false;
        }

        // Zero-fill the page (demand-zero semantics)
        unsafe {
            core::ptr::write_bytes(free_page as *mut u8, 0, PAGE_SIZE);
        }

        // Map in G-stage
        let page_table = &tvm.page_table;
        let pt_pages = tvm.pt_pages;
        let pt_page_count = tvm.pt_page_count;
        let pt_page_used = &mut tvm.pt_page_used;

        let mut alloc_fn = || {
            if *pt_page_used < pt_page_count {
                let p = pt_pages[*pt_page_used];
                *pt_page_used += 1;
                Some(p)
            } else {
                None
            }
        };
        page_table
            .map_4k(fault_gpa, free_page, &mut alloc_fn)
            .is_ok()
    }

    // ---- New COVH FIDs ----

    pub fn add_shared_pages(
        &mut self,
        tvm_id: usize,
        src_paddr: usize,
        dst_paddr: usize,
        num_pages: usize,
        gpa: usize,
    ) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };
        if gpa % PAGE_SIZE != 0 || num_pages == 0 {
            return SbiRet::invalid_address();
        }
        // Shared pages must live in host-allocatable memory.
        // Mapping TSM-private or MPT-pool pages into the guest would grant
        // it read/write access to confidential memory through the host MPT.
        let shared_len = match num_pages.checked_mul(PAGE_SIZE) {
            Some(l) => l,
            None => return SbiRet::invalid_param(),
        };
        if !is_host_range(dst_paddr, shared_len) {
            return SbiRet::invalid_address();
        }
        let page_table = &tvm.page_table;
        let pt_pages = tvm.pt_pages;
        let pt_page_count = tvm.pt_page_count;
        let pt_page_used = &mut tvm.pt_page_used;

        for i in 0..num_pages {
            let cur_gpa = gpa + i * PAGE_SIZE;
            let spa = dst_paddr + i * PAGE_SIZE;
            let mut alloc_fn = || {
                if *pt_page_used < pt_page_count {
                    let p = pt_pages[*pt_page_used];
                    *pt_page_used += 1;
                    Some(p)
                } else {
                    None
                }
            };
            if page_table.map_4k(cur_gpa, spa, &mut alloc_fn).is_err() {
                return SbiRet::failed();
            }
        }
        let _ = src_paddr; // src is the Host-side physical addr, already accessible
        SbiRet::success(0)
    }

    pub fn tvm_fence(&mut self, tvm_id: usize) -> SbiRet {
        let _tvm = match self.active_tvm.as_ref() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };
        unsafe {
            core::arch::asm!("hfence.gvma");
        }
        SbiRet::success(0)
    }

    pub fn tvm_invalidate_pages(&mut self, tvm_id: usize, gpa: usize, length: usize) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };
        if gpa % PAGE_SIZE != 0 || length == 0 || length % PAGE_SIZE != 0 {
            return SbiRet::invalid_address();
        }
        let num_pages = length / PAGE_SIZE;
        for i in 0..num_pages {
            let cur_gpa = gpa + i * PAGE_SIZE;
            if tvm.page_table.invalidate_4k(cur_gpa).is_err() {
                return SbiRet::invalid_address();
            }
        }
        unsafe {
            core::arch::asm!("hfence.gvma");
        }
        SbiRet::success(0)
    }

    pub fn tvm_validate_pages(&mut self, tvm_id: usize, gpa: usize, length: usize) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };
        if gpa % PAGE_SIZE != 0 || length == 0 || length % PAGE_SIZE != 0 {
            return SbiRet::invalid_address();
        }

        let num_pages = length / PAGE_SIZE;
        for i in 0..num_pages {
            if tvm.page_table.validate_4k(gpa + i * PAGE_SIZE).is_err() {
                return SbiRet::invalid_address();
            }
        }
        unsafe {
            core::arch::asm!("hfence.gvma");
        }
        SbiRet::success(0)
    }

    pub fn tvm_remove_pages(&mut self, tvm_id: usize, gpa: usize, length: usize) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };
        if gpa % PAGE_SIZE != 0 || length == 0 || length % PAGE_SIZE != 0 {
            return SbiRet::invalid_address();
        }

        // Find and release the SPA pages back to the free pool
        let mut tracker = page_tracker();
        let num_pages = length / PAGE_SIZE;
        for i in 0..num_pages {
            let cur_gpa = gpa + i * PAGE_SIZE;
            // Removal is only legal after invalidate + fence, so require a retained PTE.
            if let Ok(spa) = tvm.page_table.remove_invalid_4k(cur_gpa) {
                // Return page to free pool
                tracker.release_page(spa, tvm_id);
            } else {
                return SbiRet::invalid_address();
            }
        }
        unsafe {
            core::arch::asm!("hfence.gvma");
        }
        SbiRet::success(0)
    }

    pub fn destroy_tvm(&mut self, tvm_id: usize) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };
        if tvm.aia.enabled {
            // Implicit unbind: destroy has no unbind FIDs, so the bound file
            // is scrubbed (identity lines cleared, pending identities
            // claimed), withdrawn from the hart's hgeie line, unmapped from
            // the G-stage, and returned to the unassigned pool.
            if let Some(binding) = tvm.vcpu.as_ref().and_then(|v| v.imsic_binding) {
                let claimed = scrub_guest_file(binding.vgein);
                crate::println!(
                    "[TSM] COVI guest-file scrub claimed {} identit{}.",
                    claimed,
                    if claimed == 1 { "y" } else { "ies" }
                );
                unsafe {
                    core::arch::asm!("csrc hgeie, {}", in(reg) 1usize << binding.vgein);
                }
                let _ = tvm.page_table.unmap_4k(binding.gpa);
                unsafe {
                    core::arch::asm!("hfence.gvma");
                }
                if let Some(vcpu) = tvm.vcpu.as_mut() {
                    vcpu.ctx.hstatus &= !HSTATUS_VGEIN_MASK;
                    vcpu.ctx.ext_int_pending = 0;
                    vcpu.ctx.ext_int_allowed = 0;
                    vcpu.imsic_binding = None;
                }
            }
        }

        let mut tracker = page_tracker();
        tracker.release_tvm_pages(tvm_id);

        self.active_tvm = None;

        SbiRet::success(0)
    }
}

/// Shared TVM manager.
///
/// The mutex is only ever held for short configuration operations
/// (create/finalize/destroy/map/unmap …). It must **never** be held across
/// the vCPU run loop: a guest trap re-enters [`tvm_manager`] from the same
/// hart and would self-deadlock (see [`run_tvm_vcpu`]).
static TVM_MANAGER: Mutex<TvmManager> = Mutex::new(TvmManager::new());

/// Locks and returns the shared TVM manager.
pub fn tvm_manager() -> MutexGuard<'static, TvmManager> {
    TVM_MANAGER.lock()
}

/// Runs a TVM vCPU until its next exit.
///
/// Lock-boundary red line: no lock may be held while the guest
/// runs, so the vCPU is detached from the manager for the duration of the
/// run and re-attached after the exit. Guest-exit state lives in the
/// vCPU's own context (per-hart state), not in shared globals.
pub fn run_tvm_vcpu(tvm_id: usize, vcpu_id: usize) -> SbiRet {
    let mut vcpu = {
        let mut mgr = tvm_manager();
        let tvm = match mgr.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };
        if tvm.state != TvmState::Runnable {
            return SbiRet::invalid_param();
        }
        match tvm.vcpu.as_ref() {
            Some(v) if v.id == vcpu_id => {}
            _ => return SbiRet::invalid_param(),
        }
        // §7.4.2: running an unbound (AIA) vCPU is an error.
        if tvm.aia.enabled && tvm.vcpu.as_ref().and_then(|v| v.imsic_binding).is_none() {
            return SbiRet::invalid_param();
        }
        tvm.vcpu.take().expect("vcpu presence checked above")
    };

    let res = vcpu.run();
    let exit_reason = vcpu.ctx.last_exit.reason;

    let mut mgr = tvm_manager();
    match mgr.active_tvm.as_mut() {
        // The TVM must still be the one the vCPU was detached from; a
        // concurrent destroy while detached is a torn state, reported as
        // a failed run.
        Some(t) if t.id == tvm_id => t.vcpu = Some(vcpu),
        _ => return SbiRet::failed(),
    }
    if res == 0 {
        SbiRet::success(exit_reason)
    } else {
        SbiRet::failed()
    }
}
