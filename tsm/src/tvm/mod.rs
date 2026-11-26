pub mod vcpu;

use riscv_cove::host::TvmCreateParams;
use riscv_cove::interrupt::TvmAiaParams;
use spin::{Mutex, MutexGuard};
use tsm::aia;
use tsm::mm::{
    PAGE_SIZE, PageState, PageTracker, RangeStatus, imsic_geometry, is_host_range,
    page_tracker, platform,
};
use tsm::page_table::{self, GStagePageTable, leaf_is_present, leaf_pa};
use tsm::sbi::SbiRet;
use tsm::share::{
    self, SharedRegion, UnsharePlanError, YieldFailure, apply_unshare, plan_unshare,
};
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
    /// Physical hart the bind ran on (§11.5: the vCPU is bound to the
    /// current physical CPU). Recorded so run and destroy can tell whether
    /// they execute on that hart.
    pub hart_id: usize,
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
/// The whole `hstatus` and `vsiselect` are saved before the window is
/// redirected: only the VGEIN field of the temporary value differs from
/// the saved `hstatus`, so every other hypervisor bit survives. Both CSRs
/// are restored on the way out, `hstatus` first, so no `vsireg`/`vstopei`
/// access runs through the wrong window.
///
/// Returns the number of identities claimed from the file.
fn scrub_guest_file(vgein: usize) -> usize {
    let mut claimed = 0usize;
    unsafe {
        let mut hstatus: usize;
        core::arch::asm!("csrr {}, hstatus", out(reg) hstatus);
        let saved_hstatus = hstatus;
        let mut saved_vsiselect: usize;
        core::arch::asm!("csrr {}, vsiselect", out(reg) saved_vsiselect);
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

        core::arch::asm!("csrw hstatus, {}", in(reg) saved_hstatus);
        core::arch::asm!("csrw vsiselect, {}", in(reg) saved_vsiselect);
    }
    claimed
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TvmState {
    Initializing,
    Runnable,
    /// A share could not be rolled back and the TVM lost a page of its
    /// memory. It must never run again.
    Poisoned,
    Destroyed,
}

/// Outcome of a COVG share request (§12.3). Only [`ShareOutcome::Shared`] and
/// [`ShareOutcome::Poisoned`] reach the host as an exit; a refused or failed
/// request is answered to the guest itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShareOutcome {
    Shared,
    /// Table 46 INVALID_ADDRESS: the guest physical address is unusable.
    BadAddress,
    /// Table 46 INVALID_PARAM: the length or the request state is invalid.
    BadParam,
    Failed,
    Poisoned,
}

/// Outcome of a COVG unshare request (§12.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnshareOutcome {
    Unshared,
    /// Table 47 INVALID_ADDRESS: the guest physical address is unusable.
    BadAddress,
    /// Table 47 INVALID_PARAM: the length, the range or the request state is
    /// invalid.
    BadParam,
    Failed,
}

/// What the demand-zero handler did with a guest fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemandZero {
    /// A fresh confidential page was mapped; resume the guest.
    Served,
    /// The page is host-owned: it lies in a shared region or an MMIO
    /// region, so the host must serve the fault.
    HostPage,
    /// Not a demand-zero candidate (outside memory, occupied, no pool page).
    Refused,
}

/// Classification of one guest page for the invalidate/validate/remove trio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SharedPage {
    Absent,
    Present,
    Retained,
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

    /// COVH convert_pages (§10.3): takes host memory out of the host domain
    /// so the host can assign it to a TVM.
    ///
    /// A page the live TVM maps is host memory the guest still writes to (a
    /// shared page mapped by COVH FID 13). Converting it would make it
    /// assignable as page-table or payload memory while that guest mapping
    /// survives, so the call is refused. Table 13 names no code for this
    /// state; invalid_param is the project mapping. The manager holds a
    /// single TVM today; with several, every live TVM's page table would be
    /// checked.
    pub fn convert_pages(&mut self, base_paddr: usize, num_pages: usize) -> SbiRet {
        if base_paddr % PAGE_SIZE == 0 && num_pages != 0 {
            if let Some(len) = num_pages.checked_mul(PAGE_SIZE) {
                if base_paddr.checked_add(len).is_some() {
                    if let Some(tvm) = self.active_tvm.as_ref() {
                        if tvm.page_table.maps_any(base_paddr, len) {
                            return SbiRet::invalid_param();
                        }
                    }
                }
            }
        }

        let Some(layout) = platform() else {
            return SbiRet::failed();
        };
        let mut mpt = crate::rdsm_shim::RdsmMpt;
        page_tracker().convert_pages(layout, &mut mpt, base_paddr, num_pages)
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
        // Table 24 covers an invalid tsm_page_type with INVALID_PARAM. Only
        // 4 KiB pages are implemented: the larger enum values are legal but
        // unsupported and answer not_supported; anything else is not a legal
        // enum value.
        match page_type {
            0 => {}
            1..=3 => return SbiRet::not_supported(),
            _ => return SbiRet::invalid_param(),
        }
        // Table 24: an unusable address is INVALID_ADDRESS, an invalid page
        // count is INVALID_PARAM.
        if src_paddr % PAGE_SIZE != 0 || dst_paddr % PAGE_SIZE != 0 || gpa % PAGE_SIZE != 0 {
            return SbiRet::invalid_address();
        }
        if num_pages == 0 {
            return SbiRet::invalid_param();
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
        let gpa_end = match gpa.checked_add(copy_len) {
            Some(e) => e,
            None => return SbiRet::invalid_address(),
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
            return SbiRet::invalid_address();
        }
        // Table 24 INVALID_ADDRESS: a guest page that already carries a
        // mapping cannot receive a measured page.
        for i in 0..num_pages {
            if tvm
                .page_table
                .leaf_4k(gpa + i * PAGE_SIZE)
                .is_some_and(|pte| pte != 0)
            {
                return SbiRet::invalid_address();
            }
        }

        // Reserve the page tables for the whole range before anything
        // changes: a refused plan must leave the tracker, the guest pages
        // and the pool exactly as they were.
        let needed = match tvm.page_table.plan_range_4k(gpa, num_pages) {
            Ok(needed) => needed,
            Err(()) => return SbiRet::failed(),
        };
        if tvm.pt_page_count.saturating_sub(tvm.pt_page_used) < needed {
            return SbiRet::failed();
        }

        // Table 24 INVALID_ADDRESS: a destination that was never converted
        // is not confidential memory. A converted page that is already in
        // use is a state violation the tables name no code for, so it is the
        // project's invalid_param.
        let mut tracker = page_tracker();
        match tracker.status_of_range(dst_paddr, num_pages) {
            RangeStatus::Free => {}
            RangeStatus::NotConverted => return SbiRet::invalid_address(),
            RangeStatus::InUse => return SbiRet::invalid_param(),
        }

        tracker.assign_range(dst_paddr, num_pages, PageState::AssignedPayload, tvm_id);

        for i in 0..num_pages {
            let s_addr = src_paddr + i * PAGE_SIZE;
            let d_addr = dst_paddr + i * PAGE_SIZE;
            let cur_gpa = gpa + i * PAGE_SIZE;

            // Copy contents
            unsafe {
                core::ptr::copy_nonoverlapping(s_addr as *const u8, d_addr as *mut u8, PAGE_SIZE);
            }

            // The whole range was planned and the pool supplies it, so the
            // mapping cannot fail.
            let mut pool =
                page_table::PtPool::new(&tvm.pt_pages[..tvm.pt_page_count], &mut tvm.pt_page_used);
            tvm.page_table
                .map_4k(cur_gpa, d_addr, &mut pool)
                .expect("page tables planned before the first write");
        }

        SbiRet::success(0)
    }

    /// COVH FID 12 `add_zero_pages` (§10.14): maps fresh zero pages at a
    /// guest range after finalization; the guest faults them in on access.
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

        // §10.14: "This call may be made only after calling
        // sbi_covh_finalize_tvm()". Table 25 names no state error;
        // invalid_param is the project mapping, as for the memory-region and
        // measured-page handlers.
        if tvm.state != TvmState::Runnable {
            return SbiRet::invalid_param();
        }
        // Table 25 covers an invalid tsm_page_type with INVALID_PARAM. Only
        // 4 KiB pages are implemented: the larger enum values are legal but
        // unsupported and answer not_supported; anything else is not a legal
        // enum value.
        match page_type {
            0 => {}
            1..=3 => return SbiRet::not_supported(),
            _ => return SbiRet::invalid_param(),
        }
        // Table 25: an unusable address is INVALID_ADDRESS, an invalid page
        // count is INVALID_PARAM.
        if dst_paddr % PAGE_SIZE != 0 || gpa % PAGE_SIZE != 0 {
            return SbiRet::invalid_address();
        }
        if num_pages == 0 {
            return SbiRet::invalid_param();
        }
        let len = match num_pages.checked_mul(PAGE_SIZE) {
            Some(len) => len,
            None => return SbiRet::invalid_param(),
        };
        let gpa_end = match gpa.checked_add(len) {
            Some(end) if end <= (1usize << 41) => end,
            _ => return SbiRet::invalid_address(),
        };
        // Table 25 INVALID_ADDRESS: the guest range must be TVM memory and
        // must not be host-owned. A shared page is mapped by the host and an
        // MMIO page is emulated by it; neither can become a confidential
        // zero page.
        let in_memory = tvm
            .memory_regions
            .iter()
            .flatten()
            .any(|r| gpa >= r.gpa_base && gpa_end <= r.gpa_base + r.len);
        if !in_memory {
            return SbiRet::invalid_address();
        }
        let overlaps = |base: usize, region_len: usize| gpa < base + region_len && base < gpa_end;
        if tvm
            .shared_regions
            .iter()
            .flatten()
            .any(|r| overlaps(r.gpa_base, r.len))
        {
            return SbiRet::invalid_address();
        }
        if tvm
            .mmio_regions
            .iter()
            .flatten()
            .any(|r| overlaps(r.gpa_base, r.len))
        {
            return SbiRet::invalid_address();
        }

        // Table 25 INVALID_ADDRESS: a guest page that already carries a
        // mapping cannot receive a zero page.
        for i in 0..num_pages {
            if tvm
                .page_table
                .leaf_4k(gpa + i * PAGE_SIZE)
                .is_some_and(|pte| pte != 0)
            {
                return SbiRet::invalid_address();
            }
        }

        // Reserve the page tables for the whole range before anything
        // changes, as for measured pages.
        let needed = match tvm.page_table.plan_range_4k(gpa, num_pages) {
            Ok(needed) => needed,
            Err(()) => return SbiRet::failed(),
        };
        if tvm.pt_page_count.saturating_sub(tvm.pt_page_used) < needed {
            return SbiRet::failed();
        }

        // Table 25 INVALID_ADDRESS: a destination that was never converted
        // is not confidential memory (the row names base_page_address). A
        // converted page that is already in use is the project's
        // invalid_param.
        let mut tracker = page_tracker();
        match tracker.status_of_range(dst_paddr, num_pages) {
            RangeStatus::Free => {}
            RangeStatus::NotConverted => return SbiRet::invalid_address(),
            RangeStatus::InUse => return SbiRet::invalid_param(),
        }

        tracker.assign_range(dst_paddr, num_pages, PageState::AssignedPayload, tvm_id);

        for i in 0..num_pages {
            let d_addr = dst_paddr + i * PAGE_SIZE;
            let cur_gpa = gpa + i * PAGE_SIZE;

            unsafe {
                core::ptr::write_bytes(d_addr as *mut u8, 0, PAGE_SIZE);
            }

            // The whole range was planned and the pool supplies it, so the
            // mapping cannot fail.
            let mut pool =
                page_table::PtPool::new(&tvm.pt_pages[..tvm.pt_page_count], &mut tvm.pt_page_used);
            tvm.page_table
                .map_4k(cur_gpa, d_addr, &mut pool)
                .expect("page tables planned before the first write");
        }

        // A vCPU on another hart could still hold an older translation of
        // this range.
        // TODO(multi-hart): the fence is only sufficient because one hart
        // runs every operation today.
        unsafe {
            core::arch::asm!("hfence.gvma");
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
        // Table 34 lists INVALID_ADDRESS for "tvm_vcpu_imsic_gpa was
        // invalid": a page that already carries a G-stage mapping cannot
        // become the vCPU's device page.
        if tvm.page_table.leaf_4k(gpa).is_some_and(|pte| pte != 0) {
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
        // Table 37 names no TVM-state error for bind. Project mapping: a
        // binding is what makes a vCPU eligible to run, so it requires a
        // finalized TVM. Binding an Initializing TVM would store a VGEIN that
        // sbi_covh_finalize_tvm() then overwrites, leaving a recorded binding
        // without a usable guest interrupt file.
        if tvm.state != TvmState::Runnable {
            return SbiRet::invalid_param();
        }
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

        // Plan the device mapping before the binding changes any state: a
        // pool that cannot supply it must fail the call up front.
        let needed = match tvm.page_table.plan_4k(gpa) {
            Ok(needed) => needed,
            Err(()) => return SbiRet::failed(),
        };
        if tvm.pt_page_count.saturating_sub(tvm.pt_page_used) < needed {
            return SbiRet::failed();
        }

        // The file must have been converted (COVI FID 2) and fenced, and
        // not owned by any TVM. Lock order: TVM_MANAGER -> PAGE_TRACKER.
        let mut tracker = page_tracker();
        let bound = tracker.bind_imsic_page(file_paddr, tvm_id, vcpu_id, gpa);
        if bound.error != 0 {
            return SbiRet::invalid_param();
        }

        // Map the vIMSIC GPA onto the file in the G-stage: device leaf,
        // never executable. The PT pool belongs to this TVM and was sized
        // above, so the mapping cannot run out of pages; the release below
        // stays as defence in depth.
        let mut pool =
            page_table::PtPool::new(&tvm.pt_pages[..tvm.pt_page_count], &mut tvm.pt_page_used);
        if tvm.page_table.map_4k_nx(gpa, file_paddr, &mut pool).is_err() {
            let _ = tracker.release_imsic_file(file_paddr, tvm_id);
            return SbiRet::failed();
        }

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
            hart_id: hart,
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
            // Table 40 names no "AIA not initialized" error; an unarmed AIA
            // configuration is an invalid TVM state, so invalid_param is the
            // project mapping (FID 1 and FID 4 reject the same state with it).
            return SbiRet::invalid_param();
        }
        // TODO(multi-hart): a vCPU is detached from the manager while it runs
        // on another hart, so an inject in that window is refused by the
        // lookup below instead of reaching the bound file as §11.8 asks.
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

        let vcpu = match tvm.vcpu.as_ref() {
            Some(v) => v,
            None => return SbiRet::failed(),
        };

        // A vCPU carrying a binding cannot occur now that bind requires a
        // Runnable TVM, but finalize must never be the step that clears one.
        // Table 19 names no code for this state; invalid_param is the project
        // mapping.
        if vcpu.imsic_binding.is_some() {
            return SbiRet::invalid_param();
        }

        // §11.2: every vCPU of an AIA-enabled TVM must have its vIMSIC
        // GPA configured before finalize.
        if tvm.aia.enabled && vcpu.imsic_gpa.is_none() {
            return SbiRet::failed();
        }

        // Validate the whole identity argument before any state changes
        // (§10.8): zero means "no identity"; a nonzero address must be
        // 64-byte aligned. Table 19 names SBI_ERR_INVALID_PARAM for
        // "tvm_identity_addr was invalid", and the host-allocatable-range
        // check is the TSM's own hardening reported under the same code. An
        // invalid argument must leave the TVM exactly as it was.
        if identity_addr != 0 && (identity_addr % 64 != 0 || !is_host_range(identity_addr, 64)) {
            return SbiRet::invalid_param();
        }

        // No fallible step past this point: the transition to Runnable is
        // atomic with respect to what the caller can observe.
        let hgatp = tvm.page_table.make_hgatp(tvm.vmid);
        let vcpu = tvm.vcpu.as_mut().expect("vCPU presence checked above");
        vcpu.init_boot(entry_sepc, entry_arg, hgatp);

        tvm.entry_sepc = entry_sepc;
        tvm.entry_arg = entry_arg;

        if identity_addr != 0 {
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

    /// COVG share (§12.3): every populated page of the range is scrubbed and
    /// handed to the host, the leaves are cut out of the TVM, and the range
    /// is recorded as a shared region.
    pub fn share_memory_internal(&mut self, gpa: usize, len: usize) -> ShareOutcome {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) => t,
            None => return ShareOutcome::BadParam,
        };

        // Refusals leave the TVM and the tracker untouched. Table 46 splits
        // them into an unusable address and an invalid parameter or state.
        if gpa % PAGE_SIZE != 0 {
            return ShareOutcome::BadAddress;
        }
        if len == 0 || len % PAGE_SIZE != 0 {
            return ShareOutcome::BadParam;
        }
        let end = match gpa.checked_add(len) {
            Some(end) => end,
            None => return ShareOutcome::BadAddress,
        };
        // The Sv39x4 G-stage bounds the addresses this page table can hold.
        if end > (1usize << 41) {
            return ShareOutcome::BadAddress;
        }
        // Table 46 INVALID_PARAM: the range must map to a confidential
        // region (§12.3 "must lie within an existing region of confidential
        // address space").
        let in_memory = tvm
            .memory_regions
            .iter()
            .flatten()
            .any(|r| gpa >= r.gpa_base && end <= r.gpa_base + r.len);
        if !in_memory {
            return ShareOutcome::BadParam;
        }
        // A range that is already shared, or that the host emulates, is not
        // confidential memory.
        let overlaps = |base: usize, region_len: usize| gpa < base + region_len && base < end;
        if tvm
            .shared_regions
            .iter()
            .flatten()
            .any(|r| overlaps(r.gpa_base, r.len))
        {
            return ShareOutcome::BadParam;
        }
        if tvm
            .mmio_regions
            .iter()
            .flatten()
            .any(|r| overlaps(r.gpa_base, r.len))
        {
            return ShareOutcome::BadParam;
        }
        // Only a running TVM yields memory: §12.3 blocks the calling vCPU
        // until the host completes the conversion.
        if tvm.state != TvmState::Runnable {
            return ShareOutcome::BadParam;
        }
        let slot = match tvm.shared_regions.iter().position(|slot| slot.is_none()) {
            Some(slot) => slot,
            None => return ShareOutcome::Failed,
        };

        let mut tracker = page_tracker();
        match share::yield_range(
            &tvm.page_table,
            &mut tracker,
            &mut crate::rdsm_shim::RdsmMpt,
            tvm.id,
            gpa,
            len,
        ) {
            Ok(()) => {
                tvm.shared_regions[slot] = Some(SharedRegion { gpa_base: gpa, len });
                unsafe {
                    core::arch::asm!("hfence.gvma");
                }
                ShareOutcome::Shared
            }
            Err(YieldFailure::Refused) => {
                crate::println!(
                    "[TSM] COVG share refused: gpa=0x{:x}, len=0x{:x}",
                    gpa,
                    len
                );
                ShareOutcome::BadParam
            }
            Err(YieldFailure::RolledBack) => {
                crate::println!(
                    "[TSM] COVG share rolled back, nothing shared: gpa=0x{:x}, len=0x{:x}",
                    gpa,
                    len
                );
                ShareOutcome::Failed
            }
            Err(YieldFailure::Poisoned) => {
                crate::println!(
                    "[TSM] COVG share could not be rolled back, TVM poisoned: gpa=0x{:x}, len=0x{:x}",
                    gpa,
                    len
                );
                tvm.state = TvmState::Poisoned;
                unsafe {
                    core::arch::asm!("hfence.gvma");
                }
                ShareOutcome::Poisoned
            }
        }
    }

    /// COVG unshare (§12.4): unmaps every page the host had mapped into the
    /// range and shrinks the shared regions around it. Nothing is mapped
    /// back; a later guest access faults and is served by demand-zero.
    pub fn unshare_memory_internal(&mut self, gpa: usize, len: usize) -> UnshareOutcome {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) => t,
            None => return UnshareOutcome::BadParam,
        };

        // Table 47 splits its refusals into an unusable address and an
        // invalid length, range or state.
        if gpa % PAGE_SIZE != 0 {
            return UnshareOutcome::BadAddress;
        }
        if len == 0 || len % PAGE_SIZE != 0 {
            return UnshareOutcome::BadParam;
        }
        let end = match gpa.checked_add(len) {
            Some(end) => end,
            None => return UnshareOutcome::BadAddress,
        };
        if end > (1usize << 41) {
            return UnshareOutcome::BadAddress;
        }
        // Only a running TVM unshares memory: §12.3 blocks the calling vCPU
        // until the host completes the conversion.
        if tvm.state != TvmState::Runnable {
            return UnshareOutcome::BadParam;
        }
        // Table 47 INVALID_PARAM: the range must span one shared region
        // (§12.4 "must lie within an existing region of non-confidential
        // address space"), not several and not a part of one outside it. The
        // bookkeeping is planned before any leaf changes, so a missing free
        // slot fails the call before the range is unmapped.
        let (index, reshape) = match plan_unshare(&tvm.shared_regions, gpa, len) {
            Ok(plan) => plan,
            Err(UnsharePlanError::NotShared) => return UnshareOutcome::BadParam,
            Err(UnsharePlanError::NoFreeSlot) => return UnshareOutcome::Failed,
        };

        let tracker = page_tracker();
        if share::retake_range(&tvm.page_table, &tracker, gpa, len).is_err() {
            crate::println!(
                "[TSM] COVG unshare refused: gpa=0x{:x}, len=0x{:x}",
                gpa,
                len
            );
            return UnshareOutcome::BadParam;
        }
        drop(tracker);

        apply_unshare(&mut tvm.shared_regions, index, reshape);
        // TODO(multi-hart): invalidating the translation is only sufficient
        // because one hart runs every operation today; a vCPU of this TVM
        // running on another hart would need a shootdown before it resumes.
        unsafe {
            core::arch::asm!("hfence.gvma");
        }
        UnshareOutcome::Unshared
    }

    /// Serves a guest page fault with a fresh zero page. The faulting page
    /// must lie inside a memory region and outside every shared and MMIO
    /// region: a fault on host-owned memory belongs to the host.
    pub fn handle_demand_zero(&mut self, fault_gpa: usize) -> DemandZero {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) => t,
            None => return DemandZero::Refused,
        };

        // A fault reports the exact address; allocation and mapping work on
        // the page that contains it.
        let page_gpa = fault_gpa & !(PAGE_SIZE - 1);

        // Host-owned pages are classified first: a shared page belongs to
        // the host, and an emulated MMIO region usually lies outside the RAM
        // regions altogether. Serving either here would hand the guest a
        // fresh private page behind the host's back.
        if tvm
            .shared_regions
            .iter()
            .flatten()
            .any(|r| page_gpa >= r.gpa_base && page_gpa < r.gpa_base + r.len)
        {
            return DemandZero::HostPage;
        }
        if tvm
            .mmio_regions
            .iter()
            .flatten()
            .any(|r| page_gpa >= r.gpa_base && page_gpa < r.gpa_base + r.len)
        {
            return DemandZero::HostPage;
        }
        // Everything else must lie inside a reported memory region.
        let in_region = tvm
            .memory_regions
            .iter()
            .flatten()
            .any(|r| page_gpa >= r.gpa_base && page_gpa < r.gpa_base + r.len);
        if !in_region {
            return DemandZero::Refused;
        }

        // An occupied leaf is never a demand-zero candidate, and a pool that
        // cannot link the tables must leave the fault to the host.
        let needed = match tvm.page_table.plan_4k(page_gpa) {
            Ok(needed) => needed,
            Err(()) => return DemandZero::Refused,
        };
        if tvm.pt_page_count.saturating_sub(tvm.pt_page_used) < needed {
            return DemandZero::Refused;
        }

        // Find a free page from the tracker
        let mut tracker = page_tracker();
        let free_page = match tracker.find_free_page() {
            Some(p) => p,
            None => return DemandZero::Refused,
        };
        // Backstop: a tracker entry outside host-allocatable memory must not
        // reach the guest, even if the tracker is ever inconsistent.
        if !is_host_range(free_page, PAGE_SIZE) {
            return DemandZero::Refused;
        }

        // Assign it to this TVM
        if !tracker.assign_page(free_page, PageState::AssignedPayload, tvm.id) {
            return DemandZero::Refused;
        }

        // Zero-fill the page (demand-zero semantics)
        unsafe {
            core::ptr::write_bytes(free_page as *mut u8, 0, PAGE_SIZE);
        }

        let mut pool =
            page_table::PtPool::new(&tvm.pt_pages[..tvm.pt_page_count], &mut tvm.pt_page_used);
        if tvm.page_table.map_4k(page_gpa, free_page, &mut pool).is_err() {
            // Cannot happen after the plan above; undo the assignment so the
            // refused fault does not leak the page.
            tracker.unassign_page(free_page, tvm.id);
            return DemandZero::Refused;
        }
        DemandZero::Served
    }

    // ---- New COVH FIDs ----

    /// COVH FID 13 `add_tvm_shared_pages` (§10.15): maps host memory into a
    /// shared region the TVM defined with COVG share.
    pub fn add_shared_pages(
        &mut self,
        tvm_id: usize,
        base_page_address: usize,
        tsm_page_type: usize,
        num_pages: usize,
        tvm_base_page_address: usize,
    ) -> SbiRet {
        // Table 26 INVALID_PARAM: tvm_guest_id was invalid.
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };
        // §10.15: shared pages can be added only after the TVM begins
        // execution. Table 26 names no state error; invalid_param is the
        // project mapping, as for the measured and zero-page handlers.
        if tvm.state != TvmState::Runnable {
            return SbiRet::invalid_param();
        }
        // Table 26 INVALID_PARAM covers an invalid tsm_page_type. Only 4 KiB
        // pages are implemented: the larger enum values (PAGE_2MB, PAGE_1GB,
        // PAGE_512GB) are legal but unsupported and answer not_supported, the
        // way the measured and zero-page handlers do; anything else is not a
        // legal enum value.
        match tsm_page_type {
            0 => {}
            1..=3 => return SbiRet::not_supported(),
            _ => return SbiRet::invalid_param(),
        }
        // Table 26 INVALID_PARAM: num_pages was invalid.
        if num_pages == 0 {
            return SbiRet::invalid_param();
        }
        let len = match num_pages.checked_mul(PAGE_SIZE) {
            Some(len) => len,
            None => return SbiRet::invalid_param(),
        };
        // Table 26 INVALID_ADDRESS: either address was invalid. Both are
        // page-granular.
        if base_page_address % PAGE_SIZE != 0 || tvm_base_page_address % PAGE_SIZE != 0 {
            return SbiRet::invalid_address();
        }
        // The GPA range must be representable in the Sv39x4 G-stage.
        let gpa_end = match tvm_base_page_address.checked_add(len) {
            Some(end) if end <= (1usize << 41) => end,
            _ => return SbiRet::invalid_address(),
        };
        // §10.15: the GPA range must lie within a region of non-confidential
        // memory the TVM defined via the guest interface.
        let in_shared = tvm
            .shared_regions
            .iter()
            .flatten()
            .any(|r| tvm_base_page_address >= r.gpa_base && gpa_end <= r.gpa_base + r.len);
        if !in_shared {
            return SbiRet::invalid_address();
        }
        // The pages handed in must be host-allocatable memory: mapping the
        // TSM image or the MPT pool would expose confidential memory.
        if !is_host_range(base_page_address, len) {
            return SbiRet::invalid_address();
        }
        // A confidential page is not a valid base_page_address: no tracked
        // page may overlap the range.
        let tracker = page_tracker();
        if !tracker.range_is_untracked(base_page_address, num_pages) {
            return SbiRet::invalid_address();
        }
        // The pages handed in must not already be mapped into this TVM: one
        // host page is never mapped at two guest addresses.
        if tvm.page_table.maps_any(base_page_address, len) {
            return SbiRet::invalid_address();
        }
        // Only empty leaves can receive a shared mapping. Table 26's
        // INVALID_PARAM row names only tvm_guest_id, tsm_page_type and
        // num_pages, so an occupied address is an invalid
        // tvm_base_page_address.
        for i in 0..num_pages {
            let gpa = tvm_base_page_address + i * PAGE_SIZE;
            if tvm.page_table.leaf_4k(gpa).is_some_and(|pte| pte != 0) {
                return SbiRet::invalid_address();
            }
        }
        // Reserve the page tables for the whole range before the first
        // write; a short pool must leave the table untouched.
        let needed = match tvm.page_table.plan_range_4k(tvm_base_page_address, num_pages) {
            Ok(needed) => needed,
            Err(()) => return SbiRet::failed(),
        };
        if tvm.pt_page_count.saturating_sub(tvm.pt_page_used) < needed {
            return SbiRet::failed();
        }

        for i in 0..num_pages {
            let gpa = tvm_base_page_address + i * PAGE_SIZE;
            let spa = base_page_address + i * PAGE_SIZE;
            // The whole range was planned and the pool supplies it, so the
            // mapping cannot fail.
            let mut pool =
                page_table::PtPool::new(&tvm.pt_pages[..tvm.pt_page_count], &mut tvm.pt_page_used);
            // Shared memory is untrusted: the leaf is never executable
            // (CoVE §5.2.3).
            tvm.page_table
                .map_4k_nx(gpa, spa, &mut pool)
                .expect("page tables planned before the first write");
        }
        unsafe {
            core::arch::asm!("hfence.gvma");
        }
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

    /// COVH FID 17 `tvm_invalidate_pages` (§10.19): blocks present shared
    /// mappings in the range so the TVM can no longer access them.
    pub fn tvm_invalidate_pages(&mut self, tvm_id: usize, gpa: usize, length: usize) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) => t,
            None => return SbiRet::invalid_param(),
        };
        if let Err(ret) = check_trio_range(tvm, tvm_id, gpa, length) {
            return ret;
        }
        let num_pages = length / PAGE_SIZE;
        let tracker = page_tracker();
        // Whole-range preflight: no leaf changes before every page passes.
        for i in 0..num_pages {
            match classify_shared_page(tvm, &tracker, gpa + i * PAGE_SIZE) {
                Err(ret) => return ret,
                Ok(SharedPage::Present) => {}
                // Table 30: "gpa was invalid" when the page is not present.
                Ok(_) => return SbiRet::invalid_address(),
            }
        }
        for i in 0..num_pages {
            tvm.page_table
                .invalidate_4k(gpa + i * PAGE_SIZE)
                .expect("page class checked before the first write");
        }
        unsafe {
            core::arch::asm!("hfence.gvma");
        }
        SbiRet::success(0)
    }

    /// COVH FID 18 `tvm_validate_pages` (§10.20): restores mappings the host
    /// invalidated, including those it invalidated only to remove them.
    pub fn tvm_validate_pages(&mut self, tvm_id: usize, gpa: usize, length: usize) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) => t,
            None => return SbiRet::invalid_param(),
        };
        if let Err(ret) = check_trio_range(tvm, tvm_id, gpa, length) {
            return ret;
        }
        let num_pages = length / PAGE_SIZE;
        let tracker = page_tracker();
        for i in 0..num_pages {
            match classify_shared_page(tvm, &tracker, gpa + i * PAGE_SIZE) {
                Err(ret) => return ret,
                Ok(SharedPage::Retained) => {}
                // Table 31: "gpa was invalid" unless §10.19 invalidated it.
                Ok(_) => return SbiRet::invalid_address(),
            }
        }
        for i in 0..num_pages {
            tvm.page_table
                .validate_4k(gpa + i * PAGE_SIZE)
                .expect("page class checked before the first write");
        }
        unsafe {
            core::arch::asm!("hfence.gvma");
        }
        SbiRet::success(0)
    }

    /// COVH FID 19 `tvm_remove_pages` (§10.21): clears mappings the host
    /// invalidated and fenced. The pages stay host memory, so the tracker is
    /// not involved.
    pub fn tvm_remove_pages(&mut self, tvm_id: usize, gpa: usize, length: usize) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) => t,
            None => return SbiRet::invalid_param(),
        };
        if let Err(ret) = check_trio_range(tvm, tvm_id, gpa, length) {
            return ret;
        }
        let num_pages = length / PAGE_SIZE;
        let tracker = page_tracker();
        for i in 0..num_pages {
            match classify_shared_page(tvm, &tracker, gpa + i * PAGE_SIZE) {
                Err(ret) => return ret,
                Ok(SharedPage::Retained) => {}
                // Table 32: "gpa was invalid" unless it was invalidated and
                // fenced before the call.
                Ok(_) => return SbiRet::invalid_address(),
            }
        }
        for i in 0..num_pages {
            let _ = tvm
                .page_table
                .remove_invalid_4k(gpa + i * PAGE_SIZE)
                .expect("page class checked before the first write");
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

        // §10.10: destroy fails while the VMM still executes a virtual hart.
        // A running vCPU is detached from the manager, so a Runnable TVM with
        // no attached vCPU has one in flight on some hart. The arguments are
        // valid but the operation cannot complete, so this is FAILED: Table
        // 21 reserves INVALID_PARAM for an invalid tvm_guest_id.
        if tvm.state == TvmState::Runnable && tvm.vcpu.is_none() {
            return SbiRet::failed();
        }

        // A binding belongs to the hart the bind ran on (§11.5) and only that
        // hart's VS indirect window can reach its interrupt file. The host
        // may retry on the binding hart.
        // TODO(multi-hart): tearing a binding down from another hart needs a
        // per-hart work protocol; the local VS indirect window cannot scrub
        // another hart's interrupt file.
        if let Some(binding) = tvm.vcpu.as_ref().and_then(|v| v.imsic_binding) {
            if binding.hart_id != crate::current_hart_id() {
                return SbiRet::failed();
            }
        }

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

/// Common argument checks of the invalidate/validate/remove trio (§10.19 to
/// §10.21, Tables 30-32): a bad id or length is INVALID_PARAM, a bad `gpa` is
/// INVALID_ADDRESS.
fn check_trio_range(tvm: &Tvm, tvm_id: usize, gpa: usize, length: usize) -> Result<(), SbiRet> {
    if tvm.id != tvm_id {
        return Err(SbiRet::invalid_param());
    }
    if length == 0 || length % PAGE_SIZE != 0 {
        return Err(SbiRet::invalid_param());
    }
    if gpa % PAGE_SIZE != 0 {
        return Err(SbiRet::invalid_address());
    }
    match gpa.checked_add(length) {
        Some(end) if end <= (1usize << 41) => Ok(()),
        _ => Err(SbiRet::invalid_address()),
    }
}

/// Per-page preflight of the invalidate/validate/remove trio.
///
/// The trio is the host's tool for removing shared pages, so a page outside
/// every shared region is INVALID_PARAM (project policy). This TSM tears
/// confidential pages down only by destroying the TVM, which is narrower than
/// §10.19's "uniquely owned by the TVM" branch. A present leaf that maps a
/// tracked page is refused by the same code as a defence: it is a
/// confidential mapping, not a shared page.
fn classify_shared_page(
    tvm: &Tvm,
    tracker: &PageTracker,
    gpa: usize,
) -> Result<SharedPage, SbiRet> {
    let in_shared = tvm
        .shared_regions
        .iter()
        .flatten()
        .any(|r| gpa >= r.gpa_base && gpa < r.gpa_base + r.len);
    if !in_shared {
        return Err(SbiRet::invalid_param());
    }
    match tvm.page_table.leaf_4k(gpa) {
        None | Some(0) => Ok(SharedPage::Absent),
        Some(pte) if leaf_is_present(pte) => {
            if tracker.lookup(leaf_pa(pte)).is_some() {
                return Err(SbiRet::invalid_param());
            }
            Ok(SharedPage::Present)
        }
        Some(pte) => {
            // A retained entry is still a mapping the host may re-validate,
            // so its page must stay host memory as well.
            if tracker.lookup(leaf_pa(pte)).is_some() {
                return Err(SbiRet::invalid_param());
            }
            Ok(SharedPage::Retained)
        }
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
        // Table 28 INVALID_PARAM: the TVM "wasn't in TVM_RUNNABLE state". A
        // poisoned TVM never runs again (§12.3 blocks the vCPU whose share
        // could not be completed), so it is answered here like any other
        // non-runnable TVM.
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
        // §7.4.2: a vCPU is eligible to run only on the physical hart it is
        // bound to; a run on a different hart must fail. Table 28 names no
        // code for a wrong-hart run; the ids and state are valid, so FAILED is
        // the project mapping.
        if let Some(binding) = tvm.vcpu.as_ref().and_then(|v| v.imsic_binding) {
            if binding.hart_id != crate::current_hart_id() {
                return SbiRet::failed();
            }
        }
        tvm.vcpu.take().expect("vcpu presence checked above")
    };

    let res = vcpu.run();
    let exit_reason = vcpu.ctx.last_exit.reason;
    let exit_page = vcpu.ctx.last_exit.gpa & !(PAGE_SIZE - 1);

    let mut mgr = tvm_manager();
    let tvm = match mgr.active_tvm.as_mut() {
        // The TVM must still be the one the vCPU was detached from; a
        // concurrent destroy while detached is a torn state, reported as
        // a failed run.
        Some(t) if t.id == tvm_id => t,
        _ => return SbiRet::failed(),
    };
    tvm.vcpu = Some(vcpu);
    // A run that poisoned the TVM (a COVG share it could not roll back) is
    // reported as a failed run and the TVM never runs again.
    if tvm.state == TvmState::Poisoned {
        return SbiRet::failed();
    }
    if res == 0 {
        // The reason occupies the low seven bits; a host-owned page fault
        // carries the faulting guest page address above them so the host
        // knows which page to service.
        let value = if exit_reason == vcpu::EXIT_GUEST_PAGE_FAULT {
            (exit_page << 7) | exit_reason
        } else {
            exit_reason
        };
        SbiRet::success(value)
    } else {
        SbiRet::failed()
    }
}
