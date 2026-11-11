use crate::SbiRet;
use crate::rdsm_shim::{rdsm_mfence_pa, rdsm_mpt_set};
use spin::{Mutex, MutexGuard, Once};

pub const PAGE_SIZE: usize = 4096;
pub const MAX_PAGES: usize = 512;

// ── Host-allocatable memory whitelist ───────────────────────
//
// Derived once at boot from RDSM_GET_INFO. Host-supplied physical
// addresses are only accepted when the whole range falls inside one of
// these regions; everything else (firmware gap, TSM image, MPT page
// pool) is reserved and must be rejected.

static HOST_REGIONS: Once<[(usize, usize); 2]> = Once::new();

/// IMSIC supervisor/guest interrupt-file MMIO whitelist `(start, end)`,
/// derived from the RDSM-provided platform geometry. Zero-sized when the
/// platform has no supervisor IMSIC.
static IMSIC_REGION: Once<(usize, usize)> = Once::new();

/// Full supervisor/guest interrupt-file geometry (SmmTT v0.49 section
/// 6.1): bind-time address calculus derives the per-(domain, hart, guest)
/// file pages from it. Zeroed when the platform has no supervisor IMSIC.
static IMSIC_GEOMETRY: Once<crate::rdsm_shim::RdsmImsicGeometry> = Once::new();

/// Returns the supervisor/guest interrupt-file geometry, or `None` on an
/// IMSIC-less platform (fail closed).
pub fn imsic_geometry() -> Option<crate::rdsm_shim::RdsmImsicGeometry> {
    IMSIC_GEOMETRY.get().filter(|g| g.usable()).copied()
}

/// Populate the whitelists from the RDSM-provided platform layout.
pub fn init_host_regions(info: &crate::rdsm_shim::RdsmPlatformInfo) {
    HOST_REGIONS.call_once(|| {
        [
            (info.tsm_region_end, info.mpt_pool_start),
            (info.mpt_pool_end, info.ram_end),
        ]
    });
    IMSIC_REGION.call_once(|| (info.imsic.range_start, info.imsic.range_end));
    IMSIC_GEOMETRY.call_once(|| info.imsic);
}

/// Fail-closed check: is `[paddr, paddr+len)` entirely host-allocatable?
/// Containment only; page alignment is the caller's concern (each COVH
/// entry validates its own alignment requirements).
pub fn is_host_range(paddr: usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }
    let end = match paddr.checked_add(len) {
        Some(e) => e,
        None => return false,
    };
    HOST_REGIONS.get().map_or(false, |regions| {
        regions
            .iter()
            .any(|&(start, stop)| paddr >= start && end <= stop)
    })
}

/// Fail-closed check: is `[paddr, paddr+len)` entirely inside the IMSIC
/// interrupt-file MMIO region? Always false on IMSIC-less platforms.
pub fn is_imsic_range(paddr: usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }
    let end = match paddr.checked_add(len) {
        Some(e) => e,
        None => return false,
    };
    IMSIC_REGION
        .get()
        .map_or(false, |&(start, stop)| stop > start && paddr >= start && end <= stop)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageState {
    NonConfidential,
    ConvertedClean,
    AssignedTvmRoot,
    AssignedTvmState,
    AssignedVcpuState,
    AssignedPageTable,
    AssignedPayload,
    Shared,
    /// IMSIC guest/manager interrupt file page owned by the TSM (COVI).
    /// Device MMIO: contents are never read or written by software here.
    ImsicFile,
}

#[derive(Clone, Copy, Debug)]
pub struct PageEntry {
    pub paddr: usize,
    pub state: PageState,
    pub owner_tvm_id: Option<usize>,
    pub gpa: Option<usize>,
}

pub struct PageTracker {
    entries: [Option<PageEntry>; MAX_PAGES],
}

impl PageTracker {
    pub const fn new() -> Self {
        Self {
            entries: [None; MAX_PAGES],
        }
    }

    pub fn convert_pages(&mut self, base_paddr: usize, num_pages: usize) -> SbiRet {
        if base_paddr % PAGE_SIZE != 0 || num_pages == 0 {
            return SbiRet::invalid_param();
        }

        let len = match num_pages.checked_mul(PAGE_SIZE) {
            Some(l) => l,
            None => return SbiRet::invalid_param(),
        };

        // Only host-allocatable memory may be converted:
        // firmware, TSM image and MPT page pool must stay out of reach.
        if !is_host_range(base_paddr, len) {
            return SbiRet::invalid_param();
        }

        // Check if there is enough space in entries and no duplicates
        let mut free_count = 0;
        for slot in self.entries.iter() {
            if slot.is_none() {
                free_count += 1;
            } else if let Some(e) = slot {
                let e_end = e.paddr + PAGE_SIZE;
                let target_end = match base_paddr.checked_add(len) {
                    Some(t) => t,
                    None => return SbiRet::invalid_param(),
                };
                if !(target_end <= e.paddr || base_paddr >= e_end) {
                    return SbiRet::invalid_param(); // Already converted
                }
            }
        }
        if free_count < num_pages {
            return SbiRet::failed();
        }

        // 1. Strip Host (SDID=0) access in MPT
        let (err0, _) = rdsm_mpt_set(0, base_paddr, len, 0); // NONE
        if err0 != 0 {
            return SbiRet::failed();
        }

        // 2. Grant Confidential (SDID=1) access in MPT
        let (err1, _) = rdsm_mpt_set(1, base_paddr, len, 7); // RWX
        if err1 != 0 {
            return SbiRet::failed();
        }

        // 3. Add each page
        for i in 0..num_pages {
            let paddr = base_paddr + i * PAGE_SIZE;
            for slot in self.entries.iter_mut() {
                if slot.is_none() {
                    *slot = Some(PageEntry {
                        paddr,
                        state: PageState::ConvertedClean,
                        owner_tvm_id: None,
                        gpa: None,
                    });
                    break;
                }
            }
        }

        SbiRet::success(0)
    }

    pub fn reclaim_pages(&mut self, base_paddr: usize, num_pages: usize) -> SbiRet {
        if base_paddr % PAGE_SIZE != 0 || num_pages == 0 {
            return SbiRet::invalid_param();
        }

        // Verify all pages exist, are ConvertedClean, and have no owner
        for i in 0..num_pages {
            let paddr = base_paddr + i * PAGE_SIZE;
            let mut found = false;
            for slot in self.entries.iter().flatten() {
                if slot.paddr == paddr {
                    if slot.owner_tvm_id.is_some() || slot.state != PageState::ConvertedClean {
                        return SbiRet::invalid_param();
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                return SbiRet::invalid_param();
            }
        }

        let len = match num_pages.checked_mul(PAGE_SIZE) {
            Some(l) => l,
            None => return SbiRet::invalid_param(),
        };

        // 0. Scrub confidential page contents before restoring Host access (CoVE 5.2.4)
        for i in 0..num_pages {
            let paddr = base_paddr + i * PAGE_SIZE;
            unsafe {
                core::ptr::write_bytes(paddr as *mut u8, 0, PAGE_SIZE);
            }
        }

        // 1. Restore Host access (SDID=0)
        let (err0, _) = rdsm_mpt_set(0, base_paddr, len, 7); // RWX
        if err0 != 0 {
            return SbiRet::failed();
        }

        // 2. Fence
        let _ = rdsm_mfence_pa(0, 0);

        // 3. Remove entries
        for i in 0..num_pages {
            let paddr = base_paddr + i * PAGE_SIZE;
            for slot in self.entries.iter_mut() {
                if let Some(e) = slot {
                    if e.paddr == paddr {
                        *slot = None;
                        break;
                    }
                }
            }
        }

        SbiRet::success(0)
    }

    /// COVI convert_aia_imsic: turns a host-owned IMSIC interrupt file page
    /// into a TSM-managed confidential page (CoVE 11.3). The host must run
    /// the COVH global + local fence sequence afterwards before the page may
    /// be assigned to a TVM.
    ///
    /// Address violations (misalignment, outside the IMSIC whitelist) are
    /// `invalid_address`; tracker-state violations (already converted) are
    /// `invalid_param`. Unlike RAM conversion there is no scrub step — the
    /// page is device MMIO and writes to it would inject spurious
    /// interrupts.
    pub fn convert_imsic_page(&mut self, paddr: usize) -> SbiRet {
        if paddr % PAGE_SIZE != 0 || !is_imsic_range(paddr, PAGE_SIZE) {
            return SbiRet::invalid_address();
        }
        // Only guest interrupt files are COVI-convertible (CoVE §11.3):
        // decompose against the platform geometry and reject supervisor
        // files (guest index 0) and foreign domain/hart positions. On an
        // IMSIC-less platform the whitelist is empty above (fail closed).
        if let Some(g) = imsic_geometry() {
            let guests = 1usize << g.guest_index_bits;
            let harts = 1usize << g.hart_index_bits;
            let domains = (g.range_end - g.range_start) >> g.domain_index_shift;
            let off = paddr - g.range_start;
            let guest_idx = (off >> 12) & (guests - 1);
            let hart_idx = (off >> (12 + g.guest_index_bits)) & (harts - 1);
            let domain_idx = off >> g.domain_index_shift;
            if guests <= 1 || guest_idx == 0 || hart_idx >= harts || domain_idx >= domains {
                return SbiRet::invalid_address();
            }
        } else {
            return SbiRet::invalid_address();
        }
        for slot in self.entries.iter().flatten() {
            if slot.paddr == paddr {
                return SbiRet::invalid_param();
            }
        }

        // 1. Strip Host (SDID=0) access in MPT.
        let (err0, _) = rdsm_mpt_set(0, paddr, PAGE_SIZE, 0); // NONE
        if err0 != 0 {
            return SbiRet::failed();
        }

        // 2. Grant Confidential (SDID=1) read-write access in MPT.
        let (err1, _) = rdsm_mpt_set(1, paddr, PAGE_SIZE, 0b011); // RW
        if err1 != 0 {
            return SbiRet::failed();
        }

        // 3. Track the page as a TSM-owned interrupt file.
        for slot in self.entries.iter_mut() {
            if slot.is_none() {
                *slot = Some(PageEntry {
                    paddr,
                    state: PageState::ImsicFile,
                    owner_tvm_id: None,
                    gpa: None,
                });
                return SbiRet::success(0);
            }
        }
        SbiRet::failed()
    }

    /// COVI reclaim_tvm_aia_imsic: returns a TSM-owned interrupt file page
    /// to the host (CoVE 11.4). The page must not be assigned to a TVM.
    ///
    /// No scrub step (device MMIO); the MPT update restores host access and
    /// withdraws confidential access, followed by a global MPT fence.
    pub fn reclaim_imsic_page(&mut self, paddr: usize) -> SbiRet {
        if paddr % PAGE_SIZE != 0 {
            return SbiRet::invalid_address();
        }
        let mut found = false;
        for slot in self.entries.iter().flatten() {
            if slot.paddr == paddr {
                if slot.owner_tvm_id.is_some() || slot.state != PageState::ImsicFile {
                    return SbiRet::invalid_param();
                }
                found = true;
                break;
            }
        }
        if !found {
            return SbiRet::invalid_param();
        }

        // 1. Restore Host (SDID=0) access.
        let (err0, _) = rdsm_mpt_set(0, paddr, PAGE_SIZE, 0b111); // RWX
        if err0 != 0 {
            return SbiRet::failed();
        }

        // 2. Withdraw Confidential (SDID=1) access.
        let (err1, _) = rdsm_mpt_set(1, paddr, PAGE_SIZE, 0); // NONE
        if err1 != 0 {
            return SbiRet::failed();
        }

        // 3. Fence so the restored host access is visible.
        let _ = rdsm_mfence_pa(0, 0);

        // 4. Remove the tracker entry.
        for slot in self.entries.iter_mut() {
            if let Some(e) = slot {
                if e.paddr == paddr {
                    *slot = None;
                    return SbiRet::success(0);
                }
            }
        }
        SbiRet::failed()
    }

    /// COVI bind (§11.5): assigns a converted, unowned interrupt-file page
    /// to `(tvm, vcpu)` at `gpa`. The caller has already validated the
    /// address against the platform geometry; the tracker only enforces
    /// page state, so a bound vIMSIC page is always a converted guest
    /// interrupt file.
    pub fn bind_imsic_page(
        &mut self,
        paddr: usize,
        tvm_id: usize,
        vcpu_id: usize,
        gpa: usize,
    ) -> SbiRet {
        for slot in self.entries.iter_mut().flatten() {
            if slot.paddr == paddr {
                if slot.state != PageState::ImsicFile || slot.owner_tvm_id.is_some() {
                    return SbiRet::invalid_param();
                }
                slot.owner_tvm_id = Some(tvm_id);
                slot.gpa = Some(gpa);
                let _ = vcpu_id; // single-vCPU WP2: the owner tuple is TVM-scoped
                return SbiRet::success(0);
            }
        }
        SbiRet::invalid_param()
    }

    /// Whether `paddr` is an interrupt-file page currently unassigned.
    pub fn is_free_imsic_page(&self, paddr: usize) -> bool {
        self.entries.iter().flatten().any(|slot| {
            slot.paddr == paddr
                && slot.state == PageState::ImsicFile
                && slot.owner_tvm_id.is_none()
        })
    }

    /// Releases an interrupt-file page owned by `tvm_id` back to the
    /// unassigned pool. Device MMIO: no scrub, and the state stays
    /// [`PageState::ImsicFile`] (unlike RAM pages, which return to
    /// [`PageState::ConvertedClean`]).
    pub fn release_imsic_file(&mut self, paddr: usize, tvm_id: usize) -> bool {
        for slot in self.entries.iter_mut().flatten() {
            if slot.paddr == paddr && slot.owner_tvm_id == Some(tvm_id) {
                slot.owner_tvm_id = None;
                slot.gpa = None;
                return true;
            }
        }
        false
    }

    pub fn is_converted_and_free(&self, base_paddr: usize, num_pages: usize) -> bool {
        for i in 0..num_pages {
            let paddr = base_paddr + i * PAGE_SIZE;
            let mut found = false;
            for slot in self.entries.iter().flatten() {
                if slot.paddr == paddr {
                    if slot.owner_tvm_id.is_none() && slot.state == PageState::ConvertedClean {
                        found = true;
                        break;
                    }
                }
            }
            if !found {
                return false;
            }
        }
        true
    }

    pub fn assign_range(
        &mut self,
        base_paddr: usize,
        num_pages: usize,
        state: PageState,
        tvm_id: usize,
    ) -> bool {
        if !self.is_converted_and_free(base_paddr, num_pages) {
            return false;
        }

        for i in 0..num_pages {
            let paddr = base_paddr + i * PAGE_SIZE;
            for slot in self.entries.iter_mut().flatten() {
                if slot.paddr == paddr {
                    slot.state = state;
                    slot.owner_tvm_id = Some(tvm_id);
                    break;
                }
            }
        }
        true
    }

    pub fn find_free_page(&self) -> Option<usize> {
        for slot in self.entries.iter().flatten() {
            if slot.owner_tvm_id.is_none() && slot.state == PageState::ConvertedClean {
                return Some(slot.paddr);
            }
        }
        None
    }

    pub fn assign_page(&mut self, paddr: usize, state: PageState, tvm_id: usize) -> bool {
        for slot in self.entries.iter_mut().flatten() {
            if slot.paddr == paddr
                && slot.owner_tvm_id.is_none()
                && slot.state == PageState::ConvertedClean
            {
                slot.state = state;
                slot.owner_tvm_id = Some(tvm_id);
                return true;
            }
        }
        false
    }

    pub fn set_shared(&mut self, paddr: usize, gpa: usize, tvm_id: usize) -> bool {
        for slot in self.entries.iter_mut().flatten() {
            if slot.paddr == paddr && slot.owner_tvm_id == Some(tvm_id) {
                slot.state = PageState::Shared;
                slot.gpa = Some(gpa);
                return true;
            }
        }
        false
    }

    pub fn find_shared_by_gpa(&self, gpa: usize, tvm_id: usize) -> Option<usize> {
        self.entries
            .iter()
            .flatten()
            .find(|slot| {
                slot.owner_tvm_id == Some(tvm_id)
                    && slot.state == PageState::Shared
                    && slot.gpa == Some(gpa)
            })
            .map(|slot| slot.paddr)
    }

    pub fn clear_shared(&mut self, paddr: usize, tvm_id: usize) -> bool {
        for slot in self.entries.iter_mut().flatten() {
            if slot.paddr == paddr && slot.owner_tvm_id == Some(tvm_id) {
                if slot.state != PageState::Shared {
                    return false;
                }
                slot.state = PageState::AssignedPayload;
                slot.gpa = None;
                return true;
            }
        }
        false
    }

    pub fn release_page(&mut self, paddr: usize, tvm_id: usize) {
        for slot in self.entries.iter_mut().flatten() {
            if slot.paddr == paddr && slot.owner_tvm_id == Some(tvm_id) {
                slot.owner_tvm_id = None;
                slot.state = PageState::ConvertedClean;
                return;
            }
        }
    }

    pub fn release_tvm_pages(&mut self, tvm_id: usize) {
        for slot in self.entries.iter_mut().flatten() {
            if slot.owner_tvm_id == Some(tvm_id) {
                slot.owner_tvm_id = None;
                // Interrupt-file pages stay in their device state; the host
                // reclaim path (COVI FID 3) is the only route back to RAM.
                if slot.state != PageState::ImsicFile {
                    slot.state = PageState::ConvertedClean;
                } else {
                    slot.gpa = None;
                }
            }
        }
    }
}

/// Shared page-state machine.
///
/// Page convert/reclaim/assign are short configuration operations, so the
/// tracker is guarded by a single mutex. Callers may hold the TVM manager
/// lock while locking this one (lock order: `TVM_MANAGER` → `PAGE_TRACKER`);
/// the reverse order never occurs because the tracker never calls into the
/// manager.
static PAGE_TRACKER: Mutex<PageTracker> = Mutex::new(PageTracker::new());

/// Locks and returns the shared page tracker.
pub fn page_tracker() -> MutexGuard<'static, PageTracker> {
    PAGE_TRACKER.lock()
}
