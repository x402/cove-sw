use crate::sbi::SbiRet;
use rdsm_abi::{RdsmImsicGeometry, RdsmPlatformInfo};
use spin::{Mutex, MutexGuard, Once};

pub const PAGE_SIZE: usize = 4096;
pub const MAX_PAGES: usize = 512;

// ── Host-allocatable memory whitelist ───────────────────────
//
// Derived once at boot from RDSM_GET_INFO. Host-supplied physical
// addresses are only accepted when the whole range falls inside one of
// these regions; everything else (firmware gap, TSM image, MPT page
// pool) is reserved and must be rejected.

/// Memory map the TSM validates host-supplied addresses against. Fixed at
/// boot from the platform layout RDSM reports.
#[derive(Clone, Copy, Debug)]
pub struct PlatformLayout {
    host_regions: [(usize, usize); 2],
    imsic: RdsmImsicGeometry,
}

impl PlatformLayout {
    /// Derives the layout from the RDSM-reported platform info: the two
    /// host-allocatable regions are `[tsm_region_end, mpt_pool_start)` and
    /// `[mpt_pool_end, ram_end)`; everything else is reserved.
    pub fn new(info: &RdsmPlatformInfo) -> Self {
        Self {
            host_regions: [
                (info.tsm_region_end, info.mpt_pool_start),
                (info.mpt_pool_end, info.ram_end),
            ],
            imsic: info.imsic,
        }
    }

    /// Builds a layout from an explicit region pair and IMSIC geometry.
    pub fn from_parts(host_regions: [(usize, usize); 2], imsic: RdsmImsicGeometry) -> Self {
        Self {
            host_regions,
            imsic,
        }
    }

    /// Fail-closed check: is `[paddr, paddr+len)` entirely host-allocatable?
    /// Containment only; page alignment is the caller's concern (each COVH
    /// entry validates its own alignment requirements).
    pub fn is_host_range(&self, paddr: usize, len: usize) -> bool {
        if len == 0 {
            return false;
        }
        let end = match paddr.checked_add(len) {
            Some(e) => e,
            None => return false,
        };
        self.host_regions
            .iter()
            .any(|&(start, stop)| paddr >= start && end <= stop)
    }

    /// Fail-closed check: is `[paddr, paddr+len)` entirely inside the IMSIC
    /// interrupt-file MMIO region? Always false on IMSIC-less platforms.
    pub fn is_imsic_range(&self, paddr: usize, len: usize) -> bool {
        if len == 0 {
            return false;
        }
        let end = match paddr.checked_add(len) {
            Some(e) => e,
            None => return false,
        };
        let (start, stop) = (self.imsic.range_start, self.imsic.range_end);
        stop > start && paddr >= start && end <= stop
    }

    /// Returns the supervisor/guest interrupt-file geometry, or `None` on an
    /// IMSIC-less platform (fail closed).
    pub fn imsic_geometry(&self) -> Option<RdsmImsicGeometry> {
        Some(self.imsic).filter(|g| g.usable())
    }
}

/// Layout installed at boot, before the first dispatch.
static PLATFORM: Once<PlatformLayout> = Once::new();

/// Populates the whitelists from the RDSM-provided platform layout.
pub fn init_host_regions(info: &RdsmPlatformInfo) {
    PLATFORM.call_once(|| PlatformLayout::new(info));
}

/// The layout installed at boot, or `None` before boot initialisation.
pub fn platform() -> Option<&'static PlatformLayout> {
    PLATFORM.get()
}

/// Fail-closed check: is `[paddr, paddr+len)` entirely host-allocatable?
/// Always false before a layout is installed.
pub fn is_host_range(paddr: usize, len: usize) -> bool {
    platform().map_or(false, |layout| layout.is_host_range(paddr, len))
}

/// Fail-closed check: is `[paddr, paddr+len)` entirely inside the IMSIC
/// interrupt-file MMIO region? Always false on IMSIC-less platforms.
pub fn is_imsic_range(paddr: usize, len: usize) -> bool {
    platform().map_or(false, |layout| layout.is_imsic_range(paddr, len))
}

/// Returns the supervisor/guest interrupt-file geometry, or `None` on an
/// IMSIC-less platform (fail closed).
pub fn imsic_geometry() -> Option<RdsmImsicGeometry> {
    platform().and_then(|layout| layout.imsic_geometry())
}

/// MPT operations the page tracker needs from the RDSM firmware. The
/// backend is passed per call so the tracker's MPT call sequence can be
/// recorded on the host.
pub trait MptBackend {
    /// `RDSM_MPT_SET`; `Err` when the firmware returned a nonzero error.
    fn mpt_set(&mut self, target_sdid: usize, paddr: usize, len: usize, perm: u8) -> Result<(), ()>;
    fn mfence_pa(&mut self, paddr: usize, sdid: usize);
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

    pub fn convert_pages(
        &mut self,
        platform: &PlatformLayout,
        mpt: &mut impl MptBackend,
        base_paddr: usize,
        num_pages: usize,
    ) -> SbiRet {
        if base_paddr % PAGE_SIZE != 0 || num_pages == 0 {
            return SbiRet::invalid_param();
        }

        let len = match num_pages.checked_mul(PAGE_SIZE) {
            Some(l) => l,
            None => return SbiRet::invalid_param(),
        };

        // Only host-allocatable memory may be converted:
        // firmware, TSM image and MPT page pool must stay out of reach.
        if !platform.is_host_range(base_paddr, len) {
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

        // 1. Strip Host (SDID=0) access in MPT (NONE).
        if mpt.mpt_set(0, base_paddr, len, 0).is_err() {
            return SbiRet::failed();
        }

        // 2. Grant Confidential (SDID=1) access in MPT (RWX).
        if mpt.mpt_set(1, base_paddr, len, 7).is_err() {
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

    pub fn reclaim_pages(
        &mut self,
        mpt: &mut impl MptBackend,
        base_paddr: usize,
        num_pages: usize,
    ) -> SbiRet {
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

        // 1. Restore Host access (SDID=0, RWX).
        if mpt.mpt_set(0, base_paddr, len, 7).is_err() {
            return SbiRet::failed();
        }

        // 2. Fence
        mpt.mfence_pa(0, 0);

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
    pub fn convert_imsic_page(
        &mut self,
        platform: &PlatformLayout,
        mpt: &mut impl MptBackend,
        paddr: usize,
    ) -> SbiRet {
        if paddr % PAGE_SIZE != 0 || !platform.is_imsic_range(paddr, PAGE_SIZE) {
            return SbiRet::invalid_address();
        }
        // Only guest interrupt files are COVI-convertible (CoVE §11.3):
        // decompose against the platform geometry and reject supervisor
        // files (guest index 0) and foreign domain/hart positions. On an
        // IMSIC-less platform the whitelist is empty above (fail closed).
        if let Some(g) = platform.imsic_geometry() {
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

        // 1. Strip Host (SDID=0) access in MPT (NONE).
        if mpt.mpt_set(0, paddr, PAGE_SIZE, 0).is_err() {
            return SbiRet::failed();
        }

        // 2. Grant Confidential (SDID=1) read-write access in MPT (RW).
        if mpt.mpt_set(1, paddr, PAGE_SIZE, 0b011).is_err() {
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
    pub fn reclaim_imsic_page(&mut self, mpt: &mut impl MptBackend, paddr: usize) -> SbiRet {
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

        // 1. Restore Host (SDID=0) access (RWX).
        if mpt.mpt_set(0, paddr, PAGE_SIZE, 0b111).is_err() {
            return SbiRet::failed();
        }

        // 2. Withdraw Confidential (SDID=1) access (NONE).
        if mpt.mpt_set(1, paddr, PAGE_SIZE, 0).is_err() {
            return SbiRet::failed();
        }

        // 3. Fence so the restored host access is visible.
        mpt.mfence_pa(0, 0);

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
                let _ = vcpu_id; // The manager holds a single vCPU today, so the owner tuple is TVM-scoped.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    /// IMSIC geometry of the simulated platform: two domains, two harts,
    /// guest files enabled.
    fn test_imsic_geometry() -> RdsmImsicGeometry {
        RdsmImsicGeometry {
            range_start: 0x3000_8000,
            range_end: 0x3001_0000,
            domain_index_shift: 14,
            guest_index_bits: 1,
            hart_index_bits: 1,
        }
    }

    fn no_imsic() -> RdsmImsicGeometry {
        RdsmImsicGeometry::default()
    }

    fn platform_with_ram(base: usize, pages: usize) -> PlatformLayout {
        PlatformLayout::from_parts([(base, base + pages * PAGE_SIZE), (0, 0)], test_imsic_geometry())
    }

    /// MPT backend that records the tracker's calls. `fail_at` makes the
    /// n-th `mpt_set` call (1-based) fail.
    struct ScriptedBackend {
        sets: Vec<(usize, usize, usize, u8)>,
        fences: Vec<(usize, usize)>,
        fail_at: Option<usize>,
    }

    impl ScriptedBackend {
        fn new() -> Self {
            Self {
                sets: Vec::new(),
                fences: Vec::new(),
                fail_at: None,
            }
        }

        fn failing_at(call: usize) -> Self {
            Self {
                fail_at: Some(call),
                ..Self::new()
            }
        }
    }

    impl MptBackend for ScriptedBackend {
        fn mpt_set(
            &mut self,
            target_sdid: usize,
            paddr: usize,
            len: usize,
            perm: u8,
        ) -> Result<(), ()> {
            self.sets.push((target_sdid, paddr, len, perm));
            if self.fail_at == Some(self.sets.len()) {
                Err(())
            } else {
                Ok(())
            }
        }

        fn mfence_pa(&mut self, paddr: usize, sdid: usize) {
            self.fences.push((paddr, sdid));
        }
    }

    /// Page-aligned scratch RAM: the reclaim path scrubs through the raw
    /// address, so those pages must be real memory.
    #[repr(align(4096))]
    struct Ram([u8; 4 * PAGE_SIZE]);

    fn entry(tracker: &PageTracker, paddr: usize) -> PageEntry {
        *tracker
            .entries
            .iter()
            .flatten()
            .find(|slot| slot.paddr == paddr)
            .expect("page is tracked")
    }

    #[test]
    fn convert_pages_grants_the_confidential_domain() {
        let ram = Ram([0; 4 * PAGE_SIZE]);
        let base = ram.0.as_ptr() as usize;
        let platform = platform_with_ram(base, 4);
        let mut tracker = PageTracker::new();
        let mut mpt = ScriptedBackend::new();

        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base, 3),
            SbiRet::success(0)
        );
        assert_eq!(
            mpt.sets,
            std::vec![(0, base, 3 * PAGE_SIZE, 0), (1, base, 3 * PAGE_SIZE, 7)]
        );
        assert!(mpt.fences.is_empty());

        let entries: Vec<PageEntry> = tracker.entries.iter().flatten().copied().collect();
        assert_eq!(entries.len(), 3);
        for (i, slot) in entries.iter().enumerate() {
            assert_eq!(slot.paddr, base + i * PAGE_SIZE);
            assert_eq!(slot.state, PageState::ConvertedClean);
            assert_eq!(slot.owner_tvm_id, None);
            assert_eq!(slot.gpa, None);
        }
        assert!(tracker.is_converted_and_free(base, 3));
    }

    #[test]
    fn convert_pages_refuses_bad_arguments_before_touching_the_mpt() {
        let ram = Ram([0; 4 * PAGE_SIZE]);
        let base = ram.0.as_ptr() as usize;
        let platform = platform_with_ram(base, 4);
        let mut tracker = PageTracker::new();
        let mut mpt = ScriptedBackend::new();

        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base + 8, 1),
            SbiRet::invalid_param()
        );
        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base, 0),
            SbiRet::invalid_param()
        );
        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base, usize::MAX),
            SbiRet::invalid_param()
        );
        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base + PAGE_SIZE, 4),
            SbiRet::invalid_param()
        );

        assert!(mpt.sets.is_empty());
        assert!(tracker.entries.iter().all(|slot| slot.is_none()));
    }

    #[test]
    fn convert_pages_refuses_overlapping_ranges() {
        let ram = Ram([0; 4 * PAGE_SIZE]);
        let base = ram.0.as_ptr() as usize;
        let platform = platform_with_ram(base, 4);
        let mut tracker = PageTracker::new();
        let mut mpt = ScriptedBackend::new();

        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base, 1),
            SbiRet::success(0)
        );
        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base, 1),
            SbiRet::invalid_param()
        );
        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base, 2),
            SbiRet::invalid_param()
        );

        // Ownership does not make a tracked page convertible again.
        assert!(tracker.assign_page(base, PageState::AssignedPayload, 7));
        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base, 1),
            SbiRet::invalid_param()
        );
        assert_eq!(mpt.sets.len(), 2);
    }

    #[test]
    fn convert_pages_stops_at_the_tracker_capacity() {
        let base = 0x8000_0000;
        let platform = platform_with_ram(base, 600);
        let mut tracker = PageTracker::new();
        let mut mpt = ScriptedBackend::new();

        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base, MAX_PAGES),
            SbiRet::success(0)
        );
        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base + MAX_PAGES * PAGE_SIZE, 1),
            SbiRet::failed()
        );
        assert_eq!(mpt.sets.len(), 2);
    }

    #[test]
    fn reclaim_pages_scrubs_and_returns_the_pages_to_the_host() {
        let mut ram = Ram([0; 4 * PAGE_SIZE]);
        let base = ram.0.as_ptr() as usize;
        let platform = platform_with_ram(base, 4);
        let mut tracker = PageTracker::new();
        let mut mpt = ScriptedBackend::new();

        ram.0.fill(0xAA);
        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base, 2),
            SbiRet::success(0)
        );

        assert_eq!(tracker.reclaim_pages(&mut mpt, base, 2), SbiRet::success(0));
        assert!(ram.0[..2 * PAGE_SIZE].iter().all(|&b| b == 0));
        assert_eq!(
            mpt.sets,
            std::vec![
                (0, base, 2 * PAGE_SIZE, 0),
                (1, base, 2 * PAGE_SIZE, 7),
                (0, base, 2 * PAGE_SIZE, 7),
            ]
        );
        assert_eq!(mpt.fences, std::vec![(0, 0)]);
        assert!(tracker.entries.iter().all(|slot| slot.is_none()));
    }

    #[test]
    fn reclaim_pages_refuses_owned_or_untracked_pages() {
        let mut ram = Ram([0; 4 * PAGE_SIZE]);
        let base = ram.0.as_ptr() as usize;
        let platform = platform_with_ram(base, 4);
        let mut tracker = PageTracker::new();
        let mut mpt = ScriptedBackend::new();

        ram.0.fill(0xAA);
        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base, 2),
            SbiRet::success(0)
        );
        assert!(tracker.assign_page(base, PageState::AssignedPayload, 7));
        let calls_before = mpt.sets.len();

        assert_eq!(
            tracker.reclaim_pages(&mut mpt, base, 2),
            SbiRet::invalid_param()
        );
        assert_eq!(mpt.sets.len(), calls_before);
        assert!(ram.0[..2 * PAGE_SIZE].iter().all(|&b| b == 0xAA));

        assert_eq!(
            tracker.reclaim_pages(&mut mpt, base + 2 * PAGE_SIZE, 1),
            SbiRet::invalid_param()
        );
        assert_eq!(mpt.sets.len(), calls_before);
    }

    #[test]
    fn reclaim_pages_keeps_the_entries_when_the_host_restore_fails() {
        let mut ram = Ram([0; 4 * PAGE_SIZE]);
        let base = ram.0.as_ptr() as usize;
        let platform = platform_with_ram(base, 4);
        let mut tracker = PageTracker::new();
        let mut mpt = ScriptedBackend::failing_at(3);

        ram.0.fill(0xAA);
        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base, 2),
            SbiRet::success(0)
        );

        assert_eq!(tracker.reclaim_pages(&mut mpt, base, 2), SbiRet::failed());
        assert!(mpt.fences.is_empty());
        assert!(tracker.is_converted_and_free(base, 2));
    }

    #[test]
    fn convert_imsic_page_takes_a_guest_file() {
        let platform = PlatformLayout::from_parts([(0, 0), (0, 0)], test_imsic_geometry());
        let mut tracker = PageTracker::new();
        let mut mpt = ScriptedBackend::new();
        let paddr = 0x3000_D000;

        assert_eq!(
            tracker.convert_imsic_page(&platform, &mut mpt, paddr),
            SbiRet::success(0)
        );
        assert_eq!(
            mpt.sets,
            std::vec![(0, paddr, PAGE_SIZE, 0), (1, paddr, PAGE_SIZE, 3)]
        );
        assert!(tracker.is_free_imsic_page(paddr));
        assert_eq!(entry(&tracker, paddr).state, PageState::ImsicFile);
    }

    #[test]
    fn convert_imsic_page_refuses_bad_addresses_and_platforms() {
        let platform = PlatformLayout::from_parts([(0, 0), (0, 0)], test_imsic_geometry());
        let mut tracker = PageTracker::new();
        let mut mpt = ScriptedBackend::new();

        assert_eq!(
            tracker.convert_imsic_page(&platform, &mut mpt, 0x3000_D004),
            SbiRet::invalid_address()
        );
        assert_eq!(
            tracker.convert_imsic_page(&platform, &mut mpt, 0x3000_7000),
            SbiRet::invalid_address()
        );
        assert_eq!(
            tracker.convert_imsic_page(&platform, &mut mpt, 0x3000_C000),
            SbiRet::invalid_address()
        );

        let imsic_less = PlatformLayout::from_parts([(0, 0), (0, 0)], no_imsic());
        assert_eq!(
            tracker.convert_imsic_page(&imsic_less, &mut mpt, 0x3000_D000),
            SbiRet::invalid_address()
        );
        assert!(mpt.sets.is_empty());

        assert_eq!(
            tracker.convert_imsic_page(&platform, &mut mpt, 0x3000_D000),
            SbiRet::success(0)
        );
        assert_eq!(
            tracker.convert_imsic_page(&platform, &mut mpt, 0x3000_D000),
            SbiRet::invalid_param()
        );
    }

    #[test]
    fn reclaim_imsic_page_restores_host_access() {
        let platform = PlatformLayout::from_parts([(0, 0), (0, 0)], test_imsic_geometry());
        let mut tracker = PageTracker::new();
        let mut mpt = ScriptedBackend::new();
        let paddr = 0x3000_F000;

        assert_eq!(
            tracker.convert_imsic_page(&platform, &mut mpt, paddr),
            SbiRet::success(0)
        );
        assert_eq!(
            tracker.reclaim_imsic_page(&mut mpt, paddr),
            SbiRet::success(0)
        );
        assert_eq!(
            mpt.sets,
            std::vec![
                (0, paddr, PAGE_SIZE, 0),
                (1, paddr, PAGE_SIZE, 3),
                (0, paddr, PAGE_SIZE, 7),
                (1, paddr, PAGE_SIZE, 0),
            ]
        );
        assert_eq!(mpt.fences, std::vec![(0, 0)]);
        assert!(tracker.entries.iter().all(|slot| slot.is_none()));

        assert_eq!(
            tracker.reclaim_imsic_page(&mut mpt, paddr + 4),
            SbiRet::invalid_address()
        );
        assert_eq!(
            tracker.reclaim_imsic_page(&mut mpt, paddr),
            SbiRet::invalid_param()
        );

        assert_eq!(
            tracker.convert_imsic_page(&platform, &mut mpt, paddr),
            SbiRet::success(0)
        );
        assert_eq!(
            tracker.bind_imsic_page(paddr, 7, 0, 0x1000),
            SbiRet::success(0)
        );
        assert_eq!(
            tracker.reclaim_imsic_page(&mut mpt, paddr),
            SbiRet::invalid_param()
        );
        assert!(tracker.release_imsic_file(paddr, 7));
        assert_eq!(
            tracker.reclaim_imsic_page(&mut mpt, paddr),
            SbiRet::success(0)
        );
    }

    #[test]
    fn interrupt_file_binding_is_tvm_scoped() {
        let platform = PlatformLayout::from_parts([(0, 0), (0, 0)], test_imsic_geometry());
        let mut tracker = PageTracker::new();
        let mut mpt = ScriptedBackend::new();
        let paddr = 0x3000_D000;
        let gpa = 0x8000_0000;

        assert_eq!(
            tracker.convert_imsic_page(&platform, &mut mpt, paddr),
            SbiRet::success(0)
        );
        assert_eq!(
            tracker.bind_imsic_page(paddr, 7, 0, gpa),
            SbiRet::success(0)
        );
        assert_eq!(
            tracker.bind_imsic_page(paddr, 8, 0, gpa),
            SbiRet::invalid_param()
        );
        assert!(!tracker.is_free_imsic_page(paddr));

        let bound = entry(&tracker, paddr);
        assert_eq!(bound.state, PageState::ImsicFile);
        assert_eq!(bound.owner_tvm_id, Some(7));
        assert_eq!(bound.gpa, Some(gpa));

        assert!(!tracker.release_imsic_file(paddr, 8));
        assert_eq!(entry(&tracker, paddr).owner_tvm_id, Some(7));
        assert_eq!(entry(&tracker, paddr).gpa, Some(gpa));

        tracker.release_tvm_pages(7);
        let released = entry(&tracker, paddr);
        assert_eq!(released.state, PageState::ImsicFile);
        assert_eq!(released.owner_tvm_id, None);
        assert_eq!(released.gpa, None);
    }

    #[test]
    fn assignment_helpers_track_ownership() {
        let ram = Ram([0; 4 * PAGE_SIZE]);
        let base = ram.0.as_ptr() as usize;
        let platform = platform_with_ram(base, 4);
        let mut tracker = PageTracker::new();
        let mut mpt = ScriptedBackend::new();

        assert_eq!(tracker.find_free_page(), None);
        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base, 3),
            SbiRet::success(0)
        );
        assert_eq!(tracker.find_free_page(), Some(base));

        assert!(tracker.assign_page(base, PageState::AssignedPayload, 7));
        assert!(!tracker.assign_page(base, PageState::AssignedPayload, 7));
        assert!(tracker.assign_range(base + PAGE_SIZE, 2, PageState::AssignedPageTable, 7));
        assert!(!tracker.assign_range(base + PAGE_SIZE, 2, PageState::AssignedPageTable, 7));
        assert!(!tracker.assign_range(base + 2 * PAGE_SIZE, 2, PageState::AssignedPageTable, 7));

        tracker.release_page(base, 7);
        let freed = entry(&tracker, base);
        assert_eq!(freed.state, PageState::ConvertedClean);
        assert_eq!(freed.owner_tvm_id, None);

        assert!(tracker.assign_page(base, PageState::AssignedPayload, 8));
        tracker.release_page(base + PAGE_SIZE, 7);
        assert!(tracker.assign_page(base + PAGE_SIZE, PageState::AssignedPayload, 3));
        tracker.release_tvm_pages(8);
        assert_eq!(entry(&tracker, base).owner_tvm_id, None);
        assert_eq!(entry(&tracker, base + PAGE_SIZE).owner_tvm_id, Some(3));
    }

    #[test]
    fn shared_page_bookkeeping() {
        let ram = Ram([0; 4 * PAGE_SIZE]);
        let base = ram.0.as_ptr() as usize;
        let platform = platform_with_ram(base, 4);
        let mut tracker = PageTracker::new();
        let mut mpt = ScriptedBackend::new();
        let gpa = 0x8000_0000;

        assert_eq!(
            tracker.convert_pages(&platform, &mut mpt, base, 1),
            SbiRet::success(0)
        );
        // Sharing requires an owner.
        assert!(!tracker.set_shared(base, gpa, 7));

        assert!(tracker.assign_page(base, PageState::AssignedPayload, 7));
        assert!(tracker.set_shared(base, gpa, 7));
        assert_eq!(tracker.find_shared_by_gpa(gpa, 7), Some(base));
        assert_eq!(tracker.find_shared_by_gpa(gpa, 8), None);

        assert!(tracker.clear_shared(base, 7));
        let cleared = entry(&tracker, base);
        assert_eq!(cleared.state, PageState::AssignedPayload);
        assert_eq!(cleared.gpa, None);
        assert!(!tracker.clear_shared(base, 7));
    }

    #[test]
    fn platform_layout_derives_the_host_regions() {
        let info = RdsmPlatformInfo {
            ram_start: 0x8000_0000,
            ram_end: 0x9000_0000,
            tsm_region_start: 0x8040_0000,
            tsm_region_end: 0x8060_0000,
            mpt_pool_start: 0x8068_0000,
            mpt_pool_end: 0x8070_0000,
            imsic: test_imsic_geometry(),
        };
        let layout = PlatformLayout::new(&info);

        assert_eq!(
            layout.host_regions,
            [(0x8060_0000, 0x8068_0000), (0x8070_0000, 0x9000_0000)]
        );

        assert!(layout.is_host_range(0x8060_0000, 0x8_0000));
        assert!(layout.is_host_range(0x8070_0000, PAGE_SIZE));
        // Neither the MPT pool nor memory past RAM end is host memory.
        assert!(!layout.is_host_range(0x8068_0000, PAGE_SIZE));
        assert!(!layout.is_host_range(0x9000_0000, PAGE_SIZE));
        // Empty and overflowing ranges are refused.
        assert!(!layout.is_host_range(0x8060_0000, 0));
        assert!(!layout.is_host_range(usize::MAX - PAGE_SIZE, 2 * PAGE_SIZE));
        // A range straddling a region end is not contained.
        assert!(!layout.is_host_range(0x8067_F000, 2 * PAGE_SIZE));

        assert!(layout.is_imsic_range(0x3000_D000, PAGE_SIZE));
        assert!(!layout.is_imsic_range(0x3000_7000, PAGE_SIZE));
        assert!(layout.imsic_geometry().is_some());

        let imsic_less = PlatformLayout::from_parts([(0, 0), (0, 0)], no_imsic());
        assert!(!imsic_less.is_imsic_range(0x3000_D000, PAGE_SIZE));
        assert!(imsic_less.imsic_geometry().is_none());
    }
}
