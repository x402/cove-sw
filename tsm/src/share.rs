//! Moving TVM pages between confidential and shared memory (CoVE v0.7
//! §12.3, §12.4).
//!
//! A shared page is ordinary host memory. When the TVM yields a page
//! (`sbi_covg_share_memory_region`, §12.3) the TSM scrubs it, restores the
//! host's access to it and cuts it out of the TVM. The host then maps shared
//! pages into the TVM with `sbi_covh_add_tvm_shared_pages` (§10.15), so after
//! the hand-over the G-stage leaf is the only TSM record of a shared page and
//! nothing in the page tracker describes it.
//!
//! Unsharing (§12.4) is the reverse direction and is destructive: the TSM
//! unmaps every page the host had mapped into the range, and a later guest
//! access faults and is served by the demand-zero path with a fresh
//! confidential page ("inserts confidential pages on page-faults", §8.2.2).
//!
//! Both operations are all-or-nothing: a failure that can still be undone
//! leaves nothing shared, and a failure that cannot be undone marks the TVM
//! poisoned so it never runs again.

use crate::mm::{MptBackend, PageTracker, PAGE_SIZE};
use crate::page_table::{
    GStagePageTable, leaf_is_present, leaf_is_retained, leaf_pa,
};

/// Supervisor domain id of the non-confidential host domain, the MPT root
/// the host accesses memory through.
const HOST_SDID: usize = 0;

/// A guest-physical range the TVM declared non-confidential with COVG share
/// (§12.3). The TSM maps host pages into it on COVH demand and refuses any
/// other write, so it is the only place where a guest page may hold host
/// memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedRegion {
    pub gpa_base: usize,
    pub len: usize,
}

/// How the shared-region list changes when a subrange stops being shared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reshape {
    /// The unshared range was the whole region.
    Clear,
    /// One side of the region stays shared; the slot keeps it.
    Shrink(SharedRegion),
    /// Both sides stay shared: the head keeps the slot, the tail takes
    /// `tail_slot`.
    Split {
        head: SharedRegion,
        tail: SharedRegion,
        tail_slot: usize,
    },
}

/// Why a range cannot be unshared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnsharePlanError {
    /// No single shared region contains the whole range.
    NotShared,
    /// The range splits a region and no slot is free for the tail.
    NoFreeSlot,
}

/// Plans the shared-region bookkeeping of COVG unshare (§12.4) over
/// `[gpa, gpa + len)`: which slot the range belongs to and how the list
/// changes. Changes nothing, so a caller can refuse the request before any
/// mapping is touched.
pub fn plan_unshare(
    regions: &[Option<SharedRegion>; 8],
    gpa: usize,
    len: usize,
) -> Result<(usize, Reshape), UnsharePlanError> {
    let end = match gpa.checked_add(len) {
        Some(end) => end,
        None => return Err(UnsharePlanError::NotShared),
    };
    let (index, region) = match regions.iter().enumerate().find_map(|(index, slot)| match slot {
        Some(region) if gpa >= region.gpa_base && end <= region.gpa_base + region.len => {
            Some((index, *region))
        }
        _ => None,
    }) {
        Some(found) => found,
        None => return Err(UnsharePlanError::NotShared),
    };

    let region_end = region.gpa_base + region.len;
    let reshape = if gpa == region.gpa_base && end == region_end {
        Reshape::Clear
    } else if gpa == region.gpa_base {
        // Prefix: the tail stays shared.
        Reshape::Shrink(SharedRegion {
            gpa_base: end,
            len: region_end - end,
        })
    } else if end == region_end {
        // Suffix: the head stays shared.
        Reshape::Shrink(SharedRegion {
            gpa_base: region.gpa_base,
            len: gpa - region.gpa_base,
        })
    } else {
        // Middle: the head keeps the slot, the tail needs another one.
        let tail_slot = match regions.iter().position(|slot| slot.is_none()) {
            Some(slot) => slot,
            None => return Err(UnsharePlanError::NoFreeSlot),
        };
        Reshape::Split {
            head: SharedRegion {
                gpa_base: region.gpa_base,
                len: gpa - region.gpa_base,
            },
            tail: SharedRegion {
                gpa_base: end,
                len: region_end - end,
            },
            tail_slot,
        }
    };
    Ok((index, reshape))
}

/// Applies a plan produced by [`plan_unshare`].
pub fn apply_unshare(regions: &mut [Option<SharedRegion>; 8], index: usize, reshape: Reshape) {
    match reshape {
        Reshape::Clear => regions[index] = None,
        Reshape::Shrink(region) => regions[index] = Some(region),
        Reshape::Split {
            head,
            tail,
            tail_slot,
        } => {
            regions[index] = Some(head);
            regions[tail_slot] = Some(tail);
        }
    }
}

/// Why a range could not be yielded to the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YieldFailure {
    /// A populated page of the range is not payload memory of this TVM, or
    /// the host is already removing it. Nothing changed.
    Refused,
    /// The host could not be given access to one page. Every page handed
    /// out before it was taken back, so nothing is shared; the contents of
    /// the pages already scrubbed are cleared.
    RolledBack,
    /// Taking a handed-out page back failed. That page was unmapped and
    /// forgotten, so the host keeps it and no TVM can receive it; the TVM
    /// has lost a page of its memory and must not run again.
    Poisoned,
}

/// COVG share (CoVE v0.7 §12.3) over `[gpa, gpa + len)`: scrubs every
/// populated page, restores the host's access to it and cuts it out of the
/// TVM. All-or-nothing.
///
/// The caller guarantees a page-multiple, nonzero `len` and holds the
/// manager and tracker locks.
pub fn yield_range(
    pt: &GStagePageTable,
    tracker: &mut PageTracker,
    mpt: &mut impl MptBackend,
    tvm_id: usize,
    gpa: usize,
    len: usize,
) -> Result<(), YieldFailure> {
    let num_pages = len / PAGE_SIZE;

    // Classify every page before anything changes: an unpopulated page is
    // nothing to hand over, a retained leaf is mid-removal by the host, and
    // a present leaf must map a payload page of this TVM.
    for i in 0..num_pages {
        let page_gpa = gpa + i * PAGE_SIZE;
        match leaf_4k_present_or_retained(pt, page_gpa) {
            Leaf::Unpopulated => {}
            Leaf::Retained => return Err(YieldFailure::Refused),
            Leaf::Present(spa) => {
                if !tracker.is_tvms_payload(spa, tvm_id) {
                    return Err(YieldFailure::Refused);
                }
            }
        }
    }

    // Hand the pages over in ascending order. On the first failure every
    // page handed out before it is taken back, newest first.
    for i in 0..num_pages {
        let page_gpa = gpa + i * PAGE_SIZE;
        let spa = match leaf_4k_present_or_retained(pt, page_gpa) {
            Leaf::Present(spa) => spa,
            _ => continue,
        };
        // Scrub before the host can read the page (CoVE §5.2.4).
        // TODO(multi-hart): scrubbing while another vCPU of this TVM could
        // still run on another hart is unsafe; today the manager lock
        // serialises every operation on one hart.
        unsafe {
            core::ptr::write_bytes(spa as *mut u8, 0, PAGE_SIZE);
        }
        if mpt.mpt_set(HOST_SDID, spa, PAGE_SIZE, 0b111).is_err() {
            let mut poisoned = false;
            for j in (0..i).rev() {
                let head_gpa = gpa + j * PAGE_SIZE;
                let head_spa = match leaf_4k_present_or_retained(pt, head_gpa) {
                    Leaf::Present(spa) => spa,
                    _ => continue,
                };
                if mpt.mpt_set(HOST_SDID, head_spa, PAGE_SIZE, 0).is_err() {
                    // The host's access cannot be withdrawn: keep the page
                    // out of every TVM by cutting it out for good.
                    let _ = pt.unmap_4k(head_gpa);
                    tracker.forget_page(head_spa, tvm_id);
                    poisoned = true;
                }
            }
            mpt.mfence_pa(0, 0);
            return Err(if poisoned {
                YieldFailure::Poisoned
            } else {
                YieldFailure::RolledBack
            });
        }
    }

    // Cut the handed-out pages out of the TVM: the leaves go away and the
    // tracker stops describing them.
    for i in 0..num_pages {
        let page_gpa = gpa + i * PAGE_SIZE;
        let spa = match leaf_4k_present_or_retained(pt, page_gpa) {
            Leaf::Present(spa) => spa,
            _ => continue,
        };
        let _ = pt.unmap_4k(page_gpa);
        tracker.forget_page(spa, tvm_id);
    }

    mpt.mfence_pa(0, 0);
    Ok(())
}

/// COVG unshare (CoVE v0.7 §12.4) over `[gpa, gpa + len)`: unmaps every page
/// the host had mapped into the range. All-or-nothing.
///
/// Only pages the tracker does not know are shared pages; a leaf that maps a
/// tracked page belongs to a confidential mapping and is refused.
pub fn retake_range(
    pt: &GStagePageTable,
    tracker: &PageTracker,
    gpa: usize,
    len: usize,
) -> Result<(), ()> {
    let num_pages = len / PAGE_SIZE;

    for i in 0..num_pages {
        let page_gpa = gpa + i * PAGE_SIZE;
        match leaf_4k_present_or_retained(pt, page_gpa) {
            Leaf::Unpopulated => {}
            Leaf::Retained => {}
            Leaf::Present(spa) => {
                if tracker.lookup(spa).is_some() {
                    return Err(());
                }
            }
        }
    }

    for i in 0..num_pages {
        let page_gpa = gpa + i * PAGE_SIZE;
        match leaf_4k_present_or_retained(pt, page_gpa) {
            Leaf::Unpopulated => {}
            Leaf::Present(_) => {
                let _ = pt.unmap_4k(page_gpa);
            }
            Leaf::Retained => {
                let _ = pt.remove_invalid_4k(page_gpa);
            }
        }
    }

    Ok(())
}

/// Classification of the level-0 entry of one guest page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leaf {
    /// No last-level table, or an empty entry.
    Unpopulated,
    /// A present mapping to the given physical address.
    Present(usize),
    /// A mapping retained by `invalidate_4k`, waiting for validate or remove.
    Retained,
}

fn leaf_4k_present_or_retained(pt: &GStagePageTable, gpa: usize) -> Leaf {
    match pt.leaf_4k(gpa) {
        None | Some(0) => Leaf::Unpopulated,
        Some(pte) if leaf_is_present(pte) => Leaf::Present(leaf_pa(pte)),
        Some(pte) => {
            debug_assert!(leaf_is_retained(pte));
            Leaf::Retained
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mm::{PageState, PlatformLayout};
    use crate::page_table::PtPool;
    use crate::sbi::SbiRet;
    use rdsm_abi::RdsmImsicGeometry;
    use std::vec::Vec;

    const PAGE: usize = PAGE_SIZE;

    /// MPT backend that records the calls and can fail a set of them, by
    /// 0-based call index.
    struct ScriptedBackend {
        sets: Vec<(usize, usize, usize, u8)>,
        fences: Vec<(usize, usize)>,
        fail_mask: u32,
    }

    impl ScriptedBackend {
        fn new() -> Self {
            Self {
                sets: Vec::new(),
                fences: Vec::new(),
                fail_mask: 0,
            }
        }

        fn failing_at(calls: &[usize]) -> Self {
            let mut backend = Self::new();
            for &call in calls {
                backend.fail_mask |= 1 << call;
            }
            backend
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
            let index = self.sets.len();
            self.sets.push((target_sdid, paddr, len, perm));
            if self.fail_mask & (1 << index) != 0 {
                Err(())
            } else {
                Ok(())
            }
        }

        fn mfence_pa(&mut self, paddr: usize, sdid: usize) {
            self.fences.push((paddr, sdid));
        }
    }

    /// Page-aligned scratch RAM: payload pages are scrubbed through their
    /// raw address, so they must be real memory.
    #[repr(align(4096))]
    struct Ram([u8; 8 * PAGE]);

    /// Backing store for the 16 KiB root table plus the two table pages a
    /// 2 MiB span needs.
    #[repr(align(16384))]
    struct PtMemory([u8; 6 * PAGE]);

    /// Table pages after the 16 KiB root table.
    fn table_pages(root: usize) -> [usize; 2] {
        [root + 16384, root + 16384 + PAGE]
    }

    fn platform_for(base: usize) -> PlatformLayout {
        PlatformLayout::from_parts(
            [(base, base + 8 * PAGE), (0, 0)],
            RdsmImsicGeometry::default(),
        )
    }

    fn state_of(tracker: &PageTracker, paddr: usize) -> Option<PageState> {
        tracker.lookup(paddr).map(|entry| entry.state)
    }

    #[test]
    fn yield_scrubs_grants_and_cuts_out_payload_pages() {
        let mut ram = Ram([0; 8 * PAGE]);
        let mut ptmem = PtMemory([0; 6 * PAGE]);
        let ram_base = ram.0.as_mut_ptr() as usize;
        let root = ptmem.0.as_mut_ptr() as usize;
        let table = GStagePageTable::new(root);
        let pages = table_pages(root);
        let mut used = 0;

        let gpa = 0x8001_0000;
        let spa0 = ram_base;
        let spa1 = ram_base + PAGE;
        let spa2 = ram_base + 2 * PAGE;
        {
            let mut pool = PtPool::new(&pages, &mut used);
            assert_eq!(table.map_4k(gpa, spa0, &mut pool), Ok(()));
            assert_eq!(table.map_4k(gpa + PAGE, spa1, &mut pool), Ok(()));
            assert_eq!(table.map_4k(gpa + 2 * PAGE, spa2, &mut pool), Ok(()));
        }

        let platform = platform_for(ram_base);
        let mut tracker = PageTracker::new();
        let mut setup = ScriptedBackend::new();
        assert_eq!(
            tracker.convert_pages(&platform, &mut setup, ram_base, 4),
            SbiRet::success(0)
        );
        for i in 0..3 {
            assert!(tracker.assign_page(ram_base + i * PAGE, PageState::AssignedPayload, 7));
        }
        ram.0.fill(0xAA);

        let mut mpt = ScriptedBackend::new();
        // Four pages requested: the fourth GPA has no mapping.
        assert_eq!(
            yield_range(&table, &mut tracker, &mut mpt, 7, gpa, 4 * PAGE),
            Ok(())
        );

        assert!(ram.0[..3 * PAGE].iter().all(|&b| b == 0));
        assert_eq!(
            mpt.sets,
            std::vec![
                (0, spa0, PAGE, 7),
                (0, spa1, PAGE, 7),
                (0, spa2, PAGE, 7),
            ]
        );
        assert_eq!(mpt.fences, std::vec![(0, 0)]);
        for i in 0..3 {
            assert_eq!(table.leaf_4k(gpa + i * PAGE), Some(0));
        }
        for spa in [spa0, spa1, spa2] {
            assert_eq!(tracker.lookup(spa), None);
        }
        // The unrelated tracked page is untouched.
        assert_eq!(
            state_of(&tracker, ram_base + 3 * PAGE),
            Some(PageState::ConvertedClean)
        );

        // The intermediate tables are still there: mapping the fourth page
        // in the same span needs no pool page.
        assert_eq!(table.plan_4k(gpa + 3 * PAGE), Ok(0));
        let none: [usize; 0] = [];
        let mut spare_used = 0;
        let mut pool = PtPool::new(&none, &mut spare_used);
        assert_eq!(
            table.map_4k(gpa + 3 * PAGE, ram_base + 4 * PAGE, &mut pool),
            Ok(())
        );
    }

    #[test]
    fn yield_refuses_pages_it_may_not_take() {
        let mut ram = Ram([0; 8 * PAGE]);
        let mut ptmem = PtMemory([0; 6 * PAGE]);
        let ram_base = ram.0.as_mut_ptr() as usize;
        let root = ptmem.0.as_mut_ptr() as usize;
        let table = GStagePageTable::new(root);
        let pages = table_pages(root);
        let mut used = 0;

        let gpa_a = 0x8001_0000;
        let gpa_b = 0x8001_1000;
        let gpa_c = 0x8001_2000;
        let gpa_d = 0x8001_3000;
        let untracked = ram_base + 7 * PAGE;
        let other_tvm_page = ram_base + 3 * PAGE;
        let other_state_page = ram_base + 4 * PAGE;
        let retained_page = ram_base + 1 * PAGE;

        let platform = platform_for(ram_base);
        let mut tracker = PageTracker::new();
        let mut setup = ScriptedBackend::new();
        assert_eq!(
            tracker.convert_pages(&platform, &mut setup, ram_base, 6),
            SbiRet::success(0)
        );
        assert!(tracker.assign_page(other_tvm_page, PageState::AssignedPayload, 8));
        assert!(tracker.assign_page(other_state_page, PageState::AssignedPageTable, 7));
        assert!(tracker.assign_page(retained_page, PageState::AssignedPayload, 7));

        {
            let mut pool = PtPool::new(&pages, &mut used);
            assert_eq!(table.map_4k(gpa_a, untracked, &mut pool), Ok(()));
            assert_eq!(table.map_4k(gpa_b, other_tvm_page, &mut pool), Ok(()));
            assert_eq!(table.map_4k(gpa_c, other_state_page, &mut pool), Ok(()));
            assert_eq!(table.map_4k(gpa_d, retained_page, &mut pool), Ok(()));
        }
        assert_eq!(table.invalidate_4k(gpa_d), Ok(()));
        ram.0.fill(0x5A);

        let leaves = [
            table.leaf_4k(gpa_a),
            table.leaf_4k(gpa_b),
            table.leaf_4k(gpa_c),
            table.leaf_4k(gpa_d),
        ];
        let mut mpt = ScriptedBackend::new();
        for gpa in [gpa_a, gpa_b, gpa_c, gpa_d] {
            assert_eq!(
                yield_range(&table, &mut tracker, &mut mpt, 7, gpa, PAGE),
                Err(YieldFailure::Refused)
            );
        }

        assert!(mpt.sets.is_empty());
        assert!(mpt.fences.is_empty());
        assert_eq!(
            leaves,
            [
                table.leaf_4k(gpa_a),
                table.leaf_4k(gpa_b),
                table.leaf_4k(gpa_c),
                table.leaf_4k(gpa_d),
            ]
        );
        assert_eq!(tracker.lookup(other_tvm_page).map(|e| e.owner_tvm_id), Some(Some(8)));
        assert_eq!(state_of(&tracker, other_state_page), Some(PageState::AssignedPageTable));
        assert!(tracker.is_tvms_payload(retained_page, 7));
        assert!(ram.0.iter().all(|&b| b == 0x5A));
    }

    #[test]
    fn a_failed_grant_rolls_the_handed_out_pages_back() {
        let mut ram = Ram([0; 8 * PAGE]);
        let mut ptmem = PtMemory([0; 6 * PAGE]);
        let ram_base = ram.0.as_mut_ptr() as usize;
        let root = ptmem.0.as_mut_ptr() as usize;
        let table = GStagePageTable::new(root);
        let pages = table_pages(root);
        let mut used = 0;

        let gpa = 0x8001_0000;
        let spas = [ram_base, ram_base + PAGE, ram_base + 2 * PAGE];
        {
            let mut pool = PtPool::new(&pages, &mut used);
            for (i, spa) in spas.iter().enumerate() {
                assert_eq!(table.map_4k(gpa + i * PAGE, *spa, &mut pool), Ok(()));
            }
        }

        let platform = platform_for(ram_base);
        let mut tracker = PageTracker::new();
        let mut setup = ScriptedBackend::new();
        assert_eq!(
            tracker.convert_pages(&platform, &mut setup, ram_base, 3),
            SbiRet::success(0)
        );
        for spa in spas {
            assert!(tracker.assign_page(spa, PageState::AssignedPayload, 7));
        }
        ram.0.fill(0xAA);

        let mut mpt = ScriptedBackend::failing_at(&[1]);
        assert_eq!(
            yield_range(&table, &mut tracker, &mut mpt, 7, gpa, 3 * PAGE),
            Err(YieldFailure::RolledBack)
        );

        assert_eq!(
            mpt.sets,
            std::vec![
                (0, spas[0], PAGE, 7),
                (0, spas[1], PAGE, 7),
                (0, spas[0], PAGE, 0),
            ]
        );
        assert_eq!(mpt.fences, std::vec![(0, 0)]);
        for (i, spa) in spas.iter().enumerate() {
            let leaf = table.leaf_4k(gpa + i * PAGE).expect("leaf present");
            assert!(leaf_is_present(leaf));
            assert_eq!(leaf_pa(leaf), *spa);
            assert!(tracker.is_tvms_payload(*spa, 7));
        }
        // The first two pages were scrubbed before their grants, the third
        // was never reached.
        assert!(ram.0[..2 * PAGE].iter().all(|&b| b == 0));
        assert!(ram.0[2 * PAGE..3 * PAGE].iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn a_failed_revoke_poisons_the_tvm() {
        let mut ram = Ram([0; 8 * PAGE]);
        let mut ptmem = PtMemory([0; 6 * PAGE]);
        let ram_base = ram.0.as_mut_ptr() as usize;
        let root = ptmem.0.as_mut_ptr() as usize;
        let table = GStagePageTable::new(root);
        let pages = table_pages(root);
        let mut used = 0;

        let gpa = 0x8001_0000;
        let spas = [ram_base, ram_base + PAGE, ram_base + 2 * PAGE];
        {
            let mut pool = PtPool::new(&pages, &mut used);
            for (i, spa) in spas.iter().enumerate() {
                assert_eq!(table.map_4k(gpa + i * PAGE, *spa, &mut pool), Ok(()));
            }
        }

        let platform = platform_for(ram_base);
        let mut tracker = PageTracker::new();
        let mut setup = ScriptedBackend::new();
        assert_eq!(
            tracker.convert_pages(&platform, &mut setup, ram_base, 3),
            SbiRet::success(0)
        );
        for spa in spas {
            assert!(tracker.assign_page(spa, PageState::AssignedPayload, 7));
        }

        // Second grant fails, and so does the revoke of the first page.
        let mut mpt = ScriptedBackend::failing_at(&[1, 2]);
        assert_eq!(
            yield_range(&table, &mut tracker, &mut mpt, 7, gpa, 3 * PAGE),
            Err(YieldFailure::Poisoned)
        );

        assert_eq!(
            mpt.sets,
            std::vec![
                (0, spas[0], PAGE, 7),
                (0, spas[1], PAGE, 7),
                (0, spas[0], PAGE, 0),
            ]
        );
        assert_eq!(mpt.fences, std::vec![(0, 0)]);
        // The first page is cut out of the TVM for good.
        assert_eq!(table.leaf_4k(gpa), Some(0));
        assert_eq!(tracker.lookup(spas[0]), None);
        // The remaining two keep their mapping and their owner.
        for i in 1..3 {
            let leaf = table.leaf_4k(gpa + i * PAGE).expect("leaf present");
            assert!(leaf_is_present(leaf));
            assert_eq!(leaf_pa(leaf), spas[i]);
            assert!(tracker.is_tvms_payload(spas[i], 7));
        }
    }

    #[test]
    fn retake_unmaps_shared_leaves_only() {
        let mut ram = Ram([0; 8 * PAGE]);
        let mut ptmem = PtMemory([0; 6 * PAGE]);
        let ram_base = ram.0.as_mut_ptr() as usize;
        let root = ptmem.0.as_mut_ptr() as usize;
        let table = GStagePageTable::new(root);
        let pages = table_pages(root);
        let mut used = 0;

        let gpa_a = 0x8001_0000;
        let gpa_b = 0x8001_1000;
        let gpa_c = 0x8001_2000;
        let gpa_d = 0x8001_3000;
        let gpa_e = 0x8001_4000;
        let untracked_a = ram_base + 4 * PAGE;
        let untracked_b = ram_base + 5 * PAGE;
        let untracked_c = ram_base + 6 * PAGE;
        let tracked = ram_base;

        {
            let mut pool = PtPool::new(&pages, &mut used);
            assert_eq!(table.map_4k(gpa_a, untracked_a, &mut pool), Ok(()));
            assert_eq!(table.map_4k(gpa_b, untracked_b, &mut pool), Ok(()));
            assert_eq!(table.map_4k(gpa_c, untracked_c, &mut pool), Ok(()));
            assert_eq!(table.map_4k(gpa_e, tracked, &mut pool), Ok(()));
        }
        assert_eq!(table.invalidate_4k(gpa_c), Ok(()));

        let platform = platform_for(ram_base);
        let mut tracker = PageTracker::new();
        let mut setup = ScriptedBackend::new();
        assert_eq!(
            tracker.convert_pages(&platform, &mut setup, ram_base, 1),
            SbiRet::success(0)
        );
        assert!(tracker.assign_page(tracked, PageState::AssignedPayload, 7));

        // A leaf that maps a tracked page is a confidential mapping, not a
        // shared page: nothing may change.
        let leaves = [
            table.leaf_4k(gpa_a),
            table.leaf_4k(gpa_b),
            table.leaf_4k(gpa_c),
            table.leaf_4k(gpa_d),
            table.leaf_4k(gpa_e),
        ];
        assert_eq!(
            retake_range(&table, &tracker, gpa_a, 5 * PAGE),
            Err(())
        );
        assert_eq!(
            leaves,
            [
                table.leaf_4k(gpa_a),
                table.leaf_4k(gpa_b),
                table.leaf_4k(gpa_c),
                table.leaf_4k(gpa_d),
                table.leaf_4k(gpa_e),
            ]
        );
        assert!(tracker.is_tvms_payload(tracked, 7));

        // Two present leaves, one retained leaf and one absent GPA.
        assert_eq!(retake_range(&table, &tracker, gpa_a, 4 * PAGE), Ok(()));
        for gpa in [gpa_a, gpa_b, gpa_c, gpa_d] {
            assert_eq!(table.leaf_4k(gpa), Some(0));
        }
        // The tracker never knew these pages and still does not.
        assert_eq!(tracker.lookup(untracked_a), None);
        assert_eq!(tracker.lookup(untracked_b), None);
        assert_eq!(tracker.lookup(untracked_c), None);
        // The tracked page's leaf is untouched.
        assert!(leaf_is_present(table.leaf_4k(gpa_e).expect("leaf present")));
    }

    fn empty_regions() -> [Option<SharedRegion>; 8] {
        [None; 8]
    }

    fn region(gpa_base: usize, len: usize) -> SharedRegion {
        SharedRegion { gpa_base, len }
    }

    #[test]
    fn unshare_plan_shrinks_or_clears_a_containing_region() {
        let base = 0x8001_0000;
        let mut regions = empty_regions();
        regions[2] = Some(region(base, 0x4000));

        // Exact match clears the slot.
        assert_eq!(
            plan_unshare(&regions, base, 0x4000),
            Ok((2, Reshape::Clear))
        );
        // Prefix keeps the tail.
        assert_eq!(
            plan_unshare(&regions, base, 0x1000),
            Ok((
                2,
                Reshape::Shrink(region(base + 0x1000, 0x3000)),
            ))
        );
        // Suffix keeps the head.
        assert_eq!(
            plan_unshare(&regions, base + 0x3000, 0x1000),
            Ok((2, Reshape::Shrink(region(base, 0x3000))))
        );
        // Middle splits, head in place and tail in the first free slot.
        assert_eq!(
            plan_unshare(&regions, base + 0x1000, 0x1000),
            Ok((
                2,
                Reshape::Split {
                    head: region(base, 0x1000),
                    tail: region(base + 0x2000, 0x2000),
                    tail_slot: 0,
                },
            ))
        );
        // Planning changes nothing.
        assert_eq!(regions[2], Some(region(base, 0x4000)));
        assert!(regions.iter().enumerate().all(|(i, slot)| i == 2 || slot.is_none()));
    }

    #[test]
    fn unshare_plan_refuses_missing_or_split_regions() {
        let base = 0x8001_0000;
        let mut regions = empty_regions();
        regions[0] = Some(region(base, 0x1000));
        regions[1] = Some(region(base + 0x1000, 0x1000));

        // Outside every region, and a range that spans two of them.
        assert_eq!(
            plan_unshare(&regions, 0x8002_0000, 0x1000),
            Err(UnsharePlanError::NotShared)
        );
        assert_eq!(
            plan_unshare(&regions, base, 0x2000),
            Err(UnsharePlanError::NotShared)
        );
        assert_eq!(
            plan_unshare(&regions, base, 0x2001),
            Err(UnsharePlanError::NotShared)
        );
        // A range starting inside but ending outside is not contained.
        assert_eq!(
            plan_unshare(&regions, base + 0x800, 0x1000),
            Err(UnsharePlanError::NotShared)
        );

        // A middle range without a free slot cannot be split.
        regions[0] = Some(region(base, 0x4000));
        for slot in regions.iter_mut().skip(1) {
            *slot = Some(region(base + 0x10_0000, 0x1000));
        }
        assert_eq!(
            plan_unshare(&regions, base + 0x1000, 0x1000),
            Err(UnsharePlanError::NoFreeSlot)
        );
        // Every refusal above left the list as it was.
        assert_eq!(regions[0], Some(region(base, 0x4000)));
        for slot in regions.iter().skip(1) {
            assert_eq!(*slot, Some(region(base + 0x10_0000, 0x1000)));
        }
    }

    #[test]
    fn apply_unshare_writes_each_reshape() {
        let base = 0x8001_0000;

        let mut regions = empty_regions();
        regions[1] = Some(region(base, 0x3000));
        apply_unshare(&mut regions, 1, Reshape::Clear);
        assert!(regions.iter().all(|slot| slot.is_none()));

        regions[1] = Some(region(base, 0x3000));
        apply_unshare(&mut regions, 1, Reshape::Shrink(region(base, 0x1000)));
        assert_eq!(regions[1], Some(region(base, 0x1000)));

        regions[1] = Some(region(base, 0x3000));
        apply_unshare(
            &mut regions,
            1,
            Reshape::Split {
                head: region(base, 0x1000),
                tail: region(base + 0x2000, 0x1000),
                tail_slot: 5,
            },
        );
        assert_eq!(regions[1], Some(region(base, 0x1000)));
        assert_eq!(regions[5], Some(region(base + 0x2000, 0x1000)));
        assert_eq!(
            regions.iter().flatten().count(),
            2,
            "a split must not duplicate the region"
        );
    }
}
