use crate::SbiRet;
use crate::rdsm::{rdsm_mfence_pa, rdsm_mpt_set};

pub const PAGE_SIZE: usize = 4096;
pub const MAX_PAGES: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageState {
    NonConfidential,
    ConvertedClean,
    AssignedTvmRoot,
    AssignedTvmState,
    AssignedVcpuState,
    AssignedPageTable,
    AssignedPayload,
}

#[derive(Clone, Copy, Debug)]
pub struct PageEntry {
    pub paddr: usize,
    pub state: PageState,
    pub owner_tvm_id: Option<usize>,
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

        let len = num_pages * PAGE_SIZE;

        // Check if there is enough space in entries and no duplicates
        let mut free_count = 0;
        for slot in self.entries.iter() {
            if slot.is_none() {
                free_count += 1;
            } else if let Some(e) = slot {
                let e_end = e.paddr + PAGE_SIZE;
                let target_end = base_paddr + len;
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

        let len = num_pages * PAGE_SIZE;

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

    pub fn release_tvm_pages(&mut self, tvm_id: usize) {
        for slot in self.entries.iter_mut().flatten() {
            if slot.owner_tvm_id == Some(tvm_id) {
                slot.owner_tvm_id = None;
                slot.state = PageState::ConvertedClean;
            }
        }
    }
}

pub static mut PAGE_TRACKER: PageTracker = PageTracker::new();

pub fn page_tracker() -> &'static mut PageTracker {
    unsafe { &mut *core::ptr::addr_of_mut!(PAGE_TRACKER) }
}
