pub mod page_table;
pub mod vcpu;

use crate::SbiRet;
use crate::mm::{is_host_range, PAGE_SIZE, PageState, page_tracker};
use page_table::GStagePageTable;
use riscv_cove::host::TvmCreateParams;
use vcpu::Vcpu;

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
        // Phase 5.5: the parameter block must live in host-allocatable memory.
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

        let tracker = page_tracker();
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

        let tracker = page_tracker();
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
        // Phase 5.5: the source must be host-allocatable memory. Copying
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

        let tracker = page_tracker();
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

        let tracker = page_tracker();
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

        let tracker = page_tracker();
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

        vcpu.init_boot(entry_sepc, entry_arg, hgatp);

        tvm.entry_sepc = entry_sepc;
        tvm.entry_arg = entry_arg;

        if identity_addr != 0 {
            if identity_addr % 64 != 0 {
                return SbiRet::invalid_address();
            }
            // Phase 5.5: the identity block must be host-allocatable memory.
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

    pub fn run_tvm_vcpu(&mut self, tvm_id: usize, vcpu_id: usize) -> SbiRet {
        let tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };

        if tvm.state != TvmState::Runnable {
            return SbiRet::invalid_param();
        }

        let vcpu = match tvm.vcpu.as_mut() {
            Some(v) if v.id == vcpu_id => v,
            _ => return SbiRet::invalid_param(),
        };

        let res = vcpu.run();
        if res == 0 {
            let exit = vcpu::last_exit();
            SbiRet::success(exit.reason)
        } else {
            SbiRet::failed()
        }
    }

    // ---- Phase 4: COVG internal handlers (called from vcpu trap handler) ----

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
        let tracker = page_tracker();
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

    // ---- Phase 4: New COVH FIDs ----

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
        // Phase 5.5: shared pages must live in host-allocatable memory.
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
        let tracker = page_tracker();
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
        let _tvm = match self.active_tvm.as_mut() {
            Some(t) if t.id == tvm_id => t,
            _ => return SbiRet::invalid_param(),
        };

        let tracker = page_tracker();
        tracker.release_tvm_pages(tvm_id);

        self.active_tvm = None;

        SbiRet::success(0)
    }
}

pub static mut TVM_MANAGER: TvmManager = TvmManager::new();

pub fn tvm_manager() -> &'static mut TvmManager {
    unsafe { &mut *core::ptr::addr_of_mut!(TVM_MANAGER) }
}
