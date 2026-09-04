pub const PTE_V: u64 = 1 << 0;
pub const PTE_R: u64 = 1 << 1;
pub const PTE_W: u64 = 1 << 2;
pub const PTE_X: u64 = 1 << 3;
pub const PTE_U: u64 = 1 << 4;
pub const PTE_A: u64 = 1 << 6;
pub const PTE_D: u64 = 1 << 7;

pub const PTE_LEAF_RWX: u64 = PTE_V | PTE_R | PTE_W | PTE_X | PTE_U | PTE_A | PTE_D; // 0xDF

pub const HGATP_MODE_BARE: usize = 0;
pub const HGATP_MODE_SV39X4: usize = 8;
pub const HGATP_MODE_SV48X4: usize = 9;

pub struct GStagePageTable {
    pub root_paddr: usize, // 16KB aligned
}

impl GStagePageTable {
    pub fn new(root_paddr: usize) -> Self {
        // Clear the 16KB root table (2048 entries of 8 bytes)
        unsafe {
            core::ptr::write_bytes(root_paddr as *mut u8, 0, 16384);
        }
        Self { root_paddr }
    }

    pub fn map_4k(
        &self,
        gpa: usize,
        spa: usize,
        alloc_page: &mut impl FnMut() -> Option<usize>,
    ) -> Result<(), ()> {
        if gpa % 4096 != 0 || spa % 4096 != 0 {
            return Err(());
        }
        // Sv39x4: GPA valid width is 41 bits (0 .. 1 << 41)
        if gpa >= (1usize << 41) {
            return Err(());
        }

        let vpn2 = (gpa >> 30) & 0x7FF; // 11 bits: 0..2047
        let vpn1 = (gpa >> 21) & 0x1FF; // 9 bits: 0..511
        let vpn0 = (gpa >> 12) & 0x1FF; // 9 bits: 0..511

        let root_slice =
            unsafe { core::slice::from_raw_parts_mut(self.root_paddr as *mut u64, 2048) };

        // Level 2 (Root) -> Level 1 table
        let l1_paddr = if root_slice[vpn2] & PTE_V == 0 {
            let paddr = alloc_page().ok_or(())?;
            unsafe {
                core::ptr::write_bytes(paddr as *mut u8, 0, 4096);
            }
            root_slice[vpn2] = (((paddr >> 12) as u64) << 10) | PTE_V;
            paddr
        } else {
            ((root_slice[vpn2] >> 10) << 12) as usize
        };

        let l1_slice = unsafe { core::slice::from_raw_parts_mut(l1_paddr as *mut u64, 512) };

        // Level 1 -> Level 0 table
        let l0_paddr = if l1_slice[vpn1] & PTE_V == 0 {
            let paddr = alloc_page().ok_or(())?;
            unsafe {
                core::ptr::write_bytes(paddr as *mut u8, 0, 4096);
            }
            l1_slice[vpn1] = (((paddr >> 12) as u64) << 10) | PTE_V;
            paddr
        } else {
            ((l1_slice[vpn1] >> 10) << 12) as usize
        };

        let l0_slice = unsafe { core::slice::from_raw_parts_mut(l0_paddr as *mut u64, 512) };

        // Level 0 Leaf entry
        l0_slice[vpn0] = (((spa >> 12) as u64) << 10) | PTE_LEAF_RWX;

        Ok(())
    }

    /// Walk the page table without allocating; return the SPA for a mapped 4K page.
    pub fn lookup_4k(&self, gpa: usize) -> Result<usize, ()> {
        if gpa % 4096 != 0 || gpa >= (1usize << 41) {
            return Err(());
        }
        let vpn2 = (gpa >> 30) & 0x7FF;
        let vpn1 = (gpa >> 21) & 0x1FF;
        let vpn0 = (gpa >> 12) & 0x1FF;

        let root = unsafe { core::slice::from_raw_parts(self.root_paddr as *const u64, 2048) };
        if root[vpn2] & PTE_V == 0 {
            return Err(());
        }
        let l1_paddr = ((root[vpn2] >> 10) << 12) as usize;
        let l1 = unsafe { core::slice::from_raw_parts(l1_paddr as *const u64, 512) };
        if l1[vpn1] & PTE_V == 0 {
            return Err(());
        }
        let l0_paddr = ((l1[vpn1] >> 10) << 12) as usize;
        let l0 = unsafe { core::slice::from_raw_parts(l0_paddr as *const u64, 512) };
        if l0[vpn0] & PTE_V == 0 {
            return Err(());
        }
        Ok(((l0[vpn0] >> 10) << 12) as usize)
    }

    fn raw_lookup_4k(&self, gpa: usize) -> Result<u64, ()> {
        if gpa % 4096 != 0 || gpa >= (1usize << 41) {
            return Err(());
        }
        let vpn2 = (gpa >> 30) & 0x7FF;
        let vpn1 = (gpa >> 21) & 0x1FF;
        let vpn0 = (gpa >> 12) & 0x1FF;

        let root = unsafe { core::slice::from_raw_parts(self.root_paddr as *const u64, 2048) };
        if root[vpn2] & PTE_V == 0 {
            return Err(());
        }
        let l1_paddr = ((root[vpn2] >> 10) << 12) as usize;
        let l1 = unsafe { core::slice::from_raw_parts(l1_paddr as *const u64, 512) };
        if l1[vpn1] & PTE_V == 0 {
            return Err(());
        }
        let l0_paddr = ((l1[vpn1] >> 10) << 12) as usize;
        let l0 = unsafe { core::slice::from_raw_parts(l0_paddr as *const u64, 512) };
        Ok(l0[vpn0])
    }

    /// Temporarily block a present leaf mapping while retaining its identity.
    pub fn invalidate_4k(&self, gpa: usize) -> Result<(), ()> {
        let vpn0 = (gpa >> 12) & 0x1FF;
        let pte = self.raw_lookup_4k(gpa)?;
        if pte & PTE_V == 0 {
            return Err(());
        }

        let root = unsafe { core::slice::from_raw_parts(self.root_paddr as *const u64, 2048) };
        let vpn2 = (gpa >> 30) & 0x7FF;
        let vpn1 = (gpa >> 21) & 0x1FF;
        let l1_paddr = ((root[vpn2] >> 10) << 12) as usize;
        let l1 = unsafe { core::slice::from_raw_parts(l1_paddr as *const u64, 512) };
        let l0_paddr = ((l1[vpn1] >> 10) << 12) as usize;
        let l0 = unsafe { core::slice::from_raw_parts_mut(l0_paddr as *mut u64, 512) };
        l0[vpn0] = pte & !PTE_V;
        Ok(())
    }

    /// Restore a mapping retained by `invalidate_4k`.
    pub fn validate_4k(&self, gpa: usize) -> Result<(), ()> {
        let vpn0 = (gpa >> 12) & 0x1FF;
        let pte = self.raw_lookup_4k(gpa)?;
        if pte == 0 || pte & PTE_V != 0 {
            return Err(());
        }

        let root = unsafe { core::slice::from_raw_parts(self.root_paddr as *const u64, 2048) };
        let vpn2 = (gpa >> 30) & 0x7FF;
        let vpn1 = (gpa >> 21) & 0x1FF;
        let l1_paddr = ((root[vpn2] >> 10) << 12) as usize;
        let l1 = unsafe { core::slice::from_raw_parts(l1_paddr as *const u64, 512) };
        let l0_paddr = ((l1[vpn1] >> 10) << 12) as usize;
        let l0 = unsafe { core::slice::from_raw_parts_mut(l0_paddr as *mut u64, 512) };
        l0[vpn0] = pte | PTE_V;
        Ok(())
    }

    /// Remove an invalidated leaf and return the SPA it had retained.
    pub fn remove_invalid_4k(&self, gpa: usize) -> Result<usize, ()> {
        let vpn0 = (gpa >> 12) & 0x1FF;
        let pte = self.raw_lookup_4k(gpa)?;
        if pte == 0 || pte & PTE_V != 0 {
            return Err(());
        }

        let root = unsafe { core::slice::from_raw_parts(self.root_paddr as *const u64, 2048) };
        let vpn2 = (gpa >> 30) & 0x7FF;
        let vpn1 = (gpa >> 21) & 0x1FF;
        let l1_paddr = ((root[vpn2] >> 10) << 12) as usize;
        let l1 = unsafe { core::slice::from_raw_parts(l1_paddr as *const u64, 512) };
        let l0_paddr = ((l1[vpn1] >> 10) << 12) as usize;
        let l0 = unsafe { core::slice::from_raw_parts_mut(l0_paddr as *mut u64, 512) };
        l0[vpn0] = 0;
        Ok(((pte >> 10) << 12) as usize)
    }

    /// Remove a 4K mapping and return the SPA that was mapped.
    pub fn unmap_4k(&self, gpa: usize) -> Result<usize, ()> {
        let spa = self.lookup_4k(gpa)?;
        let vpn2 = (gpa >> 30) & 0x7FF;
        let vpn1 = (gpa >> 21) & 0x1FF;
        let vpn0 = (gpa >> 12) & 0x1FF;

        let root = unsafe { core::slice::from_raw_parts(self.root_paddr as *const u64, 2048) };
        let l1_paddr = ((root[vpn2] >> 10) << 12) as usize;
        let l1 = unsafe { core::slice::from_raw_parts(l1_paddr as *const u64, 512) };
        let l0_paddr = ((l1[vpn1] >> 10) << 12) as usize;
        let l0 = unsafe { core::slice::from_raw_parts_mut(l0_paddr as *mut u64, 512) };
        l0[vpn0] = 0;
        Ok(spa)
    }

    pub fn make_hgatp(&self, vmid: usize) -> usize {
        (HGATP_MODE_SV39X4 << 60) | ((vmid & 0x3FFF) << 44) | (self.root_paddr >> 12)
    }
}
