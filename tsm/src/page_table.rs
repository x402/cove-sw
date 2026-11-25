pub const PTE_V: u64 = 1 << 0;
pub const PTE_R: u64 = 1 << 1;
pub const PTE_W: u64 = 1 << 2;
pub const PTE_X: u64 = 1 << 3;
pub const PTE_U: u64 = 1 << 4;
pub const PTE_A: u64 = 1 << 6;
pub const PTE_D: u64 = 1 << 7;

pub const PTE_LEAF_RWX: u64 = PTE_V | PTE_R | PTE_W | PTE_X | PTE_U | PTE_A | PTE_D; // 0xDF
/// Device leaf (COVI vIMSIC): readable/writable, never executable.
pub const PTE_LEAF_RW: u64 = PTE_V | PTE_R | PTE_W | PTE_U | PTE_A | PTE_D; // 0xD7

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

        // Level 0 Leaf entry: refuse to silently overwrite an existing mapping
        if l0_slice[vpn0] & PTE_V != 0 {
            return Err(());
        }
        l0_slice[vpn0] = (((spa >> 12) as u64) << 10) | PTE_LEAF_RWX;

        Ok(())
    }

    /// Maps a 4K device page (COVI vIMSIC): read/write but never
    /// executable, so guest identity values cannot be laundered into
    /// instruction fetches. Same shape as [`map_4k`].
    pub fn map_4k_io(
        &self,
        gpa: usize,
        spa: usize,
        alloc_page: &mut impl FnMut() -> Option<usize>,
    ) -> Result<(), ()> {
        if gpa % 4096 != 0 || spa % 4096 != 0 {
            return Err(());
        }
        if gpa >= (1usize << 41) {
            return Err(());
        }

        let vpn2 = (gpa >> 30) & 0x7FF;
        let vpn1 = (gpa >> 21) & 0x1FF;
        let vpn0 = (gpa >> 12) & 0x1FF;

        let root_slice =
            unsafe { core::slice::from_raw_parts_mut(self.root_paddr as *mut u64, 2048) };

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
        if l0_slice[vpn0] & PTE_V != 0 {
            return Err(());
        }
        l0_slice[vpn0] = (((spa >> 12) as u64) << 10) | PTE_LEAF_RW;

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

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: usize = 4096;
    /// Root table: 2048 entries of 8 bytes.
    const ROOT_SIZE: usize = 16 * 1024;

    /// Backing store for the root table plus the tables an allocator may
    /// hand out.
    #[repr(align(4096))]
    struct PagePool([u8; PAGE * 17]);

    impl PagePool {
        fn new() -> Self {
            Self([0; PAGE * 17])
        }

        fn base(&mut self) -> usize {
            // The table code writes through this address, so the address must
            // come from an exclusive borrow of the backing store.
            self.0.as_mut_ptr() as usize
        }
    }

    /// Deterministic table allocator: hands out pool pages after the root
    /// table and returns `None` once `limit` pages are in use.
    struct PoolAlloc {
        first_paddr: usize,
        handed_out: usize,
        limit: usize,
        calls: usize,
    }

    impl PoolAlloc {
        fn new(base: usize, limit: usize) -> Self {
            Self {
                first_paddr: base + ROOT_SIZE,
                handed_out: 0,
                limit,
                calls: 0,
            }
        }

        fn alloc(&mut self) -> Option<usize> {
            self.calls += 1;
            if self.handed_out >= self.limit {
                return None;
            }
            let paddr = self.first_paddr + self.handed_out * PAGE;
            self.handed_out += 1;
            Some(paddr)
        }
    }

    /// Pool page left untouched by the table allocator.
    fn spare_page(base: usize, index: usize) -> usize {
        base + ROOT_SIZE + index * PAGE
    }

    #[test]
    fn map_lookup_and_unmap_round_trip() {
        let mut pool = PagePool::new();
        let base = pool.base();
        let table = GStagePageTable::new(base);
        let mut alloc = PoolAlloc::new(base, 13);

        let gpa0 = 0x4000_0000;
        let gpa1 = gpa0 + PAGE;
        let gpa2 = 0x8000_0000;
        let spa0 = spare_page(base, 8);
        let spa1 = spare_page(base, 9);
        let spa2 = spare_page(base, 10);

        assert_eq!(table.map_4k(gpa0, spa0, &mut || alloc.alloc()), Ok(()));
        assert_eq!(alloc.calls, 2);
        assert_eq!(table.map_4k(gpa1, spa1, &mut || alloc.alloc()), Ok(()));
        assert_eq!(alloc.calls, 2);
        assert_eq!(table.map_4k(gpa2, spa2, &mut || alloc.alloc()), Ok(()));
        assert_eq!(alloc.calls, 4);

        assert_eq!(table.lookup_4k(gpa0), Ok(spa0));
        assert_eq!(table.lookup_4k(gpa1), Ok(spa1));
        assert_eq!(table.lookup_4k(gpa2), Ok(spa2));

        assert_eq!(table.unmap_4k(gpa1), Ok(spa1));
        assert_eq!(table.lookup_4k(gpa1), Err(()));
        assert_eq!(table.lookup_4k(gpa0), Ok(spa0));
        assert_eq!(table.lookup_4k(gpa2), Ok(spa2));
    }

    #[test]
    fn leaf_bits_match_the_mapping_kind() {
        let mut pool = PagePool::new();
        let base = pool.base();
        let table = GStagePageTable::new(base);
        let mut alloc = PoolAlloc::new(base, 13);
        let spa = spare_page(base, 8);

        let gpa = 0x4000_0000;
        assert_eq!(table.map_4k(gpa, spa, &mut || alloc.alloc()), Ok(()));
        let raw = table.raw_lookup_4k(gpa).unwrap();
        assert_eq!(raw & 0xFF, PTE_LEAF_RWX);
        assert_eq!(raw >> 10, (spa >> 12) as u64);

        let io_gpa = 0x4000_1000;
        assert_eq!(table.map_4k_io(io_gpa, spa, &mut || alloc.alloc()), Ok(()));
        let raw = table.raw_lookup_4k(io_gpa).unwrap();
        assert_eq!(raw & 0xFF, PTE_LEAF_RW);
        assert_eq!(raw & PTE_X, 0);
        assert_eq!(raw >> 10, (spa >> 12) as u64);
    }

    #[test]
    fn bad_addresses_are_refused_before_allocating() {
        let mut pool = PagePool::new();
        let base = pool.base();
        let table = GStagePageTable::new(base);
        let mut alloc = PoolAlloc::new(base, 13);
        let spa = spare_page(base, 8);

        assert_eq!(
            table.map_4k(0x4000_0010, spa, &mut || alloc.alloc()),
            Err(())
        );
        assert_eq!(
            table.map_4k(0x4000_0000, spa + 0x10, &mut || alloc.alloc()),
            Err(())
        );
        assert_eq!(table.map_4k(1 << 41, spa, &mut || alloc.alloc()), Err(()));
        assert_eq!(alloc.calls, 0);

        assert_eq!(
            table.map_4k((1 << 41) - PAGE, spa, &mut || alloc.alloc()),
            Ok(())
        );
    }

    #[test]
    fn duplicate_leaf_is_refused() {
        let mut pool = PagePool::new();
        let base = pool.base();
        let table = GStagePageTable::new(base);
        let mut alloc = PoolAlloc::new(base, 13);
        let gpa = 0x4000_0000;
        let spa0 = spare_page(base, 8);
        let spa1 = spare_page(base, 9);

        assert_eq!(table.map_4k(gpa, spa0, &mut || alloc.alloc()), Ok(()));
        assert_eq!(table.map_4k(gpa, spa1, &mut || alloc.alloc()), Err(()));
        assert_eq!(table.lookup_4k(gpa), Ok(spa0));
    }

    #[test]
    fn allocator_exhaustion_fails_the_mapping() {
        let mut pool = PagePool::new();
        let base = pool.base();
        let table = GStagePageTable::new(base);
        let mut alloc = PoolAlloc::new(base, 0);
        let gpa = 0x4000_0000;
        let spa = spare_page(base, 8);

        assert_eq!(table.map_4k(gpa, spa, &mut || alloc.alloc()), Err(()));
        assert_eq!(alloc.calls, 1);
        assert_eq!(table.lookup_4k(gpa), Err(()));
    }

    #[test]
    fn invalidate_validate_and_remove_retain_the_spa() {
        let mut pool = PagePool::new();
        let base = pool.base();
        let table = GStagePageTable::new(base);
        let mut alloc = PoolAlloc::new(base, 13);
        let gpa = 0x4000_0000;
        let spa = spare_page(base, 8);

        assert_eq!(table.map_4k(gpa, spa, &mut || alloc.alloc()), Ok(()));

        assert_eq!(table.invalidate_4k(gpa), Ok(()));
        assert_eq!(table.lookup_4k(gpa), Err(()));
        assert_eq!(table.invalidate_4k(gpa), Err(()));

        assert_eq!(table.validate_4k(gpa), Ok(()));
        assert_eq!(table.lookup_4k(gpa), Ok(spa));
        assert_eq!(table.validate_4k(gpa), Err(()));

        assert_eq!(table.invalidate_4k(gpa), Ok(()));
        assert_eq!(table.remove_invalid_4k(gpa), Ok(spa));
        assert_eq!(table.raw_lookup_4k(gpa), Ok(0));
        assert_eq!(table.lookup_4k(gpa), Err(()));
    }

    #[test]
    fn removal_without_a_mapping_fails() {
        let mut pool = PagePool::new();
        let base = pool.base();
        let table = GStagePageTable::new(base);

        assert_eq!(table.remove_invalid_4k(0x4000_0000), Err(()));
        assert_eq!(table.unmap_4k(0x4000_0000), Err(()));
    }

    #[test]
    fn hgatp_encodes_mode_vmid_and_root() {
        let mut pool = PagePool::new();
        let base = pool.base();
        let table = GStagePageTable::new(base);

        assert_eq!(
            table.make_hgatp(5),
            (8 << 60) | (5 << 44) | (base >> 12)
        );
    }
}
