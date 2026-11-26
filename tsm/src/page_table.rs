pub const PTE_V: u64 = 1 << 0;
pub const PTE_R: u64 = 1 << 1;
pub const PTE_W: u64 = 1 << 2;
pub const PTE_X: u64 = 1 << 3;
pub const PTE_U: u64 = 1 << 4;
pub const PTE_A: u64 = 1 << 6;
pub const PTE_D: u64 = 1 << 7;

pub const PTE_LEAF_RWX: u64 = PTE_V | PTE_R | PTE_W | PTE_X | PTE_U | PTE_A | PTE_D; // 0xDF
/// Non-executable leaf: readable/writable, never executable. Used for device
/// pages (COVI vIMSIC) and for untrusted shared pages, which the TVM must
/// never fetch code from (CoVE §5.2.3).
pub const PTE_LEAF_RW: u64 = PTE_V | PTE_R | PTE_W | PTE_U | PTE_A | PTE_D; // 0xD7

pub const HGATP_MODE_BARE: usize = 0;
pub const HGATP_MODE_SV39X4: usize = 8;
pub const HGATP_MODE_SV48X4: usize = 9;

/// A TVM's page-table pages in hand-out order. `used` counts the pages
/// already handed out; the pool advances it and nothing returns a page.
pub struct PtPool<'a> {
    pages: &'a [usize],
    used: &'a mut usize,
}

impl<'a> PtPool<'a> {
    pub fn new(pages: &'a [usize], used: &'a mut usize) -> Self {
        Self { pages, used }
    }

    /// Pages not handed out yet.
    pub fn remaining(&self) -> usize {
        self.pages.len().saturating_sub(*self.used)
    }

    /// Hands out the next page. The caller has checked [`Self::remaining`].
    fn take(&mut self) -> usize {
        let paddr = self.pages[*self.used];
        *self.used += 1;
        paddr
    }
}

/// Whether a level-0 entry is a present mapping.
pub fn leaf_is_present(pte: u64) -> bool {
    pte & PTE_V != 0
}

/// Physical address a leaf entry maps.
pub fn leaf_pa(pte: u64) -> usize {
    ((pte >> 10) << 12) as usize
}

/// Whether a level-0 entry is a mapping retained by `invalidate_4k`:
/// nonzero with the valid bit clear.
pub fn leaf_is_retained(pte: u64) -> bool {
    pte != 0 && pte & PTE_V == 0
}

/// Non-leaf entry pointing at a lower-level table, always valid.
fn table_pte(paddr: usize) -> u64 {
    (((paddr >> 12) as u64) << 10) | PTE_V
}

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

    /// Raw level-0 entry of the 4 KiB page containing `gpa`, or `None` when
    /// no last-level table exists for it. Any address inside the page selects
    /// the same entry; a `gpa` outside the Sv39x4 aperture yields `None`.
    pub fn leaf_4k(&self, gpa: usize) -> Option<u64> {
        if gpa >= (1usize << 41) {
            return None;
        }
        let vpn2 = (gpa >> 30) & 0x7FF;
        let vpn1 = (gpa >> 21) & 0x1FF;
        let vpn0 = (gpa >> 12) & 0x1FF;

        let root = unsafe { core::slice::from_raw_parts(self.root_paddr as *const u64, 2048) };
        if root[vpn2] & PTE_V == 0 {
            return None;
        }
        let l1 = unsafe { core::slice::from_raw_parts(leaf_pa(root[vpn2]) as *const u64, 512) };
        if l1[vpn1] & PTE_V == 0 {
            return None;
        }
        let l0 = unsafe { core::slice::from_raw_parts(leaf_pa(l1[vpn1]) as *const u64, 512) };
        Some(l0[vpn0])
    }

    /// Number of page-table pages (0, 1 or 2) a 4 KiB leaf at `gpa` needs.
    /// `Err` for a malformed `gpa` or an occupied level-0 entry (present or
    /// retained). Changes nothing.
    pub fn plan_4k(&self, gpa: usize) -> Result<usize, ()> {
        if gpa % 4096 != 0 || gpa >= (1usize << 41) {
            return Err(());
        }
        let vpn2 = (gpa >> 30) & 0x7FF;
        let vpn1 = (gpa >> 21) & 0x1FF;
        let vpn0 = (gpa >> 12) & 0x1FF;

        let root = unsafe { core::slice::from_raw_parts(self.root_paddr as *const u64, 2048) };
        if root[vpn2] & PTE_V == 0 {
            return Ok(2);
        }
        let l1 = unsafe { core::slice::from_raw_parts(leaf_pa(root[vpn2]) as *const u64, 512) };
        if l1[vpn1] & PTE_V == 0 {
            return Ok(1);
        }
        let l0 = unsafe { core::slice::from_raw_parts(leaf_pa(l1[vpn1]) as *const u64, 512) };
        if l0[vpn0] != 0 {
            return Err(());
        }
        Ok(0)
    }

    /// Number of page-table pages a contiguous run of `num_pages` 4 KiB
    /// leaves starting at `gpa` needs, counting every missing table once.
    /// `Err` for a malformed or overflowing range or an occupied leaf.
    /// Changes nothing.
    pub fn plan_range_4k(&self, gpa: usize, num_pages: usize) -> Result<usize, ()> {
        if gpa % 4096 != 0 {
            return Err(());
        }
        let len = num_pages.checked_mul(4096).ok_or(())?;
        let end = gpa.checked_add(len).ok_or(())?;
        if end > (1usize << 41) {
            return Err(());
        }

        let root = unsafe { core::slice::from_raw_parts(self.root_paddr as *const u64, 2048) };
        let mut needed = 0;
        for i in 0..num_pages {
            let page = gpa + i * 4096;
            let vpn2 = (page >> 30) & 0x7FF;
            let vpn1 = (page >> 21) & 0x1FF;
            let vpn0 = (page >> 12) & 0x1FF;
            // Table keys repeat across the pages of one span; only the first
            // page of each span may count a new table.
            let first_of_2mib = i == 0 || (page - 4096) >> 21 != page >> 21;
            let first_of_1gib = i == 0 || (page - 4096) >> 30 != page >> 30;

            if root[vpn2] & PTE_V == 0 {
                if first_of_1gib {
                    needed += 1;
                }
                // Every 2 MiB span below a new level-1 table needs its own
                // level-0 table.
                if first_of_2mib {
                    needed += 1;
                }
                continue;
            }
            let l1 = unsafe { core::slice::from_raw_parts(leaf_pa(root[vpn2]) as *const u64, 512) };
            if l1[vpn1] & PTE_V == 0 {
                if first_of_2mib {
                    needed += 1;
                }
                continue;
            }
            let l0 = unsafe { core::slice::from_raw_parts(leaf_pa(l1[vpn1]) as *const u64, 512) };
            if l0[vpn0] != 0 {
                return Err(());
            }
        }
        Ok(needed)
    }

    /// Maps `gpa` to `spa` with a 4 KiB RWX leaf. `Err` (malformed address,
    /// occupied leaf, pool too small) leaves tables and pool untouched.
    pub fn map_4k(&self, gpa: usize, spa: usize, pool: &mut PtPool) -> Result<(), ()> {
        self.map_4k_with(gpa, spa, PTE_LEAF_RWX, pool)
    }

    /// Same with an RW, never executable leaf: device pages and untrusted
    /// shared pages, which must never be fetchable (CoVE §5.2.3).
    pub fn map_4k_nx(&self, gpa: usize, spa: usize, pool: &mut PtPool) -> Result<(), ()> {
        self.map_4k_with(gpa, spa, PTE_LEAF_RW, pool)
    }

    /// Whether any leaf of this table maps a page inside
    /// `[pa_base, pa_base + len)`. A retained entry counts: it is a mapping
    /// the host can re-validate, so its page still belongs to a guest
    /// address. An overflowing end reports `false`. Changes nothing.
    pub fn maps_any(&self, pa_base: usize, len: usize) -> bool {
        let end = match pa_base.checked_add(len) {
            Some(end) => end,
            None => return false,
        };
        if pa_base >= end {
            return false;
        }
        let overlaps = |start: usize, size: usize| match start.checked_add(size) {
            Some(stop) => start < end && pa_base < stop,
            None => false,
        };

        let root = unsafe { core::slice::from_raw_parts(self.root_paddr as *const u64, 2048) };
        for entry in root {
            if entry & PTE_V == 0 {
                continue;
            }
            let l1 = unsafe { core::slice::from_raw_parts(leaf_pa(*entry) as *const u64, 512) };
            for entry in l1 {
                if entry & PTE_V == 0 {
                    continue;
                }
                if entry & (PTE_R | PTE_W | PTE_X) != 0 {
                    // A level-1 superpage leaf maps 2 MiB.
                    if overlaps(leaf_pa(*entry), 1 << 21) {
                        return true;
                    }
                    continue;
                }
                let l0 = unsafe { core::slice::from_raw_parts(leaf_pa(*entry) as *const u64, 512) };
                for entry in l0 {
                    // Present or retained: only an empty entry maps nothing.
                    if *entry == 0 {
                        continue;
                    }
                    if overlaps(leaf_pa(*entry), 4096) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Two-phase mapping: [`Self::plan_4k`] decides whether the walk fits in
    /// the pool, and only then are tables linked and the leaf written. No
    /// step after the first write can fail, so a refused mapping leaves the
    /// table and the pool exactly as they were.
    fn map_4k_with(
        &self,
        gpa: usize,
        spa: usize,
        flags: u64,
        pool: &mut PtPool,
    ) -> Result<(), ()> {
        if spa % 4096 != 0 {
            return Err(());
        }
        let needed = self.plan_4k(gpa)?;
        if pool.remaining() < needed {
            return Err(());
        }

        let vpn2 = (gpa >> 30) & 0x7FF;
        let vpn1 = (gpa >> 21) & 0x1FF;
        let vpn0 = (gpa >> 12) & 0x1FF;

        unsafe {
            let root = core::slice::from_raw_parts_mut(self.root_paddr as *mut u64, 2048);
            if root[vpn2] & PTE_V == 0 {
                let paddr = pool.take();
                core::ptr::write_bytes(paddr as *mut u8, 0, 4096);
                root[vpn2] = table_pte(paddr);
            }
        }
        let l1_paddr = unsafe {
            let root = core::slice::from_raw_parts(self.root_paddr as *const u64, 2048);
            leaf_pa(root[vpn2])
        };
        unsafe {
            let l1 = core::slice::from_raw_parts_mut(l1_paddr as *mut u64, 512);
            if l1[vpn1] & PTE_V == 0 {
                let paddr = pool.take();
                core::ptr::write_bytes(paddr as *mut u8, 0, 4096);
                l1[vpn1] = table_pte(paddr);
            }
        }
        let l0_paddr = unsafe {
            let l1 = core::slice::from_raw_parts(l1_paddr as *const u64, 512);
            leaf_pa(l1[vpn1])
        };
        unsafe {
            let l0 = core::slice::from_raw_parts_mut(l0_paddr as *mut u64, 512);
            l0[vpn0] = (((spa >> 12) as u64) << 10) | flags;
        }
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
    /// Table pages the backing store can hand out after the root table.
    const TABLES: usize = 13;

    /// Backing store for the root table plus the tables a pool may hand
    /// out: four root pages and [`TABLES`] table pages.
    #[repr(align(4096))]
    struct PageMemory([u8; PAGE * (4 + TABLES)]);

    impl PageMemory {
        fn new() -> Self {
            Self([0; PAGE * (4 + TABLES)])
        }

        /// The table code writes through this address, so the address must
        /// come from an exclusive borrow of the backing store.
        fn base(&mut self) -> usize {
            self.0.as_mut_ptr() as usize
        }
    }

    /// Table pages after the root table, in hand-out order.
    fn table_pages(base: usize) -> [usize; TABLES] {
        let mut pages = [0; TABLES];
        for (i, page) in pages.iter_mut().enumerate() {
            *page = base + ROOT_SIZE + i * PAGE;
        }
        pages
    }

    /// Fake SPA values: mapping only records them, so they need not be RAM.
    const SPA_BASE: usize = 0x9000_0000;

    #[test]
    fn map_lookup_and_unmap_round_trip() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);
        let pages = table_pages(base);
        let mut used = 0;
        let mut pool = PtPool::new(&pages, &mut used);

        let gpa0 = 0x4000_0000;
        let gpa1 = gpa0 + PAGE;
        let gpa2 = 0x8000_0000;
        let spa0 = SPA_BASE;
        let spa1 = SPA_BASE + PAGE;
        let spa2 = SPA_BASE + 2 * PAGE;

        assert_eq!(table.map_4k(gpa0, spa0, &mut pool), Ok(()));
        assert_eq!(pool.remaining(), TABLES - 2);
        assert_eq!(table.map_4k(gpa1, spa1, &mut pool), Ok(()));
        assert_eq!(pool.remaining(), TABLES - 2);
        assert_eq!(table.map_4k(gpa2, spa2, &mut pool), Ok(()));
        assert_eq!(pool.remaining(), TABLES - 4);

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
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);
        let pages = table_pages(base);
        let mut used = 0;
        let mut pool = PtPool::new(&pages, &mut used);
        let spa = SPA_BASE;

        let gpa = 0x4000_0000;
        assert_eq!(table.map_4k(gpa, spa, &mut pool), Ok(()));
        let raw = table.leaf_4k(gpa).unwrap();
        assert_eq!(raw & 0xFF, PTE_LEAF_RWX);
        assert!(leaf_is_present(raw));
        assert!(!leaf_is_retained(raw));
        assert_eq!(leaf_pa(raw), spa);

        let io_gpa = 0x4000_1000;
        assert_eq!(table.map_4k_nx(io_gpa, spa, &mut pool), Ok(()));
        let raw = table.leaf_4k(io_gpa).unwrap();
        assert_eq!(raw & 0xFF, PTE_LEAF_RW);
        assert_eq!(raw & PTE_X, 0);
        assert_eq!(leaf_pa(raw), spa);
    }

    #[test]
    fn maps_any_finds_every_mapped_page_in_the_range() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);
        let pages = table_pages(base);
        let mut used = 0;
        let spa0 = SPA_BASE;
        let spa1 = SPA_BASE + PAGE;

        assert!(!table.maps_any(spa0, PAGE));
        {
            let mut pool = PtPool::new(&pages, &mut used);
            assert_eq!(table.map_4k(0x8001_0000, spa0, &mut pool), Ok(()));
            assert_eq!(table.map_4k(0x8001_1000, spa1, &mut pool), Ok(()));
        }

        // Hits, including ranges that only touch the boundary page.
        assert!(table.maps_any(spa0, PAGE));
        assert!(table.maps_any(spa1, PAGE));
        assert!(table.maps_any(spa0, 2 * PAGE));
        assert!(table.maps_any(spa0 - PAGE, 2 * PAGE));
        assert!(table.maps_any(spa1 - PAGE, 2 * PAGE));
        // Misses just outside on both sides.
        assert!(!table.maps_any(spa0 - PAGE, PAGE));
        assert!(!table.maps_any(spa1 + PAGE, PAGE));
        assert!(!table.maps_any(spa1 + PAGE, 8 * PAGE));
        // Empty and overflowing ranges map nothing.
        assert!(!table.maps_any(spa0, 0));
        assert!(!table.maps_any(usize::MAX - PAGE, 2 * PAGE));
        // A range in another span.
        assert!(!table.maps_any(0xA000_0000, 4 * PAGE));
    }

    #[test]
    fn maps_any_scans_every_table_and_keeps_retained_entries() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);
        let pages = table_pages(base);
        let mut used = 0;

        // Three leaves in three different tables: the second 2 MiB span of
        // one level-1 table, and a second 1 GiB span with its own level-1 and
        // level-0 tables. A walk that stops after the first valid level-1 or
        // level-0 table would miss the later ones.
        let same_l1 = 0x8001_0000;
        let second_2mib = 0x8020_0000;
        let second_1gib = 0xC000_0000;
        let spa_same_l1 = SPA_BASE;
        let spa_second_2mib = SPA_BASE + PAGE;
        let spa_second_1gib = SPA_BASE + 2 * PAGE;
        {
            let mut pool = PtPool::new(&pages, &mut used);
            assert_eq!(table.map_4k(same_l1, spa_same_l1, &mut pool), Ok(()));
            assert_eq!(table.map_4k(second_2mib, spa_second_2mib, &mut pool), Ok(()));
            assert_eq!(table.map_4k(second_1gib, spa_second_1gib, &mut pool), Ok(()));
        }

        assert!(table.maps_any(spa_same_l1, PAGE));
        assert!(table.maps_any(spa_second_2mib, PAGE));
        assert!(table.maps_any(spa_second_1gib, PAGE));
        assert!(!table.maps_any(spa_second_1gib + PAGE, 8 * PAGE));

        // A retained entry still maps its page: the host can re-validate it.
        assert_eq!(table.invalidate_4k(second_2mib), Ok(()));
        let retained = table.leaf_4k(second_2mib).expect("entry exists");
        assert!(!leaf_is_present(retained));
        assert!(leaf_is_retained(retained));
        assert!(table.maps_any(spa_second_2mib, PAGE));
        assert!(table.maps_any(spa_second_2mib, 2 * PAGE));
    }

    #[test]
    fn bad_addresses_are_refused_before_allocating() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);
        let pages = table_pages(base);
        let mut used = 0;
        let mut pool = PtPool::new(&pages, &mut used);

        assert_eq!(table.map_4k(0x4000_0010, SPA_BASE, &mut pool), Err(()));
        assert_eq!(table.map_4k(0x4000_0000, SPA_BASE + 0x10, &mut pool), Err(()));
        assert_eq!(table.map_4k(1 << 41, SPA_BASE, &mut pool), Err(()));
        assert_eq!(pool.remaining(), TABLES);

        assert_eq!(table.map_4k((1 << 41) - PAGE, SPA_BASE, &mut pool), Ok(()));
    }

    #[test]
    fn duplicate_leaf_is_refused() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);
        let pages = table_pages(base);
        let mut used = 0;
        let mut pool = PtPool::new(&pages, &mut used);
        let gpa = 0x4000_0000;

        assert_eq!(table.map_4k(gpa, SPA_BASE, &mut pool), Ok(()));
        assert_eq!(table.map_4k(gpa, SPA_BASE + PAGE, &mut pool), Err(()));
        assert_eq!(table.lookup_4k(gpa), Ok(SPA_BASE));
        assert_eq!(pool.remaining(), TABLES - 2);
    }

    #[test]
    fn a_pool_one_page_short_leaves_the_table_untouched() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);
        let pages = table_pages(base);
        let mut used = 0;
        let gpa = 0x4000_0000;

        // A fresh leaf in a fresh 1 GiB span needs two tables.
        assert_eq!(table.plan_4k(gpa), Ok(2));
        let mut pool = PtPool::new(&pages[..1], &mut used);
        assert_eq!(table.map_4k(gpa, SPA_BASE, &mut pool), Err(()));
        assert_eq!(used, 0);
        assert_eq!(table.leaf_4k(gpa), None);
        assert_eq!(table.plan_4k(gpa), Ok(2));
        assert!(table.lookup_4k(gpa).is_err());
    }

    #[test]
    fn plan_counts_only_the_missing_tables() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);
        let pages = table_pages(base);
        let mut used = 0;

        let gpa = 0x8001_0000;
        assert_eq!(table.plan_4k(gpa), Ok(2));
        {
            let mut pool = PtPool::new(&pages, &mut used);
            assert_eq!(table.map_4k(gpa, SPA_BASE, &mut pool), Ok(()));
        }
        // Same 2 MiB span: the level-0 table already exists.
        assert_eq!(table.plan_4k(gpa + PAGE), Ok(0));
        // New 2 MiB span: only the level-0 table is missing.
        assert_eq!(table.plan_4k(gpa + 0x20_0000), Ok(1));
        // New 1 GiB span: both tables are missing.
        assert_eq!(table.plan_4k(gpa + 0x4000_0000), Ok(2));

        assert_eq!(table.plan_4k(gpa + 8), Err(()));
        assert_eq!(table.plan_4k(1 << 41), Err(()));
    }

    #[test]
    fn plan_refuses_an_occupied_or_retained_leaf() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);
        let pages = table_pages(base);
        let mut used = 0;
        let gpa = 0x8001_0000;

        {
            let mut pool = PtPool::new(&pages, &mut used);
            assert_eq!(table.map_4k(gpa, SPA_BASE, &mut pool), Ok(()));
        }
        assert_eq!(table.plan_4k(gpa), Err(()));

        assert_eq!(table.invalidate_4k(gpa), Ok(()));
        assert_eq!(table.plan_4k(gpa), Err(()));
    }

    #[test]
    fn the_pool_advances_by_exactly_the_planned_pages() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);
        let pages = table_pages(base);
        let mut used = 0;

        let gpa = 0x8001_0000;
        {
            let mut pool = PtPool::new(&pages, &mut used);
            assert_eq!(table.plan_4k(gpa), Ok(2));
            assert_eq!(table.map_4k(gpa, SPA_BASE, &mut pool), Ok(()));
            assert_eq!(pool.remaining(), TABLES - 2);

            assert_eq!(table.plan_4k(gpa + PAGE), Ok(0));
            assert_eq!(table.map_4k(gpa + PAGE, SPA_BASE + PAGE, &mut pool), Ok(()));
            assert_eq!(pool.remaining(), TABLES - 2);

            assert_eq!(table.plan_4k(gpa + 0x20_0000), Ok(1));
            assert_eq!(
                table.map_4k(gpa + 0x20_0000, SPA_BASE + 2 * PAGE, &mut pool),
                Ok(())
            );
            assert_eq!(pool.remaining(), TABLES - 3);
        }
        assert_eq!(used, 3);
        assert_eq!(table.lookup_4k(gpa + 0x20_0000), Ok(SPA_BASE + 2 * PAGE));
    }

    #[test]
    fn leaf_inspection_around_invalidate_validate_and_remove() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);
        let pages = table_pages(base);
        let mut used = 0;
        let gpa = 0x4000_0000;
        let spa = SPA_BASE;

        {
            let mut pool = PtPool::new(&pages, &mut used);
            assert_eq!(table.map_4k(gpa, spa, &mut pool), Ok(()));
        }

        // No last-level table at all.
        assert_eq!(table.leaf_4k(0x8000_0000), None);
        assert_eq!(table.leaf_4k(1 << 41), None);
        // Any address inside the page selects the same entry.
        assert_eq!(table.leaf_4k(gpa + 0x10), Some(table.leaf_4k(gpa).unwrap()));

        let raw = table.leaf_4k(gpa).unwrap();
        assert!(leaf_is_present(raw));
        assert_eq!(leaf_pa(raw), spa);

        assert_eq!(table.invalidate_4k(gpa), Ok(()));
        let raw = table.leaf_4k(gpa).unwrap();
        assert!(!leaf_is_present(raw));
        assert!(leaf_is_retained(raw));
        assert_eq!(leaf_pa(raw), spa);

        assert_eq!(table.validate_4k(gpa), Ok(()));
        assert!(leaf_is_present(table.leaf_4k(gpa).unwrap()));

        assert_eq!(table.invalidate_4k(gpa), Ok(()));
        assert_eq!(table.remove_invalid_4k(gpa), Ok(spa));
        let raw = table.leaf_4k(gpa).unwrap();
        assert_eq!(raw, 0);
        assert!(!leaf_is_present(raw));
        assert!(!leaf_is_retained(raw));
    }

    #[test]
    fn plan_range_counts_each_missing_table_once() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);
        let pages = table_pages(base);
        let mut used = 0;

        // Four pages of a fresh 1 GiB span: one level-1 table and one
        // level-0 table.
        assert_eq!(table.plan_range_4k(0x8001_0000, 4), Ok(2));
        // Two pages straddling a 2 MiB boundary inside that span.
        assert_eq!(table.plan_range_4k(0x801F_F000, 2), Ok(3));
        // An empty range needs nothing.
        assert_eq!(table.plan_range_4k(0x8001_0000, 0), Ok(0));
        // Malformed and overflowing ranges are refused.
        assert_eq!(table.plan_range_4k(0x8001_0008, 1), Err(()));
        assert_eq!(table.plan_range_4k((1 << 41) - PAGE, 2), Err(()));
        assert_eq!(table.plan_range_4k(0x8001_0000, usize::MAX), Err(()));

        {
            let mut pool = PtPool::new(&pages, &mut used);
            assert_eq!(table.map_4k(0x8001_0000, SPA_BASE, &mut pool), Ok(()));
        }
        // The mapped span already has both tables, so an empty page in it
        // costs nothing.
        assert_eq!(table.plan_range_4k(0x8001_1000, 1), Ok(0));
        // A new 2 MiB span below the existing level-1 table costs one page.
        assert_eq!(table.plan_range_4k(0x8020_0000, 1), Ok(1));
        // An occupied leaf makes the whole range unplannable.
        assert_eq!(table.plan_range_4k(0x8001_0000, 2), Err(()));
    }

    #[test]
    fn plan_range_across_a_1gib_boundary_counts_two_level1_tables() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);

        // 0xBFFF_F000..0xC000_3000 spans the 0xBFFF_FFFF / 0xC000_0000
        // boundary: each side needs its own level-1 and level-0 table, so
        // the run costs four pages, not the two a single span would cost.
        for page in [0xBFFF_F000, 0xC000_0000, 0xC000_1000, 0xC000_2000] {
            assert_eq!(table.plan_4k(page), Ok(2));
        }
        assert_eq!(table.plan_range_4k(0xBFFF_F000, 4), Ok(4));
    }

    #[test]
    fn a_whole_range_is_reserved_before_the_first_mapping() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);
        let pages = table_pages(base);
        let mut used = 0;
        let mut pool = PtPool::new(&pages, &mut used);

        let gpa0 = 0x8001_0000;
        let gpa1 = gpa0 + PAGE;
        let needed = table.plan_range_4k(gpa0, 2).unwrap();
        assert_eq!(needed, 2);
        assert!(pool.remaining() >= needed);

        assert_eq!(table.map_4k(gpa0, SPA_BASE, &mut pool), Ok(()));
        assert_eq!(table.map_4k(gpa1, SPA_BASE + PAGE, &mut pool), Ok(()));
        assert_eq!(pool.remaining(), TABLES - needed);
        assert_eq!(used, needed);
        assert_eq!(table.lookup_4k(gpa0), Ok(SPA_BASE));
        assert_eq!(table.lookup_4k(gpa1), Ok(SPA_BASE + PAGE));
    }

    #[test]
    fn removal_without_a_mapping_fails() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);

        assert_eq!(table.remove_invalid_4k(0x4000_0000), Err(()));
        assert_eq!(table.unmap_4k(0x4000_0000), Err(()));
    }

    #[test]
    fn hgatp_encodes_mode_vmid_and_root() {
        let mut mem = PageMemory::new();
        let base = mem.base();
        let table = GStagePageTable::new(base);

        assert_eq!(
            table.make_hgatp(5),
            (8 << 60) | (5 << 44) | (base >> 12)
        );
    }
}
