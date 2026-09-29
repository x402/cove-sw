//! MPT radix tree management for Smmpt43/52/64 modes.
//!
//! This module provides the [`MptTree`] structure for managing Machine-level
//! Memory Protection Tables. It supports:
//!
//! - **MptPerm** — 3-bit XWR permission encoding (X=2, W=1, R=0).
//! - **MptPageAlloc** — Trait for allocating/freeing 4 KiB pages by PPN.
//! - **MptTree** — Radix-tree construction, permission setting, querying,
//!   and recursive destruction.
//!
//! All raw pointer accesses use `core::ptr::read_volatile` / `write_volatile`
//! to model MMIO-style table manipulation.
#![cfg_attr(test, allow(unused_extern_crates))]
#[cfg(test)]
extern crate std;

use crate::csr::MptMode;

// ── MptPerm: 3-bit XWR permission encoding ─────────────────────────────

/// XWR permission encoding for MPT leaf entries.
///
/// Encoded as 3 bits: X=bit2, W=bit1, R=bit0.
///
/// | Value | Name | Permissions |
/// |-------|------|-------------|
/// | 0b000 | NONE | No access   |
/// | 0b001 | R    | Read        |
/// | 0b011 | RW   | Read-Write  |
/// | 0b100 | X    | Execute     |
/// | 0b101 | RX   | Read-Execute|
/// | 0b111 | RWX  | Read-Write-Execute |
///
/// Reserved encodings (0b010, 0b110) return `None` from [`from_bits`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MptPerm(u8);

#[allow(non_upper_case_globals)]
impl MptPerm {
    /// No access.
    pub const NONE: Self = Self(0b000);
    /// Read only.
    pub const R: Self = Self(0b001);
    /// Read-Write.
    pub const RW: Self = Self(0b011);
    /// Execute only.
    pub const X: Self = Self(0b100);
    /// Read-Execute.
    pub const RX: Self = Self(0b101);
    /// Read-Write-Execute.
    pub const RWX: Self = Self(0b111);

    /// Construct an `MptPerm` from raw 3-bit XWR encoding.
    ///
    /// Returns `None` for reserved encodings (`0b010`, `0b110`).
    pub const fn from_bits(bits: u8) -> Option<Self> {
        match bits {
            0b000 | 0b001 | 0b011 | 0b100 | 0b101 | 0b111 => Some(Self(bits)),
            _ => None,
        }
    }

    /// Return the raw 3-bit XWR value.
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Whether the Read bit is set.
    pub const fn is_readable(self) -> bool {
        self.0 & 0b001 != 0
    }

    /// Whether the Write bit is set.
    pub const fn is_writable(self) -> bool {
        self.0 & 0b010 != 0
    }

    /// Whether the eXecute bit is set.
    pub const fn is_executable(self) -> bool {
        self.0 & 0b100 != 0
    }
}

// ── MPTE bit-layout constants ──────────────────────────────────────────

/// Valid bit position (bit 0).
const V_BIT: u64 = 1;

/// Leaf bit position (bit 1). 0 = non-leaf, 1 = leaf.
const L_BIT: u64 = 1 << 1;

/// NAPOT bit position (bit 2). 0 = non-NAPOT, 1 = NAPOT (currently unused).
#[allow(dead_code)]
const N_BIT: u64 = 1 << 2;

// ── Private MPTE helper functions ──────────────────────────────────────

/// Encode a non-leaf MPTE from a child page table PPN.
fn encode_non_leaf(ppn: usize) -> u64 {
    (ppn as u64) << 10 | V_BIT
}

/// Decode the child PPN from a non-leaf MPTE.
fn decode_non_leaf_ppn(entry: u64) -> usize {
    (entry >> 10) as usize
}

/// Check whether an MPTE is valid (V bit set).
fn is_valid(entry: u64) -> bool {
    (entry & V_BIT) != 0
}

/// Check whether an MPTE is a leaf entry (L bit set).
fn is_leaf(entry: u64) -> bool {
    (entry & L_BIT) != 0
}

/// Set the 3-bit XWR field for `page_index` (0..16) within a leaf MPTE.
///
/// The XWR[i] field occupies bits `[10 + i*3 : 8 + i*3]`.
fn set_leaf_xwr(entry: u64, page_index: usize, xwr: u8) -> u64 {
    let shift = 8 + page_index * 3;
    let mask = 0b111u64 << shift;
    (entry & !mask) | ((xwr as u64 & 0b111) << shift)
}

/// Extract the 3-bit XWR field for `page_index` (0..16) from a leaf MPTE.
fn get_leaf_xwr(entry: u64, page_index: usize) -> u8 {
    let shift = 8 + page_index * 3;
    ((entry >> shift) & 0b111) as u8
}

// ── Level / page-index extraction ──────────────────────────────────────

/// Extract the table index for a given level from a physical address.
///
/// The range offset is 16 bits (bits [15:0]). Each level above level 0 uses
/// 9 bits, except the Smmpt64 root which uses 12 bits.
fn level_index(mode: MptMode, pa: usize, level: usize) -> usize {
    let shift = 16 + 9 * level;
    let mask = if level == mode.levels() - 1 && mode == MptMode::Smmpt64 {
        0xFFF
    } else {
        0x1FF
    };
    (pa >> shift) & mask
}

/// Extract the page index within a leaf (4 bits, 16 pages per leaf).
fn page_index_in_leaf(pa: usize) -> usize {
    (pa >> 12) & 0xF
}

// ── MptPageAlloc trait ─────────────────────────────────────────────────

/// Trait for allocating and freeing 4 KiB pages by physical page number.
///
/// `alloc_page` returns a PPN (physical page number, not a byte address)
/// of a zeroed or caller-initialized 4 KiB page. For bump-allocator
/// implementations, `free_page` may be a no-op.
pub trait MptPageAlloc {
    /// Allocate a 4 KiB page and return its PPN, or `None` if exhausted.
    fn alloc_page(&mut self) -> Option<usize>;

    /// Free a previously allocated page by PPN.
    fn free_page(&mut self, ppn: usize);
}

// ── MptTree ────────────────────────────────────────────────────────────

/// A Machine-level Memory Protection Table (MPT) radix tree.
///
/// Manages the page-table structure for Smmpt43/52/64 modes. Supports
/// permission setting on individual pages or ranges, permission queries,
/// and recursive destruction.
pub struct MptTree {
    mode: MptMode,
    root_ppn: usize,
}

impl MptTree {
    /// Create a new MPT tree for the given mode.
    ///
    /// For Bare mode, returns a tree with `root_ppn = 0` and no allocation.
    /// For other modes, allocates a root page and zeros all entries.
    /// Returns `None` if the root page allocation fails.
    pub fn new(mode: MptMode, alloc: &mut impl MptPageAlloc) -> Option<Self> {
        if mode.is_bare() {
            return Some(Self { mode, root_ppn: 0 });
        }

        // The root table may span multiple contiguous 4 KiB pages
        // (e.g. Smmpt64 uses 8 pages = 32 KiB).  Call alloc_page()
        // repeatedly and verify that the returned PPNs are contiguous.
        // If the allocator cannot provide contiguous pages, return None.
        let root_pages = mode.root_pages();
        let mut root_ppn = 0;
        for i in 0..root_pages {
            let ppn = alloc.alloc_page()?;
            if i == 0 {
                root_ppn = ppn;
            } else if ppn != root_ppn + i {
                // Non-contiguous — allocator cannot support multi-page root.
                return None;
            }
        }

        let root_addr = (root_ppn << 12) as *mut u8;
        // Zero-initialise all root entries.
        unsafe {
            core::ptr::write_bytes(root_addr, 0, mode.root_size());
        }

        Some(Self { mode, root_ppn })
    }

    /// Create an `MptTree` from an existing mode and root PPN without allocation.
    pub const fn from_root(mode: MptMode, root_ppn: usize) -> Self {
        Self { mode, root_ppn }
    }

    /// Return the root PPN of this tree.
    pub const fn root_ppn(&self) -> usize {
        self.root_ppn
    }

    /// Return the addressing mode of this tree.
    pub const fn mode(&self) -> MptMode {
        self.mode
    }

    /// Set permissions on a range of physical addresses.
    ///
    /// Iterates over each 4 KiB page in `[pa, pa + len)` and calls
    /// [`set_perm_single`](Self::set_perm_single) for each page. This is a
    /// no-op for Bare mode or when `len == 0`.
    pub fn set_perm(
        &mut self,
        pa: usize,
        len: usize,
        perm: MptPerm,
        alloc: &mut impl MptPageAlloc,
    ) {
        if self.mode.is_bare() || len == 0 {
            return;
        }

        // Use saturating arithmetic to handle pa + len overflow.
        // If pa + len overflows usize, saturating_add wraps to usize::MAX,
        // and the loop covers all pages from `pa` to the end of the
        // address space.  In practice, callers should pass valid ranges.
        let start_page = pa >> 12;
        let end_page = pa.saturating_add(len).saturating_sub(1) >> 12;

        // Defensive: if start_page > end_page (e.g. pa is at the very end
        // of the address space), skip to avoid a panic on range construction.
        if start_page > end_page {
            return;
        }

        let mut page = start_page;
        while page <= end_page {
            if page.is_multiple_of(8192) && page + 8191 <= end_page {
                self.set_perm_table0(page << 12, perm, alloc);
                page += 8192;
            } else if page.is_multiple_of(16) && page + 15 <= end_page {
                self.set_perm_16pages(page << 12, perm, alloc);
                page += 16;
            } else {
                self.set_perm_single(page << 12, perm, alloc);
                page += 1;
            }
        }
    }

    /// Set permission for an entire 32 MiB range (one complete Level 0 table, 512 leaf entries).
    fn set_perm_table0(&mut self, pa: usize, perm: MptPerm, alloc: &mut impl MptPageAlloc) {
        let levels = self.mode.levels();
        let mut current_ppn = self.root_ppn;

        // Walk from root down to level 1.
        for level in (1..levels).rev() {
            let idx = level_index(self.mode, pa, level);
            let addr = (current_ppn << 12) + idx * 8;
            let entry = unsafe { core::ptr::read_volatile(addr as *const u64) };

            if !is_valid(entry) {
                if let Some(new_ppn) = alloc.alloc_page() {
                    unsafe {
                        core::ptr::write_bytes((new_ppn << 12) as *mut u8, 0, 4096);
                    }
                    let new_entry = encode_non_leaf(new_ppn);
                    unsafe {
                        core::ptr::write_volatile(addr as *mut u64, new_entry);
                    }
                    current_ppn = new_ppn;
                } else {
                    return;
                }
            } else if !is_leaf(entry) {
                current_ppn = decode_non_leaf_ppn(entry);
            } else {
                return;
            }
        }

        // At level 1, `current_ppn` is the level 0 table (or we just created it).
        let mut full_leaf = V_BIT | L_BIT;
        let p = perm.to_bits();
        for pi in 0..16 {
            full_leaf = set_leaf_xwr(full_leaf, pi, p);
        }

        for idx in 0..512 {
            let leaf_addr = (current_ppn << 12) + idx * 8;
            unsafe {
                core::ptr::write_volatile(leaf_addr as *mut u64, full_leaf);
            }
        }
    }

    /// Set permission for a 64 KiB block (16 contiguous 4 KiB pages, one leaf MPTE).
    fn set_perm_16pages(&mut self, pa: usize, perm: MptPerm, alloc: &mut impl MptPageAlloc) {
        let levels = self.mode.levels();
        let mut current_ppn = self.root_ppn;

        // Walk from root down to level 1.
        for level in (1..levels).rev() {
            let idx = level_index(self.mode, pa, level);
            let addr = (current_ppn << 12) + idx * 8;
            let entry = unsafe { core::ptr::read_volatile(addr as *const u64) };

            if !is_valid(entry) {
                if let Some(new_ppn) = alloc.alloc_page() {
                    unsafe {
                        core::ptr::write_bytes((new_ppn << 12) as *mut u8, 0, 4096);
                    }
                    let new_entry = encode_non_leaf(new_ppn);
                    unsafe {
                        core::ptr::write_volatile(addr as *mut u64, new_entry);
                    }
                    current_ppn = new_ppn;
                } else {
                    return;
                }
            } else if !is_leaf(entry) {
                current_ppn = decode_non_leaf_ppn(entry);
            } else {
                return;
            }
        }

        // Level 0: leaf entry.
        let leaf_idx = level_index(self.mode, pa, 0);
        let leaf_addr = (current_ppn << 12) + leaf_idx * 8;
        let mut full_leaf = V_BIT | L_BIT;
        let p = perm.to_bits();
        for pi in 0..16 {
            full_leaf = set_leaf_xwr(full_leaf, pi, p);
        }
        unsafe {
            core::ptr::write_volatile(leaf_addr as *mut u64, full_leaf);
        }
    }

    /// Set permission for a single 4 KiB page at `pa`.
    ///
    /// Walks the radix tree from the root, allocating intermediate tables as
    /// needed. At the leaf level, sets the XWR field for the page within the
    /// leaf entry. If a higher-level leaf already covers this address, the
    /// operation is silently skipped (current limitation).
    fn set_perm_single(&mut self, pa: usize, perm: MptPerm, alloc: &mut impl MptPageAlloc) {
        let levels = self.mode.levels();
        let mut current_ppn = self.root_ppn;

        // Walk from root down to level 1.
        for level in (1..levels).rev() {
            let idx = level_index(self.mode, pa, level);
            let addr = (current_ppn << 12) + idx * 8;
            let entry = unsafe { core::ptr::read_volatile(addr as *const u64) };

            if !is_valid(entry) {
                // Allocate a new intermediate page table.
                if let Some(new_ppn) = alloc.alloc_page() {
                    unsafe {
                        core::ptr::write_bytes((new_ppn << 12) as *mut u8, 0, 4096);
                    }
                    let new_entry = encode_non_leaf(new_ppn);
                    unsafe {
                        core::ptr::write_volatile(addr as *mut u64, new_entry);
                    }
                    current_ppn = new_ppn;
                } else {
                    return; // Allocation failure — skip.
                }
            } else if !is_leaf(entry) {
                current_ppn = decode_non_leaf_ppn(entry);
            } else {
                return; // Already a leaf at a higher level — skip.
            }
        }

        // Level 0: leaf entry.
        let leaf_idx = level_index(self.mode, pa, 0);
        let leaf_addr = (current_ppn << 12) + leaf_idx * 8;
        let mut entry = unsafe { core::ptr::read_volatile(leaf_addr as *const u64) };

        if !is_valid(entry) {
            // Initialise a new leaf with V=1, L=1, N=0, all XWR=0.
            entry = V_BIT | L_BIT;
        }

        let pi = page_index_in_leaf(pa);
        entry = set_leaf_xwr(entry, pi, perm.to_bits());
        unsafe {
            core::ptr::write_volatile(leaf_addr as *mut u64, entry);
        }
    }

    /// Query the permission for a physical address.
    ///
    /// Returns `Some(MptPerm)` if a leaf entry covers `pa`, or `None` if
    /// the address is unmapped or the mode is Bare.
    pub fn get_perm(&self, pa: usize) -> Option<MptPerm> {
        if self.mode.is_bare() {
            return None;
        }

        let levels = self.mode.levels();
        let mut current_ppn = self.root_ppn;

        for level in (0..levels).rev() {
            let idx = level_index(self.mode, pa, level);
            let addr = (current_ppn << 12) + idx * 8;
            let entry = unsafe { core::ptr::read_volatile(addr as *const u64) };

            if !is_valid(entry) {
                return None;
            }

            if is_leaf(entry) {
                let pi = page_index_in_leaf(pa);
                return MptPerm::from_bits(get_leaf_xwr(entry, pi));
            }

            // Non-leaf — follow child pointer.
            current_ppn = decode_non_leaf_ppn(entry);
        }

        None
    }

    /// Destroy the tree, recursively freeing all allocated pages.
    ///
    /// Frees all intermediate page tables and the root page. A no-op for
    /// Bare mode.
    pub fn destroy(self, alloc: &mut impl MptPageAlloc) {
        if self.mode.is_bare() {
            return;
        }

        let levels = self.mode.levels();
        self.destroy_level(self.root_ppn, levels - 1, alloc);
        // Free all root pages.  Smmpt64 has 8 contiguous root pages;
        // Smmpt43/52 have 1.  The root pages are guaranteed contiguous
        // by `MptTree::new`.
        for i in 0..self.mode.root_pages() {
            alloc.free_page(self.root_ppn + i);
        }
    }

    /// Recursively free all child pages at and below `level` in the tree
    /// rooted at `ppn`.
    fn destroy_level(&self, ppn: usize, level: usize, alloc: &mut impl MptPageAlloc) {
        if level == 0 {
            return; // Leaf level — no child page tables.
        }

        let entries = if level == self.mode.levels() - 1 {
            self.mode.root_entries()
        } else {
            512
        };

        for i in 0..entries {
            let addr = (ppn << 12) + i * 8;
            let entry = unsafe { core::ptr::read_volatile(addr as *const u64) };

            if is_valid(entry) && !is_leaf(entry) {
                let child_ppn = decode_non_leaf_ppn(entry);
                self.destroy_level(child_ppn, level - 1, alloc);
                alloc.free_page(child_ppn);
            }
        }
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    /// Mock allocator that carves pages from a heap-allocated buffer.
    ///
    /// Returns real memory addresses so that the volatile pointer accesses
    /// in MptTree work correctly on the host (x86).
    struct MockAlloc {
        buffer: Vec<u8>,
        next_offset: usize,
        freed_count: u32,
    }

    impl MockAlloc {
        /// Create a new allocator backed by a page-aligned 1 MiB buffer.
        fn new() -> Self {
            let mut buffer = Vec::new();
            // Allocate 1 MiB + 4 KiB extra so we can align `next_offset` to a
            // page boundary within the buffer.
            buffer.resize(1024 * 1024 + 4096, 0u8);
            let base = buffer.as_ptr() as usize;
            let next_offset = ((base + 0xFFF) & !0xFFF) - base;
            MockAlloc {
                buffer,
                next_offset,
                freed_count: 0,
            }
        }
    }

    impl MptPageAlloc for MockAlloc {
        fn alloc_page(&mut self) -> Option<usize> {
            if self.next_offset + 4096 > self.buffer.len() {
                return None;
            }
            let addr = &self.buffer[self.next_offset] as *const u8 as usize;
            self.next_offset += 4096;
            Some(addr >> 12) // Return PPN (byte address >> 12)
        }

        fn free_page(&mut self, _ppn: usize) {
            self.freed_count += 1;
        }
    }

    // ── Task 4.8: MptTree tests ────────────────────────────────────────

    #[test]
    fn create_tree_and_root_ppn() {
        let mut alloc = MockAlloc::new();
        let tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        assert_ne!(tree.root_ppn(), 0, "root PPN must be non-zero");
    }

    #[test]
    fn set_get_perm_roundtrip() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        let pa = 0x100_0000; // Some address in the middle of address space
        tree.set_perm(pa, 4096, MptPerm::RWX, &mut alloc);
        assert_eq!(tree.get_perm(pa), Some(MptPerm::RWX));
    }

    #[test]
    fn unset_perm_returns_none() {
        let mut alloc = MockAlloc::new();
        let tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        // Address not yet mapped.
        assert_eq!(tree.get_perm(0x200_0000), None);
    }

    #[test]
    fn set_perm_range() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        let base = 0x100_0000;
        // Set RX on a 3-page range (12 KiB).
        tree.set_perm(base, 0x3000, MptPerm::RX, &mut alloc);

        assert_eq!(tree.get_perm(base), Some(MptPerm::RX));
        assert_eq!(tree.get_perm(base + 0x1000), Some(MptPerm::RX));
        assert_eq!(tree.get_perm(base + 0x2000), Some(MptPerm::RX));
        // Same leaf, unset page (page_index_in_leaf=3 vs set 0-2).
        // Leaf entry is valid (V=1) but XWR[3]=0 → Some(NONE), not None.
        assert_eq!(tree.get_perm(base + 0x3000), Some(MptPerm::NONE));
        // Adjacent leaf (level_index(0)=257 vs 256) — leaf entry V=0 → None.
        assert_eq!(tree.get_perm(base + 0x1_0000), None);
    }

    #[test]
    fn destroy_frees_pages() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();

        // Set a permission to trigger intermediate page allocation.
        tree.set_perm(0x100_0000, 4096, MptPerm::R, &mut alloc);
        // Initial free count is 0.
        assert_eq!(alloc.freed_count, 0);

        tree.destroy(&mut alloc);
        // Smmpt43 (3 levels): root + 1 intermediate + 1 leaf = 3 pages.
        assert_eq!(alloc.freed_count, 3, "destroy must free all 3 pages");
    }

    #[test]
    fn bare_mode() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Bare, &mut alloc).unwrap();

        assert_eq!(tree.root_ppn(), 0);

        // set_perm should be a no-op in Bare mode.
        tree.set_perm(0x1000, 4096, MptPerm::RWX, &mut alloc);
        assert_eq!(tree.get_perm(0x1000), None);

        // destroy should be a no-op.
        tree.destroy(&mut alloc);
    }

    #[test]
    fn smmpt64_root_allocation() {
        let mut alloc = MockAlloc::new();
        let tree = MptTree::new(MptMode::Smmpt64, &mut alloc).unwrap();
        assert_ne!(tree.root_ppn(), 0, "Smmpt64 root PPN must be non-zero");
    }

    #[test]
    fn smmpt64_root_pages_contiguous() {
        let mut alloc = MockAlloc::new();
        let tree = MptTree::new(MptMode::Smmpt64, &mut alloc).unwrap();
        // Smmpt64 root spans 8 pages. The next allocation should start
        // after the root, not within it.
        let next_ppn = alloc.alloc_page().unwrap();
        assert!(
            next_ppn >= tree.root_ppn() + 8,
            "next page {:#x} overlaps root [{:#x}, {:#x})",
            next_ppn,
            tree.root_ppn(),
            tree.root_ppn() + 8
        );
    }

    #[test]
    fn smmpt64_root_zeroed() {
        let mut alloc = MockAlloc::new();
        let tree = MptTree::new(MptMode::Smmpt64, &mut alloc).unwrap();
        let root_addr = (tree.root_ppn() << 12) as *const u8;
        // All 8 root pages (32 KiB) should be zeroed after creation.
        for i in 0..(8 * 4096) {
            unsafe {
                assert_eq!(*root_addr.add(i), 0, "root byte {} not zeroed", i);
            }
        }
    }

    #[test]
    fn smmpt64_set_perm_does_not_corrupt_root() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt64, &mut alloc).unwrap();
        // Set a permission — this allocates intermediate pages after the root.
        tree.set_perm(0x100_0000, 4096, MptPerm::RWX, &mut alloc);
        // Verify the permission is retrievable (root table not corrupted).
        assert_eq!(tree.get_perm(0x100_0000), Some(MptPerm::RWX));
    }

    #[test]
    fn smmpt64_destroy_frees_all_root_pages() {
        let mut alloc = MockAlloc::new();
        let tree = MptTree::new(MptMode::Smmpt64, &mut alloc).unwrap();
        // Smmpt64 root spans 8 pages.  No intermediate pages allocated.
        assert_eq!(alloc.freed_count, 0);

        tree.destroy(&mut alloc);
        // All 8 root pages must be freed.
        assert_eq!(
            alloc.freed_count, 8,
            "destroy must free all 8 Smmpt64 root pages"
        );
    }

    #[test]
    fn set_perm_overwrite() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        let pa = 0x100_0000;

        // Set RWX, then overwrite with R.
        tree.set_perm(pa, 4096, MptPerm::RWX, &mut alloc);
        assert_eq!(tree.get_perm(pa), Some(MptPerm::RWX));
        tree.set_perm(pa, 4096, MptPerm::R, &mut alloc);
        assert_eq!(tree.get_perm(pa), Some(MptPerm::R));

        // Other pages in the same leaf are valid but have XWR=0 (NONE).
        assert_eq!(tree.get_perm(pa + 0x1000), Some(MptPerm::NONE));
    }

    #[test]
    fn smmpt52_destroy_frees_all_pages() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt52, &mut alloc).unwrap();
        tree.set_perm(0x100_0000, 4096, MptPerm::RWX, &mut alloc);
        // Verify the permission was set correctly.
        assert_eq!(tree.get_perm(0x100_0000), Some(MptPerm::RWX));
        assert_eq!(alloc.freed_count, 0);

        tree.destroy(&mut alloc);
        // Smmpt52 (4 levels): root + 2 intermediate + 1 leaf = 4 pages.
        assert_eq!(
            alloc.freed_count, 4,
            "destroy must free all 4 Smmpt52 pages"
        );
    }

    #[test]
    fn test_dual_mpt_cove_setup() {
        let mut alloc = MockAlloc::new();
        let mut conf_tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        let mut host_tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();

        let ram_start: usize = 0x8000_0000;
        let tsm_load_paddr: usize = 0x8040_0000;
        let host_load_paddr: usize = 0x8080_0000;

        // MPT_CONF: Low memory + full RAM RWX
        conf_tree.set_perm(0x1000, 0x1000, MptPerm::RWX, &mut alloc);
        conf_tree.set_perm(ram_start, 0x1000, MptPerm::RWX, &mut alloc);
        conf_tree.set_perm(tsm_load_paddr, 0x1000, MptPerm::RWX, &mut alloc);
        conf_tree.set_perm(host_load_paddr, 0x1000, MptPerm::RWX, &mut alloc);

        // MPT_HOST: Low memory RWX, RAM before TSM RWX, TSM NONE, Host RAM RWX
        host_tree.set_perm(0x1000, 0x1000, MptPerm::RWX, &mut alloc);
        host_tree.set_perm(ram_start, 0x1000, MptPerm::RWX, &mut alloc);
        host_tree.set_perm(tsm_load_paddr, 0x1000, MptPerm::NONE, &mut alloc);
        host_tree.set_perm(host_load_paddr, 0x1000, MptPerm::RWX, &mut alloc);

        // Verify MPT_CONF permissions
        assert_eq!(conf_tree.get_perm(0x1000), Some(MptPerm::RWX));
        assert_eq!(conf_tree.get_perm(ram_start), Some(MptPerm::RWX));
        assert_eq!(conf_tree.get_perm(tsm_load_paddr), Some(MptPerm::RWX));
        assert_eq!(conf_tree.get_perm(host_load_paddr), Some(MptPerm::RWX));

        // Verify MPT_HOST permissions
        assert_eq!(host_tree.get_perm(0x1000), Some(MptPerm::RWX));
        assert_eq!(host_tree.get_perm(ram_start), Some(MptPerm::RWX));
        assert_eq!(host_tree.get_perm(tsm_load_paddr), Some(MptPerm::NONE));
        assert_eq!(host_tree.get_perm(host_load_paddr), Some(MptPerm::RWX));
    }
}
