//! MPT radix tree management for Smmpt43/52/64 modes.
//!
//! This module provides the [`MptTree`] structure for managing Machine-level
//! Memory Protection Tables. It supports:
//!
//! - **MptPerm** — 3-bit XWR permission encoding (X=2, W=1, R=0).
//! - **MptError** — why a permission update could not be applied.
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

// ── MptError ───────────────────────────────────────────────────────────

/// Why an MPT permission update could not be applied. Nothing is changed
/// when one is returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MptError {
    /// The range is not a whole number of 4 KiB pages, or it lies partly
    /// above the addresses the mode covers.
    InvalidRange,
    /// The page pool cannot supply the intermediate tables the update
    /// needs.
    AllocationFailed,
    /// A valid leaf above level 0 covers part of the range. This
    /// implementation neither splits nor replaces such an entry.
    UnsupportedLeaf { level: usize },
    /// An entry has a shape this implementation does not define: a pointer
    /// entry or a leaf entry with a bit the specification reserves, a pointer
    /// at level 0, or the NAPOT flag on a leaf.
    MalformedEntry { level: usize },
}

// ── MPTE bit-layout constants ──────────────────────────────────────────

/// Valid bit position (bit 0).
const V_BIT: u64 = 1;

/// Leaf bit position (bit 1). 0 = non-leaf, 1 = leaf.
const L_BIT: u64 = 1 << 1;

/// NAPOT bit position (bit 2). 0 = non-NAPOT, 1 = NAPOT (currently unused).
const N_BIT: u64 = 1 << 2;

/// Bits SmmTT v0.49 section 4.2 (Figure 10) reserves in a pointer entry:
/// 63:54 above the PPN and 9:2 between the PPN and the L bit. The hardware
/// faults on every access an entry with one of them set covers.
const RESERVED_POINTER_BITS: u64 = (!0u64 << 54) | 0x3FC;

/// Bits the same section (Figure 11) reserves in a non-NAPOT leaf entry:
/// 63:56 above the XWR fields and 7:3 below them.
const RESERVED_LEAF_BITS: u64 = (!0u64 << 56) | 0xF8;

/// Number of 4 KiB pages covered by one leaf entry.
const LEAF_PAGES: usize = 16;

/// Number of entries in a level-0 table (and of leaves it covers).
const LEVEL0_ENTRIES: usize = 512;

/// Number of pages covered by one complete level-0 table.
const TABLE0_PAGES: usize = LEVEL0_ENTRIES * LEAF_PAGES;

/// Highest level count among the Smmpt modes (Smmpt64).
const MAX_LEVELS: usize = 5;

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

/// Classify a valid MPTE read from a table above level 0.
///
/// Returns the child page-table PPN of a pointer entry. A leaf entry above
/// level 0, or a pointer carrying a bit the specification reserves, is
/// rejected.
fn classify_pointer_entry(entry: u64, level: usize) -> Result<usize, MptError> {
    debug_assert!(is_valid(entry), "classify_pointer_entry on an absent entry");
    if is_leaf(entry) {
        return Err(MptError::UnsupportedLeaf { level });
    }
    if entry & RESERVED_POINTER_BITS != 0 {
        return Err(MptError::MalformedEntry { level });
    }
    Ok(decode_non_leaf_ppn(entry))
}

/// Validate one level-0 entry the update is about to overwrite.
///
/// An absent entry is fine (the update creates it); a valid entry must be a
/// non-NAPOT leaf with no reserved bit set.
fn validate_leaf_entry(entry: u64) -> Result<(), MptError> {
    if !is_valid(entry) {
        return Ok(());
    }
    if !is_leaf(entry) || entry & (N_BIT | RESERVED_LEAF_BITS) != 0 {
        return Err(MptError::MalformedEntry { level: 0 });
    }
    Ok(())
}

/// Build a complete leaf entry: valid, leaf, non-NAPOT, with `perm` in the
/// XWR field of every page.
fn full_leaf(perm: MptPerm) -> u64 {
    let mut entry = V_BIT | L_BIT;
    for page_index in 0..LEAF_PAGES {
        entry = set_leaf_xwr(entry, page_index, perm.to_bits());
    }
    entry
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

    /// Whether the next `count` calls to [`Self::alloc_page`] can succeed.
    fn can_allocate(&self, count: usize) -> bool;
}

// ── Permission update chunks ───────────────────────────────────────────

/// One unit of a permission update: a whole level-0 table (512 leaves,
/// 32 MiB), one leaf (16 pages, 64 KiB), or one page (one XWR field inside
/// a leaf entry).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chunk {
    /// The complete level-0 table starting at the given address.
    Table0(usize),
    /// The 16 pages of one leaf entry, starting at the given address.
    Leaf16(usize),
    /// One page at the given address.
    Single(usize),
}

impl Chunk {
    /// Physical address at the start of this chunk.
    const fn pa(self) -> usize {
        match self {
            Self::Table0(pa) | Self::Leaf16(pa) | Self::Single(pa) => pa,
        }
    }

    /// Number of level-0 entries this chunk overwrites.
    const fn level0_entries(self) -> usize {
        match self {
            // Every entry of the level-0 table.
            Self::Table0(_) => LEVEL0_ENTRIES,
            // The leaf's own entry; the single page shares it.
            Self::Leaf16(_) | Self::Single(_) => 1,
        }
    }
}

/// Yields the chunks covering pages `start_page..=end_page` in increasing
/// page order.
struct ChunkIter {
    page: usize,
    end_page: usize,
}

impl ChunkIter {
    /// The caller guarantees `start_page <= end_page`.
    const fn new(start_page: usize, end_page: usize) -> Self {
        Self {
            page: start_page,
            end_page,
        }
    }
}

impl Iterator for ChunkIter {
    type Item = Chunk;

    fn next(&mut self) -> Option<Chunk> {
        let page = self.page;
        if page > self.end_page {
            return None;
        }
        let chunk = if page.is_multiple_of(TABLE0_PAGES) && page + TABLE0_PAGES - 1 <= self.end_page
        {
            self.page += TABLE0_PAGES;
            Chunk::Table0(page << 12)
        } else if page.is_multiple_of(LEAF_PAGES) && page + LEAF_PAGES - 1 <= self.end_page {
            self.page += LEAF_PAGES;
            Chunk::Leaf16(page << 12)
        } else {
            self.page += 1;
            Chunk::Single(page << 12)
        };
        Some(chunk)
    }
}

// ── Permission update planning ─────────────────────────────────────────

/// Read-only plan of the page tables a permission update must create.
///
/// The plan walks the update's chunks in increasing address order and
/// remembers, per radix level, the entry slot of the last walked path and
/// whether the table behind it is one this update creates. A chunk whose
/// path passes through such a table again therefore counts it once: the
/// second pass allocates the table for the earlier chunk and finds it
/// present when it reaches the later one.
struct AllocPlan {
    /// Entry index of the last walked path, per level `1..MAX_LEVELS`.
    index: [usize; MAX_LEVELS],
    /// Whether the table at that entry slot is created by this update
    /// rather than pre-existing in memory.
    created: [bool; MAX_LEVELS],
    /// Intermediate tables the update still has to allocate.
    missing: usize,
}

impl AllocPlan {
    const fn new() -> Self {
        Self {
            index: [0; MAX_LEVELS],
            created: [false; MAX_LEVELS],
            missing: 0,
        }
    }

    /// Validate the path of `chunk` and the level-0 entries it overwrites,
    /// adding every table the update must create for it to [`Self::missing`].
    fn check(&mut self, mode: MptMode, root_ppn: usize, chunk: Chunk) -> Result<(), MptError> {
        let pa = chunk.pa();
        let levels = mode.levels();

        // `Some(ppn)` while the walk is inside a table that exists in
        // memory, `None` once it is below a table this update creates: such
        // a table is zero-filled, so the rest of the path is absent.
        let mut parent = Some(root_ppn);
        // Whether the path so far is the one the previous chunk walked.
        let mut shared = true;

        for level in (1..levels).rev() {
            let idx = level_index(mode, pa, level);
            let entry = match parent {
                Some(ppn) => unsafe {
                    core::ptr::read_volatile(((ppn << 12) + idx * 8) as *const u64)
                },
                None => 0,
            };
            // A table recorded for this exact entry slot is created by this
            // update, so the second pass finds it present.
            let recorded = shared && self.created[level] && self.index[level] == idx;

            if is_valid(entry) {
                parent = Some(classify_pointer_entry(entry, level)?);
                self.created[level] = false;
            } else if recorded {
                parent = None;
            } else {
                // This level and every level below it needs a new table.
                self.plan_new(mode, pa, level);
                return Ok(());
            }

            shared = shared && self.index[level] == idx;
            self.index[level] = idx;
        }

        // The level-0 table either exists in memory or is being created
        // (its entries are then all absent).
        let Some(table_ppn) = parent else {
            return Ok(());
        };
        let first = level_index(mode, pa, 0);
        for idx in first..first + chunk.level0_entries() {
            let entry =
                unsafe { core::ptr::read_volatile(((table_ppn << 12) + idx * 8) as *const u64) };
            validate_leaf_entry(entry)?;
        }
        Ok(())
    }

    /// Record that the table at `level` and every table below it is created
    /// by this update, and count them.
    fn plan_new(&mut self, mode: MptMode, pa: usize, level: usize) {
        for lower in 1..=level {
            self.index[lower] = level_index(mode, pa, lower);
            self.created[lower] = true;
        }
        self.missing += level;
    }
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
    /// The update is all-or-nothing: a read-only pass validates the whole
    /// range and plans the intermediate tables it needs before anything is
    /// written, so an unsupported layout, a malformed entry or an exhausted
    /// page pool leaves every entry untouched. A no-op for Bare mode or
    /// when `len == 0`.
    ///
    /// # Errors
    ///
    /// - [`MptError::InvalidRange`] — `pa` or `len` is not 4 KiB aligned, or
    ///   the range lies partly above the addresses the mode covers.
    /// - [`MptError::UnsupportedLeaf`] — a valid leaf above level 0 covers
    ///   part of the range.
    /// - [`MptError::MalformedEntry`] — an entry on the walk, or a level-0
    ///   entry the update overwrites, has a shape this implementation does
    ///   not define.
    /// - [`MptError::AllocationFailed`] — `alloc` cannot supply every table
    ///   the update needs.
    pub fn set_perm(
        &mut self,
        pa: usize,
        len: usize,
        perm: MptPerm,
        alloc: &mut impl MptPageAlloc,
    ) -> Result<(), MptError> {
        if self.mode.is_bare() || len == 0 {
            return Ok(());
        }
        if !pa.is_multiple_of(4096) || !len.is_multiple_of(4096) {
            return Err(MptError::InvalidRange);
        }
        // An address above the mode's width never reaches the tree: the
        // hardware faults on it, and the walk ignores its high bits, so the
        // update would land on a lower address.
        let width = self.mode.address_bits();
        if width < usize::BITS as usize && pa.saturating_add(len) > 1 << width {
            return Err(MptError::InvalidRange);
        }

        // Use saturating arithmetic to handle pa + len overflow.
        // If pa + len overflows usize, saturating_add wraps to usize::MAX,
        // and the loop covers all pages from `pa` to the end of the
        // address space.  In practice, callers should pass valid ranges.
        let start_page = pa >> 12;
        let end_page = pa.saturating_add(len).saturating_sub(1) >> 12;

        // Pass one: validate the whole range and count the tables it needs
        // without writing anything.
        let mut plan = AllocPlan::new();
        for chunk in ChunkIter::new(start_page, end_page) {
            plan.check(self.mode, self.root_ppn, chunk)?;
        }
        if !alloc.can_allocate(plan.missing) {
            return Err(MptError::AllocationFailed);
        }

        // Pass two: apply. Pass one counted every table this walk
        // allocates, so no step here can fail.
        for chunk in ChunkIter::new(start_page, end_page) {
            self.apply_chunk(chunk, perm, alloc);
        }
        Ok(())
    }

    /// Apply `perm` to one chunk, allocating the intermediate tables the
    /// planning pass counted.
    fn apply_chunk(&mut self, chunk: Chunk, perm: MptPerm, alloc: &mut impl MptPageAlloc) {
        let pa = chunk.pa();
        let levels = self.mode.levels();
        let mut current_ppn = self.root_ppn;

        // Walk from the root down to the level-0 table.
        for level in (1..levels).rev() {
            let idx = level_index(self.mode, pa, level);
            let addr = (current_ppn << 12) + idx * 8;
            let entry = unsafe { core::ptr::read_volatile(addr as *const u64) };

            current_ppn = if is_valid(entry) {
                // The planning pass validated the shape of every entry on
                // this path.
                decode_non_leaf_ppn(entry)
            } else {
                let new_ppn = alloc
                    .alloc_page()
                    .expect("MPT page pool exhausted after the update's tables were counted");
                unsafe {
                    core::ptr::write_bytes((new_ppn << 12) as *mut u8, 0, 4096);
                    core::ptr::write_volatile(addr as *mut u64, encode_non_leaf(new_ppn));
                }
                new_ppn
            };
        }

        match chunk {
            // One complete level-0 table: all its leaves are rewritten.
            Chunk::Table0(_) => {
                let leaf = full_leaf(perm);
                for idx in 0..LEVEL0_ENTRIES {
                    let leaf_addr = (current_ppn << 12) + idx * 8;
                    unsafe {
                        core::ptr::write_volatile(leaf_addr as *mut u64, leaf);
                    }
                }
            }
            // One leaf: all 16 page permissions in a single entry.
            Chunk::Leaf16(_) => {
                let leaf_idx = level_index(self.mode, pa, 0);
                let leaf_addr = (current_ppn << 12) + leaf_idx * 8;
                unsafe {
                    core::ptr::write_volatile(leaf_addr as *mut u64, full_leaf(perm));
                }
            }
            // One page: only its own XWR field inside the leaf entry.
            Chunk::Single(_) => {
                let leaf_idx = level_index(self.mode, pa, 0);
                let leaf_addr = (current_ppn << 12) + leaf_idx * 8;
                let entry = unsafe { core::ptr::read_volatile(leaf_addr as *const u64) };

                // Initialise a new leaf with V=1, L=1, N=0, all XWR=0.
                let entry = if is_valid(entry) {
                    entry
                } else {
                    V_BIT | L_BIT
                };
                let entry = set_leaf_xwr(entry, page_index_in_leaf(pa), perm.to_bits());
                unsafe {
                    core::ptr::write_volatile(leaf_addr as *mut u64, entry);
                }
            }
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
    use std::collections::BTreeSet;
    use std::vec::Vec;

    /// Mock allocator that carves pages from a heap-allocated buffer.
    ///
    /// Returns real memory addresses so that the volatile pointer accesses
    /// in MptTree work correctly on the host (x86).
    struct MockAlloc {
        buffer: Vec<u8>,
        next_offset: usize,
        /// Maximum number of pages this allocator hands out.
        capacity: usize,
        /// Pages handed out so far.
        allocated: usize,
        freed_count: u32,
    }

    impl MockAlloc {
        /// Create a new allocator backed by a page-aligned 1 MiB buffer.
        fn new() -> Self {
            Self::with_capacity(BUFFER_PAGES)
        }

        /// Create an allocator that refuses to hand out more than
        /// `capacity` pages, whatever the backing buffer could hold.
        fn with_capacity(capacity: usize) -> Self {
            let mut buffer = Vec::new();
            // Allocate 1 MiB + 4 KiB extra so we can align `next_offset` to a
            // page boundary within the buffer.
            buffer.resize(1024 * 1024 + 4096, 0u8);
            let base = buffer.as_ptr() as usize;
            let next_offset = ((base + 0xFFF) & !0xFFF) - base;
            MockAlloc {
                buffer,
                next_offset,
                capacity,
                allocated: 0,
                freed_count: 0,
            }
        }
    }

    /// Pages the 1 MiB backing buffer can hold.
    const BUFFER_PAGES: usize = 1024 * 1024 / 4096;

    impl MptPageAlloc for MockAlloc {
        fn alloc_page(&mut self) -> Option<usize> {
            if self.allocated >= self.capacity || self.next_offset + 4096 > self.buffer.len() {
                return None;
            }
            let addr = &self.buffer[self.next_offset] as *const u8 as usize;
            self.next_offset += 4096;
            self.allocated += 1;
            Some(addr >> 12) // Return PPN (byte address >> 12)
        }

        fn free_page(&mut self, _ppn: usize) {
            self.freed_count += 1;
        }

        fn can_allocate(&self, count: usize) -> bool {
            self.allocated + count <= self.capacity
        }
    }

    /// Read one entry of the table at `table_ppn`.
    fn read_entry(table_ppn: usize, idx: usize) -> u64 {
        unsafe { core::ptr::read_volatile(((table_ppn << 12) + idx * 8) as *const u64) }
    }

    /// Write one entry of the table at `table_ppn`.
    fn write_entry(table_ppn: usize, idx: usize, entry: u64) {
        unsafe {
            core::ptr::write_volatile(((table_ppn << 12) + idx * 8) as *mut u64, entry);
        }
    }

    /// Walk the tree down to the level-0 table that covers `pa`.
    fn level0_table(tree: &MptTree, pa: usize) -> usize {
        let mut ppn = tree.root_ppn();
        for level in (1..tree.mode().levels()).rev() {
            ppn = decode_non_leaf_ppn(read_entry(ppn, level_index(tree.mode(), pa, level)));
        }
        ppn
    }

    /// Allocate a zeroed page and return its PPN.
    fn fresh_table(alloc: &mut MockAlloc) -> usize {
        let ppn = alloc.alloc_page().unwrap();
        unsafe {
            core::ptr::write_bytes((ppn << 12) as *mut u8, 0, 4096);
        }
        ppn
    }

    // ── MptTree tests ──────────────────────────────────────────────────

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
        tree.set_perm(pa, 4096, MptPerm::RWX, &mut alloc)
            .expect("permission update");
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
        tree.set_perm(base, 0x3000, MptPerm::RX, &mut alloc)
            .expect("permission update");

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
        tree.set_perm(0x100_0000, 4096, MptPerm::R, &mut alloc)
            .expect("permission update");
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
        tree.set_perm(0x1000, 4096, MptPerm::RWX, &mut alloc)
            .expect("permission update");
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
        tree.set_perm(0x100_0000, 4096, MptPerm::RWX, &mut alloc)
            .expect("permission update");
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
        tree.set_perm(pa, 4096, MptPerm::RWX, &mut alloc)
            .expect("permission update");
        assert_eq!(tree.get_perm(pa), Some(MptPerm::RWX));
        tree.set_perm(pa, 4096, MptPerm::R, &mut alloc)
            .expect("permission update");
        assert_eq!(tree.get_perm(pa), Some(MptPerm::R));

        // Other pages in the same leaf are valid but have XWR=0 (NONE).
        assert_eq!(tree.get_perm(pa + 0x1000), Some(MptPerm::NONE));
    }

    #[test]
    fn smmpt52_destroy_frees_all_pages() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt52, &mut alloc).unwrap();
        tree.set_perm(0x100_0000, 4096, MptPerm::RWX, &mut alloc)
            .expect("permission update");
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
        conf_tree
            .set_perm(0x1000, 0x1000, MptPerm::RWX, &mut alloc)
            .expect("permission update");
        conf_tree
            .set_perm(ram_start, 0x1000, MptPerm::RWX, &mut alloc)
            .expect("permission update");
        conf_tree
            .set_perm(tsm_load_paddr, 0x1000, MptPerm::RWX, &mut alloc)
            .expect("permission update");
        conf_tree
            .set_perm(host_load_paddr, 0x1000, MptPerm::RWX, &mut alloc)
            .expect("permission update");

        // MPT_HOST: Low memory RWX, RAM before TSM RWX, TSM NONE, Host RAM RWX
        host_tree
            .set_perm(0x1000, 0x1000, MptPerm::RWX, &mut alloc)
            .expect("permission update");
        host_tree
            .set_perm(ram_start, 0x1000, MptPerm::RWX, &mut alloc)
            .expect("permission update");
        host_tree
            .set_perm(tsm_load_paddr, 0x1000, MptPerm::NONE, &mut alloc)
            .expect("permission update");
        host_tree
            .set_perm(host_load_paddr, 0x1000, MptPerm::RWX, &mut alloc)
            .expect("permission update");

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

    // ── All-or-nothing updates ─────────────────────────────────────────

    /// A three-table range (96 MiB) whose page tables must all be created.
    const THREE_TABLES_LEN: usize = 3 * 0x200_0000;

    /// Run an update whose plan needs one page more than the pool holds,
    /// and check that nothing was allocated, written or made readable.
    fn assert_allocation_failure_is_atomic(mode: MptMode, capacity: usize) {
        let mut alloc = MockAlloc::with_capacity(capacity);
        let mut tree = MptTree::new(mode, &mut alloc).unwrap();
        assert_eq!(alloc.allocated, mode.root_pages(), "root allocation");

        let cursor = alloc.next_offset;
        let root = tree.root_ppn();
        assert_eq!(
            tree.set_perm(0, THREE_TABLES_LEN, MptPerm::RWX, &mut alloc),
            Err(MptError::AllocationFailed)
        );

        // The allocator cursor and page count are untouched.
        assert_eq!(alloc.allocated, mode.root_pages());
        assert_eq!(alloc.next_offset, cursor, "allocator cursor moved");

        // Every page of the range is still unmapped.
        let mut pa = 0;
        while pa < THREE_TABLES_LEN {
            assert_eq!(tree.get_perm(pa), None, "page {pa:#x} became mapped");
            pa += 0x1000;
        }

        // The root table itself was not written.
        for idx in 0..mode.root_entries() {
            assert_eq!(
                read_entry(root, idx),
                0,
                "root entry {idx} was modified by a failed update"
            );
        }
    }

    #[test]
    fn allocation_failure_is_atomic_smmpt43() {
        // Root + one level-1 table + three level-0 tables = 4 pages; the
        // pool offers three after the root.
        assert_allocation_failure_is_atomic(MptMode::Smmpt43, 1 + 3);
    }

    #[test]
    fn allocation_failure_is_atomic_smmpt52() {
        // Root + two intermediate levels + three level-0 tables = 5 pages;
        // the pool offers four after the root.
        assert_allocation_failure_is_atomic(MptMode::Smmpt52, 1 + 4);
    }

    #[test]
    fn allocation_failure_is_atomic_smmpt64() {
        // An 8-page root + three intermediate levels + three level-0
        // tables = 14 pages; the pool offers five after the root.
        assert_allocation_failure_is_atomic(MptMode::Smmpt64, 8 + 5);
    }

    #[test]
    fn update_after_a_failed_one_still_succeeds() {
        let mut alloc = MockAlloc::with_capacity(1 + 3);
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        assert_eq!(
            tree.set_perm(0, THREE_TABLES_LEN, MptPerm::RWX, &mut alloc),
            Err(MptError::AllocationFailed)
        );
        // Root + one level-1 table + one level-0 table = 3 pages, which the
        // pool can still supply.
        assert_eq!(tree.set_perm(0, 4096, MptPerm::RWX, &mut alloc), Ok(()));
        assert_eq!(tree.get_perm(0), Some(MptPerm::RWX));
        assert_eq!(alloc.allocated, 3);
    }

    #[test]
    fn plan_counts_a_shared_table_once() {
        // Smmpt52, two pages in the same leaf, nothing below the root
        // exists yet: the plan needs three tables (one per level), and both
        // chunks share all of them.
        let mut alloc = MockAlloc::with_capacity(1 + 3);
        let mut tree = MptTree::new(MptMode::Smmpt52, &mut alloc).unwrap();
        let base = 0x100_0000;
        assert_eq!(
            tree.set_perm(base, 0x2000, MptPerm::RWX, &mut alloc),
            Ok(())
        );
        assert_eq!(tree.get_perm(base), Some(MptPerm::RWX));
        assert_eq!(tree.get_perm(base + 0x1000), Some(MptPerm::RWX));
        assert_eq!(alloc.allocated, 4, "root + three intermediate tables");
    }

    #[test]
    fn multi_table_update_uses_exactly_the_planned_tables() {
        let mut alloc = MockAlloc::with_capacity(1 + 4);
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        assert_eq!(
            tree.set_perm(0, THREE_TABLES_LEN, MptPerm::RWX, &mut alloc),
            Ok(())
        );
        // Root + one level-1 table + three level-0 tables.
        assert_eq!(alloc.allocated, 5);
        for pa in [
            0,
            0x1FF_F000,
            0x200_0000,
            0x400_0000,
            THREE_TABLES_LEN - 0x1000,
        ] {
            assert_eq!(tree.get_perm(pa), Some(MptPerm::RWX), "page {pa:#x}");
        }
    }

    #[test]
    fn zero_length_range_is_ok() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        let allocated = alloc.allocated;
        assert_eq!(tree.set_perm(0x1000, 0, MptPerm::RWX, &mut alloc), Ok(()));
        assert_eq!(alloc.allocated, allocated);
        assert_eq!(tree.get_perm(0x1000), None);
    }

    #[test]
    fn misaligned_range_is_rejected() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        assert_eq!(
            tree.set_perm(0x1001, 0x1000, MptPerm::RWX, &mut alloc),
            Err(MptError::InvalidRange)
        );
        assert_eq!(
            tree.set_perm(0x1000, 0x800, MptPerm::RWX, &mut alloc),
            Err(MptError::InvalidRange)
        );
        assert_eq!(tree.get_perm(0x1000), None);
    }

    #[test]
    fn leaf_above_level0_is_unsupported() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        let pa = 0x100_0000;
        // Root → level-1 table, and a valid leaf entry inside it.
        let level1_table = fresh_table(&mut alloc);
        write_entry(
            tree.root_ppn(),
            level_index(MptMode::Smmpt43, pa, 2),
            encode_non_leaf(level1_table),
        );
        write_entry(
            level1_table,
            level_index(MptMode::Smmpt43, pa, 1),
            V_BIT | L_BIT,
        );

        assert_eq!(
            tree.set_perm(pa, 4096, MptPerm::R, &mut alloc),
            Err(MptError::UnsupportedLeaf { level: 1 })
        );
        assert_eq!(tree.get_perm(pa), Some(MptPerm::NONE));
    }

    #[test]
    fn pointer_with_reserved_bits_is_malformed() {
        // Reserved in a pointer entry: 9:2 and 63:54.
        for bit in [2, 3, 9, 54, 63] {
            let mut alloc = MockAlloc::new();
            let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
            let pa = 0x100_0000;
            let level1_table = fresh_table(&mut alloc);
            write_entry(
                tree.root_ppn(),
                level_index(MptMode::Smmpt43, pa, 2),
                // Pointer entry with the reserved bit set.
                encode_non_leaf(level1_table) | (1 << bit),
            );

            assert_eq!(
                tree.set_perm(pa, 4096, MptPerm::R, &mut alloc),
                Err(MptError::MalformedEntry { level: 2 }),
                "reserved bit {bit}"
            );
        }
    }

    #[test]
    fn pointer_at_level0_is_malformed() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        let mapped = 0x100_0000;
        let touched = mapped + 0x1_0000;
        tree.set_perm(mapped, 4096, MptPerm::RWX, &mut alloc)
            .expect("permission update");

        let table = level0_table(&tree, touched);
        write_entry(
            table,
            level_index(MptMode::Smmpt43, touched, 0),
            // Valid, but a pointer: level 0 holds leaves only.
            encode_non_leaf(fresh_table(&mut alloc)),
        );

        assert_eq!(
            tree.set_perm(touched, 4096, MptPerm::R, &mut alloc),
            Err(MptError::MalformedEntry { level: 0 })
        );
        // The unrelated mapping is untouched.
        assert_eq!(tree.get_perm(mapped), Some(MptPerm::RWX));
    }

    #[test]
    fn napot_leaf_is_malformed() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        let base = 0x100_0000;
        tree.set_perm(base, 4096, MptPerm::RWX, &mut alloc)
            .expect("permission update");

        let table = level0_table(&tree, base);
        let idx = level_index(MptMode::Smmpt43, base, 0);
        write_entry(table, idx, V_BIT | L_BIT | N_BIT);

        // Rejected both for a single page and for the whole leaf chunk.
        assert_eq!(
            tree.set_perm(base, 4096, MptPerm::R, &mut alloc),
            Err(MptError::MalformedEntry { level: 0 })
        );
        assert_eq!(
            tree.set_perm(base, 0x1_0000, MptPerm::R, &mut alloc),
            Err(MptError::MalformedEntry { level: 0 })
        );
    }

    #[test]
    fn malformed_entry_inside_a_level0_table_is_rejected() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        // 32 MiB aligned, so the update below covers one whole level-0
        // table.
        let base = 0x200_0000;
        tree.set_perm(base, 4096, MptPerm::RWX, &mut alloc)
            .expect("permission update");

        // A pointer entry in an unrelated slot of the same level-0 table: a
        // whole-table update overwrites that slot, so it must refuse.
        let table = level0_table(&tree, base);
        write_entry(
            table,
            level_index(MptMode::Smmpt43, base + 0x1_0000, 0),
            encode_non_leaf(fresh_table(&mut alloc)),
        );

        assert_eq!(
            tree.set_perm(base, 0x200_0000, MptPerm::R, &mut alloc),
            Err(MptError::MalformedEntry { level: 0 })
        );
        assert_eq!(tree.get_perm(base), Some(MptPerm::RWX));
    }

    #[test]
    fn leaf_with_reserved_bits_is_malformed() {
        // Reserved in a non-NAPOT leaf entry: 7:3 and 63:56.
        for bit in [3, 7, 56, 63] {
            let mut alloc = MockAlloc::new();
            let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
            let base = 0x100_0000;
            tree.set_perm(base, 0x1_0000, MptPerm::RWX, &mut alloc)
                .expect("permission update");

            let table = level0_table(&tree, base);
            let idx = level_index(MptMode::Smmpt43, base, 0);
            write_entry(table, idx, read_entry(table, idx) | (1 << bit));

            // Rejected both for a single page and for the whole leaf chunk.
            assert_eq!(
                tree.set_perm(base, 4096, MptPerm::R, &mut alloc),
                Err(MptError::MalformedEntry { level: 0 }),
                "reserved bit {bit}"
            );
            assert_eq!(
                tree.set_perm(base, 0x1_0000, MptPerm::R, &mut alloc),
                Err(MptError::MalformedEntry { level: 0 }),
                "reserved bit {bit}"
            );
        }
    }

    #[test]
    fn range_above_the_mode_width_is_rejected() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        let top = 1usize << 43;
        // Starts above the width, and crosses its end.
        assert_eq!(
            tree.set_perm(top, 0x1000, MptPerm::RWX, &mut alloc),
            Err(MptError::InvalidRange)
        );
        assert_eq!(
            tree.set_perm(top - 0x1000, 0x2000, MptPerm::RWX, &mut alloc),
            Err(MptError::InvalidRange)
        );
        // Nothing was written, in particular not at the low address whose
        // table index the first range shares.
        assert_eq!(tree.get_perm(0), None);
        assert_eq!(alloc.allocated, 1, "only the root");

        // A range that ends exactly at the top of the width is inside it.
        assert_eq!(
            tree.set_perm(top - 0x2000, 0x2000, MptPerm::RWX, &mut alloc),
            Ok(())
        );
        assert_eq!(tree.get_perm(top - 0x1000), Some(MptPerm::RWX));

        // Smmpt52 covers the addresses Smmpt43 rejects.
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt52, &mut alloc).unwrap();
        assert_eq!(
            tree.set_perm(1usize << 43, 0x1000, MptPerm::RWX, &mut alloc),
            Ok(())
        );
        assert_eq!(
            tree.set_perm(1usize << 52, 0x1000, MptPerm::RWX, &mut alloc),
            Err(MptError::InvalidRange)
        );
    }

    #[test]
    fn a_rejected_later_chunk_leaves_the_earlier_ones_unwritten() {
        let mut alloc = MockAlloc::new();
        let mut tree = MptTree::new(MptMode::Smmpt43, &mut alloc).unwrap();
        // Root → level-1 table → level-0 table, with a pointer in the slot
        // of the leaf at 0x1_0000.
        let level1 = fresh_table(&mut alloc);
        write_entry(tree.root_ppn(), 0, encode_non_leaf(level1));
        let level0 = fresh_table(&mut alloc);
        write_entry(level1, 0, encode_non_leaf(level0));
        write_entry(
            level0,
            level_index(MptMode::Smmpt43, 0x1_0000, 0),
            encode_non_leaf(fresh_table(&mut alloc)),
        );

        // The first chunk would be applied to the slot at 0 before the
        // second chunk is found malformed.
        let cursor = alloc.next_offset;
        let allocated = alloc.allocated;
        assert_eq!(
            tree.set_perm(0, 0x2_0000, MptPerm::RWX, &mut alloc),
            Err(MptError::MalformedEntry { level: 0 })
        );
        assert_eq!(alloc.allocated, allocated);
        assert_eq!(alloc.next_offset, cursor);
        assert_eq!(read_entry(level0, 0), 0, "the first chunk was applied");
        assert_eq!(tree.get_perm(0), None);
    }

    /// Count the tables an update must create without the planning pass:
    /// walk every chunk on its own and collect the tables it needs in a
    /// set. Every entry already in the tree must be a pointer.
    fn per_chunk_table_count(tree: &MptTree, pa: usize, len: usize) -> usize {
        let mode = tree.mode();
        let levels = mode.levels();
        let mut tables: BTreeSet<Vec<usize>> = BTreeSet::new();
        for chunk in ChunkIter::new(pa >> 12, (pa + len - 1) >> 12) {
            // Index of the entry at every level above level 0, top down.
            let path: Vec<usize> = (1..levels)
                .rev()
                .map(|level| level_index(mode, chunk.pa(), level))
                .collect();
            let mut table = tree.root_ppn();
            let mut existing = 0;
            while existing < path.len() {
                let entry = read_entry(table, path[existing]);
                if !is_valid(entry) {
                    break;
                }
                table = decode_non_leaf_ppn(entry);
                existing += 1;
            }
            if existing == path.len() {
                continue;
            }
            // The table behind the first absent entry and every table below
            // it are new. The entries above a table identify it.
            for below in 0..path.len() - existing {
                tables.insert(path[..path.len() - below].to_vec());
            }
        }
        tables.len()
    }

    #[test]
    fn the_plan_matches_a_per_chunk_count() {
        /// Addresses below one root entry of an Smmpt43 tree.
        const ROOT_SPAN: usize = 1 << 34;
        /// Addresses below one level-0 table (32 MiB).
        const TABLE0: usize = 0x200_0000;

        /// A mapping at address 0, so the first level-1 and level-0 tables
        /// of the tree exist.
        fn first_leaf_mapped(tree: &mut MptTree, alloc: &mut MockAlloc) {
            tree.set_perm(0, 4096, MptPerm::RWX, alloc)
                .expect("permission update");
        }

        /// A level-1 table below each of the first two root entries.
        fn two_level1_tables(tree: &mut MptTree, alloc: &mut MockAlloc) {
            for root_idx in 0..2 {
                let table = fresh_table(alloc);
                write_entry(tree.root_ppn(), root_idx, encode_non_leaf(table));
            }
        }

        struct Case {
            name: &'static str,
            mode: MptMode,
            prepare: fn(&mut MptTree, &mut MockAlloc),
            pa: usize,
            len: usize,
        }
        let cases = [
            // Both chunks share every table they need.
            Case {
                name: "two leaves on an empty tree",
                mode: MptMode::Smmpt43,
                prepare: |_, _| {},
                pa: 0,
                len: 0x2_0000,
            },
            // The walk passes an existing level-1 table, then needs a
            // second level-0 table below it.
            Case {
                name: "two level-0 tables below an existing level-1 table",
                mode: MptMode::Smmpt43,
                prepare: first_leaf_mapped,
                pa: 0,
                len: 2 * TABLE0,
            },
            // The second chunk leaves the first chunk's root entry.
            Case {
                name: "the ends of two root entries",
                mode: MptMode::Smmpt43,
                prepare: |_, _| {},
                pa: ROOT_SPAN - TABLE0,
                len: 2 * TABLE0,
            },
            Case {
                name: "one leaf at each side of a root entry boundary, below existing level-1 tables",
                mode: MptMode::Smmpt43,
                prepare: two_level1_tables,
                pa: ROOT_SPAN - 0x1_0000,
                len: 0x2_0000,
            },
            // Three different chunk shapes; the last one reuses the table
            // the middle one creates.
            Case {
                name: "a single page, a whole level-0 table and a single page",
                mode: MptMode::Smmpt43,
                prepare: |_, _| {},
                pa: TABLE0 - 0x1000,
                len: TABLE0 + 0x2000,
            },
            Case {
                name: "Smmpt52, two pages in one leaf",
                mode: MptMode::Smmpt52,
                prepare: |_, _| {},
                pa: 0x100_0000,
                len: 0x2000,
            },
        ];

        for case in &cases {
            let mut reference = MockAlloc::new();
            let mut tree = MptTree::new(case.mode, &mut reference).unwrap();
            (case.prepare)(&mut tree, &mut reference);
            let prepared = reference.allocated;
            let expected = per_chunk_table_count(&tree, case.pa, case.len);
            assert!(expected > 0, "{}: the case creates no table", case.name);

            // With exactly the counted pages the update succeeds and uses
            // all of them.
            let mut exact = MockAlloc::with_capacity(prepared + expected);
            let mut tree = MptTree::new(case.mode, &mut exact).unwrap();
            (case.prepare)(&mut tree, &mut exact);
            assert_eq!(
                tree.set_perm(case.pa, case.len, MptPerm::RWX, &mut exact),
                Ok(()),
                "{}",
                case.name
            );
            assert_eq!(exact.allocated, prepared + expected, "{}", case.name);

            // With one page fewer it is refused and allocates nothing.
            let mut short = MockAlloc::with_capacity(prepared + expected - 1);
            let mut tree = MptTree::new(case.mode, &mut short).unwrap();
            (case.prepare)(&mut tree, &mut short);
            assert_eq!(
                tree.set_perm(case.pa, case.len, MptPerm::RWX, &mut short),
                Err(MptError::AllocationFailed),
                "{}",
                case.name
            );
            assert_eq!(
                short.allocated, prepared,
                "{}: a refused update allocated a page",
                case.name
            );
        }
    }
}
