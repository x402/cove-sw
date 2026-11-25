//! COVI AIA parameter validation (CoVE v0.7 §11.1–§11.2).
//!
//! A `tvm_aia_params` block describes the virtualized IMSIC of the TVM in
//! its guest physical address space, not the platform's physical
//! interrupt-file array. IMSIC addresses follow
//! `|xxxxxx|Group Index|xxxxxxxxxxx|Hart Index|Guest Index| 0 |`: the base
//! address is the entry with group, hart and guest index all zero, and the
//! declared index fields select one (hart, guest) interrupt file from it.
//!
//! The module is kept free of target dependencies — `core` and the ABI type
//! only, no CSR access, no MMIO, no allocation — so the rules can be
//! unit-tested off-target.

use riscv_cove::interrupt::TvmAiaParams;

/// Position of the guest index field: the low bits of an IMSIC address are
/// the guest interrupt-file page offset.
const PAGE_SHIFT: u32 = 12;
/// Size of one guest interrupt-file page.
const PAGE_SIZE: usize = 1 << PAGE_SHIFT;

/// Lowest legal `group_index_shift` (CoVE v0.7 §11.1: "Must be >= 24").
const GROUP_SHIFT_MIN: u32 = 24;
/// Highest representable shift in a 64-bit address.
const SHIFT_MAX: u32 = usize::BITS as u32 - 1;

/// Why a `tvm_aia_params` block was rejected.
///
/// Table 33 names no code for a malformed field inside the block; the
/// project maps every variant to `SBI_ERR_INVALID_PARAM`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AiaParamsError {
    /// `imsic_base_addr` is not 4 KiB aligned.
    BaseAlignment,
    /// `imsic_base_addr` has a bit set inside a declared index field.
    BaseIndexBits,
    /// `group_index_bits` is nonzero (single-group implementation limit).
    GroupIndexBits,
    /// `group_index_shift` is below 24 or does not fit in the address.
    GroupIndexShift,
    /// `guest_index_bits` is below `log2(guests_per_hart + 1)` or above 52.
    GuestIndexBits,
    /// The hart index field does not fit above the page and guest fields.
    HartIndexBits,
    /// `guests_per_hart` is nonzero (nested IMSIC virtualization is
    /// unsupported; §11.1 allows an implementation to reject it).
    GuestsPerHart,
}

/// Validated index-field geometry of one `tvm_aia_params` block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AiaLayout {
    /// Bits selecting the guest index.
    pub guest_mask: usize,
    /// Union of every declared index field, i.e. every bit the base zeroes.
    pub index_mask: usize,
}

impl AiaLayout {
    /// Layout of a disabled AIA state: no index fields, base 0 only.
    pub const EMPTY: Self = Self {
        guest_mask: 0,
        index_mask: 0,
    };

    /// Whether `gpa` is one (hart, guest) entry of the virtual array based
    /// at `base`.
    ///
    /// The address may differ from the base only inside the declared index
    /// fields, and its guest index must be 0. With `guests_per_hart = 0`
    /// guest index 0 is the only valid per-vCPU base, even when the block
    /// declares guest index bits.
    pub fn contains_gpa(&self, base: usize, gpa: usize) -> bool {
        (gpa & !self.index_mask) == (base & !self.index_mask) && (gpa & self.guest_mask) == 0
    }
}

/// Mask of a `bits`-wide field starting at `shift`, or `None` when the field
/// does not fit in a `usize`. A zero-width field is always representable.
fn field_mask(shift: u32, bits: u32) -> Option<usize> {
    if bits == 0 {
        return Some(0);
    }
    let shift = shift as usize;
    let bits = bits as usize;
    if shift >= usize::BITS as usize || bits > usize::BITS as usize - shift {
        return None;
    }
    let width = if bits == usize::BITS as usize {
        usize::MAX
    } else {
        (1usize << bits) - 1
    };
    Some(width << shift)
}

/// `ceil(log2(guests_per_hart + 1))`, computed as the bit width of
/// `guests_per_hart` without floating point. CoVE v0.7 §11.1: the guest
/// index field must be at least this wide.
fn guest_bits_min(guests_per_hart: u32) -> u32 {
    u32::BITS - guests_per_hart.leading_zeros()
}

/// Validates one `tvm_aia_params` block and derives its index masks.
pub fn validate(params: &TvmAiaParams) -> Result<AiaLayout, AiaParamsError> {
    if params.imsic_base_addr % PAGE_SIZE != 0 {
        return Err(AiaParamsError::BaseAlignment);
    }
    if params.group_index_bits != 0 {
        // Single-group implementation.
        return Err(AiaParamsError::GroupIndexBits);
    }
    if params.group_index_shift < GROUP_SHIFT_MIN || params.group_index_shift > SHIFT_MAX {
        return Err(AiaParamsError::GroupIndexShift);
    }
    // The guest field starts at bit 12, so at most 52 bits fit below XLEN.
    // This bound also keeps `PAGE_SHIFT + guest_index_bits` from overflowing.
    if params.guest_index_bits > 52 {
        return Err(AiaParamsError::GuestIndexBits);
    }
    // Spec minimum on its own; the implementation limit right below is what
    // rejects every nonzero `guests_per_hart` today.
    if params.guest_index_bits < guest_bits_min(params.guests_per_hart) {
        return Err(AiaParamsError::GuestIndexBits);
    }
    if params.guests_per_hart != 0 {
        // Nested IMSIC virtualization is unsupported.
        return Err(AiaParamsError::GuestsPerHart);
    }

    // Hart index sits above the guest index; `field_mask` rejects a nonzero
    // field that does not fit in the address.
    let hart_shift = PAGE_SHIFT + params.guest_index_bits;
    let guest_mask =
        field_mask(PAGE_SHIFT, params.guest_index_bits).ok_or(AiaParamsError::GuestIndexBits)?;
    let hart_mask =
        field_mask(hart_shift, params.hart_index_bits).ok_or(AiaParamsError::HartIndexBits)?;
    let index_mask = guest_mask | hart_mask;

    // The base is the entry with every index field zero.
    if params.imsic_base_addr & index_mask != 0 {
        return Err(AiaParamsError::BaseIndexBits);
    }

    Ok(AiaLayout {
        guest_mask,
        index_mask,
    })
}
