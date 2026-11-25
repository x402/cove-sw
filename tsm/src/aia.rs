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

#[cfg(test)]
mod tests {
    use super::*;

    fn params(
        base: usize,
        group_bits: u32,
        group_shift: u32,
        hart_bits: u32,
        guest_bits: u32,
        guests: u32,
    ) -> TvmAiaParams {
        TvmAiaParams {
            imsic_base_addr: base,
            group_index_bits: group_bits,
            group_index_shift: group_shift,
            hart_index_bits: hart_bits,
            guest_index_bits: guest_bits,
            guests_per_hart: guests,
        }
    }

    #[test]
    fn accepts_a_single_hart_without_guest_files() {
        let layout = validate(&params(0x8003_0000, 0, 24, 1, 0, 0)).unwrap();
        assert_eq!(
            layout,
            AiaLayout {
                guest_mask: 0,
                index_mask: 0x1000,
            }
        );
    }

    #[test]
    fn accepts_hart_and_guest_index_fields() {
        let layout = validate(&params(0x8000_0000, 0, 24, 3, 2, 0)).unwrap();
        assert_eq!(
            layout,
            AiaLayout {
                guest_mask: 0x3000,
                index_mask: 0x1F000,
            }
        );
    }

    #[test]
    fn rejects_misaligned_base() {
        assert_eq!(
            validate(&params(0x8003_0004, 0, 24, 1, 0, 0)),
            Err(AiaParamsError::BaseAlignment)
        );
    }

    #[test]
    fn rejects_group_index_bits() {
        assert_eq!(
            validate(&params(0x8000_0000, 1, 24, 1, 0, 0)),
            Err(AiaParamsError::GroupIndexBits)
        );
    }

    #[test]
    fn rejects_out_of_range_group_shift() {
        assert_eq!(
            validate(&params(0x8000_0000, 0, 23, 1, 0, 0)),
            Err(AiaParamsError::GroupIndexShift)
        );
        assert_eq!(
            validate(&params(0x8000_0000, 0, 64, 1, 0, 0)),
            Err(AiaParamsError::GroupIndexShift)
        );
    }

    #[test]
    fn rejects_out_of_range_guest_index_bits() {
        assert_eq!(
            validate(&params(0x8000_0000, 0, 24, 1, 53, 0)),
            Err(AiaParamsError::GuestIndexBits)
        );
        // 7 guests per hart need 3 guest index bits.
        assert_eq!(
            validate(&params(0x8000_0000, 0, 24, 1, 2, 7)),
            Err(AiaParamsError::GuestIndexBits)
        );
    }

    #[test]
    fn rejects_guests_per_hart() {
        // One guest per hart: 1 guest index bit meets the minimum.
        assert_eq!(
            validate(&params(0x8000_0000, 0, 24, 1, 1, 1)),
            Err(AiaParamsError::GuestsPerHart)
        );
    }

    #[test]
    fn rejects_hart_field_past_the_address_end() {
        // The guest field [12, 64) leaves no room for the hart field.
        assert_eq!(
            validate(&params(0, 0, 24, 1, 52, 0)),
            Err(AiaParamsError::HartIndexBits)
        );
    }

    #[test]
    fn rejects_base_bits_inside_index_fields() {
        // Bit 12 is inside the 2-bit guest field starting at bit 12.
        assert_eq!(
            validate(&params(0x8000_1000, 0, 24, 0, 2, 0)),
            Err(AiaParamsError::BaseIndexBits)
        );
        // Bit 12 is inside the 1-bit hart field starting at bit 12.
        assert_eq!(
            validate(&params(0x8000_1000, 0, 24, 1, 0, 0)),
            Err(AiaParamsError::BaseIndexBits)
        );
    }

    #[test]
    fn accepts_the_field_boundaries() {
        assert_eq!(
            validate(&params(0x8000_0000, 0, 24, 0, 0, 0)),
            Ok(AiaLayout::EMPTY)
        );
        assert_eq!(
            validate(&params(0x8000_0000, 0, 63, 0, 0, 0)),
            Ok(AiaLayout::EMPTY)
        );
        let guest_mask = ((1usize << 52) - 1) << 12;
        assert_eq!(
            validate(&params(0, 0, 24, 0, 52, 0)),
            Ok(AiaLayout {
                guest_mask,
                index_mask: guest_mask,
            })
        );
    }

    #[test]
    fn guest_bits_min_table() {
        assert_eq!(guest_bits_min(0), 0);
        assert_eq!(guest_bits_min(1), 1);
        assert_eq!(guest_bits_min(2), 2);
        assert_eq!(guest_bits_min(3), 2);
        assert_eq!(guest_bits_min(4), 3);
        assert_eq!(guest_bits_min(7), 3);
        assert_eq!(guest_bits_min(u32::MAX), 32);
    }

    #[test]
    fn field_mask_table() {
        assert_eq!(field_mask(12, 0), Some(0));
        assert_eq!(field_mask(12, 52), Some(((1usize << 52) - 1) << 12));
        assert_eq!(field_mask(0, 64), Some(usize::MAX));
        assert_eq!(field_mask(12, 53), None);
        assert_eq!(field_mask(64, 0), Some(0));
        assert_eq!(field_mask(64, 1), None);
        assert_eq!(field_mask(0, 65), None);
    }

    #[test]
    fn contains_gpa_accepts_only_declared_index_bits() {
        let layout = AiaLayout {
            index_mask: 0x1F000,
            guest_mask: 0x3000,
        };
        let base = 0x8000_0000;

        assert!(layout.contains_gpa(base, base));
        assert!(layout.contains_gpa(base, base + 0x4000));
        assert!(layout.contains_gpa(base, base + 0x1C000));
        assert!(!layout.contains_gpa(base, base + 0x1000));
        assert!(!layout.contains_gpa(base, base + 0x5000));
        assert!(!layout.contains_gpa(base, base + 0x20000));
    }

    #[test]
    fn empty_layout_accepts_only_the_base() {
        let base = 0x8000_0000;

        assert!(AiaLayout::EMPTY.contains_gpa(base, base));
        assert!(!AiaLayout::EMPTY.contains_gpa(base, base + 0x1000));
    }
}
