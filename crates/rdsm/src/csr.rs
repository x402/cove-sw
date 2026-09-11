//! CSR abstractions for the Smsdid and Smsdia extensions.
//!
//! Provides typed read/write access to the `mmpt` (0x382), `msdcfg`
//! (0x74E), `msideip` (0xF4F), and `msideie` (0x74F) machine-mode CSRs
//! defined by the smmtt specification.
//!
//! Field encoding/decoding logic is platform-independent and unit-tested
//! on the host. CSR read/write functions use inline `asm!` and are only
//! available on `riscv64` targets.

#[cfg(target_arch = "riscv64")]
use core::arch::asm;

// ── CSR addresses ──────────────────────────────────────────────────────

/// Machine-level Memory Protection Tables register (smmtt §3.1).
pub const CSR_MMPT: u16 = 0x382;

/// Machine Supervisor Domain Configuration register (smmtt §3.5).
pub const CSR_MSDCFG: u16 = 0x74E;

/// Machine Supervisor Domain External Interrupt Pending (smmtt §6.3.3).
pub const CSR_MSIDEIP: u16 = 0xF4F;

/// Machine Supervisor Domain External Interrupt Enable (smmtt §6.3.4).
pub const CSR_MSIDEIE: u16 = 0x74F;

// ── mmpt bit layout ───────────────────────────────────────────────────
//
// Layout matching Smmtt / NEMU implementation:
//   MODE [63:60]  (4 bits, values 0–3)
//   pad2 [59]     (1 bit, WPRI)
//   pad1 [58]     (1 bit, WPRI)
//   SDID [57:52]  (6 bits, SDIDMAX=6 → max 64 domains)
//   pad0 [51:44]  (8 bits, WPRI)
//   PPN  [43:0]   (44 bits)
//
// SDID field width is verified at runtime via WARL probing (see `probe`).

const MODE_SHIFT: usize = 60;
const MODE_BITS: usize = 4;
const SDID_SHIFT: usize = 52;
const SDID_BITS: usize = 6;
const PPN_BITS: usize = 44;

const fn field_mask(bits: usize) -> usize {
    (1 << bits) - 1
}

// ── MptMode ────────────────────────────────────────────────────────────

/// MPT addressing mode programmed into `mmpt.MODE`.
///
/// Encodings follow smmtt §3.1 for MXLEN=64.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MptMode {
    /// No page-based memory protection. SDID and PPN must be zero.
    Bare = 0,
    /// 43-bit physical address space, 3-level table (RV64).
    Smmpt43 = 1,
    /// 52-bit physical address space, 4-level table (RV64).
    Smmpt52 = 2,
    /// 64-bit physical address space, 5-level table (RV64).
    Smmpt64 = 3,
}

impl MptMode {
    /// Decode a MODE field value into an `MptMode`, or `None` if unknown.
    pub const fn from_bits(val: usize) -> Option<Self> {
        match val {
            0 => Some(Self::Bare),
            1 => Some(Self::Smmpt43),
            2 => Some(Self::Smmpt52),
            3 => Some(Self::Smmpt64),
            _ => None,
        }
    }

    /// Encode this mode as a raw integer suitable for the MODE field.
    pub const fn to_bits(self) -> usize {
        self as usize
    }

    /// Number of radix-tree levels for this mode.
    pub const fn levels(self) -> usize {
        match self {
            Self::Bare => 0,
            Self::Smmpt43 => 3,
            Self::Smmpt52 => 4,
            Self::Smmpt64 => 5,
        }
    }

    /// Number of entries in the root table.
    pub const fn root_entries(self) -> usize {
        match self {
            Self::Bare => 0,
            // Smmpt64 root has 2^12 entries; others have 2^9.
            Self::Smmpt64 => 4096,
            _ => 512,
        }
    }

    /// Root table size in bytes.
    pub const fn root_size(self) -> usize {
        self.root_entries() * 8 // MPTESIZE = 8 for all RV64 modes
    }

    /// Number of 4 KiB pages spanned by the root table.
    ///
    /// Smmpt43/52 have a 4 KiB root (1 page); Smmpt64 has a 32 KiB
    /// root (8 pages).  Bare mode has no root table.
    pub const fn root_pages(self) -> usize {
        match self {
            Self::Bare => 0,
            Self::Smmpt64 => 8,
            _ => 1,
        }
    }

    /// Number of pages covered by a single leaf MPTE (2^NUMPGINRANGE).
    pub const fn pages_per_leaf(self) -> usize {
        match self {
            Self::Bare => 0,
            // Smmpt43/52/64: NUMPGINRANGE=4 → 16 pages per leaf
            _ => 16,
        }
    }

    /// Whether this mode is Bare (no page-based protection).
    pub const fn is_bare(self) -> bool {
        matches!(self, Self::Bare)
    }
}

// ── Mmpt ───────────────────────────────────────────────────────────────

/// Typed wrapper for the `mmpt` CSR (0x382).
///
/// Encodes and decodes the MODE, SDID, and PPN fields. Use
/// [`Mmpt::read`] / [`Mmpt::write`] for hardware access (RISC-V only)
/// and [`Mmpt::from_parts`] / field accessors for construction and
/// inspection.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Mmpt(pub usize);

impl Mmpt {
    /// Bare mode: no page-based memory protection.
    pub const BARE: Self = Self(0);

    /// Construct an `mmpt` value from its fields.
    pub const fn from_parts(mode: MptMode, sdid: usize, ppn: usize) -> Self {
        Self(
            (mode.to_bits() << MODE_SHIFT)
                | ((sdid & field_mask(SDID_BITS)) << SDID_SHIFT)
                | (ppn & field_mask(PPN_BITS)),
        )
    }

    /// Decode the MODE field.
    pub const fn mode(&self) -> MptMode {
        let bits = (self.0 >> MODE_SHIFT) & field_mask(MODE_BITS);
        match MptMode::from_bits(bits) {
            Some(m) => m,
            None => MptMode::Bare,
        }
    }

    /// Decode the SDID field.
    pub const fn sdid(&self) -> usize {
        (self.0 >> SDID_SHIFT) & field_mask(SDID_BITS)
    }

    /// Decode the PPN field (physical page number of the MPT root).
    pub const fn ppn(&self) -> usize {
        self.0 & field_mask(PPN_BITS)
    }

    /// Raw register value.
    pub const fn bits(self) -> usize {
        self.0
    }

    /// Read the `mmpt` CSR.
    #[cfg(target_arch = "riscv64")]
    #[inline]
    pub fn read() -> Self {
        let val: usize;
        unsafe {
            asm!(
                "csrr {val}, {csr}",
                val = out(reg) val,
                csr = const CSR_MMPT,
                options(nomem),
            );
        }
        Self(val)
    }

    /// Write this value to the `mmpt` CSR.
    #[cfg(target_arch = "riscv64")]
    #[inline]
    pub fn write(&self) {
        unsafe {
            asm!(
                "csrw {csr}, {val}",
                csr = const CSR_MMPT,
                val = in(reg) self.0,
                options(nomem),
            );
        }
    }
}

// ── Msdcfg ─────────────────────────────────────────────────────────────

/// Typed wrapper for the `msdcfg` CSR (0x74E).
///
/// Currently only the SIDN field (bits 5:0) is accessed; other fields
/// (SEDA, SETA, SRL, SML, etc.) belong to extensions not implemented in
/// this phase.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Msdcfg(usize);

impl Msdcfg {
    const SIDN_SHIFT: usize = 0;
    const SIDN_BITS: usize = 6;

    /// Decode the SIDN field (bits 5:0).
    pub const fn sidn(&self) -> usize {
        (self.0 >> Self::SIDN_SHIFT) & field_mask(Self::SIDN_BITS)
    }

    /// Set the SIDN field (bits 5:0), preserving other bits.
    pub fn set_sidn(&mut self, sidn: usize) {
        let mask = field_mask(Self::SIDN_BITS) << Self::SIDN_SHIFT;
        self.0 = (self.0 & !mask) | ((sidn & field_mask(Self::SIDN_BITS)) << Self::SIDN_SHIFT);
    }

    /// Raw register value.
    pub const fn bits(self) -> usize {
        self.0
    }

    /// Read the `msdcfg` CSR.
    #[cfg(target_arch = "riscv64")]
    #[inline]
    pub fn read() -> Self {
        let val: usize;
        unsafe {
            asm!(
                "csrr {val}, {csr}",
                val = out(reg) val,
                csr = const CSR_MSDCFG,
                options(nomem),
            );
        }
        Self(val)
    }

    /// Write this value to the `msdcfg` CSR.
    #[cfg(target_arch = "riscv64")]
    #[inline]
    pub fn write(&self) {
        unsafe {
            asm!(
                "csrw {csr}, {val}",
                csr = const CSR_MSDCFG,
                val = in(reg) self.0,
                options(nomem),
            );
        }
    }
}

// ── msideip / msideie ──────────────────────────────────────────────────

/// Read `msideip` (CSR 0xF4F) — 64-bit, bit *i* = SID *i* has pending
/// external interrupt.
#[cfg(target_arch = "riscv64")]
#[inline]
pub fn read_msideip() -> usize {
    let val: usize;
    unsafe {
        asm!(
            "csrr {val}, {csr}",
            val = out(reg) val,
            csr = const CSR_MSIDEIP,
            options(nomem),
        );
    }
    val
}

/// Read `msideie` (CSR 0x74F) — 64-bit, bit *i* = SID *i* is enabled to
/// cause MSDEI.
#[cfg(target_arch = "riscv64")]
#[inline]
pub fn read_msideie() -> usize {
    let val: usize;
    unsafe {
        asm!(
            "csrr {val}, {csr}",
            val = out(reg) val,
            csr = const CSR_MSIDEIE,
            options(nomem),
        );
    }
    val
}

/// Write `msideie` (CSR 0x74F).
#[cfg(target_arch = "riscv64")]
#[inline]
pub fn write_msideie(mask: usize) {
    unsafe {
        asm!(
            "csrw {csr}, {val}",
            csr = const CSR_MSIDEIE,
            val = in(reg) mask,
            options(nomem),
        );
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpt_mode_roundtrip() {
        for mode in [
            MptMode::Bare,
            MptMode::Smmpt43,
            MptMode::Smmpt52,
            MptMode::Smmpt64,
        ] {
            let bits = mode.to_bits();
            assert_eq!(MptMode::from_bits(bits), Some(mode));
        }
        assert_eq!(MptMode::from_bits(4), None);
        assert_eq!(MptMode::from_bits(15), None);
    }

    #[test]
    fn mpt_mode_levels_and_sizes() {
        assert_eq!(MptMode::Bare.levels(), 0);
        assert_eq!(MptMode::Smmpt43.levels(), 3);
        assert_eq!(MptMode::Smmpt52.levels(), 4);
        assert_eq!(MptMode::Smmpt64.levels(), 5);

        assert_eq!(MptMode::Smmpt43.root_entries(), 512);
        assert_eq!(MptMode::Smmpt64.root_entries(), 4096);

        assert_eq!(MptMode::Smmpt43.root_size(), 4096);
        assert_eq!(MptMode::Smmpt64.root_size(), 32768);

        assert_eq!(MptMode::Smmpt43.pages_per_leaf(), 16);

        assert_eq!(MptMode::Bare.root_pages(), 0);
        assert_eq!(MptMode::Smmpt43.root_pages(), 1);
        assert_eq!(MptMode::Smmpt52.root_pages(), 1);
        assert_eq!(MptMode::Smmpt64.root_pages(), 8);
    }

    #[test]
    fn mmpt_bare() {
        let bare = Mmpt::BARE;
        assert_eq!(bare.mode(), MptMode::Bare);
        assert_eq!(bare.sdid(), 0);
        assert_eq!(bare.ppn(), 0);
        assert_eq!(bare.bits(), 0);
    }

    #[test]
    fn mmpt_from_parts_and_fields() {
        let mmpt = Mmpt::from_parts(MptMode::Smmpt43, 5, 0x12345);
        assert_eq!(mmpt.mode(), MptMode::Smmpt43);
        assert_eq!(mmpt.sdid(), 5);
        assert_eq!(mmpt.ppn(), 0x12345);
    }

    #[test]
    fn mmpt_smmpt64_mode_bit() {
        let mmpt = Mmpt::from_parts(MptMode::Smmpt64, 0, 0);
        assert_eq!(mmpt.mode(), MptMode::Smmpt64);
        assert_eq!(mmpt.sdid(), 0);
        assert_eq!(mmpt.ppn(), 0);
        // MODE=3 at bits [63:60]
        assert_eq!(mmpt.bits(), 3 << 60);
    }

    #[test]
    fn mmpt_sdid_truncation() {
        // SDID is 6 bits — values > 63 should be truncated.
        let mmpt = Mmpt::from_parts(MptMode::Smmpt43, 0xFF, 0);
        assert_eq!(mmpt.sdid(), 0x3F);
    }

    #[test]
    fn mmpt_ppn_truncation() {
        // PPN is 54 bits — high bits should be truncated.
        let mmpt = Mmpt::from_parts(MptMode::Smmpt43, 0, 1 << 60);
        assert_eq!(mmpt.ppn(), 0); // bit 60 is in MODE field, not PPN
    }

    #[test]
    fn msdcfg_sidn_roundtrip() {
        let mut cfg = Msdcfg(0);
        cfg.set_sidn(42);
        assert_eq!(cfg.sidn(), 42);
        assert_eq!(cfg.bits() & 0x3F, 42);
    }

    #[test]
    fn msdcfg_sidn_preserves_other_bits() {
        let mut cfg = Msdcfg(0xFF00); // high bits set
        cfg.set_sidn(7);
        assert_eq!(cfg.sidn(), 7);
        assert_eq!(cfg.bits() & !0x3F, 0xFF00);
    }

    #[test]
    fn msdcfg_sidn_truncation() {
        let mut cfg = Msdcfg(0);
        cfg.set_sidn(0xFF);
        assert_eq!(cfg.sidn(), 0x3F);
    }
}
