//! Trap-safe hardware probing for Smsdid, Smmpt, and Smsdia extensions.
//!
//! The [`TrapSafeCsr`] trait abstracts CSR access so that the probe
//! functions in this module remain platform-independent.  The platform
//! provides an implementation that can handle CSR read/write traps
//! gracefully (e.g. by catching the illegal-instruction exception and
//! returning `None` / `false`).
#![cfg_attr(test, allow(unused_extern_crates))]
#[cfg(test)]
extern crate std;

// ── TrapSafeCsr trait (task 8.1) ─────────────────────────────────────────

/// Trait for trap-safe CSR access.
///
/// Implementations must catch or suppress the illegal-instruction
/// exception that occurs when reading from or writing to a CSR that
/// is not implemented by the hardware.  This keeps the `rdsm` crate
/// independent of platform-specific trap handling (e.g. `mstatus.MPRV`
/// delegation or custom trap-and-emulate logic).
pub trait TrapSafeCsr {
    /// Read a CSR at `csr_addr`.
    ///
    /// Returns `None` if the CSR is not implemented (the access trapped)
    /// or if the value could not be obtained.
    fn read_csr(&self, csr_addr: u16) -> Option<usize>;

    /// Write `value` to the CSR at `csr_addr`.
    ///
    /// Returns `false` if the CSR is not implemented (the access trapped)
    /// or the write could not be completed.
    fn write_csr(&mut self, csr_addr: u16, value: usize) -> bool;
}

// ── probe_smsdid (task 8.2) ─────────────────────────────────────────────

/// Probe whether the Smsdid (Supervisor Domain ID) extension is
/// implemented.
///
/// Attempts to read the `mmpt` CSR (0x382).  If the read succeeds
/// (returns `Some`), the extension is present.
pub fn probe_smsdid(probe: &impl TrapSafeCsr) -> bool {
    probe.read_csr(crate::csr::CSR_MMPT).is_some()
}

// ── probe_smmpt (task 8.3) ──────────────────────────────────────────────

/// Probe the highest supported Smmpt addressing mode.
///
/// Tries to write each mode (Smmpt64, Smmpt52, Smmpt43) into the `mmpt`
/// CSR and reads back.  The first mode whose readback matches the written
/// value is the highest implemented mode.
///
/// The original `mmpt` value is saved before probing and restored
/// afterwards.
///
/// Returns `None` if only Bare mode is supported (or `mmpt` is not
/// accessible).
pub fn probe_smmpt(probe: &mut impl TrapSafeCsr) -> Option<crate::csr::MptMode> {
    use crate::csr::{CSR_MMPT, Mmpt, MptMode};

    // Save the original mmpt value.
    let original = probe.read_csr(CSR_MMPT)?;

    // Try modes from highest to lowest.
    let modes = [MptMode::Smmpt64, MptMode::Smmpt52, MptMode::Smmpt43];
    let mut supported = None;

    for mode in modes {
        // Build an mmpt value with this mode, SDID=0, PPN=0.
        let test_val = Mmpt::from_parts(mode, 0, 0).bits();
        if probe.write_csr(CSR_MMPT, test_val)
            && let Some(readback) = probe.read_csr(CSR_MMPT)
        {
            // Extract MODE field manually (bits [63:60]).
            let mode_bits = (readback >> 60) & 0xF;
            let readback_mode = MptMode::from_bits(mode_bits).unwrap_or(MptMode::Bare);
            if readback_mode == mode {
                supported = Some(mode);
                break;
            }
        }
    }

    // Restore the original value.
    probe.write_csr(CSR_MMPT, original);

    supported
}

// ── probe_sdid_len (task 8.4) ───────────────────────────────────────────

/// Probe the SDID field width (in bits) via WARL (Write-Anything
/// Read-Legal) probing.
///
/// Writes all 1s into the SDID field of the `mmpt` CSR, reads back,
/// extracts the SDID field, and counts the number of set bits.
///
/// Returns `0` if the `mmpt` CSR is not implemented.
pub fn probe_sdid_len(probe: &mut impl TrapSafeCsr) -> usize {
    use crate::csr::CSR_MMPT;

    // Save the original mmpt value.
    let original = match probe.read_csr(CSR_MMPT) {
        Some(val) => val,
        None => return 0,
    };

    // Write all 1s to the SDID field (bits [57:52]).
    let all_ones_sdid = original | (0x3F << 52);
    if !probe.write_csr(CSR_MMPT, all_ones_sdid) {
        // Restore and return 0.
        probe.write_csr(CSR_MMPT, original);
        return 0;
    }

    // Read back and extract the SDID field manually (bits [57:52]).
    let readback = probe.read_csr(CSR_MMPT).unwrap_or(0);
    let sdid_field = (readback >> 52) & 0x3F;

    // Restore the original value.
    probe.write_csr(CSR_MMPT, original);

    // The number of implemented SDID bits equals the number of set bits
    // after a WARL write of all-1s.
    sdid_field.count_ones() as usize
}

// ── probe_smsdia (task 8.5) ─────────────────────────────────────────────

/// Probe whether the Smsdia (Supervisor Domain Interrupts and
/// Aliasing) extension is implemented.
///
/// Attempts to read the `msdcfg` CSR (0x74E).  If the read succeeds
/// (returns `Some`), the extension is present.
pub fn probe_smsdia(probe: &impl TrapSafeCsr) -> bool {
    probe.read_csr(crate::csr::CSR_MSDCFG).is_some()
}

// ── Unit tests (task 8.6) ────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::csr::{CSR_MMPT, CSR_MSDCFG, Mmpt, MptMode};

    /// A mock CSR provider that simulates WARL behaviour for testing
    /// the probe functions on the host.
    struct MockCsr {
        /// Raw `mmpt` register value (only meaningful when `has_mmpt`).
        mmpt_val: usize,
        /// Raw `msdcfg` register value (only meaningful when `has_msdcfg`).
        msdcfg_val: usize,
        /// Whether the `mmpt` CSR is implemented.
        has_mmpt: bool,
        /// Whether the `msdcfg` CSR is implemented.
        has_msdcfg: bool,
        /// The set of Smmpt modes the simulated hardware supports
        /// (in probe order — highest first).  An empty vec means only
        /// Bare is supported.
        supported_modes: std::vec::Vec<MptMode>,
        /// The number of implemented SDID bits (0–6).  Unimplemented
        /// bits are masked to zero on readback after a write.
        sdid_width: usize,
    }

    impl TrapSafeCsr for MockCsr {
        fn read_csr(&self, csr_addr: u16) -> Option<usize> {
            match csr_addr {
                CSR_MMPT if self.has_mmpt => Some(self.mmpt_val),
                CSR_MSDCFG if self.has_msdcfg => Some(self.msdcfg_val),
                _ => None,
            }
        }

        fn write_csr(&mut self, csr_addr: u16, value: usize) -> bool {
            match csr_addr {
                CSR_MMPT if self.has_mmpt => {
                    let written = Mmpt(value);
                    let mode = written.mode();
                    let ppn = written.ppn();
                    let sdid = written.sdid();

                    // Determine the effective mode after WARL.
                    let actual_mode =
                        if mode == MptMode::Bare || self.supported_modes.contains(&mode) {
                            mode
                        } else {
                            // Silently downgrade to the highest supported
                            // mode (or Bare if none).
                            self.supported_modes
                                .first()
                                .copied()
                                .unwrap_or(MptMode::Bare)
                        };

                    // Mask SDID to the implemented width.
                    let sdid_mask = match self.sdid_width {
                        0..=5 => (1 << self.sdid_width) - 1,
                        _ => 0x3F, // 6 or more bits → full 6-bit field
                    };
                    let actual_sdid = sdid & sdid_mask;

                    self.mmpt_val = Mmpt::from_parts(actual_mode, actual_sdid, ppn).bits();
                    true
                }
                CSR_MSDCFG if self.has_msdcfg => {
                    self.msdcfg_val = value;
                    true
                }
                _ => false,
            }
        }
    }

    // ── probe_smsdid tests ──────────────────────────────────────────

    #[test]
    fn probe_smsdid_present() {
        let mock = MockCsr {
            mmpt_val: 0,
            msdcfg_val: 0,
            has_mmpt: true,
            has_msdcfg: false,
            supported_modes: std::vec![],
            sdid_width: 6,
        };
        assert!(probe_smsdid(&mock));
    }

    #[test]
    fn probe_smsdid_absent() {
        let mock = MockCsr {
            mmpt_val: 0,
            msdcfg_val: 0,
            has_mmpt: false,
            has_msdcfg: false,
            supported_modes: std::vec![],
            sdid_width: 6,
        };
        assert!(!probe_smsdid(&mock));
    }

    // ── probe_smmpt tests ───────────────────────────────────────────

    #[test]
    fn probe_smmpt_smmpt43() {
        let mut mock = MockCsr {
            mmpt_val: 0,
            msdcfg_val: 0,
            has_mmpt: true,
            has_msdcfg: false,
            supported_modes: std::vec![MptMode::Smmpt43],
            sdid_width: 6,
        };
        assert_eq!(probe_smmpt(&mut mock), Some(MptMode::Smmpt43));
    }

    #[test]
    fn probe_smmpt_smmpt64() {
        let mut mock = MockCsr {
            mmpt_val: 0,
            msdcfg_val: 0,
            has_mmpt: true,
            has_msdcfg: false,
            supported_modes: std::vec![MptMode::Smmpt64, MptMode::Smmpt52, MptMode::Smmpt43,],
            sdid_width: 6,
        };
        assert_eq!(probe_smmpt(&mut mock), Some(MptMode::Smmpt64));
    }

    #[test]
    fn probe_smmpt_none() {
        let mut mock = MockCsr {
            mmpt_val: 0,
            msdcfg_val: 0,
            has_mmpt: true,
            has_msdcfg: false,
            supported_modes: std::vec![], // only Bare
            sdid_width: 6,
        };
        assert_eq!(probe_smmpt(&mut mock), None);
    }

    #[test]
    fn probe_smmpt_restores_original() {
        let original_val = Mmpt::from_parts(MptMode::Smmpt43, 5, 0x12345).bits();
        let mut mock = MockCsr {
            mmpt_val: original_val,
            msdcfg_val: 0,
            has_mmpt: true,
            has_msdcfg: false,
            supported_modes: std::vec![MptMode::Smmpt43],
            sdid_width: 4,
        };
        let _ = probe_smmpt(&mut mock);
        assert_eq!(mock.mmpt_val, original_val);
    }

    // ── probe_sdid_len tests ────────────────────────────────────────

    #[test]
    fn probe_sdid_len_6bits() {
        let mut mock = MockCsr {
            mmpt_val: 0,
            msdcfg_val: 0,
            has_mmpt: true,
            has_msdcfg: false,
            supported_modes: std::vec![MptMode::Smmpt43],
            sdid_width: 6,
        };
        assert_eq!(probe_sdid_len(&mut mock), 6);
    }

    #[test]
    fn probe_sdid_len_3bits() {
        let mut mock = MockCsr {
            mmpt_val: 0,
            msdcfg_val: 0,
            has_mmpt: true,
            has_msdcfg: false,
            supported_modes: std::vec![MptMode::Smmpt43],
            sdid_width: 3,
        };
        assert_eq!(probe_sdid_len(&mut mock), 3);
    }

    #[test]
    fn probe_sdid_len_not_implemented() {
        let mut mock = MockCsr {
            mmpt_val: 0,
            msdcfg_val: 0,
            has_mmpt: false,
            has_msdcfg: false,
            supported_modes: std::vec![],
            sdid_width: 0,
        };
        assert_eq!(probe_sdid_len(&mut mock), 0);
    }

    #[test]
    fn probe_sdid_len_restores_original() {
        let original_val = Mmpt::from_parts(MptMode::Smmpt43, 5, 0x12345).bits();
        let mut mock = MockCsr {
            mmpt_val: original_val,
            msdcfg_val: 0,
            has_mmpt: true,
            has_msdcfg: false,
            supported_modes: std::vec![MptMode::Smmpt43],
            sdid_width: 4,
        };
        let _ = probe_sdid_len(&mut mock);
        assert_eq!(mock.mmpt_val, original_val);
    }

    // ── probe_smsdia tests ──────────────────────────────────────────

    #[test]
    fn probe_smsdia_present() {
        let mock = MockCsr {
            mmpt_val: 0,
            msdcfg_val: 0,
            has_mmpt: false,
            has_msdcfg: true,
            supported_modes: std::vec![],
            sdid_width: 6,
        };
        assert!(probe_smsdia(&mock));
    }

    #[test]
    fn probe_smsdia_absent() {
        let mock = MockCsr {
            mmpt_val: 0,
            msdcfg_val: 0,
            has_mmpt: false,
            has_msdcfg: false,
            supported_modes: std::vec![],
            sdid_width: 6,
        };
        assert!(!probe_smsdia(&mock));
    }
}
