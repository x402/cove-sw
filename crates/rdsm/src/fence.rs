//! `MFENCE.PA` and `MINVAL.PA` instruction wrappers.
//!
//! These instructions synchronize MPT (Memory Protection Table) updates.
//! They are M-mode only and defined by the smmtt specification (§3.2, §3.3).
//!
//! **⚠️ Temporary encoding convention**: The func7 values for
//! `MFENCE.PA` and `MINVAL.PA` are **not** specified in smmtt spec v0.49.
//! The encodings used here are temporary and must be adjusted when the
//! spec finalizes. All encoding constants are centralized in this file
//! for easy modification.

#[cfg(target_arch = "riscv64")]
use core::arch::asm;

// ── Instruction encoding constants ────────────────────────────────────
//
// R-type format: func7[31:25] | rs2[24:20] | rs1[19:15] | func3[14:12] | rd[11:7] | opcode[6:0]
//
// opcode = SYSTEM = 0b1110011 = 0x73
// func3  = PRIV   = 0b000     = 0x0
// rd     = x0     = 0         (no destination register)

#[cfg(target_arch = "riscv64")]
const OPCODE_SYSTEM: u32 = 0x73;
#[cfg(target_arch = "riscv64")]
const FUNC3_PRIV: u32 = 0x0;

/// MFENCE.PA: func7 = 0001110 = 0x0E
// TODO: temporary encoding convention, adjust when spec finalizes
#[cfg(target_arch = "riscv64")]
const FUNC7_MFENCE_PA: u32 = 0x0E;

/// MINVAL.PA: func7 = 0001111 = 0x0F
// TODO: temporary encoding convention, adjust when spec finalizes
#[cfg(target_arch = "riscv64")]
const FUNC7_MINVAL_PA: u32 = 0x0F;

/// Execute `MFENCE.PA` to synchronize MPT updates.
///
/// `MFENCE.PA` orders MPT table updates with subsequent protection checks.
/// Only valid in M-mode.
///
/// # Arguments
/// * `paddr` — Physical address to fence. Use `0` for all addresses.
/// * `sdid`  — Supervisor domain ID to fence. Use `0` for all supervisor
///   domains.
///
/// # Behavior (per smmtt §3.2)
/// | `paddr` (rs1) | `sdid` (rs2) | Effect |
/// | x0 (=0)       | x0 (=0)      | Fence all MPT entries for all SDs |
/// | x0 (=0)       | ≠x0          | Fence all MPT entries for the given SDID |
/// | ≠x0           | x0 (=0)      | Fence leaf MPT entries for the given PA, all SDs |
/// | ≠x0           | ≠x0          | Fence leaf MPT entries for the given PA and SDID |
///
/// When `paddr` or `sdid` is `0`, the corresponding register operand is
/// `x0` (the hard-wired zero register), matching the spec semantics.
// TODO: temporary encoding convention, adjust when spec finalizes
#[cfg(target_arch = "riscv64")]
#[inline]
pub fn mfence_pa(paddr: usize, sdid: usize) {
    unsafe {
        match (paddr == 0, sdid == 0) {
            (true, true) => asm!(
                ".insn r {opcode}, {func3}, {func7}, x0, x0, x0",
                opcode = const OPCODE_SYSTEM,
                func3 = const FUNC3_PRIV,
                func7 = const FUNC7_MFENCE_PA,
                options(nostack, preserves_flags),
            ),
            (true, false) => asm!(
                ".insn r {opcode}, {func3}, {func7}, x0, x0, {sdid}",
                opcode = const OPCODE_SYSTEM,
                func3 = const FUNC3_PRIV,
                func7 = const FUNC7_MFENCE_PA,
                sdid = in(reg) sdid,
                options(nostack, preserves_flags),
            ),
            (false, true) => asm!(
                ".insn r {opcode}, {func3}, {func7}, x0, {paddr}, x0",
                opcode = const OPCODE_SYSTEM,
                func3 = const FUNC3_PRIV,
                func7 = const FUNC7_MFENCE_PA,
                paddr = in(reg) paddr,
                options(nostack, preserves_flags),
            ),
            (false, false) => asm!(
                ".insn r {opcode}, {func3}, {func7}, x0, {paddr}, {sdid}",
                opcode = const OPCODE_SYSTEM,
                func3 = const FUNC3_PRIV,
                func7 = const FUNC7_MFENCE_PA,
                paddr = in(reg) paddr,
                sdid = in(reg) sdid,
                options(nostack, preserves_flags),
            ),
        }
    }
}

/// Execute `MINVAL.PA` for batch MPT invalidation.
///
/// `MINVAL.PA` is only ordered against `SFENCE.W.INVAL` and
/// `SFENCE.INVAL.IR`. When executed in order: `SFENCE.W.INVAL`,
/// `MINVAL.PA`, `SFENCE.INVAL.IR` has the same effect as `MFENCE.PA`
/// with the same rs1/rs2. Only valid in M-mode. Requires both Svinval
/// and Smsdid to be implemented.
///
/// # Arguments
/// * `paddr` — Physical address to invalidate. Use `0` for all addresses.
/// * `sdid`  — Supervisor domain ID. Use `0` for all SDs.
///
/// See [`mfence_pa`] for the rs1/rs2 semantics table.
// TODO: temporary encoding convention, adjust when spec finalizes
#[cfg(target_arch = "riscv64")]
#[inline]
pub fn minval_pa(paddr: usize, sdid: usize) {
    unsafe {
        match (paddr == 0, sdid == 0) {
            (true, true) => asm!(
                ".insn r {opcode}, {func3}, {func7}, x0, x0, x0",
                opcode = const OPCODE_SYSTEM,
                func3 = const FUNC3_PRIV,
                func7 = const FUNC7_MINVAL_PA,
                options(nostack, preserves_flags),
            ),
            (true, false) => asm!(
                ".insn r {opcode}, {func3}, {func7}, x0, x0, {sdid}",
                opcode = const OPCODE_SYSTEM,
                func3 = const FUNC3_PRIV,
                func7 = const FUNC7_MINVAL_PA,
                sdid = in(reg) sdid,
                options(nostack, preserves_flags),
            ),
            (false, true) => asm!(
                ".insn r {opcode}, {func3}, {func7}, x0, {paddr}, x0",
                opcode = const OPCODE_SYSTEM,
                func3 = const FUNC3_PRIV,
                func7 = const FUNC7_MINVAL_PA,
                paddr = in(reg) paddr,
                options(nostack, preserves_flags),
            ),
            (false, false) => asm!(
                ".insn r {opcode}, {func3}, {func7}, x0, {paddr}, {sdid}",
                opcode = const OPCODE_SYSTEM,
                func3 = const FUNC3_PRIV,
                func7 = const FUNC7_MINVAL_PA,
                paddr = in(reg) paddr,
                sdid = in(reg) sdid,
                options(nostack, preserves_flags),
            ),
        }
    }
}
