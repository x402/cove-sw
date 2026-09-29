//! Machine-interrupt policy: the Smmtt MSDEI claim path.
//!
//! Runtime transports the machine timer/software/external interrupts
//! itself; every other machine-interrupt cause is offered to the installed
//! [`MachineInterruptPolicy`] ahead of Runtime's fail-stop default. The
//! Smmtt MSDEI (cause 14) belongs to the RDSM: per Smmtt v0.49 §6.3.4–6.3.5,
//! `mip.MSDEIP` is read-only and reports `msideip & msideie != 0`, so the
//! only way to acknowledge the interrupt is clearing the pending SID bits
//! in `msideie` — writing 0 wholesale would disable every SID's MSDEI.

use log::warn;

use rdsm::interrupt::MsdeiTrap;
use runtime::machine_irq::MachineInterruptPolicy;

/// Claims the MSDEI and acknowledges its pending SIDs. Any other cause is
/// reported unowned and stays with Runtime's fail-stop default.
pub struct RdsmMachineIrq;

/// The instance the embedding firmware installs during boot.
pub static RDSM_MACHINE_IRQ: RdsmMachineIrq = RdsmMachineIrq;

impl MachineInterruptPolicy for RdsmMachineIrq {
    fn handle_interrupt(&self, cause: usize) -> bool {
        if cause != MsdeiTrap::MSDEI_CODE {
            return false;
        }
        let msdei = MsdeiTrap::current();
        let pending = msdei.pending_mask();
        if pending == 0 {
            warn!("RDSM: MSDEI interrupt but no pending SIDs");
        } else {
            warn!("RDSM: MSDEI interrupt, pending SID mask: {:#x}", pending);
            for sid in msdei.pending_sids() {
                warn!("RDSM:   pending SID {}", sid);
            }
        }
        // Acknowledge by clearing exactly the pending SID bits in msideie;
        // msideip is read-only, so this is the sole acknowledge channel.
        rdsm::csr::write_msideie(ack_msideie(rdsm::csr::read_msideie(), pending));
        true
    }
}

/// The acknowledgement mask: `msideie` with exactly the pending SID bits
/// cleared, never anything else and never the whole register.
fn ack_msideie(current_msideie: usize, pending: usize) -> usize {
    current_msideie & !pending
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_msdei_causes_are_unowned() {
        let policy = RdsmMachineIrq;
        assert!(!policy.handle_interrupt(0));
        assert!(!policy.handle_interrupt(5));
        assert!(!policy.handle_interrupt(13));
        assert!(!policy.handle_interrupt(15));
        assert!(!policy.handle_interrupt(usize::MAX));
    }

    #[test]
    fn msdei_cause_is_claimed() {
        // On the host the CSR stubs read 0, so the handler takes the
        // "no pending SIDs" branch and acknowledges a no-op write.
        let policy = RdsmMachineIrq;
        assert!(policy.handle_interrupt(MsdeiTrap::MSDEI_CODE));
    }

    #[test]
    fn ack_clears_exactly_pending_bits() {
        assert_eq!(ack_msideie(0b1011, 0b0010), 0b1001);
        assert_eq!(ack_msideie(0b1011, 0b1000), 0b0011);
        assert_eq!(ack_msideie(0xFFFF, 0xFFFF), 0);
    }

    #[test]
    fn ack_never_disables_unrelated_sids() {
        // Zero pending must leave msideie untouched (never write 0).
        assert_eq!(ack_msideie(0b1111, 0), 0b1111);
        // Bits not pending stay enabled even when others are acknowledged.
        assert_eq!(ack_msideie(0xFF, 1 << 63), 0xFF & !(1 << 63));
    }
}
