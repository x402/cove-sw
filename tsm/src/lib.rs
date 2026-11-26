//! Target-independent TSM logic.
//!
//! The library holds the parts of the TSM that need neither CSR access nor
//! MMIO nor `ecall`: the page tracker's state machine ([`mm`]), the G-stage
//! page table ([`page_table`]), the sharing hand-over between the TVM and the
//! host ([`share`]), the COVI AIA parameter validation ([`aia`]) and the SBI
//! return encoding ([`sbi`]). They are plain logic, so they are unit-tested on
//! the host.
//!
//! The firmware binary (`src/main.rs`) supplies everything around them: CSR
//! access, MMIO, the private RDSM call wrappers, the global allocator and
//! the entry points.

#![no_std]

#[cfg(test)]
extern crate std;

pub mod aia;
pub mod mm;
pub mod page_table;
pub mod sbi;
pub mod share;
