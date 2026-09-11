//! RDSM (Root Domain Security Manager) substrate for RISC-V Supervisor
//! Domains Access Protection (smmtt).
//!
//! This crate provides the M-mode software primitives for managing
//! Supervisor Domain isolation via the Smsdid, Smmpt, and Smsdia ISA
//! extensions:
//!
//! - **CSR abstractions** ([`csr`]): `mmpt`, `msdcfg`, `msideip`,
//!   `msideie` read/write wrappers with field encoding/decoding.
//! - **Instruction wrappers** ([`fence`]): `MFENCE.PA` and `MINVAL.PA`
//!   inline-asm wrappers for MPT cache synchronization.
//! - **MPT radix tree** ([`mpt`]): [`mpt::MptTree`] manages the
//!   Machine-level Memory Protection Tables for Smmpt43/52/64 modes.
//! - **Supervisor Domain management** ([`domain`]): SDID allocation,
//!   domain context switching via `mmpt` CSR programming.
//! - **Interrupt domain management** ([`interrupt`]): SIDN allocation,
//!   `msdcfg.SIDN` switching, MSDEI interrupt decoding.
//! - **Fault decoding** ([`fault`]): [`fault::MptFault`] decodes MPT
//!   access violations from trap context.
//! - **Hardware probing** ([`probe`]): trap-safe detection of Smsdid,
//!   Smmpt, and Smsdia extensions via the [`probe::TrapSafeCsr`] trait.
//!
//! The crate is `no_std` and has no external dependencies. CSR and
//! instruction wrappers use inline `asm!` and only compile on RISC-V
//! targets; field encoding/decoding logic and pure-software types
//! (allocators, fault decoders) are platform-independent and unit-tested
//! on the host.
#![no_std]

pub mod csr;
pub mod domain;
pub mod fault;
pub mod fence;
pub mod interrupt;
pub mod mpt;
pub mod payload;
pub mod probe;
