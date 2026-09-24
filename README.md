# cove-sw — RISC-V CoVE software stack (RDSM + TSM)

[English](README.md) | [简体中文](README.zh-CN.md)

`cove-sw` is the software stack for RISC-V CoVE (Confidential VM Extension)
on systems implementing Smmtt (Supervisor Domains Access Protection):
an M-mode **RDSM** (Root Domain Security Manager) firmware layer and an
HS-mode **TSM** (TEE Security Manager) kernel, developed in one repository
because the SBI protocol between them is private to this stack.

## What the stack does

- **RDSM** (M-mode, linked into the SBI firmware) programs the machine-level
  memory protection tables (`mmpt` CSR / MPT) and sequences TEECALL/TEERET
  context switches between the non-confidential host domain (SDID=0) and the
  confidential domain (SDID=1).
- **TSM** (HS-mode, confidential domain) manages the TVM lifecycle, converts
  physical pages into confidential memory, builds G-stage page tables and
  runs vCPU contexts.
- **test-host** (HS-mode, host domain) is a test VMM that drives the CoVE
  host extension end to end; **test-guest** is a VS-mode payload running
  inside a TVM.

The stack is validated end-to-end on NEMU (Smmtt v0.49 model, `smmtt`
branch) with a RustSBI Prototyper firmware that links the RDSM layer at
compile time behind its `rdsm` cargo feature.

## Repository layout

| Path | Contents |
|---|---|
| `crates/rdsm-abi` | Private RDSM↔TSM ABI, single source of truth: EID/FID constants, `cove-payload` header, platform-info types |
| `crates/rdsm` | RDSM substrate (`no_std`, no dependencies besides `rdsm-abi`): MPT tree, CSR codecs, domain state, trap-safe probing |
| `crates/rdsm-mech` | RDSM mechanism layer — all `unsafe` hardware access lives here (public API is safe): raw-asm CSR save/restore, `mmpt`/interrupt-domain programming, image loading, per-hart storage |
| `crates/rdsm-policy` | RDSM policy layer — `unsafe`-free sequencing: boot init, private SBI dispatch (RDSM / SUPD / COVH / COVI), TEECALL/TEERET |
| `tsm` | HS-mode TSM kernel |
| `test-host` | HS-mode test VMM (host domain) |
| `test-guest` | VS-mode guest payload (TVM workload) |
| `xtask` | `cargo xtask pack`: packs TSM @ 0x80400000 and test-host @ 0x80800000 into `cove-payload.bin` |
| `vendor/` | Vendored dependencies (`data-model`, `riscv-page-tables`, … from the Rivos `salus` series) |
| `tests/e2e` | End-to-end regression script |
| `docs/rdsm-tsm-protocol.md` | RDSM↔TSM interface and boot protocol specification |

## Full-stack source layout

The firmware build uses path dependencies, so the three repositories must
be checked out side by side:

```
<root>/
├── NEMU/      github.com/x402/NEMU, branch smmtt   — Smmtt v0.49 hardware model
├── rustsbi/   github.com/x402/rustsbi              — Prototyper firmware (rdsm feature)
└── cove-sw/   this repository
```

## Build

Rust nightly toolchain pinned in `rust-toolchain.toml`; target
`riscv64gc-unknown-none-elf`.

```bash
cargo xtask pack
# → target/riscv64gc-unknown-none-elf/release/cove-payload.bin
cargo check -p tsm -p test-host -p test-guest --target riscv64gc-unknown-none-elf
```

## End-to-end regression

```bash
tests/e2e/run_e2e.sh
```

The script packs the payload, builds the Prototyper firmware with the `rdsm`
feature, boots NEMU and verifies 17 ordered serial markers covering the
complete TEECALL/TEERET lifecycle: RDSM init → TSM ready → host extension
discovery → confidential page conversion → TVM create / finalize / run →
guest exit → TVM teardown → page reclaim.

## Specification baseline

| Specification | Version |
|---|---|
| Smmtt (Supervisor Domains Access Protection) | v0.49 draft |
| CoVE ABI | v0.7 |
| RISC-V SBI | v3.0 |

## Status

Experimental research stack under active development. Global state is
implemented multi-hart-safe (per-hart state); run-time validation currently
covers a single hart on NEMU.
