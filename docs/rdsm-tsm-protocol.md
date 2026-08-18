# RDSM <-> TSM Interface and Boot Protocol Specification

## 1. Overview and Architecture

This document defines the interface and control transfer protocol between the **Root Domain Security Manager (RDSM)** operating in M-mode and the **Trusted Security Manager (TSM)** operating in HS-mode in the confidential domain (SDID=1), on a RISC-V platform equipped with Smmtt (Supervisor Domains Access Protection, v0.49) and CoVE (Confidential VM Extension, v0.7).

### Domain Layout

```
                  +-----------------------------------+
                  |   M-mode Firmware: RDSM (RustSBI) |
                  +-----------------+-----------------+
                                    |
          +-------------------------+-------------------------+
          | (SDID = 0, SIDN = 0)                              | (SDID = 1, SIDN = 1)
          v                                                   v
+-----------------------+                           +-------------------+
|  Host Domain (VMM)    |                           | Confidential TSM  |
|  - e.g. test-host     |                           | - HS-mode TSM     |
|  - HS-mode            |                           |   (TVM Manager)   |
|  - MPT_HOST (SDID=0)  |                           | - MPT_CONF(SDID=1)|
+-----------------------+                           +---------+---------+
                                                              |
                                                    +---------v---------+
                                                    | Confidential TVMs |
                                                    | - VS-mode Guests  |
                                                    | - test-guest      |
                                                    +-------------------+
```

---

## 2. Memory Map and Packaging Format

### 2.1 Default Physical Memory Layout

```
+---------------------+ 0x8000_0000 - Prototyper (RDSM) Text/Data/BSS
| RDSM Firmware (2MB) | (Protected by PMP from S/HS access)
+---------------------+ 0x8020_0000 - Payload Staging / Header
| Cove Payload Image  | [PayloadHeader (4KB) | TSM.bin | test-host.bin]
+---------------------+ 0x8040_0000 - TSM Base Address
| TSM Reserved RAM    | (Accessible only in SDID=1; inaccessible to SDID=0)
+---------------------+ 0x8080_0000 - Host Base Address
| Host System Memory  | (Accessible in SDID=0 and SDID=1)
+---------------------+ 0x9000_0000 (or platform RAM end)
```

### 2.2 Standard PayloadHeader (4096 bytes)

```rust
#[repr(C, align(4096))]
pub struct PayloadHeader {
    pub magic: u32,             // 0x434F5645 ("COVE" in ASCII)
    pub version: u32,           // 1
    pub tsm_offset: u64,        // Byte offset of TSM.bin from start of payload
    pub tsm_size: u64,          // Size of TSM.bin in bytes
    pub tsm_load_paddr: u64,    // Physical target address to copy TSM.bin (0x80400000)
    pub tsm_entry_paddr: u64,   // Entry point physical address for TSM (0x80400000)
    pub host_offset: u64,       // Byte offset of host.bin from start of payload
    pub host_size: u64,         // Size of host.bin in bytes
    pub host_load_paddr: u64,   // Physical target address to copy host.bin (0x80800000)
    pub host_entry_paddr: u64,  // Entry point physical address for host (0x80800000)
    pub reserved: [u8; 4024],
}
```

---

## 3. RDSM SBI Extension Specification (`sbi_rdsm`)

- **Extension ID (EID)**: `0x5244534D` (`"RDSM"` in ASCII)

### 3.1 Function IDs (FID)

| FID | Name | Arguments | Returns | Description |
|---|---|---|---|---|
| `0` | `RDSM_GET_INFO` | None | `sbiret { error, value: *const RdsmInfo }` | Returns RDSM capabilities and domain info |
| `1` | `RDSM_MPT_SET` | `a0=target_sdid`, `a1=paddr`, `a2=len`, `a3=xwr_perm` | `sbiret { error, value }` | Updates MPT entry for specified SDID |
| `2` | `RDSM_MFENCE_PA`| `a0=paddr`, `a1=sdid` | `sbiret { error, value }` | Issues `MFENCE.PA` (and multi-hart broadcast) |
| `3` | `RDSM_TEERET` | `a0=reason`, `a1=arg1`, `a2=arg2` | Does not return directly to caller | Transfers control back to RDSM |

### 3.2 TEERET Reasons (`a0`)

- `TSM_SHUTDOWN = 0`: TSM shutdown / halt.
- `TSM_SWITCH = 1`: Context switch / COVH call handling return.
- `TSM_READY = 2`: Initial TSM boot complete; signal RDSM to switch to Host domain (`SDID=0`).

---

## 4. Boot and Domain Switching State Machine

```
[Boot Hart Reset]
       |
       v
[RDSM: set_pmp()]  <-- Protect M-mode code/data and MPT tables from S/HS
       |
       v
[RDSM: rdsm_init()]
  - Build MPT_HOST (SDID=0): exclude TSM range [0x80400000, 0x80800000)
  - Build MPT_CONF (SDID=1): allow full RAM
  - Read PayloadHeader at 0x80200000
  - Copy TSM.bin to 0x80400000, host.bin to 0x80800000
  - Initialize THCS for TSM execution context
       |
       v
[RDSM: Activate SDID=1 (Confidential Domain)]
  - mmpt = {MODE=Smmpt, SDID=1, PPN=MPT_CONF_ROOT}
  - MFENCE.PA x0, x0
  - MRET to TSM Entry (0x80400000)
       |
       v
[TSM (SDID=1): Boot]
  - Clear BSS, setup stack
  - Console: "[TSM] Booting... TSM_READY"
  - ecall RDSM_TEERET(TSM_READY)
       |
       v
[RDSM Trap Handler (EID=0x5244534D, FID=3, a0=TSM_READY)]
  - Save TSM context in THCS
  - Console: "[RDSM] Switching to Host Domain (SDID=0)..."
  - mmpt = {MODE=Smmpt, SDID=0, PPN=MPT_HOST_ROOT}
  - MFENCE.PA x0, x0
  - MRET to Host Entry (0x80800000)
       |
       v
[Host (SDID=0): Boot]
  - Clear BSS, setup stack
  - Console: "[HOST] Booting... HOST_STARTED"
  - Enters main VMM event loop / wait
```
