use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

// Payload format constants come from rdsm-abi (single source of truth
// shared with the RDSM firmware layer).
pub const COVE_MAGIC: u32 = rdsm_abi::COVE_PAYLOAD_MAGIC;
pub const COVE_VERSION: u32 = rdsm_abi::COVE_PAYLOAD_VERSION;

pub const TSM_LOAD_PADDR: u64 = 0x80400000;
pub const TSM_ENTRY_PADDR: u64 = 0x80400000;
pub const HOST_LOAD_PADDR: u64 = 0x80800000;
pub const HOST_ENTRY_PADDR: u64 = 0x80800000;

#[repr(C, align(4096))]
pub struct PayloadHeader {
    pub magic: u32,
    pub version: u32,
    pub tsm_offset: u64,
    pub tsm_size: u64,
    pub tsm_load_paddr: u64,
    pub tsm_entry_paddr: u64,
    pub host_offset: u64,
    pub host_size: u64,
    pub host_load_paddr: u64,
    pub host_entry_paddr: u64,
    pub reserved: [u8; 4024],
}

const _: () = assert!(core::mem::size_of::<PayloadHeader>() == 4096);

impl PayloadHeader {
    pub fn new(tsm_offset: u64, tsm_size: u64, host_offset: u64, host_size: u64) -> Self {
        Self {
            magic: COVE_MAGIC,
            version: COVE_VERSION,
            tsm_offset,
            tsm_size,
            tsm_load_paddr: TSM_LOAD_PADDR,
            tsm_entry_paddr: TSM_ENTRY_PADDR,
            host_offset,
            host_size,
            host_load_paddr: HOST_LOAD_PADDR,
            host_entry_paddr: HOST_ENTRY_PADDR,
            reserved: [0u8; 4024],
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(
                self as *const Self as *const u8,
                core::mem::size_of::<Self>(),
            )
        }
    }
}

fn align_up(val: u64, align: u64) -> u64 {
    (val + align - 1) & !(align - 1)
}

pub fn run_pack(output_path: Option<PathBuf>) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let tsm_root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let target_dir = tsm_root
        .join("target")
        .join("riscv64gc-unknown-none-elf")
        .join("release");

    println!("[xtask] Building test-guest...");
    let status = Command::new("cargo")
        .current_dir(tsm_root)
        .args([
            "build",
            "-p",
            "test-guest",
            "--target",
            "riscv64gc-unknown-none-elf",
            "--release",
        ])
        .status()?;
    if !status.success() {
        return Err("Failed to build test-guest".into());
    }

    let guest_elf = target_dir.join("test-guest");
    let guest_bin = target_dir.join("test-guest.bin");
    println!("[xtask] Converting test-guest to binary...");
    let status = Command::new("rust-objcopy")
        .args([
            "-O",
            "binary",
            guest_elf.to_str().unwrap(),
            guest_bin.to_str().unwrap(),
        ])
        .status()?;
    if !status.success() {
        return Err("Failed to run rust-objcopy for test-guest".into());
    }

    println!("[xtask] Building test-host...");
    let status = Command::new("cargo")
        .current_dir(tsm_root)
        .args([
            "build",
            "-p",
            "test-host",
            "--target",
            "riscv64gc-unknown-none-elf",
            "--release",
        ])
        .status()?;
    if !status.success() {
        return Err("Failed to build test-host".into());
    }

    let host_elf = target_dir.join("test-host");
    let host_bin = target_dir.join("test-host.bin");
    println!("[xtask] Converting test-host to binary...");
    let status = Command::new("rust-objcopy")
        .args([
            "-O",
            "binary",
            host_elf.to_str().unwrap(),
            host_bin.to_str().unwrap(),
        ])
        .status()?;
    if !status.success() {
        return Err("Failed to run rust-objcopy for test-host".into());
    }

    println!("[xtask] Building tsm...");
    let status = Command::new("cargo")
        .current_dir(tsm_root)
        .args([
            "build",
            "-p",
            "tsm",
            "--target",
            "riscv64gc-unknown-none-elf",
            "--release",
        ])
        .status()?;
    if !status.success() {
        return Err("Failed to build tsm".into());
    }

    let tsm_elf = target_dir.join("tsm");
    let tsm_bin = target_dir.join("tsm.bin");
    println!("[xtask] Converting tsm to binary...");
    let status = Command::new("rust-objcopy")
        .args([
            "-O",
            "binary",
            tsm_elf.to_str().unwrap(),
            tsm_bin.to_str().unwrap(),
        ])
        .status()?;
    if !status.success() {
        return Err("Failed to run rust-objcopy for tsm".into());
    }

    let tsm_data = fs::read(&tsm_bin)?;
    let host_data = fs::read(&host_bin)?;

    let tsm_offset = 4096u64;
    let tsm_size = tsm_data.len() as u64;
    let host_offset = align_up(tsm_offset + tsm_size, 4096);
    let host_size = host_data.len() as u64;

    let header = PayloadHeader::new(tsm_offset, tsm_size, host_offset, host_size);

    let out_file_path = output_path.unwrap_or_else(|| target_dir.join("cove-payload.bin"));
    if let Some(parent) = out_file_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut out_file = File::create(&out_file_path)?;
    out_file.write_all(header.as_bytes())?;

    // Write TSM
    out_file.seek(SeekFrom::Start(tsm_offset))?;
    out_file.write_all(&tsm_data)?;

    // Write Host
    out_file.seek(SeekFrom::Start(host_offset))?;
    out_file.write_all(&host_data)?;

    println!(
        "[xtask] Successfully packed cove-payload.bin at {}",
        out_file_path.display()
    );
    println!(
        "- TSM  : offset = 0x{:x}, size = {} B, load = 0x{:x}, entry = 0x{:x}",
        tsm_offset, tsm_size, TSM_LOAD_PADDR, TSM_ENTRY_PADDR
    );
    println!(
        "- Host : offset = 0x{:x}, size = {} B, load = 0x{:x}, entry = 0x{:x}",
        host_offset, host_size, HOST_LOAD_PADDR, HOST_ENTRY_PADDR
    );

    Ok(out_file_path)
}
