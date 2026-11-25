use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-changed=link.ld");
    // The firmware image is placed by `link.ld`; host test binaries link
    // with the platform's default linker script instead.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("none") {
        println!(
            "cargo:rustc-link-arg=-T{}",
            manifest_dir.join("link.ld").display()
        );
    }
}
