use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-changed=link.ld");
    println!(
        "cargo:rustc-link-arg=-T{}",
        manifest_dir.join("link.ld").display()
    );
}
