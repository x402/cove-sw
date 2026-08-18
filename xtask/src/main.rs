mod pack;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "xtask")]
#[command(about = "TSM workspace management and build tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build TSM and test-host and pack them into cove-payload.bin
    Pack {
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Pack { output } => {
            pack::run_pack(output)?;
        }
    }

    Ok(())
}
