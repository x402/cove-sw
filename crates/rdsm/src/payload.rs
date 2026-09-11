//! CoVE payload header definitions.
//!
//! The single source of truth lives in [`rdsm_abi`]; this module re-exports
//! it so existing `rdsm::payload::…` paths keep working.

pub use rdsm_abi::{COVE_PAYLOAD_MAGIC, COVE_PAYLOAD_VERSION, PayloadHeader};
