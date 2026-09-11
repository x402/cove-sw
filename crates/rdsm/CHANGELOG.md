# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial RDSM substrate crate: CSR abstractions (`mmpt`, `msdcfg`,
  `msideip`, `msideie`), instruction wrappers (`MFENCE.PA`, `MINVAL.PA`),
  MPT radix tree management, Supervisor Domain management, interrupt
  domain management, MPT fault decoding, and trap-safe hardware probing.
