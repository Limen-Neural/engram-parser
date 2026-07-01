# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Changed

- **License:** switched from GPL-3.0-or-later to dual MIT/Apache-2.0 for maximum adoption and ecosystem health.

### Added

- **GitHub Actions CI** — `fmt`, `clippy`, `build`, and `test` on push/PR to `main`.
- **Boundary documentation** — README scope/ownership section linked to Linear LIM-9.

## [0.1.0] - 2026-06-01

### Added

- Initial release: pure-Rust, zero-dependency GGUF v3 deserializer.
- MoE expert enumeration and per-expert weight extraction (stacked and per-expert conventions).
- Layout-aware dtype handling for F32, F16, BF16, and opaque quant types.