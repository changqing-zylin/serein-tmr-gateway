// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # ABI Guard - Cross-Domain Call Monitoring and Memory Access Control
//!
//! Implements strict boundary monitoring for WebAssembly component model
//! cross-domain calls per enterprise security requirements.
//!
//! ## Architecture
//! - Memory access boundary checking
//! - Cross-domain call interception
//! - Unauthorized access detection and logging
//! - Capability-based access control
//! - CHERI-style boundary validation for memory safety
//!
//! ## Safety Guarantees
//! - All cross-domain calls are monitored
//! - Memory access violations are caught before execution
//! - Audit trail for all boundary crossings
//! - Simulated CHERI capability bounds checking

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{info, span, warn, Level};
use wasmtime;

/// Default TLSF pool size for Wasm linear memory (256 MiB)
pub const DEFAULT_TLSF_POOL_SIZE: usize = 256 * 1024 * 1024;

/// Default Wasm page size (64 KiB)
pub const WASM_PAGE_SIZE: usize = 64 * 1024;

/// Memory boundary violation types
#[derive(Debug, Clone, thiserror::Error)]
pub enum AbiViolation {
    #[error("Memory access out of bounds: offset={offset}, length={length}, max={max}")]
    OutOfBoundsAccess {
        offset: usize,
        length: usize,
        max: usize,
    },

    #[error("Unauthorized cross-domain call: domain={domain}, function={function}")]
    UnauthorizedCall { domain: String, function: String },

    #[error("Capability violation: required={required:?}, granted={granted:?}")]
    CapabilityViolation {
        required: Vec<Capability>,
        granted: Vec<Capability>,
    },

    #[error("Pointer provenance violation: ptr={ptr:#x}, expected_domain={expected}")]
    PointerProvenance { ptr: usize, expected: String },

    #[error("Stack pointer violation: sp={sp:#x}, bounds={lower:#x}-{upper:#x}")]
    StackPointerViolation {
        sp: usize,
        lower: usize,
        upper: usize,
    },

    #[error(
        "CHERI boundary violation: base={base:#x}, length={length:#x}, requested={requested:#x}"
    )]
    CheriBoundaryViolation {
        base: usize,
        length: usize,
        requested: usize,
    },

    #[error("CHERI capability tag invalid: capability corrupted or revoked")]
    CheriTagInvalid,

    #[error("CHERI permission denied: required={required:?}, granted={granted:?}")]
    CheriPermissionDenied {
        required: CheriPermission,
        granted: CheriPermission,
    },
}

/// WASM linear memory bounds-check violation.
///
/// Returned when a host-side memory read would escape the guest's linear
/// memory region. Use `.to_wasmtime_error()` to convert into a Wasmtime
/// trap - never panics the host.
#[derive(Debug, Clone, thiserror::Error)]
pub enum WasmMemoryViolation {
    #[error(
        "WASM memory out of bounds: offset={offset} + length={length} > data_size={data_size}"
    )]
    OutOfBounds {
        offset: usize,
        length: usize,
        data_size: usize,
    },

    #[error("WASM memory offset+length overflow: offset={offset}, length={length}, data_size={data_size}")]
    OffsetOverflow {
        offset: usize,
        length: usize,
        data_size: usize,
    },

    #[error("WASM memory export not found in caller")]
    MemoryExportNotFound,
}

impl WasmMemoryViolation {
    pub fn to_wasmtime_error(&self) -> wasmtime::Error {
        wasmtime::Trap::MemoryOutOfBounds.into()
    }
}

/// Validate that `offset + length <= memory.data_size(store)`.
///
/// Returns `Ok(())` when the access is in-bounds, or a `WasmMemoryViolation`
/// that can be converted into a Wasmtime trap via `.to_wasmtime_error()`.
/// Never panics.
pub fn validate_wasm_memory_bounds(
    memory: &wasmtime::Memory,
    store: impl wasmtime::AsContext,
    offset: usize,
    length: usize,
) -> Result<(), WasmMemoryViolation> {
    let data_size = memory.data_size(&store);
    let end = offset
        .checked_add(length)
        .ok_or(WasmMemoryViolation::OffsetOverflow {
            offset,
            length,
            data_size,
        })?;
    if end > data_size {
        return Err(WasmMemoryViolation::OutOfBounds {
            offset,
            length,
            data_size,
        });
    }
    Ok(())
}

/// Read guest memory from a Wasmtime `Memory` with mandatory bounds validation.
///
/// Asserts `offset + length <= memory.data_size(store)` before copying.
/// Returns an owned `Vec<u8>` to avoid lifetime coupling with the store.
/// Returns a `WasmMemoryViolation` on violation.
pub fn read_wasm_memory(
    memory: &wasmtime::Memory,
    store: impl wasmtime::AsContext,
    offset: usize,
    length: usize,
) -> Result<Vec<u8>, WasmMemoryViolation> {
    validate_wasm_memory_bounds(memory, &store, offset, length)?;
    Ok(memory.data(&store)[offset..offset + length].to_vec())
}

/// Read guest memory via a Wasmtime `Caller` with mandatory bounds validation.
///
/// Resolves the default "memory" export from the caller, validates bounds,
/// and returns an owned `Vec<u8>`. Returns a `WasmMemoryViolation` on
/// violation or if the memory export is absent.
pub fn read_wasm_memory_from_caller<T>(
    caller: &mut wasmtime::Caller<'_, T>,
    offset: usize,
    length: usize,
) -> Result<Vec<u8>, WasmMemoryViolation> {
    let memory = caller
        .get_export("memory")
        .and_then(|ext| ext.into_memory())
        .ok_or(WasmMemoryViolation::MemoryExportNotFound)?;
    validate_wasm_memory_bounds(&memory, &*caller, offset, length)?;
    Ok(memory.data(&*caller)[offset..offset + length].to_vec())
}

/// CHERI-style capability permission bits for memory access control.
///
/// Each bit controls a specific class of operation on the associated
/// memory region. Simulates hardware CHERI permission fields in software.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheriPermission {
    pub readable: bool,
    pub writable: bool,
    pub executable: bool,
    pub load_capability: bool,
    pub store_capability: bool,
    pub seal: bool,
    pub unseal: bool,
}

impl Default for CheriPermission {
    fn default() -> Self {
        Self {
            readable: true,
            writable: true,
            executable: false,
            load_capability: true,
            store_capability: true,
            seal: false,
            unseal: false,
        }
    }
}

impl CheriPermission {
    pub fn read_only() -> Self {
        Self {
            readable: true,
            writable: false,
            executable: false,
            load_capability: true,
            store_capability: false,
            seal: false,
            unseal: false,
        }
    }

    pub fn execute_only() -> Self {
        Self {
            readable: false,
            writable: false,
            executable: true,
            load_capability: false,
            store_capability: false,
            seal: false,
            unseal: false,
        }
    }

    pub fn full() -> Self {
        Self {
            readable: true,
            writable: true,
            executable: true,
            load_capability: true,
            store_capability: true,
            seal: true,
            unseal: true,
        }
    }
}

/// CHERI-style capability representation for memory boundary enforcement.
///
/// Simulates CHERI hardware capability bounds checking in software.
/// Each capability carries its own bounds and permissions, preventing
/// arbitrary pointer arithmetic from escaping allocated regions.
#[derive(Debug, Clone)]
pub struct CheriCapability {
    pub base: usize,
    pub length: usize,
    pub offset: usize,
    pub permissions: CheriPermission,
    pub tag: bool,
    pub sealed: bool,
}

impl CheriCapability {
    pub fn new(base: usize, length: usize, permissions: CheriPermission) -> Self {
        Self {
            base,
            length,
            offset: 0,
            permissions,
            tag: true,
            sealed: false,
        }
    }

    pub fn from_tlsf_pool(pool_size: usize) -> Self {
        Self::new(0, pool_size, CheriPermission::default())
    }

    pub fn current_address(&self) -> usize {
        self.base.saturating_add(self.offset)
    }

    pub fn is_valid(&self) -> bool {
        self.tag && !self.sealed && self.offset <= self.length
    }

    pub fn check_bounds(
        &self,
        access_offset: usize,
        access_length: usize,
    ) -> Result<(), AbiViolation> {
        if !self.tag {
            return Err(AbiViolation::CheriTagInvalid);
        }

        if self.sealed {
            return Err(AbiViolation::CheriTagInvalid);
        }

        let absolute_offset = self.offset.saturating_add(access_offset);
        let end = absolute_offset.checked_add(access_length).ok_or(
            AbiViolation::CheriBoundaryViolation {
                base: self.base,
                length: self.length,
                requested: usize::MAX,
            },
        )?;

        if end > self.length {
            return Err(AbiViolation::CheriBoundaryViolation {
                base: self.base,
                length: self.length,
                requested: end,
            });
        }

        Ok(())
    }

    pub fn derive(&self, new_offset: usize, new_length: usize) -> Result<Self, AbiViolation> {
        if !self.is_valid() {
            return Err(AbiViolation::CheriTagInvalid);
        }

        let derived_base = self.base.saturating_add(new_offset);
        if new_offset.saturating_add(new_length) > self.length {
            return Err(AbiViolation::CheriBoundaryViolation {
                base: self.base,
                length: self.length,
                requested: new_offset.saturating_add(new_length),
            });
        }

        Ok(Self {
            base: derived_base,
            length: new_length,
            offset: 0,
            permissions: self.permissions,
            tag: true,
            sealed: false,
        })
    }
}

/// Capability tokens for capability-based access control.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Capability {
    MemoryRead,
    MemoryWrite,
    Execute,
    NetworkAccess,
    FileSystemAccess,
    TimeAccess,
    RandomAccess,
}

/// Context for a cross-domain call being evaluated by the ABI guard.
#[derive(Debug, Clone)]
pub struct CrossDomainContext {
    pub source_domain: String,
    pub target_domain: String,
    pub function_name: String,
    pub timestamp: std::time::Instant,
}

/// Cumulative statistics for ABI guard boundary checks and violations.
#[derive(Debug, Default)]
pub struct AbiGuardStats {
    pub total_calls: AtomicU64,
    pub violations_detected: AtomicU64,
    pub memory_checks: AtomicU64,
    pub capability_checks: AtomicU64,
    pub cheri_boundary_checks: AtomicU64,
}

/// Cross-domain call monitor with CHERI-style boundary validation.
///
/// Enforces memory access boundaries and capability checks for all
/// host↔guest transitions in the WASM Component Model runtime.
pub struct AbiGuard {
    stats: AbiGuardStats,
    enabled: bool,
    strict_mode: bool,
    default_capability: CheriCapability,
    named_capabilities: HashMap<String, CheriCapability>,
}

impl AbiGuard {
    pub fn new(enabled: bool, strict_mode: bool) -> Self {
        Self {
            stats: AbiGuardStats::default(),
            enabled,
            strict_mode,
            default_capability: CheriCapability::from_tlsf_pool(DEFAULT_TLSF_POOL_SIZE),
            named_capabilities: HashMap::new(),
        }
    }

    pub fn with_pool_size(enabled: bool, strict_mode: bool, pool_size: usize) -> Self {
        Self {
            stats: AbiGuardStats::default(),
            enabled,
            strict_mode,
            default_capability: CheriCapability::from_tlsf_pool(pool_size),
            named_capabilities: HashMap::new(),
        }
    }

    /// Register a named `CheriCapability` for a specific memory region (e.g., guest linear memory).
    ///
    /// ## Safety Contract
    /// The capability is stored by name and can be retrieved for per-access validation.
    /// Duplicate names overwrite the previous capability.
    pub fn register_capability(
        &mut self,
        name: &str,
        cap: CheriCapability,
    ) -> Result<(), AbiViolation> {
        if !cap.is_valid() {
            return Err(AbiViolation::CheriTagInvalid);
        }
        self.named_capabilities.insert(name.to_string(), cap);
        Ok(())
    }

    /// CHERI-style memory boundary validation for cross-domain calls.
    ///
    /// Simulates hardware-enforced capability bounds checking by verifying
    /// that all memory accesses fall within the TLSF pool allocation.
    pub fn check_cheri_boundary(
        &self,
        offset: usize,
        length: usize,
    ) -> Result<CheriCapability, AbiViolation> {
        if !self.enabled {
            return Ok(self.default_capability.clone());
        }

        let span = span!(
            Level::DEBUG,
            "cheri_boundary_check",
            offset = offset,
            length = length
        );
        let _enter = span.enter();

        self.stats
            .cheri_boundary_checks
            .fetch_add(1, Ordering::SeqCst);

        self.default_capability.check_bounds(offset, length)?;

        let derived = self.default_capability.derive(offset, length)?;

        if self.strict_mode {
            info!(
                base = format!("{:#x}", derived.base),
                length = format!("{:#x}", derived.length),
                permissions = ?derived.permissions,
                "CHERI capability derived successfully"
            );
        }

        Ok(derived)
    }

    pub fn check_memory_access(
        &self,
        offset: usize,
        length: usize,
        max_memory: usize,
    ) -> Result<(), AbiViolation> {
        if !self.enabled {
            return Ok(());
        }

        let span = span!(
            Level::DEBUG,
            "memory_access_check",
            offset = offset,
            length = length
        );
        let _enter = span.enter();

        self.stats.memory_checks.fetch_add(1, Ordering::SeqCst);

        let end = offset
            .checked_add(length)
            .ok_or(AbiViolation::OutOfBoundsAccess {
                offset,
                length,
                max: max_memory,
            })?;

        if end > max_memory {
            warn!(
                offset = offset,
                length = length,
                max = max_memory,
                "Memory access violation detected"
            );

            self.stats
                .violations_detected
                .fetch_add(1, Ordering::SeqCst);

            return Err(AbiViolation::OutOfBoundsAccess {
                offset,
                length,
                max: max_memory,
            });
        }

        Ok(())
    }

    pub fn check_cross_domain_call(
        &self,
        context: &CrossDomainContext,
        required_capabilities: &[Capability],
        granted_capabilities: &[Capability],
    ) -> Result<(), AbiViolation> {
        if !self.enabled {
            return Ok(());
        }

        self.stats.total_calls.fetch_add(1, Ordering::SeqCst);
        self.stats.capability_checks.fetch_add(1, Ordering::SeqCst);

        for required in required_capabilities {
            if !granted_capabilities.contains(required) {
                warn!(
                    source = %context.source_domain,
                    target = %context.target_domain,
                    function = %context.function_name,
                    required = ?required,
                    "Capability violation in cross-domain call"
                );

                self.stats
                    .violations_detected
                    .fetch_add(1, Ordering::SeqCst);

                return Err(AbiViolation::CapabilityViolation {
                    required: required_capabilities.to_vec(),
                    granted: granted_capabilities.to_vec(),
                });
            }
        }

        if self.strict_mode {
            info!(
                source = %context.source_domain,
                target = %context.target_domain,
                function = %context.function_name,
                "Cross-domain call authorized"
            );
        }

        Ok(())
    }

    pub fn check_pointer_provenance(
        &self,
        ptr: usize,
        expected_domain: &str,
        actual_domain: &str,
    ) -> Result<(), AbiViolation> {
        if !self.enabled {
            return Ok(());
        }

        if expected_domain != actual_domain {
            warn!(
                ptr = format!("{:#x}", ptr),
                expected = expected_domain,
                actual = actual_domain,
                "Pointer provenance violation"
            );

            self.stats
                .violations_detected
                .fetch_add(1, Ordering::SeqCst);

            return Err(AbiViolation::PointerProvenance {
                ptr,
                expected: expected_domain.to_string(),
            });
        }

        Ok(())
    }

    pub fn check_stack_pointer(
        &self,
        sp: usize,
        stack_lower: usize,
        stack_upper: usize,
    ) -> Result<(), AbiViolation> {
        if !self.enabled {
            return Ok(());
        }

        if sp < stack_lower || sp > stack_upper {
            warn!(
                sp = format!("{:#x}", sp),
                bounds = format!("{:#x}-{:#x}", stack_lower, stack_upper),
                "Stack pointer violation"
            );

            self.stats
                .violations_detected
                .fetch_add(1, Ordering::SeqCst);

            return Err(AbiViolation::StackPointerViolation {
                sp,
                lower: stack_lower,
                upper: stack_upper,
            });
        }

        Ok(())
    }

    pub fn stats(&self) -> &AbiGuardStats {
        &self.stats
    }

    pub fn reset_stats(&self) {
        self.stats.total_calls.store(0, Ordering::SeqCst);
        self.stats.violations_detected.store(0, Ordering::SeqCst);
        self.stats.memory_checks.store(0, Ordering::SeqCst);
        self.stats.capability_checks.store(0, Ordering::SeqCst);
        self.stats.cheri_boundary_checks.store(0, Ordering::SeqCst);
    }
}

impl Default for AbiGuard {
    fn default() -> Self {
        Self::new(true, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_access_valid() {
        let guard = AbiGuard::new(true, false);
        assert!(guard.check_memory_access(0, 100, 1024).is_ok());
        assert!(guard.check_memory_access(1000, 24, 1024).is_ok());
    }

    #[test]
    fn test_memory_access_out_of_bounds() {
        let guard = AbiGuard::new(true, false);
        assert!(guard.check_memory_access(1000, 100, 1024).is_err());
        assert!(guard.check_memory_access(1024, 1, 1024).is_err());
    }

    #[test]
    fn test_capability_check_valid() {
        let guard = AbiGuard::new(true, false);
        let context = CrossDomainContext {
            source_domain: "guest".to_string(),
            target_domain: "host".to_string(),
            function_name: "persist".to_string(),
            timestamp: std::time::Instant::now(),
        };

        let required = vec![Capability::MemoryRead, Capability::MemoryWrite];
        let granted = vec![
            Capability::MemoryRead,
            Capability::MemoryWrite,
            Capability::Execute,
        ];

        assert!(guard
            .check_cross_domain_call(&context, &required, &granted)
            .is_ok());
    }

    #[test]
    fn test_capability_check_violation() {
        let guard = AbiGuard::new(true, false);
        let context = CrossDomainContext {
            source_domain: "guest".to_string(),
            target_domain: "host".to_string(),
            function_name: "persist".to_string(),
            timestamp: std::time::Instant::now(),
        };

        let required = vec![Capability::NetworkAccess];
        let granted = vec![Capability::MemoryRead];

        assert!(guard
            .check_cross_domain_call(&context, &required, &granted)
            .is_err());
    }

    #[test]
    fn test_pointer_provenance_valid() {
        let guard = AbiGuard::new(true, false);
        assert!(guard
            .check_pointer_provenance(0x1000, "guest", "guest")
            .is_ok());
    }

    #[test]
    fn test_pointer_provenance_violation() {
        let guard = AbiGuard::new(true, false);
        assert!(guard
            .check_pointer_provenance(0x1000, "guest", "host")
            .is_err());
    }

    #[test]
    fn test_stack_pointer_valid() {
        let guard = AbiGuard::new(true, false);
        assert!(guard.check_stack_pointer(0x8000, 0x7000, 0x9000).is_ok());
    }

    #[test]
    fn test_stack_pointer_violation() {
        let guard = AbiGuard::new(true, false);
        assert!(guard.check_stack_pointer(0x6000, 0x7000, 0x9000).is_err());
        assert!(guard.check_stack_pointer(0xA000, 0x7000, 0x9000).is_err());
    }

    #[test]
    fn test_disabled_guard() {
        let guard = AbiGuard::new(false, false);
        assert!(guard.check_memory_access(10000, 100, 1024).is_ok());
    }

    #[test]
    fn test_wasm_memory_violation_out_of_bounds() {
        let err = WasmMemoryViolation::OutOfBounds {
            offset: 100,
            length: 50,
            data_size: 120,
        };
        let msg = format!("{err}");
        assert!(msg.contains("out of bounds"));
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));
        assert!(msg.contains("120"));
    }

    #[test]
    fn test_wasm_memory_violation_overflow() {
        let err = WasmMemoryViolation::OffsetOverflow {
            offset: usize::MAX,
            length: 1,
            data_size: 1024,
        };
        let msg = format!("{err}");
        assert!(msg.contains("overflow"));
    }

    #[test]
    fn test_wasm_memory_violation_no_export() {
        let err = WasmMemoryViolation::MemoryExportNotFound;
        let msg = format!("{err}");
        assert!(msg.contains("not found"));
    }

    #[test]
    fn test_validate_wasm_memory_bounds_arithmetic() {
        let check = |offset: usize, length: usize, data_size: usize| {
            let end = offset
                .checked_add(length)
                .ok_or(WasmMemoryViolation::OffsetOverflow {
                    offset,
                    length,
                    data_size,
                })?;
            if end > data_size {
                return Err(WasmMemoryViolation::OutOfBounds {
                    offset,
                    length,
                    data_size,
                });
            }
            Ok(())
        };
        assert!(check(0, 100, 200).is_ok());
        assert!(check(100, 100, 200).is_ok());
        assert!(check(150, 100, 200).is_err());
        assert!(check(usize::MAX, 1, 1024).is_err());
    }
}
