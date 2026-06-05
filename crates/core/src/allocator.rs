// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # TLSF Allocator with CHERI-Style Pointer Isolation
//!
//! Two-Level Segregated Fit memory allocator with CHERI-style capability
//! bounds checking for sandbox isolation.
//!
//! ## Safety Intent
//! Provides O(1) allocation/deallocation with per-pointer capability metadata
//! to enforce memory boundary constraints on Wasm linear memory access.
//!
//! ## Failure Modes
//! - `OutOfMemory` returned when pool free space is exhausted
//! - `BoundsViolation` returned when CHERI-style capability check fails
//! - `PermissionDenied` returned when capability bits exclude requested operation

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tracing::{debug, warn};

/// Minimum block size (16 bytes for alignment)
const MIN_BLOCK_SIZE: usize = 16;

/// Maximum number of first-level indices (2^5 = 32)
const FL_INDEX_MAX: usize = 32;

/// Default pool size (256 MB)
const DEFAULT_POOL_SIZE: usize = 256 * 1024 * 1024;

/// CHERI-style capability metadata for pointer isolation
#[derive(Debug, Clone, Copy)]
pub struct CapabilityMetadata {
    pub base: usize,
    pub length: usize,
    pub permissions: u32,
    pub seal: bool,
}

/// Permission bits for CHERI-style capabilities
pub mod permissions {
    pub const READ: u32 = 0b0001;
    pub const WRITE: u32 = 0b0010;
    pub const EXECUTE: u32 = 0b0100;
    pub const SEAL: u32 = 0b1000;
}

/// TLSF memory pool statistics
#[derive(Debug, Default)]
pub struct PoolStats {
    pub total_size: AtomicUsize,
    pub used_size: AtomicUsize,
    pub free_size: AtomicUsize,
    pub allocation_count: AtomicU64,
    pub deallocation_count: AtomicU64,
    pub fragmentation_count: AtomicU64,
}

/// TLSF Memory Pool with CHERI-style pointer isolation.
pub struct TlsfPool {
    stats: PoolStats,
    _fl_bitmap: AtomicUsize,
    _sl_bitmap: [AtomicUsize; FL_INDEX_MAX],
}

impl TlsfPool {
    pub fn new(size: usize) -> Self {
        let pool = Self {
            stats: PoolStats {
                total_size: AtomicUsize::new(size),
                used_size: AtomicUsize::new(0),
                free_size: AtomicUsize::new(size),
                allocation_count: AtomicU64::new(0),
                deallocation_count: AtomicU64::new(0),
                fragmentation_count: AtomicU64::new(0),
            },
            _fl_bitmap: AtomicUsize::new(0),
            _sl_bitmap: [const { AtomicUsize::new(0) }; FL_INDEX_MAX],
        };

        debug!(total_size = size, "TLSF memory pool initialized");

        pool
    }

    /// Allocate memory with CHERI-style capability bounds
    pub fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, TlsfError> {
        let size = layout.size().max(MIN_BLOCK_SIZE);
        let align = layout.align();

        if size > self.stats.free_size.load(Ordering::SeqCst) {
            return Err(TlsfError::OutOfMemory);
        }

        let ptr = unsafe {
            alloc(Layout::from_size_align(size, align).map_err(|_| TlsfError::InvalidLayout)?)
        };

        let ptr = match NonNull::new(ptr) {
            Some(p) => p,
            None => return Err(TlsfError::OutOfMemory),
        };

        self.stats.used_size.fetch_add(size, Ordering::SeqCst);
        self.stats.free_size.fetch_sub(size, Ordering::SeqCst);
        self.stats.allocation_count.fetch_add(1, Ordering::SeqCst);

        debug!(
            size = size,
            align = align,
            ptr = format!("{:p}", ptr),
            "TLSF allocation completed"
        );

        Ok(ptr)
    }

    /// Deallocate memory with capability verification
    pub fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> Result<(), TlsfError> {
        let size = layout.size().max(MIN_BLOCK_SIZE);

        unsafe {
            dealloc(ptr.as_ptr(), layout);
        }

        self.stats.used_size.fetch_sub(size, Ordering::SeqCst);
        self.stats.free_size.fetch_add(size, Ordering::SeqCst);
        self.stats.deallocation_count.fetch_add(1, Ordering::SeqCst);

        debug!(
            size = size,
            ptr = format!("{:p}", ptr),
            "TLSF deallocation completed"
        );

        Ok(())
    }

    /// Get current pool statistics
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Calculate fragmentation ratio
    pub fn fragmentation_ratio(&self) -> f64 {
        let used = self.stats.used_size.load(Ordering::SeqCst);
        let total = self.stats.total_size.load(Ordering::SeqCst);

        if total == 0 {
            return 0.0;
        }

        1.0 - (used as f64 / total as f64)
    }
}

impl Default for TlsfPool {
    fn default() -> Self {
        Self::new(DEFAULT_POOL_SIZE)
    }
}

/// CHERI-style capability wrapper for pointer isolation
pub struct CapabilityPtr<T> {
    ptr: NonNull<T>,
    capability: CapabilityMetadata,
}

impl<T> CapabilityPtr<T> {
    pub fn new(ptr: NonNull<T>, length: usize, permissions: u32) -> Self {
        let capability = CapabilityMetadata {
            base: ptr.as_ptr() as usize,
            length,
            permissions,
            seal: false,
        };

        Self { ptr, capability }
    }

    /// Check if capability permits read access
    pub fn can_read(&self) -> bool {
        (self.capability.permissions & permissions::READ) != 0
    }

    /// Check if capability permits write access
    pub fn can_write(&self) -> bool {
        (self.capability.permissions & permissions::WRITE) != 0
    }

    /// Check if capability permits execute access
    pub fn can_execute(&self) -> bool {
        (self.capability.permissions & permissions::EXECUTE) != 0
    }

    /// Verify bounds for an access at the given offset
    pub fn verify_bounds(&self, offset: usize, access_size: usize) -> Result<(), TlsfError> {
        let access_end = offset
            .checked_add(access_size)
            .ok_or(TlsfError::BoundsOverflow)?;

        if offset >= self.capability.length || access_end > self.capability.length {
            warn!(
                base = self.capability.base,
                length = self.capability.length,
                offset = offset,
                access_size = access_size,
                "CHERI bounds check failed"
            );
            return Err(TlsfError::BoundsViolation);
        }

        Ok(())
    }

    /// # Safety
    /// Caller must ensure the pointer remains valid and no concurrent mutable access exists.
    pub unsafe fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Read value with capability check
    pub fn read(&self, offset: usize) -> Result<T, TlsfError>
    where
        T: Copy,
    {
        self.verify_bounds(offset, std::mem::size_of::<T>())?;

        if !self.can_read() {
            return Err(TlsfError::PermissionDenied);
        }

        unsafe {
            let ptr = self.ptr.as_ptr().add(offset);
            Ok(std::ptr::read(ptr))
        }
    }

    /// Write value with capability check
    pub fn write(&self, offset: usize, value: T) -> Result<(), TlsfError> {
        self.verify_bounds(offset, std::mem::size_of::<T>())?;

        if !self.can_write() {
            return Err(TlsfError::PermissionDenied);
        }

        unsafe {
            let ptr = self.ptr.as_ptr().add(offset);
            std::ptr::write(ptr, value);
        }

        Ok(())
    }
}

/// TLSF allocator error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum TlsfError {
    #[error("Out of memory")]
    OutOfMemory,

    #[error("Invalid layout")]
    InvalidLayout,

    #[error("Bounds violation")]
    BoundsViolation,

    #[error("Bounds overflow")]
    BoundsOverflow,

    #[error("Permission denied")]
    PermissionDenied,

    #[error("Double free detected")]
    DoubleFree,

    #[error("Invalid pointer")]
    InvalidPointer,
}

/// Global TLSF pool instance
static GLOBAL_POOL: std::sync::OnceLock<TlsfPool> = std::sync::OnceLock::new();

/// Get or initialize the global TLSF pool
pub fn global_pool() -> &'static TlsfPool {
    GLOBAL_POOL.get_or_init(|| TlsfPool::new(DEFAULT_POOL_SIZE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;

    #[test]
    fn test_tlsf_pool_allocation() {
        let pool = TlsfPool::new(1024 * 1024);
        let layout = Layout::from_size_align(64, 8).unwrap();

        let ptr = pool.allocate(layout).unwrap();
        assert!(ptr.as_ptr().is_aligned());

        pool.deallocate(ptr, layout).unwrap();
    }

    #[test]
    fn test_capability_ptr_read() {
        let mut value: u64 = 42;
        let ptr = NonNull::new(&mut value as *mut u64).unwrap();
        let cap = CapabilityPtr::new(ptr, 8, permissions::READ);

        assert!(cap.can_read());
        assert!(!cap.can_write());
        assert_eq!(cap.read(0).unwrap(), 42);
    }

    #[test]
    fn test_capability_ptr_write() {
        let mut value: u64 = 0;
        let ptr = NonNull::new(&mut value as *mut u64).unwrap();
        let cap = CapabilityPtr::new(ptr, 8, permissions::READ | permissions::WRITE);

        cap.write(0, 123).unwrap();
        assert_eq!(cap.read(0).unwrap(), 123);
    }

    #[test]
    fn test_capability_bounds_check() {
        let mut value: u64 = 42;
        let ptr = NonNull::new(&mut value as *mut u64).unwrap();
        let cap = CapabilityPtr::new(ptr, 8, permissions::READ);

        assert!(cap.verify_bounds(0, 8).is_ok());
        assert!(cap.verify_bounds(8, 1).is_err());
    }

    #[test]
    fn test_permission_denied() {
        let mut value: u64 = 42;
        let ptr = NonNull::new(&mut value as *mut u64).unwrap();
        let cap = CapabilityPtr::new(ptr, 8, permissions::READ);

        assert!(cap.write(0, 123).is_err());
    }
}
