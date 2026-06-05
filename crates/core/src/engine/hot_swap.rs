// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Bumpless Transfer - Lock-Free Atomic Pointer Hot-Swapping
//!
//! Implements enterprise security compliant module replacement using `arc-swap` for
//! zero-downtime hot-swapping of critical system components.
//!
//! ## Architecture
//! - Lock-free read access via `ArcSwap<Any>`
//! - Atomic pointer swapping for module replacement
//! - Memory reclamation via epoch-based reclamation
//!
//! ## Safety Guarantees
//! - Readers never block writers
//! - Old module remains valid until all readers complete
//! - No use-after-free or data races

use arc_swap::ArcSwap;
use std::any::Any;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

use crate::SIS_FUEL_QUOTA;

/// Global fuel quota for Joule-level consumption monitoring (10 billion units)
/// Re-exported from crate root for single source of truth
pub const GLOBAL_FUEL_QUOTA: u64 = SIS_FUEL_QUOTA;

/// Trait for hot-swappable modules implementing Bumpless Transfer.
///
/// ## Safety Contract
/// Implementors MUST guarantee thread-safety and provide meaningful
/// version information for audit trails.
pub trait HotSwappable: Send + Sync + std::fmt::Debug {
    fn module_name(&self) -> &str;
    fn version(&self) -> &str;
}

/// Atomic container for hot-swappable modules.
///
/// Implements bumpless transfer protocol with zero memory leaks:
/// - Uses ArcSwap for lock-free reads during swap operations
/// - Explicit drop of returned old Arc to reclaim memory
/// - Swap count tracking for audit and monitoring
pub struct HotSwapContainer<T: Send + Sync> {
    inner: ArcSwap<T>,
    module_name: &'static str,
    swap_count: std::sync::atomic::AtomicU64,
}

impl<T: Send + Sync> HotSwapContainer<T> {
    /// Creates a new HotSwapContainer with the given initial module.
    ///
    /// ## Safety
    /// The initial module must be safely shareable across threads (Arc-wrapped).
    pub fn new(initial: T, module_name: &'static str) -> Self {
        Self {
            inner: ArcSwap::new(Arc::new(initial)),
            module_name,
            swap_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Returns an Arc clone of the current module for lock-free read access.
    ///
    /// ## Performance
    /// O(1) lock-free read - no writers can block this operation.
    #[inline]
    pub fn load(&self) -> Arc<T> {
        Arc::clone(&self.inner.load())
    }

    /// Atomically swaps the module, returning the old module Arc for explicit reclamation.
    ///
    /// ## Bumpless Transfer Protocol
    /// 1. Atomic swap via ArcSwap - readers see either old or new, never invalid state
    /// 2. Returns old Arc - caller MUST drop it to reclaim memory
    /// 3. Increment swap count for audit trail
    ///
    /// ## Memory Safety
    /// The old Arc is returned for explicit management. If not dropped,
    /// memory will not be reclaimed. This is intentional for zero-copy
    /// transfer scenarios where the caller needs to process the old module.
    pub fn swap(&self, new_module: T) -> Arc<T> {
        let start = Instant::now();
        let old = self.inner.swap(Arc::new(new_module));
        let count = self
            .swap_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;

        info!(
            module = self.module_name,
            swap_count = count,
            duration_us = start.elapsed().as_micros(),
            "Bumpless Transfer completed"
        );

        old
    }

    /// Atomic swap with immediate memory reclamation (no return).
    ///
    /// ## Use Case
    /// Use when the old module result is not needed - this performs
    /// immediate reclamation without waiting for caller drop.
    #[inline]
    pub fn swap_and_forget(&self, new_module: T) {
        let old = self.swap(new_module);
        drop(old);
    }

    /// Returns the total number of swaps performed.
    pub fn swap_count(&self) -> u64 {
        self.swap_count.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Type-erased hot-swap container for heterogeneous module storage.
///
/// ## Safety Contract
/// Every `downcast<T>` call validates `TypeId` **before** attempting
/// `Box<dyn Any>::downcast_ref`.  A mismatch returns
/// `AbiError::TypeMismatch` instead of panicking the host.
pub struct TypeErasedContainer {
    inner: ArcSwap<Box<dyn Any + Send + Sync>>,
    module_name: &'static str,
}

#[derive(Debug, thiserror::Error)]
pub enum AbiError {
    #[error(
        "Type mismatch in TypeErasedContainer '{container}': expected {expected}, found {found}"
    )]
    TypeMismatch {
        container: String,
        expected: String,
        found: String,
    },
}

impl TypeErasedContainer {
    /// Creates a new TypeErasedContainer with the given initial value.
    pub fn new<T: Any + Send + Sync>(initial: T, module_name: &'static str) -> Self {
        Self {
            inner: ArcSwap::new(Arc::new(Box::new(initial))),
            module_name,
        }
    }

    /// Returns an Arc clone of the current boxed value.
    #[inline]
    pub fn load(&self) -> Arc<Box<dyn Any + Send + Sync>> {
        self.inner.load_full()
    }

    /// Downcast the type-erased value to `T` with explicit `TypeId` validation.
    ///
    /// Returns `Err(AbiError::TypeMismatch)` if the stored value's `TypeId`
    /// does not match `TypeId::of::<T>()`.  Never panics.
    pub fn downcast<T: 'static + Send + Sync + Clone>(&self) -> Result<T, AbiError> {
        let boxed = self.inner.load();
        let any = boxed.as_ref() as &dyn Any;

        if any.is::<T>() {
            any.downcast_ref::<T>()
                .cloned()
                .ok_or_else(|| AbiError::TypeMismatch {
                    container: self.module_name.to_string(),
                    expected: std::any::type_name::<T>().to_string(),
                    found: "<unknown>".to_string(),
                })
        } else {
            Err(AbiError::TypeMismatch {
                container: self.module_name.to_string(),
                expected: std::any::type_name::<T>().to_string(),
                found: std::any::type_name_of_val(any).to_string(),
            })
        }
    }

    /// Atomically swaps the boxed value.
    ///
    /// ## Memory Safety
    /// Old value Arc is returned - caller MUST drop to reclaim memory.
    pub fn swap<T: Any + Send + Sync>(&self, new_module: T) -> Arc<Box<dyn Any + Send + Sync>> {
        let start = Instant::now();
        let old = self.inner.swap(Arc::new(Box::new(new_module)));

        info!(
            module = self.module_name,
            duration_us = start.elapsed().as_micros(),
            "Type-erased Bumpless Transfer completed"
        );

        old
    }
}

/// SIS (Safety Instrumented System) Interlock State
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SisInterlockState {
    Normal,
    PreShutdown,
    EmergencyShutdown,
    Locked,
}

/// SIS Interlock Controller for ESD (Emergency Shutdown) logic.
///
/// ## Safety Intent
/// Enforces hard fuel quota limits to prevent runaway computation.
/// Transitions through warning states before emergency shutdown.
pub struct SisInterlock {
    state: std::sync::atomic::AtomicU8,
    fuel_consumed: std::sync::atomic::AtomicU64,
    fuel_limit: u64,
}

impl SisInterlock {
    /// Creates a new SIS Interlock with the specified fuel limit.
    pub fn new(fuel_limit: u64) -> Self {
        Self {
            state: std::sync::atomic::AtomicU8::new(SisInterlockState::Normal as u8),
            fuel_consumed: std::sync::atomic::AtomicU64::new(0),
            fuel_limit,
        }
    }

    /// Consumes fuel and returns the current interlock state.
    ///
    /// ## State Transitions
    /// - Normal → PreShutdown at 90% quota
    /// - PreShutdown → EmergencyShutdown at 100% quota
    pub fn consume_fuel(&self, amount: u64) -> SisInterlockState {
        let total = self
            .fuel_consumed
            .fetch_add(amount, std::sync::atomic::Ordering::SeqCst);
        let new_total = total + amount;

        if new_total >= self.fuel_limit {
            self.trigger_esd();
            SisInterlockState::EmergencyShutdown
        } else if new_total >= self.fuel_limit * 9 / 10 {
            self.transition_to(SisInterlockState::PreShutdown);
            SisInterlockState::PreShutdown
        } else {
            SisInterlockState::Normal
        }
    }

    /// Triggers Emergency Shutdown (ESD) - irreversible safety state.
    pub fn trigger_esd(&self) {
        self.transition_to(SisInterlockState::EmergencyShutdown);
        warn!("SIS ESD triggered - fuel limit exceeded");
    }

    /// Transitions to a new SIS state.
    pub fn transition_to(&self, new_state: SisInterlockState) {
        self.state
            .store(new_state as u8, std::sync::atomic::Ordering::SeqCst);
    }

    /// Returns the current SIS interlock state.
    pub fn current_state(&self) -> SisInterlockState {
        match self.state.load(std::sync::atomic::Ordering::SeqCst) {
            0 => SisInterlockState::Normal,
            1 => SisInterlockState::PreShutdown,
            2 => SisInterlockState::EmergencyShutdown,
            3 => SisInterlockState::Locked,
            _ => SisInterlockState::Normal,
        }
    }

    /// Returns remaining fuel quota.
    pub fn fuel_remaining(&self) -> u64 {
        let consumed = self.fuel_consumed.load(std::sync::atomic::Ordering::SeqCst);
        self.fuel_limit.saturating_sub(consumed)
    }

    /// Resets the SIS interlock to Normal state (for testing only).
    pub fn reset(&self) {
        self.fuel_consumed
            .store(0, std::sync::atomic::Ordering::SeqCst);
        self.transition_to(SisInterlockState::Normal);
    }
}

impl Default for SisInterlock {
    fn default() -> Self {
        Self::new(GLOBAL_FUEL_QUOTA)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hot_swap_container() {
        let container = HotSwapContainer::new(42u64, "test-module");
        assert_eq!(*container.load(), 42);

        let old = container.swap(100u64);
        assert_eq!(*old, 42);
        assert_eq!(*container.load(), 100);
        assert_eq!(container.swap_count(), 1);

        drop(old);
    }

    #[test]
    fn test_hot_swap_container_forget() {
        let container = HotSwapContainer::new(42u64, "test-module");
        assert_eq!(*container.load(), 42);

        container.swap_and_forget(100u64);
        assert_eq!(*container.load(), 100);
    }

    #[test]
    fn test_sis_interlock_normal() {
        let sis = SisInterlock::new(1000);
        assert_eq!(sis.current_state(), SisInterlockState::Normal);
        assert_eq!(sis.consume_fuel(100), SisInterlockState::Normal);
        assert_eq!(sis.fuel_remaining(), 900);
    }

    #[test]
    fn test_sis_interlock_eshutdown() {
        let sis = SisInterlock::new(1000);
        assert_eq!(sis.consume_fuel(1000), SisInterlockState::EmergencyShutdown);
        assert_eq!(sis.current_state(), SisInterlockState::EmergencyShutdown);
    }

    #[test]
    fn test_sis_interlock_preshutdown() {
        let sis = SisInterlock::new(1000);
        assert_eq!(sis.consume_fuel(900), SisInterlockState::PreShutdown);
        assert_eq!(sis.current_state(), SisInterlockState::PreShutdown);
    }
}
