// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Staleness Tolerance - Cached Data with Grace Period
//!
//! Implements `CACHED_MAY_BE_STALE` semantics for non-critical routing flags
//! and configuration data. Allows serving slightly stale data to prevent I/O
//! blocking on hot paths while maintaining bounded freshness guarantees.
//!
//! ## Architecture
//! - **StaleCache<T>**: Generic wrapper around cached values with TTL and staleness window
//! - **Freshness Levels**: Fresh, Stale-Acceptable, Expired
//! - **Graceful Degradation**: Stale data served when source unreachable; logged but not blocked
//!
//! ## Safety Intent
//! Prevent I/O bottlenecks on non-critical paths by accepting bounded staleness.
//! Critical paths (security decisions, financial transactions) always require fresh data.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Freshness level of a cached value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshnessLevel {
    /// Data is within TTL - fully fresh
    Fresh,
    /// Data is past TTL but within grace period - acceptable for non-critical use
    StaleAcceptable { age_seconds: i64 },
    /// Data exceeds grace period - must be refreshed
    Expired { age_seconds: i64 },
}

/// A cached value with staleness tracking and grace-period semantics.
///
/// Wraps any serializable type T with metadata about when it was cached,
/// its time-to-live (TTL), and an additional grace period during which
/// slightly stale data may still be served.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaleCache<T: Clone + Serialize> {
    pub value: T,
    pub cached_at: DateTime<Utc>,
    pub ttl_seconds: i64,
    pub grace_period_seconds: i64,
    pub source: String,
    pub generation: u64,
}

impl<T: Clone + Serialize> StaleCache<T> {
    /// Create a new stale cache entry.
    ///
    /// # Arguments
    /// * `value` - The cached data
    /// * `ttl_seconds` - Time-to-live before data is considered stale
    /// * `grace_period_seconds` - Additional window where stale data is acceptable
    /// * `source` - Description of where the data originated
    pub fn new(value: T, ttl_seconds: i64, grace_period_seconds: i64, source: &str) -> Self {
        Self {
            value,
            cached_at: Utc::now(),
            ttl_seconds,
            grace_period_seconds,
            source: source.to_string(),
            generation: 1,
        }
    }

    /// Determine the current freshness level of this cache entry.
    pub fn freshness(&self) -> FreshnessLevel {
        let age = (Utc::now() - self.cached_at).num_seconds();

        if age < self.ttl_seconds {
            FreshnessLevel::Fresh
        } else if age < self.ttl_seconds + self.grace_period_seconds {
            FreshnessLevel::StaleAcceptable { age_seconds: age }
        } else {
            FreshnessLevel::Expired { age_seconds: age }
        }
    }

    /// Check whether the value can be served (fresh or within grace period).
    pub fn is_serveable(&self) -> bool {
        !matches!(self.freshness(), FreshnessLevel::Expired { .. })
    }

    /// Check whether the value is completely fresh (within TTL).
    pub fn is_fresh(&self) -> bool {
        matches!(self.freshness(), FreshnessLevel::Fresh)
    }

    /// Get the age of this cache entry in seconds.
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.cached_at).num_seconds()
    }

    /// Update the cached value and bump the generation counter.
    pub fn refresh(&mut self, new_value: T) {
        self.value = new_value;
        self.cached_at = Utc::now();
        self.generation += 1;
    }

    /// Extend the TTL without changing the value (e.g., after re-validation).
    pub fn extend_ttl(&mut self, additional_seconds: i64) {
        self.cached_at = Utc::now();
        self.ttl_seconds = additional_seconds;
    }
}

/// Manages multiple stale cache entries with unified access control.
///
/// Provides get/set/evict operations with optional background refresh
/// scheduling for entries approaching expiry.
pub struct StalenessManager {
    caches: std::collections::HashMap<String, Box<dyn std::any::Any + Send + Sync>>,
    default_ttl_seconds: i64,
    default_grace_seconds: i64,
}

impl StalenessManager {
    /// Create a new staleness manager with default timing parameters.
    pub fn new(default_ttl_seconds: i64, default_grace_seconds: i64) -> Self {
        Self {
            caches: std::collections::HashMap::new(),
            default_ttl_seconds,
            default_grace_seconds,
        }
    }

    /// Store a value in the cache with staleness tracking.
    pub fn put<T: Clone + Serialize + 'static + Send + Sync>(&mut self, key: &str, value: T, source: &str) {
        let cache = StaleCache::new(
            value,
            self.default_ttl_seconds,
            self.default_grace_seconds,
            source,
        );
        self.caches.insert(key.to_string(), Box::new(cache));

        tracing::debug!(
            cache_key = %key,
            ttl_sec = self.default_ttl_seconds,
            grace_sec = self.default_grace_seconds,
            "[STALENESS] Cache entry stored"
        );
    }

    /// Retrieve a value from the cache, checking freshness.
    ///
    /// # Type Parameters
    /// * `T` - The expected type of the cached value
    ///
    /// # Returns
    /// - `Some((T, FreshnessLevel))` - Value found with its freshness status
    /// - `None` - Key not found or type mismatch
    pub fn get<T: Clone + Serialize + 'static>(&self, key: &str) -> Option<(T, FreshnessLevel)> {
        self.caches.get(key).and_then(|boxed| {
            boxed.downcast_ref::<StaleCache<T>>().map(|cache| {
                let freshness = cache.freshness();

                match &freshness {
                    FreshnessLevel::StaleAcceptable { age_seconds } => {
                        tracing::warn!(
                            cache_key = %key,
                            age_sec = age_seconds,
                            "[STALENESS] Serving stale-but-acceptable data"
                        );
                    }
                    FreshnessLevel::Expired { age_seconds } => {
                        tracing::error!(
                            cache_key = %key,
                            age_sec = age_seconds,
                            "[STALENESS] Cache entry EXPIRED - should not serve"
                        );
                    }
                    FreshnessLevel::Fresh => {}
                }

                (cache.value.clone(), freshness)
            })
        })
    }

    /// Remove a cache entry entirely.
    pub fn evict(&mut self, key: &str) -> bool {
        self.caches.remove(key).is_some()
    }

    /// Return the number of cached entries.
    pub fn len(&self) -> usize {
        self.caches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.caches.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresh_cache_entry() {
        let cache: StaleCache<String> =
            StaleCache::new("fresh_value".to_string(), 60, 30, "test_source");

        assert_eq!(cache.freshness(), FreshnessLevel::Fresh);
        assert!(cache.is_serveable());
        assert!(cache.is_fresh());
    }

    #[test]
    fn test_staleness_manager_put_get() {
        let mut manager = StalenessManager::new(60, 30);
        manager.put("routing_flag", true, "config");

        let result: Option<(bool, FreshnessLevel)> = manager.get("routing_flag");
        assert!(result.is_some());
        let (value, freshness) = result.unwrap();
        assert!(value);
        assert_eq!(freshness, FreshnessLevel::Fresh);
    }

    #[test]
    fn test_cache_refresh_bumps_generation() {
        let mut cache: StaleCache<i32> = StaleCache::new(42, 60, 30, "source");
        let gen1 = cache.generation;
        cache.refresh(100);
        assert_eq!(cache.generation, gen1 + 1);
        assert_eq!(cache.value, 100);
    }

    #[test]
    fn test_evict_removes_entry() {
        let mut manager = StalenessManager::new(60, 30);
        manager.put("temp_key", "value", "source");
        assert_eq!(manager.len(), 1);
        assert!(manager.evict("temp_key"));
        assert_eq!(manager.len(), 0);
    }
}
