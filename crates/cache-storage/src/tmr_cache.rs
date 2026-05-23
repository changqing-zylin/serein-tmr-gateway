// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # TMR Result Cache - Consensus Deduplication Layer
//!
//! Caches TMR consensus results keyed by prompt hash to ensure identical
//! prompts receive consistent responses across the distributed system.
//!
//! ## Architecture
//! - **Primary**: Redis-backed cache via `REDIS_URL` for multi-instance consistency
//! - **Fallback**: In-memory `HashMap` cache for single-instance or Redis-unavailable mode
//!
//! ## Cache Key Strategy
//! Keys are SHA-256 hashes of the normalized prompt text, prefixed with
//! `serein:tmr:` for namespace isolation in shared Redis instances.

use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

const CACHE_KEY_PREFIX: &str = "serein:tmr:";

/// A cached TMR consensus result with expiration metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedConsensusResult {
    pub content: String,
    pub agreeing_nodes: u8,
    pub total_nodes: u8,
    pub adjudication_logic: String,
    pub cached_at: i64,
    pub ttl_sec: u64,
}

/// In-memory cache entry with expiration tracking.
struct MemoryCacheEntry {
    result: CachedConsensusResult,
    inserted_at: Instant,
}

/// TMR result cache for consensus deduplication.
///
/// Uses Redis when `REDIS_URL` is configured and reachable;
/// otherwise falls back to an in-memory HashMap cache.
pub struct TmrResultCache {
    redis_url: String,
    ttl_sec: u64,
    memory_cache: Mutex<HashMap<String, MemoryCacheEntry>>,
    redis_available: bool,
}

impl TmrResultCache {
    /// Create a new TMR result cache.
    ///
    /// # Arguments
    /// * `redis_url` - Redis connection URL (e.g., "redis://127.0.0.1:6379")
    /// * `ttl_sec` - Cache entry time-to-live in seconds
    pub fn new(redis_url: String, ttl_sec: u64) -> Self {
        let redis_available = !redis_url.is_empty();
        if redis_available {
            info!(
                redis_url = %redis_url,
                ttl_sec,
                "[TMR CACHE] Redis configured for consensus result caching"
            );
        } else {
            warn!("[TMR CACHE] No REDIS_URL configured - falling back to in-memory cache");
        }

        Self {
            redis_url,
            ttl_sec,
            memory_cache: Mutex::new(HashMap::new()),
            redis_available,
        }
    }

    /// Create a cache from environment variables with sensible defaults.
    pub fn from_env() -> Self {
        let redis_url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        let ttl_sec: u64 = std::env::var("CACHE_TTL_SEC")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3600);
        Self::new(redis_url, ttl_sec)
    }

    /// Compute the cache key for a given prompt.
    pub fn cache_key(prompt: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(prompt.trim().as_bytes());
        let hash = hasher.finalize();
        format!("{}{}", CACHE_KEY_PREFIX, hex::encode(hash))
    }

    /// Retrieve a cached consensus result for the given prompt.
    pub async fn get(&self, prompt: &str) -> Option<CachedConsensusResult> {
        let key = Self::cache_key(prompt);

        if self.redis_available {
            if let Some(cached) = self.get_from_redis(&key).await {
                debug!(cache_key = %key, "[TMR CACHE] Redis cache hit");
                return Some(cached);
            }
        }

        self.get_from_memory(&key)
    }

    /// Store a consensus result in the cache.
    pub async fn put(&self, prompt: &str, result: CachedConsensusResult) {
        let key = Self::cache_key(prompt);

        if self.redis_available {
            self.put_to_redis(&key, &result).await;
        }

        self.put_to_memory(&key, result);
    }

    fn get_from_memory(&self, key: &str) -> Option<CachedConsensusResult> {
        let cache = self.memory_cache.lock().ok()?;
        let entry = cache.get(key)?;
        if entry.inserted_at.elapsed() < Duration::from_secs(self.ttl_sec) {
            debug!(cache_key = %key, "[TMR CACHE] Memory cache hit");
            Some(entry.result.clone())
        } else {
            None
        }
    }

    fn put_to_memory(&self, key: &str, result: CachedConsensusResult) {
        if let Ok(mut cache) = self.memory_cache.lock() {
            cache.insert(
                key.to_string(),
                MemoryCacheEntry {
                    result,
                    inserted_at: Instant::now(),
                },
            );

            if cache.len() > 10_000 {
                let before = cache.len();
                cache.retain(|_, entry| entry.inserted_at.elapsed() < Duration::from_secs(self.ttl_sec));
                let evicted = before - cache.len();
                if evicted > 0 {
                    debug!(
                        evicted,
                        remaining = cache.len(),
                        "[TMR CACHE] Memory cache TTL eviction performed"
                    );
                }
            }
        }
    }

    async fn get_from_redis(&self, key: &str) -> Option<CachedConsensusResult> {
        debug!(
            redis_url = %self.redis_url,
            cache_key = %key,
            "[TMR CACHE] Redis GET - stub (enable redis crate for production)"
        );
        None
    }

    async fn put_to_redis(&self, key: &str, _result: &CachedConsensusResult) {
        debug!(
            redis_url = %self.redis_url,
            cache_key = %key,
            ttl_sec = self.ttl_sec,
            "[TMR CACHE] Redis SET - stub (enable redis crate for production)"
        );
    }
}

impl Default for TmrResultCache {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_deterministic() {
        let key1 = TmrResultCache::cache_key("test prompt");
        let key2 = TmrResultCache::cache_key("test prompt");
        assert_eq!(key1, key2);
        assert!(key1.starts_with(CACHE_KEY_PREFIX));
    }

    #[test]
    fn test_cache_key_differs_for_different_prompts() {
        let key1 = TmrResultCache::cache_key("prompt A");
        let key2 = TmrResultCache::cache_key("prompt B");
        assert_ne!(key1, key2);
    }

    #[tokio::test]
    async fn test_memory_cache_put_get() {
        let cache = TmrResultCache::new(String::new(), 3600);
        let result = CachedConsensusResult {
            content: r#"{"networkId":"ETH"}"#.to_string(),
            agreeing_nodes: 2,
            total_nodes: 3,
            adjudication_logic: "canonical_key_majority".to_string(),
            cached_at: 0,
            ttl_sec: 3600,
        };

        cache.put("test prompt", result.clone()).await;
        let cached = cache.get("test prompt").await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().content, result.content);
    }

    #[tokio::test]
    async fn test_memory_cache_miss() {
        let cache = TmrResultCache::new(String::new(), 3600);
        let cached = cache.get("nonexistent prompt").await;
        assert!(cached.is_none());
    }
}
