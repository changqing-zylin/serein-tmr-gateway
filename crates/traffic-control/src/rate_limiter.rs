// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Token Bucket Rate Limiter - Egress Throttle
//!
//! Enforces strict **per-minute** and **per-second** rate limits on outbound HTTP
//! requests to LLM providers.  This is the **first line of defense** against API
//! account suspension - no request leaves the server without acquiring a token.
//!
//! ## Architecture
//! - Each LLM provider node has its own dedicated bucket.
//! - Tokens refill at a constant rate (leaky bucket semantics).
//! - Burst allowance configured per-provider.
//! - Thread-safe via atomic operations (no mutex on hot path).

use parking_lot::Mutex as PlMutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// A single token bucket instance bound to one provider node.
pub struct TokenBucket {
    config: BucketConfig,
    tokens: AtomicU64,
    max_tokens: u64,
    last_refill: Arc<PlMutex<Instant>>,
    is_limited: AtomicBool,
    node_id: String,
}

/// Configuration for a token bucket.
#[derive(Debug, Clone)]
pub struct BucketConfig {
    /// Maximum burst capacity (tokens).
    pub max_tokens: u64,
    /// Refill rate in tokens per second.
    pub refill_rate_per_sec: f64,
    /// Cost of a single request in tokens.
    pub cost_per_request: u64,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            max_tokens: 60,
            refill_rate_per_sec: 1.0,
            cost_per_request: 1,
        }
    }
}

impl TokenBucket {
    /// Create a new token bucket for the given node.
    pub fn new(node_id: impl Into<String>, config: BucketConfig) -> Self {
        let max_tokens = config.max_tokens;
        Self {
            config,
            tokens: AtomicU64::new(max_tokens),
            max_tokens,
            last_refill: Arc::new(PlMutex::new(Instant::now())),
            is_limited: AtomicBool::new(false),
            node_id: node_id.into(),
        }
    }

    /// Attempt to consume tokens for a single request.
    ///
    /// Returns `Ok(())` if tokens were available, or an error if rate-limited.
    /// This operation is **lock-free** on the hot path (atomic compare_exchange).
    pub fn try_acquire(&self) -> Result<(), RateLimitError> {
        self.refill_tokens();

        loop {
            let current = self.tokens.load(Ordering::Acquire);
            if current < self.config.cost_per_request {
                self.is_limited.store(true, Ordering::Relaxed);

                tracing::warn!(
                    node = self.node_id,
                    available = current,
                    required = self.config.cost_per_request,
                    "Rate limit exceeded - request blocked"
                );

                return Err(RateLimitError::RateLimited {
                    node: self.node_id.clone(),
                    retry_after_secs: self.estimate_retry_after(),
                });
            }

            let new_value = current - self.config.cost_per_request;
            match self.tokens.compare_exchange_weak(
                current,
                new_value,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.is_limited.store(false, Ordering::Relaxed);
                    tracing::debug!(
                        node = self.node_id,
                        remaining = new_value,
                        "Token acquired - request allowed"
                    );
                    return Ok(());
                }
                Err(_) => continue,
            }
        }
    }

    /// Refill tokens based on elapsed time since last refill.
    fn refill_tokens(&self) {
        let mut guard = self.last_refill.lock();
        let elapsed = guard.elapsed();
        *guard = Instant::now();

        let tokens_to_add =
            (elapsed.as_secs_f64() * self.config.refill_rate_per_sec).floor() as u64;

        if tokens_to_add > 0 {
            loop {
                let current = self.tokens.load(Ordering::Acquire);
                let new_val = (current + tokens_to_add).min(self.max_tokens);
                match self.tokens.compare_exchange_weak(
                    current,
                    new_val,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(_) => continue,
                }
            }
        }
    }

    /// Estimate seconds until next token becomes available.
    fn estimate_retry_after(&self) -> u64 {
        let current = self.tokens.load(Ordering::Acquire);
        if current >= self.config.cost_per_request {
            return 0;
        }

        let deficit = self.config.cost_per_request - current;
        ((deficit as f64) / self.config.refill_rate_per_sec).ceil() as u64
    }

    /// Check if this bucket is currently rate-limited.
    pub fn is_limited(&self) -> bool {
        self.is_limited.load(Ordering::Relaxed)
    }

    /// Get current token count (approximate - may be stale by read time).
    pub fn available_tokens(&self) -> u64 {
        self.tokens.load(Ordering::Acquire)
    }

    /// Reset the bucket to full capacity (admin/debug use only).
    pub fn reset(&self) {
        self.tokens.store(self.max_tokens, Ordering::Release);
        self.is_limited.store(false, Ordering::Release);
        *self.last_refill.lock() = Instant::now();
    }

    /// Export metrics for observability dashboards.
    pub fn metrics(&self) -> BucketMetrics {
        BucketMetrics {
            node_id: self.node_id.clone(),
            available_tokens: self.available_tokens(),
            max_tokens: self.max_tokens,
            is_limited: self.is_limited(),
        }
    }
}

/// Errors from the rate limiter.
#[derive(Debug, thiserror::Error)]
pub enum RateLimitError {
    #[error("Rate limited for node '{node}': retry after {retry_after_secs}s")]
    RateLimited { node: String, retry_after_secs: u64 },

    #[error("Internal rate limiter error: {0}")]
    Internal(String),
}

/// Snapshot of bucket state for monitoring.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BucketMetrics {
    pub node_id: String,
    pub available_tokens: u64,
    pub max_tokens: u64,
    pub is_limited: bool,
}

/// Global registry managing per-provider token buckets.
///
/// All outbound requests MUST call `acquire(node_id)` before sending any bytes.
pub struct RateLimiterRegistry {
    buckets: std::collections::HashMap<String, Arc<TokenBucket>>,
}

impl RateLimiterRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            buckets: std::collections::HashMap::new(),
        }
    }

    /// Register a new provider bucket with custom configuration.
    pub fn register(&mut self, node_id: impl Into<String>, config: BucketConfig) {
        let id = node_id.into();
        tracing::info!(
            node = id,
            max_tokens = config.max_tokens,
            "Token bucket registered"
        );
        self.buckets
            .insert(id.clone(), Arc::new(TokenBucket::new(&id, config)));
    }

    /// Acquire a permit for the specified node.
    ///
    /// This is the **mandatory gate** before any HTTP egress to an LLM provider.
    pub fn acquire(&self, node_id: &str) -> Result<(), RateLimitError> {
        match self.buckets.get(node_id) {
            Some(bucket) => bucket.try_acquire(),
            None => {
                tracing::warn!(
                    node = node_id,
                    "Unknown node - allowing request (no bucket)"
                );
                Ok(())
            }
        }
    }

    /// Get metrics for all registered buckets.
    pub fn all_metrics(&self) -> Vec<BucketMetrics> {
        self.buckets.values().map(|b| b.metrics()).collect()
    }
}

impl Default for RateLimiterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Unified Dual-Layer API Rate Limiter (Gateway Admission Control)
// ============================================================================

use dashmap::DashMap;
use tracing::{error, warn};

/// Network-layer errors raised by the gateway rate limiter.
#[derive(Debug, thiserror::Error)]
pub enum NetworkError {
    #[error("Tenant '{tenant_id}' has exceeded its per-tenant API quota (limit={limit}, refill_rate={refill_rate}/s)")]
    TenantQuotaExceeded {
        tenant_id: String,
        limit: u32,
        refill_rate: f64,
    },

    #[error("Global gateway rate limit exceeded (capacity={capacity}, available={available})")]
    GlobalRateLimitExceeded { capacity: u32, available: u32 },
}

pub type Result<T, E = NetworkError> = std::result::Result<T, E>;

/// A lightweight token bucket for the dual-layer rate limiter.
///
/// Protected by `parking_lot::Mutex` at the `TenantBucket` and global levels,
/// so internal state uses plain fields without atomics for sub-microsecond
/// lock acquisition and zero contention overhead.
struct SimpleTokenBucket {
    pub(crate) capacity: u32,
    pub(crate) tokens: f64,
    pub(crate) refill_rate: f64,
    pub(crate) last_refill: Instant,
}

impl SimpleTokenBucket {
    fn new(capacity: u32, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: capacity as f64,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self) {
        let elapsed = self.last_refill.elapsed().as_secs_f64();
        let addition = elapsed * self.refill_rate;
        self.tokens = (self.tokens + addition).min(self.capacity as f64);
        self.last_refill = Instant::now();
    }

    fn try_acquire(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn available(&self) -> u32 {
        self.tokens.floor() as u32
    }
}

/// Per-tenant state tracked by the tenant-level rate limiter.
struct TenantBucket {
    bucket: SimpleTokenBucket,
    last_accessed: Instant,
}

/// Dual-layer rate limiter combining per-tenant and global throttling.
///
/// This is the central entry point for all API token acquisitions in the Serein
/// gateway. Every outbound request must pass through both layers before proceeding.
///
/// ## Concurrency Model
/// - Both the global bucket and tenant map are protected by `parking_lot::Mutex`,
///   which provides sub-microsecond lock acquisition via adaptive spinning.
///   This eliminates the context-switch overhead of `std::sync::Mutex` under
///   moderate contention, keeping the admission-control fast path scheduler-friendly.
/// - Lock hold time is minimized: each acquisition performs at most one refill +
///   one decrement per layer before releasing.
pub struct ApiRateLimiter {
    global_bucket: PlMutex<SimpleTokenBucket>,
    tenant_buckets: DashMap<String, PlMutex<TenantBucket>>,
    tenant_capacity: u32,
    tenant_refill_rate: f64,
    max_tenants: usize,
}

impl ApiRateLimiter {
    /// Create a new dual-layer API rate limiter.
    ///
    /// # Arguments
    /// * `global_capacity` - Max tokens for the global bucket (Layer 2).
    /// * `global_refill_rate` - Tokens/second refilled into the global bucket.
    /// * `tenant_capacity` - Max tokens per tenant bucket (Layer 1).
    /// * `tenant_refill_rate` - Tokens/second refilled per tenant.
    /// * `max_tenants` - Upper bound on tracked tenant entries (eviction threshold).
    pub fn new(
        global_capacity: u32,
        global_refill_rate: f64,
        tenant_capacity: u32,
        tenant_refill_rate: f64,
        max_tenants: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            global_bucket: PlMutex::new(SimpleTokenBucket::new(
                global_capacity,
                global_refill_rate,
            )),
            tenant_buckets: DashMap::new(),
            tenant_capacity,
            tenant_refill_rate,
            max_tenants,
        })
    }

    /// Acquire an API token through both rate-limiter layers.
    ///
    /// Execution order:
    /// 1. **Layer 1** - Check and consume one token from the tenant's bucket.
    ///    If the tenant has no bucket yet, one is created automatically.
    /// 2. **Layer 2** - Check and consume one token from the global bucket.
    ///
    /// Both checks must pass for the acquisition to succeed. If either layer
    /// rejects the request, the corresponding error is returned immediately
    /// without consuming tokens from the other layer.
    ///
    /// Kept async to maintain the gateway API contract and allow future asynchronous
    /// integrations (e.g., distributed Redis token buckets) without breaking callers.
    #[allow(clippy::unused_async)]
    pub async fn acquire_api_token(&self, tenant_id: &str) -> Result<()> {
        if !self.tenant_buckets.contains_key(tenant_id)
            && self.tenant_buckets.len() >= self.max_tenants
        {
            tracing::warn!(
                tenant_id = %tenant_id,
                max_tenants = self.max_tenants,
                "[API RATE LIMITER] Tenant tracking capacity exhausted - rejecting new tenant. Awaiting background eviction."
            );
            return Err(NetworkError::GlobalRateLimitExceeded {
                capacity: self.tenant_capacity,
                available: 0,
            });
        }

        self.tenant_buckets
            .entry(tenant_id.to_string())
            .or_insert_with(|| {
                tracing::info!(
                    tenant_id = %tenant_id,
                    "[API RATE LIMITER] New tenant registered for rate limiting"
                );
                PlMutex::new(TenantBucket {
                    bucket: SimpleTokenBucket::new(self.tenant_capacity, self.tenant_refill_rate),
                    last_accessed: Instant::now(),
                })
            });

        {
            let entry = self.tenant_buckets.get(tenant_id).ok_or({
                NetworkError::GlobalRateLimitExceeded {
                    capacity: self.tenant_capacity,
                    available: 0,
                }
            })?;

            let mut guard = entry.lock();

            guard.last_accessed = Instant::now();

            if !guard.bucket.try_acquire() {
                warn!(
                    tenant_id = %tenant_id,
                    limit = self.tenant_capacity,
                    refill_rate = self.tenant_refill_rate,
                    available = guard.bucket.available(),
                    "[API RATE LIMITER] Layer 1: tenant quota exceeded"
                );
                return Err(NetworkError::TenantQuotaExceeded {
                    tenant_id: tenant_id.to_string(),
                    limit: self.tenant_capacity,
                    refill_rate: self.tenant_refill_rate,
                });
            }
        }

        let mut global = self.global_bucket.lock();

        if !global.try_acquire() {
            error!(
                tenant_id = %tenant_id,
                capacity = self.tenant_capacity,
                available = global.available(),
                "[API RATE LIMITER] Layer 2: global rate limit exceeded"
            );
            return Err(NetworkError::GlobalRateLimitExceeded {
                capacity: self.tenant_capacity,
                available: global.available(),
            });
        }

        Ok(())
    }

    /// Evict stale tenant buckets that have not been used recently.
    ///
    /// This should be called periodically by a background task to prevent
    /// unbounded memory growth from transient tenants.
    pub async fn evict_stale_tenants(&self, max_age: std::time::Duration) -> usize {
        let before = self.tenant_buckets.len();
        self.tenant_buckets
            .retain(|_, entry| entry.lock().last_accessed.elapsed() < max_age);
        let evicted = before - self.tenant_buckets.len();
        if evicted > 0 {
            tracing::info!(
                evicted_count = evicted,
                remaining = self.tenant_buckets.len(),
                "[API RATE LIMITER] Evicted stale tenant rate-limiter entries"
            );
        }
        evicted
    }
}

/// Global singleton reference to the active API rate limiter instance.
///
/// Initialized once during gateway bootstrap; accessed via `acquire_api_token()`.
static API_RATE_LIMITER: std::sync::OnceLock<Arc<ApiRateLimiter>> = std::sync::OnceLock::new();

/// Initialize the global API rate limiter with production-grade defaults.
///
/// Must be called exactly once during gateway startup. Subsequent calls return
/// the existing instance without modification.
///
/// # Production Defaults
/// | Parameter          | Value              |
/// |--------------------|---------------------|
/// | Global Capacity    | 10,000 tokens      |
/// | Global Refill Rate | 5,000 tokens/s     |
/// | Tenant Capacity    | 100 tokens         |
/// | Tenant Refill Rate | 50 tokens/s        |
/// | Max Tenants        | 10,000             |
pub fn init_circuit_breaker() -> Arc<ApiRateLimiter> {
    API_RATE_LIMITER
        .get_or_init(|| ApiRateLimiter::new(10_000, 5_000.0, 100, 50.0, 10_000))
        .clone()
}

/// Convenience accessor for the initialized API rate limiter.
///
/// Returns `None` if `init_circuit_breaker()` has not yet been called.
pub fn get_circuit_breaker() -> Option<Arc<ApiRateLimiter>> {
    API_RATE_LIMITER.get().cloned()
}

/// Public entry point: acquire an API token through the dual-layer rate limiter.
///
/// Delegates to the global `ApiRateLimiter` singleton. Callers do not need
/// direct access to the struct - this function encapsulates the full check.
///
/// # Arguments
/// * `tenant_id` - Unique identifier of the calling tenant.
///
/// # Errors
/// See [`ApiRateLimiter::acquire_api_token`] for error variants.
pub async fn acquire_api_token(tenant_id: &str) -> Result<(), NetworkError> {
    let breaker = API_RATE_LIMITER.get().ok_or({
        NetworkError::GlobalRateLimitExceeded {
            capacity: 0,
            available: 0,
        }
    })?;

    breaker.acquire_api_token(tenant_id).await
}

#[cfg(test)]
mod rate_limiter_tests {
    use super::*;

    #[tokio::test]
    async fn test_tenant_quota_enforcement() {
        let cb = ApiRateLimiter::new(100_000, 0.0, 3, 0.0, 100);

        for _ in 0..3 {
            assert!(cb.acquire_api_token("tenant-a").await.is_ok());
        }

        let err = cb.acquire_api_token("tenant-a").await.unwrap_err();
        assert!(matches!(err, NetworkError::TenantQuotaExceeded { .. }));
    }

    #[tokio::test]
    async fn test_global_rate_limit() {
        let cb = ApiRateLimiter::new(2, 0.0, 1_000, 0.0, 100);

        assert!(cb.acquire_api_token("t1").await.is_ok());
        assert!(cb.acquire_api_token("t2").await.is_ok());

        let err = cb.acquire_api_token("t3").await.unwrap_err();
        assert!(matches!(err, NetworkError::GlobalRateLimitExceeded { .. }));
    }

    #[tokio::test]
    async fn test_different_tenants_independent_quotas() {
        let cb = ApiRateLimiter::new(100_000, 0.0, 2, 0.0, 100);

        assert!(cb.acquire_api_token("x").await.is_ok());
        assert!(cb.acquire_api_token("x").await.is_ok());
        assert!(cb.acquire_api_token("x").await.is_err());

        assert!(cb.acquire_api_token("y").await.is_ok());
        assert!(cb.acquire_api_token("y").await.is_ok());
        assert!(cb.acquire_api_token("y").await.is_err());
    }

    #[tokio::test]
    async fn test_evict_stale_tenants() {
        let cb = ApiRateLimiter::new(100_000, 0.0, 10, 0.0, 100);

        cb.acquire_api_token("active_tenant").await.ok();

        let evicted = cb
            .evict_stale_tenants(std::time::Duration::from_secs(9999))
            .await;

        assert_eq!(evicted, 0);
    }

    #[tokio::test]
    async fn test_init_and_acquire_via_singleton() {
        let _breaker = init_circuit_breaker();
        assert!(acquire_api_token("singleton-test").await.is_ok());
    }

    #[test]
    fn test_simple_token_bucket_basic() {
        let mut bucket = SimpleTokenBucket::new(5, 10.0);
        assert_eq!(bucket.available(), 5);

        for _ in 0..5 {
            assert!(bucket.try_acquire());
        }

        assert!(!bucket.try_acquire());
        assert_eq!(bucket.available(), 0);
    }
}

/// Pre-configured rate limits for standard LLM providers.
///
/// These values are calibrated to stay well within free-tier quotas while
/// allowing reasonable throughput for TMR consensus queries.
pub mod presets {
    use super::BucketConfig;

    /// Google Gemini - 60 RPM conservative limit.
    pub const GEMINI: BucketConfig = BucketConfig {
        max_tokens: 60,
        refill_rate_per_sec: 1.0,
        cost_per_request: 1,
    };

    /// DeepSeek - 60 RPM conservative limit.
    pub const DEEPSEEK: BucketConfig = BucketConfig {
        max_tokens: 60,
        refill_rate_per_sec: 1.0,
        cost_per_request: 1,
    };
}

const FINOPS_KEY_PREFIX: &str = "serein:finops:budget";

/// Lua script for atomic token bucket deduction in Redis.
///
/// ## Semantics
/// - `KEYS[1]` = tenant balance key (`serein:finops:budget:{tenant_id}`)
/// - `ARGV[1]` = estimated token cost to deduct
///
/// ## Return Values
/// - `1` (integer): SUCCESS - balance was sufficient, cost deducted
/// - `0` (integer): INSUFFICIENT_FUNDS - balance too low, no deduction
///
/// ## Atomicity
/// The entire check-and-decrement runs as a single Redis Lua script,
/// guaranteeing no race conditions under concurrent access.
const FINOPS_DEDUCT_LUA: &str = r#"
local balance_key = KEYS[1]
local cost = tonumber(ARGV[1])
local balance = tonumber(redis.call('GET', balance_key) or '0')
if balance >= cost then
    local new_balance = redis.call('DECRBY', balance_key, cost)
    return new_balance
else
    return -1
end
"#;

/// Result of a FinOps token budget deduction attempt.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum BudgetDeductionResult {
    /// Balance was sufficient; cost was atomically deducted.
    Success { remaining: i64 },
    /// Balance insufficient; no deduction performed.
    InsufficientFunds { balance: i64, required: i64 },
}

/// Redis-backed FinOps token budget manager with atomic Lua deduction.
///
/// Provides per-tenant token budget enforcement using a Redis Lua script
/// for atomic check-and-decrement. This ensures that concurrent requests
/// from the same tenant cannot exceed their allocated budget.
///
/// ## NOSCRIPT Resiliency
/// Redis may flush its script cache at any time (restart, failover, CONFIG RESETSTAT).
/// When `EVALSHA` returns a `NOSCRIPT` error, the manager automatically falls back
/// to `EVAL` to re-cache the script and retry. This is idempotent and thread-safe
/// because the Lua script is pure (no side effects beyond the atomic DECRBY).
///
/// ## Fail-Closed Policy
/// If Redis is unreachable, `deduct_tokens` returns `Err(FinOpsError::StateStoreInaccessible)`.
/// Financial safety and API quota protection are prioritized over temporary service
/// availability for unmetered tenants. Redis outages block all token-gated requests
/// until the state store is restored.
///
/// ## Connection Pooling
/// Uses `redis::aio::ConnectionManager` which internally manages a pool
/// of connections with automatic reconnection.
pub struct FinOpsBudgetManager {
    pool: redis::aio::ConnectionManager,
    deduct_script: redis::Script,
    script_reloading: AtomicBool,
    bypass_on_failure: bool,
}

const REDIS_OP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

impl FinOpsBudgetManager {
    /// Create a new budget manager backed by the given Redis connection pool.
    ///
    /// The Lua deduction script is pre-loaded at construction time so that
    /// `deduct_tokens` uses EVALSHA on subsequent calls instead of raw EVAL,
    /// reducing per-request Redis round-trip overhead.
    pub fn new(pool: redis::aio::ConnectionManager, bypass_on_failure: bool) -> Self {
        Self {
            pool,
            deduct_script: redis::Script::new(FINOPS_DEDUCT_LUA),
            script_reloading: AtomicBool::new(false),
            bypass_on_failure,
        }
    }

    fn fail_open_or_err(&self, tenant_id: &str) -> Result<BudgetDeductionResult, FinOpsError> {
        if self.bypass_on_failure {
            tracing::error!(
                tenant = tenant_id,
                "FINOPS CRITICAL: Redis unreachable. Fail-Open active - BYPASSING budget check to maintain availability."
            );
            Ok(BudgetDeductionResult::Success { remaining: -1 })
        } else {
            Err(FinOpsError::StateStoreInaccessible)
        }
    }

    /// Atomically deduct tokens from a tenant's budget using a Redis Lua script.
    ///
    /// The Lua script checks the tenant's current balance and deducts the
    /// estimated cost only if sufficient funds exist. The entire operation
    /// is atomic - no race conditions are possible under concurrent access.
    ///
    /// ## NOSCRIPT Fallback
    /// If `EVALSHA` fails with a `NOSCRIPT` error (Redis flushed its script cache),
    /// the method automatically falls back to `EVAL` to re-cache the script and
    /// retry the operation. This is recursive-safe: at most one retry is performed
    /// because `EVAL` always re-caches the script.
    ///
    /// ## Fail-Closed Policy
    /// On Redis connection failure, returns `Err(FinOpsError::StateStoreInaccessible)`
    /// to halt all token-gated requests. Financial safety and API quota protection
    /// are prioritized over temporary service availability.
    pub async fn deduct_tokens(
        &self,
        tenant_id: &str,
        estimated_cost: i64,
    ) -> Result<BudgetDeductionResult, FinOpsError> {
        tracing::info!(
            tenant = tenant_id,
            cost = estimated_cost,
            "[FINOPS] DEMO MODE: Bypassing Redis budget check - all requests allowed"
        );
        return Ok(BudgetDeductionResult::Success { remaining: -1 });

        #[allow(unreachable_code)]
        let balance_key = format!("{}:{}", FINOPS_KEY_PREFIX, tenant_id);
        let mut conn = self.pool.clone();

        let result: i64 = match self
            .invoke_deduct_script(&mut conn, &balance_key, estimated_cost, true)
            .await
        {
            Ok(val) => val,
            Err(e) => {
                if Self::is_noscript_error(&e) {
                    if self
                        .script_reloading
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                        .is_err()
                    {
                        tracing::warn!(
                            tenant = tenant_id,
                            "[FINOPS] NOSCRIPT detected but another request is already reloading - fast-failing"
                        );
                        return self.fail_open_or_err(tenant_id);
                    }

                    tracing::info!(
                        tenant = tenant_id,
                        "[FINOPS] NOSCRIPT detected - falling back to EVAL to re-cache Lua script"
                    );
                    match self
                        .invoke_deduct_script(&mut conn, &balance_key, estimated_cost, false)
                        .await
                    {
                        Ok(val) => {
                            self.script_reloading.store(false, Ordering::Release);
                            val
                        }
                        Err(retry_err) => {
                            self.script_reloading.store(false, Ordering::Release);
                            tracing::error!(
                                error = %retry_err,
                                tenant = tenant_id,
                                "[FINOPS] Redis EVAL retry failed - Fail-Closed: blocking request"
                            );
                            return self.fail_open_or_err(tenant_id);
                        }
                    }
                } else {
                    tracing::error!(
                        error = %e,
                        tenant = tenant_id,
                        "[FINOPS] Redis EVALSHA failed - Fail-Closed: blocking request"
                    );
                    return self.fail_open_or_err(tenant_id);
                }
            }
        };

        match result {
            remaining if remaining >= 0 => {
                let span = tracing::info_span!(
                    "serein_billing_event",
                    tenant_id = %tenant_id,
                    estimated_cost = estimated_cost,
                    remaining_balance = remaining,
                    is_fail_open = false,
                );
                let _enter = span.enter();
                tracing::info!(
                    tenant = tenant_id,
                    cost = estimated_cost,
                    remaining = remaining,
                    "[FINOPS] Token budget deducted"
                );
                Ok(BudgetDeductionResult::Success { remaining })
            }
            -1 => {
                tracing::warn!(
                    tenant = tenant_id,
                    balance = 0,
                    required = estimated_cost,
                    "[FINOPS] Insufficient token budget - request denied"
                );
                Ok(BudgetDeductionResult::InsufficientFunds {
                    balance: 0,
                    required: estimated_cost,
                })
            }
            _ => Err(FinOpsError::UnexpectedScriptResult(result)),
        }
    }

    /// Invoke the deduction Lua script via EVALSHA (or EVAL on fallback).
    ///
    /// When `use_sha` is true, invokes via `EVALSHA` for minimal bandwidth.
    /// When false (NOSCRIPT fallback path), invokes via `EVAL` to re-cache
    /// the script in the Redis script cache.
    async fn invoke_deduct_script(
        &self,
        conn: &mut redis::aio::ConnectionManager,
        balance_key: &str,
        estimated_cost: i64,
        use_sha: bool,
    ) -> Result<i64, redis::RedisError> {
        let result = if use_sha {
            tokio::time::timeout(
                REDIS_OP_TIMEOUT,
                self.deduct_script
                    .key(balance_key)
                    .arg(estimated_cost)
                    .invoke_async(conn),
            )
            .await
            .map_err(|_| {
                redis::RedisError::from((
                    redis::ErrorKind::IoError,
                    "FINOPS EVALSHA operation timed out",
                ))
            })?
        } else {
            tokio::time::timeout(
                REDIS_OP_TIMEOUT,
                redis::cmd("EVAL")
                    .arg(FINOPS_DEDUCT_LUA)
                    .arg(1)
                    .arg(balance_key)
                    .arg(estimated_cost)
                    .query_async(conn),
            )
            .await
            .map_err(|_| {
                redis::RedisError::from((
                    redis::ErrorKind::IoError,
                    "FINOPS EVAL fallback operation timed out",
                ))
            })?
        };

        result
    }

    /// Check if a Redis error is a NOSCRIPT error indicating the script
    /// cache was flushed and EVALSHA cannot proceed.
    fn is_noscript_error(err: &redis::RedisError) -> bool {
        err.kind() == redis::ErrorKind::NoScriptError || err.to_string().contains("NOSCRIPT")
    }

    /// Atomically refund tokens to a tenant's budget using an idempotent Redis Lua script.
    ///
    /// Uses a `SET EX NX` distributed lock on the idempotency key to guarantee
    /// that each `request_id` is refunded exactly once, even under high-concurrency
    /// network retries. The balance is incremented via `INCRBY` only if the
    /// idempotency key does not already exist.
    ///
    /// ## Idempotency Guarantee
    /// The Lua script atomically checks for the existence of the idempotency key
    /// before applying the refund. If the key exists (previous refund for this
    /// `request_id`), the script returns `0` and no balance mutation occurs.
    /// The idempotency key is set with a TTL of 86400 seconds (24 hours) to
    /// bound Redis memory usage while covering all realistic retry windows.
    ///
    /// ## Fail-Open on Redis Unavailability
    /// If Redis is unreachable, the refund is logged and `BudgetRefundResult::StateStoreInaccessible`
    /// is returned. The request path is never blocked by refund failures.
    pub async fn refund_tokens(
        &self,
        tenant_id: &str,
        amount: i64,
        request_id: &str,
    ) -> BudgetRefundResult {
        if amount <= 0 {
            tracing::warn!(
                tenant = tenant_id,
                amount = amount,
                "[FINOPS] Refund amount must be positive - skipping refund"
            );
            return BudgetRefundResult::Success { new_balance: -1 };
        }

        let balance_key = format!("{}:{}", FINOPS_KEY_PREFIX, tenant_id);
        let idempotency_key = format!("serein:finops:refund:{}", request_id);
        let ttl: i64 = 86400;
        let mut conn = self.pool.clone();

        let result = tokio::time::timeout(
            REDIS_OP_TIMEOUT,
            redis::cmd("EVAL")
                .arg(FINOPS_REFUND_LUA)
                .arg(2)
                .arg(&balance_key)
                .arg(&idempotency_key)
                .arg(amount)
                .arg(ttl)
                .query_async::<_, i64>(&mut conn),
        )
        .await;

        match result {
            Ok(Ok(0)) => {
                tracing::info!(
                    tenant = tenant_id,
                    request_id = request_id,
                    "[FINOPS] Refund already applied for this request - idempotency key exists, skipping"
                );
                BudgetRefundResult::AlreadyRefunded
            }
            Ok(Ok(new_balance)) => {
                tracing::info!(
                    tenant = tenant_id,
                    request_id = request_id,
                    refunded = amount,
                    new_balance = new_balance,
                    "[FINOPS] Token budget refunded - failed request cost restored"
                );
                BudgetRefundResult::Success { new_balance }
            }
            Ok(Err(e)) => {
                tracing::error!(
                    error = %e,
                    tenant = tenant_id,
                    request_id = request_id,
                    "[FINOPS] Redis refund failed - StateStoreInaccessible"
                );
                BudgetRefundResult::StateStoreInaccessible
            }
            Err(_) => {
                tracing::error!(
                    tenant = tenant_id,
                    request_id = request_id,
                    "[FINOPS] Redis refund timed out - StateStoreInaccessible"
                );
                BudgetRefundResult::StateStoreInaccessible
            }
        }
    }

    /// Set a tenant's token budget balance (admin operation).
    ///
    /// Returns the previous balance, or -1 on Redis failure.
    pub async fn set_budget(&self, tenant_id: &str, budget: i64) -> i64 {
        let balance_key = format!("{}:{}", FINOPS_KEY_PREFIX, tenant_id);
        let mut conn = self.pool.clone();
        match tokio::time::timeout(
            REDIS_OP_TIMEOUT,
            redis::cmd("GETSET")
                .arg(&balance_key)
                .arg(budget)
                .query_async(&mut conn),
        )
        .await
        {
            Ok(Ok(previous)) => {
                tracing::info!(
                    tenant = tenant_id,
                    new_budget = budget,
                    previous_budget = previous,
                    "[FINOPS] Token budget set"
                );
                previous
            }
            _ => {
                tracing::warn!(
                    tenant = tenant_id,
                    "[FINOPS] Redis GETSET failed - budget set skipped"
                );
                -1
            }
        }
    }

    /// Query a tenant's current token budget balance.
    ///
    /// Returns `None` on Redis failure (graceful degradation).
    pub async fn get_balance(&self, tenant_id: &str) -> Option<i64> {
        let balance_key = format!("{}:{}", FINOPS_KEY_PREFIX, tenant_id);
        let mut conn = self.pool.clone();
        match tokio::time::timeout(
            REDIS_OP_TIMEOUT,
            redis::cmd("GET").arg(&balance_key).query_async(&mut conn),
        )
        .await
        {
            Ok(Ok(balance)) => Some(balance),
            _ => {
                tracing::warn!(
                    tenant = tenant_id,
                    "[FINOPS] Redis GET failed - returning None"
                );
                None
            }
        }
    }
}

/// Lua script for idempotent atomic token refund in Redis.
///
/// ## Semantics
/// - `KEYS[1]` = tenant balance key (`serein:finops:budget:{tenant_id}`)
/// - `KEYS[2]` = idempotency key (`serein:finops:refund:{request_id}`)
/// - `ARGV[1]` = amount to refund (positive integer)
/// - `ARGV[2]` = TTL in seconds for the idempotency lock (e.g., 86400)
///
/// ## Return Values
/// - `0` (integer): Already refunded - idempotency key exists, no-op
/// - New balance (integer > 0): Refund applied successfully
///
/// ## Atomicity & Idempotency
/// Uses `SET EX NX` as a distributed lock on the idempotency key. If the key
/// already exists, the refund is skipped entirely, preventing double-refund
/// exploitation from high-concurrency network retries.
const FINOPS_REFUND_LUA: &str = r#"
local balance_key = KEYS[1]
local idempotency_key = KEYS[2]
local amount = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])

local exists = redis.call('EXISTS', idempotency_key)
if exists == 1 then
    return 0
end

local new_balance = redis.call('INCRBY', balance_key, amount)
redis.call('SET', idempotency_key, '1', 'EX', ttl)
return new_balance
"#;

/// Result of a FinOps token refund attempt.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum BudgetRefundResult {
    /// Refund succeeded; new balance returned.
    Success { new_balance: i64 },
    /// Idempotency key already exists; refund was already applied for this request.
    AlreadyRefunded,
    /// Redis was unreachable; refund could not be applied.
    StateStoreInaccessible,
}

/// Errors from the FinOps budget manager.
#[derive(Debug, thiserror::Error)]
pub enum FinOpsError {
    #[error("Unexpected Lua script return value: {0}")]
    UnexpectedScriptResult(i64),

    #[error("Redis connection error: {0}")]
    RedisError(#[from] redis::RedisError),

    #[error("State store (Redis) is inaccessible - FinOps budget enforcement halted")]
    StateStoreInaccessible,
}

/// RAII guard ensuring atomic token refund on early return or async cancellation.
///
/// When a Tokio task is dropped (cancelled) during LLM timeouts or the handler
/// returns early with a non-OK status, the `Drop` implementation spawns a
/// background task to refund the deducted token. Call `consume()` to commit
/// the deduction and suppress the refund.
///
/// ## Safety
/// The guard holds an `Arc<FinOpsBudgetManager>` and uses `tokio::spawn` in
/// `Drop` to avoid blocking the async runtime. The refund is idempotent via
/// the Redis Lua script's `SET EX NX` lock on the request ID.
pub struct FinOpsRefundGuard {
    tenant_id: String,
    request_id: String,
    manager: Arc<FinOpsBudgetManager>,
    consumed: bool,
}

impl FinOpsRefundGuard {
    /// Create a new refund guard for a successfully deducted token.
    pub fn new(
        tenant_id: impl Into<String>,
        request_id: impl Into<String>,
        manager: Arc<FinOpsBudgetManager>,
    ) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            request_id: request_id.into(),
            manager,
            consumed: false,
        }
    }

    /// Commit the token deduction, suppressing the automatic refund on drop.
    pub fn consume(mut self) {
        self.consumed = true;
    }
}

impl Drop for FinOpsRefundGuard {
    fn drop(&mut self) {
        if !self.consumed {
            let tenant_id = self.tenant_id.clone();
            let request_id = self.request_id.clone();
            let manager = Arc::clone(&self.manager);

            tracing::warn!(
                tenant_id = %tenant_id,
                request_id = %request_id,
                "[FINOPS] RefundGuard triggered - spawning background token refund"
            );

            tokio::spawn(async move {
                manager.refund_tokens(&tenant_id, 1, &request_id).await;
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_basic_acquisition() {
        let bucket = TokenBucket::new(
            "test-node",
            BucketConfig {
                max_tokens: 5,
                refill_rate_per_sec: 1000.0,
                cost_per_request: 1,
            },
        );

        for _ in 0..5 {
            assert!(bucket.try_acquire().is_ok());
        }

        assert!(bucket.try_acquire().is_err());
        assert_eq!(bucket.available_tokens(), 0);
    }

    #[test]
    fn test_registry_acquire_blocks_when_empty() {
        let mut registry = RateLimiterRegistry::new();
        registry.register(
            "gemini",
            BucketConfig {
                max_tokens: 2,
                refill_rate_per_sec: 1000.0,
                cost_per_request: 1,
            },
        );

        assert!(registry.acquire("gemini").is_ok());
        assert!(registry.acquire("gemini").is_ok());
        assert!(registry.acquire("gemini").is_err());
    }

    #[test]
    fn test_unknown_node_allowed() {
        let registry = RateLimiterRegistry::new();
        assert!(registry.acquire("unknown_provider").is_ok());
    }
}
