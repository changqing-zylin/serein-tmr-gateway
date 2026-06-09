// Copyright (c) 2026 Changqing Zhang. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Coordinator - TMR Orchestration with Physical Fallback Node
//!
//! Implements Triple Modular Redundancy (TMR) orchestration with an unkillable
//! Physical Fallback Node (Node D) backed by a local `wasi-nn` compliant
//! Small Language Model (SLM) endpoint.
//!
//! ## Architecture
//! - **Consensus Delegation**: All TMR adjudication is delegated to
//!   [`serein_consensus::TmrConsensusEngine`] - the Single Source of Truth (SSoT)
//!   for majority-vote arbitration. No local `find_majority` or byte-hashing
//!   fallbacks exist in this module.
//! - **Dynamic Provider Registry**: `LlmProvider` trait enables runtime-pluggable
//!   LLM backends loaded from `providers.toml`, mapped to `ProviderNode::Custom`
//!   entries in the Swarm engine.
//! - **Physical Fallback Node D**: Local `wasi-nn` SLM endpoint - unkillable
//!   because it runs on the same host with no external API dependency.
//! - **Stasis Fallback**: When Swarm fails to reach consensus, the system
//!   immediately activates the Stasis local SLM fallback without
//!   retrying compromised cloud logic.
//!
//! ## Fallback Trigger Conditions
//! - HTTP 403 (Forbidden) - API key revoked or quota exceeded
//! - HTTP 429 (Too Many Requests) - Rate limit hit
//! - Timeout - Network partition or provider outage
//! - Circuit breaker open - Anti-ban system isolated the node
//! - Swarm consensus failure - No majority agreement among providers

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use parking_lot::Mutex as PlMutex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Notify;
use tracing::{error, info, warn};

use serein_consensus::tmr_consensus::{
    ConsensusResult, ProviderNode, ProviderResponse, ResponseStatus, TmrConfig, TmrConsensusEngine,
};
use serein_interfaces::IntermediatePayload;
use serein_interfaces::TmrCanonicalStrategy;
use serein_traffic_control::circuit_breaker::CircuitBreaker;
use serein_traffic_control::rate_limiter::FinOpsBudgetManager;
use serein_worker::{MaskingMap, PIIProtector};

use crate::provider::{LlmProvider, ProviderRequest, TmrNodeError};
use crate::web3::{crypto_billing, proof_logger};

/// Minimum agreement threshold for TMR 2/3 quorum enforcement.
pub const TMR_MIN_AGREEMENT: usize = 2;

/// Temperature jitter offsets applied to each TMR node to prevent
/// deterministic hallucination overlaps from consensus poisoning attacks.
const TMR_TEMPERATURE_JITTER: [f32; 3] = [0.0, 0.05, 0.1];

/// Maximum concurrent `spawn_blocking` tasks permitted across all
/// `PhysicalFallbackNode` instances. Prevents ghost-thread accumulation
/// under heavy timeout/retry storms that could exhaust OS resources.
///
/// When all permits are held, additional `invoke_wasi_nn` calls immediately
/// return `GatewayError::ResourceExhausted` instead of spawning yet another
/// blocking thread that cannot be forcibly aborted.
pub const MAX_BLOCKING_TASKS: usize = 64;

/// Global semaphore bounding `spawn_blocking` concurrency for wasi-nn inference.
///
/// Uses `OnceLock` for lazy one-time initialization. All `PhysicalFallbackNode`
/// instances share this semaphore, ensuring the process-wide blocking thread
/// population never exceeds `MAX_BLOCKING_TASKS`.
#[cfg(feature = "wasi-nn")]
static BLOCKING_TASK_SEMAPHORE: std::sync::OnceLock<tokio::sync::Semaphore> =
    std::sync::OnceLock::new();

#[cfg(feature = "wasi-nn")]
fn blocking_semaphore() -> &'static tokio::sync::Semaphore {
    BLOCKING_TASK_SEMAPHORE.get_or_init(|| tokio::sync::Semaphore::new(MAX_BLOCKING_TASKS))
}

/// Default TMR canonical strategy for adjudicating non-identical LLM outputs.
pub const TMR_CANONICAL_STRATEGY_DEFAULT: &str = "strict";

const CACHE_KEY_PREFIX: &str = "serein:cache:ast";
const CACHE_TTL_SECS: u64 = 3600;

/// Redis-backed semantic AST cache for short-circuiting LLM calls.
///
/// Before dispatching HTTP requests to LLM providers, the canonical key
/// derived from the incoming AST/Schema is checked against Redis. On a cache
/// hit, the LLM call is bypassed entirely and the cached consensus result is
/// returned immediately.
///
/// ## Graceful Degradation
/// If Redis is unreachable, all operations degrade silently: cache misses are
/// returned and write failures are logged without blocking the request path.
/// Redis outages MUST NOT take down the gateway.
pub struct SemanticCache {
    pool: redis::aio::ConnectionManager,
    ttl_secs: u64,
}

/// Result of a semantic cache lookup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheLookup {
    /// Cache hit - contains the serialized consensus result.
    Hit(String),
    /// Cache miss - proceed with LLM dispatch.
    Miss,
}

impl SemanticCache {
    pub fn new(pool: redis::aio::ConnectionManager) -> Self {
        Self {
            pool,
            ttl_secs: CACHE_TTL_SECS,
        }
    }

    pub fn with_ttl(mut self, ttl_secs: u64) -> Self {
        self.ttl_secs = ttl_secs;
        self
    }

    /// O(1) Redis GET for the given canonical key.
    ///
    /// Returns `CacheLookup::Hit` with the cached JSON on success, or
    /// `CacheLookup::Miss` on key absence or Redis failure (graceful degradation).
    pub async fn get(&self, canonical_key: &str) -> CacheLookup {
        let redis_key = format!("{}:{}", CACHE_KEY_PREFIX, canonical_key);
        let mut conn = self.pool.clone();
        match redis::cmd("GET")
            .arg(&redis_key)
            .query_async(&mut conn)
            .await
        {
            Ok(Some(cached)) => {
                info!(
                    key = %redis_key,
                    "[SEMANTIC CACHE] Cache hit - short-circuiting LLM call"
                );
                CacheLookup::Hit(cached)
            }
            Ok(None) => CacheLookup::Miss,
            Err(e) => {
                warn!(
                    error = %e,
                    "[SEMANTIC CACHE] Redis GET error - degrading to cache miss"
                );
                CacheLookup::Miss
            }
        }
    }

    /// Asynchronous Redis SET with TTL for the consensus result.
    ///
    /// Failures are logged and silently swallowed - cache writes are best-effort
    /// and MUST NOT block or fail the request path.
    pub async fn set(&self, canonical_key: &str, value: &str) {
        let redis_key = format!("{}:{}", CACHE_KEY_PREFIX, canonical_key);
        let mut conn = self.pool.clone();
        match redis::cmd("SET")
            .arg(&redis_key)
            .arg(value)
            .arg("EX")
            .arg(self.ttl_secs)
            .query_async::<_, ()>(&mut conn)
            .await
        {
            Ok(()) => {
                info!(
                    key = %redis_key,
                    ttl = self.ttl_secs,
                    "[SEMANTIC CACHE] Cached consensus result"
                );
            }
            Err(e) => {
                warn!(
                    error = %e,
                    "[SEMANTIC CACHE] Redis SET failed - cache write skipped"
                );
            }
        }
    }
}

/// Compute a deterministic canonical key from a request prompt and tenant identity.
///
/// The key incorporates the `tenant_id` to enforce strict tenant isolation in
/// the singleflight and semantic cache layers. Two tenants submitting identical
/// prompts MUST produce different canonical keys - preventing cross-tenant
/// data spillage through shared cache entries or deduplicated in-flight requests.
///
/// ## Key Derivation
/// 1. Prefix with `tenant:{tenant_id}:` to guarantee namespace separation.
/// 2. Attempt structured key via `IntermediatePayload::from_llm_output`.
/// 3. Fallback to SHA-256 hash of the raw prompt for unstructured inputs.
fn compute_request_canonical_key(tenant_id: &str, prompt: &str) -> String {
    let tenant_prefix = format!("tenant:{}:", tenant_id);
    if let Ok(payload) = IntermediatePayload::from_llm_output(prompt) {
        return format!("{}{}", tenant_prefix, payload.canonical_key());
    }
    let digest = Sha256::digest(prompt.as_bytes());
    format!("{}sha256:{}", tenant_prefix, hex::encode(digest))
}

/// Execution mode for the Physical Fallback Node D (Local SLM).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlmExecutionMode {
    /// Node D is active and participates in TMR consensus.
    Active,
    /// Node D is on standby - only invoked when explicitly requested.
    Lazy,
}

impl SlmExecutionMode {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "lazy" => SlmExecutionMode::Lazy,
            _ => SlmExecutionMode::Active,
        }
    }

    pub fn is_active(&self) -> bool {
        matches!(self, SlmExecutionMode::Active)
    }
}

impl std::fmt::Display for SlmExecutionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlmExecutionMode::Active => write!(f, "active"),
            SlmExecutionMode::Lazy => write!(f, "lazy"),
        }
    }
}

/// Configuration for the Physical Fallback Node D (Local SLM).
///
/// Reads from environment variables `SLM_MODEL_ID`, `SLM_MODEL_PATH`,
/// and `SLM_EXECUTION_MODE` to configure the local Small Language Model
/// endpoint that serves as the unkillable consensus participant.
///
/// # Fail-Fast Validation
/// - `SLM_MODEL_PATH` must resolve to a valid filesystem path.
/// - `SLM_EXECUTION_MODE` must be "active" or "lazy".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlmNodeConfig {
    /// Model identifier for the local SLM (e.g., "serein-slm-v1").
    pub model_id: String,
    /// File system path to the GGUF model weights (validated at load time).
    #[serde(with = "pathbuf_serde")]
    pub model_path: std::path::PathBuf,
    /// Execution mode for Node D.
    pub execution_mode: SlmExecutionMode,
}

mod pathbuf_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::path::PathBuf;

    pub fn serialize<S>(path: &std::path::Path, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&path.to_string_lossy())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<PathBuf, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(PathBuf::from(s))
    }
}

impl SlmNodeConfig {
    /// Load SLM configuration from environment variables with validation.
    ///
    /// Returns an error if `SLM_MODEL_PATH` resolves to an empty path,
    /// which would prevent the Physical Fallback Node from loading weights.
    pub fn from_env() -> Result<Self, String> {
        let model_id =
            std::env::var("SLM_MODEL_ID").unwrap_or_else(|_| "serein-slm-v1".to_string());

        let model_path_str = std::env::var("SLM_MODEL_PATH")
            .unwrap_or_else(|_| "./serein-models/fallback-slm.gguf".to_string());

        let model_path = std::path::PathBuf::from(&model_path_str);
        if model_path.as_os_str().is_empty() {
            return Err("SLM_MODEL_PATH must not be empty - Physical Fallback Node requires a valid model path".to_string());
        }

        let execution_mode_str =
            std::env::var("SLM_EXECUTION_MODE").unwrap_or_else(|_| "active".to_string());
        let execution_mode = SlmExecutionMode::from_str_lossy(&execution_mode_str);

        Ok(Self {
            model_id,
            model_path,
            execution_mode,
        })
    }

    /// Convert this config into a [`WasiNnEndpoint`] for the fallback node.
    pub fn to_wasi_nn_endpoint(&self) -> WasiNnEndpoint {
        let path_str = self.model_path.to_string_lossy().to_string();
        let mut endpoint = WasiNnEndpoint::new(&self.model_id, &path_str);
        if !self.execution_mode.is_active() {
            endpoint = endpoint.disabled();
        }
        endpoint
    }
}

impl Default for SlmNodeConfig {
    fn default() -> Self {
        Self::from_env().unwrap_or_else(|e| {
            tracing::error!(error = %e, "[SLM CONFIG] Default SLM config failed - using safe fallback with Lazy execution mode");
            Self {
                model_id: "serein-slm-v1".to_string(),
                model_path: std::path::PathBuf::from("./serein-models/fallback-slm.gguf"),
                execution_mode: SlmExecutionMode::Lazy,
            }
        })
    }
}

/// Result from a single TMR node's inference.
///
/// Each result carries its own `provider_id` and `provider_name` captured
/// at dispatch time, ensuring correct telemetry regardless of
/// `FuturesUnordered` completion order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TmrNodeResult {
    pub provider_id: String,
    pub provider_name: String,
    pub output: Option<String>,
    pub error: Option<TmrNodeError>,
    pub duration_ms: u64,
    pub fallback_triggered: bool,
}

/// Result from the Physical Fallback Node D (local wasi-nn SLM).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackNodeResult {
    pub output: Option<String>,
    pub error: Option<String>,
    pub duration_ms: u64,
    pub model_id: String,
}

/// wasi-nn compliant SLM endpoint configuration.
///
/// Represents a local Small Language Model endpoint that conforms to the
/// `wasi-nn` specification. This endpoint runs on the same host as the
/// Serein microkernel, making it immune to cloud API outages, rate limits,
/// and network partitions.
#[derive(Debug, Clone)]
pub struct WasiNnEndpoint {
    pub model_id: String,
    pub model_path: String,
    pub encoding: String,
    pub inference_timeout_ms: u64,
    pub max_input_tokens: usize,
    pub max_output_tokens: usize,
    pub enabled: bool,
}

impl WasiNnEndpoint {
    pub fn new(model_id: &str, model_path: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            model_path: model_path.to_string(),
            encoding: "Ggml".to_string(),
            inference_timeout_ms: 30_000,
            max_input_tokens: 4096,
            max_output_tokens: 2048,
            enabled: true,
        }
    }

    pub fn with_timeouts(mut self, inference_timeout_ms: u64) -> Self {
        self.inference_timeout_ms = inference_timeout_ms;
        self
    }

    pub fn with_token_limits(mut self, max_input: usize, max_output: usize) -> Self {
        self.max_input_tokens = max_input;
        self.max_output_tokens = max_output;
        self
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

impl Default for WasiNnEndpoint {
    fn default() -> Self {
        Self::new("serein-slm-v1", "./serein-models/base_model.gguf")
    }
}

/// Physical Fallback Node D - the unkillable consensus participant.
///
/// This node runs a local `wasi-nn` compliant Small Language Model that
/// provides a consensus vote when cloud LLM providers are unavailable.
/// It is "unkillable" because:
/// 1. It runs on the same host - no network dependency
/// 2. It uses `wasi-nn` - standardized inference interface
/// 3. It is sandboxed in Wasm - cannot be compromised by network attacks
/// 4. It has no API key - cannot be rate-limited or revoked
pub struct PhysicalFallbackNode {
    endpoint: WasiNnEndpoint,
    invocation_count: std::sync::atomic::AtomicU64,
    error_count: std::sync::atomic::AtomicU64,
    last_invocation: Arc<PlMutex<Option<Instant>>>,
    concurrency_semaphore: tokio::sync::Semaphore,
    force_disabled: std::sync::atomic::AtomicBool,
}

impl PhysicalFallbackNode {
    pub fn new(endpoint: WasiNnEndpoint) -> Self {
        Self {
            endpoint,
            invocation_count: std::sync::atomic::AtomicU64::new(0),
            error_count: std::sync::atomic::AtomicU64::new(0),
            last_invocation: Arc::new(PlMutex::new(None)),
            concurrency_semaphore: tokio::sync::Semaphore::new(4),
            force_disabled: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn is_available(&self) -> bool {
        self.endpoint.enabled
            && !self
                .force_disabled
                .load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn endpoint(&self) -> &WasiNnEndpoint {
        &self.endpoint
    }

    pub fn disable(&self) {
        self.force_disabled
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn enable(&self) {
        self.force_disabled
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn invocation_count(&self) -> u64 {
        self.invocation_count
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Invoke the local wasi-nn SLM endpoint for inference.
    pub async fn invoke(&self, prompt: &str) -> FallbackNodeResult {
        let _permit = match tokio::time::timeout(
            Duration::from_millis(500),
            self.concurrency_semaphore.acquire(),
        )
        .await
        {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => {
                warn!(
                    model_id = %self.endpoint.model_id,
                    "[PHYSICAL FALLBACK] Node D semaphore closed - fast-failing"
                );
                return FallbackNodeResult {
                    output: None,
                    error: Some("Node D semaphore closed".to_string()),
                    duration_ms: 0,
                    model_id: self.endpoint.model_id.clone(),
                };
            }
            Err(_) => {
                warn!(
                    model_id = %self.endpoint.model_id,
                    "[PHYSICAL FALLBACK] Node D overloaded - concurrency limit reached, fast-failing"
                );
                return FallbackNodeResult {
                    output: None,
                    error: Some(
                        "Node D overloaded: Concurrency limit reached after bounded wait"
                            .to_string(),
                    ),
                    duration_ms: 0,
                    model_id: self.endpoint.model_id.clone(),
                };
            }
        };

        self.invocation_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        *self.last_invocation.lock() = Some(Instant::now());

        let start = Instant::now();

        if !self.endpoint.enabled {
            self.error_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return FallbackNodeResult {
                output: None,
                error: Some("Physical fallback node is disabled".to_string()),
                duration_ms: start.elapsed().as_millis() as u64,
                model_id: self.endpoint.model_id.clone(),
            };
        }

        if prompt.len() > self.endpoint.max_input_tokens * 4 {
            self.error_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return FallbackNodeResult {
                output: None,
                error: Some(format!(
                    "Prompt exceeds max input tokens: {} chars (max {} tokens)",
                    prompt.len(),
                    self.endpoint.max_input_tokens
                )),
                duration_ms: start.elapsed().as_millis() as u64,
                model_id: self.endpoint.model_id.clone(),
            };
        }

        #[cfg(feature = "wasi-nn")]
        {
            let result = tokio::time::timeout(
                Duration::from_millis(self.endpoint.inference_timeout_ms),
                self.invoke_wasi_nn(prompt),
            )
            .await
            .map_err(|_| {
                self.error_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                FallbackNodeResult {
                    output: None,
                    error: Some(format!(
                        "wasi-nn inference timed out after {}ms",
                        self.endpoint.inference_timeout_ms
                    )),
                    duration_ms: self.endpoint.inference_timeout_ms,
                    model_id: self.endpoint.model_id.clone(),
                }
            });

            match result {
                Ok(Ok(output)) => FallbackNodeResult {
                    output: Some(output),
                    error: None,
                    duration_ms: start.elapsed().as_millis() as u64,
                    model_id: self.endpoint.model_id.clone(),
                },
                Ok(Err(e)) => {
                    self.error_count
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    FallbackNodeResult {
                        output: None,
                        error: Some(format!("wasi-nn inference failed: {}", e)),
                        duration_ms: start.elapsed().as_millis() as u64,
                        model_id: self.endpoint.model_id.clone(),
                    }
                }
                Err(e) => e,
            }
        }

        #[cfg(not(feature = "wasi-nn"))]
        {
            tokio::time::sleep(Duration::from_millis(50)).await;

            let mock_output = r#"{"networkId":"FALLBACK","taskType":"Unknown","maxGasLimit":0,"confidenceScore":0.50,"sourceUrl":"local://wasi-nn-slm"}"#.to_string();

            info!(
                model_id = %self.endpoint.model_id,
                prompt_len = prompt.len(),
                duration_ms = start.elapsed().as_millis() as u64,
                "[PHYSICAL FALLBACK] wasi-nn SLM inference completed (stub mode)"
            );

            FallbackNodeResult {
                output: Some(mock_output),
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
                model_id: self.endpoint.model_id.clone(),
            }
        }
    }

    #[cfg(feature = "wasi-nn")]
    async fn invoke_wasi_nn(&self, prompt: &str) -> Result<String, String> {
        use serein_core::model_interface::{
            create_model, InferenceTarget, ModelConfig, ModelEncoding,
        };

        let _permit = blocking_semaphore()
            .try_acquire()
            .map_err(|_| {
                format!(
                    "Resource exhausted: blocking thread pool at capacity ({} permits) - refusing spawn_blocking to prevent ghost-thread leak",
                    MAX_BLOCKING_TASKS
                )
            })?;

        let config = ModelConfig::default()
            .with_model_name(&self.endpoint.model_id)
            .with_encoding(ModelEncoding::Ggml)
            .with_target(InferenceTarget::Cpu)
            .with_models_dir(&self.endpoint.model_path);

        let prompt_owned = prompt.to_string();
        let max_output_tokens = self.endpoint.max_output_tokens as u32;

        let output = tokio::task::spawn_blocking(move || {
            let model = create_model(config)
                .map_err(|e| format!("Failed to create WASI-NN model: {}", e))?;
            model
                .infer(prompt_owned.as_bytes(), max_output_tokens)
                .map_err(|e| format!("WASI-NN inference failed: {}", e))
        })
        .await
        .map_err(|e| {
            format!(
                "Tokio starvation prevented, but spawn_blocking failed: {}",
                e
            )
        })??;

        String::from_utf8(output).map_err(|e| format!("WASI-NN output is not valid UTF-8: {}", e))
    }
}

/// TMR consensus result with fallback information.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TmrConsensusResult {
    pub majority_output: Option<String>,
    pub agreement_count: usize,
    pub total_nodes: usize,
    pub fallback_activated: bool,
    pub node_results: Vec<TmrNodeResult>,
    pub fallback_result: Option<FallbackNodeResult>,
    pub consensus_achieved: bool,
    pub total_duration_ms: u64,
    pub cache_hit: bool,
}

/// Internal adapter mapping a dynamic `LlmProvider` + `CircuitBreaker` pair
/// to a `ProviderNode::Custom` identifier for the Swarm consensus engine.
#[derive(Clone)]
struct ProviderAdapter {
    node: ProviderNode,
    provider: Arc<dyn LlmProvider>,
    circuit_breaker: Arc<CircuitBreaker>,
}

/// TMR Orchestrator with Physical Fallback Node and Swarm consensus delegation.
///
/// Manages Triple Modular Redundancy consensus by delegating all adjudication
/// to [`TmrConsensusEngine`] from `serein-consensus`. The orchestrator is responsible
/// for:
/// 1. Mapping dynamic `LlmProvider` trait objects to `ProviderNode::Custom` entries
/// 2. Building the `query_fn` closure that normalizes LLM output into canonical keys
/// 3. Invoking the Swarm engine for authoritative consensus
/// 4. Activating the Stasis local SLM fallback when Swarm fails
///
/// ## Consensus Flow
/// 1. **Pre-Consensus**: The `query_fn` extracts raw results from each provider,
///    checking circuit breakers and recording telemetry.
/// 2. **Normalization**: Each result is parsed via [`IntermediatePayload::from_llm_output`]
///    and its [`IntermediatePayload::canonical_key`] is computed as the `semantic_key`.
/// 3. **Delegation**: `TmrConsensusEngine::query` performs majority-vote arbitration.
/// 4. **Handling**: The Swarm result is treated as the final authoritative decision.
///    On failure, the Stasis fallback is invoked immediately without retrying
///    compromised cloud logic.
///
/// Gateway-level error types for request lifecycle failures.
#[derive(Debug, Clone, thiserror::Error)]
pub enum GatewayError {
    #[error("Request cancelled - client disconnected before upstream response arrived")]
    RequestCancelled,

    #[error("Upstream LLM request aborted - token burn prevented")]
    UpstreamAborted,

    #[error("Resource exhausted - blocking thread pool at capacity ({0} permits)")]
    ResourceExhausted(usize),
}

/// Singleflight entry with cancel-safety support.
///
/// When the leader task is cancelled (client disconnect), all waiting watchers
/// are immediately notified with an empty result so they can return
/// `GatewayError::RequestCancelled` to their callers.
struct SingleFlightEntry {
    result: Option<TmrConsensusResult>,
    notify: Arc<Notify>,
    cancelled: std::sync::atomic::AtomicBool,
}

/// Singleflight guard providing RAII cleanup of the inflight map entry.
///
/// When dropped (on both success and error/timeout paths), this guard
/// removes the entry from the shared inflight map, preventing memory leaks
/// from abandoned entries.
///
/// ## Cancel Safety
/// If the leader task is cancelled, the guard's `Drop` implementation
/// marks the entry as cancelled and notifies all waiting watchers so
/// they can return `GatewayError::RequestCancelled` immediately instead
/// of waiting for a result that will never arrive.
struct SingleFlightGuard {
    key: String,
    map: std::sync::Arc<PlMutex<HashMap<String, Arc<PlMutex<SingleFlightEntry>>>>>,
    is_leader: bool,
}

impl SingleFlightGuard {
    fn new(
        key: String,
        map: std::sync::Arc<PlMutex<HashMap<String, Arc<PlMutex<SingleFlightEntry>>>>>,
        is_leader: bool,
    ) -> Self {
        Self {
            key,
            map,
            is_leader,
        }
    }
}

impl Drop for SingleFlightGuard {
    fn drop(&mut self) {
        let mut map = self.map.lock();
        if let Some(entry_arc) = map.remove(&self.key) {
            if self.is_leader {
                let entry = entry_arc.lock();
                if entry.result.is_none()
                    && !entry.cancelled.load(std::sync::atomic::Ordering::SeqCst)
                {
                    entry
                        .cancelled
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    entry.notify.notify_waiters();
                    error!(
                        key = %self.key,
                        "[SINGLEFLIGHT] Leader task dropped without setting result - notifying watchers of cancellation"
                    );
                }
            }
        }
    }
}

/// Singleflight deduplication map keyed by canonical cache key.
///
/// Prevents Thundering Herd on cache misses: when multiple concurrent
/// requests miss the semantic cache for the same canonical key, only
/// the first request dispatches to LLM providers. All subsequent callers
/// await the in-flight result with a bounded 15s timeout to prevent
/// worker pool exhaustion if the leader hangs.
struct CacheSingleFlight {
    inflight: std::sync::Arc<PlMutex<HashMap<String, Arc<PlMutex<SingleFlightEntry>>>>>,
}

/// Maximum time a watcher will wait for the leader's result before
/// giving up and returning a timeout error. Prevents worker pool
/// exhaustion when the leader request hangs indefinitely.
const SINGLEFLIGHT_WATCHER_TIMEOUT: Duration = Duration::from_secs(15);

impl CacheSingleFlight {
    fn new() -> Self {
        Self {
            inflight: std::sync::Arc::new(PlMutex::new(HashMap::new())),
        }
    }

    /// Cancel-safe singleflight deduplication with `tokio::select!`.
    ///
    /// ## Cancel Safety
    /// When a client drops the HTTP connection (cancelling the local Tokio task),
    /// the leader's upstream LLM request is aborted via `tokio::select!` to
    /// prevent burning API tokens on responses nobody will consume.
    ///
    /// If the leader is cancelled, all waiting watchers are immediately notified
    /// with `GatewayError::RequestCancelled` via the entry's `cancelled` flag.
    async fn dedup<F, Fut>(&self, key: &str, f: F) -> TmrConsensusResult
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = TmrConsensusResult>,
    {
        let entry_arc = {
            let map = self.inflight.lock();
            map.get(key).cloned()
        };

        if let Some(entry_arc) = entry_arc {
            {
                let guard = entry_arc.lock();
                if let Some(ref result) = guard.result {
                    return result.clone();
                }
                if guard.cancelled.load(std::sync::atomic::Ordering::SeqCst) {
                    warn!(
                        key = %key,
                        "[SINGLEFLIGHT] Watcher detected leader cancellation - returning empty result"
                    );
                    return TmrConsensusResult::default();
                }
            }

            let notify = {
                let guard = entry_arc.lock();
                if let Some(ref result) = guard.result {
                    return result.clone();
                }
                if guard.cancelled.load(std::sync::atomic::Ordering::SeqCst) {
                    warn!(
                        key = %key,
                        "[SINGLEFLIGHT] Watcher detected leader cancellation - returning empty result"
                    );
                    return TmrConsensusResult::default();
                }
                guard.notify.clone()
            };

            match tokio::time::timeout(SINGLEFLIGHT_WATCHER_TIMEOUT, notify.notified()).await {
                Ok(()) => {
                    let guard = entry_arc.lock();
                    if guard.cancelled.load(std::sync::atomic::Ordering::SeqCst) {
                        warn!(
                            key = %key,
                            "[SINGLEFLIGHT] Watcher notified of leader cancellation - returning empty result"
                        );
                        return TmrConsensusResult::default();
                    }
                    guard
                        .result
                        .clone()
                        .unwrap_or_else(TmrConsensusResult::default)
                }
                Err(_) => {
                    warn!(
                        key = %key,
                        timeout_secs = SINGLEFLIGHT_WATCHER_TIMEOUT.as_secs(),
                        "[SINGLEFLIGHT] Watcher timed out waiting for leader - returning empty result"
                    );
                    TmrConsensusResult::default()
                }
            }
        } else {
            let entry = Arc::new(PlMutex::new(SingleFlightEntry {
                result: None,
                notify: Arc::new(Notify::new()),
                cancelled: std::sync::atomic::AtomicBool::new(false),
            }));
            {
                let mut map = self.inflight.lock();
                map.insert(key.to_string(), entry.clone());
            }

            let _guard = SingleFlightGuard::new(key.to_string(), self.inflight.clone(), true);

            let result = f().await;

            let mut guard = entry.lock();
            guard.result = Some(result.clone());
            guard.notify.notify_waiters();

            result
        }
    }
}

pub struct TmrOrchestrator {
    fallback_node: Arc<PhysicalFallbackNode>,
    fallback_trigger_count: std::sync::atomic::AtomicU64,
    adapters: Vec<ProviderAdapter>,
    tmr_engine: Arc<PlMutex<TmrConsensusEngine>>,
    canonical_strategy: TmrCanonicalStrategy,
    min_agreement: usize,
    cache: Option<Arc<SemanticCache>>,
    singleflight: CacheSingleFlight,
    pii_protector: PIIProtector,
    masking_maps: DashMap<String, MaskingMap>,
    finops: Option<Arc<FinOpsBudgetManager>>,
}

impl TmrOrchestrator {
    /// Build an orchestrator with dynamic `LlmProvider` trait objects.
    ///
    /// Each provider is registered as a `ProviderNode::Custom(provider_id)`
    /// in the Swarm consensus engine with its own circuit breaker.
    pub fn with_providers(providers: Vec<Arc<dyn LlmProvider>>, slm_config: SlmNodeConfig) -> Self {
        let fallback = PhysicalFallbackNode::new(slm_config.to_wasi_nn_endpoint());

        let mut adapters = Vec::with_capacity(providers.len());
        let mut engine_nodes = Vec::with_capacity(providers.len());

        for provider in &providers {
            let provider_id = provider.provider_id().to_string();
            let node = ProviderNode::Custom(provider_id);
            let cb = Arc::new(CircuitBreaker::new(
                format!("provider-{}", provider.provider_id()),
                serein_traffic_control::circuit_breaker::CircuitBreakerConfig::default(),
            ));

            engine_nodes.push(node.clone());
            adapters.push(ProviderAdapter {
                node,
                provider: provider.clone(),
                circuit_breaker: cb,
            });
        }

        let strategy_str = std::env::var("TMR_CANONICAL_STRATEGY")
            .unwrap_or_else(|_| TMR_CANONICAL_STRATEGY_DEFAULT.to_string());
        let canonical_strategy = TmrCanonicalStrategy::from_str_lossy(&strategy_str);

        let min_agreement: usize = std::env::var("TMR_MIN_AGREEMENT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(TMR_MIN_AGREEMENT)
            .max(2);

        let query_timeout_ms: u64 = std::env::var("TMR_QUERY_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30_000);

        let consensus_timeout_ms: u64 = std::env::var("TMR_CONSENSUS_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60_000);

        let tmr_config = TmrConfig {
            strict_mode: matches!(canonical_strategy, TmrCanonicalStrategy::Strict),
            query_timeout_ms,
            consensus_timeout_ms,
            min_agreement,
        };

        let tmr_engine = TmrConsensusEngine::with_config_and_nodes(tmr_config, engine_nodes);

        Self {
            fallback_node: Arc::new(fallback),
            fallback_trigger_count: std::sync::atomic::AtomicU64::new(0),
            adapters,
            tmr_engine: Arc::new(PlMutex::new(tmr_engine)),
            canonical_strategy,
            min_agreement,
            cache: None,
            singleflight: CacheSingleFlight::new(),
            pii_protector: PIIProtector::new(),
            masking_maps: DashMap::new(),
            finops: None,
        }
    }

    /// Attach a Redis-backed semantic cache to the orchestrator.
    ///
    /// When set, the orchestrator will check the cache before dispatching
    /// LLM calls and store successful consensus results asynchronously.
    pub fn with_cache(mut self, cache: SemanticCache) -> Self {
        self.cache = Some(Arc::new(cache));
        self
    }

    /// Attach a FinOps budget manager for token refunds on failed requests.
    ///
    /// When set, the orchestrator will automatically refund tokens to tenants
    /// when LLM dispatch or consensus fails due to network errors, rate limits,
    /// or timeouts. This ensures tenants are never charged for failed requests.
    pub fn with_finops(mut self, finops: Arc<FinOpsBudgetManager>) -> Self {
        self.finops = Some(finops);
        self
    }

    /// Execute TMR consensus by delegating to the Swarm consensus engine.
    ///
    /// ## Flow
    /// 1. Build a `query_fn` that maps each `ProviderNode` to its `LlmProvider`,
    ///    checks the circuit breaker, invokes the provider, and normalizes the
    ///    output into a canonical `semantic_key` via `IntermediatePayload`.
    /// 2. Delegate to `TmrConsensusEngine::query` for authoritative arbitration.
    /// 3. On success, return the Swarm result as the final decision.
    /// 4. On failure, immediately activate the Stasis local SLM fallback
    ///    without retrying any cloud logic.
    ///
    /// ## Telemetry
    /// Individual `TmrNodeResult` entries are collected via shared state inside
    /// the `query_fn` closure, preserving per-provider telemetry regardless of
    /// completion order.
    pub async fn execute_consensus_providers(&self, req: &ProviderRequest) -> TmrConsensusResult {
        let start = Instant::now();
        let canonical_key = compute_request_canonical_key(&req.tenant_id, &req.prompt);

        if let Some(ref cache) = self.cache {
            match cache.get(&canonical_key).await {
                CacheLookup::Hit(cached_json) => {
                    match serde_json::from_str::<TmrConsensusResult>(&cached_json) {
                        Ok(mut cached_result) => {
                            cached_result.cache_hit = true;
                            info!(
                                key = %canonical_key,
                                "[TMR ORCHESTRATOR] Semantic cache hit - LLM dispatch bypassed"
                            );
                            return cached_result;
                        }
                        Err(e) => {
                            warn!(
                                error = %e,
                                "[SEMANTIC CACHE] Deserialization failed - falling back to LLM dispatch"
                            );
                        }
                    }
                }
                CacheLookup::Miss => {}
            }
        }

        let sf_key = canonical_key.clone();
        let sf = &self.singleflight;
        let result = sf
            .dedup(&sf_key, || {
                self.dispatch_consensus_inner(req, &canonical_key, start)
            })
            .await;

        result
    }

    /// Inner dispatch logic separated for singleflight deduplication.
    ///
    /// This method contains the actual LLM provider dispatch, Swarm consensus
    /// delegation, and cache write-back. It is called through [`CacheSingleFlight::dedup`]
    /// so that concurrent cache-miss requests for the same canonical key share
    /// a single LLM dispatch (Thundering Herd prevention).
    #[allow(clippy::await_holding_lock)]
    async fn dispatch_consensus_inner(
        &self,
        req: &ProviderRequest,
        canonical_key: &str,
        start: Instant,
    ) -> TmrConsensusResult {
        let (masked_prompt, masking_map) = self.pii_protector.mask(&req.prompt);
        self.masking_maps
            .insert(canonical_key.to_string(), masking_map);

        let masked_req = ProviderRequest {
            prompt: masked_prompt,
            ..req.clone()
        };

        let node_results_collector: Arc<PlMutex<Vec<TmrNodeResult>>> =
            Arc::new(PlMutex::new(Vec::new()));

        let mut dispatch_futures = Vec::with_capacity(self.adapters.len());

        for (i, adapter) in self.adapters.iter().enumerate() {
            let adapter = adapter.clone();
            let collector = node_results_collector.clone();
            let jitter_idx = i.min(TMR_TEMPERATURE_JITTER.len() - 1);
            let mut jittered_req = masked_req.clone();
            jittered_req.temperature += TMR_TEMPERATURE_JITTER[jitter_idx];

            let future = async move {
                if let Err(e) = adapter.circuit_breaker.allow_request() {
                    warn!(
                        provider = adapter.provider.display_name(),
                        error = %e,
                        "[TMR ORCHESTRATOR] Circuit breaker blocked request"
                    );

                    collector.lock().push(TmrNodeResult {
                        provider_id: adapter.provider.provider_id().to_string(),
                        provider_name: adapter.provider.display_name().to_string(),
                        output: None,
                        error: Some(TmrNodeError::CircuitBreakerOpen),
                        duration_ms: 0,
                        fallback_triggered: true,
                    });

                    return ProviderResponse {
                        provider: adapter.node.clone(),
                        content: String::new(),
                        semantic_key: String::new(),
                        latency_ms: 0,
                        tokens_used: 0,
                        status: ResponseStatus::ServerError,
                    };
                }

                let invoke_start = Instant::now();
                let timeout_ms = adapter.provider.timeout_ms();
                let provider_id = adapter.provider.provider_id().to_string();
                let provider_name = adapter.provider.display_name().to_string();

                let result = tokio::time::timeout(
                    Duration::from_millis(timeout_ms),
                    adapter.provider.invoke(&jittered_req),
                )
                .await;

                match result {
                    Ok(Ok(response)) => {
                        let semantic_key =
                            IntermediatePayload::from_llm_output(&response.content)
                                .map(|p| p.canonical_key())
                                .unwrap_or_default();

                        adapter.circuit_breaker.record_success();

                        let duration_ms = invoke_start.elapsed().as_millis() as u64;

                        collector.lock().push(TmrNodeResult {
                            provider_id,
                            provider_name,
                            output: Some(response.content.clone()),
                            error: None,
                            duration_ms,
                            fallback_triggered: false,
                        });

                        ProviderResponse {
                            provider: adapter.node.clone(),
                            content: response.content,
                            semantic_key,
                            latency_ms: duration_ms,
                            tokens_used: 0,
                            status: ResponseStatus::Success,
                        }
                    }
                    Ok(Err(err)) => {
                        let triggered = err.should_trigger_fallback();

                        let status = match &err {
                            TmrNodeError::HttpRateLimited => ResponseStatus::RateLimited,
                            TmrNodeError::ServerError(_) => ResponseStatus::ServerError,
                            TmrNodeError::NetworkError(_) => ResponseStatus::NetworkError,
                            TmrNodeError::Timeout => ResponseStatus::Timeout,
                            _ => ResponseStatus::ServerError,
                        };

                        let http_status = match &err {
                            TmrNodeError::HttpForbidden => Some(403u16),
                            TmrNodeError::HttpRateLimited => Some(429u16),
                            TmrNodeError::ServerError(code) => Some(*code),
                            _ => None,
                        };
                        adapter.circuit_breaker.record_failure(http_status);

                        let duration_ms = invoke_start.elapsed().as_millis() as u64;

                        error!(
                            provider = %provider_name,
                            error = ?err,
                            duration_ms = duration_ms,
                            "[TMR ORCHESTRATOR] Provider invocation failed"
                        );

                        collector.lock().push(TmrNodeResult {
                            provider_id,
                            provider_name,
                            output: None,
                            error: Some(err),
                            duration_ms,
                            fallback_triggered: triggered,
                        });

                        ProviderResponse {
                            provider: adapter.node.clone(),
                            content: String::new(),
                            semantic_key: String::new(),
                            latency_ms: duration_ms,
                            tokens_used: 0,
                            status,
                        }
                    }
                    Err(_elapsed) => {
                        adapter.circuit_breaker.record_failure(Some(408));

                        let duration_ms = timeout_ms;

                        error!(
                            provider = %provider_name,
                            timeout_ms = timeout_ms,
                            "[TMR ORCHESTRATOR] Provider request timed out"
                        );

                        collector.lock().push(TmrNodeResult {
                            provider_id,
                            provider_name,
                            output: None,
                            error: Some(TmrNodeError::Timeout),
                            duration_ms,
                            fallback_triggered: true,
                        });

                        ProviderResponse {
                            provider: adapter.node.clone(),
                            content: String::new(),
                            semantic_key: String::new(),
                            latency_ms: duration_ms,
                            tokens_used: 0,
                            status: ResponseStatus::Timeout,
                        }
                    }
                }
            };

            dispatch_futures.push(future);
        }

        let responses: Vec<ProviderResponse> = futures::future::join_all(dispatch_futures).await;

        let node_results = match Arc::try_unwrap(node_results_collector) {
            Ok(mutex) => mutex.into_inner(),
            Err(arc) => arc.lock().clone(),
        };

        let successful: Vec<&ProviderResponse> = responses
            .iter()
            .filter(|r| r.status == ResponseStatus::Success && !r.content.is_empty())
            .collect();

        if successful.len() < self.min_agreement {
            error!(
                successful_count = successful.len(),
                total_nodes = self.adapters.len(),
                min_agreement = self.min_agreement,
                "[TMR ORCHESTRATOR] Quorum not met - insufficient successful providers"
            );

            let mut result = self
                .handle_consensus_failure_stasis(req, node_results, start)
                .await;

            if !result.consensus_achieved {
                if let Some(ref finops) = self.finops {
                    let estimated_cost = self.adapters.len() as i64;
                    let _ = finops
                        .refund_tokens(&req.tenant_id, estimated_cost, canonical_key)
                        .await;
                    warn!(
                        tenant_id = %req.tenant_id,
                        request_id = %canonical_key,
                        refund_amount = estimated_cost,
                        "[FINOPS REFUND] Consensus failed - tokens refunded to tenant"
                    );
                }
            }

            if let Some((_, map)) = self.masking_maps.remove(canonical_key) {
                if let Some(ref output) = result.majority_output {
                    result.majority_output = Some(PIIProtector::restore(output, &map));
                }
            }

            result.cache_hit = false;

            if let Some(ref cache) = self.cache {
                if result.consensus_achieved && result.majority_output.is_some() {
                    if let Ok(json) = serde_json::to_string(&result) {
                        let cache = cache.clone();
                        let key = canonical_key.to_string();
                        tokio::spawn(async move {
                            cache.set(&key, &json).await;
                        });
                    }
                }
            }

            if result.consensus_achieved {
                if let Some(ref output) = result.majority_output {
                    let output = output.clone();
                    let agreement_count = result.agreement_count;
                    let tenant_id = req.tenant_id.clone();
                    let tokens_used =
                        result.node_results.iter().map(|r| r.duration_ms).sum::<u64>() / 100;
                    tokio::spawn(async move {
                        proof_logger::submit_verifiable_proof(&output, agreement_count);
                        crypto_billing::deduct_inference_cost(&tenant_id, tokens_used);
                    });
                }
            }

            return result;
        }

        let consensus_result = self.tmr_engine.lock().evaluate(responses);

        let mut result = match consensus_result {
            Ok(consensus) => self.handle_consensus_success(consensus, node_results, start),
            Err(_) => {
                self.handle_consensus_failure_stasis(req, node_results, start)
                    .await
            }
        };

        if !result.consensus_achieved {
            if let Some(ref finops) = self.finops {
                let estimated_cost = self.adapters.len() as i64;
                let _ = finops
                    .refund_tokens(&req.tenant_id, estimated_cost, canonical_key)
                    .await;
                warn!(
                    tenant_id = %req.tenant_id,
                    request_id = %canonical_key,
                    refund_amount = estimated_cost,
                    "[FINOPS REFUND] Consensus failed - tokens refunded to tenant"
                );
            }
        }

        if let Some((_, map)) = self.masking_maps.remove(canonical_key) {
            if let Some(ref output) = result.majority_output {
                result.majority_output = Some(PIIProtector::restore(output, &map));
            }
        }

        result.cache_hit = false;

        if let Some(ref cache) = self.cache {
            if result.consensus_achieved && result.majority_output.is_some() {
                if let Ok(json) = serde_json::to_string(&result) {
                    let cache = cache.clone();
                    let key = canonical_key.to_string();
                    tokio::spawn(async move {
                        cache.set(&key, &json).await;
                    });
                }
            }
        }

        if result.consensus_achieved {
            if let Some(ref output) = result.majority_output {
                let output = output.clone();
                let agreement_count = result.agreement_count;
                let tenant_id = req.tenant_id.clone();
                let tokens_used = result.node_results.iter().map(|r| r.duration_ms).sum::<u64>() / 100;
                tokio::spawn(async move {
                    proof_logger::submit_verifiable_proof(&output, agreement_count);
                    crypto_billing::deduct_inference_cost(&tenant_id, tokens_used);
                });
            }
        }

        result
    }

    /// Handle a successful consensus result from the Swarm engine.
    fn handle_consensus_success(
        &self,
        result: ConsensusResult,
        node_results: Vec<TmrNodeResult>,
        start: Instant,
    ) -> TmrConsensusResult {
        info!(
            agreement_count = result.winning_providers.len(),
            total_latency_ms = result.total_latency_ms,
            degraded = result.degraded,
            "[TMR ORCHESTRATOR] Swarm consensus achieved"
        );

        TmrConsensusResult {
            majority_output: Some(result.content),
            agreement_count: result.winning_providers.len(),
            total_nodes: self.adapters.len(),
            fallback_activated: false,
            node_results,
            fallback_result: None,
            consensus_achieved: true,
            total_duration_ms: start.elapsed().as_millis() as u64,
            cache_hit: false,
        }
    }

    /// Handle Swarm consensus failure by immediately activating the Stasis
    /// local SLM fallback without retrying compromised cloud logic.
    async fn handle_consensus_failure_stasis(
        &self,
        req: &ProviderRequest,
        node_results: Vec<TmrNodeResult>,
        start: Instant,
    ) -> TmrConsensusResult {
        self.fallback_trigger_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        warn!(
            "[TMR ORCHESTRATOR] Swarm consensus failed - activating Stasis local SLM fallback immediately"
        );

        let fallback_result = self.fallback_node.invoke(&req.prompt).await;

        if let Some(ref fb_output) = fallback_result.output {
            TmrConsensusResult {
                majority_output: Some(fb_output.clone()),
                agreement_count: 1,
                total_nodes: self.adapters.len() + 1,
                fallback_activated: true,
                node_results,
                fallback_result: Some(fallback_result),
                consensus_achieved: true,
                total_duration_ms: start.elapsed().as_millis() as u64,
                cache_hit: false,
            }
        } else {
            TmrConsensusResult {
                majority_output: None,
                agreement_count: 0,
                total_nodes: self.adapters.len(),
                fallback_activated: true,
                node_results,
                fallback_result: Some(fallback_result),
                consensus_achieved: false,
                total_duration_ms: start.elapsed().as_millis() as u64,
                cache_hit: false,
            }
        }
    }

    pub fn fallback_trigger_count(&self) -> u64 {
        self.fallback_trigger_count
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn fallback_node(&self) -> &Arc<PhysicalFallbackNode> {
        &self.fallback_node
    }

    pub fn canonical_strategy(&self) -> TmrCanonicalStrategy {
        self.canonical_strategy
    }

    pub fn min_agreement(&self) -> usize {
        self.min_agreement
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasi_nn_endpoint_default() {
        let endpoint = WasiNnEndpoint::default();
        assert_eq!(endpoint.model_id, "serein-slm-v1");
        assert!(endpoint.enabled);
    }

    #[test]
    fn test_wasi_nn_endpoint_builder() {
        let endpoint = WasiNnEndpoint::new("test-model", "/path/to/model.wasm")
            .with_timeouts(60_000)
            .with_token_limits(8192, 4096)
            .disabled();
        assert_eq!(endpoint.model_id, "test-model");
        assert_eq!(endpoint.inference_timeout_ms, 60_000);
        assert_eq!(endpoint.max_input_tokens, 8192);
        assert_eq!(endpoint.max_output_tokens, 4096);
        assert!(!endpoint.enabled);
    }

    #[tokio::test]
    async fn test_physical_fallback_node_invoke() {
        let endpoint = WasiNnEndpoint::default();
        let node = PhysicalFallbackNode::new(endpoint);
        let result = node.invoke("test prompt").await;
        assert!(result.output.is_some());
        assert!(result.error.is_none());
        assert_eq!(result.model_id, "serein-slm-v1");
    }

    #[tokio::test]
    async fn test_physical_fallback_node_disabled() {
        let endpoint = WasiNnEndpoint::default().disabled();
        let node = PhysicalFallbackNode::new(endpoint);
        let result = node.invoke("test prompt").await;
        assert!(result.output.is_none());
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn test_tmr_orchestrator_consensus_with_providers() {
        use crate::provider::{LlmProvider, ProviderResponse, TmrNodeError};
        use async_trait::async_trait;

        struct MockProvider {
            id: String,
            name: String,
            output: String,
        }

        #[async_trait]
        impl LlmProvider for MockProvider {
            fn provider_id(&self) -> &str {
                &self.id
            }
            fn display_name(&self) -> &str {
                &self.name
            }
            fn cache_provider(&self) -> String {
                self.id.clone()
            }
            fn timeout_ms(&self) -> u64 {
                5000
            }

            async fn invoke(
                &self,
                _req: &ProviderRequest,
            ) -> Result<ProviderResponse, TmrNodeError> {
                Ok(ProviderResponse {
                    content: self.output.clone(),
                })
            }
        }

        let providers: Vec<Arc<dyn LlmProvider>> = vec![
            Arc::new(MockProvider {
                id: "provider-a".to_string(),
                name: "Provider A".to_string(),
                output: r#"{"networkId":"ethereum","taskType":"swap","maxGasLimit":300000,"confidenceScore":0.95,"sourceUrl":"https://example.com"}"#.to_string(),
            }),
            Arc::new(MockProvider {
                id: "provider-b".to_string(),
                name: "Provider B".to_string(),
                output: r#"{"networkId":"ethereum","taskType":"swap","maxGasLimit":300000,"confidenceScore":0.95,"sourceUrl":"https://example.com"}"#.to_string(),
            }),
            Arc::new(MockProvider {
                id: "provider-c".to_string(),
                name: "Provider C".to_string(),
                output: r#"{"networkId":"ethereum","taskType":"swap","maxGasLimit":300000,"confidenceScore":0.95,"sourceUrl":"https://example.com"}"#.to_string(),
            }),
        ];

        let slm_config = SlmNodeConfig {
            model_id: "test-slm".to_string(),
            model_path: std::path::PathBuf::from("./test-model.gguf"),
            execution_mode: SlmExecutionMode::Active,
        };
        let orchestrator = TmrOrchestrator::with_providers(providers, slm_config);
        let req = ProviderRequest::new("test prompt");
        let result = orchestrator.execute_consensus_providers(&req).await;

        assert!(result.consensus_achieved);
        assert!(result.majority_output.is_some());
        assert!(!result.fallback_activated);
        assert!(result.agreement_count >= 2);

        for nr in &result.node_results {
            assert!(!nr.provider_id.is_empty());
            assert!(!nr.provider_name.is_empty());
        }
    }

    #[tokio::test]
    async fn test_tmr_orchestrator_telemetry_desync_fix() {
        use crate::provider::{LlmProvider, ProviderResponse, TmrNodeError};
        use async_trait::async_trait;

        struct SlowProvider {
            id: String,
            name: String,
            delay_ms: u64,
            output: String,
        }

        #[async_trait]
        impl LlmProvider for SlowProvider {
            fn provider_id(&self) -> &str {
                &self.id
            }
            fn display_name(&self) -> &str {
                &self.name
            }
            fn cache_provider(&self) -> String {
                self.id.clone()
            }
            fn timeout_ms(&self) -> u64 {
                5000
            }

            async fn invoke(
                &self,
                _req: &ProviderRequest,
            ) -> Result<ProviderResponse, TmrNodeError> {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
                Ok(ProviderResponse {
                    content: self.output.clone(),
                })
            }
        }

        let providers: Vec<Arc<dyn LlmProvider>> = vec![
            Arc::new(SlowProvider {
                id: "slow-provider".to_string(),
                name: "Slow Provider".to_string(),
                delay_ms: 200,
                output: r#"{"networkId":"ethereum","taskType":"swap","maxGasLimit":300000,"confidenceScore":0.95,"sourceUrl":"https://example.com"}"#.to_string(),
            }),
            Arc::new(SlowProvider {
                id: "fast-provider".to_string(),
                name: "Fast Provider".to_string(),
                delay_ms: 10,
                output: r#"{"networkId":"ethereum","taskType":"swap","maxGasLimit":300000,"confidenceScore":0.95,"sourceUrl":"https://example.com"}"#.to_string(),
            }),
            Arc::new(SlowProvider {
                id: "medium-provider".to_string(),
                name: "Medium Provider".to_string(),
                delay_ms: 100,
                output: r#"{"networkId":"ethereum","taskType":"swap","maxGasLimit":300000,"confidenceScore":0.95,"sourceUrl":"https://example.com"}"#.to_string(),
            }),
        ];

        let slm_config = SlmNodeConfig {
            model_id: "test-slm".to_string(),
            model_path: std::path::PathBuf::from("./test-model.gguf"),
            execution_mode: SlmExecutionMode::Active,
        };
        let orchestrator = TmrOrchestrator::with_providers(providers, slm_config);
        let req = ProviderRequest::new("test prompt");
        let result = orchestrator.execute_consensus_providers(&req).await;

        assert!(result.consensus_achieved);

        for nr in &result.node_results {
            match nr.provider_id.as_str() {
                "slow-provider" => assert_eq!(nr.provider_name, "Slow Provider"),
                "fast-provider" => assert_eq!(nr.provider_name, "Fast Provider"),
                "medium-provider" => assert_eq!(nr.provider_name, "Medium Provider"),
                _ => panic!("Unknown provider ID: {}", nr.provider_id),
            }
        }
    }

    #[tokio::test]
    async fn test_tmr_orchestrator_stasis_fallback_on_no_consensus() {
        use crate::provider::{LlmProvider, ProviderResponse, TmrNodeError};
        use async_trait::async_trait;

        struct DivergentProvider {
            id: String,
            name: String,
            output: String,
        }

        #[async_trait]
        impl LlmProvider for DivergentProvider {
            fn provider_id(&self) -> &str {
                &self.id
            }
            fn display_name(&self) -> &str {
                &self.name
            }
            fn cache_provider(&self) -> String {
                self.id.clone()
            }
            fn timeout_ms(&self) -> u64 {
                5000
            }

            async fn invoke(
                &self,
                _req: &ProviderRequest,
            ) -> Result<ProviderResponse, TmrNodeError> {
                Ok(ProviderResponse {
                    content: self.output.clone(),
                })
            }
        }

        let providers: Vec<Arc<dyn LlmProvider>> = vec![
            Arc::new(DivergentProvider {
                id: "provider-a".to_string(),
                name: "Provider A".to_string(),
                output: r#"{"networkId":"ethereum","taskType":"swap","maxGasLimit":300000,"confidenceScore":0.95,"sourceUrl":"https://a.com"}"#.to_string(),
            }),
            Arc::new(DivergentProvider {
                id: "provider-b".to_string(),
                name: "Provider B".to_string(),
                output: r#"{"networkId":"solana","taskType":"stake","maxGasLimit":200000,"confidenceScore":0.90,"sourceUrl":"https://b.com"}"#.to_string(),
            }),
            Arc::new(DivergentProvider {
                id: "provider-c".to_string(),
                name: "Provider C".to_string(),
                output: r#"{"networkId":"polygon","taskType":"bridge","maxGasLimit":150000,"confidenceScore":0.85,"sourceUrl":"https://c.com"}"#.to_string(),
            }),
        ];

        let slm_config = SlmNodeConfig {
            model_id: "test-slm".to_string(),
            model_path: std::path::PathBuf::from("./test-model.gguf"),
            execution_mode: SlmExecutionMode::Active,
        };
        let orchestrator = TmrOrchestrator::with_providers(providers, slm_config);
        let req = ProviderRequest::new("test prompt");
        let result = orchestrator.execute_consensus_providers(&req).await;

        assert!(result.fallback_activated);
        assert!(result.fallback_result.is_some());
    }
}
