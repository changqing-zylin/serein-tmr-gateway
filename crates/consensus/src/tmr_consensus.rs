// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! TMR Consensus Engine - Triple Modular Redundancy Arbitrator
//!
//! Implements a **2-out-of-3 consensus** protocol for LLM provider fault tolerance.
//! When any single provider is tripped by the Circuit Breaker, the engine gracefully
//! degrades to use the remaining healthy nodes.
//!
//! ## Consensus Modes
//! - **Strict (2/3)**: Requires at least 2 identical responses.
//! - **Degraded (1/2)**: Falls back when only 2 providers are available.
//! - **Best Effort (1/N)**: Returns first successful response if majority impossible.
//!
//! ## Canonical Hashing
//! Uses SHA-256 over a deterministically normalized JSON payload for canonical
//! hashing. JSON objects are parsed, recursively key-sorted via `BTreeMap`,
//! and re-serialized with compact formatting (no whitespace) before hashing.
//! This ensures semantically identical JSON payloads produce identical hashes
//! regardless of LLM-introduced whitespace or key-order variance.
//!
//! For non-JSON inputs, BLAKE3 is used as a fallback hash.
//!
//! ## CPU Offloading
//! All CPU-bound canonical hashing is offloaded to a dedicated rayon worker pool
//! to prevent Tokio async executor starvation. Backpressure is applied when the
//! rayon queue exceeds 1000 pending tasks (returns HTTP 429).
//!
//! ## Integration
//! - Works with `CircuitBreaker` from serein-traffic-control for provider health.
//! - Uses `JitteredBackoff` from this crate for transient error recovery.
//! - Supports both built-in provider identifiers and dynamic `Custom(String)` nodes
//!   for runtime-pluggable LLM backends.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use futures::stream::{FuturesUnordered, StreamExt};

/// Maximum number of pending rayon tasks before backpressure is applied.
/// When exceeded, `canonical_semantic_key` returns a 429 Too Many Requests error.
const RAYON_BACKPRESSURE_LIMIT: usize = 1000;

static RAYON_PENDING: AtomicUsize = AtomicUsize::new(0);

/// RAII guard that decrements the global rayon pending counter on drop.
///
/// The guard directly references the module-level `RAYON_PENDING` static,
/// ensuring deterministic decrement regardless of how the scope exits
/// (panic, early return, or normal completion).
struct PendingGuard;

impl Drop for PendingGuard {
    fn drop(&mut self) {
        RAYON_PENDING.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Compute a canonical semantic key from raw LLM output using SHA-256 canonical hash.
///
/// Parses the raw JSON string, sorts keys recursively, re-serializes with compact
/// formatting, and computes a deterministic SHA-256 hash. CPU-intensive work is
/// dispatched to the global rayon thread pool via a `tokio::sync::oneshot`
/// bridge, allowing the async task to yield without blocking the Tokio executor.
///
/// ## Tokio-to-Rayon Bridge
/// 1. Check backpressure counter against `RAYON_BACKPRESSURE_LIMIT`.
/// 2. Create a `oneshot` channel and dispatch `compute_canonical_key_sync`
///    to `rayon::spawn` (runs on the dedicated rayon thread pool).
/// 3. `.await` the oneshot receiver - the Tokio task yields until the
///    rayon worker sends the result back.
///
/// ## Backpressure
/// If the rayon queue has more than `RAYON_BACKPRESSURE_LIMIT` pending tasks,
/// this function returns a 429 indicator string instead of queuing more work.
///
/// ## Fallback
/// Non-JSON inputs are hashed directly as UTF-8 bytes.
pub async fn canonical_semantic_key(raw_content: &str) -> String {
    let current_pending = RAYON_PENDING.load(Ordering::SeqCst);
    if current_pending > RAYON_BACKPRESSURE_LIMIT {
        tracing::warn!(
            target: "consensus",
            pending = current_pending,
            limit = RAYON_BACKPRESSURE_LIMIT,
            "Backpressure active"
        );
        return "429_TOO_MANY_REQUESTS_BACKPRESSURE".to_string();
    }

    RAYON_PENDING.fetch_add(1, Ordering::SeqCst);
    let raw_owned = raw_content.to_string();
    let (tx, rx) = tokio::sync::oneshot::channel();

    rayon::spawn(move || {
        let _guard = PendingGuard;

        if tx.is_closed() { return; }
        let result = compute_canonical_key_sync(&raw_owned);
        let _ = tx.send(result);
    });

    match rx.await {
        Ok(hash) => hash,
        Err(_) => "500_RAYON_WORKER_DROPPED".to_string()
    }
}

/// Recursively sort all object keys in a `serde_json::Value` by converting
/// `Map` entries into a `BTreeMap` and back, then recursing into nested
/// objects and arrays.
fn sort_json_keys(value: &mut serde_json::Value) {
    use std::collections::BTreeMap;
    match value {
        serde_json::Value::Object(map) => {
            let taken = std::mem::take(map);
            let sorted: BTreeMap<String, serde_json::Value> = taken.into_iter().collect();
            *map = sorted.into_iter().collect();
            for v in map.values_mut() {
                sort_json_keys(v);
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                sort_json_keys(v);
            }
        }
        _ => {}
    }
}

/// Compute a canonical SHA-256 hash from a raw LLM response string.
///
/// Parses the input as JSON, recursively sorts all object keys, re-serializes
/// with compact formatting (no whitespace), then hashes the canonical byte
/// representation with SHA-256 and returns the hex-encoded digest.
///
/// Non-JSON inputs return an error rather than falling back to raw hashing,
/// ensuring callers are explicitly aware of parse failures.
///
/// ## Determinism Guarantee
/// Two semantically identical JSON payloads with different key ordering
/// or whitespace will produce the same canonical hash.
pub fn compute_canonical_hash(raw_output: &str) -> Result<String, anyhow::Error> {
    use sha2::{Digest, Sha256};

    let trimmed = raw_output.trim();
    let mut value: serde_json::Value = serde_json::from_str(trimmed)
        .map_err(|e| anyhow::anyhow!("failed to parse raw_output as JSON: {}", e))?;

    sort_json_keys(&mut value);

    let canonical = serde_json::to_string(&value)
        .map_err(|e| anyhow::anyhow!("failed to serialize canonical JSON: {}", e))?;

    let hash = Sha256::digest(canonical.as_bytes());
    Ok(hex::encode(hash))
}

/// Strip leading/trailing whitespace and Markdown code fences from LLM output.
///
/// Handles variants common across LLM providers:
/// - ```json\n{ ... }\n```
/// - ```\n{ ... }\n```
/// - Raw JSON with leading/trailing whitespace
fn strip_markdown_fences(response: &str) -> String {
    let mut s = response.trim();

    if s.starts_with("```") {
        if let Some(after_open) = s.strip_prefix("```json")
            .or_else(|| s.strip_prefix("```"))
        {
            s = after_open.trim_start();
            if let Some(pos) = s.rfind("```") {
                s = s[..pos].trim();
            }
        }
    }

    s.to_string()
}

/// Synchronous canonical key computation for use within rayon or spawn_blocking.
fn compute_canonical_key_sync(raw: &str) -> String {
    let sanitized = strip_markdown_fences(raw);
    let trimmed = sanitized.trim();
    if !trimmed.starts_with('{') && !trimmed.starts_with('[') {
        let hash = blake3::hash(trimmed.as_bytes());
        return format!("blake3:{}", hash.to_hex());
    }
    match compute_canonical_hash(trimmed) {
        Ok(hex_hash) => format!("sha256:{}", hex_hash),
        Err(_) => {
            let hash = blake3::hash(trimmed.as_bytes());
            format!("blake3:{}", hash.to_hex())
        }
    }
}

/// Unique identifier for an LLM provider node in the TMR cluster.
///
/// Built-in variants cover the standard cloud providers. The `Custom` variant
/// supports runtime-registered dynamic providers (e.g., those loaded from
/// `providers.toml` via the oracle's `LlmProvider` trait).
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderNode {
    GoogleGemini,
    DeepSeek,
    Groq,
    /// Runtime-registered dynamic provider identified by a unique string key.
    Custom(String),
}

impl std::fmt::Display for ProviderNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GoogleGemini => write!(f, "gemini"),
            Self::DeepSeek => write!(f, "deepseek"),
            Self::Groq => write!(f, "groq"),
            Self::Custom(id) => write!(f, "custom:{}", id),
        }
    }
}

/// Health status reported by the circuit breaker for each provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeHealth {
    Healthy,
    Degraded,
    Tripped,
}

/// Response payload from a single LLM provider replica.
#[derive(Debug, Clone)]
pub struct ProviderResponse {
    pub provider: ProviderNode,
    pub content: String,
    pub semantic_key: String,
    pub latency_ms: u64,
    pub tokens_used: u32,
    pub status: ResponseStatus,
}

/// Status of the provider response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseStatus {
    Success,
    RateLimited,
    ServerError,
    NetworkError,
    Timeout,
}

/// Consensus result after arbitration completes.
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// The agreed-upon response content.
    pub content: String,
    /// Which providers contributed to consensus.
    pub winning_providers: Vec<ProviderNode>,
    /// Total latency including all parallel queries.
    pub total_latency_ms: u64,
    /// Whether degradation was required (not all 3 nodes available).
    pub degraded: bool,
    /// Token consumption across all queried providers.
    pub total_tokens_used: u32,
}

/// Errors during TMR consensus arbitration.
#[derive(Debug, thiserror::Error)]
pub enum ConsensusError {
    #[error("All providers unavailable or tripped")]
    AllProvidersUnavailable,

    #[error("No consensus reached: {0}")]
    NoConsensus(String),

    #[error("Provider {provider} error: {error}")]
    ProviderError { provider: String, error: String },

    #[error("Backpressure: rayon queue exceeded {0} pending tasks")]
    Backpressure(usize),
}

/// Configuration for the TMR consensus engine.
#[derive(Debug, Clone)]
pub struct TmrConfig {
    /// Enable strict 2-out-of-3 mode (disable for testing).
    pub strict_mode: bool,
    /// Timeout per-provider query.
    pub query_timeout_ms: u64,
    /// Maximum total consensus timeout.
    pub consensus_timeout_ms: u64,
    /// Minimum number of agreeing providers required for consensus.
    ///
    /// Defaults to 2 (2-of-3 quorum). Must be at least 1.
    pub min_agreement: usize,
}

impl Default for TmrConfig {
    fn default() -> Self {
        Self {
            strict_mode: true,
            query_timeout_ms: 30_000,
            consensus_timeout_ms: 60_000,
            min_agreement: 2,
        }
    }
}

/// Triple Modular Redundancy consensus engine.
///
/// Dispatches queries to multiple LLM providers in parallel and arbitrates
/// results using configurable consensus rules. Supports both built-in
/// `ProviderNode` variants and dynamically registered `Custom` nodes.
///
/// ## Single Source of Truth (SSoT)
/// This engine is the sole authority for TMR consensus adjudication.
/// Callers must delegate all majority-vote logic here - no local
/// `find_majority` or byte-hashing fallbacks are permitted.
pub struct TmrConsensusEngine {
    config: TmrConfig,
    node_health: HashMap<ProviderNode, NodeHealth>,
}

impl TmrConsensusEngine {
    /// Create a new TMR consensus engine with default configuration
    /// and the three built-in cloud providers marked as healthy.
    pub fn new() -> Self {
        Self {
            config: TmrConfig::default(),
            node_health: HashMap::from([
                (ProviderNode::GoogleGemini, NodeHealth::Healthy),
                (ProviderNode::DeepSeek, NodeHealth::Healthy),
                (ProviderNode::Groq, NodeHealth::Healthy),
            ]),
        }
    }

    /// Create with custom configuration and the three built-in providers.
    pub fn with_config(config: TmrConfig) -> Self {
        Self {
            config,
            node_health: HashMap::from([
                (ProviderNode::GoogleGemini, NodeHealth::Healthy),
                (ProviderNode::DeepSeek, NodeHealth::Healthy),
                (ProviderNode::Groq, NodeHealth::Healthy),
            ]),
        }
    }

    /// Create an engine with only the specified provider nodes.
    ///
    /// Use this constructor when providers are loaded dynamically at runtime
    /// (e.g., from `providers.toml`). All nodes start with `Healthy` status.
    pub fn with_nodes(nodes: Vec<ProviderNode>) -> Self {
        let node_health: HashMap<ProviderNode, NodeHealth> = nodes
            .into_iter()
            .map(|n| (n, NodeHealth::Healthy))
            .collect();

        Self {
            config: TmrConfig::default(),
            node_health,
        }
    }

    /// Create an engine with custom configuration and specified provider nodes.
    pub fn with_config_and_nodes(config: TmrConfig, nodes: Vec<ProviderNode>) -> Self {
        let node_health: HashMap<ProviderNode, NodeHealth> = nodes
            .into_iter()
            .map(|n| (n, NodeHealth::Healthy))
            .collect();

        Self {
            config,
            node_health,
        }
    }

    /// Register a provider node with an initial health status.
    ///
    /// If the node already exists, its health status is updated.
    pub fn register_node(&mut self, node: ProviderNode, health: NodeHealth) {
        self.node_health.insert(node.clone(), health);
    }

    /// Update the health status of a specific provider node.
    ///
    /// Called by the Circuit Breaker when state transitions occur.
    pub fn update_node_health(&mut self, node: ProviderNode, health: NodeHealth) {
        let prev = self.node_health.insert(node.clone(), health);

        tracing::info!(
            target: "consensus",
            provider = %node,
            new_health = ?health,
            previous = ?prev,
            "Node health updated"
        );
    }

    /// Get the list of currently healthy (non-tripped) providers.
    fn healthy_nodes(&self) -> Vec<&ProviderNode> {
        self.node_health
            .iter()
            .filter(|(_, &h)| h != NodeHealth::Tripped)
            .map(|(n, _)| n)
            .collect()
    }

    /// Execute a consensus query across all available providers.
    ///
    /// # Arguments
    /// * `query_fn` - Async function that takes a `ProviderNode` and returns
    ///   a `ProviderResponse`. The `semantic_key` field of each response is
    ///   used for consensus comparison - callers must normalize LLM output
    ///   into a canonical key before returning.
    ///
    /// # Returns
    /// - `Ok(ConsensusResult)` - When consensus is reached (strict or degraded).
    /// - `Err(ConsensusError)` - When no consensus is possible.
    pub async fn query<F, Fut>(
        &self,
        query_fn: F,
    ) -> Result<ConsensusResult, ConsensusError>
    where
        F: Fn(ProviderNode) -> Fut + Clone + Send,
        Fut: std::future::Future<Output = ProviderResponse> + Send + 'static,
    {
        let available_nodes = self.healthy_nodes();

        if available_nodes.is_empty() {
            return Err(ConsensusError::AllProvidersUnavailable);
        }

        let start_time = std::time::Instant::now();

        let mut tasks: FuturesUnordered<
            std::pin::Pin<Box<dyn std::future::Future<Output = ProviderResponse> + Send>>,
        > = FuturesUnordered::new();

        for node in available_nodes.clone() {
            let qf = query_fn.clone();
            let node_clone = (*node).clone();
            let timeout = Duration::from_millis(self.config.query_timeout_ms);

            tasks.push(Box::pin(async move {
                let node_for_response = node_clone.clone();
                let result =
                    tokio::time::timeout(timeout, qf(node_clone)).await;

                match result {
                    Ok(response) => response,
                    Err(_) => ProviderResponse {
                        provider: node_for_response,
                        content: String::new(),
                        semantic_key: String::new(),
                        latency_ms: timeout.as_millis() as u64,
                        tokens_used: 0,
                        status: ResponseStatus::Timeout,
                    },
                }
            }));
        }

        let mut responses: Vec<ProviderResponse> = Vec::new();
        let mut consensus_map: HashMap<String, (usize, String, Vec<ProviderNode>)> =
            HashMap::new();

        while let Some(response) = tasks.next().await {
            if response.status == ResponseStatus::Success && !response.semantic_key.is_empty() {
                let entry = consensus_map
                    .entry(response.semantic_key.clone())
                    .or_insert((0, String::new(), Vec::new()));

                entry.0 += 1;
                entry.1 = response.content.clone();
                entry.2.push(response.provider.clone());
            }

            responses.push(response);

            let required = if self.config.strict_mode {
                if available_nodes.len() >= 3 { self.config.min_agreement } else { available_nodes.len() }
            } else {
                1
            };

            if let Some((_, content, providers)) =
                consensus_map.values().find(|(c, _, _)| *c >= required)
            {
                let total_latency = start_time.elapsed().as_millis() as u64;
                let total_tokens: u32 = responses
                    .iter()
                    .map(|r| r.tokens_used)
                    .sum();

                let degraded = available_nodes.len() < 3;

                tracing::info!(
                    target: "consensus",
                    consensus_key = consensus_map.keys().next().map(|k| k.as_str()).unwrap_or(""),
                    winners = ?providers,
                    total_latency_ms = total_latency,
                    degraded,
                    "Consensus reached"
                );

                return Ok(ConsensusResult {
                    content: content.clone(),
                    winning_providers: providers.clone(),
                    total_latency_ms: total_latency,
                    degraded,
                    total_tokens_used: total_tokens,
                });
            }
        }

        Err(ConsensusError::NoConsensus(format!(
            "Received {} responses, no majority achieved",
            responses.len()
        )))
    }

    /// Evaluate pre-collected provider responses for consensus without dispatching.
    ///
    /// Unlike [`query`](Self::query), this method does not perform its own
    /// concurrent dispatch. It accepts an already-collected `Vec<ProviderResponse>`
    /// and performs only the adjudication phase - building a consensus map from
    /// semantic keys and determining whether a majority quorum exists.
    ///
    /// This enables callers to use custom fault-tolerant dispatch strategies
    /// (e.g., `join_all` with per-provider circuit breakers and temperature
    /// jitter) while still delegating authoritative consensus arbitration to
    /// this engine as the Single Source of Truth.
    pub fn evaluate(
        &self,
        responses: Vec<ProviderResponse>,
    ) -> Result<ConsensusResult, ConsensusError> {
        if responses.is_empty() {
            return Err(ConsensusError::AllProvidersUnavailable);
        }

        let available_nodes = self.healthy_nodes();
        let total_nodes = if available_nodes.is_empty() {
            responses.len()
        } else {
            available_nodes.len()
        };

        let mut consensus_map: HashMap<String, (usize, String, Vec<ProviderNode>)> =
            HashMap::new();
        let mut total_tokens: u32 = 0;
        let mut total_latency: u64 = 0;

        for response in &responses {
            total_tokens += response.tokens_used;
            total_latency = total_latency.max(response.latency_ms);

            if response.status == ResponseStatus::Success && !response.semantic_key.is_empty() {
                let entry = consensus_map
                    .entry(response.semantic_key.clone())
                    .or_insert((0, String::new(), Vec::new()));
                entry.0 += 1;
                entry.1 = response.content.clone();
                entry.2.push(response.provider.clone());
            }
        }

        let required = if self.config.strict_mode {
            if total_nodes >= 3 {
                self.config.min_agreement
            } else {
                total_nodes
            }
        } else {
            1
        };

        if let Some((count, content, providers)) =
            consensus_map.values().max_by_key(|(c, _, _)| *c)
        {
            if *count >= required {
                let degraded = total_nodes < 3;

                tracing::info!(
                    target: "consensus",
                    winners = ?providers,
                    agreement_count = count,
                    total_latency_ms = total_latency,
                    degraded,
                    "Consensus reached via evaluate()"
                );

                return Ok(ConsensusResult {
                    content: content.clone(),
                    winning_providers: providers.clone(),
                    total_latency_ms: total_latency,
                    degraded,
                    total_tokens_used: total_tokens,
                });
            }
        }

        Err(ConsensusError::NoConsensus(format!(
            "Received {} responses, no majority achieved (best agreement: {}, required: {})",
            responses.len(),
            consensus_map.values().map(|(c, _, _)| *c).max().unwrap_or(0),
            required
        )))
    }

    /// Get current health snapshot of all nodes.
    pub fn health_snapshot(&self) -> HashMap<String, NodeHealth> {
        self.node_health
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect()
    }
}

impl Default for TmrConsensusEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_node() {
        let mut engine = TmrConsensusEngine::with_nodes(vec![]);
        engine.register_node(ProviderNode::Custom("test".to_string()), NodeHealth::Healthy);

        let snapshot = engine.health_snapshot();
        assert!(snapshot.contains_key("custom:test"));
    }

    #[tokio::test]
    async fn test_canonical_semantic_key_json() {
        let result = canonical_semantic_key(r#"{"z":1,"a":2}"#).await;
        assert!(result.starts_with("sha256:"));
    }

    #[tokio::test]
    async fn test_canonical_semantic_key_non_json() {
        let result = canonical_semantic_key("plain text response").await;
        assert!(result.starts_with("blake3:"));
    }

    #[tokio::test]
    async fn test_canonical_semantic_key_key_order_invariance() {
        let key_a = canonical_semantic_key(r#"{"b":2,"a":1}"#).await;
        let key_b = canonical_semantic_key(r#"{"a":1,"b":2}"#).await;
        assert_eq!(key_a, key_b, "Key-reordered JSON must produce identical canonical keys");
    }

    #[tokio::test]
    async fn test_tmr_strict_consensus_two_of_three() {
        let engine = TmrConsensusEngine::new();

        async fn mock_response(provider: ProviderNode) -> ProviderResponse {
            tokio::time::sleep(Duration::from_millis(5)).await;

            let (content, semantic_key) = match provider {
                ProviderNode::GoogleGemini | ProviderNode::DeepSeek => {
                    ("consensus_answer".to_string(), "CHAIN:ETH:ACTION:swap:AMOUNT:90".to_string())
                }
                _ => {
                    ("different_answer".to_string(), "CHAIN:ETH:ACTION:audit:AMOUNT:180".to_string())
                }
            };

            ProviderResponse {
                provider,
                content,
                semantic_key,
                latency_ms: 5,
                tokens_used: 100,
                status: ResponseStatus::Success,
            }
        }

        let result = engine.query(mock_response).await;
        assert!(result.is_ok(), "2-of-3 consensus must succeed");

        let consensus = result.unwrap();
        assert_eq!(consensus.content, "consensus_answer");
        assert!(consensus.winning_providers.len() >= 2, "At least 2 providers must agree");
        assert!(!consensus.degraded, "Full 3-node consensus must not be degraded");
    }

    #[tokio::test]
    async fn test_tmr_degraded_consensus_two_nodes() {
        let mut engine = TmrConsensusEngine::new();
        engine.update_node_health(ProviderNode::Groq, NodeHealth::Tripped);

        async fn mock_response(provider: ProviderNode) -> ProviderResponse {
            tokio::time::sleep(Duration::from_millis(5)).await;
            let content = "agreed_degraded_response".to_string();
            let semantic_key = "agreed_degraded_key".to_string();
            ProviderResponse {
                provider,
                content,
                semantic_key,
                latency_ms: 5,
                tokens_used: 50,
                status: ResponseStatus::Success,
            }
        }

        let result = engine.query(mock_response).await;
        assert!(result.is_ok(), "Degraded consensus with 2 nodes must succeed");

        let consensus = result.unwrap();
        assert!(consensus.degraded, "2-node consensus must be marked as degraded");
    }

    #[tokio::test]
    async fn test_tmr_all_providers_unavailable() {
        let mut engine = TmrConsensusEngine::with_nodes(vec![
            ProviderNode::Custom("p1".to_string()),
            ProviderNode::Custom("p2".to_string()),
        ]);
        engine.update_node_health(ProviderNode::Custom("p1".to_string()), NodeHealth::Tripped);
        engine.update_node_health(ProviderNode::Custom("p2".to_string()), NodeHealth::Tripped);

        async fn mock_response(provider: ProviderNode) -> ProviderResponse {
            ProviderResponse {
                provider,
                content: String::new(),
                semantic_key: String::new(),
                latency_ms: 0,
                tokens_used: 0,
                status: ResponseStatus::ServerError,
            }
        }

        let result = engine.query(mock_response).await;
        assert!(result.is_err(), "All-tripped providers must return an error");
        match result.unwrap_err() {
            ConsensusError::AllProvidersUnavailable => {}
            other => panic!("Expected AllProvidersUnavailable, got: {}", other),
        }
    }

    #[tokio::test]
    async fn test_tmr_no_consensus_all_different() {
        let engine = TmrConsensusEngine::with_nodes(vec![
            ProviderNode::Custom("a".to_string()),
            ProviderNode::Custom("b".to_string()),
            ProviderNode::Custom("c".to_string()),
        ]);

        async fn mock_response(provider: ProviderNode) -> ProviderResponse {
            let key = match &provider {
                ProviderNode::Custom(id) => id.clone(),
                _ => "default".to_string(),
            };
            ProviderResponse {
                provider,
                content: format!("unique_{}", key),
                semantic_key: format!("unique_key_{}", key),
                latency_ms: 5,
                tokens_used: 50,
                status: ResponseStatus::Success,
            }
        }

        let result = engine.query(mock_response).await;
        assert!(result.is_err(), "3 different responses must fail consensus");
        match result.unwrap_err() {
            ConsensusError::NoConsensus(_) => {}
            other => panic!("Expected NoConsensus, got: {}", other),
        }
    }

    #[tokio::test]
    async fn test_tmr_timeout_counts_as_failure() {
        let engine = TmrConsensusEngine::with_config(TmrConfig {
            strict_mode: true,
            query_timeout_ms: 50,
            consensus_timeout_ms: 200,
            min_agreement: 2,
        });

        async fn mock_response(provider: ProviderNode) -> ProviderResponse {
            match provider {
                ProviderNode::GoogleGemini => {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    ProviderResponse {
                        provider,
                        content: "late_answer".to_string(),
                        semantic_key: "LATE_KEY".to_string(),
                        latency_ms: 500,
                        tokens_used: 100,
                        status: ResponseStatus::Success,
                    }
                }
                ProviderNode::DeepSeek => ProviderResponse {
                    provider,
                    content: "fast_answer".to_string(),
                    semantic_key: "FAST_KEY".to_string(),
                    latency_ms: 5,
                    tokens_used: 100,
                    status: ResponseStatus::Success,
                },
                _ => ProviderResponse {
                    provider,
                    content: String::new(),
                    semantic_key: String::new(),
                    latency_ms: 50,
                    tokens_used: 0,
                    status: ResponseStatus::Timeout,
                },
            }
        }

        let result = engine.query(mock_response).await;
        assert!(result.is_err(), "1 success + 1 timeout + 1 late must fail 2/3 consensus");
    }

    #[tokio::test]
    #[ignore = "Mutates global RAYON_PENDING state, causes race conditions in parallel tests"]
    async fn test_backpressure_returns_429() {
        let original = RAYON_PENDING.load(Ordering::SeqCst);
        RAYON_PENDING.store(RAYON_BACKPRESSURE_LIMIT + 100, Ordering::SeqCst);

        let result = canonical_semantic_key(r#"{"test":1}"#).await;

        RAYON_PENDING.store(original, Ordering::SeqCst);

        assert_eq!(result, "429_TOO_MANY_REQUESTS_BACKPRESSURE", "Must return 429 when backpressure limit exceeded");
    }

    #[tokio::test]
    #[ignore = "Mutates global RAYON_PENDING state, causes race conditions in parallel tests"]
    async fn test_backpressure_normal_flow() {
        let original = RAYON_PENDING.load(Ordering::SeqCst);
        RAYON_PENDING.store(0, Ordering::SeqCst);

        let result = canonical_semantic_key(r#"{"test":1}"#).await;

        RAYON_PENDING.store(original, Ordering::SeqCst);

        assert!(result.starts_with("sha256:"), "Must return valid hash when under backpressure limit");
    }

    #[test]
    fn test_compute_canonical_key_sync_json() {
        let result = compute_canonical_key_sync(r#"{"b":2,"a":1}"#);
        assert!(result.starts_with("sha256:"));

        let result_reordered = compute_canonical_key_sync(r#"{"a":1,"b":2}"#);
        assert_eq!(result, result_reordered, "Sync key must be key-order invariant");
    }

    #[test]
    fn test_compute_canonical_key_sync_non_json() {
        let result = compute_canonical_key_sync("not json at all");
        assert!(result.starts_with("blake3:"));
    }

    #[test]
    fn test_compute_canonical_key_sync_whitespace_json() {
        let compact = compute_canonical_key_sync(r#"{"a":1}"#);
        let padded = compute_canonical_key_sync(r#"  { "a" : 1 }  "#);
        assert_eq!(compact, padded, "Whitespace-padded JSON must produce identical key");
    }

    #[test]
    fn test_canonical_key_sync_markdown_fenced_json() {
        let fenced =
            compute_canonical_key_sync("```json\n{\"a\":1}\n```");
        assert!(
            fenced.starts_with("sha256:"),
            "Markdown-fenced JSON must produce SHA-256 canonical key"
        );
    }

    #[test]
    fn test_canonical_key_sync_bare_fenced_json() {
        let fenced =
            compute_canonical_key_sync("```\n{\"a\":1}\n```");
        assert!(
            fenced.starts_with("sha256:"),
            "Bare-fenced JSON must produce SHA-256 canonical key"
        );
    }

    #[test]
    fn test_canonical_key_sync_formatting_agnostic() {
        let minified = compute_canonical_key_sync(r#"{"a":1,"b":2}"#);
        let pretty = compute_canonical_key_sync("{\n  \"a\": 1,\n  \"b\": 2\n}");
        let fenced = compute_canonical_key_sync("```json\n{\"a\":1,\"b\":2}\n```");
        assert_eq!(minified, pretty, "Pretty vs minified JSON must match");
        assert_eq!(minified, fenced, "Fenced vs minified JSON must match");
        assert_eq!(pretty, fenced, "Fenced vs pretty JSON must match");
    }

    #[test]
    fn test_canonical_key_sync_non_json_after_fence_strip() {
        let fenced_text =
            compute_canonical_key_sync("```\nthis is plain text, not json\n```");
        assert!(
            fenced_text.starts_with("blake3:"),
            "Fenced plain text must fall back to BLAKE3"
        );
    }

    #[test]
    fn test_compute_canonical_hash_key_order_invariance() {
        let hash_a = compute_canonical_hash(r#"{"z":1,"a":2,"m":3}"#).unwrap();
        let hash_b = compute_canonical_hash(r#"{"a":2,"m":3,"z":1}"#).unwrap();
        assert_eq!(hash_a, hash_b, "Key-reordered JSON must produce identical SHA-256 hash");
    }

    #[test]
    fn test_compute_canonical_hash_whitespace_invariance() {
        let hash_a = compute_canonical_hash(r#"{"a":1}"#).unwrap();
        let hash_b = compute_canonical_hash(r#"  { "a" : 1 }  "#).unwrap();
        assert_eq!(hash_a, hash_b, "Whitespace-padded JSON must produce identical SHA-256 hash");
    }

    #[test]
    fn test_compute_canonical_hash_nested_key_order() {
        let hash_a = compute_canonical_hash(r#"{"outer":{"z":1,"a":2},"b":3}"#).unwrap();
        let hash_b = compute_canonical_hash(r#"{"b":3,"outer":{"a":2,"z":1}}"#).unwrap();
        assert_eq!(hash_a, hash_b, "Deeply nested key-reordered JSON must produce identical SHA-256 hash");
    }

    #[test]
    fn test_compute_canonical_hash_different_values() {
        let hash_a = compute_canonical_hash(r#"{"key":"value_a"}"#).unwrap();
        let hash_b = compute_canonical_hash(r#"{"key":"value_b"}"#).unwrap();
        assert_ne!(hash_a, hash_b, "Different values must produce different SHA-256 hashes");
    }

    #[test]
    fn test_compute_canonical_hash_non_json_error() {
        let result = compute_canonical_hash("not json at all");
        assert!(result.is_err(), "Non-JSON input must return an error");
    }

    /// Taiwei (Consensus) specification: deeply nested arrays must produce
    /// identical canonical hashes regardless of internal key ordering within
    /// objects nested inside array elements.
    #[test]
    fn test_compute_canonical_hash_deeply_nested_arrays() {
        let payload_a = r#"{
            "results": [
                {"id": 1, "items": [{"z": 9, "a": 1}, {"m": 3, "b": 2}]},
                {"id": 2, "items": [{"x": 5, "y": 6}]}
            ]
        }"#;
        let payload_b = r#"{
            "results": [
                {"id": 1, "items": [{"a": 1, "z": 9}, {"b": 2, "m": 3}]},
                {"id": 2, "items": [{"y": 6, "x": 5}]}
            ]
        }"#;

        let hash_a = compute_canonical_hash(payload_a)
            .expect("Payload A must parse as valid JSON");
        let hash_b = compute_canonical_hash(payload_b)
            .expect("Payload B must parse as valid JSON");

        assert_eq!(
            hash_a, hash_b,
            "Deeply nested arrays with key-reordered internal objects must produce identical SHA-256 hashes"
        );
    }

    /// Taiwei (Consensus) specification: arrays with different element ordering
    /// must produce different canonical hashes (arrays are order-sensitive).
    #[test]
    fn test_compute_canonical_hash_array_order_sensitivity() {
        let payload_a = r#"{"items": [1, 2, 3]}"#;
        let payload_b = r#"{"items": [3, 2, 1]}"#;

        let hash_a = compute_canonical_hash(payload_a).unwrap();
        let hash_b = compute_canonical_hash(payload_b).unwrap();

        assert_ne!(
            hash_a, hash_b,
            "Arrays with different element ordering must produce different hashes (order-sensitive)"
        );
    }
}
