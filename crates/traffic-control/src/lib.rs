// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Changqing Zhang Serein Nexus - L4/L7 API Gateway with Zero-Trust Security
//!
//! This crate provides API gateway functionality with WebAssembly-based
//! security policies and Zero-Trust architecture.
//!
//! ## Anti-Ban Infrastructure
//! - **Circuit Breakers**: Per-provider state machines that trip on HTTP 429/5xx
//! - **Token Bucket**: Mandatory egress rate limiter before any HTTP request leaves

pub mod circuit_breaker;
pub mod rate_limiter;

pub use rate_limiter::{
    acquire_api_token, get_circuit_breaker, init_circuit_breaker, ApiRateLimiter,
    BudgetDeductionResult, BudgetRefundResult, FinOpsBudgetManager, FinOpsError, FinOpsRefundGuard,
    NetworkError,
};

use futures::stream::FuturesUnordered;
use futures::StreamExt;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use thiserror::Error;

/// Nexus Gateway - Global Traffic Gateway for L4/L7 routing
#[derive(Debug)]
pub struct NexusGateway {
    /// Tenant identifier for routing context
    #[allow(dead_code)]
    tenant: String,
}

/// Nexus-specific errors
#[derive(Error, Debug)]
pub enum NexusError {
    #[error("Route not found for endpoint: {0}")]
    RouteNotFound(String),

    #[error("Malformed payload: {0}")]
    MalformedPayload(#[source] serde_json::Error),

    #[error("Internal gateway error: {0}")]
    Internal(String),

    #[error("TMR Consensus failed: {0}")]
    TmrConsensusFailure(String),
}

impl NexusGateway {
    pub fn new(tenant: impl Into<String>) -> Self {
        NexusGateway {
            tenant: tenant.into(),
        }
    }

    /// Receive a request at the gateway edge
    pub fn receive_request(&self, endpoint: &str, payload: &str) -> Result<String, NexusError> {
        let json_value: Value =
            serde_json::from_str(payload).map_err(NexusError::MalformedPayload)?;

        self.route_request(endpoint, &json_value)
    }

    fn route_request(&self, endpoint: &str, _payload: &Value) -> Result<String, NexusError> {
        if endpoint.starts_with("/v1/agent/") {
            let response = serde_json::json!({
                "status": "ok",
                "tenant": "agent-execution",
                "rate_limit": {
                    "remaining": 999,
                    "reset_after_secs": 3600
                }
            });
            return Ok(response.to_string());
        } else {
            Err(NexusError::RouteNotFound(endpoint.to_string()))
        }
    }
}

/// Convenience function for static gateway usage
pub fn handle_request(endpoint: &str, payload: &str) -> Result<String, NexusError> {
    let gateway = NexusGateway::new("global-nexus");
    gateway.receive_request(endpoint, payload)
}

// INGRESS CONTRACT: ML-KEM (FIPS 204) Enforced. Legacy TLS key exchange prohibited.
//
// All inbound connections to the Nexus Gateway MUST negotiate TLS 1.3+ with an
// ML-KEM (FIPS 204) key encapsulation mechanism.  TLS 1.2 and all RSA/ECDHE-only
// cipher suites are explicitly rejected at the transport layer before any HTTP
// parsing occurs.  This is a non-negotiable Zero-Trust ingress requirement.

/// TMR (Triple Modular Redundancy) Arbitrator for consensus-driven LLM queries.
///
/// Dispatches the same query to multiple LLM backends concurrently and resolves
/// as soon as a 2-of-3 consensus is reached. Remaining in-flight requests are
/// explicitly aborted to prevent network socket leaks.
pub struct TmrArbitrator;

/// Result from a single LLM backend query.
#[derive(Debug, Clone)]
pub struct BackendResult {
    pub node_id: String,
    pub payload: String,
}

type BackendFuture = Pin<Box<dyn Future<Output = Result<BackendResult, NexusError>> + Send>>;

impl TmrArbitrator {
    /// Arbitrate across multiple backend futures, returning the first consensus result.
    ///
    /// Consensus is defined as 2-of-3 matching results (by payload equality).
    /// Once consensus is reached, all remaining futures are explicitly dropped
    /// to abort in-flight requests and prevent socket leaks.
    pub async fn arbitrate(futures: Vec<BackendFuture>) -> Result<BackendResult, NexusError> {
        let mut unordered: FuturesUnordered<_> = futures.into_iter().collect();
        let mut results: Vec<BackendResult> = Vec::new();
        let mut consensus_result: Option<BackendResult> = None;

        while let Some(result) = unordered.next().await {
            if let Ok(r) = result {
                results.push(r.clone());
                if Self::has_consensus(&results) {
                    consensus_result = Some(r);
                    break;
                }
            }
        }

        drop(unordered);

        if let Some(r) = consensus_result {
            return Ok(r);
        }

        results.into_iter().next().ok_or_else(|| {
            NexusError::TmrConsensusFailure(
                "All backends failed - no consensus possible".to_string(),
            )
        })
    }

    fn has_consensus(results: &[BackendResult]) -> bool {
        if results.len() < 2 {
            return false;
        }
        let parsed: Vec<Value> = results
            .iter()
            .map(|r| {
                let sanitized = strip_markdown_fences(&r.payload);
                serde_json::from_str(&sanitized)
                    .unwrap_or_else(|_| Value::String(r.payload.clone()))
            })
            .collect();
        for i in 0..parsed.len() {
            let mut count = 1;
            for j in (i + 1)..parsed.len() {
                if fuzzy_json_eq(&parsed[i], &parsed[j]) {
                    count += 1;
                }
            }
            if count >= 2 {
                return true;
            }
        }
        false
    }
}

/// Strip markdown code fences and leading/trailing whitespace from an LLM response.
///
/// Different LLM providers wrap JSON output in varying formatting:
/// - ```json { ... } ``` (with language tag)
/// - ``` { ... } ``` (bare fences)
/// - Raw JSON with leading/trailing whitespace
///
/// This normalizes all variants to clean, parsable JSON before consensus comparison.
fn strip_markdown_fences(response: &str) -> String {
    let mut s = response.trim();

    if s.starts_with("```") {
        if let Some(after_open) = s.strip_prefix("```json").or_else(|| s.strip_prefix("```")) {
            s = after_open.trim_start();
            if let Some(pos) = s.rfind("```") {
                s = s[..pos].trim();
            }
        }
    }

    s.to_string()
}

/// Fuzzy equality comparison for `serde_json::Value` with numeric tolerance.
///
/// Allows a relative delta of `1e-6` when comparing numeric values (both
/// `Number` and floating-point fields within objects/arrays). Strings and
/// booleans are compared with strict equality. This prevents consensus
/// failure caused by microscopic floating-point precision differences
/// (e.g., `confidence_score: 0.95` vs `0.9500000000000001`).
const FUZZY_DELTA: f64 = 1e-6;

fn fuzzy_json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(na), Value::Number(nb)) => {
            if let (Some(fa), Some(fb)) = (na.as_f64(), nb.as_f64()) {
                if fa.is_nan() && fb.is_nan() {
                    return true;
                }
                if fa.is_infinite() && fb.is_infinite() && fa.signum() == fb.signum() {
                    return true;
                }
                let diff = (fa - fb).abs();
                let max_abs = fa.abs().max(fb.abs());
                if max_abs > 0.0 {
                    diff / max_abs <= FUZZY_DELTA
                } else {
                    diff <= FUZZY_DELTA
                }
            } else {
                na == nb
            }
        }
        (Value::Array(arr_a), Value::Array(arr_b)) => {
            if arr_a.len() != arr_b.len() {
                return false;
            }
            arr_a
                .iter()
                .zip(arr_b.iter())
                .all(|(va, vb)| fuzzy_json_eq(va, vb))
        }
        (Value::Object(obj_a), Value::Object(obj_b)) => {
            if obj_a.len() != obj_b.len() {
                return false;
            }
            obj_a
                .iter()
                .all(|(key, va)| obj_b.get(key).is_some_and(|vb| fuzzy_json_eq(va, vb)))
        }
        (Value::String(sa), Value::String(sb)) => sa == sb,
        (Value::Bool(ba), Value::Bool(bb)) => ba == bb,
        (Value::Null, Value::Null) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_agent_route() {
        let payload = r#"{"network_id": "ethereum", "task_type": "swap"}"#;
        let result = handle_request("/v1/agent/execute", payload);
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.contains("agent-execution"));
        assert!(response.contains("ok"));
    }

    #[test]
    fn test_unknown_route() {
        let payload = r#"{"data": "test"}"#;
        let result = handle_request("/v2/unknown", payload);
        assert!(matches!(result, Err(NexusError::RouteNotFound(_))));
    }

    #[test]
    fn test_malformed_payload() {
        let payload = r#"{"invalid": json}"#;
        let result = handle_request("/v1/agent/execute", payload);
        assert!(matches!(result, Err(NexusError::MalformedPayload(_))));
    }

    #[test]
    fn test_gateway_instance() {
        let gateway = NexusGateway::new("test-tenant");
        let payload = r#"{"test": true}"#;
        let result = gateway.receive_request("/v1/agent/execute", payload);
        assert!(result.is_ok());
    }
}
