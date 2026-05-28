// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Circuit Breaker - LLM Provider Resiliency Pattern
//!
//! Implements the **state-machine circuit breaker** for individual LLM provider nodes.
//!
//! ## State Machine
//! - `Closed` → Normal operation, requests pass through.
//! - `Open` → Provider is tripped (HTTP 429/5xx). All requests fail fast.
//! - `HalfOpen` → Probe state: single request allowed to test recovery.
//!
//! ## Anti-Ban Contract
//! - **NO aggressive while-loop retries** when tripped.
//! - Return error immediately and let TMR consensus degrade gracefully.
//! - Trip on `HTTP 429` or `5xx` status codes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

/// Circuit breaker states encoded as atomic u8 for lock-free reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[repr(u8)]
pub enum CircuitState {
    /// Normal operation - requests flow through.
    Closed = 0,
    /// Provider tripped - all requests fail fast.
    Open = 1,
    /// Probe state - single request allowed to test recovery.
    HalfOpen = 2,
}

impl From<u8> for CircuitState {
    fn from(v: u8) -> Self {
        match v {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed,
        }
    }
}

/// Configuration parameters for a single circuit breaker instance.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before tripping the circuit.
    pub failure_threshold: u32,
    /// Duration the circuit remains in Open state before attempting HalfOpen.
    pub open_duration: Duration,
    /// Number of successes required in HalfOpen to transition back to Closed.
    pub half_open_max_attempts: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            open_duration: Duration::from_secs(30),
            half_open_max_attempts: 1,
        }
    }
}

/// Atomic snapshot of the circuit breaker's operational status.
///
/// Bundles `State` and `LastFailureTime` into a single struct protected by
/// `parking_lot::RwLock`, ensuring atomic read-modify-write semantics for
/// all status transitions. This eliminates the race condition window between
/// independent `AtomicU8` state and `Mutex<Option<Instant>>` timestamp updates.
#[derive(Debug, Clone)]
struct CircuitStatus {
    state: CircuitState,
    last_failure_time: Option<Instant>,
}

impl CircuitStatus {
    fn new() -> Self {
        Self {
            state: CircuitState::Closed,
            last_failure_time: None,
        }
    }
}

/// Thread-safe circuit breaker for a single LLM provider endpoint.
///
/// Uses `parking_lot::RwLock<CircuitStatus>` for atomic state transitions
/// that bundle state and timestamp updates. Atomic counters remain lock-free
/// for high-frequency success/failure recording on the hot path.
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    status: RwLock<CircuitStatus>,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    half_open_successes: AtomicU64,
    probes_in_flight: AtomicU64,
    node_id: String,
}

impl CircuitBreaker {
    /// Create a new circuit breaker for the given LLM node.
    pub fn new(node_id: impl Into<String>, config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            status: RwLock::new(CircuitStatus::new()),
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            half_open_successes: AtomicU64::new(0),
            probes_in_flight: AtomicU64::new(0),
            node_id: node_id.into(),
        }
    }

    /// Read the current circuit state (lock-free read via RwLock).
    pub fn state(&self) -> CircuitState {
        self.status.read().state
    }

    /// Check if a request is allowed through the circuit.
    ///
    /// Returns `Ok(())` if the request may proceed, or an error if the circuit
    /// is open and rejecting traffic.
    pub fn allow_request(&self) -> Result<(), CircuitBreakerError> {
        let status = self.status.read();
        let current_state = status.state;

        match current_state {
            CircuitState::Closed => Ok(()),

            CircuitState::Open => {
                let elapsed = status.last_failure_time.map(|t| t.elapsed());

                if let Some(elapsed) = elapsed {
                    if elapsed >= self.config.open_duration {
                        drop(status);
                        self.transition_to(CircuitState::HalfOpen);
                        tracing::info!(
                            node = self.node_id,
                            open_elapsed_ms = elapsed.as_millis(),
                            "Circuit transitioning to HalfOpen"
                        );
                        return Ok(());
                    }
                }

                Err(CircuitBreakerError::CircuitOpen {
                    node: self.node_id.clone(),
                    reason: "Failure threshold exceeded".to_string(),
                })
            }

            CircuitState::HalfOpen => {
                drop(status);
                let current_probes = self.probes_in_flight.fetch_add(1, Ordering::AcqRel);
                if current_probes >= self.config.half_open_max_attempts as u64 {
                    self.probes_in_flight.fetch_sub(1, Ordering::AcqRel);
                    Err(CircuitBreakerError::CircuitOpen {
                        node: self.node_id.clone(),
                        reason: "HalfOpen probe slot exhausted".to_string(),
                    })
                } else {
                    Ok(())
                }
            }
        }
    }

    /// Record a successful response from the provider.
    pub fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
        self.failure_count.store(0, Ordering::Release);

        let current_state = self.state();
        if current_state == CircuitState::HalfOpen {
            self.probes_in_flight.fetch_sub(1, Ordering::AcqRel);
            self.half_open_successes.fetch_add(1, Ordering::AcqRel);

            if self.half_open_successes.load(Ordering::Acquire)
                >= self.config.half_open_max_attempts as u64
            {
                self.transition_to(CircuitState::Closed);
                tracing::info!(
                    node = self.node_id,
                    "Circuit recovered - transitioned to Closed"
                );
            }
        }

        tracing::debug!(
            node = self.node_id,
            state = ?self.state(),
            success_count = self.success_count.load(Ordering::Relaxed),
            "Circuit success recorded"
        );
    }

    /// Record a failed response from the provider.
    ///
    /// Automatically trips the circuit if the failure threshold is exceeded.
    /// **Does NOT retry** - returns error for TMR degradation.
    pub fn record_failure(&self, status_code: Option<u16>) {
        let failures = self.failure_count.fetch_add(1, Ordering::AcqRel) + 1;

        {
            let mut status = self.status.write();
            status.last_failure_time = Some(Instant::now());
        }

        tracing::warn!(
            node = self.node_id,
            status_code = ?status_code,
            consecutive_failures = failures,
            threshold = self.config.failure_threshold,
            "Circuit failure recorded"
        );

        if failures >= self.config.failure_threshold as u64 {
            self.transition_to(CircuitState::Open);
            tracing::error!(
                node = self.node_id,
                failures,
                "CIRCUIT TRIPPED - Node isolated. No retries will be attempted."
            );
        }

        if self.state() == CircuitState::HalfOpen {
            self.probes_in_flight.fetch_sub(1, Ordering::AcqRel);
            self.transition_to(CircuitState::Open);
            tracing::warn!(
                node = self.node_id,
                "HalfOpen probe failed - circuit re-opened"
            );
        }
    }

    /// Transition to a new internal state with atomic status snapshot.
    fn transition_to(&self, new_state: CircuitState) {
        let mut status = self.status.write();
        status.state = new_state;

        match new_state {
            CircuitState::Closed => {
                self.failure_count.store(0, Ordering::Release);
                self.half_open_successes.store(0, Ordering::Release);
                self.probes_in_flight.store(0, Ordering::Release);
            }
            CircuitState::Open => {
                self.half_open_successes.store(0, Ordering::Release);
            }
            CircuitState::HalfOpen => {}
        }
    }

    /// Get diagnostic metrics for monitoring dashboards.
    pub fn metrics(&self) -> CircuitMetrics {
        CircuitMetrics {
            node_id: self.node_id.clone(),
            state: self.state(),
            consecutive_failures: self.failure_count.load(Ordering::Relaxed),
            total_successes: self.success_count.load(Ordering::Relaxed),
        }
    }

    /// Acquire a ProbeGuard for RAII-safe HalfOpen probe execution.
    ///
    /// Returns `Ok(ProbeGuard)` if the circuit allows a probe, or an error
    /// if the circuit is Open. The guard ensures `record_failure` is called
    /// automatically if dropped without explicit resolution, preventing
    /// probe stampede and in-flight counter leaks.
    pub fn acquire_probe_guard(self: &Arc<Self>) -> Result<ProbeGuard, CircuitBreakerError> {
        self.allow_request()?;
        Ok(ProbeGuard::new(Arc::clone(self)))
    }
}

/// Errors produced by the circuit breaker.
#[derive(Debug, thiserror::Error)]
pub enum CircuitBreakerError {
    #[error("Circuit OPEN for node '{node}': {reason}")]
    CircuitOpen { node: String, reason: String },

    #[error("Internal circuit error: {0}")]
    Internal(String),
}

/// RAII guard that ensures `record_failure` is called if a HalfOpen probe
/// is acquired but never explicitly resolved via `record_success` or
/// `record_failure`. Prevents probe leaks when the orchestrator times out
/// or drops the future before resolution.
///
/// ## Usage
/// ```ignore
/// let guard = circuit_breaker.acquire_probe_guard()?;
/// // ... perform request ...
/// guard.mark_success(); // or guard.mark_failure(Some(status_code));
/// // If dropped without marking, record_failure is called automatically.
/// ```
pub struct ProbeGuard {
    breaker: Arc<CircuitBreaker>,
    resolved: bool,
}

impl ProbeGuard {
    fn new(breaker: Arc<CircuitBreaker>) -> Self {
        Self {
            breaker,
            resolved: false,
        }
    }

    /// Mark the probe as successful. Disables the RAII failure-on-drop.
    pub fn mark_success(mut self) {
        self.resolved = true;
        self.breaker.record_success();
    }

    /// Mark the probe as failed with an optional HTTP status code.
    /// Disables the RAII failure-on-drop.
    pub fn mark_failure(mut self, status_code: Option<u16>) {
        self.resolved = true;
        self.breaker.record_failure(status_code);
    }
}

impl Drop for ProbeGuard {
    fn drop(&mut self) {
        if !self.resolved {
            tracing::warn!(
                node = self.breaker.node_id,
                "[CIRCUIT BREAKER] ProbeGuard dropped without explicit resolution - \
                 recording failure to prevent probe leak"
            );
            let current = self.breaker.probes_in_flight.load(Ordering::Acquire);
            if current > 0 {
                self.breaker.probes_in_flight.fetch_sub(1, Ordering::AcqRel);
            }
            self.breaker.record_failure(None);
        }
    }
}

/// Snapshot of circuit breaker state for external observability.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CircuitMetrics {
    pub node_id: String,
    pub state: CircuitState,
    pub consecutive_failures: u64,
    pub total_successes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_closed_allows_requests() {
        let cb = CircuitBreaker::new(
            "test-gemini",
            CircuitBreakerConfig {
                failure_threshold: 3,
                open_duration: Duration::from_millis(100),
                ..Default::default()
            },
        );

        assert!(cb.allow_request().is_ok());
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_trips_after_threshold() {
        let cb = CircuitBreaker::new(
            "test-deepseek",
            CircuitBreakerConfig {
                failure_threshold: 2,
                open_duration: Duration::from_secs(60),
                ..Default::default()
            },
        );

        cb.record_failure(Some(429));
        cb.record_failure(Some(500));

        assert_eq!(cb.state(), CircuitState::Open);
        assert!(cb.allow_request().is_err());
    }

    #[test]
    fn test_circuit_recovers_after_halfopen() {
        let cb = CircuitBreaker::new(
            "test-groq",
            CircuitBreakerConfig {
                failure_threshold: 2,
                open_duration: Duration::from_millis(50),
                ..Default::default()
            },
        );

        cb.record_failure(Some(429));
        cb.record_failure(Some(429));
        assert_eq!(cb.state(), CircuitState::Open);

        std::thread::sleep(Duration::from_millis(60));

        assert!(cb.allow_request().is_ok());
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }
}
