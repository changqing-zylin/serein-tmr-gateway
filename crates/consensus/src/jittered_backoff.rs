// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Jittered Exponential Backoff - Anti-Throttling Retry Strategy
//!
//! Implements randomized exponential backoff to prevent synchronized request spikes
//! that trigger LLM provider rate limiters (especially Google Gemini).
//!
//! ## Design Rationale
//! - **Full Jitter**: `random_between(0, base * 2^attempt)` - optimal for distributed systems
//! - **Decorrelated Jitter**: `random_between(base, 3 * previous_delay)` - better tail latency
//! - **HTTP 429/5xx**: Retryable server errors use Full Jitter to avoid thundering herd
//! - **No aggressive retries**: Max 3 attempts for transient errors only.

use rand::Rng;
use rand::SeedableRng;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryableHttpStatus {
    TooManyRequests,
    ServerError(u16),
}

impl RetryableHttpStatus {
    pub fn from_status_code(code: u16) -> Option<Self> {
        match code {
            429 => Some(RetryableHttpStatus::TooManyRequests),
            500..=599 => Some(RetryableHttpStatus::ServerError(code)),
            _ => None,
        }
    }

    pub fn status_code(&self) -> u16 {
        match self {
            RetryableHttpStatus::TooManyRequests => 429,
            RetryableHttpStatus::ServerError(c) => *c,
        }
    }
}

/// Supported jitter strategies for backoff calculation.
#[derive(Debug, Clone, Copy, Default)]
pub enum JitterStrategy {
    #[default]
    Full,
    /// Decorrelated jitter: random in [base, 3 * previous_delay)
    Decorrelated,
    /// Equal jitter: (cap / 2) + random(0, cap / 2)
    Equal,
}

/// Configuration for the backoff policy.
#[derive(Debug, Clone)]
pub struct BackoffConfig {
    /// Initial delay before first retry.
    pub base_delay: Duration,
    /// Maximum delay ceiling (prevents unbounded growth).
    pub max_delay: Duration,
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: u32,
    /// Jitter strategy to apply.
    pub jitter_strategy: JitterStrategy,
    /// Multiplicative factor for exponential growth.
    pub multiplier: f64,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            base_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(30),
            max_retries: 3,
            jitter_strategy: JitterStrategy::default(),
            multiplier: 2.0,
        }
    }
}

/// Stateful backoff calculator with jitter.
///
/// Each instance tracks its own attempt counter and previous delay for
/// decorrelated jitter calculations.
pub struct JitteredBackoff {
    config: BackoffConfig,
    current_attempt: u32,
    previous_delay: Duration,
    rng: rand::rngs::StdRng,
}

impl JitteredBackoff {
    /// Create a new backoff instance with default configuration.
    pub fn new() -> Self {
        Self::with_config(BackoffConfig::default())
    }

    /// Create a new backoff instance with custom configuration.
    pub fn with_config(config: BackoffConfig) -> Self {
        let base_delay = config.base_delay;
        Self {
            config,
            current_attempt: 0,
            previous_delay: base_delay,
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }

    /// Calculate the next backoff duration with applied jitter.
    ///
    /// Returns `None` if all retry attempts are exhausted.
    pub fn next_delay(&mut self) -> Option<Duration> {
        if self.current_attempt >= self.config.max_retries {
            return None;
        }

        let delay = match self.config.jitter_strategy {
            JitterStrategy::Full => self.full_jitter(),
            JitterStrategy::Decorrelated => self.decorrelated_jitter(),
            JitterStrategy::Equal => self.equal_jitter(),
        };

        let capped = delay.min(self.config.max_delay);
        self.previous_delay = capped;
        self.current_attempt += 1;

        tracing::debug!(
            attempt = self.current_attempt,
            delay_ms = capped.as_millis(),
            strategy = ?self.config.jitter_strategy,
            "Backoff delay calculated"
        );

        Some(capped)
    }

    /// Full jitter: uniform random in [0, min(cap, base * 2^attempt)]
    fn full_jitter(&mut self) -> Duration {
        let exp = self.config.multiplier.powi(self.current_attempt as i32);
        let raw = self.config.base_delay.as_secs_f64() * exp;
        let cap = self.config.max_delay.as_secs_f64().min(raw);
        let jittered = self.rng.gen::<f64>() * cap;

        Duration::from_secs_f64(jittered.max(0.001))
    }

    /// Decorrelated jitter: random in [base, 3 * previous_delay)
    fn decorrelated_jitter(&mut self) -> Duration {
        let lower = self.config.base_delay.as_secs_f64();
        let upper = 3.0 * self.previous_delay.as_secs_f64();
        let jittered = self.rng.gen_range(lower..upper);

        Duration::from_secs_f64(jittered)
    }

    /// Equal jitter: (cap / 2) + random(0, cap / 2)
    fn equal_jitter(&mut self) -> Duration {
        let exp = self.config.multiplier.powi(self.current_attempt as i32);
        let raw = self.config.base_delay.as_secs_f64() * exp;
        let cap = self.config.max_delay.as_secs_f64().min(raw);
        let half = cap / 2.0;
        let jittered = half + (self.rng.gen::<f64>() * half);

        Duration::from_secs_f64(jittered)
    }

    /// Reset the attempt counter (e.g., after a successful operation).
    pub fn reset(&mut self) {
        self.current_attempt = 0;
        self.previous_delay = self.config.base_delay;
    }

    /// Get the number of attempts made so far.
    pub fn attempts(&self) -> u32 {
        self.current_attempt
    }

    /// Check if more retries are available.
    pub fn can_retry(&self) -> bool {
        self.current_attempt < self.config.max_retries
    }

    /// Async sleep with the next calculated backoff delay.
    ///
    /// Returns `Ok(())` after sleeping, or `None` if retries exhausted.
    pub async fn wait_next(&mut self) -> Result<(), ()> {
        match self.next_delay() {
            Some(delay) => {
                tokio::time::sleep(delay).await;
                Ok(())
            }
            None => Err(()),
        }
    }
}

impl Default for JitteredBackoff {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute an async operation with automatic jittered backoff retry.
///
/// Retries on transient network errors AND HTTP 429/5xx responses.
/// Uses Full Jitter strategy for HTTP errors to prevent thundering herd
/// when multiple clients hit the same rate-limited endpoint simultaneously.
///
/// # Arguments
/// * `operation` - The async function to execute. Receives attempt number.
/// * `config` - Optional custom backoff configuration.
///
/// # Returns
/// - `Ok(T)` - The successful result from the operation.
/// - `Err(E)` - The last error after all retries exhausted.
pub async fn execute_with_backoff<F, Fut, T, E>(
    operation: F,
    config: Option<BackoffConfig>,
) -> Result<T, E>
where
    F: Fn(u32) -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let mut backoff = JitteredBackoff::with_config(config.unwrap_or_default());

    loop {
        match operation(backoff.attempts()).await {
            Ok(result) => {
                tracing::debug!(
                    attempts = backoff.attempts(),
                    "Operation succeeded"
                );
                return Ok(result);
            }
            Err(err) => {
                if !backoff.can_retry() {
                    tracing::warn!(
                        attempts = backoff.attempts(),
                        error = ?err,
                        "All retry attempts exhausted"
                    );
                    return Err(err);
                }

                let delay = backoff.next_delay().unwrap();

                tracing::info!(
                    attempt = backoff.attempts(),
                    error = ?err,
                    delay_ms = delay.as_millis(),
                    "Transient error - retrying with jittered backoff"
                );

                tokio::time::sleep(delay).await;
            }
        }
    }
}

/// Outcome of an HTTP operation that may produce a retryable status code.
#[derive(Debug, Clone)]
pub enum HttpOutcome<T> {
    Success(T),
    RetryableStatus(u16),
}

/// Execute an async HTTP operation with jittered backoff on 429/5xx.
///
/// When the operation returns a retryable HTTP status (429 or 5xx),
/// applies Full Jitter exponential backoff before retrying. This prevents
/// thundering herd scenarios where multiple clients retry simultaneously
/// after a rate limit or server error.
///
/// # Arguments
/// * `operation` - The async function to execute. Returns `HttpOutcome<T>` for
///   successful responses or `Err(E)` for network/transport errors.
/// * `config` - Optional custom backoff configuration.
///
/// # Returns
/// - `Ok(T)` - The successful result from the operation.
/// - `Err(HttpBackoffError<E>)` - The last error after all retries exhausted.
pub async fn execute_with_http_backoff<F, Fut, T, E>(
    operation: F,
    config: Option<BackoffConfig>,
) -> Result<T, HttpBackoffError<E>>
where
    F: Fn(u32) -> Fut,
    Fut: std::future::Future<Output = Result<HttpOutcome<T>, E>>,
    E: std::fmt::Debug,
{
    let mut backoff = JitteredBackoff::with_config(config.unwrap_or_else(|| BackoffConfig {
        jitter_strategy: JitterStrategy::Full,
        ..Default::default()
    }));

    loop {
        match operation(backoff.attempts()).await {
            Ok(HttpOutcome::Success(result)) => {
                tracing::debug!(
                    attempts = backoff.attempts(),
                    "HTTP operation succeeded"
                );
                return Ok(result);
            }
            Ok(HttpOutcome::RetryableStatus(code)) => {
                if !backoff.can_retry() {
                    tracing::warn!(
                        status_code = code,
                        attempts = backoff.attempts(),
                        "HTTP {} - all retry attempts exhausted", code
                    );
                    return Err(HttpBackoffError::RetryableStatusExhausted { status_code: code, attempts: backoff.attempts() });
                }

                let delay = backoff.next_delay().unwrap();
                tracing::info!(
                    status_code = code,
                    attempt = backoff.attempts(),
                    delay_ms = delay.as_millis(),
                    "HTTP {} - retrying with Full Jitter backoff", code
                );
                tokio::time::sleep(delay).await;
            }
            Err(err) => {
                if !backoff.can_retry() {
                    tracing::warn!(
                        attempts = backoff.attempts(),
                        error = ?err,
                        "All retry attempts exhausted"
                    );
                    return Err(HttpBackoffError::Transport(err));
                }

                let delay = backoff.next_delay().unwrap();
                tracing::info!(
                    attempt = backoff.attempts(),
                    error = ?err,
                    delay_ms = delay.as_millis(),
                    "Network error - retrying with jittered backoff"
                );
                tokio::time::sleep(delay).await;
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum HttpBackoffError<E> {
    #[error("HTTP {status_code} after {attempts} retry attempts")]
    RetryableStatusExhausted { status_code: u16, attempts: u32 },
    #[error("Transport error after retries: {0:?}")]
    Transport(E),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_jitter_increases_with_attempts() {
        let mut backoff = JitteredBackoff::with_config(BackoffConfig {
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            max_retries: 5,
            ..Default::default()
        });

        let delays: Vec<Duration> = (0..5).filter_map(|_| backoff.next_delay()).collect();

        assert_eq!(delays.len(), 5);

        for window in delays.windows(2) {
            let avg_earlier = (window[0].as_millis() + window[1].as_millis()) / 2;
            assert!(avg_earlier > 0, "Delays should be positive");
        }
    }

    #[tokio::test]
    async fn test_execute_with_backoff_success_on_first_try() {
        let result = execute_with_backoff(
            |_attempt| async { Ok::<_, String>("success".to_string()) },
            Some(BackoffConfig {
                max_retries: 3,
                ..Default::default()
            }),
        )
        .await;

        assert_eq!(result.unwrap(), "success");
    }

    #[tokio::test]
    async fn test_execute_with_backoff_retries_then_succeeds() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let config = Some(BackoffConfig {
            base_delay: Duration::from_millis(10),
            max_retries: 5,
            ..Default::default()
        });

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();
        let op = move |_: u32| -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send>> {
            let count = attempts_clone.fetch_add(1, Ordering::SeqCst) + 1;
            if count < 3 {
                Box::pin(async move { Err("transient_error".to_string()) })
            } else {
                Box::pin(async move { Ok("recovered".to_string()) })
            }
        };

        let result = execute_with_backoff(op, config).await;

        assert_eq!(result.unwrap(), "recovered");
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut backoff = JitteredBackoff::with_config(BackoffConfig {
            max_retries: 3,
            ..Default::default()
        });

        let _ = backoff.next_delay();
        let _ = backoff.next_delay();
        assert_eq!(backoff.attempts(), 2);

        backoff.reset();
        assert_eq!(backoff.attempts(), 0);
        assert!(backoff.can_retry());
    }
}
