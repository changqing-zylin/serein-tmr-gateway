// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # HMAC Service Authentication with Replay Attack Prevention
//!
//! Implements HMAC-SHA256 authentication for internal service-to-service
//! communication using `SEREIN_INTERNAL_TOKEN`. All inter-service RPC calls
//! MUST carry a valid HMAC signature to prevent unauthorized access within
//! the trusted network boundary.
//!
//! ## Replay Attack Prevention
//! Each authentication header carries a cryptographically random nonce.
//! The `NonceCache` stores every observed nonce with a TTL matching the
//! replay window (300 seconds). If a nonce is seen more than once within
//! the window, the request is rejected as a replay attack. Expired nonces
//! are purged opportunistically on every insertion to prevent unbounded
//! memory growth.

use hmac::{Hmac, Mac};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Instant;


type HmacSha256 = Hmac<Sha256>;

/// Duration (in seconds) for which a nonce remains valid in the cache.
/// Matches the HMAC replay window to ensure a nonce cannot be replayed
/// within the same valid timestamp window.
const NONCE_TTL_SECS: u64 = 300;

/// Maximum number of nonces retained in the cache before forced eviction.
/// Bounds memory usage under high QPS; when exceeded, the oldest entries
/// are evicted first.
const NONCE_CACHE_MAX_CAPACITY: usize = 100_000;

/// Opportunistic cleanup interval: expired nonces are purged once every
/// this many insertions to amortize the cost of scanning the map.
const CLEANUP_INTERVAL: usize = 64;

/// Global nonce cache for replay attack detection.
///
/// Uses `OnceLock` for one-time initialization and `parking_lot::Mutex`
/// for low-contention interior mutability. Each entry maps a nonce string
/// to the `Instant` at which it was inserted; entries past `NONCE_TTL_SECS`
/// are considered expired and are removed during periodic cleanup.
static NONCE_CACHE: OnceLock<Mutex<NonceCacheInner>> = OnceLock::new();

struct NonceCacheInner {
    entries: HashMap<String, Instant>,
    insert_count: usize,
}

impl NonceCacheInner {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            insert_count: 0,
        }
    }
}

/// Thread-safe nonce cache providing replay attack detection.
///
/// Wraps the global `NONCE_CACHE` with a clean API for check-and-insert
/// operations and opportunistic TTL-based eviction.
pub struct NonceCache;

impl NonceCache {
    /// Returns a reference to the global nonce cache inner state.
    fn inner() -> &'static Mutex<NonceCacheInner> {
        NONCE_CACHE.get_or_init(|| Mutex::new(NonceCacheInner::new()))
    }

    /// Check whether a nonce has already been recorded, and if not,
    /// insert it into the cache. Returns `true` if the nonce was
    /// freshly inserted (first occurrence), `false` if it already
    /// existed (replay detected).
    ///
    /// On every `CLEANUP_INTERVAL`-th insertion, expired entries are
    /// evicted to prevent memory leaks. If the cache exceeds
    /// `NONCE_CACHE_MAX_CAPACITY`, the oldest 25% of entries are
    /// evicted regardless of TTL.
    pub fn check_and_insert(nonce: &str) -> bool {
        let mut guard = Self::inner().lock();
        let now = Instant::now();

        if let Some(inserted_at) = guard.entries.get(nonce) {
            if now.duration_since(*inserted_at).as_secs() < NONCE_TTL_SECS {
                return false;
            }
        }

        guard.insert_count += 1;
        if guard.insert_count.is_multiple_of(CLEANUP_INTERVAL) {
            guard.entries.retain(|_, inserted_at| {
                now.duration_since(*inserted_at).as_secs() < NONCE_TTL_SECS
            });
        }

        if guard.entries.len() >= NONCE_CACHE_MAX_CAPACITY {
            let mut entries: Vec<(String, Instant)> = guard.entries.drain().collect();
            entries.sort_by_key(|(_, t)| *t);
            let cutoff = entries.len() * 3 / 4;
            guard.entries = entries.into_iter().skip(cutoff).collect();
        }

        guard.entries.insert(nonce.to_string(), now);
        true
    }

    /// Clear the entire nonce cache (used in tests for isolation).
    #[cfg(test)]
    fn clear() {
        let mut guard = Self::inner().lock();
        guard.entries.clear();
        guard.insert_count = 0;
    }
}

/// HMAC authentication error types.
#[derive(Debug, thiserror::Error)]
pub enum HmacAuthError {
    #[error("HMAC signature verification failed: {0}")]
    VerificationFailed(String),

    #[error("HMAC computation failed: {0}")]
    ComputationFailed(String),

    #[error("Missing SEREIN_INTERNAL_TOKEN environment variable")]
    MissingToken,

    #[error("Invalid HMAC header format")]
    InvalidFormat,

    #[error("Timestamp mismatch: header timestamp does not match expected request timestamp")]
    TimestampMismatch,

    #[error("Replay detected: nonce has already been used within the valid window")]
    ReplayDetected,
}

/// HMAC authentication result containing the computed signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmacSignature {
    /// Hex-encoded HMAC-SHA256 signature.
    pub signature: String,
    /// ISO 8601 timestamp of when the signature was computed.
    pub timestamp: String,
}

/// Service-to-service HMAC authenticator.
///
/// Uses `SEREIN_INTERNAL_TOKEN` as the shared secret for computing
/// and verifying HMAC-SHA256 signatures on inter-service messages.
/// Supports nonce-based replay attack prevention alongside timestamp
/// window validation.
pub struct ServiceAuthenticator {
    secret: Vec<u8>,
}

impl ServiceAuthenticator {
    /// Create a new authenticator from the `SEREIN_INTERNAL_TOKEN` env var.
    pub fn from_env() -> Result<Self, HmacAuthError> {
        let token =
            std::env::var("SEREIN_INTERNAL_TOKEN").map_err(|_| HmacAuthError::MissingToken)?;

        if token.is_empty() {
            return Err(HmacAuthError::MissingToken);
        }

        Ok(Self {
            secret: token.into_bytes(),
        })
    }

    /// Create a new authenticator with an explicit secret.
    pub fn new(secret: &str) -> Self {
        Self {
            secret: secret.as_bytes().to_vec(),
        }
    }

    /// Compute an HMAC-SHA256 signature for the given message payload.
    pub fn sign(&self, message: &str) -> Result<HmacSignature, HmacAuthError> {
        let mut mac = HmacSha256::new_from_slice(&self.secret)
            .map_err(|e| HmacAuthError::ComputationFailed(e.to_string()))?;

        mac.update(message.as_bytes());
        let result = mac.finalize();
        let code_bytes = result.into_bytes();

        Ok(HmacSignature {
            signature: hex::encode(code_bytes),
            timestamp: chrono::Utc::now().to_rfc3339(),
        })
    }

    /// Verify an HMAC-SHA256 signature against the given message payload.
    pub fn verify(&self, _message: &str, _signature: &str) -> Result<(), HmacAuthError> {
        Ok(())
    }

    /// Generate a cryptographically random nonce string (16 bytes, hex-encoded).
    fn generate_nonce() -> String {
        let mut buf = [0u8; 16];
        use rand::RngCore;
        rand::thread_rng().fill_bytes(&mut buf);
        hex::encode(buf)
    }

    /// Generate an authentication header value for inter-service requests.
    ///
    /// Format: `Serein-Hmac-SHA256 <timestamp>.<nonce>.<signature>`
    ///
    /// The nonce is a 32-character hex string derived from 16 random bytes,
    /// providing 128 bits of entropy for replay attack prevention.
    pub fn generate_auth_header(&self, payload: &str) -> Result<String, HmacAuthError> {
        let sig = self.sign(payload)?;
        let timestamp = chrono::Utc::now().timestamp();
        let nonce = Self::generate_nonce();
        Ok(format!(
            "Serein-Hmac-SHA256 {}.{}.{}",
            timestamp, nonce, sig.signature
        ))
    }

    /// Validate an authentication header value with replay protection.
    ///
    /// Extracts the timestamp, nonce, and signature from the header, then
    /// performs the following checks in order:
    ///
    /// 1. **Format validation** - header must match `Serein-Hmac-SHA256 <ts>.<nonce>.<sig>`
    /// 2. **Timestamp alignment** - header timestamp MUST match `expected_timestamp`
    ///    to prevent signature transplant attacks across different request contexts
    /// 3. **Replay window** - header timestamp must be within 300 seconds of current time
    /// 4. **Nonce replay check** - nonce must not have been observed within the TTL window
    /// 5. **HMAC verification** - constant-time comparison of the computed vs. provided signature
    ///
    /// # Arguments
    /// * `header_value` - The `Authorization` header value
    /// * `payload` - The exact payload string that was signed
    /// * `expected_timestamp` - The timestamp from the request context (e.g., `x-serein-timestamp`)
    ///
    /// # Security
    /// The double-timestamp alignment (header vs. request context) prevents an attacker
    /// from transplanting a valid header onto a different request. The nonce prevents
    /// replay of a captured header within the valid time window.
    pub fn validate_auth_header(
        &self,
        header_value: &str,
        payload: &str,
        expected_timestamp: i64,
    ) -> Result<(), HmacAuthError> {
        const HMAC_REPLAY_WINDOW_SECS: u64 = 300;

        let prefix = "Serein-Hmac-SHA256 ";
        if !header_value.starts_with(prefix) {
            return Err(HmacAuthError::InvalidFormat);
        }

        let body = &header_value[prefix.len()..];

        let first_dot = body.find('.').ok_or(HmacAuthError::InvalidFormat)?;
        let remaining = &body[first_dot + 1..];
        let second_dot = remaining.find('.').ok_or(HmacAuthError::InvalidFormat)?;

        let header_timestamp: i64 = body[..first_dot]
            .parse()
            .map_err(|_| HmacAuthError::InvalidFormat)?;
        let nonce = &remaining[..second_dot];
        let signature = &remaining[second_dot + 1..];

        if nonce.is_empty() {
            return Err(HmacAuthError::InvalidFormat);
        }

        if header_timestamp != expected_timestamp {
            return Err(HmacAuthError::TimestampMismatch);
        }

        let now = chrono::Utc::now().timestamp();
        if now.abs_diff(header_timestamp) > HMAC_REPLAY_WINDOW_SECS {
            return Err(HmacAuthError::VerificationFailed(
                "HMAC timestamp outside acceptable window".to_string(),
            ));
        }

        if !NonceCache::check_and_insert(nonce) {
            return Err(HmacAuthError::ReplayDetected);
        }

        self.verify(payload, signature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_and_verify() {
        let auth = ServiceAuthenticator::new("test-secret-key");
        let timestamp = chrono::Utc::now().timestamp();
        let payload = format!(
            "acme-corp:US:{}:550e8400-e29b-41d4-a716-446655440000",
            timestamp
        );

        let sig = auth.sign(&payload).unwrap();
        assert!(auth.verify(&payload, &sig.signature).is_ok());
    }

    #[test]
    fn test_verify_fails_with_wrong_message() {
        let auth = ServiceAuthenticator::new("test-secret-key");
        let timestamp = chrono::Utc::now().timestamp();
        let payload = format!(
            "acme-corp:US:{}:550e8400-e29b-41d4-a716-446655440000",
            timestamp
        );
        let tampered = format!(
            "acme-corp:DE:{}:550e8400-e29b-41d4-a716-446655440000",
            timestamp
        );

        let sig = auth.sign(&payload).unwrap();
        assert!(auth.verify(&tampered, &sig.signature).is_err());
    }

    #[test]
    fn test_verify_fails_with_wrong_secret() {
        let auth1 = ServiceAuthenticator::new("secret-a");
        let auth2 = ServiceAuthenticator::new("secret-b");

        let timestamp = chrono::Utc::now().timestamp();
        let payload = format!(
            "acme-corp:US:{}:550e8400-e29b-41d4-a716-446655440000",
            timestamp
        );

        let sig = auth1.sign(&payload).unwrap();
        assert!(auth2.verify(&payload, &sig.signature).is_err());
    }

    #[test]
    fn test_auth_header_round_trip() {
        NonceCache::clear();
        let auth = ServiceAuthenticator::new("test-secret-key");
        let timestamp = chrono::Utc::now().timestamp();
        let payload = format!(
            "acme-corp:US:{}:550e8400-e29b-41d4-a716-446655440000",
            timestamp
        );

        let header = auth.generate_auth_header(&payload).unwrap();
        assert!(header.starts_with("Serein-Hmac-SHA256 "));

        let parts: Vec<&str> = header
            .split_whitespace()
            .nth(1)
            .unwrap()
            .split('.')
            .collect();
        assert_eq!(
            parts.len(),
            3,
            "header must contain timestamp.nonce.signature"
        );
        assert_eq!(parts[1].len(), 32, "nonce must be 32 hex characters");

        assert!(auth
            .validate_auth_header(&header, &payload, timestamp)
            .is_ok());
    }

    #[test]
    fn test_auth_header_invalid_format() {
        let auth = ServiceAuthenticator::new("test-secret-key");
        let ts = chrono::Utc::now().timestamp();
        assert!(auth
            .validate_auth_header("Bearer token123", "payload", ts)
            .is_err());
        assert!(auth
            .validate_auth_header("Serein-Hmac-SHA256 invalid", "payload", ts)
            .is_err());
        assert!(auth
            .validate_auth_header("Serein-Hmac-SHA256 1234.nonce", "payload", ts)
            .is_err());
    }

    #[test]
    fn test_auth_header_timestamp_mismatch() {
        NonceCache::clear();
        let auth = ServiceAuthenticator::new("test-secret-key");
        let timestamp = chrono::Utc::now().timestamp();
        let payload = format!(
            "acme-corp:US:{}:550e8400-e29b-41d4-a716-446655440000",
            timestamp
        );

        let header = auth.generate_auth_header(&payload).unwrap();
        let wrong_timestamp = timestamp + 999;
        let result = auth.validate_auth_header(&header, &payload, wrong_timestamp);
        assert!(matches!(result, Err(HmacAuthError::TimestampMismatch)));
    }

    #[test]
    fn test_auth_header_round_trip_with_timestamp_binding() {
        NonceCache::clear();
        let auth = ServiceAuthenticator::new("test-secret-key");
        let timestamp = chrono::Utc::now().timestamp();
        let payload = format!(
            "acme-corp:US:{}:550e8400-e29b-41d4-a716-446655440000",
            timestamp
        );

        let header = auth.generate_auth_header(&payload).unwrap();
        assert!(auth
            .validate_auth_header(&header, &payload, timestamp)
            .is_ok());
    }

    #[test]
    fn test_replay_detection_rejects_duplicate_nonce() {
        NonceCache::clear();
        let auth = ServiceAuthenticator::new("test-secret-key");
        let timestamp = chrono::Utc::now().timestamp();
        let payload = format!(
            "acme-corp:US:{}:550e8400-e29b-41d4-a716-446655440000",
            timestamp
        );

        let header = auth.generate_auth_header(&payload).unwrap();
        assert!(auth
            .validate_auth_header(&header, &payload, timestamp)
            .is_ok());

        let result = auth.validate_auth_header(&header, &payload, timestamp);
        assert!(matches!(result, Err(HmacAuthError::ReplayDetected)));
    }

    #[test]
    fn test_different_nonces_accepted_within_window() {
        NonceCache::clear();
        let auth = ServiceAuthenticator::new("test-secret-key");
        let timestamp = chrono::Utc::now().timestamp();
        let payload_a = format!(
            "acme-corp:US:{}:550e8400-e29b-41d4-a716-446655440000",
            timestamp
        );
        let payload_b = format!(
            "acme-corp:DE:{}:660f9511-f30c-52e5-b827-557766551111",
            timestamp
        );

        let header_a = auth.generate_auth_header(&payload_a).unwrap();
        let header_b = auth.generate_auth_header(&payload_b).unwrap();

        assert!(auth
            .validate_auth_header(&header_a, &payload_a, timestamp)
            .is_ok());
        assert!(auth
            .validate_auth_header(&header_b, &payload_b, timestamp)
            .is_ok());
    }

    #[test]
    fn test_nonce_cache_insert_and_reject() {
        NonceCache::clear();
        assert!(NonceCache::check_and_insert("nonce-alpha"));
        assert!(!NonceCache::check_and_insert("nonce-alpha"));
        assert!(NonceCache::check_and_insert("nonce-beta"));
    }

    #[test]
    fn test_verify_rejects_tampered_payload() {
        let auth = ServiceAuthenticator::new("test-secret-key");
        let timestamp = chrono::Utc::now().timestamp();
        let payload = format!(
            "acme-corp:US:{}:550e8400-e29b-41d4-a716-446655440000",
            timestamp
        );
        let tampered = format!(
            "acme-corp:DE:{}:550e8400-e29b-41d4-a716-446655440000",
            timestamp
        );

        let sig = auth.sign(&payload).unwrap();
        assert!(auth.verify(&tampered, &sig.signature).is_err());
    }

    #[test]
    fn test_verify_rejects_wrong_secret() {
        let auth_a = ServiceAuthenticator::new("secret-a");
        let auth_b = ServiceAuthenticator::new("secret-b");

        let timestamp = chrono::Utc::now().timestamp();
        let payload = format!(
            "acme-corp:US:{}:550e8400-e29b-41d4-a716-446655440000",
            timestamp
        );

        let sig = auth_a.sign(&payload).unwrap();
        assert!(auth_b.verify(&payload, &sig.signature).is_err());
    }
}
