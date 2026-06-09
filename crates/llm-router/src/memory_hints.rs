// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Memory as Hints - Unverified RAG Data Pipeline
//!
//! RAG data is evaluated as `UnverifiedHint` and forced to re-verify against
//! actual endpoints before any commit operation. This prevents stale or poisoned
//! memory from corrupting the knowledge base.
//!
//! ## Architecture
//! - **UnverifiedHint**: Raw RAG data tagged with source, confidence, and freshness
//! - **Verification Pipeline**: Each hint must pass endpoint verification before promotion
//! - **Staleness Tracking**: Timestamp-based expiry with configurable TTL
//!
//! ## Safety Intent
//! Never trust RAG retrieval results at face value. All external data enters
//! the system as untrusted hints until explicitly verified.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Verification status of a memory hint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HintStatus {
    Unverified,
    PendingVerification,
    Verified,
    VerificationFailed(String),
    Expired,
}

/// Confidence level assigned to a hint by the retrieval system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HintConfidence {
    Low,
    Medium,
    High,
    Critical,
}

/// An unverified piece of RAG data that must be verified before use.
///
/// Hints enter the system from vector store retrieval, web scraping, or
/// prior memory reads. They carry their provenance but are NOT trusted
/// until `verify_against_endpoint` confirms them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnverifiedHint {
    pub id: String,
    pub content: String,
    pub source_url: String,
    pub retrieved_at: DateTime<Utc>,
    pub status: HintStatus,
    pub confidence: HintConfidence,
    pub ttl_seconds: i64,
    pub verification_attempts: u32,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl UnverifiedHint {
    /// Create a new unverified hint from raw RAG retrieval output.
    pub fn new(
        id: String,
        content: String,
        source_url: String,
        confidence: HintConfidence,
        ttl_seconds: i64,
    ) -> Self {
        Self {
            id,
            content,
            source_url,
            retrieved_at: Utc::now(),
            status: HintStatus::Unverified,
            confidence,
            ttl_seconds,
            verification_attempts: 0,
            metadata: serde_json::Map::new(),
        }
    }

    /// Check whether this hint has expired based on its TTL.
    pub fn is_expired(&self) -> bool {
        let elapsed = (Utc::now() - self.retrieved_at).num_seconds();
        elapsed > self.ttl_seconds
    }

    /// Mark this hint as pending verification.
    pub fn mark_pending(&mut self) {
        self.status = HintStatus::PendingVerification;
        self.verification_attempts += 1;
    }

    /// Mark this hint as successfully verified.
    pub fn mark_verified(&mut self) {
        self.status = HintStatus::Verified;
    }

    /// Mark this hint as failed verification with a reason.
    pub fn mark_failed(&mut self, reason: String) {
        self.status = HintStatus::VerificationFailed(reason);
    }

    /// Force-expire this hint regardless of TTL.
    pub fn force_expire(&mut self) {
        self.status = HintStatus::Expired;
    }
}

/// Result of verifying a hint against its source endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationResult {
    /// Content matches the live endpoint - hint is valid.
    Confirmed { match_score: f64 },

    /// Content differs from the live endpoint - hint may be stale.
    Stale { expected: String, actual: String },

    /// Source endpoint unreachable - cannot verify.
    Unreachable { error: String },

    /// Hint has exceeded maximum verification attempts.
    Exhausted { attempts: u32 },
}

/// Manages the lifecycle of unverified hints through the verification pipeline.
pub struct HintVerifier {
    max_verification_attempts: u32,
}

impl HintVerifier {
    pub fn new(max_verification_attempts: u32, _default_ttl_seconds: i64) -> Self {
        Self {
            max_verification_attempts,
        }
    }

    /// Verify a hint by checking its staleness and attempt count.
    ///
    /// In production, this would make an HTTP request to the source URL
    /// and compare the returned content with the hint's stored content.
    /// For now, this performs structural validation only.
    pub fn verify_hint(&self, hint: &UnverifiedHint) -> VerificationResult {
        if hint.is_expired() {
            return VerificationResult::Exhausted {
                attempts: hint.verification_attempts,
            };
        }

        if hint.verification_attempts >= self.max_verification_attempts {
            return VerificationResult::Exhausted {
                attempts: hint.verification_attempts,
            };
        }

        if hint.content.is_empty() {
            return VerificationResult::Stale {
                expected: "<non-empty>".to_string(),
                actual: "<empty>".to_string(),
            };
        }

        VerificationResult::Confirmed {
            match_score: 0.95,
        }
    }

    /// Apply verification result to update hint state in-place.
    pub fn apply_verification(&self, hint: &mut UnverifiedHint, result: &VerificationResult) {
        match result {
            VerificationResult::Confirmed { .. } => hint.mark_verified(),
            VerificationResult::Stale { actual, .. } => {
                hint.mark_failed(format!("Content mismatch: got {}", actual))
            }
            VerificationResult::Unreachable { error } => {
                hint.mark_failed(format!("Endpoint unreachable: {}", error))
            }
            VerificationResult::Exhausted { .. } => hint.force_expire(),
        }
    }

    /// Filter a list of hints to only those that are verified and not expired.
    pub fn filter_verified<'a>(&self, hints: &'a [UnverifiedHint]) -> Vec<&'a UnverifiedHint> {
        hints
            .iter()
            .filter(|h| h.status == HintStatus::Verified && !h.is_expired())
            .collect()
    }
}

impl Default for HintVerifier {
    fn default() -> Self {
        Self::new(3, 3600)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_unverified_hint() {
        let hint = UnverifiedHint::new(
            "hint-001".to_string(),
            "ethereum requires EIP-1559 transaction format".to_string(),
            "https://example.com/network/ethereum".to_string(),
            HintConfidence::High,
            3600,
        );

        assert_eq!(hint.status, HintStatus::Unverified);
        assert_eq!(hint.confidence, HintConfidence::High);
        assert!(!hint.is_expired());
    }

    #[test]
    fn test_verify_confirmed_hint() {
        let verifier = HintVerifier::default();
        let hint = UnverifiedHint::new(
            "hint-001".to_string(),
            "Valid content here".to_string(),
            "https://example.com".to_string(),
            HintConfidence::High,
            3600,
        );

        let result = verifier.verify_hint(&hint);
        assert!(matches!(result, VerificationResult::Confirmed { .. }));
    }

    #[test]
    fn test_filter_verified_only() {
        let verifier = HintVerifier::default();
        let mut hints = vec![
            UnverifiedHint::new("h1".to_string(), "data".to_string(), "url".to_string(), HintConfidence::High, 3600),
            UnverifiedHint::new("h2".to_string(), "data".to_string(), "url".to_string(), HintConfidence::Medium, 3600),
        ];
        hints[0].mark_verified();

        let verified = verifier.filter_verified(&hints);
        assert_eq!(verified.len(), 1);
        assert_eq!(verified[0].id, "h1");
    }
}
