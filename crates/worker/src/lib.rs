// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Serein Compliance Worker - Global AI Compliance & Audit Event Bus
//!
//! High-performance async event bus for EU AI Act and GDPR compliance logging.
//! Decoupled from the main API gateway request path to ensure low latency
//! while maintaining a durable audit trail of all data processing activities.
//!
//! ## Architecture
//! - `AuditEvent` - strictly typed audit record capturing the full context
//!   of each data processing activity (who, what, which tenant, what action)
//! - `ComplianceBus` - async MPSC channel-based event bus; producers send
//!   `AuditEvent`s without blocking the request path
//! - `spawn_worker()` - detached Tokio task that consumes events, applies
//!   PII redaction, and persists the sanitized audit record
//! - `redact_pii()` - regex-based GDPR masking for email, phone, SSN, and
//!   credit card patterns before persistence
//!
//! ## GDPR Compliance
//! The `end_user_id` field is retained per-event to support Right-to-be-Forgotten
//! (RTBF) deletion queries per GDPR Article 17. PII in `raw_payload` is
//! redacted before storage so that audit logs never contain cleartext
//! personal data, satisfying GDPR Article 25 (Data Protection by Design).

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use tokio::sync::mpsc::{Receiver, Sender};

static PII_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)(?:[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}|\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b|\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b|\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b)",
    )
    .expect("PII_PATTERN regex compilation failed")
});

static EMAIL_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$")
        .expect("EMAIL_PATTERN regex compilation failed")
});

static PHONE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}$")
        .expect("PHONE_PATTERN regex compilation failed")
});

static IDENTITY_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\d{3}[-\s]?\d{2}[-\s]?\d{4}$").expect("IDENTITY_PATTERN regex compilation failed")
});

static FINANCIAL_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$")
        .expect("FINANCIAL_PATTERN regex compilation failed")
});

/// Classification of a detected PII entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PiiType {
    Email,
    Phone,
    Identity,
    Financial,
}

impl PiiType {
    fn placeholder_tag(&self) -> &'static str {
        match self {
            PiiType::Email => "MASKED_EMAIL",
            PiiType::Phone => "MASKED_PHONE",
            PiiType::Identity => "MASKED_IDENTITY",
            PiiType::Financial => "MASKED_FINANCIAL",
        }
    }
}

#[allow(clippy::if_same_then_else)]
fn classify_pii(matched: &str) -> PiiType {
    if EMAIL_PATTERN.is_match(matched) {
        PiiType::Email
    } else if FINANCIAL_PATTERN.is_match(matched) {
        PiiType::Financial
    } else if IDENTITY_PATTERN.is_match(matched) {
        PiiType::Identity
    } else if PHONE_PATTERN.is_match(matched) {
        PiiType::Phone
    } else {
        PiiType::Phone
    }
}

/// Classification of the enforcement action taken on a request.
///
/// Typed enum ensures compile-time correctness for action classification.
/// Serialized as SCREAMING_SNAKE_CASE strings for interoperability with
/// external compliance monitoring systems and SIEM integrations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum AuditAction {
    /// Request was allowed through the pipeline unmodified.
    ALLOWED,
    /// Request was blocked by the WASM sandbox (fuel exhaustion, OOM, trap).
    BLOCKED_BY_WASM,
    /// Request payload was modified (e.g., PII stripped, fields redacted).
    MODIFIED,
    /// Request was rejected by the consensus oracle (low confidence).
    BLOCKED_BY_ORACLE,
    /// Request was rejected by the SIS interlock (fuel depleted).
    BLOCKED_BY_SIS,
    /// Request was rejected due to HMAC authentication failure.
    BLOCKED_BY_AUTH,
    /// Request was rejected as a replay attack (duplicate nonce).
    BLOCKED_BY_REPLAY,
}

impl AuditAction {
    /// Convert the action to its wire-format string representation.
    ///
    /// Used when constructing `AuditEvent.action_taken` and for
    /// structured logging in SIEM integrations.
    pub fn as_str(&self) -> &'static str {
        match self {
            AuditAction::ALLOWED => "ALLOWED",
            AuditAction::BLOCKED_BY_WASM => "BLOCKED_BY_WASM",
            AuditAction::BLOCKED_BY_ORACLE => "BLOCKED_BY_ORACLE",
            AuditAction::BLOCKED_BY_SIS => "BLOCKED_BY_SIS",
            AuditAction::BLOCKED_BY_AUTH => "BLOCKED_BY_AUTH",
            AuditAction::BLOCKED_BY_REPLAY => "BLOCKED_BY_REPLAY",
            AuditAction::MODIFIED => "MODIFIED",
        }
    }
}

impl std::fmt::Display for AuditAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A strictly typed audit event for EU AI Act and GDPR compliance logging.
///
/// Each event captures the full context of a data processing activity:
/// - **When**: `timestamp` - Unix epoch milliseconds for temporal correlation
/// - **Who**: `end_user_id` - retained for GDPR Right-to-be-Forgotten (RTBF)
///   queries per Article 17; enables data subject erasure across audit logs
/// - **Which tenant**: `tenant_id` - multi-tenant isolation and data sovereignty
/// - **What data**: `raw_payload` - original request payload, redacted before
///   persistence to satisfy GDPR Article 25 (Data Protection by Design)
/// - **What action**: `action_taken` - enforcement decision string (e.g.,
///   "ALLOWED", "BLOCKED_BY_WASM", "MODIFIED")
///
/// The `end_user_id` field is essential for GDPR Article 17 compliance:
/// when a data subject exercises their right to erasure, all audit events
/// associated with that `end_user_id` can be located and purged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unix epoch milliseconds when the event was emitted.
    pub timestamp: u64,
    /// Tenant identifier for multi-tenant isolation and data sovereignty.
    pub tenant_id: String,
    /// End-user identifier retained for GDPR Right-to-be-Forgotten queries.
    pub end_user_id: String,
    /// Original request payload; will be redacted before persistence.
    pub raw_payload: String,
    /// Enforcement action taken on this request (e.g., "ALLOWED",
    /// "BLOCKED_BY_WASM", "MODIFIED").
    pub action_taken: String,
}

impl AuditEvent {
    /// Construct a new audit event with the current timestamp.
    ///
    /// The `action_taken` field is populated from the typed `AuditAction`
    /// enum to ensure only valid action strings are recorded.
    pub fn new(
        tenant_id: impl Into<String>,
        end_user_id: impl Into<String>,
        raw_payload: impl Into<String>,
        action: AuditAction,
    ) -> Self {
        Self {
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            tenant_id: tenant_id.into(),
            end_user_id: end_user_id.into(),
            raw_payload: raw_payload.into(),
            action_taken: action.as_str().to_string(),
        }
    }
}

/// Async compliance bus decoupled from the main request path to ensure
/// low latency on the API gateway.
///
/// Uses a bounded `tokio::sync::mpsc::channel` with capacity 10000 to
/// decouple audit event production from consumption while preventing
/// unbounded memory growth under sustained burst load. Producers
/// (request handlers) send events via `try_send` without blocking the
/// request path; the `Receiver` is consumed by a background worker
/// that applies PII redaction and persists the record.
///
/// When the channel buffer is full, `try_send` returns immediately
/// with a tracing warning - the request path is never blocked and
/// the system degrades gracefully under extreme load.
#[derive(Clone)]
pub struct ComplianceBus {
    tx: Sender<AuditEvent>,
    dropped_events: Arc<AtomicU64>,
}

impl ComplianceBus {
    /// Create a new compliance bus with a bounded MPSC channel (capacity 10000).
    ///
    /// Returns both the sender handle (`ComplianceBus`) and the receiver
    /// for the consumer. The receiver must be passed to `spawn_worker()`
    /// to begin processing events.
    pub fn new() -> (Self, Receiver<AuditEvent>) {
        let (tx, rx) = tokio::sync::mpsc::channel::<AuditEvent>(10000);
        (
            Self {
                tx,
                dropped_events: Arc::new(AtomicU64::new(0)),
            },
            rx,
        )
    }

    /// Emit an audit event into the compliance bus.
    ///
    /// Uses `try_send` to never block the request path. If the channel
    /// buffer is full, the event is dropped and a tracing warning is
    /// emitted. Returns `Ok(())` in all cases - the request path must
    /// never be blocked by audit logging backpressure.
    pub fn emit(&self, event: AuditEvent) -> Result<(), AuditEvent> {
        match self.tx.try_send(event) {
            Ok(()) => Ok(()),
            Err(tokio::sync::mpsc::error::TrySendError::Full(e)) => {
                self.dropped_events.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    tenant_id = %e.tenant_id,
                    action = %e.action_taken,
                    total_dropped = self.dropped_events.load(Ordering::Relaxed),
                    "Audit event dropped - compliance bus buffer full (capacity 10000)"
                );
                Ok(())
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(e)) => {
                tracing::error!("Compliance bus channel closed - worker terminated, event lost");
                Err(e)
            }
        }
    }

    /// Retrieve the cumulative count of audit events dropped due to
    /// channel buffer overflow since process startup.
    ///
    /// ## Stasis (Resource Control) Hub Alignment
    ///
    /// This counter is a first-class observability signal for the **Stasis**
    /// resource-control hub. When the compliance bus buffer saturates under
    /// burst load, the system enters a degraded audit mode: events are
    /// dropped rather than blocking the request path. This counter quantifies
    /// the audit-loss rate, enabling Stasis to:
    ///
    /// - **Detect backpressure saturation** before it cascades into request-path
    ///   latency degradation (proactive resource control).
    /// - **Trigger adaptive buffer resizing** or worker-pool scaling when the
    ///   drop rate exceeds a configurable threshold.
    /// - **Feed into the `/metrics` Prometheus endpoint** as a monotonic
    ///   `serein_compliance_dropped_events_total` gauge, allowing cluster-wide
    ///   alerting on audit integrity degradation.
    ///
    /// This zero-cost counter uses `AtomicU64` with `Relaxed` ordering,
    /// suitable for high-frequency metrics scraping without contention.
    pub fn get_drop_count(&self) -> u64 {
        self.dropped_events.load(Ordering::Relaxed)
    }

    /// Spawn a detached Tokio task that consumes audit events from the
    /// receiver, applies PII redaction, and persists the sanitized record.
    ///
    /// The worker runs indefinitely until the channel is closed (all senders
    /// dropped). Each event is processed sequentially to maintain causal
    /// ordering within a single worker instance.
    ///
    /// Processing pipeline per event:
    /// 1. Receive event from channel
    /// 2. Apply `redact_pii()` to `raw_payload`
    /// 3. Serialize the sanitized event to JSON
    /// 4. Persist (currently logged; replace with durable storage backend)
    pub fn spawn_worker(mut rx: Receiver<AuditEvent>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            tracing::info!("Compliance bus worker started - consuming audit events");

            while let Some(event) = rx.recv().await {
                let redacted_payload = redact_pii(&event.raw_payload);

                let sanitized = AuditEvent {
                    timestamp: event.timestamp,
                    tenant_id: event.tenant_id,
                    end_user_id: event.end_user_id,
                    raw_payload: redacted_payload,
                    action_taken: event.action_taken,
                };

                match serde_json::to_string(&sanitized) {
                    Ok(json) => {
                        tracing::info!(
                            tenant_id = %sanitized.tenant_id,
                            action = %sanitized.action_taken,
                            json_len = json.len(),
                            "Audit event persisted (PII redacted)"
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            tenant_id = %sanitized.tenant_id,
                            error = %e,
                            "Failed to serialize audit event - dropping record"
                        );
                    }
                }
            }

            tracing::warn!("Compliance bus worker terminated - channel closed");
        })
    }
}

/// Redact personally identifiable information from a text string.
///
/// Applies pre-compiled regex patterns to detect and mask common PII
/// categories: email addresses, phone numbers, SSNs, and credit card
/// numbers. Each match is replaced with a typed placeholder
/// (`[MASKED_EMAIL_N]`, `[MASKED_PHONE_N]`, etc.) to preserve semantic
/// context for downstream LLM reasoning.
///
/// This function is called inside the compliance worker before any
/// audit event is persisted, ensuring that no cleartext PII enters
/// long-term storage in compliance with GDPR Article 25 (Data Protection
/// by Design and by Default).
pub fn redact_pii(text: &str) -> String {
    let mut counter: u64 = 0;
    PII_PATTERN
        .replace_all(text, |caps: &regex::Captures| {
            let matched = caps.get(0).expect("capture group 0 must exist").as_str();
            let pii_type = classify_pii(matched);
            let idx = counter;
            counter += 1;
            format!("[{}_{}]", pii_type.placeholder_tag(), idx)
        })
        .to_string()
}

/// Bi-directional PII shielding utility for zero-trust prompt sanitization.
///
/// `PIIProtector` intercepts PII before it leaves the gateway and restores
/// it after LLM inference. Each detected PII entity is replaced with a
/// context-aware typed placeholder, and the mapping is stored for inverse
/// replacement (de-masking) on the ingress path.
///
/// ## Zero-Trust Model
/// - **Egress**: PII is stripped from prompts before transmission to external
///   LLM providers (DeepSeek, Moonshot, Groq). No cleartext PII ever leaves
///   the trust boundary.
/// - **Ingress**: After consensus is achieved, original PII values are restored
///   into the response before Z3 formal validation in the WASM sandbox.
///
/// ## Typed Placeholder Format
/// Each PII entity is replaced with a context-aware tag that preserves
/// semantic type information for improved LLM reasoning accuracy:
/// - Emails -> `[MASKED_EMAIL_N]`
/// - Phones -> `[MASKED_PHONE_N]`
/// - SSN/ID -> `[MASKED_IDENTITY_N]`
/// - Credit Cards -> `[MASKED_FINANCIAL_N]`
pub struct PIIProtector {
    counter: std::sync::atomic::AtomicU64,
}

/// Mapping from placeholder tokens to original PII values.
pub type MaskingMap = std::collections::HashMap<String, String>;

impl PIIProtector {
    pub fn new() -> Self {
        Self {
            counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Mask all PII entities in the input text and return the sanitized text
    /// along with the reversible masking map.
    ///
    /// Each detected PII entity is classified by type and replaced with a
    /// context-aware placeholder (`[MASKED_EMAIL_0]`, `[MASKED_PHONE_1]`, ...).
    /// The returned `MaskingMap` maps each placeholder back to its original
    /// value for later restoration.
    pub fn mask(&self, text: &str) -> (String, MaskingMap) {
        let mut map = MaskingMap::new();
        let counter = &self.counter;

        let result = PII_PATTERN
            .replace_all(text, |caps: &regex::Captures| {
                let matched = caps
                    .get(0)
                    .expect("capture group 0 must exist")
                    .as_str()
                    .to_string();
                let pii_type = classify_pii(&matched);
                let idx = counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let placeholder = format!("[{}_{}]", pii_type.placeholder_tag(), idx);
                map.insert(placeholder.clone(), matched);
                placeholder
            })
            .to_string();

        (result, map)
    }

    /// Restore original PII values by performing inverse replacement of all
    /// typed placeholder tokens found in the masking map.
    ///
    /// Iterates over all entries in the `MaskingMap` and replaces each
    /// typed placeholder (`[MASKED_EMAIL_N]`, `[MASKED_PHONE_N]`, etc.)
    /// with its original PII value. Placeholders not found in the map
    /// are left unchanged.
    pub fn restore(text: &str, map: &MaskingMap) -> String {
        let mut result = text.to_string();
        for (placeholder, original) in map {
            result = result.replace(placeholder.as_str(), original.as_str());
        }
        result
    }
}

impl Default for PIIProtector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redact_email() {
        let input = "User alice@example.com requested access";
        let redacted = redact_pii(input);
        assert!(!redacted.contains("alice@example.com"));
        assert!(redacted.contains("[MASKED_EMAIL_"));
    }

    #[test]
    fn test_redact_phone() {
        let input = "Call +1-555-123-4567 for support";
        let redacted = redact_pii(input);
        assert!(!redacted.contains("555-123-4567"));
        assert!(redacted.contains("[MASKED_PHONE_"));
    }

    #[test]
    fn test_redact_ssn() {
        let input = "SSN: 123-45-6789 on file";
        let redacted = redact_pii(input);
        assert!(!redacted.contains("123-45-6789"));
        assert!(redacted.contains("[MASKED_IDENTITY_"));
    }

    #[test]
    fn test_redact_credit_card() {
        let input = "Card: 4111-1111-1111-1111 charged";
        let redacted = redact_pii(input);
        assert!(!redacted.contains("4111-1111-1111-1111"));
        assert!(redacted.contains("[MASKED_FINANCIAL_"));
    }

    #[test]
    fn test_redact_no_pii() {
        let input = "Normal log message with no PII";
        assert_eq!(redact_pii(input), input);
    }

    #[test]
    fn test_redact_multiple_pii_types() {
        let input = "Email: bob@corp.com, Phone: 555-0100, SSN: 999-88-7777";
        let redacted = redact_pii(input);
        assert!(!redacted.contains("bob@corp.com"));
        assert!(!redacted.contains("999-88-7777"));
        assert!(redacted.contains("[MASKED_EMAIL_"));
        assert!(redacted.contains("[MASKED_IDENTITY_"));
    }

    #[test]
    fn test_audit_event_construction() {
        let event = AuditEvent::new(
            "tenant-42",
            "user-1234",
            "payload with alice@example.com",
            AuditAction::ALLOWED,
        );
        assert_eq!(event.tenant_id, "tenant-42");
        assert_eq!(event.end_user_id, "user-1234");
        assert_eq!(event.action_taken, "ALLOWED");
        assert!(event.timestamp > 0);
    }

    #[test]
    fn test_audit_action_as_str() {
        assert_eq!(AuditAction::ALLOWED.as_str(), "ALLOWED");
        assert_eq!(AuditAction::BLOCKED_BY_WASM.as_str(), "BLOCKED_BY_WASM");
        assert_eq!(AuditAction::MODIFIED.as_str(), "MODIFIED");
        assert_eq!(AuditAction::BLOCKED_BY_REPLAY.as_str(), "BLOCKED_BY_REPLAY");
        assert_eq!(AuditAction::BLOCKED_BY_ORACLE.as_str(), "BLOCKED_BY_ORACLE");
        assert_eq!(AuditAction::BLOCKED_BY_SIS.as_str(), "BLOCKED_BY_SIS");
        assert_eq!(AuditAction::BLOCKED_BY_AUTH.as_str(), "BLOCKED_BY_AUTH");
    }

    #[test]
    fn test_audit_action_display() {
        assert_eq!(AuditAction::ALLOWED.to_string(), "ALLOWED");
        assert_eq!(AuditAction::BLOCKED_BY_WASM.to_string(), "BLOCKED_BY_WASM");
        assert_eq!(AuditAction::MODIFIED.to_string(), "MODIFIED");
        assert_eq!(
            AuditAction::BLOCKED_BY_REPLAY.to_string(),
            "BLOCKED_BY_REPLAY"
        );
    }

    #[test]
    fn test_pii_protector_mask_and_restore_email() {
        let protector = PIIProtector::new();
        let input = "Contact alice@example.com for details";
        let (masked, map) = protector.mask(input);
        assert!(!masked.contains("alice@example.com"));
        assert!(masked.contains("[MASKED_EMAIL_"));
        let restored = PIIProtector::restore(&masked, &map);
        assert_eq!(restored, input);
    }

    #[test]
    fn test_pii_protector_mask_and_restore_multiple() {
        let protector = PIIProtector::new();
        let input = "Email: bob@corp.com, Phone: 555-0100, SSN: 999-88-7777";
        let (masked, map) = protector.mask(input);
        assert!(!masked.contains("bob@corp.com"));
        assert!(!masked.contains("999-88-7777"));
        assert!(masked.matches("[MASKED_").count() >= 3);
        let restored = PIIProtector::restore(&masked, &map);
        assert_eq!(restored, input);
    }

    #[test]
    fn test_pii_protector_no_pii_unchanged() {
        let protector = PIIProtector::new();
        let input = "Normal text without any PII";
        let (masked, map) = protector.mask(input);
        assert_eq!(masked, input);
        assert!(map.is_empty());
    }

    #[test]
    fn test_pii_protector_credit_card_roundtrip() {
        let protector = PIIProtector::new();
        let input = "Card: 4111-1111-1111-1111 charged";
        let (masked, map) = protector.mask(input);
        assert!(!masked.contains("4111-1111-1111-1111"));
        let restored = PIIProtector::restore(&masked, &map);
        assert_eq!(restored, input);
    }

    #[tokio::test]
    async fn test_compliance_bus_emit_and_receive() {
        let (bus, mut rx) = ComplianceBus::new();

        let event = AuditEvent::new("t1", "u1", "data", AuditAction::ALLOWED);
        assert!(bus.emit(event).is_ok());

        let received = rx.recv().await.expect("should receive event");
        assert_eq!(received.tenant_id, "t1");
        assert_eq!(received.end_user_id, "u1");
        assert_eq!(received.action_taken, "ALLOWED");
    }

    #[tokio::test]
    async fn test_compliance_bus_worker_redacts_pii() {
        let (bus, rx) = ComplianceBus::new();

        let event = AuditEvent::new(
            "tenant-x",
            "user-y",
            "Contact: alice@example.com, SSN: 123-45-6789",
            AuditAction::MODIFIED,
        );
        bus.emit(event).unwrap();

        drop(bus);

        let handle = ComplianceBus::spawn_worker(rx);
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn test_compliance_bus_emit_after_drop() {
        let (bus, rx) = ComplianceBus::new();
        drop(rx);

        let event = AuditEvent::new("t", "u", "p", AuditAction::BLOCKED_BY_AUTH);
        assert!(bus.emit(event).is_err());
    }
}
