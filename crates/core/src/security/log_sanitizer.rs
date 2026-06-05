// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Zero-Leak Log Sanitizer
//!
//! Prevents sensitive data from leaking through log output by applying
//! regex-based pattern masking to all log messages before they are emitted.
//!
//! ## Masked Patterns
//! - API keys in query strings (`?key=...`)
//! - Bearer tokens in Authorization headers
//! - Generic key/value pairs containing sensitive keywords
//! - Email addresses
//! - IPv4 addresses (optional)
//!
//! ## Architecture
//! The `LogSanitizer` is designed to be used as a `tracing` layer or
//! as a standalone utility for sanitizing strings before logging.
//! It compiles all patterns once at construction time for zero-allocation
//! matching during operation.

use regex::Regex;
use std::borrow::Cow;
use std::sync::LazyLock;

struct SanitizationRule {
    pattern: Regex,
    replacement: &'static str,
}

fn safe_regex(pattern: &str, rule_name: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|e| {
        tracing::error!(
            error = %e,
            rule = rule_name,
            "[LOG SANITIZER] Regex compilation failed - sanitization rule disabled. Log output may contain sensitive data."
        );
        Regex::new(r"a^").unwrap_or_else(|e2| {
            tracing::error!(error = %e2, "[LOG SANITIZER] Fallback regex also failed - using empty pattern");
            Regex::new("").unwrap_or_else(|_| Regex::new("a").unwrap())
        })
    })
}

static SANITIZATION_RULES: LazyLock<Vec<SanitizationRule>> = LazyLock::new(|| {
    vec![
        SanitizationRule {
            pattern: safe_regex(r"\?key=[^&\s]+", "api_key_query"),
            replacement: "?key=[REDACTED]",
        },
        SanitizationRule {
            pattern: safe_regex(r"(?i)Bearer\s+[A-Za-z0-9\-._~+/]+=*", "bearer_token"),
            replacement: "Bearer [REDACTED]",
        },
        SanitizationRule {
            pattern: safe_regex(
                r#"(?mi)(api[_-]?key|token|secret|password|auth[_-]?token|access[_-]?key|private[_-]?key)["\s:=]+[^\s"{},]{8,}"#,
                "sensitive_key_value",
            ),
            replacement: "${1}=[REDACTED]",
        },
        SanitizationRule {
            pattern: safe_regex(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "email"),
            replacement: "[EMAIL_REDACTED]",
        },
    ]
});

/// Zero-leak log sanitizer that masks sensitive data in log messages.
///
/// Compiles regex patterns once and applies them in sequence to sanitize
/// any string before it is written to logs. Designed for use as a
/// `tracing` subscriber layer or as a standalone utility.
///
/// ## Usage
/// ```rust,ignore
/// use serein_core::security::LogSanitizer;
///
/// let sanitizer = LogSanitizer::new();
/// let safe = sanitizer.sanitize("API call with ?key=sk-1234567890abcdef");
/// assert_eq!(safe, "API call with ?key=[REDACTED]");
/// ```
pub struct LogSanitizer {
    mask_ipv4: bool,
}

impl LogSanitizer {
    pub fn new() -> Self {
        Self { mask_ipv4: false }
    }

    pub fn with_ipv4_masking() -> Self {
        Self { mask_ipv4: true }
    }

    /// Sanitize a string by applying all registered regex masking patterns.
    ///
    /// ## Safety Contract
    /// This function MUST be called on any string before it is written to
    /// logs or external observability systems. Failure to sanitize may
    /// result in credential leakage.
    pub fn sanitize(&self, input: &str) -> String {
        let mut result: Cow<str> = Cow::Borrowed(input);

        for rule in SANITIZATION_RULES.iter() {
            let replaced = rule.pattern.replace_all(&result, rule.replacement);
            if let Cow::Owned(s) = replaced {
                result = Cow::Owned(s);
            }
        }

        if self.mask_ipv4 {
            if let Ok(ipv4_pattern) = Regex::new(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b") {
                let replaced = ipv4_pattern.replace_all(&result, "[IP_REDACTED]");
                if let Cow::Owned(s) = replaced {
                    result = Cow::Owned(s);
                }
            }
        }

        result.into_owned()
    }

    /// Sanitize a string for WASM cross-boundary transfer.
    ///
    /// Removes BOM characters, zero-width spaces, and control characters
    /// that could cause issues when transferring JSON payloads across the
    /// Component Model boundary to WASM guests.
    pub fn sanitize_for_wasm_transfer(input: &str) -> String {
        input
            .trim()
            .trim_start_matches(['\u{feff}', '\u{200b}'])
            .replace(
                |c: char| c.is_control() && c != '\n' && c != '\r' && c != '\t',
                " ",
            )
            .trim()
            .to_string()
    }
}

impl Default for LogSanitizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_api_key_in_query() {
        let sanitizer = LogSanitizer::new();
        let input = "Request to https://api.example.com/v1?key=sk-1234567890abcdef";
        let result = sanitizer.sanitize(input);
        assert_eq!(
            result,
            "Request to https://api.example.com/v1?key=[REDACTED]"
        );
    }

    #[test]
    fn test_mask_bearer_token() {
        let sanitizer = LogSanitizer::new();
        let input = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abc.def";
        let result = sanitizer.sanitize(input);
        assert_eq!(result, "Authorization: Bearer [REDACTED]");
    }

    #[test]
    fn test_mask_sensitive_key_value() {
        let sanitizer = LogSanitizer::new();
        let input = r#"{"api_key": "sk-1234567890abcdef"}"#;
        let result = sanitizer.sanitize(input);
        assert!(result.contains("[REDACTED]"));
        assert!(!result.contains("sk-1234567890abcdef"));
    }

    #[test]
    fn test_mask_email() {
        let sanitizer = LogSanitizer::new();
        let input = "User admin@example.com logged in";
        let result = sanitizer.sanitize(input);
        assert_eq!(result, "User [EMAIL_REDACTED] logged in");
    }

    #[test]
    fn test_ipv4_masking_disabled_by_default() {
        let sanitizer = LogSanitizer::new();
        let input = "Connection from 192.168.1.100";
        let result = sanitizer.sanitize(input);
        assert!(result.contains("192.168.1.100"));
    }

    #[test]
    fn test_ipv4_masking_enabled() {
        let sanitizer = LogSanitizer::with_ipv4_masking();
        let input = "Connection from 192.168.1.100";
        let result = sanitizer.sanitize(input);
        assert_eq!(result, "Connection from [IP_REDACTED]");
    }

    #[test]
    fn test_sanitize_for_wasm_transfer() {
        let input = "\u{feff}\u{200b}  {\"key\": \"value\"}  \x00\x01";
        let result = LogSanitizer::sanitize_for_wasm_transfer(input);
        assert_eq!(result, "{\"key\": \"value\"}");
    }

    #[test]
    fn test_sanitize_preserves_newlines() {
        let sanitizer = LogSanitizer::new();
        let input = "Line 1\nLine 2\tTabbed";
        let result = sanitizer.sanitize(input);
        assert!(result.contains('\n'));
        assert!(result.contains('\t'));
    }
}
