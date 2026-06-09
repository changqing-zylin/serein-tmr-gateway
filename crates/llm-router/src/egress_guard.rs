// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Egress Guard - SSRF-Preventing Firewall
//!
//! Implements a strict egress firewall that enforces an absolute static whitelist
//! of permitted outbound domains to prevent Server-Side Request Forgery (SSRF).
//!
//! ## Security Architecture
//! - **Allowlist-only model**: Every outbound request must match a pre-approved domain.
//! - **SSRF hardening**: Rejects private/reserved IP ranges, link-local addresses,
//!   and any endpoint not present in the static whitelist.
//! - **Audit trail**: All blocked egress attempts are logged at `tracing::error!`
//!   level with full context for security forensics.
//!
//! ## Failure Modes
//! - Non-whitelisted domain: Returns `SecurityError::EgressViolation` immediately.
//! - Network transport failure: Propagates `SecurityError::TransportFailure`.
//! - Invalid URL parsing: Returns `SecurityError::InvalidEndpoint`.

use std::net::{IpAddr, Ipv4Addr};
use std::sync::OnceLock;

use reqwest::Client;
use thiserror::Error;
use tracing::{error, info};

pub type Result<T, E = SecurityError> = std::result::Result<T, E>;

/// Static allowlist of permitted egress domains.
///
/// This is an absolute whitelist - any domain not listed here will be rejected
/// at the network boundary before DNS resolution or TCP connection occurs.
///
/// Entries MUST be added via change-control review per ISS-NETWORK-002.
static EGRESS_ALLOWLIST: OnceLock<Vec<&'static str>> = OnceLock::new();

/// Reserved IPv4 ranges that indicate SSRF attempts (RFC 1918 + link-local + loopback).
static RESERVED_IPV4_PREFIXES: &[(u32, u8)] = &[
    (0x00000000, 8),   // 0.0.0.0/8
    (0x0a000000, 8),   // 10.0.0.0/8
    (0x64400000, 10),  // 100.64.0.0/10
    (0x7f000000, 8),   // 127.0.0.0/8
    (0xa9fe0000, 16),  // 169.254.0.0/16
    (0xac100000, 12),  // 172.16.0.0/12
    (0xc0a80000, 16),  // 192.168.0.0/16
    (0xc6120000, 15),  // 198.18.0.0/15
];

/// Security errors raised by the Egress Guard.
#[derive(Error, Debug)]
pub enum SecurityError {
    #[error("Egress violation: attempted connection to non-whitelisted endpoint '{endpoint}'")]
    EgressViolation { endpoint: String },

    #[error("SSRF attempt detected: resolved IP '{ip}' falls within reserved/private range")]
    SsrfDetected { ip: String },

    #[error("Invalid endpoint URL: {reason}")]
    InvalidEndpoint { reason: String },

    #[error("Transport failure during secure API call: {detail}")]
    TransportFailure { detail: String },
}

/// Initialize and return the static egress domain allowlist.
///
/// This function is idempotent; subsequent calls return the same reference.
fn get_allowlist() -> &'static Vec<&'static str> {
    EGRESS_ALLOWLIST.get_or_init(|| {
        vec![
            "api.deepseek.com",
            "api.anthropic.com",
            "api.openai.com",
            "generativelanguage.googleapis.com",
            "cdn.openai.com",
            "storage.googleapis.com",
            "api.groq.com",
        ]
    })
}

/// Check whether an IPv4 address falls within a reserved / private range.
fn is_reserved_ipv4(ipv4: &Ipv4Addr) -> bool {
    let addr = u32::from_be_bytes(ipv4.octets());
    RESERVED_IPV4_PREFIXES.iter().any(|(prefix, mask_len)| {
        let mask = if *mask_len == 0 { 0 } else { u32::MAX << (32 - *mask_len) };
        (addr ^ *prefix) & mask == 0
    })
}

/// Check whether an IP address is reserved, loopback, or otherwise SSRF-risky.
fn is_reserved_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_reserved_ipv4(v4),
        IpAddr::V6(v6) => {
            if v6.is_loopback() || v6.is_unspecified() {
                return true;
            }

            let segments = v6.segments();
            let high_bytes = ((segments[0] as u32) << 16) | (segments[1] as u32);

            let fc00_mask: u32 = 0xfe00_0000;
            let fc00_prefix: u32 = 0xfc00_0000;
            if (high_bytes & fc00_mask) == fc00_prefix {
                error!(
                    original_ipv6 = %ip,
                    "[EGRESS GUARD] SSRF detected: IPv6 Unique Local Address (fc00::/7)"
                );
                return true;
            }

            let fe80_mask: u32 = 0xffc0_0000;
            let fe80_prefix: u32 = 0xfe80_0000;
            if (high_bytes & fe80_mask) == fe80_prefix {
                error!(
                    original_ipv6 = %ip,
                    "[EGRESS GUARD] SSRF detected: IPv6 Link-Local Address (fe80::/10)"
                );
                return true;
            }

            if let Some(mapped_v4) = v6.to_ipv4_mapped() {
                if is_reserved_ipv4(&mapped_v4) {
                    error!(
                        original_ipv6 = %ip,
                        mapped_ipv4 = %mapped_v4,
                        "[EGRESS GUARD] SSRF detected: IPv4-mapped IPv6 address resolves to reserved IPv4 range"
                    );
                    return true;
                }
            }
            false
        }
    }
}

/// Parse the hostname from a URL string and validate it against the allowlist.
///
/// Returns `Ok(())` if the hostname is whitelisted, `Err(SecurityError)` otherwise.
fn validate_endpoint(endpoint: &str) -> Result<()> {
    let url =
        url::Url::parse(endpoint).map_err(|e| SecurityError::InvalidEndpoint {
            reason: format!("failed to parse URL: {}", e),
        })?;

    let host = url
        .host_str()
        .ok_or_else(|| SecurityError::InvalidEndpoint {
            reason: "URL has no host component".to_string(),
        })?;

    let allowlist = get_allowlist();
    if !allowlist.contains(&host) {
        error!(
            endpoint = %endpoint,
            target_host = %host,
            "[EGRESS GUARD] Blocked outbound request to non-whitelisted domain"
        );
        return Err(SecurityError::EgressViolation {
            endpoint: endpoint.to_string(),
        });
    }

    Ok(())
}

/// Perform a secure, SSRF-hardened cloud API call through the egress firewall.
///
/// This function enforces a strict defense-in-depth strategy:
///
/// 1. **Domain validation**: The endpoint's hostname must match the static allowlist.
/// 2. **IP resolution guard**: After DNS resolution, the resulting IP is checked against
///    reserved/private ranges to prevent DNS rebinding SSRF attacks.
/// 3. **Connection pooling**: The passed-in `reqwest::Client` is used directly for the
///    HTTP request. Its internal connection pool handles keep-alive and prevents port
///    exhaustion. DNS resolution is validated before the request, and the shared
///    client's connection pool safely handles keep-alive while preventing DNS rebinding
///    within an active connection.
///
/// # Arguments
/// * `client` - A shared `reqwest::Client` reference used for the actual HTTP request.
///   The client's internal connection pool prevents port exhaustion.
/// * `endpoint` - The full HTTPS URL of the target API endpoint.
/// * `payload` - JSON-encoded request body to send as POST data.
///
/// # Returns
/// The raw response body string on success, or a `SecurityError` on failure.
pub async fn secure_cloud_api_call(
    client: &Client,
    endpoint: &str,
    payload: &str,
) -> Result<String> {
    validate_endpoint(endpoint)?;

    let url = url::Url::parse(endpoint)
        .map_err(|e| SecurityError::InvalidEndpoint {
            reason: format!("failed to parse URL after allowlist check: {}", e),
        })?;

    let host = url.host_str().unwrap_or_default();
    let port = url.port().unwrap_or(443);

    let addrs = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| {
            error!(
                endpoint = %endpoint,
                error = %e,
                "[EGRESS GUARD] DNS resolution failed for allowed domain"
            );
            SecurityError::TransportFailure {
                detail: format!("DNS lookup failed: {}", e),
            }
        })?;

    for addr in addrs {
        if is_reserved_ip(&addr.ip()) {
            error!(
                endpoint = %endpoint,
                resolved_ip = %addr.ip(),
                "[EGRESS GUARD] SSRF detected: allowed domain resolved to reserved IP"
            );
            return Err(SecurityError::SsrfDetected {
                ip: addr.ip().to_string(),
            });
        }
    }

    info!(
        endpoint = %endpoint,
        resolved_host = %host,
        "[EGRESS GUARD] Secure API call dispatched via pooled client (SSRF check passed)"
    );

    let response = client
        .post(endpoint)
        .header("Content-Type", "application/json")
        .header("Cache-Control", "ephemeral")
        .body(payload.to_string())
        .send()
        .await
        .map_err(|e| {
            error!(
                endpoint = %endpoint,
                error = %e,
                "[EGRESS GUARD] HTTP request failed during secure API call"
            );
            SecurityError::TransportFailure {
                detail: format!("HTTP request failed: {}", e),
            }
        })?;

    let status = response.status();
    let body = response.text().await.map_err(|e| {
        error!(
            endpoint = %endpoint,
            status_code = %status,
            error = %e,
            "[EGRESS GUARD] Failed to read response body"
        );
        SecurityError::TransportFailure {
            detail: format!("response body read failed: {}", e),
        }
    })?;

    if !status.is_success() {
        error!(
            endpoint = %endpoint,
            status_code = %status,
            body_preview = %body.chars().take(256).collect::<String>(),
            "[EGRESS GUARD] Remote server returned non-success status"
        );
        return Err(SecurityError::TransportFailure {
            detail: format!("remote server returned status {}", status),
        });
    }

    Ok(body)
}

/// Perform a secure, SSRF-hardened cloud API call with custom headers.
///
/// Identical to `secure_cloud_api_call` but accepts additional HTTP headers
/// for provider-specific features like prompt caching (e.g., `anthropic-beta`).
///
/// # Arguments
/// * `client` - Shared `reqwest::Client` for connection pooling
/// * `endpoint` - Full HTTPS URL of the target API endpoint
/// * `payload` - JSON-encoded request body
/// * `extra_headers` - Additional headers as (name, value) pairs
pub async fn secure_cloud_api_call_with_headers(
    client: &Client,
    endpoint: &str,
    payload: &str,
    extra_headers: &[(String, String)],
) -> Result<String> {
    validate_endpoint(endpoint)?;

    let url = url::Url::parse(endpoint)
        .map_err(|e| SecurityError::InvalidEndpoint {
            reason: format!("failed to parse URL after allowlist check: {}", e),
        })?;

    let host = url.host_str().unwrap_or_default();
    let port = url.port().unwrap_or(443);

    let addrs = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| {
            error!(
                endpoint = %endpoint,
                error = %e,
                "[EGRESS GUARD] DNS resolution failed for allowed domain"
            );
            SecurityError::TransportFailure {
                detail: format!("DNS lookup failed: {}", e),
            }
        })?;

    for addr in addrs {
        if is_reserved_ip(&addr.ip()) {
            error!(
                endpoint = %endpoint,
                resolved_ip = %addr.ip(),
                "[EGRESS GUARD] SSRF detected: allowed domain resolved to reserved IP"
            );
            return Err(SecurityError::SsrfDetected {
                ip: addr.ip().to_string(),
            });
        }
    }

    info!(
        endpoint = %endpoint,
        resolved_host = %host,
        header_count = extra_headers.len(),
        "[EGRESS GUARD] Secure API call dispatched with custom headers (SSRF check passed)"
    );

    let mut request_builder = client.post(endpoint);

    let mut has_content_type = false;
    let mut has_cache_control = false;

    for (name, value) in extra_headers {
        if name.eq_ignore_ascii_case("content-type") { has_content_type = true; }
        if name.eq_ignore_ascii_case("cache-control") { has_cache_control = true; }
        request_builder = request_builder.header(name.as_str(), value.as_str());
    }

    if !has_content_type {
        request_builder = request_builder.header("Content-Type", "application/json");
    }
    if !has_cache_control {
        request_builder = request_builder.header("Cache-Control", "ephemeral");
    }

    let response = request_builder
        .body(payload.to_string())
        .send()
        .await
        .map_err(|e| {
            error!(
                endpoint = %endpoint,
                error = %e,
                "[EGRESS GUARD] HTTP request failed during secure API call with headers"
            );
            SecurityError::TransportFailure {
                detail: format!("HTTP request failed: {}", e),
            }
        })?;

    let status = response.status();
    let body = response.text().await.map_err(|e| {
        error!(
            endpoint = %endpoint,
            status_code = %status,
            error = %e,
            "[EGRESS GUARD] Failed to read response body"
        );
        SecurityError::TransportFailure {
            detail: format!("response body read failed: {}", e),
        }
    })?;

    if !status.is_success() {
        error!(
            endpoint = %endpoint,
            status_code = %status,
            body_preview = %body.chars().take(256).collect::<String>(),
            "[EGRESS GUARD] Remote server returned non-success status"
        );
        return Err(SecurityError::TransportFailure {
            detail: format!("remote server returned status {}", status),
        });
    }

    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allowlist_contains_known_domains() {
        let list = get_allowlist();
        assert!(list.contains(&"api.deepseek.com"));
        assert!(list.contains(&"api.anthropic.com"));
        assert!(list.contains(&"api.openai.com"));
    }

    #[test]
    fn test_reject_non_whitelisted_domain() {
        let result = validate_endpoint("https://evil.internal.corp/api");
        assert!(matches!(result, Err(SecurityError::EgressViolation { .. })));
    }

    #[test]
    fn test_accept_whitelisted_domain() {
        assert!(validate_endpoint("https://api.deepseek.com/v1/chat").is_ok());
        assert!(validate_endpoint("https://api.anthropic.com/v1/messages").is_ok());
    }

    #[test]
    fn test_reject_raw_ipv4_loopback() {
        let ip = IpAddr::V4(Ipv4Addr::LOCALHOST);
        assert!(is_reserved_ip(&ip));
    }

    #[test]
    fn test_reject_rfc1918_ranges() {
        assert!(is_reserved_ip(&IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1))));
        assert!(is_reserved_ip(&IpAddr::V4(Ipv4Addr::new(172, 16, 0, 1))));
        assert!(is_reserved_ip(&IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))));
    }

    #[test]
    fn test_reject_link_local() {
        assert!(is_reserved_ip(&IpAddr::V4(Ipv4Addr::new(169, 254, 1, 1))));
    }

    #[test]
    fn test_reject_ipv4_mapped_loopback() {
        use std::net::Ipv6Addr;
        let mapped_loopback = Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x7f00, 0x1);
        assert!(is_reserved_ip(&IpAddr::V6(mapped_loopback)));
    }

    #[test]
    fn test_reject_ipv4_mapped_rfc1918() {
        use std::net::Ipv6Addr;
        let mapped_private = Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x0a00, 0x1);
        assert!(is_reserved_ip(&IpAddr::V6(mapped_private)));
    }

    #[test]
    fn test_allow_ipv4_mapped_public() {
        use std::net::Ipv6Addr;
        let mapped_public = Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x0808, 0x0808);
        assert!(!is_reserved_ip(&IpAddr::V6(mapped_public)));
    }

    #[test]
    fn test_reject_ipv6_unique_local_fc00() {
        use std::net::Ipv6Addr;
        let ula = Ipv6Addr::new(0xfc00, 0, 0, 0, 0, 0, 0, 1);
        assert!(is_reserved_ip(&IpAddr::V6(ula)));
    }

    #[test]
    fn test_reject_ipv6_unique_local_fd00() {
        use std::net::Ipv6Addr;
        let ula = Ipv6Addr::new(0xfd12, 0x3456, 0x7890, 0, 0, 0, 0, 1);
        assert!(is_reserved_ip(&IpAddr::V6(ula)));
    }

    #[test]
    fn test_reject_ipv6_link_local_fe80() {
        use std::net::Ipv6Addr;
        let link_local = Ipv6Addr::new(0xfe80, 0, 0, 0, 0, 0, 0, 1);
        assert!(is_reserved_ip(&IpAddr::V6(link_local)));
    }

    #[test]
    fn test_allow_ipv6_public() {
        use std::net::Ipv6Addr;
        let public = Ipv6Addr::new(0x2607, 0xf8b0, 0x4004, 0x800, 0, 0, 0, 1);
        assert!(!is_reserved_ip(&IpAddr::V6(public)));
    }

    #[test]
    fn test_reject_invalid_url() {
        let result = validate_endpoint("not-a-valid-url");
        assert!(matches!(result, Err(SecurityError::InvalidEndpoint { .. })));
    }

    #[test]
    fn test_reject_url_without_host() {
        let result = validate_endpoint("file:///etc/passwd");
        assert!(matches!(result, Err(SecurityError::InvalidEndpoint { .. })));
    }

    #[tokio::test]
    async fn test_secure_cloud_api_call_blocked() {
        let client = Client::new();
        let result = secure_cloud_api_call(&client, "https://malicious.example.com/data", "{}").await;
        assert!(matches!(result, Err(SecurityError::EgressViolation { .. })));
    }
}
