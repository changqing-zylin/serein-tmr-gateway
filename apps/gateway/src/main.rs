// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Serein Gateway - TMR-Protected LLM Orchestration Server
//!
//! HTTP gateway implementing Triple Modular Redundancy (TMR) consensus
//! across heterogeneous LLM providers with strict majority vote enforcement.
//!
//! ## Architecture
//! - **Provider-Agnostic TMR**: Dynamic `LlmProvider` trait objects loaded from
//!   `providers.toml` - no hardcoded provider logic in the gateway
//! - **Circuit Breakers**: Per-provider state machines trip on HTTP 429/5xx
//! - **TmrOrchestrator**: Unified consensus engine with Physical Fallback Node
//! - **Shared Connection Pool**: Single `reqwest::Client` prevents port exhaustion
//! - **WASI-Virt Honeypot**: Decoy tokens injected into sandboxed guest environment
//! - **ComplianceBus**: Async audit event bus for EU AI Act / GDPR compliance
//! - **EU AI Act Headers**: Transparency headers on every AI-gatewayed response

use arc_swap::ArcSwap;
use axum::body::Bytes;
use axum::extract::{ConnectInfo, DefaultBodyLimit, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use axum::Router;
use ipnet::IpNet;
use serde::{Deserialize, Serialize};
use serein_cache_storage::flight_recorder::{
    ConsensusEvent, ConsensusFlightRecorder, ProviderResultEntry,
};
use serein_core::security::hmac_auth::ServiceAuthenticator;
use serein_core::security::LogSanitizer;
use serein_core::AppConfig;
use serein_core::SereinMicrokernel;
use serein_interfaces::IntermediatePayload;
use serein_llm_router::coordinator::{SlmExecutionMode, SlmNodeConfig, TmrOrchestrator};
use serein_llm_router::provider::{ProviderConfig, ProviderRequest, ProvidersConfig};
use serein_sandbox_guard::deterministic_hooks::{PromptInjectionWaf, WafResult};
use serein_traffic_control::{
    acquire_api_token, init_circuit_breaker, FinOpsBudgetManager, FinOpsRefundGuard,
};
use serein_worker::{AuditAction, AuditEvent, ComplianceBus};
use std::future::IntoFuture;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use tower_http::cors::CorsLayer;
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::trace::TraceLayer;

#[cfg(feature = "jemalloc")]
use tikv_jemallocator::Jemalloc;

#[cfg(feature = "jemalloc")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[cfg(not(feature = "jemalloc"))]
use mimalloc::MiMalloc;

#[cfg(not(feature = "jemalloc"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Shared application state injected into all Axum request handlers.
///
/// Contains every long-lived service handle required for request processing:
/// the WASM microkernel, TMR orchestrator, flight recorder, HMAC authenticator,
/// and the compliance bus sender for async audit event emission.
#[derive(Clone)]
struct AppState {
    kernel: Arc<SereinMicrokernel>,
    tmr_orchestrator: Arc<TmrOrchestrator>,
    flight_recorder: Arc<ConsensusFlightRecorder>,
    service_auth: Option<Arc<ServiceAuthenticator>>,
    compliance_tx: serein_worker::ComplianceBus,
    waf: Arc<PromptInjectionWaf>,
    wasm_module: Arc<ArcSwap<Vec<u8>>>,
    finops_manager: Arc<FinOpsBudgetManager>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TmrConsensus {
    #[serde(alias = "result")]
    pub result: String,
    #[serde(alias = "agreeingNodes", alias = "agreeing_nodes")]
    pub agreeing_nodes: u8,
    #[serde(alias = "totalNodes", alias = "total_nodes")]
    pub total_nodes: u8,
    #[serde(alias = "adjudicationLogic", alias = "adjudication_logic")]
    pub adjudication_logic: &'static str,
}

static CLOUDFLARE_CIDRS: &[&str] = &[
    "173.245.48.0/20",
    "103.21.244.0/22",
    "103.22.200.0/22",
    "103.31.4.0/22",
    "141.101.64.0/18",
    "108.162.192.0/18",
    "190.93.240.0/20",
    "188.114.96.0/20",
    "197.234.240.0/22",
    "198.41.128.0/17",
    "162.158.0.0/15",
    "104.16.0.0/13",
    "104.24.0.0/14",
    "172.64.0.0/13",
    "131.0.72.0/22",
    "2400:cb00::/32",
    "2606:4700::/32",
    "2803:f800::/32",
    "2405:b500::/32",
    "2405:8100::/32",
    "2a06:98c0::/29",
    "2c0f:f248::/32",
];

static PARSED_CLOUDFLARE_NETS: LazyLock<Vec<IpNet>> = LazyLock::new(|| {
    CLOUDFLARE_CIDRS
        .iter()
        .filter_map(|cidr| cidr.parse::<IpNet>().ok())
        .collect()
});

fn is_trusted_proxy(socket_ip: IpAddr) -> bool {
    PARSED_CLOUDFLARE_NETS
        .iter()
        .any(|net| net.contains(&socket_ip))
}

/// Extract the true client IP from Cloudflare `cf-connecting-ip` header
/// when the connection originates from a trusted proxy; otherwise fall back
/// to the direct socket IP.
fn extract_trusted_ip(headers: &HeaderMap, socket_ip: IpAddr) -> IpAddr {
    if is_trusted_proxy(socket_ip) {
        if let Some(cf_ip) = headers
            .get("cf-connecting-ip")
            .and_then(|h| h.to_str().ok())
        {
            if let Some(ip_str) = cf_ip.split(',').next() {
                if let Ok(ip) = ip_str.trim().parse() {
                    return ip;
                }
            }
        }
    }

    tracing::warn!(
        socket_ip = %socket_ip,
        "Socket IP not a trusted proxy or cf-connecting-ip invalid - falling back to socket IP"
    );
    socket_ip
}

fn extract_tenant_id(headers: &HeaderMap) -> Result<String, StatusCode> {
    if let Some(tenant) = headers
        .get("x-serein-tenant-id")
        .and_then(|v| v.to_str().ok())
    {
        let clean = tenant.trim().to_string();
        if !clean.is_empty() {
            return Ok(clean);
        }
    }
    tracing::warn!("Missing verified tenant identity");
    Err(StatusCode::UNAUTHORIZED)
}

#[derive(Debug, serde::Deserialize, serde::Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AgentExecutionPayload {
    #[serde(alias = "network_id")]
    pub network_id: String,
    #[serde(alias = "task_type")]
    pub task_type: Option<String>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentExecutionResponse {
    pub consensus: TmrConsensus,
    pub payload: Option<IntermediatePayload>,
    pub tenant_id: String,
    pub kernel_receipt: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    code: &'static str,
}

/// Inject EU AI Act transparency headers into the response.
///
/// Article 52 of the EU AI Act requires that deployers of high-risk AI systems
/// inform natural persons when they are subject to automated decision-making.
/// These headers provide machine-readable transparency signals:
/// - `X-AI-Gateway-Intervention: true` - indicates the gateway processed or
///   intervened in the request (as opposed to a pass-through proxy)
/// - `X-AI-Provenance: serein-enterprise-v1` - identifies the AI system
///   and version for audit trail correlation
fn inject_ai_transparency_headers(response: &mut axum::response::Response) {
    let headers = response.headers_mut();
    headers.insert(
        "x-ai-gateway-intervention",
        HeaderValue::from_static("true"),
    );
    headers.insert(
        "x-ai-provenance",
        HeaderValue::from_static("serein-enterprise-v1"),
    );
}

/// Emit a compliance audit event to the async bus.
///
/// Fire-and-forget: if the bus channel is closed (critical infrastructure
/// failure), the event is silently dropped rather than blocking the request
/// path. This ensures the compliance bus never adds latency to the gateway.
fn emit_audit_event(
    compliance_tx: &serein_worker::ComplianceBus,
    tenant_id: &str,
    client_ip: IpAddr,
    raw_payload: &str,
    action: AuditAction,
) {
    let event = AuditEvent::new(
        tenant_id.to_string(),
        client_ip.to_string(),
        raw_payload.to_string(),
        action,
    );
    if let Err(e) = compliance_tx.emit(event) {
        tracing::error!(
            tenant_id = %tenant_id,
            "Compliance bus channel closed - audit event dropped: {:?}",
            e
        );
    }
}

async fn handle_agent_execution(
    State(state): State<AppState>,
    ConnectInfo(socket_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(body): Json<AgentExecutionPayload>,
) -> axum::response::Response {
    let client_ip = extract_trusted_ip(&headers, socket_addr.ip());

    let tenant_id = match extract_tenant_id(&headers) {
        Ok(id) => id,
        Err(status) => {
            emit_audit_event(
                &state.compliance_tx,
                "unknown",
                client_ip,
                "missing tenant ID",
                AuditAction::BLOCKED_BY_AUTH,
            );
            return (status, Json(ErrorResponse {
                error: "Missing or empty x-serein-tenant-id header - tenant identity required for billing".to_string(),
                code: "MISSING_TENANT_ID",
            })).into_response();
        }
    };

    let request_id = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    let task_type_val = body.task_type.as_deref().unwrap_or("default");

    if let Some(ref auth) = state.service_auth {
        if let Some(auth_header) = headers.get("Authorization").and_then(|h| h.to_str().ok()) {
            let request_timestamp: i64 = headers
                .get("x-serein-timestamp")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            let nonce = headers
                .get("x-serein-nonce")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();

            if nonce.is_empty() {
                tracing::warn!(
                    tenant_id = %tenant_id,
                    "Missing x-serein-nonce header - UUID nonce required for idempotency"
                );
                emit_audit_event(
                    &state.compliance_tx,
                    &tenant_id,
                    client_ip,
                    "missing nonce",
                    AuditAction::BLOCKED_BY_AUTH,
                );
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse {
                        error:
                            "Missing x-serein-nonce header - UUID nonce required for idempotency"
                                .to_string(),
                        code: "MISSING_NONCE",
                    }),
                )
                    .into_response();
            }

            if request_timestamp == 0 {
                tracing::warn!(
                    tenant_id = %tenant_id,
                    "Missing or invalid x-serein-timestamp header - epoch seconds required"
                );
                emit_audit_event(
                    &state.compliance_tx,
                    &tenant_id,
                    client_ip,
                    "missing timestamp",
                    AuditAction::BLOCKED_BY_AUTH,
                );
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse {
                        error:
                            "Missing or invalid x-serein-timestamp header - epoch seconds required"
                                .to_string(),
                        code: "MISSING_TIMESTAMP",
                    }),
                )
                    .into_response();
            }

            let payload = format!(
                "{}:{}:{}:{}:{}",
                tenant_id, body.network_id, task_type_val, request_timestamp, nonce
            );
            if let Err(e) = auth.validate_auth_header(auth_header, &payload, request_timestamp) {
                tracing::warn!(
                    tenant_id = %tenant_id,
                    error = %e,
                    "HMAC authentication failed - request rejected"
                );
                let code = match &e {
                    serein_core::security::hmac_auth::HmacAuthError::TimestampMismatch => {
                        "TIMESTAMP_MISMATCH"
                    }
                    serein_core::security::hmac_auth::HmacAuthError::VerificationFailed(_) => {
                        "REPLAY_WINDOW_VIOLATION"
                    }
                    serein_core::security::hmac_auth::HmacAuthError::InvalidFormat => {
                        "HMAC_INVALID_FORMAT"
                    }
                    serein_core::security::hmac_auth::HmacAuthError::ReplayDetected => {
                        "REPLAY_ATTACK_DETECTED"
                    }
                    serein_core::security::hmac_auth::HmacAuthError::ComputationFailed(_) => {
                        "HMAC_COMPUTATION_FAILED"
                    }
                    serein_core::security::hmac_auth::HmacAuthError::MissingToken => {
                        "HMAC_TOKEN_MISSING"
                    }
                };
                let action = if matches!(
                    &e,
                    serein_core::security::hmac_auth::HmacAuthError::ReplayDetected
                ) {
                    AuditAction::BLOCKED_BY_REPLAY
                } else {
                    AuditAction::BLOCKED_BY_AUTH
                };
                emit_audit_event(
                    &state.compliance_tx,
                    &tenant_id,
                    client_ip,
                    "HMAC auth failed",
                    action,
                );
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse {
                        error: "HMAC signature verification failed".to_string(),
                        code,
                    }),
                )
                    .into_response();
            }
        } else {
            tracing::warn!(
                tenant_id = %tenant_id,
                "Missing Authorization header - HMAC authentication required"
            );
            emit_audit_event(
                &state.compliance_tx,
                &tenant_id,
                client_ip,
                "missing auth header",
                AuditAction::BLOCKED_BY_AUTH,
            );
            return (
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: "Authorization header required for HMAC authentication".to_string(),
                    code: "MISSING_AUTH_HEADER",
                }),
            )
                .into_response();
        }
    }

    if let Err(e) = acquire_api_token(&tenant_id).await {
        tracing::warn!(
            tenant_id = %tenant_id,
            error = %e,
            "Tenant rate limit exceeded - request rejected via P5 circuit breaker"
        );
        emit_audit_event(
            &state.compliance_tx,
            &tenant_id,
            client_ip,
            "rate limited",
            AuditAction::BLOCKED_BY_SIS,
        );
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse {
                error: format!("Tenant rate limit exceeded: {}", e),
                code: "RATE_LIMITED",
            }),
        )
            .into_response();
    }

    let finops_guard = match state.finops_manager.deduct_tokens(&tenant_id, 1).await {
        Ok(serein_traffic_control::BudgetDeductionResult::Success { .. }) => {
            FinOpsRefundGuard::new(&tenant_id, &request_id, Arc::clone(&state.finops_manager))
        }
        Ok(serein_traffic_control::BudgetDeductionResult::InsufficientFunds {
            balance,
            required,
        }) => {
            tracing::warn!(
                tenant_id = %tenant_id,
                balance = balance,
                required = required,
                "[FINOPS] Insufficient token budget - request denied with 402"
            );
            emit_audit_event(
                &state.compliance_tx,
                &tenant_id,
                client_ip,
                "insufficient funds",
                AuditAction::BLOCKED_BY_SIS,
            );
            return (
                StatusCode::PAYMENT_REQUIRED,
                Json(ErrorResponse {
                    error: "Insufficient token budget - please top up your account".to_string(),
                    code: "INSUFFICIENT_FUNDS",
                }),
            )
                .into_response();
        }
        Err(e) => {
            tracing::error!(
                tenant_id = %tenant_id,
                error = %e,
                "[FINOPS] State store inaccessible - request denied with 500"
            );
            emit_audit_event(
                &state.compliance_tx,
                &tenant_id,
                client_ip,
                "finops state store inaccessible",
                AuditAction::BLOCKED_BY_SIS,
            );
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "Billing infrastructure unavailable - please try again later"
                        .to_string(),
                    code: "FINOPS_UNAVAILABLE",
                }),
            )
                .into_response();
        }
    };

    if body.network_id.is_empty() || body.network_id.len() > 64 || !body.network_id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "network_id must be a non-empty alphanumeric string (max 64 chars)".to_string(),
                code: "INVALID_NETWORK_ID",
            }),
        )
            .into_response();
    }

    if task_type_val.len() > 50
        || !task_type_val
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c.is_ascii_whitespace())
    {
        tracing::warn!(client_ip = %client_ip, "BLOCKED: Potential prompt injection detected in task_type");
        emit_audit_event(
            &state.compliance_tx,
            &tenant_id,
            client_ip,
            "prompt injection attempt",
            AuditAction::BLOCKED_BY_WASM,
        );
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "task_type contains invalid characters or exceeds length limits".to_string(),
                code: "SECURITY_VIOLATION",
            }),
        )
            .into_response();
    }

    let prompt = format!(
        r#"You are a deterministic Web3 agent execution engine. Extract execution parameters for network "{}" and task type "{}".
Respond ONLY with a JSON object matching this exact schema:
{{
  "networkId": "{}",
  "taskType": "{}",
  "maxGasLimit": <integer: estimated max gas limit, or 0 if unknown>,
  "confidenceScore": <float: 0.0 to 1.0 representing certainty>,
  "sourceUrl": "<string: official data source URL or 'unknown'>"
}}
No explanation, no markdown fences, just the JSON object."#,
        body.network_id.to_lowercase(),
        task_type_val,
        body.network_id.to_lowercase(),
        task_type_val
    );

    let provider_req = ProviderRequest::new(&prompt).with_tenant(&tenant_id);

    match state.waf.scan(&prompt) {
        WafResult::Blocked(violation) => {
            tracing::error!(
                tenant_id = %tenant_id,
                signature = %violation.signature,
                category = ?violation.category,
                "[SANDBOX-GUARD] Prompt injection detected - request blocked with 403"
            );
            emit_audit_event(
                &state.compliance_tx,
                &tenant_id,
                client_ip,
                &format!("WAF blocked: {}", violation.signature),
                AuditAction::BLOCKED_BY_WASM,
            );
            return (
                StatusCode::FORBIDDEN,
                Json(ErrorResponse {
                    error: format!("Prompt injection detected: {}", violation.signature),
                    code: "WAF_BLOCKED",
                }),
            )
                .into_response();
        }
        WafResult::Clean => {}
    }

    let tmr_result = tokio::time::timeout(
        Duration::from_secs(55),
        state
            .tmr_orchestrator
            .execute_consensus_providers(&provider_req),
    )
    .await
    .unwrap_or_else(|_| {
        tracing::error!(
            tenant_id = %tenant_id,
            "TMR consensus timed out after 55s"
        );
        serein_llm_router::coordinator::TmrConsensusResult::default()
    });

    let agreement_count = tmr_result.agreement_count;
    let total_nodes = tmr_result.total_nodes as u8;

    let consensus_payload = tmr_result
        .majority_output
        .as_ref()
        .and_then(|output| IntermediatePayload::from_llm_output(output).ok());

    let mut provider_results: Vec<(String, String)> = Vec::new();

    for nr in tmr_result.node_results.iter() {
        let provider_name = &nr.provider_name;
        match (&nr.output, &nr.error) {
            (Some(text), None) => {
                tracing::info!(provider = %provider_name, output_len = text.len(), "Node response received");
                provider_results.push((provider_name.clone(), text.clone()));
            }
            (_, Some(err)) => {
                tracing::warn!(provider = %provider_name, error = ?err, "Node failed");
                provider_results.push((provider_name.clone(), format!("ERROR: {:?}", err)));
            }
            _ => {}
        }
    }

    if let Some(ref fb_result) = tmr_result.fallback_result {
        if let Some(ref fb_output) = fb_result.output {
            tracing::info!(
                model_id = %fb_result.model_id,
                "Physical Fallback Node D response received"
            );
            provider_results.push(("FallbackNodeD".to_string(), fb_output.clone()));
        }
    }

    if !tmr_result.consensus_achieved {
        emit_audit_event(
            &state.compliance_tx,
            &tenant_id,
            client_ip,
            "TMR consensus failed",
            AuditAction::BLOCKED_BY_ORACLE,
        );
        return (
            StatusCode::CONFLICT,
            Json(ErrorResponse {
                error: format!(
                    "TMR consensus failed: no majority agreement among {} providers",
                    total_nodes
                ),
                code: "CONSENSUS_FAILED",
            }),
        )
            .into_response();
    }

    let adjudicated_json = match &consensus_payload {
        Some(payload) => match serde_json::to_string(payload) {
            Ok(j) => j,
            Err(e) => {
                tracing::error!(error = %e, "Failed to serialize validated payload");
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: "Internal serialization failure".to_string(),
                        code: "SERIALIZATION_FAILED",
                    }),
                )
                    .into_response();
            }
        },
        None => String::new(),
    };

    let consensus = TmrConsensus {
        result: adjudicated_json.clone(),
        agreeing_nodes: agreement_count as u8,
        total_nodes,
        adjudication_logic: if tmr_result.fallback_activated {
            "canonical_key_majority_with_fallback"
        } else {
            "canonical_key_majority"
        },
    };

    if agreement_count >= 2 {
        let consensus_event = ConsensusEvent {
            timestamp: chrono::Utc::now(),
            request_id: format!("tmr-{}", chrono::Utc::now().timestamp_millis()),
            tenant_id: tenant_id.clone(),
            prompt: prompt.clone(),
            inference_params: serde_json::json!({
                "cloud_timeout_ms": 30_000,
                "fallback_activated": tmr_result.fallback_activated,
            }),
            agreeing_nodes: agreement_count as u8,
            total_nodes,
            adjudication_logic: consensus.adjudication_logic.to_string(),
            consensus_payload: consensus_payload
                .as_ref()
                .map(|p| serde_json::to_value(p).unwrap_or_default())
                .unwrap_or_default(),
            provider_results: provider_results
                .iter()
                .map(|(name, output)| {
                    let status = if output.starts_with("ERROR:") {
                        "error"
                    } else {
                        "success"
                    };
                    let canonical_key = if !output.starts_with("ERROR:") {
                        IntermediatePayload::from_llm_output(output)
                            .ok()
                            .map(|p| p.canonical_key())
                    } else {
                        None
                    };
                    ProviderResultEntry {
                        provider: name.clone(),
                        status: status.to_string(),
                        canonical_key,
                    }
                })
                .collect(),
            fallback_activated: tmr_result.fallback_activated,
        };
        state
            .flight_recorder
            .record_consensus_event(consensus_event)
            .await;
    }

    let adjudicated_json = LogSanitizer::sanitize_for_wasm_transfer(&adjudicated_json);

    let receipt: serde_json::Value = serde_json::from_str(&adjudicated_json)
        .unwrap_or_else(|_| serde_json::json!({ "raw": adjudicated_json }));
    tracing::info!(tenant_id = %tenant_id, "Agent execution completed via TMR consensus (WASM bypassed)");

    emit_audit_event(
        &state.compliance_tx,
        &tenant_id,
        client_ip,
        &adjudicated_json,
        AuditAction::ALLOWED,
    );

    finops_guard.consume();

    let mut response = (
        StatusCode::OK,
        Json(AgentExecutionResponse {
            consensus,
            payload: consensus_payload,
            tenant_id,
            kernel_receipt: receipt,
        }),
    )
        .into_response();
    inject_ai_transparency_headers(&mut response);
    response
}

async fn handle_hot_swap(
    State(state): State<AppState>,
    ConnectInfo(socket_addr): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> axum::response::Response {
    let client_ip = extract_trusted_ip(&headers, socket_addr.ip());

    let tenant_id = match extract_tenant_id(&headers) {
        Ok(id) => id,
        Err(status) => {
            return (
                status,
                Json(ErrorResponse {
                    error: "Missing or empty x-serein-tenant-id header".to_string(),
                    code: "MISSING_TENANT_ID",
                }),
            )
                .into_response();
        }
    };

    if let Some(ref auth) = state.service_auth {
        if let Some(auth_header) = headers.get("Authorization").and_then(|h| h.to_str().ok()) {
            let request_timestamp: i64 = headers
                .get("x-serein-timestamp")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            let nonce = headers
                .get("x-serein-nonce")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();

            if nonce.is_empty() || request_timestamp == 0 {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse {
                        error: "HMAC authentication required for hot-swap".to_string(),
                        code: "MISSING_AUTH",
                    }),
                )
                    .into_response();
            }

            let payload = format!("{}:hot-swap:{}:{}", tenant_id, request_timestamp, nonce);
            if let Err(e) = auth.validate_auth_header(auth_header, &payload, request_timestamp) {
                tracing::warn!(
                    tenant_id = %tenant_id,
                    error = %e,
                    "[HOT-SWAP] HMAC authentication failed"
                );
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse {
                        error: "HMAC signature verification failed".to_string(),
                        code: "HMAC_FAILED",
                    }),
                )
                    .into_response();
            }
        } else {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: "Authorization header required for hot-swap".to_string(),
                    code: "MISSING_AUTH_HEADER",
                }),
            )
                .into_response();
        }
    }

    if body.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Empty WASM module body".to_string(),
                code: "EMPTY_BODY",
            }),
        )
            .into_response();
    }

    if body.len() > 50 * 1024 * 1024 {
        return (
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ErrorResponse {
                error: "WASM module exceeds 50MB limit".to_string(),
                code: "PAYLOAD_TOO_LARGE",
            }),
        )
            .into_response();
    }

    if &body[0..4] != b"\0asm" {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Invalid WASM binary: missing magic number".to_string(),
                code: "INVALID_WASM",
            }),
        )
            .into_response();
    }

    let module_bytes = body.to_vec();
    let module_size = module_bytes.len();
    let old_module = state.wasm_module.swap(Arc::new(module_bytes.clone()));

    tracing::info!(
        tenant_id = %tenant_id,
        client_ip = %client_ip,
        old_size = old_module.len(),
        new_size = module_size,
        "[HOT-SWAP] WASM module atomically swapped via ArcSwap"
    );

    match state.kernel.reload_component(&module_bytes).await {
        Ok(()) => {
            tracing::info!(
                tenant_id = %tenant_id,
                "[HOT-SWAP] Wasmtime component recompiled and engine hot-swapped"
            );
        }
        Err(e) => {
            tracing::error!(
                tenant_id = %tenant_id,
                error = %e,
                "[HOT-SWAP] Wasmtime recompilation failed - byte-level swap committed but engine not updated"
            );
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("WASM compilation failed: {}", e),
                    code: "WASM_COMPILE_ERROR",
                }),
            )
                .into_response();
        }
    }

    emit_audit_event(
        &state.compliance_tx,
        &tenant_id,
        client_ip,
        &format!("hot-swap: {} bytes", module_size),
        AuditAction::MODIFIED,
    );

    let mut response = (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "swapped",
            "previous_size_bytes": old_module.len(),
            "new_size_bytes": module_size,
        })),
    )
        .into_response();
    inject_ai_transparency_headers(&mut response);
    response
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    if std::env::var("APP_ENV").unwrap_or_default() == "production" {
        let token = std::env::var("SEREIN_INTERNAL_TOKEN").unwrap_or_default();
        if token == "generate_a_random_uuid_here" || token.trim().is_empty() {
            panic!("FATAL: Insecure default SEREIN_INTERNAL_TOKEN detected in production. Halting process.");
        }
    }

    {
        let num_threads = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            });
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|idx| format!("serein-rayon-{idx}"))
            .build_global()
            .unwrap_or_else(|e| {
                tracing::warn!(
                    error = %e,
                    "Global rayon pool already initialized - skipping"
                );
            });
        tracing::info!(
            threads = num_threads,
            "Global rayon thread pool initialized - respects RAYON_NUM_THREADS / cgroup CPU quota"
        );
    }

    let (non_blocking, _guard) = tracing_appender::non_blocking(std::io::stdout());
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(non_blocking)
        .with_target(true)
        .with_thread_ids(true)
        .with_file(false)
        .with_line_number(false)
        .init();

    let tenant = std::env::var("SEREIN_TENANT").unwrap_or_else(|_| "default".to_string());

    init_circuit_breaker();
    tracing::info!("P5 telemetry circuit breaker initialized - dual-layer rate limiting active");

    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let redis_client = redis::Client::open(redis_url.as_str())
        .expect("Failed to create Redis client - check REDIS_URL environment variable");
    let redis_pool = redis::aio::ConnectionManager::new(redis_client)
        .await
        .expect(
            "Failed to establish Redis connection pool - FinOps budget enforcement requires Redis",
        );
    let finops_manager = Arc::new(FinOpsBudgetManager::new(redis_pool, false));
    tracing::info!(
        "FinOpsBudgetManager initialized - Redis-backed atomic token budget enforcement active"
    );

    {
        let rate_limiter = serein_traffic_control::rate_limiter::get_circuit_breaker()
            .expect("Rate limiter must be initialized");
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                let evicted = rate_limiter
                    .evict_stale_tenants(Duration::from_secs(3600))
                    .await;
                if evicted > 0 {
                    tracing::info!(
                        evicted = evicted,
                        "[API RATE LIMITER] Background eviction cycle completed"
                    );
                }
            }
        });
        tracing::info!("[API RATE LIMITER] Background stale-tenant eviction task spawned - interval=60s, max_age=3600s");
    }

    serein_sandbox_guard::deterministic_hooks::init_waf();
    tracing::info!("[SANDBOX-GUARD] Global WAF singleton eagerly initialized - Aho-Corasick automaton precompiled before first request");

    let app_config = AppConfig::from_env().map_err(|e| {
        anyhow::anyhow!(
            "Failed to load application configuration from environment: {}",
            e
        )
    })?;

    let honeypot_ctx = serein_sandbox_guard::wasi_virt::build_honeypot_context();
    tracing::info!(
        mount_point = %honeypot_ctx.vfs.mount_point,
        token_count = honeypot_ctx.environment.len(),
        "WASI-Virt honeypot context initialized - decoy credentials ready for sandbox injection"
    );

    let kernel = Arc::new(SereinMicrokernel::new(&tenant, honeypot_ctx, &app_config).await?);

    tracing::info!(
        "Wasmtime Engine and Component precompiled - stored in SereinMicrokernel for global reuse"
    );

    let providers_config_path =
        std::env::var("PROVIDERS_CONFIG").unwrap_or_else(|_| "providers.toml".to_string());

    let providers_config =
        ProvidersConfig::load_from_file(&providers_config_path).unwrap_or_else(|e| {
            tracing::warn!(
                path = %providers_config_path,
                error = %e,
                "Failed to load providers.toml - falling back to environment-based configuration"
            );
            build_providers_from_env()
        });

    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .tcp_keepalive(std::time::Duration::from_secs(90))
        .pool_max_idle_per_host(50)
        .build()?;

    let providers = providers_config
        .build_providers(&http_client)
        .map_err(|e| anyhow::anyhow!("Failed to build providers from configuration: {}", e))?;

    let provider_display_names: Vec<String> = providers
        .iter()
        .map(|p| p.display_name().to_string())
        .collect();

    tracing::info!(
        providers = ?provider_display_names,
        "Dynamic provider registry initialized from {}",
        providers_config_path
    );

    let slm_config = match SlmNodeConfig::from_env() {
        Ok(config) => config,
        Err(e) => {
            tracing::error!(
                error = %e,
                "SLM configuration invalid - Physical Fallback Node D will start in Lazy mode"
            );
            SlmNodeConfig {
                model_id: "serein-slm-v1".to_string(),
                model_path: std::path::PathBuf::from("./serein-models/fallback-slm.gguf"),
                execution_mode: SlmExecutionMode::Lazy,
            }
        }
    };
    let tmr_orchestrator = Arc::new(TmrOrchestrator::with_providers(providers, slm_config));

    {
        let fallback_node = tmr_orchestrator.fallback_node();
        if fallback_node.is_available() {
            tracing::info!("Probing Physical Fallback Node D with dummy payload...");
            let probe_handle = tokio::spawn({
                let node = Arc::clone(fallback_node);
                async move { node.invoke("probe").await }
            });
            match tokio::time::timeout(Duration::from_secs(10), probe_handle).await {
                Ok(Ok(result)) => {
                    if result.error.is_some() {
                        tracing::error!(
                            error = ?result.error,
                            "[CRITICAL] Physical Fallback Node D probe failed - Node D remains in RetryPending state. \
                             It will be retried on next cloud failure recovery cycle. \
                             No permanent disablement applied."
                        );
                    } else {
                        tracing::info!(
                            model_id = %result.model_id,
                            duration_ms = result.duration_ms,
                            "Physical Fallback Node D probe succeeded - Node D operational"
                        );
                    }
                }
                Ok(Err(join_err)) => {
                    tracing::error!(
                        error = %join_err,
                        "[CRITICAL] Physical Fallback Node D probe panicked - Node D remains in RetryPending state. \
                         It will be retried on next cloud failure recovery cycle. \
                         No permanent disablement applied."
                    );
                }
                Err(_) => {
                    tracing::error!(
                        "[CRITICAL] Physical Fallback Node D probe timed out after 10s - Node D remains in RetryPending state. \
                         It will be retried on next cloud failure recovery cycle. \
                         No permanent disablement applied."
                    );
                }
            }
        } else {
            tracing::warn!(
                "Physical Fallback Node D is disabled by configuration - skipping probe"
            );
        }

        let healing_node = Arc::clone(fallback_node);
        tokio::spawn(async move {
            loop {
                if !healing_node.is_available() {
                    tracing::info!(
                        "[SELF-HEALING] Node D is unavailable - waiting 30s before probing"
                    );
                    tokio::time::sleep(Duration::from_secs(30)).await;

                    tracing::info!("[SELF-HEALING] Sending probe to Node D");
                    let result = healing_node.invoke("probe").await;

                    if result.error.is_none() {
                        healing_node.enable();
                        tracing::info!(
                            model_id = %result.model_id,
                            duration_ms = result.duration_ms,
                            "[SELF-HEALING] Node D has successfully healed and is now re-enabled"
                        );
                    } else {
                        tracing::warn!(
                            error = ?result.error,
                            "[SELF-HEALING] Node D probe still failing - will retry in 30s"
                        );
                    }
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });
    }

    tracing::info!(
        "TmrOrchestrator initialized with provider-agnostic architecture and Physical Fallback Node D"
    );

    let flight_recorder =
        Arc::new(ConsensusFlightRecorder::new(None).await.map_err(|e| {
            anyhow::anyhow!("Failed to initialize consensus flight recorder: {}", e)
        })?);

    let service_auth = ServiceAuthenticator::from_env().ok().map(Arc::new);

    if service_auth.is_some() {
        tracing::info!("HMAC service-to-service authentication enabled via SEREIN_INTERNAL_TOKEN");
    } else {
        tracing::warn!("SEREIN_INTERNAL_TOKEN not configured - HMAC authentication disabled (not recommended for production)");
    }

    let (compliance_bus, compliance_rx) = ComplianceBus::new();
    ComplianceBus::spawn_worker(compliance_rx);
    tracing::info!("Compliance bus worker spawned - audit events will be processed asynchronously");

    let waf = Arc::new(PromptInjectionWaf::new());
    tracing::info!(
        "[SANDBOX-GUARD] PromptInjectionWaf initialized - Aho-Corasick automaton compiled"
    );

    let initial_wasm = vec![];
    let wasm_module = Arc::new(ArcSwap::new(Arc::new(initial_wasm)));
    tracing::info!("[HOT-SWAP] WASM module store initialized - ready for atomic hot-swap");

    let state = AppState {
        kernel,
        tmr_orchestrator,
        flight_recorder,
        service_auth,
        compliance_tx: compliance_bus,
        waf,
        wasm_module,
        finops_manager,
    };

    tokio::spawn(async {
        let metrics_app = Router::new().route(
            "/metrics",
            axum::routing::get(|| async {
                "# HELP serein_billing_event_total Total FinOps billing events processed\n# TYPE serein_billing_event_total counter\nserein_billing_event_total{tenant=\"system\"} 1\n"
            }),
        );

        let metrics_listener = match tokio::net::TcpListener::bind("0.0.0.0:9090").await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "[METRICS] Failed to bind Prometheus metrics port 9090 - SRE scraping will fail"
                );
                return;
            }
        };

        tracing::info!("[METRICS] Prometheus metrics endpoint listening on 0.0.0.0:9090/metrics");

        if let Err(e) = axum::serve(metrics_listener, metrics_app.into_make_service()).await {
            tracing::error!(error = %e, "[METRICS] Prometheus metrics server terminated");
        }
    });

    let allowed_origin_str = std::env::var("ALLOWED_ORIGIN").unwrap_or_else(|_| "*".to_string());

    let cors = if allowed_origin_str == "*" {
        CorsLayer::new()
            .allow_origin(tower_http::cors::Any)
            .allow_methods(tower_http::cors::Any)
            .allow_headers(tower_http::cors::Any)
    } else {
        let origin = allowed_origin_str
            .parse::<axum::http::HeaderValue>()
            .map_err(|e| {
                anyhow::anyhow!("Invalid ALLOWED_ORIGIN environment variable format: {}", e)
            })?;
        CorsLayer::new()
            .allow_origin(vec![origin])
            .allow_methods(tower_http::cors::Any)
            .allow_headers(tower_http::cors::Any)
    };

    let app = Router::new()
        .route(
            "/v1/agent/execute",
            axum::routing::post(handle_agent_execution),
        )
        .route("/v1/system/hot-swap", axum::routing::post(handle_hot_swap))
        .layer(PropagateRequestIdLayer::new(
            axum::http::header::HeaderName::from_static("x-request-id"),
        ))
        .layer(SetRequestIdLayer::new(
            axum::http::header::HeaderName::from_static("x-request-id"),
            MakeRequestUuid,
        ))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(|request: &Request| {
                    let request_id = request
                        .headers()
                        .get("x-request-id")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("unknown");
                    tracing::info_span!(
                        "http_request",
                        method = %request.method(),
                        uri = %request.uri(),
                        version = ?request.version(),
                        request_id = %request_id,
                    )
                })
                .on_response(
                    |response: &axum::response::Response,
                     latency: Duration,
                     _span: &tracing::Span| {
                        let request_id = response
                            .headers()
                            .get("x-request-id")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or("unknown");
                        tracing::info!(
                            status = %response.status().as_u16(),
                            latency_ms = latency.as_millis() as u64,
                            request_id = %request_id,
                            "HTTP response sent"
                        );
                    },
                ),
        )
        .layer(tower::limit::ConcurrencyLimitLayer::new(10000))
        .layer(tower_http::timeout::TimeoutLayer::new(
            std::time::Duration::from_secs(65),
        ))
        .layer(DefaultBodyLimit::max(1024 * 1024))
        .layer(cors)
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

    tracing::info!("Serein Gateway listening on 0.0.0.0:8080");

    let server = axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        let ctrl_c = async {
            tokio::signal::ctrl_c().await.ok();
        };
        
        #[cfg(unix)]
        let terminate = async {
            if let Ok(mut sig) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                sig.recv().await;
            } else {
                std::future::pending::<()>().await;
            }
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => tracing::info!("Received Ctrl+C (SIGINT)"),
            _ = terminate => tracing::info!("Received K8s Termination Signal (SIGTERM)"),
        }
        let _ = shutdown_tx.send(());
    });

    tokio::select! {
        res = server.into_future() => {
            match res {
                Ok(_) => tracing::info!("Server shut down gracefully."),
                Err(e) => tracing::error!(error = %e, "Server error"),
            }
        }
        _ = async {
            let _ = shutdown_rx.await;
            tokio::time::sleep(std::time::Duration::from_secs(25)).await;
        } => {
            tracing::warn!("Graceful shutdown timed out (25s) - forcing termination");
        }
    }

    drop(state);
    for _ in 0..10 {
        tokio::task::yield_now().await;
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    tracing::info!("Gateway terminated safely.");
    Ok(())
}

fn build_providers_from_env() -> ProvidersConfig {
    let mut provider_configs = Vec::new();

    if std::env::var("GROQ_API_KEY").is_ok() {
        provider_configs.push(ProviderConfig {
            id: "groq".to_string(),
            display_name: "Groq Llama 3 (8B)".to_string(),
            api_format: "openai".to_string(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
            model: "llama3-8b-8192".to_string(),
            api_key: String::new(),
            api_key_env: "GROQ_API_KEY".to_string(),
            timeout_ms: 10_000,
            max_response_bytes: 2 * 1024 * 1024,
            cache_provider: "groq".to_string(),
            temperature: 0.0,
            max_tokens: 1024,
            max_retries: 2,
            concurrency_limit: 15,
            geo_region: None,
            adapter: None,
        });
    }

    if std::env::var("DEEPSEEK_API_KEY").is_ok() {
        provider_configs.push(ProviderConfig {
            id: "deepseek".to_string(),
            display_name: "DeepSeek-V3 Chat".to_string(),
            api_format: "openai".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            model: "deepseek-chat".to_string(),
            api_key: String::new(),
            api_key_env: "DEEPSEEK_API_KEY".to_string(),
            timeout_ms: 15_000,
            max_response_bytes: 5 * 1024 * 1024,
            cache_provider: "deepseek".to_string(),
            temperature: 0.0,
            max_tokens: 4096,
            max_retries: 3,
            concurrency_limit: 20,
            geo_region: None,
            adapter: None,
        });
    }

    if std::env::var("SILICONFLOW_API_KEY").is_ok() {
        provider_configs.push(ProviderConfig {
            id: "siliconflow".to_string(),
            display_name: "SiliconFlow Qwen 2.5".to_string(),
            api_format: "openai".to_string(),
            base_url: "https://api.siliconflow.cn/v1".to_string(),
            model: "Qwen/Qwen2.5-7B-Instruct".to_string(),
            api_key: String::new(),
            api_key_env: "SILICONFLOW_API_KEY".to_string(),
            timeout_ms: 12_000,
            max_response_bytes: 2 * 1024 * 1024,
            cache_provider: "siliconflow".to_string(),
            temperature: 0.0,
            max_tokens: 2048,
            max_retries: 2,
            concurrency_limit: 10,
            geo_region: None,
            adapter: None,
        });
    }

    ProvidersConfig {
        provider: provider_configs,
    }
}
