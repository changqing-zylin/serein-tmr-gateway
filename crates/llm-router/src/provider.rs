// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Provider - LLM Provider Abstraction Layer
//!
//! Defines the `LlmProvider` trait and its concrete implementations for
//! heterogeneous LLM provider invocation. Providers are loaded from
//! `providers.toml` and built into trait objects at runtime.
//!
//! ## Architecture
//! - **ProviderRequest / ProviderResponse**: Decoupled request/response types
//!   that eliminate transport-layer coupling from the trait signature
//! - **LlmProvider trait**: Provider-agnostic invocation interface with
//!   injected `reqwest::Client` for transport decoupling
//! - **GenericOpenAiProvider**: Unified adapter for OpenAI-compatible APIs
//!   (DeepSeek, Groq, vLLM, Ollama, etc.)
//! - **GeminiProvider**: Google Gemini REST API adapter
//! - **WasmAdapterProvider**: WASM-plugin-driven provider for non-standard
//!   auth schemes (IAM signing, token refresh). Routes through a `.wasm`
//!   adapter component that implements the `serein-adapter.wit` contract.
//! - **ProviderConfig**: TOML-deserializable configuration for dynamic loading

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

/// Provider-agnostic request structure.
///
/// Decouples the `LlmProvider::invoke` signature from raw string prompts,
/// enabling structured input with optional images and temperature control.
///
/// ## Tenant Isolation
/// The `tenant_id` field is mandatory for singleflight and cache key
/// derivation. It ensures that requests from different tenants are never
/// deduplicated together, preventing cross-tenant data spillage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderRequest {
    pub prompt: String,
    pub tenant_id: String,
    pub images: Option<Vec<String>>,
    pub temperature: f32,
}

impl ProviderRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            tenant_id: String::new(),
            images: None,
            temperature: 0.0,
        }
    }

    /// Attach a tenant identifier for singleflight and cache isolation.
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = tenant_id.into();
        self
    }
}

/// Provider-agnostic response structure.
///
/// Wraps the LLM output in a typed container, decoupling consumers
/// from transport-layer response formats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderResponse {
    pub content: String,
}

/// Error from a TMR cloud node that may trigger fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TmrNodeError {
    HttpForbidden,
    HttpRateLimited,
    Timeout,
    CircuitBreakerOpen,
    ServerError(u16),
    ParseError(String),
    NetworkError(String),
}

impl TmrNodeError {
    pub fn should_trigger_fallback(&self) -> bool {
        matches!(
            self,
            TmrNodeError::HttpForbidden
                | TmrNodeError::HttpRateLimited
                | TmrNodeError::Timeout
                | TmrNodeError::CircuitBreakerOpen
        )
    }

    pub fn from_http_status(status: u16) -> Option<Self> {
        match status {
            403 => Some(TmrNodeError::HttpForbidden),
            429 => Some(TmrNodeError::HttpRateLimited),
            s if (500..=599).contains(&s) => Some(TmrNodeError::ServerError(s)),
            _ => None,
        }
    }
}

/// Provider build and configuration errors.
#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("Provider configuration error: {0}")]
    ConfigError(String),

    #[error("Unknown api_format '{format}' for provider '{id}'")]
    UnknownFormat { format: String, id: String },
}

/// Provider-agnostic trait for LLM invocation.
///
/// Abstracts the HTTP call details for any LLM provider into a single
/// async interface. Each provider implementation handles:
/// - URL construction and authentication header injection
/// - Request body serialization in the provider's native format
/// - Response parsing and error classification
/// - Safety limits (response size cap, timeout enforcement)
///
/// ## Transport Decoupling
/// The `reqwest::Client` is injected during provider initialization and
/// stored as a struct field, removing it from the `invoke` signature.
/// This decouples the trait from the HTTP transport layer (OCP compliance).
#[async_trait]
pub trait LlmProvider: Send + Sync + 'static {
    /// Unique identifier for this provider (e.g., "gemini", "deepseek", "groq").
    fn provider_id(&self) -> &str;

    /// Human-readable display name for logging (e.g., "Gemini 2.0 Flash").
    fn display_name(&self) -> &str;

    /// Prompt-cache provider identifier string for cache directive formatting.
    fn cache_provider(&self) -> String;

    /// Per-provider request timeout in milliseconds.
    fn timeout_ms(&self) -> u64;

    /// Invoke the LLM with the given request and return the response.
    ///
    /// ## Error Classification
    /// Implementations MUST map HTTP status codes to `TmrNodeError` variants:
    /// - 403 → `TmrNodeError::HttpForbidden`
    /// - 429 → `TmrNodeError::HttpRateLimited`
    /// - 5xx → `TmrNodeError::ServerError(code)`
    /// - Network/timeout → `TmrNodeError::NetworkError(msg)`
    /// - Parse failure → `TmrNodeError::ParseError(msg)`
    async fn invoke(&self, req: &ProviderRequest) -> Result<ProviderResponse, TmrNodeError>;
}

/// Static configuration for a single LLM provider loaded from `providers.toml`.
///
/// Each entry in the TOML `[[provider]]` array maps directly to this struct.
/// The `api_format` field determines which `LlmProvider` implementation is used:
/// - `"openai"` → `GenericOpenAiProvider` (DeepSeek, Groq, vLLM, Ollama, etc.)
/// - `"gemini"` → `GeminiProvider` (Google Gemini REST API)
///
/// ## Sovereign Routing / Geo-fencing
/// The `geo_region` field enables compliance-aware provider filtering. When a
/// required region is specified at scheduling time, providers whose `geo_region`
/// does not match the compliance requirement are dynamically stripped from the
/// TMR consensus pool. This ensures data residency requirements (e.g., EU GDPR)
/// are enforced at the routing layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Unique identifier (e.g., "gemini", "deepseek", "groq").
    pub id: String,

    /// Human-readable display name for logging.
    pub display_name: String,

    /// API format: "openai" or "gemini".
    #[serde(default = "default_api_format")]
    pub api_format: String,

    /// Base URL for the provider API (e.g., "https://api.deepseek.com/v1").
    /// The OpenAI compatibility layer automatically appends "/chat/completions".
    pub base_url: String,

    /// Model identifier passed in the request body.
    pub model: String,

    /// API key (loaded from environment variable or TOML).
    #[serde(default)]
    pub api_key: String,

    /// Environment variable name holding the API key.
    /// If set, takes precedence over `api_key` field.
    #[serde(default)]
    pub api_key_env: String,

    /// Per-provider timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    /// Maximum response body size in bytes (safety limit).
    #[serde(default = "default_max_response_bytes")]
    pub max_response_bytes: usize,

    /// Prompt-cache provider variant for cache directive formatting.
    #[serde(default)]
    pub cache_provider: String,

    /// Sampling temperature - forced to 0.0 for deterministic TMR consensus.
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Maximum tokens in the LLM response.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Maximum retry attempts on transient failures.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Maximum concurrent requests to this provider.
    #[serde(default = "default_concurrency_limit")]
    pub concurrency_limit: u32,

    /// Geographic region for sovereign routing / geo-fencing compliance.
    ///
    /// When set, this provider will only be eligible for TMR consensus
    /// scheduling when the required region matches. Providers without a
    /// `geo_region` are always eligible (no restriction).
    ///
    /// ## Example values
    /// - `"EU"` - EU data residency (GDPR compliance)
    /// - `"US"` - US data residency
    /// - `"CN"` - China data residency
    /// - `"APAC"` - Asia-Pacific data residency
    #[serde(default)]
    pub geo_region: Option<String>,

    /// WASM adapter plugin filename for non-OpenAI-compatible auth schemes.
    ///
    /// When set, the TMR engine loads the specified `.wasm` component to
    /// handle provider-specific authentication (IAM signing, token refresh,
    /// non-standard headers) before dispatching the HTTP request. The adapter
    /// produces an `http-request-spec` that the host executes via its own
    /// reqwest client pool - the WASM plugin never touches the network.
    ///
    /// Providers using standard OpenAI-compatible auth (Bearer token) do not
    /// need an adapter and should leave this field unset.
    ///
    /// ## Example
    /// - `"baidu_adapter.wasm"` - Baidu Qianfan IAM v3 signing adapter
    #[serde(default)]
    pub adapter: Option<String>,
}

fn default_api_format() -> String {
    "openai".to_string()
}

fn default_timeout_ms() -> u64 {
    15_000
}

fn default_max_response_bytes() -> usize {
    5 * 1024 * 1024
}

fn default_temperature() -> f32 {
    0.0
}

fn default_max_tokens() -> u32 {
    1024
}

fn default_max_retries() -> u32 {
    3
}

fn default_concurrency_limit() -> u32 {
    10
}

impl ProviderConfig {
    /// Resolve the effective API key: prefer environment variable, fall back to inline value.
    pub fn resolve_api_key(&self) -> Option<String> {
        if !self.api_key_env.is_empty() {
            std::env::var(&self.api_key_env).ok().filter(|k| !k.is_empty())
        } else if !self.api_key.is_empty() {
            Some(self.api_key.clone())
        } else {
            None
        }
    }

    /// Resolve the cache provider identifier string.
    pub fn resolve_cache_provider(&self) -> String {
        if !self.cache_provider.is_empty() {
            self.cache_provider.clone()
        } else {
            self.id.clone()
        }
    }

    /// Check whether this provider satisfies the given geo-fencing requirement.
    ///
    /// A provider is eligible if:
    /// - It has no `geo_region` set (no restriction), OR
    /// - Its `geo_region` matches the required region (case-insensitive)
    pub fn satisfies_geo_requirement(&self, required_region: &str) -> bool {
        match &self.geo_region {
            None => true,
            Some(region) => region.eq_ignore_ascii_case(required_region),
        }
    }

    /// Build an `LlmProvider` trait object from this configuration.
    ///
    /// The `reqwest::Client` is injected into the provider for connection
    /// pooling and transport decoupling.
    pub fn build_provider(&self, client: reqwest::Client) -> Result<Arc<dyn LlmProvider>, ProviderError> {
        if self.adapter.is_some() {
            return Ok(Arc::new(WasmAdapterProvider::from_config(self, client)));
        }
        match self.api_format.to_lowercase().as_str() {
            "openai" => Ok(Arc::new(GenericOpenAiProvider::from_config(self, client))),
            "gemini" => Ok(Arc::new(GeminiProvider::from_config(self, client))),
            other => Err(ProviderError::UnknownFormat {
                format: other.to_string(),
                id: self.id.clone(),
            }),
        }
    }
}

/// Trait for filtering providers based on compliance requirements.
///
/// Implementations determine which providers are eligible for TMR consensus
/// scheduling based on sovereign routing / geo-fencing constraints.
pub trait GeoFilter: Send + Sync {
    /// Filter providers, retaining only those eligible for the given region.
    fn filter_providers(
        &self,
        providers: &[ProviderConfig],
        required_region: &str,
    ) -> Vec<ProviderConfig>;
}

/// Sovereign routing filter that enforces geo-fencing compliance.
///
/// Before scheduling LLMs for TMR consensus, this filter dynamically strips
/// out any providers whose `geo_region` does not match the compliance
/// requirement. Providers without a `geo_region` are always eligible.
///
/// ## Data Residency Enforcement
/// When a required region is specified (e.g., "EU" for GDPR compliance),
/// only providers in that region - or providers with no region restriction -
/// are included in the filtered result. This ensures that sensitive data
/// never leaves the required jurisdiction.
pub struct SovereignGeoFilter;

impl SovereignGeoFilter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SovereignGeoFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl GeoFilter for SovereignGeoFilter {
    fn filter_providers(
        &self,
        providers: &[ProviderConfig],
        required_region: &str,
    ) -> Vec<ProviderConfig> {
        let eligible: Vec<ProviderConfig> = providers
            .iter()
            .filter(|p| p.satisfies_geo_requirement(required_region))
            .cloned()
            .collect();

        let stripped_count = providers.len() - eligible.len();
        if stripped_count > 0 {
            tracing::warn!(
                required_region = required_region,
                total_providers = providers.len(),
                eligible_providers = eligible.len(),
                stripped = stripped_count,
                "[GEO-FILTER] Providers stripped due to geo-fencing compliance"
            );
        } else {
            tracing::debug!(
                required_region = required_region,
                eligible_providers = eligible.len(),
                "[GEO-FILTER] All providers eligible for region"
            );
        }

        eligible
    }
}

/// Generic OpenAI-compatible LLM provider.
///
/// Works with any API that follows the OpenAI chat completions schema:
/// - DeepSeek (`https://api.deepseek.com/v1`)
/// - Groq (`https://api.groq.com/openai/v1`)
/// - Local vLLM (`http://localhost:8000/v1`)
/// - Ollama (`http://localhost:11434/v1`)
///
/// ## Transport Decoupling
/// The `reqwest::Client` is stored as a struct field, injected during
/// initialization. This removes the HTTP transport from the `LlmProvider`
/// trait signature, enabling alternative transport implementations.
#[derive(Debug, Clone)]
pub struct GenericOpenAiProvider {
    id: String,
    display_name: String,
    base_url: String,
    model: String,
    api_key: Option<String>,
    timeout_ms: u64,
    max_response_bytes: usize,
    cache_provider: String,
    client: reqwest::Client,
}

impl GenericOpenAiProvider {
    pub fn from_config(config: &ProviderConfig, client: reqwest::Client) -> Self {
        Self {
            id: config.id.clone(),
            display_name: config.display_name.clone(),
            base_url: config.base_url.clone(),
            model: config.model.clone(),
            api_key: config.resolve_api_key(),
            timeout_ms: config.timeout_ms,
            max_response_bytes: config.max_response_bytes,
            cache_provider: config.resolve_cache_provider(),
            client,
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "is_zero_f32", default)]
    temperature: f32,
    stream: bool,
}

fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    content: String,
}

#[async_trait]
impl LlmProvider for GenericOpenAiProvider {
    fn provider_id(&self) -> &str {
        &self.id
    }

    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn cache_provider(&self) -> String {
        self.cache_provider.clone()
    }

    fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }

    async fn invoke(&self, req: &ProviderRequest) -> Result<ProviderResponse, TmrNodeError> {
        let api_key = self.api_key.clone().ok_or_else(|| {
            TmrNodeError::NetworkError(format!("{} API key not configured", self.display_name))
        })?;

        let body = OpenAiRequest {
            model: self.model.clone(),
            messages: vec![OpenAiMessage {
                role: "user",
                content: req.prompt.clone(),
            }],
            temperature: req.temperature,
            stream: false,
        };

        let endpoint_url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let mut request = self.client
            .post(&endpoint_url)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body);

        if self.base_url.contains("googleapis.com") {
            request = request.header("x-goog-api-key", &api_key);
        }

        let response = request.send().await.map_err(|e| {
            TmrNodeError::NetworkError(format!("{} request failed: {}", self.display_name, e))
        })?;

        let status = response.status();
        if status.as_u16() == 403 {
            return Err(TmrNodeError::HttpForbidden);
        }
        if status.as_u16() == 429 {
            return Err(TmrNodeError::HttpRateLimited);
        }
        if status.as_u16() >= 500 {
            return Err(TmrNodeError::ServerError(status.as_u16()));
        }

        let bytes = response.bytes().await.map_err(|e| {
            TmrNodeError::ParseError(format!("Failed to read {} bytes: {}", self.display_name, e))
        })?;

        if bytes.len() > self.max_response_bytes {
            return Err(TmrNodeError::ParseError(format!(
                "{} response exceeded {}B safety limit",
                self.display_name, self.max_response_bytes
            )));
        }

        let resp: OpenAiResponse = serde_json::from_slice(&bytes).map_err(|e| {
            TmrNodeError::ParseError(format!("{} response parse failed: {}", self.display_name, e))
        })?;

        resp.choices
            .first()
            .map(|c| ProviderResponse { content: c.message.content.clone() })
            .ok_or_else(|| {
                TmrNodeError::ParseError(format!("{} returned no choices", self.display_name))
            })
    }
}

/// Google Gemini REST API provider.
///
/// Uses the Gemini-native `generateContent` endpoint rather than the
/// OpenAI-compatible schema. The API key is passed as a query parameter.
///
/// ## Transport Decoupling
/// The `reqwest::Client` is stored as a struct field, injected during
/// initialization. This removes the HTTP transport from the `LlmProvider`
/// trait signature.
#[derive(Debug, Clone)]
pub struct GeminiProvider {
    id: String,
    display_name: String,
    base_url: String,
    api_key: Option<String>,
    timeout_ms: u64,
    max_response_bytes: usize,
    client: reqwest::Client,
}

impl GeminiProvider {
    pub fn from_config(config: &ProviderConfig, client: reqwest::Client) -> Self {
        Self {
            id: config.id.clone(),
            display_name: config.display_name.clone(),
            base_url: config.base_url.clone(),
            api_key: config.resolve_api_key(),
            timeout_ms: config.timeout_ms,
            max_response_bytes: config.max_response_bytes,
            client,
        }
    }
}

#[derive(Debug, Serialize)]
struct GeminiRequestBody {
    contents: Vec<GeminiRequestBodyContent>,
    #[serde(rename = "generationConfig", skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Serialize)]
struct GeminiRequestBodyContent {
    parts: Vec<GeminiRequestBodyPart>,
}

#[derive(Debug, Serialize)]
struct GeminiRequestBodyPart {
    text: String,
}

#[derive(Debug, Serialize)]
struct GeminiGenerationConfig {
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseBody {
    candidates: Vec<GeminiResponseBodyCandidate>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseBodyCandidate {
    content: GeminiResponseBodyContent,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseBodyContent {
    parts: Vec<GeminiResponseBodyPart>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseBodyPart {
    text: String,
}

#[async_trait]
impl LlmProvider for GeminiProvider {
    fn provider_id(&self) -> &str {
        &self.id
    }

    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn cache_provider(&self) -> String {
        "gemini".to_string()
    }

    fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }

    async fn invoke(&self, req: &ProviderRequest) -> Result<ProviderResponse, TmrNodeError> {
        let api_key = self.api_key.clone().ok_or_else(|| {
            TmrNodeError::NetworkError(format!("{} API key not configured", self.display_name))
        })?;

        let url = format!("{}?key={}", self.base_url, api_key);

        let generation_config = if req.temperature > 0.0 {
            Some(GeminiGenerationConfig {
                temperature: req.temperature,
            })
        } else {
            None
        };

        let body = GeminiRequestBody {
            contents: vec![GeminiRequestBodyContent {
                parts: vec![GeminiRequestBodyPart { text: req.prompt.clone() }],
            }],
            generation_config,
        };

        let response = self.client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                let redacted = e.to_string().replace(&api_key, "[REDACTED]");
                TmrNodeError::NetworkError(format!("{} request failed: {}", self.display_name, redacted))
            })?;

        let status = response.status();
        if status.as_u16() == 403 {
            return Err(TmrNodeError::HttpForbidden);
        }
        if status.as_u16() == 429 {
            return Err(TmrNodeError::HttpRateLimited);
        }
        if status.as_u16() >= 500 {
            return Err(TmrNodeError::ServerError(status.as_u16()));
        }

        let bytes = response.bytes().await.map_err(|e| {
            TmrNodeError::ParseError(format!("Failed to read {} bytes: {}", self.display_name, e))
        })?;

        if bytes.len() > self.max_response_bytes {
            return Err(TmrNodeError::ParseError(format!(
                "{} response exceeded {}B safety limit",
                self.display_name, self.max_response_bytes
            )));
        }

        let resp: GeminiResponseBody = serde_json::from_slice(&bytes).map_err(|e| {
            TmrNodeError::ParseError(format!("{} response parse failed: {}", self.display_name, e))
        })?;

        resp.candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .map(|p| ProviderResponse { content: p.text.clone() })
            .ok_or_else(|| {
                TmrNodeError::ParseError(format!("{} returned no content", self.display_name))
            })
    }
}

/// WASM-adapter-driven LLM provider for non-standard authentication schemes.
///
/// Routes requests through a `.wasm` adapter component that implements the
/// `serein-adapter.wit` contract. The adapter handles provider-specific
/// authentication (IAM signing, token refresh, non-standard headers) and
/// produces an `http-request-spec` that the host executes via its own
/// reqwest client pool.
///
/// The `base_url` from `ProviderConfig` is injected into the
/// `StandardizedRequest` so the WASM plugin can construct the correct
/// provider endpoint dynamically - no hardcoded URLs.
///
/// ## Transport Decoupling
/// The `reqwest::Client` is stored as a struct field, injected during
/// initialization. The WASM adapter never touches the network directly.
#[derive(Debug, Clone)]
pub struct WasmAdapterProvider {
    id: String,
    display_name: String,
    base_url: String,
    model: String,
    api_key: Option<String>,
    timeout_ms: u64,
    max_response_bytes: usize,
    cache_provider: String,
    adapter_path: String,
    client: reqwest::Client,
}

impl WasmAdapterProvider {
    pub fn from_config(config: &ProviderConfig, client: reqwest::Client) -> Self {
        let adapter_path = config
            .adapter
            .clone()
            .unwrap_or_else(|| format!("{}_adapter.wasm", config.id));

        Self {
            id: config.id.clone(),
            display_name: config.display_name.clone(),
            base_url: config.base_url.clone(),
            model: config.model.clone(),
            api_key: config.resolve_api_key(),
            timeout_ms: config.timeout_ms,
            max_response_bytes: config.max_response_bytes,
            cache_provider: config.resolve_cache_provider(),
            adapter_path,
            client,
        }
    }
}

#[async_trait]
impl LlmProvider for WasmAdapterProvider {
    fn provider_id(&self) -> &str {
        &self.id
    }

    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn cache_provider(&self) -> String {
        self.cache_provider.clone()
    }

    fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }

    async fn invoke(&self, req: &ProviderRequest) -> Result<ProviderResponse, TmrNodeError> {
        let api_key = self.api_key.clone().ok_or_else(|| {
            TmrNodeError::NetworkError(format!("{} API key not configured", self.display_name))
        })?;

        tracing::info!(
            "Delegating request transformation to WASM adapter: {}",
            self.adapter_path
        );

        // OPTIMIZE: Pool this Wasmtime Engine globally in production to avoid cold-start overhead.
        let mut wasm_config = wasmtime::Config::new();
        wasm_config.wasm_component_model(true);
        let engine = wasmtime::Engine::new(&wasm_config).map_err(|e| {
            TmrNodeError::NetworkError(format!("Failed to create WASM engine: {}", e))
        })?;

        let wasm_bytes = std::fs::read(&self.adapter_path).map_err(|e| {
            TmrNodeError::NetworkError(format!(
                "Failed to read adapter '{}': {}",
                self.adapter_path, e
            ))
        })?;

        let component = wasmtime::component::Component::new(&engine, &wasm_bytes).map_err(|e| {
            TmrNodeError::NetworkError(format!(
                "Failed to compile adapter '{}': {}",
                self.adapter_path, e
            ))
        })?;

        let linker = wasmtime::component::Linker::new(&engine);
        let mut store = wasmtime::Store::new(&engine, ());

        let instance =
            linker.instantiate(&mut store, &component).map_err(|e| {
                TmrNodeError::NetworkError(format!(
                    "Failed to instantiate adapter '{}': {}",
                    self.adapter_path, e
                ))
            })?;

        let transform_func = {
            let mut exports = instance.exports(&mut store);
            if let Some(mut interface) = exports.instance("serein:adapter/provider-adapter@1.0.0") {
                interface.func("transform-request")
            } else {
                None
            }
        }.or_else(|| instance.get_func(&mut store, "transform-request"))
         .ok_or_else(|| TmrNodeError::NetworkError("Export 'transform-request' not found in component".to_string()))?;

        let req_val = wasmtime::component::Val::Record(vec![
            (
                "model".to_string(),
                wasmtime::component::Val::String(self.model.clone()),
            ),
            (
                "prompt".to_string(),
                wasmtime::component::Val::String(req.prompt.clone()),
            ),
            (
                "temperature".to_string(),
                wasmtime::component::Val::Float32(req.temperature),
            ),
            (
                "base-url".to_string(),
                wasmtime::component::Val::String(self.base_url.clone()),
            ),
        ]);

        let api_key_val = wasmtime::component::Val::String(api_key.clone());

        let mut results = [wasmtime::component::Val::Bool(false)];
        transform_func
            .call(&mut store, &[req_val, api_key_val], &mut results)
            .map_err(|e| {
                TmrNodeError::NetworkError(format!(
                    "WASM adapter 'transform-request' call trapped: {}",
                    e
                ))
            })?;

        let http_spec = match &results[0] {
            wasmtime::component::Val::Result(inner) => match inner {
                Ok(spec_opt) => match spec_opt {
                    Some(box_val) => box_val,
                    None => {
                        return Err(TmrNodeError::NetworkError(
                            "WASM adapter 'transform-request' returned empty result".to_string(),
                        ));
                    }
                },
                Err(err_opt) => {
                    let err_msg = match err_opt {
                        Some(box_val) => match &**box_val {
                            wasmtime::component::Val::String(s) => s.clone(),
                            other => format!("{:?}", other),
                        },
                        None => "unknown error".to_string(),
                    };
                    return Err(TmrNodeError::NetworkError(format!(
                        "WASM adapter 'transform-request' returned error: {}",
                        err_msg
                    )));
                }
            },
            other => {
                return Err(TmrNodeError::NetworkError(format!(
                    "WASM adapter 'transform-request' returned unexpected value: {:?}",
                    other
                )));
            }
        };

        let (spec_url, spec_headers, spec_body) = match &**http_spec {
            wasmtime::component::Val::Record(fields) => {
                let mut url = String::new();
                let mut headers: Vec<(String, String)> = Vec::new();
                let mut body = String::new();
                for (name, val) in fields {
                    match name.as_str() {
                        "url" => {
                            if let wasmtime::component::Val::String(s) = val {
                                url = s.clone();
                            }
                        }
                        "headers" => {
                            if let wasmtime::component::Val::List(items) = val {
                                for item in items {
                                    if let wasmtime::component::Val::Tuple(tuple) = item {
                                        if tuple.len() == 2 {
                                            let h_name = match &tuple[0] {
                                                wasmtime::component::Val::String(s) => s.clone(),
                                                _ => continue,
                                            };
                                            let h_value = match &tuple[1] {
                                                wasmtime::component::Val::String(s) => s.clone(),
                                                _ => continue,
                                            };
                                            headers.push((h_name, h_value));
                                        }
                                    }
                                }
                            }
                        }
                        "body" => {
                            if let wasmtime::component::Val::String(s) = val {
                                body = s.clone();
                            }
                        }
                        _ => {}
                    }
                }
                (url, headers, body)
            }
            other => {
                return Err(TmrNodeError::NetworkError(format!(
                    "WASM adapter returned unexpected HttpRequestSpec: {:?}",
                    other
                )));
            }
        };

        let mut http_request = self
            .client
            .post(&spec_url)
            .body(spec_body)
            .header("Content-Type", "application/json");
        for (name, value) in &spec_headers {
            http_request = http_request.header(name.as_str(), value.as_str());
        }

        let response = http_request.send().await.map_err(|e| {
            TmrNodeError::NetworkError(format!("{} request failed: {}", self.display_name, e))
        })?;

        let status = response.status();
        if let Some(err) = TmrNodeError::from_http_status(status.as_u16()) {
            return Err(err);
        }

        let raw_body = response.text().await.map_err(|e| {
            TmrNodeError::ParseError(format!(
                "Failed to read {} response text: {}",
                self.display_name, e
            ))
        })?;

        if raw_body.len() > self.max_response_bytes {
            return Err(TmrNodeError::ParseError(format!(
                "{} response exceeded {}B safety limit",
                self.display_name, self.max_response_bytes
            )));
        }

        let parse_func = {
            let mut exports = instance.exports(&mut store);
            if let Some(mut interface) = exports.instance("serein:adapter/provider-adapter@1.0.0") {
                interface.func("parse-response")
            } else {
                None
            }
        }.or_else(|| instance.get_func(&mut store, "parse-response"))
         .ok_or_else(|| TmrNodeError::NetworkError("Export 'parse-response' not found in component".to_string()))?;

        let raw_body_val = wasmtime::component::Val::String(raw_body);
        let mut parse_results = [wasmtime::component::Val::Bool(false)];
        parse_func
            .call(&mut store, &[raw_body_val], &mut parse_results)
            .map_err(|e| {
                TmrNodeError::ParseError(format!(
                    "WASM adapter 'parse-response' call trapped: {}",
                    e
                ))
            })?;

        let content = match &parse_results[0] {
            wasmtime::component::Val::Result(inner) => match inner {
                Ok(val_opt) => match val_opt {
                    Some(box_val) => match &**box_val {
                        wasmtime::component::Val::String(s) => s.clone(),
                        other => {
                            return Err(TmrNodeError::ParseError(format!(
                                "WASM adapter 'parse-response' returned non-string: {:?}",
                                other
                            )));
                        }
                    },
                    None => {
                        return Err(TmrNodeError::ParseError(
                            "WASM adapter 'parse-response' returned empty result".to_string(),
                        ));
                    }
                },
                Err(err_opt) => {
                    let err_msg = match err_opt {
                        Some(box_val) => match &**box_val {
                            wasmtime::component::Val::String(s) => s.clone(),
                            other => format!("{:?}", other),
                        },
                        None => "unknown error".to_string(),
                    };
                    return Err(TmrNodeError::ParseError(format!(
                        "WASM adapter 'parse-response' returned error: {}",
                        err_msg
                    )));
                }
            },
            other => {
                return Err(TmrNodeError::ParseError(format!(
                    "WASM adapter 'parse-response' returned unexpected value: {:?}",
                    other
                )));
            }
        };

        Ok(ProviderResponse { content })
    }
}

/// TOML-deserializable configuration file for dynamic provider loading.
///
/// ## Example `providers.toml`
/// ```toml
/// [[provider]]
/// id = "gemini"
/// display_name = "Gemini 2.0 Flash"
/// api_format = "gemini"
/// base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
/// model = "gemini-2.0-flash"
/// api_key_env = "GEMINI_API_KEY"
/// timeout_ms = 15000
///
/// [[provider]]
/// id = "deepseek"
/// display_name = "DeepSeek Chat"
/// api_format = "openai"
/// base_url = "https://api.deepseek.com/v1"
/// model = "deepseek-chat"
/// api_key_env = "DEEPSEEK_API_KEY"
/// timeout_ms = 45000
///
/// [[provider]]
/// id = "groq"
/// display_name = "Groq (LLaMA 3 8B)"
/// api_format = "openai"
/// base_url = "https://api.groq.com/openai/v1"
/// model = "llama3-8b-8192"
/// api_key_env = "GROQ_API_KEY"
/// geo_region = "US"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvidersConfig {
    pub provider: Vec<ProviderConfig>,
}

impl ProvidersConfig {
    /// Load provider configuration from a TOML file.
    pub fn load_from_file(path: &str) -> Result<Self, ProviderError> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            ProviderError::ConfigError(format!("Failed to read {}: {}", path, e))
        })?;
        Self::parse_toml(&content)
    }

    /// Parse provider configuration from a TOML string.
    pub fn parse_toml(content: &str) -> Result<Self, ProviderError> {
        toml::from_str(content).map_err(|e| {
            ProviderError::ConfigError(format!("Failed to parse providers TOML: {}", e))
        })
    }

    /// Build all configured providers into trait objects.
    ///
    /// The `reqwest::Client` is injected into each provider for connection
    /// pooling and transport decoupling.
    pub fn build_providers(&self, client: &reqwest::Client) -> Result<Vec<Arc<dyn LlmProvider>>, ProviderError> {
        self.provider
            .iter()
            .map(|config| config.build_provider(client.clone()))
            .collect()
    }

    /// Filter providers by geo-fencing compliance requirement.
    ///
    /// Returns a new `ProvidersConfig` containing only providers that satisfy
    /// the required region. Providers without a `geo_region` are always included.
    pub fn filter_by_geo(&self, required_region: &str) -> ProvidersConfig {
        let filter = SovereignGeoFilter::new();
        ProvidersConfig {
            provider: filter.filter_providers(&self.provider, required_region),
        }
    }

    /// Build providers filtered by geo-fencing compliance requirement.
    ///
    /// Convenience method that combines `filter_by_geo` and `build_providers`
    /// into a single call. Only providers satisfying the required region are
    /// built into trait objects.
    pub fn build_providers_for_region(
        &self,
        client: &reqwest::Client,
        required_region: &str,
    ) -> Result<Vec<Arc<dyn LlmProvider>>, ProviderError> {
        let filtered = self.filter_by_geo(required_region);
        filtered.build_providers(client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tmr_node_error_fallback_triggers() {
        assert!(TmrNodeError::HttpForbidden.should_trigger_fallback());
        assert!(TmrNodeError::HttpRateLimited.should_trigger_fallback());
        assert!(TmrNodeError::Timeout.should_trigger_fallback());
        assert!(TmrNodeError::CircuitBreakerOpen.should_trigger_fallback());
        assert!(!TmrNodeError::ServerError(500).should_trigger_fallback());
        assert!(!TmrNodeError::ParseError("test".to_string()).should_trigger_fallback());
    }

    #[test]
    fn test_tmr_node_error_from_http_status() {
        assert!(matches!(TmrNodeError::from_http_status(403), Some(TmrNodeError::HttpForbidden)));
        assert!(matches!(TmrNodeError::from_http_status(429), Some(TmrNodeError::HttpRateLimited)));
        assert!(matches!(TmrNodeError::from_http_status(500), Some(TmrNodeError::ServerError(500))));
        assert!(TmrNodeError::from_http_status(200).is_none());
    }

    #[test]
    fn test_provider_request_new() {
        let req = ProviderRequest::new("test prompt");
        assert_eq!(req.prompt, "test prompt");
        assert!(req.images.is_none());
        assert_eq!(req.temperature, 0.0);
    }

    #[test]
    fn test_provider_config_resolve_cache_provider() {
        let config = ProviderConfig {
            id: "deepseek".to_string(),
            display_name: "DeepSeek".to_string(),
            api_format: "openai".to_string(),
            base_url: "https://api.deepseek.com".to_string(),
            model: "deepseek-chat".to_string(),
            api_key: String::new(),
            api_key_env: String::new(),
            timeout_ms: 15_000,
            max_response_bytes: 5 * 1024 * 1024,
            cache_provider: "deepseek".to_string(),
            temperature: 0.0,
            max_tokens: 4096,
            max_retries: 3,
            concurrency_limit: 20,
            geo_region: None,
            adapter: None,
        };
        assert_eq!(config.resolve_cache_provider(), "deepseek");

        let config_no_cache = ProviderConfig {
            cache_provider: String::new(),
            ..config.clone()
        };
        assert_eq!(config_no_cache.resolve_cache_provider(), "deepseek");
    }

    #[test]
    fn test_providers_config_parse_toml() {
        let toml = r#"
[[provider]]
id = "test"
display_name = "Test Provider"
api_format = "openai"
base_url = "https://api.test.com/v1"
model = "test-model"
api_key = "test-key"
"#;
        let config = ProvidersConfig::parse_toml(toml).unwrap();
        assert_eq!(config.provider.len(), 1);
        assert_eq!(config.provider[0].id, "test");
    }
}
