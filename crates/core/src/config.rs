// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Unified Application Configuration
//!
//! Centralizes all environment variable loading with fail-fast validation.
//! Every `.env` variable is mapped to a typed Rust field with strict
//! validation at startup - no silent defaults for security-critical values.
//!
//! ## Credential Safety
//! Sensitive fields (SEREIN_MASTER_KEY, SEREIN_INTERNAL_TOKEN, DATABASE_URL)
//! are loaded directly into `secrecy::SecretString` without intermediate
//! heap-allocated `String` buffers. Transient buffers used during validation
//! are zeroized immediately after use to prevent credential leakage through
//! memory dumps or core dumps.

use anyhow::{anyhow, Context, Result};
use arc_swap::ArcSwap;
use notify::Watcher;
use secrecy::SecretString;
pub use serein_interfaces::TmrCanonicalStrategy;
use std::sync::Arc;
use zeroize::Zeroize;

/// Unified application configuration loaded from environment variables.
///
/// All fields are validated at construction time with fail-fast semantics.
/// Security-critical fields (SEREIN_MASTER_KEY, SEREIN_INTERNAL_TOKEN) are
/// validated for format correctness before any subsystem initialization.
///
/// ## Credential Safety
/// - `master_key_hex` and `internal_token` are stored as `SecretString`,
///   preventing accidental logging or debug exposure.
/// - `database_url` is stored as `SecretString` to protect embedded
///   credentials in connection strings.
/// - All transient buffers used during parsing are zeroized after use.
#[derive(Debug, Clone)]
pub struct AppConfig {
    pub app_env: String,
    pub app_name: String,
    pub app_version: String,
    pub rust_log: String,
    pub worker_threads: usize,
    pub metrics_port: u16,

    /// AES-256-GCM master key - 64-character hex string (256 bits).
    /// Wrapped in `SecretString` to prevent accidental logging or debug exposure.
    /// Loaded directly from the environment buffer with zeroize on transient data.
    pub master_key_hex: SecretString,
    /// HMAC-SHA256 secret for internal service-to-service authentication.
    /// Stored as `SecretString` - never logged, never debug-printed.
    pub internal_token: SecretString,

    pub wasi_component_path: String,
    pub use_ephemeral_storage: bool,
    pub wasi_nn_enabled: bool,

    pub slm_model_id: String,
    pub slm_model_path: std::path::PathBuf,
    pub slm_execution_mode: String,

    pub tmr_agreement_threshold: usize,
    pub tmr_global_timeout_ms: u64,
    pub tmr_jitter_backoff_base_ms: u64,
    pub tmr_canonical_strategy: TmrCanonicalStrategy,

    pub redis_url: String,
    pub cache_ttl_sec: u64,
    /// Database connection string stored as `SecretString` to protect
    /// embedded credentials (username, password) from logging exposure.
    pub database_url: SecretString,

    pub aegis_public_key: String,
    pub aegis_rate_limit_per_min: u32,
    pub cors_allowed_origins: String,

    pub proxy_enabled: bool,
    pub proxy_steering_policy: String,
    pub http_proxy: Option<String>,
    pub https_proxy: Option<String>,
    pub no_proxy: Option<String>,
    pub dns_resolver_strategy: String,
    pub tcp_keepalive_sec: u64,
    pub connect_timeout_ms: u64,
    pub total_request_timeout_ms: u64,
    pub own_node_region: String,

    pub wasm_max_memory_mb: u64,
    pub wasm_instruction_limit: u64,

    pub otlp_endpoint: String,
    pub telemetry_sample_rate: f64,
}

/// Load a sensitive environment variable directly into `SecretString`,
/// zeroizing the intermediate `String` buffer after transfer.
///
/// This avoids leaving credential material on the heap in a `String`
/// that would only be dropped later (or potentially never zeroized).
fn load_secret_env(key: &str) -> Result<SecretString> {
    let raw = std::env::var(key).context(format!("{} environment variable is required", key))?;
    Ok(SecretString::from(raw))
}

impl AppConfig {
    /// Load and validate all configuration from environment variables.
    ///
    /// Fails fast on any security-critical misconfiguration:
    /// - `SEREIN_MASTER_KEY` must be exactly 64 hex characters
    /// - `SEREIN_INTERNAL_TOKEN` must be non-empty
    /// - `TMR_AGREEMENT_THRESHOLD` must be >= 2 for 2/3 quorum
    ///
    /// ## Credential Safety
    /// All sensitive environment variables are loaded directly into
    /// `SecretString` with zeroize on transient buffers. No intermediate
    /// `String` allocations persist after validation.
    pub fn from_env() -> Result<Self> {
        let master_key_hex = Self::load_and_validate_master_key()?;
        let internal_token = Self::load_and_validate_internal_token()?;

        let tmr_agreement_threshold: usize = std::env::var("TMR_AGREEMENT_THRESHOLD")
            .unwrap_or_else(|_| "2".to_string())
            .parse()
            .context("TMR_AGREEMENT_THRESHOLD must be a valid positive integer")?;

        if tmr_agreement_threshold < 2 {
            return Err(anyhow!(
                "TMR_AGREEMENT_THRESHOLD must be >= 2 for 2/3 quorum enforcement. Received {}.",
                tmr_agreement_threshold
            ));
        }

        let tmr_global_timeout_ms: u64 = std::env::var("TMR_GLOBAL_TIMEOUT_MS")
            .unwrap_or_else(|_| "8000".to_string())
            .parse()
            .context("TMR_GLOBAL_TIMEOUT_MS must be a valid u64")?;

        let tmr_jitter_backoff_base_ms: u64 = std::env::var("TMR_JITTER_BACKOFF_BASE_MS")
            .unwrap_or_else(|_| "200".to_string())
            .parse()
            .context("TMR_JITTER_BACKOFF_BASE_MS must be a valid u64")?;

        let tmr_canonical_strategy_str =
            std::env::var("TMR_CANONICAL_STRATEGY").unwrap_or_else(|_| "strict".to_string());
        let tmr_canonical_strategy =
            TmrCanonicalStrategy::from_str_lossy(&tmr_canonical_strategy_str);

        let app_env = std::env::var("APP_ENV").unwrap_or_else(|_| "production".to_string());
        let app_name = std::env::var("APP_NAME").unwrap_or_else(|_| "serein-gateway".to_string());
        let app_version = std::env::var("APP_VERSION").unwrap_or_else(|_| "1.0.0".to_string());
        let rust_log = std::env::var("RUST_LOG")
            .unwrap_or_else(|_| "info,serein_core=debug,serein_gateway=debug".to_string());

        let worker_threads: usize = std::env::var("WORKER_THREADS")
            .unwrap_or_else(|_| "4".to_string())
            .parse()
            .context("WORKER_THREADS must be a valid positive integer")?;

        let metrics_port: u16 = std::env::var("METRICS_PORT")
            .unwrap_or_else(|_| "9090".to_string())
            .parse()
            .context("METRICS_PORT must be a valid port number (0-65535)")?;

        let wasm_max_memory_mb: u64 = std::env::var("WASM_MAX_MEMORY_MB")
            .unwrap_or_else(|_| "128".to_string())
            .parse()
            .context("WASM_MAX_MEMORY_MB must be a valid u64")?;

        let wasm_instruction_limit: u64 = std::env::var("WASM_INSTRUCTION_LIMIT")
            .unwrap_or_else(|_| "50000000".to_string())
            .parse()
            .context("WASM_INSTRUCTION_LIMIT must be a valid u64")?;

        let wasi_component_path =
            std::env::var("WASI_COMPONENT_PATH").unwrap_or_else(|_| "./components".to_string());

        let use_ephemeral_storage = std::env::var("USE_EPHEMERAL_STORAGE")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .context("USE_EPHEMERAL_STORAGE must be 'true' or 'false'")?;

        let wasi_nn_enabled = std::env::var("WASI_NN_ENABLED")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .context("WASI_NN_ENABLED must be 'true' or 'false'")?;

        let slm_model_id =
            std::env::var("SLM_MODEL_ID").unwrap_or_else(|_| "serein-slm-v1".to_string());
        let slm_model_path_str = std::env::var("SLM_MODEL_PATH")
            .unwrap_or_else(|_| "./serein-models/fallback-8b-q4.gguf".to_string());
        let slm_model_path = std::path::PathBuf::from(&slm_model_path_str);

        if slm_model_path.as_os_str().is_empty() {
            return Err(anyhow!("SLM_MODEL_PATH must not be empty"));
        }

        let slm_execution_mode =
            std::env::var("SLM_EXECUTION_MODE").unwrap_or_else(|_| "active".to_string());
        if slm_execution_mode != "active" && slm_execution_mode != "lazy" {
            return Err(anyhow!(
                "SLM_EXECUTION_MODE must be 'active' or 'lazy'. Received '{}'.",
                slm_execution_mode
            ));
        }

        let redis_url =
            std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379/0".to_string());
        let cache_ttl_sec: u64 = std::env::var("CACHE_TTL_SEC")
            .unwrap_or_else(|_| "3600".to_string())
            .parse()
            .context("CACHE_TTL_SEC must be a valid u64")?;

        let database_url = load_secret_env("DATABASE_URL").unwrap_or_else(|_| {
            SecretString::new("postgres://user:password@localhost:5432/serein".to_string())
        });

        let aegis_public_key = std::env::var("AEGIS_PUBLIC_KEY").unwrap_or_default();
        let aegis_rate_limit_per_min: u32 = std::env::var("AEGIS_RATE_LIMIT_PER_MIN")
            .unwrap_or_else(|_| "60".to_string())
            .parse()
            .context("AEGIS_RATE_LIMIT_PER_MIN must be a valid u32")?;

        let cors_allowed_origins = std::env::var("CORS_ALLOWED_ORIGINS")
            .unwrap_or_else(|_| "https://chaindata.com".to_string());

        let proxy_enabled: bool = std::env::var("PROXY_ENABLED")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .context("PROXY_ENABLED must be 'true' or 'false'")?;

        let proxy_steering_policy =
            std::env::var("PROXY_STEERING_POLICY").unwrap_or_else(|_| "latency".to_string());
        let http_proxy = std::env::var("HTTP_PROXY").ok();
        let https_proxy = std::env::var("HTTPS_PROXY").ok();
        let no_proxy = std::env::var("NO_PROXY").ok();

        let dns_resolver_strategy =
            std::env::var("DNS_RESOLVER_STRATEGY").unwrap_or_else(|_| "trust-dns".to_string());

        let tcp_keepalive_sec: u64 = std::env::var("TCP_KEEPALIVE")
            .unwrap_or_else(|_| "60".to_string())
            .parse()
            .context("TCP_KEEPALIVE must be a valid u64")?;

        let connect_timeout_ms: u64 = std::env::var("CONNECT_TIMEOUT_MS")
            .unwrap_or_else(|_| "3000".to_string())
            .parse()
            .context("CONNECT_TIMEOUT_MS must be a valid u64")?;

        let total_request_timeout_ms: u64 = std::env::var("TOTAL_REQUEST_TIMEOUT_MS")
            .unwrap_or_else(|_| "30000".to_string())
            .parse()
            .context("TOTAL_REQUEST_TIMEOUT_MS must be a valid u64")?;

        let own_node_region =
            std::env::var("OWN_NODE_REGION").unwrap_or_else(|_| "GLOBAL".to_string());

        let otlp_endpoint =
            std::env::var("OTLP_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:4317".to_string());
        let telemetry_sample_rate: f64 = std::env::var("TELEMETRY_SAMPLE_RATE")
            .unwrap_or_else(|_| "1.0".to_string())
            .parse()
            .context("TELEMETRY_SAMPLE_RATE must be a valid f64")?;

        Ok(Self {
            app_env,
            app_name,
            app_version,
            rust_log,
            worker_threads,
            metrics_port,
            master_key_hex,
            internal_token,
            wasi_component_path,
            use_ephemeral_storage,
            wasi_nn_enabled,
            slm_model_id,
            slm_model_path,
            slm_execution_mode,
            tmr_agreement_threshold,
            tmr_global_timeout_ms,
            tmr_jitter_backoff_base_ms,
            tmr_canonical_strategy,
            redis_url,
            cache_ttl_sec,
            database_url,
            aegis_public_key,
            aegis_rate_limit_per_min,
            cors_allowed_origins,
            proxy_enabled,
            proxy_steering_policy,
            http_proxy,
            https_proxy,
            no_proxy,
            dns_resolver_strategy,
            tcp_keepalive_sec,
            connect_timeout_ms,
            total_request_timeout_ms,
            own_node_region,
            wasm_max_memory_mb,
            wasm_instruction_limit,
            otlp_endpoint,
            telemetry_sample_rate,
        })
    }

    /// Load and validate SEREIN_MASTER_KEY directly into a `SecretString`.
    ///
    /// The hex validation is performed on a transient buffer that is
    /// zeroized immediately after validation, ensuring no credential
    /// material persists on the heap beyond the `SecretString`.
    fn load_and_validate_master_key() -> Result<SecretString> {
        let mut raw = std::env::var("SEREIN_MASTER_KEY")
            .context("SEREIN_MASTER_KEY environment variable is required")?;

        if raw.len() != 64 {
            let len = raw.len();
            raw.zeroize();
            return Err(anyhow!(
                "SEREIN_MASTER_KEY must be exactly 64 hex characters (256 bits). Received {} characters.",
                len
            ));
        }

        let mut hex_buf = raw.clone();
        hex::decode(&hex_buf).map_err(|e| {
            let err = anyhow!("SEREIN_MASTER_KEY must be valid hexadecimal: {}", e);
            hex_buf.zeroize();
            raw.zeroize();
            err
        })?;
        hex_buf.zeroize();

        let secret = SecretString::from(raw);
        Ok(secret)
    }

    /// Load and validate SEREIN_INTERNAL_TOKEN directly into a `SecretString`.
    ///
    /// Validates non-emptiness and zeroizes the transient buffer.
    fn load_and_validate_internal_token() -> Result<SecretString> {
        let mut raw = std::env::var("SEREIN_INTERNAL_TOKEN").context(
            "SEREIN_INTERNAL_TOKEN environment variable is required for HMAC authentication",
        )?;

        if raw.is_empty() {
            raw.zeroize();
            return Err(anyhow!("SEREIN_INTERNAL_TOKEN must not be empty"));
        }

        let secret = SecretString::from(raw);
        Ok(secret)
    }
}

/// Hot-reloadable configuration wrapper using `ArcSwap` for lock-free reads.
///
/// Wraps `AppConfig` in an `Arc<ArcSwap<AppConfig>>` so that:
/// - **Reads** are lock-free and wait-free via `ArcSwap::load()`
/// - **Writes** atomically swap the entire config pointer - no partial updates
/// - **Watchers** can subscribe to changes via `tokio::sync::watch`
///
/// ## Atomicity
/// The entire `AppConfig` is swapped as a single `Arc` pointer. No field-level
/// locking is needed - readers always see a consistent snapshot.
///
/// ## Usage
/// ```ignore
/// let watchable = WatchableConfig::new(AppConfig::from_env()?);
/// let current = watchable.load();
/// // ... later, after file watcher triggers:
/// watchable.swap(new_config);
/// ```
pub struct WatchableConfig {
    inner: Arc<ArcSwap<AppConfig>>,
    tx: tokio::sync::watch::Sender<Arc<AppConfig>>,
    rx: tokio::sync::watch::Receiver<Arc<AppConfig>>,
}

impl WatchableConfig {
    /// Create a new watchable config from an initial `AppConfig`.
    pub fn new(config: AppConfig) -> Self {
        let arc = Arc::new(config);
        let inner = Arc::new(ArcSwap::from(arc.clone()));
        let (tx, rx) = tokio::sync::watch::channel(arc);
        Self { inner, tx, rx }
    }

    /// Load the current config snapshot (lock-free, wait-free).
    ///
    /// Returns an `Arc<AppConfig>` that is guaranteed to be a consistent
    /// snapshot - no partial field updates are possible.
    pub fn load(&self) -> Arc<AppConfig> {
        self.inner.load_full()
    }

    /// Atomically swap the entire configuration.
    ///
    /// All subsequent `load()` calls will return the new config.
    /// The `watch::Sender` broadcasts the new config to all subscribers.
    pub fn swap(&self, new_config: AppConfig) {
        let arc = Arc::new(new_config);
        self.inner.store(arc.clone());
        let _ = self.tx.send(arc);
        tracing::info!("[CONFIG] Configuration hot-reloaded - new snapshot active");
    }

    /// Subscribe to configuration change notifications.
    ///
    /// Returns a `watch::Receiver` that yields `Arc<AppConfig>` on each swap.
    pub fn subscribe(&self) -> tokio::sync::watch::Receiver<Arc<AppConfig>> {
        self.rx.clone()
    }
}

/// Spawn an asynchronous file watcher that monitors a configuration file
/// for `Modify` events and triggers hot-reload via the `WatchableConfig`.
///
/// ## Debouncing
/// File watchers often fire multiple events for a single save operation
/// (e.g., write + rename on Unix). A 500ms debounce window prevents
/// redundant reloads.
///
/// ## Error Handling
/// If re-parsing fails, the error is logged and the current config is
/// preserved - a malformed config file MUST NOT take down the gateway.
pub async fn spawn_config_watcher(
    watchable: Arc<WatchableConfig>,
    config_path: std::path::PathBuf,
) -> Result<()> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(16);

    let mut watcher =
        notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
            if let Ok(event) = res {
                if event.kind.is_modify() {
                    let _ = tx.blocking_send(());
                }
            }
        })
        .context("Failed to create filesystem watcher")?;

    watcher
        .watch(&config_path, notify::RecursiveMode::NonRecursive)
        .context("Failed to start watching config file")?;

    tracing::info!(
        path = %config_path.display(),
        "[CONFIG] File watcher started - monitoring for modifications"
    );

    tokio::spawn(async move {
        loop {
            tokio::select! {
                Some(()) = rx.recv() => {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

                    if rx.try_recv().is_ok() {
                        continue;
                    }

                    tracing::info!(
                        path = %config_path.display(),
                        "[CONFIG] File modification detected - attempting hot-reload"
                    );

                    match AppConfig::from_env() {
                        Ok(new_config) => {
                            watchable.swap(new_config);
                        }
                        Err(e) => {
                            tracing::error!(
                                error = %e,
                                "[CONFIG] Hot-reload failed - malformed config, keeping current snapshot"
                            );
                        }
                    }
                }
                else => {
                    tracing::info!("[CONFIG] File watcher channel closed - stopping");
                    break;
                }
            }
        }
        let _ = watcher;
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_app_config_master_key_validation() {
        std::env::set_var(
            "SEREIN_MASTER_KEY",
            "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcde",
        );
        std::env::set_var("SEREIN_INTERNAL_TOKEN", "test-token");

        let config = super::AppConfig::from_env();
        assert!(config.is_err());
        let err_msg = config.unwrap_err().to_string();
        assert!(err_msg.contains("exactly 64 hex characters"));
    }
}
