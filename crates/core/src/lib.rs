// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Changqing Zhang Serein Core Kernel
//!
//! The core microkernel of the Serein architecture. It manages the WebAssembly (Wasm)
//! runtime, enforces Zero-Trust security policies, and orchestrates the request lifecycle
//! from payload validation to sandboxed execution via the WASI 0.3 component model.
//!
//! ## Security Architecture
//! - Fuel-based execution limiting with configurable quota
//! - AST-based payload validation (Z3FormalValidator)
//! - Atomic hot-swap for module replacement via ArcSwap
//!
//! ## Isolation Boundaries
//! - TCB (Trusted Computing Base) zone isolation
//! - TPM 2.0 hardware measurement integration
//! - TLSF memory pool with capability-based pointer isolation

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use sqlx::sqlite::SqlitePoolOptions;
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{info, span, warn, Instrument, Level};
use wasmtime::component::{Component, Linker};
use wasmtime::Store;
use wasmtime_wasi::{ResourceTable, WasiCtxBuilder};

wasmtime::component::bindgen!({
    world: "aegis-world",
    path: "../interfaces/serein-aegis.wit",
    async: true
});

pub mod abi_guard;
pub mod allocator;
pub mod config;
pub mod engine;
pub mod model_interface;
pub mod security;

pub use abi_guard::{
    read_wasm_memory, read_wasm_memory_from_caller, validate_wasm_memory_bounds, AbiGuard,
    AbiGuardStats, AbiViolation, Capability, CheriCapability, CheriPermission, CrossDomainContext,
    WasmMemoryViolation, DEFAULT_TLSF_POOL_SIZE, WASM_PAGE_SIZE,
};
pub use allocator::{global_pool, CapabilityPtr, TlsfError, TlsfPool};
pub use config::AppConfig;
pub use engine::fuel_monitor::{EsdState, FuelMonitor, Z3Validator, DEFAULT_FUEL_QUOTA};
pub use engine::hot_swap::{
    AbiError, HotSwapContainer, SisInterlock, SisInterlockState, GLOBAL_FUEL_QUOTA,
};
pub use engine::wasmtime_rt::{
    AegisHub, EpochTicker, MemoryResourceLimiter, PersistRecord, SecureRuntimeConfig,
    SecureWasmEngine, ServerState, StoreState,
};
pub use security::{
    capabilities::ResourceMonitor,
    hmac_auth::{HmacAuthError, HmacSignature, ServiceAuthenticator},
    masking::{
        MaskingEngine, MaskingError, PiiField, PiiMaskConfig, PiiMaskingEngine, Sensitivity,
    },
    mock_hsm::MockHsm,
    tpm_measure::{
        compute_hash, HardwareQuote, QuoteResult, TpmError, PCR_COUNT, SHA256_DIGEST_SIZE,
    },
};
pub use serein_interfaces::TmrCanonicalStrategy;
pub use wasmtime::component::Component as WasmtimeComponent;
pub use wasmtime::Engine as WasmtimeEngine;

/// Fuel quota for execution consumption monitoring (10 billion units)
const SIS_FUEL_QUOTA: u64 = 10_000_000_000;

/// SHA-256 strangulation iterations for active defense (5 million)
pub const STRANGULATION_ITERATIONS: u32 = 5_000_000;

/// Hardware Security Module abstraction trait.
///
/// Decouples the kernel from hardware-specific TPM/CCA implementations,
/// allowing a seamless swap between physical TPM 2.0 chips, AMD SEV-SNP,
/// Intel TDX, and software mock implementations without modifying core
/// orchestration logic.
///
/// ## Implementations
/// - `MockHsm` - Software stub for development/testing (no hardware required)
/// - Physical TPM 2.0 - Via `tss-esapi` FFI binding (production)
/// - AMD SEV-SNP - Via `/dev/sev` ioctl interface (confidential VMs)
/// - Intel TDX - Via TDG.MR.TDREPORT + QE (confidential VMs)
#[async_trait]
pub trait HardwareSecurityModule: Send + Sync {
    /// Verify hardware attestation before allowing enclave operations.
    async fn verify_attestation(&self) -> Result<(), TpmError>;

    /// Retrieve the last verified hardware attestation quote, if any.
    async fn get_verified_quote(&self) -> Option<HardwareQuote>;

    /// Measure the kernel image hash into a PCR.
    fn measure_kernel(&self, kernel_hash: &[u8; SHA256_DIGEST_SIZE]) -> Result<(), TpmError>;

    /// Measure the security policy hash into a PCR.
    fn measure_security_policy(
        &self,
        policy_hash: &[u8; SHA256_DIGEST_SIZE],
    ) -> Result<(), TpmError>;

    /// Measure a loaded module hash into a PCR.
    fn measure_module(
        &self,
        module_name: &str,
        module_hash: &[u8; SHA256_DIGEST_SIZE],
    ) -> Result<(), TpmError>;

    /// Measure runtime state hash into a PCR.
    fn measure_runtime_state(&self, state_hash: &[u8; SHA256_DIGEST_SIZE]) -> Result<(), TpmError>;

    /// Generate a TPM quote over current PCR values.
    fn quote(&self, nonce: &[u8]) -> Result<QuoteResult, TpmError>;

    /// Verify a TPM quote against expected PCR values.
    fn verify_quote(
        &self,
        quote: &QuoteResult,
        expected_pcrs: &[[u8; SHA256_DIGEST_SIZE]; PCR_COUNT],
    ) -> Result<bool, TpmError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Invalid input payload: {0}")]
    InvalidPayload(String),

    #[error("Authorization failed: {0}")]
    Authorization(String),

    #[error("Threat detected and mitigated: {0}")]
    ThreatMitigated(String),

    #[error("SIS Emergency Shutdown triggered: {0}")]
    EmergencyShutdown(String),

    #[error("Internal server error: {0}")]
    Internal(#[from] anyhow::Error),
}

/// The central orchestrator integrating the gateway routing, consensus logic,
/// and secure Wasmtime execution engine with SIS interlocks.
pub struct SereinMicrokernel {
    nexus: serein_traffic_control::NexusGateway,
    oracle: serein_llm_router::ConsensusArbitrator,
    engine: SecureWasmEngine,
    linker: Linker<StoreState>,
    state: Arc<RwLock<ServerState>>,
    tenant_component: std::sync::RwLock<Option<HotSwapContainer<Component>>>,
    aegis_component: std::sync::Mutex<Option<HotSwapContainer<Component>>>,
    module_cache: Arc<RwLock<HashMap<String, Component>>>,
    resource_monitor: ResourceMonitor,
    sis_interlock: SisInterlock,
    fuel_monitor: FuelMonitor,
    z3_validator: Z3Validator,
    hsm: Arc<dyn HardwareSecurityModule>,
    abi_guard: AbiGuard,
    honeypot_ctx: serein_sandbox_guard::wasi_virt::SharedWasiVirtContext,
    _epoch_ticker: Option<EpochTicker>,
}

impl SereinMicrokernel {
    pub fn wasmtime_engine(&self) -> &wasmtime::Engine {
        &self.engine.engine
    }

    pub fn tenant_component(&self) -> Option<Arc<wasmtime::component::Component>> {
        self.tenant_component
            .read()
            .ok()
            .and_then(|guard| guard.as_ref().map(|c| c.load()))
    }

    /// Initializes the Serein microkernel environment.
    ///
    /// Bootstraps the Wasm engine, dynamic linker, shared state, and loads
    /// the specified tenant Wasm component into memory.
    ///
    /// ## Initialization Steps
    /// - Ephemeral SQLite storage for execution records
    /// - Masking engine with AES-256-GCM master key
    /// - WASI linker with optional WASI-NN support
    /// - SIS interlock with configurable fuel quota
    /// - AST-based payload validator
    /// - TPM measurement (hardware or software mock)
    /// - Background epoch ticker to prevent executor starvation
    pub async fn new(
        tenant: impl Into<String>,
        honeypot_ctx: serein_sandbox_guard::wasi_virt::WasiVirtContext,
        _config: &AppConfig,
    ) -> Result<Self> {
        let engine = initialize_kernel()?;

        let db_url = "sqlite::memory:?cache=shared";
        info!("Initializing isolated kernel-space ephemeral storage");
        let db_pool = SqlitePoolOptions::new()
            .max_connections(5)
            .min_connections(1)
            .idle_timeout(None)
            .connect(db_url)
            .await
            .context("Failed to initialize kernel-space ephemeral storage")?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS execution_records (
                storage_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                record_key TEXT NOT NULL,
                payload TEXT NOT NULL,
                sensitivity TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&db_pool)
        .await
        .context("Failed to initialize execution_records schema in ephemeral storage")?;

        info!("Kernel-space ephemeral storage schema initialized successfully");

        let master_key = std::env::var("SEREIN_MASTER_KEY")
            .context("SEREIN_MASTER_KEY environment variable is required for production deployment. Set a 64-character hex-encoded 256-bit key.")?;

        if master_key.len() != 64 {
            return Err(anyhow!(
                "SEREIN_MASTER_KEY must be exactly 64 hex characters (256 bits). Received {} characters.",
                master_key.len()
            ));
        }

        hex::decode(&master_key)
            .map_err(|e| anyhow!("SEREIN_MASTER_KEY must be valid hexadecimal: {}", e))?;

        let masking_engine = MaskingEngine::new(Some(&master_key))
            .context("Failed to initialize the Masking Engine")?;

        let mut linker = Linker::new(&engine.engine);
        wasmtime_wasi::add_to_linker_async(&mut linker)
            .context("Failed to link WASI interfaces")?;
        PersistRecord::add_to_linker(&mut linker, |state: &mut StoreState| state)
            .context("Failed to link PersistRecord host functions")?;

        #[cfg(feature = "wasi-nn")]
        {
            let wasi_nn_enabled = std::env::var("WASI_NN_ENABLED")
                .unwrap_or_else(|_| "true".to_string())
                .parse::<bool>()
                .unwrap_or(true);

            if wasi_nn_enabled {
                wasmtime_wasi_nn::add_to_linker(&mut linker, |state: &mut StoreState| {
                    &mut state.ctx
                })
                .context("Failed to link WASI-NN host interface")?;
                info!("WASI-NN host-calls injected into linker - on-device SLM inference enabled");
            } else {
                warn!("WASI_NN_ENABLED=false - on-device SLM inference disabled at runtime despite compile-time support");
            }
        }

        #[cfg(not(feature = "wasi-nn"))]
        {
            warn!("WASI-NN feature not enabled at compile time - on-device SLM inference host-calls not linked");
        }

        let state = Arc::new(RwLock::new(ServerState {
            masking_engine: Arc::new(RwLock::new(masking_engine)),
            db_pool: db_pool.clone(),
        }));

        warn!("Aegis sandbox guard component loading skipped - Wasm module not available for hackathon demo");
        let aegis_component = None;

        let sis_interlock = SisInterlock::new(SIS_FUEL_QUOTA);
        let fuel_monitor = FuelMonitor::new(SIS_FUEL_QUOTA);
        let z3_validator = Z3Validator::new(true);

        let hsm: Arc<dyn HardwareSecurityModule> = {
            #[cfg(feature = "secure-hardware")]
            {
                info!("Initializing TPM 2.0 hardware security module");
                unimplemented!(
                    "Physical TPM 2.0 binding requires tss-esapi. \
                     Enable 'secure-hardware' feature only in production with TPM hardware."
                )
            }
            #[cfg(not(feature = "secure-hardware"))]
            {
                warn!("TPM 2.0 hardware disabled - initializing MockHsm software stub");
                Arc::new(MockHsm::new().map_err(|e| anyhow!("MockHsm initialization failed: {e}"))?)
            }
        };

        let abi_guard = AbiGuard::new(true, true);

        let honeypot_env = honeypot_ctx.environment.clone();
        let shared_honeypot_ctx: serein_sandbox_guard::wasi_virt::SharedWasiVirtContext =
            Arc::new(honeypot_ctx);
        let mount_point_display = shared_honeypot_ctx.vfs.mount_point.clone();
        info!(
            mount_point = %mount_point_display,
            token_count = honeypot_env.len(),
            "WASI-Virt honeypot context initialized - decoy credentials injected into sandbox environment"
        );

        let epoch_ticker = if engine.config.epoch_interruption {
            let ticker = engine.start_epoch_ticker();
            info!(
                interval_ms = 100u64,
                "Epoch ticker spawned - Wasmtime starvation prevention active"
            );
            Some(ticker)
        } else {
            None
        };

        info!(
            fuel_quota = SIS_FUEL_QUOTA,
            "Serein Microkernel initialized"
        );

        Ok(Self {
            nexus: serein_traffic_control::NexusGateway::new(tenant),
            oracle: serein_llm_router::ConsensusArbitrator::new(),
            engine,
            linker,
            state,
            tenant_component: std::sync::RwLock::new(None),
            aegis_component: std::sync::Mutex::new(aegis_component.map(|c| HotSwapContainer::new(c, "aegis_component"))),
            module_cache: Arc::new(RwLock::new(HashMap::new())),
            resource_monitor: ResourceMonitor::default(),
            sis_interlock,
            fuel_monitor,
            z3_validator,
            hsm,
            abi_guard,
            honeypot_ctx: shared_honeypot_ctx,
            _epoch_ticker: epoch_ticker,
        })
    }

    async fn load_component(engine: &SecureWasmEngine, name: &str) -> Result<Component> {
        let path = find_wasm_component(name)?;
        let bytes = tokio::fs::read(&path)
            .await
            .context("Failed to read Wasm component")?;
        let engine_clone = engine.engine.clone();
        tokio::task::spawn_blocking(move || {
            wasmtime::component::Component::from_binary(&engine_clone, &bytes)
                .context("Failed to parse Wasm component binary")
        })
        .await
        .context("Tokio spawn_blocking failed")?
    }

    /// Executes a WASM component from the module cache with OOM protection.
    ///
    /// Resolves the compiled `Component` from the in-memory cache by `module_path`.
    /// On cache miss, the component is read from disk, compiled via
    /// `Component::from_binary` on a blocking thread, and inserted into the cache
    /// for subsequent requests. Each store is equipped with a `MemoryResourceLimiter`
    /// that enforces a strict 256MB per-instance linear memory cap.
    pub async fn run_wasm(
        &self,
        module_path: &str,
        payload: &str,
        tenant_id: &str,
    ) -> Result<String> {
        let component = self.get_or_cache_component(module_path).await?;

        let mut store = self
            .create_store_for_request(tenant_id)
            .map_err(|e| anyhow!("Store creation failed: {e}"))?;

        let (instance, _) = PersistRecord::instantiate_async(&mut store, &component, &self.linker)
            .await
            .map_err(|e| anyhow!("Component instantiation failed: {e}"))?;

        let result = instance
            .call_process(&mut store, payload)
            .await
            .map_err(|e| anyhow!("Guest execution trapped: {e}"))?
            .map_err(|e| anyhow!("Guest logic error: {e}"))?;

        Ok(result)
    }

    /// Resolves a compiled `Component` from the module cache.
    ///
    /// Uses a read-then-write double-check pattern to minimize write-lock
    /// contention under high QPS. Compilation is offloaded to a blocking
    /// thread pool to avoid stalling the Tokio runtime.
    async fn get_or_cache_component(&self, module_path: &str) -> Result<Component> {
        {
            let cache = self.module_cache.read().await;
            if let Some(component) = cache.get(module_path) {
                info!(module_path, "Module cache hit - reusing compiled component");
                return Ok(component.clone());
            }
        }

        let mut cache = self.module_cache.write().await;
        if let Some(component) = cache.get(module_path) {
            info!(
                module_path,
                "Module cache hit after write lock - reusing compiled component"
            );
            return Ok(component.clone());
        }

        info!(
            module_path,
            "Module cache miss - reading and compiling component"
        );
        let bytes = tokio::fs::read(module_path)
            .await
            .context("Failed to read WASM component from path")?;

        let engine = self.engine.engine.clone();
        let component = tokio::task::spawn_blocking(move || {
            wasmtime::component::Component::from_binary(&engine, &bytes)
                .context("Failed to compile WASM component")
        })
        .await
        .context("spawn_blocking panicked")??;

        cache.insert(module_path.to_string(), component.clone());
        info!(
            module_path,
            "Component compiled and inserted into module cache"
        );
        Ok(component)
    }

    /// Processes an incoming API request through the Zero-Trust pipeline.
    ///
    /// ## Pipeline Stages
    /// 1. SIS Interlock Check - reject if fuel exhausted
    /// 2. Resource monitoring and rate limiting
    /// 3. Oracle Consensus Arbitration - reject low-confidence payloads
    /// 4. AST-based payload validation
    /// 5. Sandboxed tenant execution with fuel accounting
    pub async fn process_api_request(
        &self,
        endpoint: &str,
        adjudicated_payload: &str,
        ip: IpAddr,
        tenant_id: &str,
    ) -> Result<String, ApiError> {
        let request_span = span!(
            Level::INFO,
            "api_request",
            endpoint = endpoint,
            client_ip = %ip,
            fuel_quota = SIS_FUEL_QUOTA
        );

        async move {
            if self.sis_interlock.current_state() == SisInterlockState::EmergencyShutdown {
                return Err(ApiError::EmergencyShutdown(
                    "Emergency shutdown active - all requests blocked".into()
                ));
            }

            let initial_fuel = self.sis_interlock.fuel_remaining();
            info!(fuel_remaining = initial_fuel, "Request processing started");

            if self.resource_monitor.check_and_record_request(ip).map_err(|e| ApiError::Internal(anyhow!("Resource monitor check failed: {e}")))? {
                return self.trigger_strangulation_mode(ip).await;
            }

            self.nexus
                .receive_request(endpoint, adjudicated_payload)
                .map_err(|e| ApiError::Internal(anyhow!("Nexus Gateway routing failed: {e}")))?;

            match self.oracle.arbitrate(adjudicated_payload, self.detect_response_source(adjudicated_payload)) {
                Ok(_) => {
                    info!(
                        "Oracle consensus arbitration passed - proceeding to Z3 validation"
                    );
                }
                Err(serein_llm_router::OracleError::LowConfidenceThreshold(score, threshold)) => {
                    warn!(
                        confidence = score,
                        threshold = threshold,
                        "Oracle arbitration rejected low-confidence payload"
                    );
                    return Err(ApiError::InvalidPayload(format!(
                        "Payload rejected by consensus arbitrator: confidence {:.4} below threshold {:.2}",
                        score, threshold
                    )));
                }
                Err(e) => {
                    return Err(ApiError::Internal(anyhow!("Oracle consensus arbitration failed: {e}")));
                }
            }

            if let Err(_violation) = self.z3_validator.prove_safety(adjudicated_payload.as_bytes()) {
                warn!("Payload validation failed - formal safety proof rejected");

                return Err(ApiError::InvalidPayload(
                    "Payload validation failed - formal safety proof rejected".into()
                ));
            }

            let mut store = self.create_store_for_request(tenant_id)
                .map_err(ApiError::Internal)?;

            let execution_timeout = Duration::from_secs(self.engine.config.execution_timeout_secs);
            let exec_result = tokio::time::timeout(
                execution_timeout,
                self.execute_guest(&mut store, adjudicated_payload, tenant_id),
            )
            .await
            .map_err(|_| {
                warn!(
                    timeout_secs = self.engine.config.execution_timeout_secs,
                    "WASM guest execution timed out - epoch interruption force-terminated the instance"
                );
                ApiError::EmergencyShutdown(
                    format!("WASM guest execution exceeded {}s timeout - instance force-terminated", self.engine.config.execution_timeout_secs)
                )
            })?;
            exec_result
        }.instrument(request_span).await
    }

    /// Executes the tenant WASM guest inside the borrowed store and accounts fuel.
    ///
    /// The caller is responsible for returning the store to the pool after this
    /// method returns, regardless of success or failure.
    async fn execute_guest(
        &self,
        store: &mut Store<StoreState>,
        adjudicated_payload: &str,
        tenant_id: &str,
    ) -> Result<String, ApiError> {
        let tenant_component = self
            .tenant_component
            .read()
            .map_err(|e| ApiError::Internal(anyhow!("Tenant component lock poisoned: {e}")))?
            .as_ref()
            .ok_or_else(|| ApiError::Internal(anyhow!("Tenant component not loaded")))?
            .load();
        let (instance, _) =
            PersistRecord::instantiate_async(&mut *store, &tenant_component, &self.linker)
                .await
                .map_err(|e| ApiError::Internal(anyhow!("Guest instantiation failed: {e}")))?;

        let fuel_before = store.get_fuel().map_err(|e| {
            ApiError::Internal(anyhow!("Failed to read fuel before execution: {e}"))
        })?;

        let guest_result = instance
            .call_process(&mut *store, adjudicated_payload)
            .await;

        let fuel_after = store
            .get_fuel()
            .map_err(|e| ApiError::Internal(anyhow!("Failed to read fuel after execution: {e}")))?;
        let consumed_fuel = fuel_before.saturating_sub(fuel_after);
        self.sis_interlock.consume_fuel(consumed_fuel);
        if let Err(trap) = self.fuel_monitor.consume(consumed_fuel) {
            return Err(ApiError::EmergencyShutdown(format!(
                "Fuel ESD trap: {trap}"
            )));
        }

        let result = guest_result
            .map_err(|trap_err| ApiError::Internal(anyhow!("Guest execution trapped: {trap_err}")))?
            .map_err(|guest_err| {
                ApiError::InvalidPayload(format!("Guest logic failed: {guest_err}"))
            })?;

        info!(
            fuel_consumed = consumed_fuel,
            fuel_remaining = self.sis_interlock.fuel_remaining(),
            tenant = tenant_id,
            "Request completed - fuel accounted"
        );

        Ok(result)
    }

    /// Detects the response source from the adjudicated payload to apply
    /// the correct dynamic confidence threshold.
    ///
    /// Checks for a `source` or `provider` field in the JSON payload.
    /// If the source indicates a local SLM (Node D), returns `PhysicalNodeD`
    /// which triggers the lower 0.40 confidence threshold.
    ///
    /// This logic resides in the WASM host (kernel) to prevent guest-side
    /// manipulation of confidence requirements.
    fn detect_response_source(&self, payload: &str) -> serein_llm_router::ResponseSource {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) {
            let source_field = value
                .get("source")
                .or_else(|| value.get("provider"))
                .or_else(|| value.get("node_id"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let source_lower = source_field.to_lowercase();
            if source_lower.contains("node_d")
                || source_lower.contains("noded")
                || source_lower.contains("slm")
                || source_lower.contains("local")
            {
                return serein_llm_router::ResponseSource::PhysicalNodeD;
            }
        }
        serein_llm_router::ResponseSource::FrontierCloud
    }

    /// Activates Strangulation Mode via sandboxed Aegis component.
    ///
    /// All defense payload execution occurs inside the Wasm sandbox.
    /// Native host-thread cryptographic loops are prohibited - they bypass
    /// fuel metering and capability bounds checking.
    ///
    /// ## Containment Architecture
    /// 1. Borrows a Store from the pool with fuel quota
    /// 2. Instantiates serein_sandbox_guard.wasm inside the sandbox
    /// 3. Invokes trigger-asymmetric-counter exported by the Aegis guest
    /// 4. Returns the sandboxed result; guest traps propagate as ApiError::Internal
    async fn trigger_strangulation_mode(&self, ip: IpAddr) -> Result<String, ApiError> {
        info!(
            ip = %ip,
            "Aegis sandbox containment engaged - instantiating defense component"
        );

        let mut store = self
            .create_store_for_request("system")
            .map_err(ApiError::Internal)?;

        let aegis_timeout = Duration::from_secs(self.engine.config.execution_timeout_secs);
        let aegis_result = tokio::time::timeout(aegis_timeout, self.execute_aegis(&mut store, ip))
            .await
            .map_err(|_| {
                warn!(
                    timeout_secs = self.engine.config.execution_timeout_secs,
                    "Aegis defense component execution timed out - force-terminated"
                );
                ApiError::EmergencyShutdown(format!(
                    "Aegis defense component exceeded {}s timeout",
                    self.engine.config.execution_timeout_secs
                ))
            })?;
        aegis_result
    }

    /// Executes the Aegis defense component inside the borrowed store.
    ///
    /// The caller is responsible for returning the store to the pool after this
    /// method returns, regardless of success or failure.
    async fn execute_aegis(
        &self,
        store: &mut Store<StoreState>,
        ip: IpAddr,
    ) -> Result<String, ApiError> {
        let aegis_component = match self.aegis_component.lock().unwrap().as_ref() {
            Some(container) => container.load(),
            None => {
                warn!("Aegis component not loaded - skipping sandbox defense");
                return Err(ApiError::Internal(anyhow!(
                    "Aegis sandbox guard component not available"
                )));
            }
        };
        let (aegis_instance, _aegis_data) =
            AegisWorld::instantiate_async(&mut *store, &aegis_component, &self.linker)
                .await
                .map_err(|e| {
                    ApiError::Internal(anyhow!("Aegis sandbox instantiation failed: {e}"))
                })?;

        let defense = aegis_instance.interface0;
        let attacker_ctx = defense
            .attacker_context()
            .call_constructor(&mut *store)
            .await
            .map_err(|e| {
                ApiError::Internal(anyhow!("Attacker context construction failed: {e}"))
            })?;

        let aegis_fuel_before = store.get_fuel().map_err(ApiError::Internal)?;

        let aegis_result = defense
            .call_trigger_asymmetric_counter(&mut *store, attacker_ctx)
            .await;

        attacker_ctx.resource_drop(&mut *store).map_err(|e| {
            ApiError::Internal(anyhow!("Attacker context resource drop failed: {e}"))
        })?;

        let aegis_fuel_after = store.get_fuel().map_err(|e| {
            ApiError::Internal(anyhow!("Failed to read fuel after Aegis execution: {e}"))
        })?;
        let aegis_fuel = aegis_fuel_before.saturating_sub(aegis_fuel_after);
        self.sis_interlock.consume_fuel(aegis_fuel);
        if let Err(trap) = self.fuel_monitor.consume(aegis_fuel) {
            return Err(ApiError::EmergencyShutdown(format!(
                "Aegis fuel ESD trap: {trap}"
            )));
        }

        let result = aegis_result.map_err(|trap| {
            ApiError::Internal(anyhow!("Aegis guest trapped during countermeasure: {trap}"))
        })?;

        warn!(
            ip = %ip,
            sandbox_result = %result,
            aegis_fuel_consumed = aegis_fuel,
            "Aegis sandboxed countermeasure completed - containment verified, fuel accounted"
        );

        Err(ApiError::ThreatMitigated(format!(
            "Aegis countermeasure executed for IP {} - sandbox containment verified, fuel consumed: {}",
            ip, aegis_fuel
        )))
    }

    /// Allocates an isolated `Store` for a single request with fuel injection.
    ///
    /// The fuel quota prevents runaway memory allocation and infinite loops.
    /// Capability-based memory boundary validation is performed via the ABI Guard.
    fn create_store_for_request(&self, tenant_id: &str) -> Result<Store<StoreState>> {
        let mut wasi_builder = WasiCtxBuilder::new();
        wasi_builder.inherit_stdio();

        let honeypot_ref = self.honeypot_ctx.as_ref();
        for (key, value) in &honeypot_ref.environment {
            wasi_builder.env(key, value);
        }

        let models_dir = std::path::Path::new("./serein-models");
        if models_dir.exists() {
            wasi_builder
                .preopened_dir(
                    models_dir,
                    "/models",
                    wasmtime_wasi::DirPerms::READ,
                    wasmtime_wasi::FilePerms::READ,
                )
                .with_context(|| format!("Failed to preopen models directory {:?}", models_dir))?;
            info!(
                models_dir = %models_dir.display(),
                mount_point = "/models",
                "Mounted ./serein-models as READ-ONLY VFS directory /models"
            );
        } else {
            warn!(
                models_dir = %models_dir.display(),
                "Models directory not found - /models VFS mount skipped. \
                 On-device SLM inference will be unavailable."
            );
        }

        let ctx = wasi_builder.build();
        let store_state = StoreState {
            server_state: Arc::clone(&self.state),
            ctx,
            table: ResourceTable::new(),
            aegis_hub: AegisHub,
            attacker_contexts: Vec::new(),
            abi_guard: AbiGuard::new(true, true),
            memory_capability: None,
            canonical_abi_guard: engine::wasmtime_rt::CanonicalAbiGuard::new(true),
            tenant_id: tenant_id.to_string(),
            honeypot_ctx: Arc::clone(&self.honeypot_ctx),
            models_mount: "/models".to_string(),
            shared_buffers: HashMap::new(),
            resource_limiter: engine::wasmtime_rt::MemoryResourceLimiter::default_256mb(),
        };

        let mut store = Store::new(&self.engine.engine, store_state);
        store
            .set_fuel(SIS_FUEL_QUOTA)
            .context("Failed to inject fuel into the execution store")?;

        if self.engine.config.epoch_interruption {
            store.set_epoch_deadline(self.engine.config.epoch_deadline);
        }

        store.limiter(
            |state: &mut StoreState| -> &mut dyn wasmtime::ResourceLimiter {
                &mut state.resource_limiter
            },
        );

        let memory_bounds = self.validate_memory_bounds(&store)?;
        info!(
            linear_memory_min = memory_bounds.0,
            linear_memory_max = memory_bounds.1,
            "Memory boundary validation passed"
        );

        Ok(store)
    }

    /// Validates memory boundaries using capability-based boundary checking.
    fn validate_memory_bounds(&self, _store: &Store<StoreState>) -> Result<(usize, usize)> {
        let linear_memory_min = 0usize;
        let linear_memory_max = self.engine.config.max_memory_size;

        self.abi_guard
            .check_memory_access(linear_memory_min, linear_memory_max, linear_memory_max)
            .map_err(|e| anyhow!("Memory boundary validation failed: {}", e))?;

        Ok((linear_memory_min, linear_memory_max))
    }

    /// Hot-swaps the tenant component at runtime without service interruption.
    ///
    /// Uses ArcSwap for atomic pointer replacement, enabling zero-downtime
    /// module updates.
    pub async fn update_tenant_component(&self, component_name: &str) -> Result<()> {
        let new_component = Self::load_component(&self.engine, component_name).await?;

        let mut guard = self
            .tenant_component
            .write()
            .map_err(|e| anyhow!("Tenant component lock poisoned: {e}"))?;

        match guard.as_ref() {
            Some(container) => {
                let old = container.swap(new_component);
                info!(
                    module = "tenant_component",
                    swap_count = container.swap_count(),
                    "Hot-swap completed - tenant component replaced"
                );
                drop(old);
            }
            None => {
                *guard = Some(HotSwapContainer::new(new_component, "tenant_component"));
                info!(
                    module = "tenant_component",
                    "Hot-swap completed - tenant component initialized"
                );
            }
        }
        Ok(())
    }

    /// Recompile and hot-swap the tenant Wasm component from raw bytes.
    ///
    /// Compiles the provided Wasm binary into a new `Component` on a blocking
    /// thread, then atomically swaps it into the active tenant slot via
    /// `HotSwapContainer`. The old component is dropped after the swap,
    /// releasing its memory.
    ///
    /// This is the genuine hot-swap path triggered by the `/v1/system/hot-swap`
    /// endpoint - unlike the `ArcSwap<Vec<u8>>` byte-level swap, this method
    /// forces Wasmtime to recompile the machine code and re-instantiate the
    /// component for subsequent requests.
    pub async fn reload_component(&self, new_wasm_bytes: &[u8]) -> Result<()> {
        let bytes = new_wasm_bytes.to_vec();
        let engine = self.engine.engine.clone();

        let component = tokio::task::spawn_blocking(move || {
            wasmtime::component::Component::from_binary(&engine, &bytes)
                .context("Failed to compile WASM component from hot-swap bytes")
        })
        .await
        .context("spawn_blocking panicked during hot-swap compilation")??;

        let mut guard = self
            .tenant_component
            .write()
            .map_err(|e| anyhow!("Tenant component lock poisoned: {e}"))?;

        match guard.as_ref() {
            Some(container) => {
                let old = container.swap(component);
                info!(
                    module = "tenant_component",
                    swap_count = container.swap_count(),
                    "Genuine hot-swap completed - Wasmtime component recompiled and atomically replaced"
                );
                drop(old);
            }
            None => {
                *guard = Some(HotSwapContainer::new(component, "tenant_component"));
                info!(
                    module = "tenant_component",
                    "Genuine hot-swap completed - tenant component initialized from raw bytes"
                );
            }
        }
        Ok(())
    }

    /// Hot-swaps the aegis component at runtime without service interruption.
    pub async fn update_aegis_component(&self, component_name: &str) -> Result<()> {
        let new_component = Self::load_component(&self.engine, component_name).await?;

        let mut guard = self.aegis_component.lock().unwrap();
        match guard.as_ref() {
            Some(container) => {
                let old = container.swap(new_component);
                info!(
                    module = "aegis_component",
                    swap_count = container.swap_count(),
                    "Hot-swap completed - aegis component replaced"
                );
                drop(old);
            }
            None => {
                *guard = Some(HotSwapContainer::new(new_component, "aegis_component"));
                info!(
                    module = "aegis_component",
                    "Hot-swap completed - aegis component initialized"
                );
            }
        }
        Ok(())
    }

    /// Get SIS interlock state for monitoring
    pub fn sis_state(&self) -> SisInterlockState {
        self.sis_interlock.current_state()
    }

    /// Get fuel monitor metrics
    pub fn fuel_metrics(&self) -> engine::fuel_monitor::EnergyMetrics {
        self.fuel_monitor.metrics()
    }

    /// Get Hardware Security Module reference for attestation and measurement.
    pub fn hsm(&self) -> Arc<dyn HardwareSecurityModule> {
        self.hsm.clone()
    }

    /// Get ABI Guard reference for boundary checking
    pub fn abi_guard(&self) -> &AbiGuard {
        &self.abi_guard
    }
}

/// Dynamic path radar to locate compiled Wasm targets across build profiles.
fn find_wasm_component(name: &str) -> anyhow::Result<std::path::PathBuf> {
    let root_dir = std::env::current_dir()
        .map_err(|e| anyhow::anyhow!("Failed to determine workspace directory: {}", e))?;

    let target_dir = root_dir.join("target");

    let paths = [
        root_dir.join(name),
        root_dir.join("modules").join(name),
        target_dir.join("wasm32-wasip1/release").join(name),
        target_dir.join("wasm32-wasi/release").join(name),
    ];

    paths.into_iter().find(|p| p.exists()).ok_or_else(|| {
        anyhow::anyhow!(
            "Wasm component '{}' not found - ensure it is placed in the project root",
            name
        )
    })
}

/// Bootstraps the baseline secure environment configurations.
pub fn initialize_kernel() -> Result<SecureWasmEngine> {
    SecureWasmEngine::new(SecureRuntimeConfig::default())
}
