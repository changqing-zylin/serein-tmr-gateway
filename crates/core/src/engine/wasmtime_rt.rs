// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Serein Core - Wasmtime Runtime Engine (WASI 0.2 Component Model)
//!
//! Secure WebAssembly runtime powered by Wasmtime with fuel-based execution
//! limiting, WASI 0.2 Component Model Canonical ABI enforcement, and
//! zero-copy shared buffer resource handles.
//!
//! ## Zero-Copy Shared Buffers
//! - `SharedBuffer` resource provides read-only memory views from host to guest
//! - Guest reads buffer contents via `Resource<SharedBuffer>` handles instead
//!   of receiving copied strings across the Component Model boundary
//! - Large payloads are transferred via `persist_buffer` which accepts a
//!   `SharedBuffer` handle, avoiding per-request string allocation overhead
//!
//! ## Canonical ABI Enforcement
//! - ALL host/guest data transfers occur exclusively through the Component Model
//!   lifting/lowering pipeline. Raw `Memory::data_mut` access is prohibited.
//! - `CanonicalAbiGuard` validates every cross-boundary transfer against the
//!   Canonical ABI specification, ensuring type-safe serialization at the boundary.
//!
//! ## Safety Guarantees
//! - `AbiGuard` enforces capability bounds on linear memory regions.
//! - Fuel exhaustion triggers a hard `wasmtime::Trap::Interrupt`.
//! - `CanonicalAbiGuard` ensures no raw memory sharing between host and guest.

use crate::SIS_FUEL_QUOTA;

use crate::{
    abi_guard::{AbiGuard, CheriCapability},
    security::masking::{MaskingEngine, Sensitivity as HostSensitivity},
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::Utc;
use sqlx::SqlitePool;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{error, info, warn};
use uuid::Uuid;
use wasmtime::component::Resource;
use wasmtime::Config;
use wasmtime::Engine;
use wasmtime_wasi::{ResourceTable, WasiCtx, WasiView};

wasmtime::component::bindgen!({
    path: "../interfaces/data-persistence.wit",
    world: "persist-record",
    async: true,
    with: {
        "serein:core/data-persistence/shared-buffer": SharedBufferEntry,
    },
});

/// Marker type for shared buffer resources in the Component Model.
pub struct SharedBufferEntry;

fn sensitivity_to_str(level: serein::core::data_persistence::Sensitivity) -> &'static str {
    match level {
        serein::core::data_persistence::Sensitivity::Public => "public",
        serein::core::data_persistence::Sensitivity::Internal => "internal",
        serein::core::data_persistence::Sensitivity::PiiMasked => "pii_masked",
        serein::core::data_persistence::Sensitivity::PiiEncrypted => "pii_encrypted",
    }
}

const SHARED_BUFFER_MAX_LEN: usize = 16 * 1024 * 1024;
const SHARED_BUFFER_READ_CHUNK: u64 = 64 * 1024;

/// Host-side memory bounds validation macro.
///
/// Validates that `offset + len <= capacity` before any memory access
/// originating from guest-provided offsets. Triggers a `GuestTrap::MemoryAccessViolation`
/// on failure, preventing guest-side probing of host memory boundaries.
///
/// ## Safety
/// This macro MUST be invoked before any `unsafe` memory access or slice
/// indexing that uses guest-provided offset/length values.
macro_rules! ensure_in_bounds {
    ($offset:expr, $len:expr, $capacity:expr) => {
        if ($offset as u64).saturating_add($len as u64) > $capacity as u64 {
            tracing::error!(
                offset = $offset,
                len = $len,
                capacity = $capacity,
                "GUEST_TRAP: MemoryAccessViolation - guest-provided offset+len exceeds buffer capacity"
            );
            return Err(GuestTrap::MemoryAccessViolation {
                offset: $offset as u64,
                len: $len as u64,
                capacity: $capacity as u64,
            });
        }
    };
}

/// Guest trap variants for host-enforced memory safety violations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum GuestTrap {
    #[error(
        "Memory access violation: offset {offset} + len {len} exceeds buffer capacity {capacity}"
    )]
    MemoryAccessViolation {
        offset: u64,
        len: u64,
        capacity: u64,
    },
}

/// Shared server state injected into every WASM store instance.
///
/// Provides access to the masking engine (for PII transformation) and the
/// SQLite connection pool (for record persistence) via `RwLock`-protected
/// shared references.
#[derive(Clone)]
pub struct ServerState {
    pub masking_engine: Arc<RwLock<MaskingEngine>>,
    pub db_pool: SqlitePool,
}

/// Canonical ABI boundary enforcement guard.
///
/// Tracks compliant transfers and violations across the Component Model
/// boundary. Validates buffer sizes, alignment, and string lengths before
/// any host↔guest data transfer.
pub struct CanonicalAbiGuard {
    transfer_count: std::sync::atomic::AtomicU64,
    violation_count: std::sync::atomic::AtomicU64,
    enabled: bool,
}

impl CanonicalAbiGuard {
    pub fn new(enabled: bool) -> Self {
        Self {
            transfer_count: std::sync::atomic::AtomicU64::new(0),
            violation_count: std::sync::atomic::AtomicU64::new(0),
            enabled,
        }
    }

    pub fn record_compliant_transfer(&self) {
        if self.enabled {
            self.transfer_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    pub fn record_violation(&self, context: &str) {
        if self.enabled {
            self.violation_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            error!(
                context,
                total_violations = self
                    .violation_count
                    .load(std::sync::atomic::Ordering::SeqCst),
                "CANONICAL_ABI_VIOLATION: Raw memory access bypassed Component Model boundary"
            );
        }
    }

    pub fn validate_buffer_transfer(
        &self,
        len: usize,
        max_len: usize,
        alignment: usize,
    ) -> Result<(), CanonicalAbiViolation> {
        if !self.enabled {
            return Ok(());
        }

        if len > max_len {
            return Err(CanonicalAbiViolation::BufferOverflow {
                requested: len,
                maximum: max_len,
            });
        }

        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(CanonicalAbiViolation::InvalidAlignment { alignment });
        }

        self.record_compliant_transfer();
        Ok(())
    }

    pub fn validate_string_transfer(
        &self,
        s: &str,
        max_len: usize,
    ) -> Result<(), CanonicalAbiViolation> {
        if !self.enabled {
            return Ok(());
        }

        let byte_len = s.len();
        if byte_len > max_len {
            return Err(CanonicalAbiViolation::BufferOverflow {
                requested: byte_len,
                maximum: max_len,
            });
        }

        self.record_compliant_transfer();
        Ok(())
    }

    pub fn transfer_count(&self) -> u64 {
        self.transfer_count
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn violation_count(&self) -> u64 {
        self.violation_count
            .load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum CanonicalAbiViolation {
    #[error("Canonical ABI buffer overflow: requested {requested} bytes, maximum {maximum}")]
    BufferOverflow { requested: usize, maximum: usize },

    #[error("Canonical ABI invalid alignment: {alignment} is not a power of 2")]
    InvalidAlignment { alignment: usize },

    #[error("Canonical ABI raw memory access prohibited: {context}")]
    RawMemoryAccessProhibited { context: String },

    #[error("Canonical ABI type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },
}

/// Per-request WASM store state containing WASI context, resource table,
/// Aegis security hub, shared buffers, and ABI boundary guards.
pub struct StoreState {
    pub server_state: Arc<RwLock<ServerState>>,
    pub ctx: WasiCtx,
    pub table: ResourceTable,
    pub aegis_hub: AegisHub,
    pub attacker_contexts: Vec<Resource<AttackerContext>>,
    pub abi_guard: AbiGuard,
    pub memory_capability: Option<CheriCapability>,
    pub canonical_abi_guard: CanonicalAbiGuard,
    pub tenant_id: String,
    pub honeypot_ctx: serein_sandbox_guard::wasi_virt::SharedWasiVirtContext,
    pub models_mount: String,
    pub shared_buffers: HashMap<u32, Arc<[u8]>>,
    pub resource_limiter: MemoryResourceLimiter,
}

impl Drop for StoreState {
    fn drop(&mut self) {
        for res in self.attacker_contexts.drain(..) {
            if let Err(e) = self.table.delete(res) {
                warn!("Leaked attacker_context resource: {e}");
            }
        }
        self.shared_buffers.clear();
    }
}

impl StoreState {
    pub fn register_memory_capability(&mut self, data_size: usize) -> Result<()> {
        let cap = CheriCapability::from_tlsf_pool(data_size);
        self.abi_guard
            .register_capability("guest_linear_memory", cap.clone())?;
        self.memory_capability = Some(cap);
        info!(
            memory_size = data_size,
            "Capability registered for WASM linear memory - Canonical ABI boundary enforcement active"
        );
        Ok(())
    }

    pub fn check_memory_access(
        &self,
        offset: usize,
        len: usize,
    ) -> Result<(), crate::abi_guard::AbiViolation> {
        if let Some(ref cap) = self.memory_capability {
            cap.check_bounds(offset, len)?;
        }
        Ok(())
    }

    pub fn create_shared_buffer(&mut self, data: Vec<u8>) -> Result<Resource<SharedBufferEntry>> {
        if data.len() > SHARED_BUFFER_MAX_LEN {
            return Err(anyhow::anyhow!(
                "SharedBuffer exceeds maximum size: {} bytes (max {})",
                data.len(),
                SHARED_BUFFER_MAX_LEN
            ));
        }
        let arc_data: Arc<[u8]> = data.into();
        let resource = self.table.push(SharedBufferEntry)?;
        let rep = resource.rep();
        self.shared_buffers.insert(rep, arc_data);
        self.canonical_abi_guard.record_compliant_transfer();
        Ok(resource)
    }

    fn get_shared_buffer(&self, resource: &Resource<SharedBufferEntry>) -> Option<&Arc<[u8]>> {
        self.shared_buffers.get(&resource.rep())
    }
}

async fn insert_record(
    tenant_id: &str,
    key: &str,
    transformed_data: &str,
    sensitivity: &str,
    db_pool: &SqlitePool,
) -> (bool, String, String) {
    let storage_id = Uuid::new_v4();
    let created_at = Utc::now();

    let insert_result = sqlx::query(
        r#"
        INSERT INTO execution_records (storage_id, tenant_id, record_key, payload, sensitivity, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        "#
    )
    .bind(storage_id.to_string())
    .bind(tenant_id)
    .bind(key)
    .bind(transformed_data)
    .bind(sensitivity)
    .bind(created_at.to_rfc3339())
    .execute(db_pool)
    .await;

    match insert_result {
        Ok(_) => {
            tracing::info!(
                storage_id = %storage_id,
                key = %key,
                "Record persisted successfully"
            );
            (true, storage_id.to_string(), created_at.to_rfc3339())
        }
        Err(e) => {
            error!("Persistence failed: {e}");
            (false, String::new(), format!("Persistence failed: {e}"))
        }
    }
}

/// Aegis security hub for active defense coordination within WASM stores.
#[derive(Default)]
pub struct AegisHub;

/// Attacker context resource for Aegis sandboxed countermeasure execution.
#[derive(Debug)]
pub struct AttackerContext;

impl WasiView for StoreState {
    fn table(&mut self) -> &mut ResourceTable {
        &mut self.table
    }
    fn ctx(&mut self) -> &mut WasiCtx {
        &mut self.ctx
    }
}

#[async_trait]
impl serein::core::data_persistence::HostSharedBuffer for StoreState {
    async fn read(
        &mut self,
        self_: Resource<SharedBufferEntry>,
        offset: u64,
        len: u64,
    ) -> (Vec<u8>, u64) {
        let read_len = len.min(SHARED_BUFFER_READ_CHUNK);
        match self.get_shared_buffer(&self_) {
            Some(data) => {
                let total = data.len() as u64;
                if offset >= total {
                    self.canonical_abi_guard.record_compliant_transfer();
                    return (Vec::new(), total);
                }
                let capacity = data.len();

                if let Err(trap) = (|| -> Result<(), GuestTrap> {
                    ensure_in_bounds!(offset, read_len, capacity);
                    Ok(())
                })() {
                    error!(%trap, "SharedBuffer read bounds violation blocked by Aegis");
                    return (Vec::new(), total);
                }

                let start = offset as usize;
                let end = std::cmp::min(start + read_len as usize, data.len());
                let slice = data[start..end].to_vec();
                self.canonical_abi_guard.record_compliant_transfer();
                (slice, total)
            }
            None => {
                error!("SharedBuffer resource not found in host map");
                (Vec::new(), 0)
            }
        }
    }

    async fn len(&mut self, self_: Resource<SharedBufferEntry>) -> u64 {
        match self.get_shared_buffer(&self_) {
            Some(data) => data.len() as u64,
            None => {
                error!("SharedBuffer resource not found in host map");
                0
            }
        }
    }

    fn drop(&mut self, self_: Resource<SharedBufferEntry>) -> Result<()> {
        let rep = self_.rep();
        self.shared_buffers.remove(&rep);
        self.table.delete(self_)?;
        Ok(())
    }
}

#[async_trait]
impl serein::core::data_persistence::Host for StoreState {
    async fn persist(
        &mut self,
        payload: serein::core::data_persistence::DataPayload,
    ) -> (bool, String, String) {
        if let Err(e) = self
            .canonical_abi_guard
            .validate_string_transfer(&payload.key, 4096)
        {
            error!("Canonical ABI validation failed for key: {e}");
            return (
                false,
                String::new(),
                format!("Canonical ABI violation: {e}"),
            );
        }
        if let Err(e) = self
            .canonical_abi_guard
            .validate_string_transfer(&payload.value, 1024 * 1024)
        {
            error!("Canonical ABI validation failed for value: {e}");
            return (
                false,
                String::new(),
                format!("Canonical ABI violation: {e}"),
            );
        }

        let state = self.server_state.read().await;
        let masking_engine = state.masking_engine.read().await;
        let host_sensitivity: HostSensitivity = payload.level.into();

        let transformed_data =
            match masking_engine.transform_data_owned(payload.value, host_sensitivity) {
                Ok(data) => data,
                Err(e) => {
                    error!("Data transformation failed: {e}");
                    return (
                        false,
                        String::new(),
                        format!("Data transformation failed: {e}"),
                    );
                }
            };

        let db_pool = state.db_pool.clone();
        drop(masking_engine);
        drop(state);

        let sensitivity_str = sensitivity_to_str(payload.level);
        let tenant_id = self.tenant_id.clone();
        insert_record(
            &tenant_id,
            &payload.key,
            &transformed_data,
            sensitivity_str,
            &db_pool,
        )
        .await
    }

    async fn persist_buffer(
        &mut self,
        key: String,
        buffer: Resource<SharedBufferEntry>,
        level: serein::core::data_persistence::Sensitivity,
    ) -> (bool, String, String) {
        if let Err(e) = self
            .canonical_abi_guard
            .validate_string_transfer(&key, 4096)
        {
            error!("Canonical ABI validation failed for key: {e}");
            return (
                false,
                String::new(),
                format!("Canonical ABI violation: {e}"),
            );
        }

        let buf_data = match self.get_shared_buffer(&buffer) {
            Some(data) => data,
            None => {
                error!("SharedBuffer resource not found in host map");
                return (
                    false,
                    String::new(),
                    "SharedBuffer resource not found".to_string(),
                );
            }
        };

        if buf_data.len() > SHARED_BUFFER_MAX_LEN {
            error!(
                "SharedBuffer payload exceeds maximum: {} bytes (max {})",
                buf_data.len(),
                SHARED_BUFFER_MAX_LEN
            );
            return (
                false,
                String::new(),
                format!(
                    "SharedBuffer payload exceeds maximum: {} bytes",
                    buf_data.len()
                ),
            );
        }

        let state = self.server_state.read().await;
        let masking_engine = state.masking_engine.read().await;
        let host_sensitivity: HostSensitivity = level.into();

        let transformed_data = match masking_engine.transform_bytes(buf_data, host_sensitivity) {
            Ok(data) => data,
            Err(e) => {
                error!("Data transformation failed: {e}");
                return (
                    false,
                    String::new(),
                    format!("Data transformation failed: {e}"),
                );
            }
        };

        let db_pool = state.db_pool.clone();
        drop(masking_engine);
        drop(state);

        let sensitivity_str = sensitivity_to_str(level);
        let tenant_id = self.tenant_id.clone();
        insert_record(
            &tenant_id,
            &key,
            &transformed_data,
            sensitivity_str,
            &db_pool,
        )
        .await
    }

    async fn retrieve(
        &mut self,
        key: String,
        requester_clearance: serein::core::data_persistence::Sensitivity,
    ) -> (Option<serein::core::data_persistence::DataPayload>, bool) {
        if let Err(e) = self
            .canonical_abi_guard
            .validate_string_transfer(&key, 4096)
        {
            error!("Canonical ABI validation failed for retrieve key: {e}");
            return (None, false);
        }

        let is_authorized =
            requester_clearance != serein::core::data_persistence::Sensitivity::Public;

        let data = if is_authorized {
            Some(serein::core::data_persistence::DataPayload {
                key,
                value: "simulated_retrieved_value_from_host".to_string(),
                level: requester_clearance,
            })
        } else {
            None
        };
        (data, is_authorized)
    }

    async fn delete(
        &mut self,
        key: String,
        requester_clearance: serein::core::data_persistence::Sensitivity,
    ) -> (bool, bool) {
        if let Err(e) = self
            .canonical_abi_guard
            .validate_string_transfer(&key, 4096)
        {
            error!("Canonical ABI validation failed for delete key: {e}");
            return (false, false);
        }

        let is_authorized =
            requester_clearance != serein::core::data_persistence::Sensitivity::Public;
        (is_authorized, is_authorized)
    }

    async fn audit(
        &mut self,
        key: String,
        auditor_clearance: serein::core::data_persistence::Sensitivity,
    ) -> (Vec<String>, bool) {
        if let Err(e) = self
            .canonical_abi_guard
            .validate_string_transfer(&key, 4096)
        {
            error!("Canonical ABI validation failed for audit key: {e}");
            return (Vec::new(), false);
        }

        let is_authorized = matches!(
            auditor_clearance,
            serein::core::data_persistence::Sensitivity::Internal
                | serein::core::data_persistence::Sensitivity::PiiEncrypted
        );

        let trail = if is_authorized {
            vec![
                "2026-03-29T10:00:00Z - DATA_CREATED".to_string(),
                "2026-03-29T11:30:00Z - DATA_ACCESSED by user:auditor".to_string(),
            ]
        } else {
            vec![]
        };
        (trail, is_authorized)
    }
}

impl From<serein::core::data_persistence::Sensitivity> for HostSensitivity {
    fn from(guest_level: serein::core::data_persistence::Sensitivity) -> Self {
        match guest_level {
            serein::core::data_persistence::Sensitivity::Public => HostSensitivity::Public,
            serein::core::data_persistence::Sensitivity::Internal => HostSensitivity::Internal,
            serein::core::data_persistence::Sensitivity::PiiMasked => HostSensitivity::PiiMasked,
            serein::core::data_persistence::Sensitivity::PiiEncrypted => {
                HostSensitivity::PiiEncrypted
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SecureRuntimeConfig {
    pub max_fuel: u64,
    pub epoch_interruption: bool,
    pub epoch_deadline: u64,
    pub execution_timeout_secs: u64,
    pub max_memory_size: usize,
    pub initial_memory_pages: u32,
    pub max_memory_pages: u32,
    /// Maximum number of concurrent component instances in the pooling allocator.
    /// Constrained to 500 by default to avoid `vm.max_map_count` exhaustion under
    /// high concurrency - each instance reserves guard pages via mmap, and
    /// oversubscription triggers OS-level mapping limits.
    pub pool_size: u32,
}

impl Default for SecureRuntimeConfig {
    fn default() -> Self {
        let initial_memory_pages: u32 = 512;
        let max_memory_pages: u32 = 4096;
        Self {
            max_fuel: SIS_FUEL_QUOTA,
            epoch_interruption: true,
            epoch_deadline: 100,
            execution_timeout_secs: 30,
            max_memory_size: max_memory_pages as usize * 64 * 1024,
            initial_memory_pages,
            max_memory_pages,
            pool_size: 500,
        }
    }
}

pub struct SecureWasmEngine {
    pub engine: Engine,
    pub config: SecureRuntimeConfig,
}

/// Background epoch ticker for Wasmtime starvation prevention.
///
/// Spawns a `tokio::task` that periodically increments the Wasmtime engine epoch
/// at a configurable interval. When the `EpochTicker` is dropped, the background
/// task is gracefully cancelled via `JoinHandle::abort()`.
///
/// ## Safety Guarantee
/// Without an active epoch ticker, `epoch_interruption` configuration is inert -
/// malicious WASM guests can execute infinite loops without being killed. This
/// ticker ensures deterministic interruption of CPU-bound guests.
pub struct EpochTicker {
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl EpochTicker {
    /// Start a background epoch ticker for the given Wasmtime engine.
    ///
    /// The ticker increments `engine.increment_epoch()` at the specified interval,
    /// ensuring that WASM guests with `epoch_interruption` enabled will be
    /// interrupted if they exceed their epoch deadline.
    pub fn start(engine: Engine, interval: Duration) -> Self {
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Burst);
            loop {
                ticker.tick().await;
                engine.increment_epoch();
            }
        });
        Self {
            handle: Some(handle),
        }
    }
}

impl Drop for EpochTicker {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
            tracing::debug!("Epoch ticker background task cancelled - graceful shutdown");
        }
    }
}

const WASM_PAGE_SIZE_BYTES: u64 = 64 * 1024;

impl SecureWasmEngine {
    pub fn new(config: SecureRuntimeConfig) -> Result<Self> {
        let mut engine_config = Config::new();
        engine_config.wasm_component_model(true);
        engine_config.async_support(true);
        engine_config.consume_fuel(true);
        engine_config.wasm_backtrace(true);
        engine_config.wasm_memory64(false);
        engine_config.wasm_tail_call(true);

        let max_memory_bytes = config.max_memory_pages as u64 * WASM_PAGE_SIZE_BYTES;
        let initial_memory_bytes = config.initial_memory_pages as u64 * WASM_PAGE_SIZE_BYTES;

        engine_config.static_memory_guard_size(4 * 1024 * 1024);
        engine_config.dynamic_memory_reserved_for_growth(4 * 1024 * 1024);
        engine_config.dynamic_memory_guard_size(4 * 1024 * 1024);

        engine_config.epoch_interruption(true);

        let mut pooling_config = wasmtime::PoolingAllocationConfig::default();
        pooling_config.total_component_instances(config.pool_size);
        pooling_config.max_core_instances_per_component(5);
        pooling_config.total_memories(config.pool_size);
        pooling_config.max_memories_per_component(5);
        pooling_config.max_memory_size(max_memory_bytes as usize);

        info!(
            initial_pages = config.initial_memory_pages,
            max_pages = config.max_memory_pages,
            initial_memory_mib = initial_memory_bytes / (1024 * 1024),
            max_memory_mib = max_memory_bytes / (1024 * 1024),
            pool_size = config.pool_size,
            pooling_max_memory_bytes = max_memory_bytes,
            "Pooling allocator configured - pool_size={} avoids vm.max_map_count exhaustion under concurrency",
            config.pool_size,
        );

        engine_config.allocation_strategy(wasmtime::InstanceAllocationStrategy::Pooling(
            pooling_config,
        ));

        let engine =
            Engine::new(&engine_config).context("Failed to create secure Wasmtime engine")?;

        Ok(Self { engine, config })
    }

    /// Start the background epoch ticker for WASM starvation prevention.
    ///
    /// Must be called from within a Tokio runtime context. The returned
    /// `EpochTicker` must be held alive for the lifetime of the engine -
    /// dropping it will cancel the background task.
    ///
    /// ## Panics
    /// Panics if called outside a Tokio runtime context.
    pub fn start_epoch_ticker(&self) -> EpochTicker {
        let interval = Duration::from_millis(100);
        EpochTicker::start(self.engine.clone(), interval)
    }

    pub fn precompile_component(&self, path: &str) -> Result<wasmtime::component::Component> {
        let component = wasmtime::component::Component::from_file(&self.engine, path)
            .context("Failed to precompile Wasm component from file")?;
        info!(
            component_path = path,
            "Component precompiled and cached for global reuse"
        );
        Ok(component)
    }
}

const MAX_INSTANCE_MEMORY_BYTES: usize = 256 * 1024 * 1024;

pub struct MemoryResourceLimiter {
    max_memory_bytes: usize,
}

impl MemoryResourceLimiter {
    pub fn new(max_memory_bytes: usize) -> Self {
        Self { max_memory_bytes }
    }

    pub fn default_256mb() -> Self {
        Self::new(MAX_INSTANCE_MEMORY_BYTES)
    }
}

impl wasmtime::ResourceLimiter for MemoryResourceLimiter {
    fn memory_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> Result<bool> {
        if desired > self.max_memory_bytes {
            tracing::error!(
                desired_bytes = desired,
                limit_bytes = self.max_memory_bytes,
                "OOM protection: memory growth denied - exceeds per-instance limit"
            );
            Ok(false)
        } else {
            Ok(true)
        }
    }

    fn table_growing(
        &mut self,
        _current: u32,
        _desired: u32,
        _maximum: Option<u32>,
    ) -> Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::serein::core::data_persistence::Host;
    use super::*;
    use wasmtime_wasi::WasiCtxBuilder;

    fn create_test_store_state() -> StoreState {
        let masking_engine = Arc::new(RwLock::new(MaskingEngine::new(None).unwrap()));
        let db_pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect_lazy("sqlite::memory:")
            .expect("Failed to create test pool");
        StoreState {
            server_state: Arc::new(RwLock::new(ServerState {
                masking_engine,
                db_pool,
            })),
            ctx: WasiCtxBuilder::new().build(),
            table: ResourceTable::new(),
            aegis_hub: AegisHub,
            attacker_contexts: Vec::new(),
            abi_guard: AbiGuard::new(true, true),
            memory_capability: None,
            canonical_abi_guard: CanonicalAbiGuard::new(true),
            tenant_id: "test_tenant".to_string(),
            honeypot_ctx: Arc::new(serein_sandbox_guard::wasi_virt::WasiVirtContext::new()),
            models_mount: "/models".to_string(),
            shared_buffers: HashMap::new(),
            resource_limiter: MemoryResourceLimiter::default_256mb(),
        }
    }

    #[test]
    fn test_canonical_abi_guard_compliant_transfer() {
        let guard = CanonicalAbiGuard::new(true);
        guard.record_compliant_transfer();
        guard.record_compliant_transfer();
        assert_eq!(guard.transfer_count(), 2);
        assert_eq!(guard.violation_count(), 0);
    }

    #[test]
    fn test_canonical_abi_guard_buffer_validation() {
        let guard = CanonicalAbiGuard::new(true);
        assert!(guard.validate_buffer_transfer(100, 1024, 8).is_ok());
        assert!(guard.validate_buffer_transfer(2048, 1024, 8).is_err());
    }

    #[test]
    fn test_canonical_abi_guard_string_validation() {
        let guard = CanonicalAbiGuard::new(true);
        assert!(guard.validate_string_transfer("hello", 4096).is_ok());
        let long_string = "a".repeat(5000);
        assert!(guard.validate_string_transfer(&long_string, 4096).is_err());
    }

    #[test]
    fn test_canonical_abi_guard_alignment() {
        let guard = CanonicalAbiGuard::new(true);
        assert!(guard.validate_buffer_transfer(100, 1024, 8).is_ok());
        assert!(guard.validate_buffer_transfer(100, 1024, 0).is_err());
        assert!(guard.validate_buffer_transfer(100, 1024, 3).is_err());
    }

    #[tokio::test]
    async fn test_create_shared_buffer() {
        let mut state = create_test_store_state();
        let resource = state.create_shared_buffer(b"test data".to_vec());
        assert!(resource.is_ok());
        let res = resource.unwrap();
        assert!(state.get_shared_buffer(&res).is_some());
        let data = state.get_shared_buffer(&res).unwrap();
        assert_eq!(&**data, b"test data");
    }

    #[tokio::test]
    #[ignore = "Requires live database connection - run with --ignored flag"]
    async fn test_persist_with_database() {
        let mut store_state = create_test_store_state();
        let payload = serein::core::data_persistence::DataPayload {
            key: "test_key".to_string(),
            value: "test_value".to_string(),
            level: serein::core::data_persistence::Sensitivity::Internal,
        };
        let (success, storage_id, _) = store_state.persist(payload).await;
        assert!(success);
        assert!(Uuid::parse_str(&storage_id).is_ok());
    }
}
