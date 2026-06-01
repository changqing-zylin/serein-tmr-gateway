// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # WASI-Virt - Honeypot Injection & Virtual Filesystem Redirection
//!
//! Implements the WASI-Virt honeypot subsystem for the Aegis active defense
//! framework. Injects canary credentials (honey-tokens) into the guest
//! environment and redirects filesystem access to a virtual memory-backed
//! directory to detect and contain credential exfiltration attempts.
//!
//! ## Security Architecture
//! - **Honey-tokens**: Fake AWS and database credentials injected as environment
//!   variables. Any outbound request containing these tokens triggers an immediate
//!   security alert and sandbox termination.
//! - **VFS Redirection**: Guest filesystem writes are redirected to a virtual
//!   memory-backed `/dev/shm` directory, preventing persistent storage abuse
//!   and enabling forensic analysis of guest write patterns.
//! - **Canary Monitoring**: All guest environment variable reads and file writes
//!   are monitored for honey-token access patterns.
//!
//! ## Failure Modes
//! - Honey-token access by guest: Triggers Aegis countermeasure
//! - VFS write failure: Returns EIO to guest, logs the event
//! - Memory pressure on VFS: Evicts oldest entries with audit log

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use tracing::{info, warn, error};
use dashmap::DashMap;

#[cfg(feature = "host-engine")]
use moka::sync::Cache;

#[cfg(feature = "host-engine")]
use std::sync::OnceLock;
#[cfg(feature = "host-engine")]
use wasmtime::{Config, Engine, InstanceAllocationStrategy, Module, PoolingAllocationConfig};

/// Size of a single WASM page in bytes (64 KiB).
///
/// Hard physical boundary for the WASM sandbox: defines the fundamental
/// memory allocation unit used by all memory page calculations.
pub const WASM_PAGE_SIZE_BYTES: u64 = 64 * 1024;

#[cfg(feature = "host-engine")]
static GLOBAL_ENGINE: OnceLock<Engine> = OnceLock::new();

/// Maximum number of compiled WASM modules retained in the LRU cache.
///
/// Bounds the heap usage of the module cache to prevent OOM under high
/// concurrency with many distinct tenant modules. Each cached `Module`
/// holds the compiled Cranelift IR, so this limit directly controls
/// the maximum JIT compilation memory footprint.
#[cfg(feature = "host-engine")]
const MODULE_CACHE_MAX_CAPACITY: usize = 256;

/// Thread-safe LRU cache for compiled WASM `Module` instances.
///
/// Eliminates Cranelift cold-start storms by caching the compiled module
/// keyed by a tenant ID or module content hash. On cache hits, the
/// expensive `Module::new` compilation is skipped entirely, reducing
/// latency from hundreds of milliseconds to microseconds.
///
/// ## Bounding
/// The cache is strictly bounded to `MODULE_CACHE_MAX_CAPACITY` entries.
/// Eviction follows an LRU policy - least recently used modules are
/// discarded first under memory pressure.
#[cfg(feature = "host-engine")]
static MODULE_CACHE: OnceLock<Cache<String, Module>> = OnceLock::new();

#[cfg(feature = "host-engine")]
#[derive(Debug, Clone)]
pub struct WasiVirtEngineConfig {
    pub max_memory_pages: u32,
    pub total_component_instances: u32,
    pub max_core_instances_per_component: u32,
    pub total_memories: u32,
    pub max_memories_per_component: u32,
    pub async_support: bool,
    pub epoch_interruption: bool,
    pub consume_fuel: bool,
}

/// Hard physical boundary for the WASM sandbox: maximum number of WASM
/// component instances to prevent virtual memory exhaustion.
/// 1024 instances × 10MB per instance = ~10GB upper bound.
pub const MAX_INSTANCES_HARD_CAP: u32 = 1024;

/// Hard physical boundary for the WASM sandbox: maximum memory pages per
/// WASM instance (160 pages × 64KB = 10MB). Prevents a single guest from
/// consuming excessive virtual memory.
pub const MAX_MEMORY_PAGES_HARD_CAP: u32 = 160;

/// Hard physical boundary for the WASM sandbox: conservative total memory
/// reservation capacity in bytes. Limits the aggregate virtual memory committed
/// by the pooling allocator.
/// 1024 instances × 10MB = 10GB total reservation ceiling.
pub const MEMORY_RESERVATION_CAPACITY: u64 = MAX_INSTANCES_HARD_CAP as u64
    * MAX_MEMORY_PAGES_HARD_CAP as u64
    * WASM_PAGE_SIZE_BYTES;

#[cfg(feature = "host-engine")]
impl Default for WasiVirtEngineConfig {
    fn default() -> Self {
        Self {
            max_memory_pages: MAX_MEMORY_PAGES_HARD_CAP,
            total_component_instances: MAX_INSTANCES_HARD_CAP,
            max_core_instances_per_component: MAX_INSTANCES_HARD_CAP,
            total_memories: MAX_INSTANCES_HARD_CAP,
            max_memories_per_component: 1,
            async_support: true,
            epoch_interruption: true,
            consume_fuel: true,
        }
    }
}

#[cfg(feature = "host-engine")]
impl WasiVirtEngineConfig {
    pub fn max_memory_bytes(&self) -> usize {
        (self.max_memory_pages as u64 * WASM_PAGE_SIZE_BYTES) as usize
    }
}

#[cfg(feature = "host-engine")]
fn create_pooling_engine_config(cfg: &WasiVirtEngineConfig) -> Config {
    let mut config = Config::new();
    config.wasm_component_model(true);
    config.async_support(cfg.async_support);
    config.consume_fuel(cfg.consume_fuel);
    config.wasm_backtrace(true);
    config.wasm_memory64(false);
    config.epoch_interruption(cfg.epoch_interruption);

    config.static_memory_guard_size(4 * 1024 * 1024);
    config.dynamic_memory_reserved_for_growth(4 * 1024 * 1024);
    config.dynamic_memory_guard_size(4 * 1024 * 1024);

    let clamped_pages = cfg.max_memory_pages.min(MAX_MEMORY_PAGES_HARD_CAP);
    let clamped_instances = cfg.total_component_instances.min(MAX_INSTANCES_HARD_CAP);
    let max_memory_bytes = (clamped_pages as u64 * WASM_PAGE_SIZE_BYTES) as usize;

    let mut pooling = PoolingAllocationConfig::default();
    pooling.total_component_instances(clamped_instances);
    pooling.max_core_instances_per_component(
        cfg.max_core_instances_per_component.min(MAX_INSTANCES_HARD_CAP),
    );
    pooling.total_memories(cfg.total_memories.min(MAX_INSTANCES_HARD_CAP));
    pooling.max_memories_per_component(cfg.max_memories_per_component.max(1));
    pooling.max_memory_size(max_memory_bytes);

    info!(
        max_pages = clamped_pages,
        max_memory_mib = max_memory_bytes / (1024 * 1024),
        total_component_instances = clamped_instances,
        memory_reservation_gib = MEMORY_RESERVATION_CAPACITY / (1024 * 1024 * 1024),
        "WASI-Virt global engine: pooling allocator configured with hard caps"
    );

    config.allocation_strategy(InstanceAllocationStrategy::Pooling(pooling));
    config
}

#[cfg(feature = "host-engine")]
pub fn global_engine() -> &'static Engine {
    GLOBAL_ENGINE.get_or_init(|| {
        let config = WasiVirtEngineConfig::default();
        let pooling_config = create_pooling_engine_config(&config);
        match Engine::new(&pooling_config) {
            Ok(engine) => engine,
            Err(e) => {
                tracing::error!(error = %e, "WASI-Virt: failed to initialize global Wasmtime Engine");
                panic!("WASI-Virt: failed to initialize global Wasmtime Engine: {}", e);
            }
        }
    })
}

#[cfg(feature = "host-engine")]
pub fn init_global_engine(cfg: WasiVirtEngineConfig) -> anyhow::Result<&'static Engine> {
    if let Some(engine) = GLOBAL_ENGINE.get() {
        return Ok(engine);
    }
    let config = create_pooling_engine_config(&cfg);
    let engine = Engine::new(&config)?;
    info!("WASI-Virt global engine initialized - singleton via OnceLock");
    let _ = GLOBAL_ENGINE.set(engine);
    GLOBAL_ENGINE
        .get()
        .ok_or_else(|| anyhow::anyhow!("WASI-Virt: global engine not found after initialization"))
}

/// Access the global module LRU cache.
///
/// The cache is lazily initialized on first access with a maximum capacity
/// of `MODULE_CACHE_MAX_CAPACITY` entries. Thread-safe: concurrent access
/// is handled internally by `moka::sync::Cache`.
#[cfg(feature = "host-engine")]
pub fn module_cache() -> &'static Cache<String, Module> {
    MODULE_CACHE.get_or_init(|| {
        Cache::builder()
            .max_capacity(MODULE_CACHE_MAX_CAPACITY as u64)
            .build()
    })
}

/// Retrieve a compiled `Module` from the LRU cache, or compile it on cache miss.
///
/// ## Cache Key
/// The key is typically a tenant ID or a content hash of the WASM binary.
/// Using tenant ID ensures per-tenant module isolation while enabling
/// reuse across multiple requests from the same tenant.
///
/// ## Performance
/// - **Cache hit**: Returns in microseconds (no Cranelift compilation)
/// - **Cache miss**: Compiles via `Module::new` and inserts into the cache
///   for subsequent requests. Compilation typically takes 50-500ms depending
///   on module complexity.
///
/// ## Bounding
/// If the cache is at capacity, the least recently used module is evicted
/// to make room for the new entry. This prevents unbounded heap growth.
#[cfg(feature = "host-engine")]
pub fn get_or_compile_module(engine: &Engine, cache_key: &str, wasm_bytes: &[u8]) -> anyhow::Result<Module> {
    if let Some(cached) = module_cache().get(cache_key) {
        info!(
            key = %cache_key,
            "[WASI-VIRT] Module cache hit - skipping Cranelift compilation"
        );
        return Ok(cached);
    }

    let start = std::time::Instant::now();
    let module = Module::new(engine, wasm_bytes)?;
    let compile_ms = start.elapsed().as_millis();

    module_cache().insert(cache_key.to_string(), module.clone());

    info!(
        key = %cache_key,
        compile_ms = compile_ms,
        cache_size = module_cache().entry_count(),
        "[WASI-VIRT] Module compiled and cached - Cranelift JIT overhead amortized"
    );

    Ok(module)
}

/// Honey-token credentials injected into the guest environment.
///
/// These are fake credentials that should NEVER be used by legitimate code.
/// Any access to these values by the guest WASM module indicates a
/// credential exfiltration attempt and triggers the Aegis countermeasure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoneypotTokens {
    pub aws_access_key_id: String,
    pub aws_secret_access_key: String,
    pub db_password: String,
    pub db_connection_string: String,
    pub api_token: String,
}

impl HoneypotTokens {
    /// Generate a new set of honey-token credentials with distinctive canary markers.
    ///
    /// The tokens contain embedded canary identifiers that are registered with
    /// the Aegis monitoring system. Any outbound network request containing
    /// these identifiers will trigger an immediate security alert.
    pub fn generate() -> Self {
        let canary_id = Self::generate_canary_id();
        Self {
            aws_access_key_id: format!("AKIA3HONEYPOT{}FAKE", canary_id),
            aws_secret_access_key: format!("HoneypotSecret{}FakeKey/Not/Real/At/All", canary_id),
            db_password: format!("hp_db_pass_{}_canary", canary_id),
            db_connection_string: format!(
                "postgresql://honeypot:{}@db.internal.fake:5432/trap_db",
                canary_id
            ),
            api_token: format!("hp_tok_{}_aegis_canary", canary_id),
        }
    }

    /// Generate a 64-bit entropy-based canary ID using `fastrand`.
    ///
    /// Replaces the former `SystemTime`-based approach to guarantee zero
    /// collisions during simultaneous cluster-wide sandbox initialization.
    /// `fastrand` provides a fast, statistically uniform PRNG suitable for
    /// canary identifiers (not cryptographic keys).
    fn generate_canary_id() -> String {
        let id = fastrand::u64(..);
        format!("{:016x}", id)
    }

    /// Check if a given string contains any honey-token value.
    ///
    /// Returns the name of the compromised token if found, or `None` if
    /// the string is clean.
    pub fn detect_token_leak(&self, data: &str) -> Option<&str> {
        if data.contains(&self.aws_access_key_id) {
            return Some("AWS_ACCESS_KEY_ID");
        }
        if data.contains(&self.aws_secret_access_key) {
            return Some("AWS_SECRET_ACCESS_KEY");
        }
        if data.contains(&self.db_password) {
            return Some("DB_PASSWORD");
        }
        if data.contains(&self.db_connection_string) {
            return Some("DB_CONNECTION_STRING");
        }
        if data.contains(&self.api_token) {
            return Some("API_TOKEN");
        }
        None
    }
}

impl Default for HoneypotTokens {
    fn default() -> Self {
        Self::generate()
    }
}

/// A virtual file entry in the memory-backed filesystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualFile {
    pub path: String,
    pub content: Vec<u8>,
    pub created_at: u64,
    pub modified_at: u64,
    pub size: usize,
}

/// Virtual filesystem backed by in-memory storage.
///
/// Redirects all guest filesystem writes to a virtual `/dev/shm` directory,
/// preventing persistent storage abuse while enabling forensic analysis.
///
/// ## Thread Safety
/// All internal state uses lock-free or fine-grained concurrent primitives:
/// - `DashMap` for the file store (sharded internal locks, no global contention)
/// - `AtomicU64` for counters and byte tracking (lock-free with `SeqCst` ordering)
///
/// All mutating methods take `&self`, enabling concurrent access from multiple
/// WASI host-call threads without external locking.
pub struct VirtualFileSystem {
    pub mount_point: String,
    pub max_total_bytes: usize,
    pub max_file_count: usize,
    files: DashMap<String, VirtualFile>,
    total_bytes_used: AtomicU64,
    write_count: AtomicU64,
    read_count: AtomicU64,
}

impl Serialize for VirtualFileSystem {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("VirtualFileSystem", 6)?;
        state.serialize_field("mount_point", &self.mount_point)?;
        state.serialize_field("max_total_bytes", &self.max_total_bytes)?;
        state.serialize_field("max_file_count", &self.max_file_count)?;

        let files_map: HashMap<String, VirtualFile> = self
            .files
            .iter()
            .map(|ref_multi| (ref_multi.key().clone(), ref_multi.value().clone()))
            .collect();
        state.serialize_field("files", &files_map)?;

        state.serialize_field("total_bytes_used", &self.total_bytes_used.load(Ordering::SeqCst))?;
        state.serialize_field("write_count", &self.write_count.load(Ordering::SeqCst))?;
        state.serialize_field("read_count", &self.read_count.load(Ordering::SeqCst))?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for VirtualFileSystem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct VfsHelper {
            mount_point: String,
            max_total_bytes: usize,
            max_file_count: usize,
            files: HashMap<String, VirtualFile>,
            total_bytes_used: u64,
            write_count: u64,
            read_count: u64,
        }

        let helper = VfsHelper::deserialize(deserializer)?;
        let dash_map = DashMap::with_capacity(helper.files.len());
        for (k, v) in helper.files {
            dash_map.insert(k, v);
        }

        Ok(Self {
            mount_point: helper.mount_point,
            max_total_bytes: helper.max_total_bytes,
            max_file_count: helper.max_file_count,
            files: dash_map,
            total_bytes_used: AtomicU64::new(helper.total_bytes_used),
            write_count: AtomicU64::new(helper.write_count),
            read_count: AtomicU64::new(helper.read_count),
        })
    }
}

impl std::fmt::Debug for VirtualFileSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualFileSystem")
            .field("mount_point", &self.mount_point)
            .field("max_total_bytes", &self.max_total_bytes)
            .field("max_file_count", &self.max_file_count)
            .field("file_count", &self.files.len())
            .field("total_bytes_used", &self.total_bytes_used.load(Ordering::SeqCst))
            .field("write_count", &self.write_count.load(Ordering::SeqCst))
            .field("read_count", &self.read_count.load(Ordering::SeqCst))
            .finish()
    }
}

impl Clone for VirtualFileSystem {
    fn clone(&self) -> Self {
        let dash_map = DashMap::with_capacity(self.files.len());
        for ref_multi in self.files.iter() {
            dash_map.insert(ref_multi.key().clone(), ref_multi.value().clone());
        }
        Self {
            mount_point: self.mount_point.clone(),
            max_total_bytes: self.max_total_bytes,
            max_file_count: self.max_file_count,
            files: dash_map,
            total_bytes_used: AtomicU64::new(self.total_bytes_used.load(Ordering::SeqCst)),
            write_count: AtomicU64::new(self.write_count.load(Ordering::SeqCst)),
            read_count: AtomicU64::new(self.read_count.load(Ordering::SeqCst)),
        }
    }
}

impl VirtualFileSystem {
    const DEFAULT_MOUNT_POINT: &'static str = "/dev/shm/serein-vfs";
    const DEFAULT_MAX_BYTES: usize = 64 * 1024 * 1024;
    const DEFAULT_MAX_FILE_COUNT: usize = 1000;

    pub fn new() -> Self {
        Self {
            mount_point: Self::DEFAULT_MOUNT_POINT.to_string(),
            max_total_bytes: Self::DEFAULT_MAX_BYTES,
            max_file_count: Self::DEFAULT_MAX_FILE_COUNT,
            files: DashMap::new(),
            total_bytes_used: AtomicU64::new(0),
            write_count: AtomicU64::new(0),
            read_count: AtomicU64::new(0),
        }
    }

    pub fn with_mount_point(mut self, mount: &str) -> Self {
        self.mount_point = mount.to_string();
        self
    }

    pub fn with_max_bytes(mut self, max: usize) -> Self {
        self.max_total_bytes = max;
        self
    }

    pub fn with_max_file_count(mut self, max: usize) -> Self {
        self.max_file_count = max.max(1);
        self
    }

    /// Redirect a guest path to the virtual filesystem mount point.
    ///
    /// All paths are remapped under the virtual mount point to prevent
    /// the guest from accessing the real host filesystem. Path traversal
    /// is blocked by normalizing `../` and `./` segments before prefixing
    /// with the mount point.
    pub fn redirect_path(&self, guest_path: &str) -> String {
        let stripped = guest_path.trim_start_matches('/');
        let normalized = Self::normalize_path_segments(stripped);
        format!("{}/{}", self.mount_point, normalized)
    }

    /// Normalize a path by resolving and stripping `.` and `..` segments.
    ///
    /// Uses a stack-based approach: each `..` pops the last segment (if any),
    /// and `.` segments are discarded. This guarantees the result cannot
    /// escape the VFS namespace via path traversal.
    fn normalize_path_segments(path: &str) -> String {
        let mut stack: Vec<&str> = Vec::new();
        for segment in path.split('/') {
            match segment {
                "" | "." => {}
                ".." => {
                    stack.pop();
                }
                _ => stack.push(segment),
            }
        }
        stack.join("/")
    }

    /// Write data to the virtual filesystem.
    ///
    /// Returns an error if the write would exceed the total byte budget.
    /// Thread-safe: uses DashMap for fine-grained file-level locking and
    /// AtomicU64 for lock-free counter updates.
    pub fn write_file(&self, guest_path: &str, data: &[u8]) -> Result<(), VfsError> {
        let vfs_path = self.redirect_path(guest_path);
        let data_len = data.len();

        if data_len > self.max_total_bytes {
            warn!(
                guest_path = %guest_path,
                requested_bytes = data_len,
                budget_bytes = self.max_total_bytes,
                "[WASI-VIRT] VFS write rejected - single file exceeds total memory budget"
            );
            return Err(VfsError::BudgetExceeded {
                requested: data_len,
                available: self.max_total_bytes.saturating_sub(self.total_bytes_used.load(Ordering::SeqCst) as usize),
            });
        }

        let is_new_file = !self.files.contains_key(&vfs_path);
        if is_new_file && self.files.len() >= self.max_file_count {
            warn!(
                guest_path = %guest_path,
                current_files = self.files.len(),
                max_files = self.max_file_count,
                "[WASI-VIRT] VFS write rejected - inode count limit reached"
            );
            return Err(VfsError::InodeExhausted {
                current: self.files.len(),
                maximum: self.max_file_count,
            });
        }

        let existing_size = self.files.get(&vfs_path).map(|f| f.size).unwrap_or(0);

        // Atomic CAS loop for total_bytes_used to prevent TOCTOU quota bypass.
        //
        // The quota check `new_total > self.max_total_bytes` is performed
        // *inside* the atomic update, so no concurrent writer can slip past
        // the budget between the load and the store.
        let new_total = match self.total_bytes_used.fetch_update(
            Ordering::SeqCst,
            Ordering::SeqCst,
            |current| {
                let current_usize = current as usize;
                let proposed = current_usize.saturating_sub(existing_size) + data_len;
                if proposed > self.max_total_bytes {
                    None
                } else {
                    Some(proposed as u64)
                }
            },
        ) {
            Ok(updated) => updated as usize,
            Err(_) => {
                let current_used = self.total_bytes_used.load(Ordering::SeqCst) as usize;
                warn!(
                    guest_path = %guest_path,
                    requested_bytes = data_len,
                    budget_bytes = self.max_total_bytes,
                    used_bytes = current_used,
                    "[WASI-VIRT] VFS write rejected - memory budget exceeded (atomic CAS)"
                );
                return Err(VfsError::BudgetExceeded {
                    requested: data_len,
                    available: self.max_total_bytes.saturating_sub(current_used),
                });
            }
        };

        let now = Self::current_timestamp();
        let created_at = self
            .files
            .get(&vfs_path)
            .map(|f| f.created_at)
            .unwrap_or(now);

        let file = VirtualFile {
            path: vfs_path.clone(),
            content: data.to_vec(),
            created_at,
            modified_at: now,
            size: data_len,
        };

        self.files.insert(vfs_path.clone(), file);
        self.write_count.fetch_add(1, Ordering::SeqCst);

        info!(
            guest_path = %guest_path,
            size = data_len,
            total_used = new_total,
            "[WASI-VIRT] Virtual file write completed"
        );

        Ok(())
    }

    /// Read data from the virtual filesystem.
    ///
    /// Thread-safe: DashMap allows concurrent reads without blocking writers.
    pub fn read_file(&self, guest_path: &str) -> Result<Vec<u8>, VfsError> {
        let vfs_path = self.redirect_path(guest_path);
        self.files
            .get(&vfs_path)
            .map(|f| {
                self.read_count.fetch_add(1, Ordering::SeqCst);
                f.content.clone()
            })
            .ok_or_else(|| VfsError::FileNotFound {
                path: guest_path.to_string(),
            })
    }

    /// Delete a file from the virtual filesystem.
    ///
    /// Thread-safe: DashMap removal is atomic per key.
    pub fn delete_file(&self, guest_path: &str) -> Result<(), VfsError> {
        let vfs_path = self.redirect_path(guest_path);
        if let Some((_, file)) = self.files.remove(&vfs_path) {
            self.total_bytes_used.fetch_sub(file.size as u64, Ordering::SeqCst);
            info!(
                guest_path = %guest_path,
                freed_bytes = file.size,
                "[WASI-VIRT] Virtual file deleted"
            );
            Ok(())
        } else {
            Err(VfsError::FileNotFound {
                path: guest_path.to_string(),
            })
        }
    }

    /// List all files in the virtual filesystem.
    ///
    /// Returns a snapshot of current file references. The collection is
    /// point-in-time consistent per shard but not globally atomic.
    pub fn list_files(&self) -> Vec<VirtualFile> {
        self.files.iter().map(|r| r.value().clone()).collect()
    }

    /// Return the current number of files in the VFS.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Return the total bytes currently used.
    pub fn bytes_used(&self) -> u64 {
        self.total_bytes_used.load(Ordering::SeqCst)
    }

    /// Return the total write operation count.
    pub fn write_count(&self) -> u64 {
        self.write_count.load(Ordering::SeqCst)
    }

    /// Return the total read operation count.
    pub fn read_count(&self) -> u64 {
        self.read_count.load(Ordering::SeqCst)
    }

    fn current_timestamp() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_millis() as u64,
            Err(e) => {
                warn!(
                    error = %e,
                    "[WASI-VIRT] System clock drifted backwards - falling back to timestamp 0"
                );
                0
            }
        }
    }
}

impl Default for VirtualFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// VFS operation errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum VfsError {
    #[error("VFS memory budget exceeded: requested {requested} bytes, {available} available")]
    BudgetExceeded { requested: usize, available: usize },

    #[error("VFS inode exhaustion: {current} files exist, maximum {maximum}")]
    InodeExhausted { current: usize, maximum: usize },

    #[error("Virtual file not found: {path}")]
    FileNotFound { path: String },

    #[error("VFS path traversal detected: {path}")]
    PathTraversalDetected { path: String },
}

/// The complete WASI-Virt sandbox context combining honeypot tokens
/// and virtual filesystem for guest containment.
///
/// ## Thread Safety
/// - `vfs`: Fully concurrent via `DashMap` + `AtomicU64` (all `&self` methods)
/// - `leak_detected`: `AtomicBool` for lock-free leak flagging
/// - `leaked_token`: Protected by `RwLock` for rare-write, frequent-read pattern
/// - `environment`: Read-only after construction, no synchronization needed
pub struct WasiVirtContext {
    pub tokens: HoneypotTokens,
    pub vfs: VirtualFileSystem,
    pub environment: HashMap<String, String>,
    leak_detected: AtomicBool,
    leaked_token: RwLock<Option<String>>,
}

impl Serialize for WasiVirtContext {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("WasiVirtContext", 5)?;
        state.serialize_field("tokens", &self.tokens)?;
        state.serialize_field("vfs", &self.vfs)?;
        state.serialize_field("environment", &self.environment)?;
        state.serialize_field("leak_detected", &self.leak_detected.load(Ordering::SeqCst))?;
        state.serialize_field("leaked_token", &*self.leaked_token.read().unwrap_or_else(|e| e.into_inner()))?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for WasiVirtContext {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WasiVirtContextHelper {
            tokens: HoneypotTokens,
            vfs: VirtualFileSystem,
            environment: HashMap<String, String>,
            leak_detected: bool,
            leaked_token: Option<String>,
        }

        let helper = WasiVirtContextHelper::deserialize(deserializer)?;
        Ok(Self {
            tokens: helper.tokens,
            vfs: helper.vfs,
            environment: helper.environment,
            leak_detected: AtomicBool::new(helper.leak_detected),
            leaked_token: RwLock::new(helper.leaked_token),
        })
    }
}

impl std::fmt::Debug for WasiVirtContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasiVirtContext")
            .field("tokens", &self.tokens)
            .field("vfs", &self.vfs)
            .field("environment", &self.environment)
            .field("leak_detected", &self.leak_detected.load(Ordering::SeqCst))
            .field("leaked_token", &*self.leaked_token.read().unwrap_or_else(|e| e.into_inner()))
            .finish()
    }
}

impl Clone for WasiVirtContext {
    fn clone(&self) -> Self {
        Self {
            tokens: self.tokens.clone(),
            vfs: self.vfs.clone(),
            environment: self.environment.clone(),
            leak_detected: AtomicBool::new(self.leak_detected.load(Ordering::SeqCst)),
            leaked_token: RwLock::new(self.leaked_token.read().unwrap_or_else(|e| e.into_inner()).clone()),
        }
    }
}

impl WasiVirtContext {
    pub fn new() -> Self {
        let tokens = HoneypotTokens::generate();
        let mut environment = HashMap::new();
        environment.insert("AWS_ACCESS_KEY_ID".to_string(), tokens.aws_access_key_id.clone());
        environment.insert("AWS_SECRET_ACCESS_KEY".to_string(), tokens.aws_secret_access_key.clone());
        environment.insert("DB_PASSWORD".to_string(), tokens.db_password.clone());
        environment.insert("DATABASE_URL".to_string(), tokens.db_connection_string.clone());
        environment.insert("API_TOKEN".to_string(), tokens.api_token.clone());

        Self {
            tokens,
            vfs: VirtualFileSystem::new(),
            environment,
            leak_detected: AtomicBool::new(false),
            leaked_token: RwLock::new(None),
        }
    }

    /// Scan guest output for honey-token leaks.
    ///
    /// If a honey-token is detected in guest output, this method atomically
    /// marks the context as compromised and returns the name of the leaked token.
    /// Thread-safe: uses `AtomicBool` for the flag and `RwLock` for the token name.
    pub fn scan_for_leaks(&self, data: &str) -> Option<String> {
        if let Some(token_name) = self.tokens.detect_token_leak(data) {
            self.leak_detected.store(true, Ordering::SeqCst);
            *self.leaked_token.write().unwrap_or_else(|e| e.into_inner()) = Some(token_name.to_string());
            error!(
                leaked_token = token_name,
                "[WASI-VIRT] HONEY-TOKEN LEAK DETECTED - guest attempted credential exfiltration"
            );
            return Some(token_name.to_string());
        }
        None
    }

    /// Check whether a honey-token leak has been detected.
    pub fn leak_detected(&self) -> bool {
        self.leak_detected.load(Ordering::SeqCst)
    }

    /// Get the name of the leaked token, if any.
    pub fn leaked_token(&self) -> Option<String> {
        self.leaked_token.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Get the guest environment variables with injected honey-tokens.
    pub fn guest_environment(&self) -> &HashMap<String, String> {
        &self.environment
    }

    /// Redirect a guest filesystem path through the VFS layer.
    pub fn redirect_path(&self, guest_path: &str) -> String {
        self.vfs.redirect_path(guest_path)
    }
}

impl Default for WasiVirtContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a complete honeypot context for WASI-Virt sandbox initialization.
///
/// This function creates the full WASI-Virt context with:
/// 1. Fresh honey-token credentials injected as environment variables
/// 2. Virtual filesystem mounted at `/dev/shm/serein-vfs`
/// 3. Leak detection monitoring enabled
///
/// ## Usage
/// Call this function when initializing a new WASM guest sandbox to
/// inject the honeypot environment. The returned context should be
/// stored in the `StoreState` and referenced during guest execution.
pub fn build_honeypot_context() -> WasiVirtContext {
    let context = WasiVirtContext::new();
    info!(
        mount_point = %context.vfs.mount_point,
        token_count = context.environment.len(),
        "[WASI-VIRT] Honeypot context built - canary credentials injected, VFS mounted"
    );
    context
}

/// Thread-safe wrapper for WASI-Virt context used in StoreState.
///
/// Since `WasiVirtContext` is now internally thread-safe (DashMap + AtomicU64 +
/// AtomicBool + RwLock), a simple `Arc` wrapper suffices - no external lock needed.
pub type SharedWasiVirtContext = Arc<WasiVirtContext>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_honeypot_tokens_generate() {
        let tokens = HoneypotTokens::generate();
        assert!(tokens.aws_access_key_id.starts_with("AKIA3HONEYPOT"));
        assert!(tokens.aws_secret_access_key.contains("HoneypotSecret"));
        assert!(tokens.db_password.starts_with("hp_db_pass_"));
        assert!(tokens.db_connection_string.contains("honeypot"));
        assert!(tokens.api_token.starts_with("hp_tok_"));
    }

    #[test]
    fn test_detect_token_leak() {
        let tokens = HoneypotTokens::generate();
        assert_eq!(
            tokens.detect_token_leak(&format!("key={}", tokens.aws_access_key_id)),
            Some("AWS_ACCESS_KEY_ID")
        );
        assert_eq!(
            tokens.detect_token_leak(&format!("pass={}", tokens.db_password)),
            Some("DB_PASSWORD")
        );
        assert_eq!(tokens.detect_token_leak("clean data"), None);
    }

    #[test]
    fn test_vfs_write_and_read() {
        let vfs = VirtualFileSystem::new();
        vfs.write_file("/test.txt", b"hello world").unwrap();
        let data = vfs.read_file("/test.txt").unwrap();
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn test_vfs_path_redirection() {
        let vfs = VirtualFileSystem::new();
        assert_eq!(
            vfs.redirect_path("/etc/passwd"),
            "/dev/shm/serein-vfs/etc/passwd"
        );
    }

    #[test]
    fn test_vfs_budget_enforcement() {
        let vfs = VirtualFileSystem::new().with_max_bytes(100);
        vfs.write_file("/small.txt", b"short").unwrap();
        let result = vfs.write_file("/big.txt", &[0u8; 200]);
        assert!(matches!(result, Err(VfsError::BudgetExceeded { .. })));
    }

    #[test]
    fn test_vfs_delete_file() {
        let vfs = VirtualFileSystem::new();
        vfs.write_file("/temp.txt", b"data").unwrap();
        vfs.delete_file("/temp.txt").unwrap();
        assert!(vfs.read_file("/temp.txt").is_err());
    }

    #[test]
    fn test_build_honeypot_context() {
        let ctx = build_honeypot_context();
        assert!(ctx.environment.contains_key("AWS_ACCESS_KEY_ID"));
        assert!(ctx.environment.contains_key("DB_PASSWORD"));
        assert!(!ctx.leak_detected());
    }

    #[test]
    fn test_scan_for_leaks() {
        let ctx = build_honeypot_context();
        let aws_key = ctx.environment.get("AWS_ACCESS_KEY_ID").unwrap().clone();
        let result = ctx.scan_for_leaks(&format!("exfil: {}", aws_key));
        assert_eq!(result, Some("AWS_ACCESS_KEY_ID".to_string()));
        assert!(ctx.leak_detected());
    }

    #[test]
    fn test_vfs_file_not_found() {
        let vfs = VirtualFileSystem::new();
        let result = vfs.read_file("/nonexistent.txt");
        assert!(matches!(result, Err(VfsError::FileNotFound { .. })));
    }

    #[test]
    fn test_vfs_inode_exhaustion() {
        let vfs = VirtualFileSystem::new().with_max_file_count(2);
        vfs.write_file("/file1.txt", b"data1").unwrap();
        vfs.write_file("/file2.txt", b"data2").unwrap();
        let result = vfs.write_file("/file3.txt", b"data3");
        assert!(matches!(result, Err(VfsError::InodeExhausted { .. })));
    }

    #[test]
    fn test_vfs_inode_update_existing_file() {
        let vfs = VirtualFileSystem::new().with_max_file_count(1);
        vfs.write_file("/file1.txt", b"data1").unwrap();
        let result = vfs.write_file("/file1.txt", b"updated_data");
        assert!(result.is_ok());
    }
}
