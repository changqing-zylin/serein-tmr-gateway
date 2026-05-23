// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Flight Recorder - Structured Audit Trail for TMR Consensus
//!
//! Append-only audit log for all TMR consensus events, provider responses,
//! and security incidents. Uses a bounded MPSC channel with a background
//! SQLite writer thread to ensure the hot-path is never blocked by disk I/O.
//!
//! ## Architecture
//! - **Bounded Channel**: `sync_channel(10000)` prevents OOM under burst load;
//!   backpressure drops events with a warning rather than blocking the gateway
//! - **Batched WAL Writes**: Background thread accumulates up to 64 records
//!   before committing a WAL transaction, amortizing fsync overhead
//! - **Daily Log Rotation**: `ConsensusFlightRecorder` rotates JSONL output
//!   files at UTC midnight for downstream SSD training pipelines
//! - **TTL Garbage Collection**: SQLite records older than 30 days are pruned
//!   daily to prevent unbounded disk growth

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::mpsc::{self, SyncSender, Receiver};
use std::time::Duration as StdDuration;
use tracing::{debug, error, info, warn};

const CHANNEL_BOUND: usize = 10000;
const BATCH_SIZE: usize = 64;
const BATCH_FLUSH_TIMEOUT: StdDuration = StdDuration::from_millis(200);

/// Classification of auditable events in the TMR consensus lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    PromptSent,
    ResponseReceived,
    ResponseError,
    CircuitBreakerTripped,
    CircuitBreakerReset,
    RateLimitHit,
    TokenBucketExhausted,
    CacheHit,
    CacheMiss,
    SecurityEvent,
}

/// A single auditable event record persisted in the SQLite flight log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlightRecord {
    pub id: i64,
    pub timestamp: DateTime<Utc>,
    pub event_type: EventType,
    pub node_name: String,
    pub request_id: String,
    pub payload_summary: String,
    pub raw_prompt: Option<String>,
    pub raw_response: Option<String>,
    pub error_message: Option<String>,
    pub duration_ms: Option<u64>,
    pub token_count: Option<u32>,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// Query parameters for filtering flight records from the audit log.
#[derive(Debug, Clone, Default)]
pub struct RecordQuery {
    pub event_type: Option<EventType>,
    pub node_name: Option<String>,
    pub request_id: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

enum FlightRecorderCommand {
    RecordEvent(Box<FlightRecorderRecord>),
    Shutdown,
}

struct FlightRecorderRecord {
    timestamp: String,
    event_type: String,
    node_name: String,
    request_id: String,
    payload_summary: String,
    raw_prompt: Option<String>,
    raw_response: Option<String>,
    error_message: Option<String>,
    duration_ms: Option<i64>,
    token_count: Option<i64>,
    metadata: String,
}

/// Arguments for recording a single flight event on the hot-path.
pub struct RecordEventArgs<'a> {
    pub event_type: EventType,
    pub node_name: &'a str,
    pub request_id: &'a str,
    pub payload_summary: &'a str,
    pub raw_prompt: Option<String>,
    pub raw_response: Option<String>,
    pub error_message: Option<String>,
    pub duration_ms: Option<u64>,
    pub token_count: Option<u32>,
}

/// Non-blocking flight recorder backed by a bounded MPSC channel and SQLite WAL.
///
/// All `record_*` methods use `try_send` - the gateway hot-path is never
/// blocked by disk I/O. When the channel is full, events are dropped with
/// a warning rather than stalling the request.
pub struct FlightRecorder {
    tx: SyncSender<FlightRecorderCommand>,
}

/// Errors raised during flight recorder initialization.
#[derive(Debug, thiserror::Error)]
pub enum FlightRecorderError {
    #[error("SQLite open failed: {0}")]
    DatabaseOpen(#[source] rusqlite::Error),

    #[error("SQLite schema initialization failed: {0}")]
    SchemaInit(#[source] rusqlite::Error),
}

fn insert_batch(conn: &mut Connection, batch: &[FlightRecorderCommand]) {
    if batch.is_empty() {
        return;
    }

    let tx = match conn.transaction() {
        Ok(tx) => tx,
        Err(e) => {
            error!("[FLIGHT RECORDER] Failed to begin batch transaction: {}", e);
            return;
        }
    };

    let mut stmt = match tx.prepare_cached(
        "INSERT INTO flight_records (
            timestamp, event_type, node_name, request_id,
            payload_summary, raw_prompt, raw_response,
            error_message, duration_ms, token_count, metadata
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)"
    ) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("[FLIGHT RECORDER] Failed to prepare cached statement: {}", e);
            return;
        }
    };

    let mut inserted = 0usize;
    for command in batch {
        if let FlightRecorderCommand::RecordEvent(rec) = command {
            match stmt.execute(params![
                rec.timestamp,
                rec.event_type,
                rec.node_name,
                rec.request_id,
                rec.payload_summary,
                rec.raw_prompt,
                rec.raw_response,
                rec.error_message,
                rec.duration_ms,
                rec.token_count,
                rec.metadata
            ]) {
                Ok(_) => inserted += 1,
                Err(e) => error!(
                    node = %rec.node_name,
                    request_id = %rec.request_id,
                    "Failed to insert flight record in batch: {}", e
                ),
            }
        }
    }

    drop(stmt);

    if let Err(e) = tx.commit() {
        error!("[FLIGHT RECORDER] Failed to commit batch transaction: {}", e);
    } else {
        debug!(
            batch_size = inserted,
            "[FLIGHT RECORDER] Batch committed {} records to SQLite", inserted
        );
    }
}

fn background_writer(rx: Receiver<FlightRecorderCommand>, mut conn: Connection) {
    let mut batch: Vec<FlightRecorderCommand> = Vec::with_capacity(BATCH_SIZE);

    loop {
        let timeout_reached = match rx.recv_timeout(BATCH_FLUSH_TIMEOUT) {
            Ok(command) => {
                match command {
                    FlightRecorderCommand::Shutdown => {
                        if !batch.is_empty() {
                            insert_batch(&mut conn, &batch);
                            batch.clear();
                        }
                        info!("[FLIGHT RECORDER] Background writer thread shutting down");
                        return;
                    }
                    cmd => {
                        batch.push(cmd);
                        batch.len() >= BATCH_SIZE
                    }
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => true,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                if !batch.is_empty() {
                    insert_batch(&mut conn, &batch);
                }
                info!("[FLIGHT RECORDER] Channel disconnected - flushing remaining batch and exiting");
                return;
            }
        };

        if timeout_reached && !batch.is_empty() {
            insert_batch(&mut conn, &batch);
            batch.clear();
        }

        while batch.len() < BATCH_SIZE {
            match rx.try_recv() {
                Ok(command) => {
                    match command {
                        FlightRecorderCommand::Shutdown => {
                            insert_batch(&mut conn, &batch);
                            info!("[FLIGHT RECORDER] Background writer thread shutting down");
                            return;
                        }
                        cmd => batch.push(cmd),
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    insert_batch(&mut conn, &batch);
                    info!("[FLIGHT RECORDER] Channel disconnected - flushing and exiting");
                    return;
                }
            }
        }
    }
}

impl FlightRecorder {
    pub fn new() -> Result<Self, FlightRecorderError> {
        let write_conn = Connection::open("serein_audit.db")
            .map_err(FlightRecorderError::DatabaseOpen)?;

        write_conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA wal_autocheckpoint = 1000;
             PRAGMA busy_timeout = 5000;
             PRAGMA temp_store = MEMORY;
             CREATE TABLE IF NOT EXISTS flight_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                node_name TEXT NOT NULL,
                request_id TEXT NOT NULL,
                payload_summary TEXT NOT NULL,
                raw_prompt TEXT,
                raw_response TEXT,
                error_message TEXT,
                duration_ms INTEGER,
                token_count INTEGER,
                metadata TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_flight_records_ts ON flight_records(timestamp);
            CREATE INDEX IF NOT EXISTS idx_flight_records_req ON flight_records(request_id);",
        )
        .map_err(FlightRecorderError::SchemaInit)?;

        let (tx, rx) = mpsc::sync_channel::<FlightRecorderCommand>(CHANNEL_BOUND);

        std::thread::Builder::new()
            .name("serein-flight-recorder".to_string())
            .spawn(move || {
                background_writer(rx, write_conn);
            })
            .map_err(|e| FlightRecorderError::DatabaseOpen(
                rusqlite::Error::InvalidParameterName(format!(
                    "Failed to spawn flight recorder writer thread: {}", e
                ))
            ))?;

        info!(
            channel_bound = CHANNEL_BOUND,
            batch_size = BATCH_SIZE,
            flush_timeout_ms = BATCH_FLUSH_TIMEOUT.as_millis(),
            "[FLIGHT RECORDER] Background writer spawned - batched WAL2 writes, non-blocking try_send hot-path"
        );

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(86400));
            loop {
                interval.tick().await;
                tracing::info!("[FLIGHT RECORDER] Running daily TTL garbage collection...");

                let _ = tokio::task::spawn_blocking(|| {
                    if let Ok(gc_conn) = rusqlite::Connection::open("serein_audit.db") {
                        let _ = gc_conn.execute("DELETE FROM flight_records WHERE timestamp < datetime('now', '-30 days');", []);
                        let _ = gc_conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);");
                    }
                }).await;
            }
        });

        Ok(Self {
            tx,
        })
    }

    pub fn record_event(&self, args: RecordEventArgs) {
        let timestamp = Utc::now().to_rfc3339();

        let et_str = serde_json::to_string(&args.event_type).unwrap_or_else(|_| "\"Unknown\"".to_string());
        let et_clean = et_str.trim_matches('"').to_string();

        let summary = truncate(args.payload_summary, 500);
        let prompt = args.raw_prompt.map(|s| truncate(&s, 10_000));
        let response = args.raw_response.map(|s| truncate(&s, 10_000));

        let command = FlightRecorderCommand::RecordEvent(Box::new(FlightRecorderRecord {
            timestamp,
            event_type: et_clean,
            node_name: args.node_name.to_string(),
            request_id: args.request_id.to_string(),
            payload_summary: summary,
            raw_prompt: prompt,
            raw_response: response,
            error_message: args.error_message,
            duration_ms: args.duration_ms.map(|d| d as i64),
            token_count: args.token_count.map(|t| t as i64),
            metadata: "{}".to_string(),
        }));

        if let Err(e) = self.tx.try_send(command) {
            match e {
                mpsc::TrySendError::Full(_) => {
                    warn!(
                        node = %args.node_name,
                        request_id = %args.request_id,
                        "[FLIGHT RECORDER] Channel full - event dropped, gateway NOT blocked"
                    );
                }
                mpsc::TrySendError::Disconnected(_) => {
                    error!(
                        "[FLIGHT RECORDER] Channel disconnected - background writer terminated"
                    );
                }
            }
        }
    }

    pub fn record_response(
        &self,
        node_name: &str,
        request_id: &str,
        prompt: &str,
        response: &str,
        duration_ms: u64,
        token_count: u32,
    ) {
        self.record_event(RecordEventArgs {
            event_type: EventType::ResponseReceived,
            node_name,
            request_id,
            payload_summary: &format!("Response from {} ({} tokens)", node_name, token_count),
            raw_prompt: Some(prompt.to_string()),
            raw_response: Some(response.to_string()),
            error_message: None,
            duration_ms: Some(duration_ms),
            token_count: Some(token_count),
        });
    }

    pub fn record_error(
        &self,
        node_name: &str,
        request_id: &str,
        prompt: &str,
        error: &str,
        duration_ms: u64,
    ) {
        self.record_event(RecordEventArgs {
            event_type: EventType::ResponseError,
            node_name,
            request_id,
            payload_summary: &format!("Error from {}: {}", node_name, error),
            raw_prompt: Some(prompt.to_string()),
            raw_response: None,
            error_message: Some(error.to_string()),
            duration_ms: Some(duration_ms),
            token_count: None,
        });
    }

    pub async fn query(&self, query: &RecordQuery) -> Vec<FlightRecord> {
        let limit = query.limit.unwrap_or(100);
        let offset = query.offset.unwrap_or(0);

        tokio::task::spawn_blocking(move || {
            let read_conn = match Connection::open_with_flags(
                "serein_audit.db",
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
            ) {
                Ok(c) => c,
                Err(e) => {
                    error!("Failed to open read-only SQLite connection: {}", e);
                    return vec![];
                }
            };

            if let Err(e) = read_conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA read_uncommitted = ON;
                 PRAGMA wal_autocheckpoint = 1000;
                 PRAGMA busy_timeout = 5000;"
            ) {
                error!("Failed to set read-only PRAGMAs: {}", e);
                return vec![];
            }

            let sql = format!(
                "SELECT id, timestamp, event_type, node_name, request_id,
                        payload_summary, raw_prompt, raw_response, error_message,
                        duration_ms, token_count, metadata
                 FROM flight_records
                 ORDER BY timestamp DESC
                 LIMIT {} OFFSET {}",
                limit, offset
            );

            let mut stmt = match read_conn.prepare(&sql) {
                Ok(s) => s,
                Err(e) => {
                    error!("SQLite prepare failed: {}", e);
                    return vec![];
                }
            };

            let row_iter = match stmt.query_map([], |row| {
                let ts_str: String = row.get(1)?;
                let ts = DateTime::parse_from_rfc3339(&ts_str)
                    .map(|d| d.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                let et_str: String = row.get(2)?;
                let et_json = format!("\"{}\"", et_str);
                let et: EventType = serde_json::from_str(&et_json).unwrap_or(EventType::SecurityEvent);

                let meta_str: String = row.get(11)?;
                let meta = serde_json::from_str(&meta_str).unwrap_or_default();

                Ok(FlightRecord {
                    id: row.get(0)?,
                    timestamp: ts,
                    event_type: et,
                    node_name: row.get(3)?,
                    request_id: row.get(4)?,
                    payload_summary: row.get(5)?,
                    raw_prompt: row.get(6)?,
                    raw_response: row.get(7)?,
                    error_message: row.get(8)?,
                    duration_ms: row.get::<_, Option<i64>>(9)?.map(|v| v as u64),
                    token_count: row.get::<_, Option<i64>>(10)?.map(|v| v as u32),
                    metadata: meta,
                })
            }) {
                Ok(iter) => iter,
                Err(e) => {
                    error!("SQLite query_map failed: {}", e);
                    return vec![];
                }
            };

            let mut results = Vec::new();
            for record in row_iter.flatten() {
                results.push(record);
            }
            results
        }).await.unwrap_or_default()
    }

    pub async fn count(&self) -> usize {
        tokio::task::spawn_blocking(move || {
            let read_conn = match Connection::open_with_flags(
                "serein_audit.db",
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
            ) {
                Ok(c) => c,
                Err(e) => {
                    error!("Failed to open read-only SQLite connection for count: {}", e);
                    return 0usize;
                }
            };

            if let Err(e) = read_conn.execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA read_uncommitted = ON;
                 PRAGMA wal_autocheckpoint = 1000;
                 PRAGMA busy_timeout = 5000;"
            ) {
                error!("Failed to set read-only PRAGMAs for count: {}", e);
                return 0usize;
            }

            read_conn
                .query_row("SELECT COUNT(*) FROM flight_records", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap_or(0) as usize
        }).await.unwrap_or(0)
    }

    pub async fn export_json(&self) -> Result<String, serde_json::Error> {
        let records = self
            .query(&RecordQuery {
                limit: Some(10_000),
                ..Default::default()
            })
            .await;
        serde_json::to_string_pretty(&records)
    }
}

impl Drop for FlightRecorder {
    fn drop(&mut self) {
        let _ = self.tx.try_send(FlightRecorderCommand::Shutdown);
    }
}

/// TMR consensus adjudication event for the SSD training corpus pipeline.
///
/// Captured only when 2+ nodes agree (2/3 quorum). Written as JSONL for
/// downstream self-supervised distillation (SSD) model training.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConsensusEvent {
    pub timestamp: DateTime<Utc>,
    pub request_id: String,
    pub tenant_id: String,
    pub prompt: String,
    pub inference_params: serde_json::Value,
    pub agreeing_nodes: u8,
    pub total_nodes: u8,
    pub adjudication_logic: String,
    pub consensus_payload: serde_json::Value,
    pub provider_results: Vec<ProviderResultEntry>,
    pub fallback_activated: bool,
}

/// Per-provider result entry within a consensus event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderResultEntry {
    pub provider: String,
    pub status: String,
    pub canonical_key: Option<String>,
}

/// Async flight recorder that writes consensus events to a rotating JSONL corpus.
///
/// Uses a Tokio MPSC channel with a background writer task that rotates
/// output files at UTC midnight. Designed for the SSD (Self-Supervised
/// Distillation) training pipeline - events are only recorded when 2/3
/// quorum is achieved.
pub struct ConsensusFlightRecorder {
    corpus_path: PathBuf,
    tx: tokio::sync::mpsc::Sender<String>,
}

impl ConsensusFlightRecorder {
    pub async fn new(corpus_path: Option<PathBuf>) -> Result<Self, FlightRecorderError> {
        let path = corpus_path.unwrap_or_else(|| {
            PathBuf::from("./serein-telemetry/ssd_training_corpus.jsonl")
        });

        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| FlightRecorderError::DatabaseOpen(
                    rusqlite::Error::InvalidParameterName(format!(
                        "Failed to create telemetry directory {:?}: {}", parent, e
                    ))
                ))?;
        }

        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(10000);

        let path_display = path.display().to_string();
        let path_clone = path.clone();
        tokio::spawn(async move {
            use tokio::io::AsyncWriteExt;
            let mut current_date = chrono::Utc::now().format("%Y-%m-%d").to_string();

            let parent_dir = match path_clone.parent() {
                Some(p) => p,
                None => {
                    tracing::error!("[FLIGHT RECORDER] Log path has no parent directory - writer task aborting to prevent panic");
                    return;
                }
            };
            let file_path = parent_dir.join(format!("ssd_training_corpus_{}.jsonl", current_date));

            let file = match tokio::fs::OpenOptions::new().create(true).append(true).open(&file_path).await {
                Ok(f) => f,
                Err(e) => {
                    tracing::error!("[FLIGHT RECORDER] Failed to open corpus file {:?}: {}. Writer task aborting to prevent panic.", file_path, e);
                    return;
                }
            };
            let mut writer = tokio::io::BufWriter::with_capacity(128 * 1024, file);

            loop {
                match rx.recv().await {
                    Some(json_line) => {
                        let now_date = chrono::Utc::now().format("%Y-%m-%d").to_string();
                        if now_date != current_date {
                            let new_file_path = parent_dir.join(format!("ssd_training_corpus_{}.jsonl", now_date));

                            match tokio::fs::OpenOptions::new().create(true).append(true).open(&new_file_path).await {
                                Ok(new_file) => {
                                    let _ = writer.flush().await;
                                    writer = tokio::io::BufWriter::with_capacity(128 * 1024, new_file);
                                    current_date = now_date;
                                }
                                Err(e) => {
                                    tracing::error!("[FLIGHT RECORDER] Log rotation failed: {}. Continuing with existing file to prevent data loss.", e);
                                }
                            }
                        }
                        let line_with_newline = format!("{}\n", json_line);
                        let _ = writer.write_all(line_with_newline.as_bytes()).await;
                    }
                    None => {
                        let _ = writer.flush().await;
                        return;
                    }
                }
            }
        });

        info!(
            corpus_path = %path_display,
            "[FLIGHT RECORDER] Consensus flight recorder initialized - mpsc channel + background Tokio writer active"
        );

        Ok(Self {
            corpus_path: path,
            tx,
        })
    }

    pub async fn record_consensus_event(&self, event: ConsensusEvent) {
        if event.agreeing_nodes < 2 {
            warn!(
                request_id = %event.request_id,
                agreeing_nodes = event.agreeing_nodes,
                total_nodes = event.total_nodes,
                "[FLIGHT RECORDER] Consensus event rejected - no 2/3 majority achieved"
            );
            return;
        }

        let json_line = match serde_json::to_string(&event) {
            Ok(j) => j,
            Err(e) => {
                error!(
                    request_id = %event.request_id,
                    error = %e,
                    "[FLIGHT RECORDER] Failed to serialize consensus event to JSON"
                );
                return;
            }
        };

        if let Err(e) = self.tx.try_send(json_line) {
            match e {
                tokio::sync::mpsc::error::TrySendError::Full(_) => {
                    static DROP_COUNT: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(0);
                    let count =
                        DROP_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if count.is_multiple_of(1000) {
                        tracing::error!(
                            drop_count = count + 1,
                            "[CRITICAL] Telemetry buffer full! Over 1000 events dropped. Increase CHANNEL_BOUND or check disk I/O."
                        );
                    }
                }
                tokio::sync::mpsc::error::TrySendError::Closed(_) => {
                    error!(
                        "[FLIGHT RECORDER] Consensus event channel closed - background writer terminated"
                    );
                }
            }
        } else {
            info!(
                request_id = %event.request_id,
                agreeing_nodes = event.agreeing_nodes,
                total_nodes = event.total_nodes,
                "[FLIGHT RECORDER] TMR consensus event dispatched to background writer"
            );
        }
    }

    pub fn corpus_path(&self) -> &PathBuf {
        &self.corpus_path
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}... [truncated, {} bytes]", &s[..max_len], s.len())
    }
}
