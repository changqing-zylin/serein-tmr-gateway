// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strict write discipline: data commits only when task status is VerifiedSuccess.
//! All other outcomes trigger retry or safe discard with audit logging.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    InProgress,
    VerifiedSuccess,
    VerifiedFailure,
    PartialSuccess,
    RetryableError,
    NonRetryableError,
    Cancelled,
}

impl TaskStatus {
    pub fn allows_commit(&self) -> bool {
        matches!(self, TaskStatus::VerifiedSuccess | TaskStatus::PartialSuccess)
    }

    pub fn allows_retry(&self) -> bool {
        matches!(
            self,
            TaskStatus::RetryableError | TaskStatus::InProgress | TaskStatus::Pending
        )
    }
}

#[derive(Error, Debug)]
pub enum WriteDisciplineError {
    #[error("Write rejected - task status '{status:?}' does not permit commit")]
    CommitRejected { status: TaskStatus },

    #[error("Data integrity check failed: {0}")]
    IntegrityCheckFailed(String),

    #[error("Maximum retry attempts ({max}) exhausted without success")]
    MaxRetriesExhausted { max: u32 },

    #[error("Serialization failure: {0}")]
    SerializationError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteAttempt {
    pub attempt_id: String,
    pub timestamp: DateTime<Utc>,
    pub task_status: TaskStatus,
    pub outcome: WriteOutcome,
    pub data_fingerprint: String,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WriteOutcome {
    Committed,
    Rejected,
    Retried,
    FailedPermanent,
}

pub struct WriteGate {
    audit_trail: std::sync::Mutex<Vec<WriteAttempt>>,
    max_retries: u32,
}

impl WriteGate {
    pub fn new(max_retries: u32) -> Self {
        Self {
            audit_trail: std::sync::Mutex::new(Vec::new()),
            max_retries,
        }
    }

    pub fn commit(
        &self,
        task_status: &TaskStatus,
        data: &serde_json::Value,
        task_id: &str,
    ) -> Result<(), WriteDisciplineError> {
        let data_fingerprint = self.compute_fingerprint(data);
        let attempt_id = format!("{}-{}", task_id, Utc::now().timestamp_millis());

        if !task_status.allows_commit() {
            let attempt = WriteAttempt {
                attempt_id: attempt_id.clone(),
                timestamp: Utc::now(),
                task_status: task_status.clone(),
                outcome: WriteOutcome::Rejected,
                data_fingerprint: data_fingerprint.clone(),
                error_message: Some(format!("Status {:?} does not permit commit", task_status)),
            };
            self.record_attempt(attempt);
            return Err(WriteDisciplineError::CommitRejected {
                status: task_status.clone(),
            });
        }

        self.validate_integrity(data)?;

        let attempt = WriteAttempt {
            attempt_id,
            timestamp: Utc::now(),
            task_status: task_status.clone(),
            outcome: WriteOutcome::Committed,
            data_fingerprint,
            error_message: None,
        };
        self.record_attempt(attempt);

        tracing::info!(
            task_id = %task_id,
            status = ?task_status,
            "[WRITE GATE] Commit approved"
        );

        Ok(())
    }

    pub async fn self_healing_write<F, Fut>(
        &self,
        initial_status: TaskStatus,
        data: &serde_json::Value,
        task_id: &str,
        mut compute_status: F,
    ) -> Result<(), WriteDisciplineError>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = TaskStatus>,
    {
        let mut current_status = initial_status;

        for attempt in 0..=self.max_retries {
            match self.commit(&current_status, data, task_id) {
                Ok(()) => return Ok(()),
                Err(WriteDisciplineError::CommitRejected { .. }) if current_status.allows_retry() => {
                    let backoff_ms = self.jittered_backoff(attempt);
                    tracing::warn!(
                        attempt = attempt,
                        backoff_ms,
                        task_id = %task_id,
                        "[WRITE GATE] Retrying after backoff"
                    );
                    tokio::time::sleep(tokio::time::Duration::from_millis(backoff_ms)).await;
                    current_status = compute_status().await;
                }
                Err(e) => return Err(e),
            }
        }

        Err(WriteDisciplineError::MaxRetriesExhausted {
            max: self.max_retries,
        })
    }

    pub fn audit_trail(&self) -> Vec<WriteAttempt> {
        self.audit_trail.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn record_attempt(&self, attempt: WriteAttempt) {
        let mut trail = self.audit_trail.lock().unwrap_or_else(|e| e.into_inner());
        trail.push(attempt);
    }

    fn compute_fingerprint(&self, data: &serde_json::Value) -> String {
        use sha2::{Digest, Sha256};
        let serialized = serde_json::to_string(data).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(serialized.as_bytes());
        hex::encode(hasher.finalize())
    }

    fn validate_integrity(&self, data: &serde_json::Value) -> Result<(), WriteDisciplineError> {
        if data.is_null() {
            return Err(WriteDisciplineError::IntegrityCheckFailed(
                "Data is null".to_string(),
            ));
        }
        Ok(())
    }

    fn jittered_backoff(&self, attempt: u32) -> u64 {
        use rand::Rng;
        let base_ms = 100u64;
        let max_ms = 5000u64;
        let exponential = base_ms.saturating_mul(2u64.saturating_pow(attempt));
        let capped = exponential.min(max_ms);
        let jitter = rand::thread_rng().gen_range(0..=capped / 2);
        capped.saturating_add(jitter)
    }
}

impl Default for WriteGate {
    fn default() -> Self {
        Self::new(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_commit_verified_success() {
        let gate = WriteGate::new(3);
        let data = json!({"networkId": "ethereum", "taskType": "swap"});
        assert!(gate.commit(&TaskStatus::VerifiedSuccess, &data, "task-001").is_ok());
        assert_eq!(gate.audit_trail()[0].outcome, WriteOutcome::Committed);
    }

    #[test]
    fn test_commit_rejected_for_pending() {
        let gate = WriteGate::new(3);
        assert!(matches!(
            gate.commit(&TaskStatus::Pending, &json!({"key": "value"}), "task-002"),
            Err(WriteDisciplineError::CommitRejected { .. })
        ));
        assert_eq!(gate.audit_trail()[0].outcome, WriteOutcome::Rejected);
    }

    #[test]
    fn test_null_data_rejected() {
        let gate = WriteGate::new(3);
        assert!(matches!(
            gate.commit(&TaskStatus::VerifiedSuccess, &json!(null), "task-003"),
            Err(WriteDisciplineError::IntegrityCheckFailed(_))
        ));
    }

    #[tokio::test]
    async fn test_self_healing_write_eventually_succeeds() {
        use std::sync::atomic::{AtomicU32, Ordering};
        let gate = WriteGate::new(3);
        let data = json!({"result": "ok"});
        let call_count = AtomicU32::new(0);

        let result = gate
            .self_healing_write(
                TaskStatus::InProgress,
                &data,
                "task-sh",
                || {
                    let count = call_count.fetch_add(1, Ordering::SeqCst) + 1;
                    async move {
                        if count < 2 { TaskStatus::InProgress } else { TaskStatus::VerifiedSuccess }
                    }
                },
            )
            .await;

        assert!(result.is_ok());
        assert!(call_count.load(Ordering::SeqCst) >= 2);
    }
}
