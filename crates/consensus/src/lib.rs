// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Serein Consensus - TMR Arbitration & Anti-Throttling Backoff
//!
//! ## Anti-Ban Infrastructure
//! - **Jittered Backoff**: Randomized exponential backoff to prevent synchronized request spikes
//! - **TMR Consensus**: 2-out-of-3 provider arbitration with graceful degradation

pub mod jittered_backoff;
pub mod tmr_consensus;

pub use jittered_backoff::{
    execute_with_backoff, execute_with_http_backoff, BackoffConfig, HttpBackoffError,
    HttpOutcome, JitterStrategy, JitteredBackoff, RetryableHttpStatus,
};

pub use tmr_consensus::{
    canonical_semantic_key, compute_canonical_hash, ConsensusError, ConsensusResult,
    NodeHealth, ProviderNode, ProviderResponse, ResponseStatus, TmrConfig,
    TmrConsensusEngine,
};
