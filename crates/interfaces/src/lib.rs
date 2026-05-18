// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WIT interface definitions for the Serein WASI 0.3 component model.
//!
//! Consumed by workspace crates via `wit-bindgen`. No Rust source - only `.wit`
//! files under this package define the component-model contracts.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// Canonical strategy for TMR adjudication of non-identical LLM JSON outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TmrCanonicalStrategy {
    /// Strict comparison: requires exact canonical key match for consensus.
    Strict,
    /// Fuzzy comparison: allows minor structural differences in JSON output.
    Fuzzy,
}

impl TmrCanonicalStrategy {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "fuzzy" => TmrCanonicalStrategy::Fuzzy,
            _ => TmrCanonicalStrategy::Strict,
        }
    }
}

impl std::fmt::Display for TmrCanonicalStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TmrCanonicalStrategy::Strict => write!(f, "strict"),
            TmrCanonicalStrategy::Fuzzy => write!(f, "fuzzy"),
        }
    }
}

/// Canonical intermediate payload extracted from LLM output for TMR consensus comparison.
///
/// Supports both camelCase (LLM-native) and snake_case (Rust-native) field aliases
/// for resilient deserialization across heterogeneous provider outputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct IntermediatePayload {
    #[serde(alias = "networkId", alias = "network_id")]
    pub network_id: String,
    #[serde(alias = "taskType", alias = "task_type")]
    pub task_type: String,
    #[serde(alias = "maxGasLimit", alias = "max_gas_limit")]
    pub max_gas_limit: u32,
    #[serde(alias = "confidenceScore", alias = "confidence_score")]
    pub confidence_score: f64,
    #[serde(alias = "sourceUrl", alias = "source_url")]
    pub source_url: String,
}

impl IntermediatePayload {
    pub fn from_llm_output(raw: &str) -> Result<Self, String> {
        let sanitized = sanitize_llm_response(raw);
        let mut bytes = sanitized.into_bytes();
        simd_json::from_slice::<Self>(&mut bytes)
            .or_else(|_| serde_json::from_str(&String::from_utf8_lossy(&bytes)))
            .map_err(|e| format!("Failed to parse payload: {}", e))
    }

    pub fn canonical_key(&self) -> String {
        format!(
            "NETWORK:{}:TASK:{}:GAS:{}",
            self.network_id.to_lowercase().trim(),
            self.task_type.to_lowercase().trim(),
            self.max_gas_limit
        )
    }
}

/// Strip markdown code fences from LLM output before JSON parsing.
///
/// LLMs frequently wrap JSON in ```json...``` blocks. This function
/// extracts the inner content for clean deserialization.
///
/// Handles multiple variations:
/// - Fenced blocks with language tag: ```json { ... } ```
/// - Bare fenced blocks: ``` { ... } ```
/// - Leading/trailing whitespace and stray newlines
pub fn sanitize_llm_response(response: &str) -> String {
    static FENCE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"```(?:json)?\s*([\s\S]*?)\s*```").unwrap()
    });

    let extracted = if let Some(caps) = FENCE_REGEX.captures(response) {
        caps.get(1)
            .map_or_else(|| response.to_string(), |m: regex::Match| m.as_str().to_string())
    } else {
        response.to_string()
    };

    let trimmed = extracted.trim().to_string();

    static NESTED_FENCE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"```(?:json)?\s*([\s\S]*?)\s*```").unwrap()
    });

    if let Some(caps) = NESTED_FENCE_REGEX.captures(&trimmed) {
        caps.get(1)
            .map_or(trimmed.clone(), |m: regex::Match| m.as_str().trim().to_string())
    } else {
        trimmed
    }
}
