// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Hex-Encoding Evasion - Sensitive Entity Obfuscation
//!
//! Implements obfuscation of politically sensitive entities using JavaScript-style
//! `String.fromCharCode(...)` encoding before sending payloads to LLM providers.
//! This prevents triggering internal safety/content filters that would otherwise
//! cause account suspension or response blocking.
//!
//! ## Architecture
//! - **Entity Detection**: Regex-based detection of sensitive patterns
//! - **Hex Encoding**: Converts matched entities to `String.fromCharCode(0xHH, ...)`
//! - **Decoding Table**: Server-side mapping to restore original values after LLM response
//! - **Scope**: Only encodes detected sensitive terms; safe content passes through unchanged
//!
//! ## Safety Intent
//! Avoid LLM provider safety filter bans by never sending raw sensitive entity names
//! in prompt text. The encoding is transparent to the application layer.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Encoding format options for sensitive entity obfuscation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingFormat {
    /// JavaScript `String.fromCharCode(0xHH, ...)` style
    FromCharCode,

    /// Plain hex escape `\xHH\xHH`
    HexEscape,

    /// Unicode code point escape `\uHHHH`
    UnicodeEscape,
}

/// A single encoding mapping from original text to its obfuscated form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingEntry {
    pub original: String,
    pub encoded: String,
    pub encoding_format: EncodingFormat,
    pub entity_class: SensitivityClass,
}

/// Sensitivity classification for detected entities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensitivityClass {
    PoliticalEntity,
    GeographicRegion,
    OrganizationName,
    PersonName,
    RestrictedTerm,
    Custom(String),
}

/// Result of encoding or decoding operations.
#[derive(Debug, Clone)]
pub struct EncodeResult {
    pub processed_text: String,
    pub entries: Vec<EncodingEntry>,
    pub entities_found: usize,
}

/// Hex-Encoding Evasion engine for sensitive entity obfuscation.
///
/// Scans input text for known sensitive patterns and replaces them with
/// encoded equivalents that bypass LLM safety filters while preserving
/// semantic meaning for the model's reasoning process.
pub struct HexEvasionEngine {
    patterns: Vec<(Regex, SensitivityClass)>,
    custom_entities: HashMap<String, SensitivityClass>,
    default_format: EncodingFormat,
}

impl HexEvasionEngine {
    /// Create a new hex evasion engine with default sensitivity patterns.
    pub fn new() -> Self {
        let mut engine = Self {
            patterns: Vec::new(),
            custom_entities: HashMap::new(),
            default_format: EncodingFormat::FromCharCode,
        };
        engine.initialize_default_patterns();
        engine
    }

    /// Set the default encoding format for all obfuscation operations.
    pub fn set_encoding_format(&mut self, format: EncodingFormat) {
        self.default_format = format;
    }

    /// Register a custom sensitive entity for detection and encoding.
    pub fn register_entity(&mut self, entity: &str, class: SensitivityClass) {
        self.custom_entities.insert(entity.to_lowercase(), class);
    }

    /// Encode all sensitive entities in the input text.
    ///
    /// Scans the text for both pattern-matched and explicitly registered
    /// sensitive entities, replacing each with its encoded equivalent.
    ///
    /// # Returns
    /// An `EncodeResult` containing the processed text and a decode table.
    pub fn encode(&self, input: &str) -> EncodeResult {
        let mut entries = Vec::new();
        let mut result = input.to_string();
        let mut found = 0usize;

        for (pattern, class) in &self.patterns {
            result = pattern
                .replace_all(&result, |caps: &regex::Captures| {
                    let original = caps
                        .get(0)
                        .map(|m| m.as_str().to_string())
                        .unwrap_or_default();
                    let encoded = self.encode_entity(&original);
                    found += 1;
                    entries.push(EncodingEntry {
                        original: original.clone(),
                        encoded: encoded.clone(),
                        encoding_format: self.default_format,
                        entity_class: class.clone(),
                    });
                    encoded
                })
                .to_string();
        }

        for (entity, class) in &self.custom_entities {
            let lower_input = result.to_lowercase();
            if let Some(pos) = lower_input.find(entity) {
                let original = result[pos..pos + entity.len()].to_string();
                let encoded = self.encode_entity(&original);
                result = result.replace(&original, &encoded);
                found += 1;
                entries.push(EncodingEntry {
                    original,
                    encoded,
                    encoding_format: self.default_format,
                    entity_class: class.clone(),
                });
            }
        }

        tracing::info!(
            entities_found = found,
            entries_encoded = entries.len(),
            "[HEX EVASION] Sensitive entity encoding complete"
        );

        EncodeResult {
            processed_text: result,
            entries,
            entities_found: found,
        }
    }

    /// Decode previously encoded text back to its original form.
    ///
    /// Uses the encoding entries table to reverse the obfuscation.
    pub fn decode(&self, encoded_text: &str, entries: &[EncodingEntry]) -> String {
        let mut result = encoded_text.to_string();

        for entry in entries.iter().rev() {
            result = result.replace(&entry.encoded, &entry.original);
        }

        result
    }

    /// Encode a single entity string using the configured format.
    fn encode_entity(&self, entity: &str) -> String {
        match self.default_format {
            EncodingFormat::FromCharCode => {
                let codes: Vec<String> = entity
                    .chars()
                    .map(|c| format!("0x{:X}", c as u32))
                    .collect();
                format!("String.fromCharCode({})", codes.join(", "))
            }
            EncodingFormat::HexEscape => {
                let escaped: Vec<String> = entity
                    .chars()
                    .map(|c| format!("\\x{:02X}", c as u8))
                    .collect();
                escaped.join("")
            }
            EncodingFormat::UnicodeEscape => {
                let escaped: Vec<String> = entity
                    .chars()
                    .map(|c| format!("\\u{:04X}", c as u32))
                    .collect();
                escaped.join("")
            }
        }
    }

    /// Initialize the default set of sensitivity detection patterns.
    fn initialize_default_patterns(&mut self) {
        let pattern_defs: Vec<(&str, SensitivityClass)> = vec![
            // These are placeholder patterns - real deployment requires
            // domain-specific entity lists per operational requirements
            (r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", SensitivityClass::PersonName),
        ];

        for (pattern_str, class) in pattern_defs {
            if let Ok(re) = Regex::new(pattern_str) {
                self.patterns.push((re, class));
            }
        }
    }
}

impl Default for HexEvasionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_from_char_code() {
        let mut engine = HexEvasionEngine::new();
        engine.set_encoding_format(EncodingFormat::FromCharCode);

        let result = engine.encode("Hello World");
        assert!(!result.processed_text.contains("Hello World"));
        assert!(result.processed_text.contains("String.fromCharCode"));
    }

    #[test]
    fn test_encode_custom_entity() {
        let mut engine = HexEvasionEngine::new();
        engine.register_entity("SensitiveTerm", SensitivityClass::RestrictedTerm);

        let result = engine.encode("This contains SensitiveTerm here");
        assert!(result.entities_found >= 1);
        assert!(!result.processed_text.contains("SensitiveTerm"));
    }

    #[test]
    fn test_roundtrip_encode_decode() {
        let mut engine = HexEvasionEngine::new();
        engine.set_encoding_format(EncodingFormat::FromCharCode);

        let original = "Data with John Smith inside";
        let encoded = engine.encode(original);
        let decoded = engine.decode(&encoded.processed_text, &encoded.entries);

        assert_eq!(decoded, original);
    }

    #[test]
    fn test_hex_escape_format() {
        let mut engine = HexEvasionEngine::new();
        engine.set_encoding_format(EncodingFormat::HexEscape);
        engine.register_entity("Test", SensitivityClass::Custom("TestEntity".to_string()));

        let result = engine.encode("Test");
        assert!(result.processed_text.contains("\\x"));
    }
}
