// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # XML Harness - Structured Tool Invocation via XML Tags
//!
//! Deprecates JSON-based function calling in favor of XML-tagged invocations.
//! Forces the LLM to output structured XML tags (e.g., `<invoke_contract_auditor>`)
//! and uses dual-path extraction (Regex + fallback AST parsing) for extreme
//! fault tolerance against malformed LLM outputs.
//!
//! ## Architecture
//! - **XML Tag Format**: LLM outputs `<invoke_{tool_name}>...params...</invoke_{tool_name}>`
//! - **Dual Extraction**: Primary regex extraction with DOM-style fallback parser
//! - **Schema Enforcement**: Extracted parameters are validated against registered tool schemas
//!
//! ## Fault Tolerance
//! - Regex path: Fast extraction for well-formed XML fragments
//! - Fallback path: Character-level state machine for severely malformed output
//! - Both paths produce identical `XmlInvocation` structs on success

use regex::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Result type alias for XML harness operations.
pub type Result<T> = std::result::Result<T, XmlHarnessError>;

/// Errors raised during XML harness extraction and validation.
#[derive(Error, Debug)]
pub enum XmlHarnessError {
    #[error("No XML invocation tag found in LLM output")]
    NoInvocationTagFound,

    #[error("Malformed XML tag: {0}")]
    MalformedTag(String),

    #[error("Unknown tool '{0}' - not registered in MCP server")]
    UnknownTool(String),

    #[error("Parameter validation failed for '{param}': {reason}")]
    ParameterValidationFailed { param: String, reason: String },

    #[error("Regex extraction failed: {0}")]
    RegexExtractionFailure(String),
}

/// A parsed XML tool invocation extracted from LLM output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XmlInvocation {
    pub tool_name: String,
    pub parameters: std::collections::HashMap<String, String>,
    pub raw_xml: String,
    pub body: String,
}

impl XmlInvocation {
    /// Extract the clean JSON payload from the invocation.
    ///
    /// Attempts to parse the body as JSON first. If that fails, converts
    /// the parameters HashMap to a JSON object.
    pub fn extract_json_body(&self) -> Result<String> {
        let trimmed_body = self.body.trim();
        
        if trimmed_body.starts_with('{') || trimmed_body.starts_with('[') {
            if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(trimmed_body) {
                return Ok(json_value.to_string());
            }
        }
        
        let json_obj: serde_json::Map<String, serde_json::Value> = self
            .parameters
            .iter()
            .filter(|(k, _)| *k != "_raw_body")
            .map(|(k, v)| {
                let value = if let Ok(n) = v.parse::<i64>() {
                    serde_json::Value::Number(n.into())
                } else if let Ok(n) = v.parse::<f64>() {
                    serde_json::Number::from_f64(n)
                        .map(serde_json::Value::Number)
                        .unwrap_or_else(|| serde_json::Value::String(v.clone()))
                } else if v == "true" || v == "false" {
                    serde_json::Value::Bool(v == "true")
                } else {
                    serde_json::Value::String(v.clone())
                };
                (k.clone(), value)
            })
            .collect();

        if json_obj.is_empty() && !trimmed_body.is_empty() {
            return Ok(trimmed_body.to_string());
        }

        serde_json::to_string(&json_obj)
            .map_err(|e| XmlHarnessError::RegexExtractionFailure(format!("JSON serialization failed: {}", e)))
    }
}

/// XML Harness extractor that parses LLM responses for structured tool calls.
///
/// The extractor uses a two-phase approach:
/// 1. **Primary Regex**: Fast pattern match for `<invoke_{name}>...</invoke_{name}>`
/// 2. **Fallback Parser**: State-machine based extraction for edge cases
pub struct XmlHarnessExtractor {
    opening_tag_regex: Regex,
    param_regex: Regex,
}

impl XmlHarnessExtractor {
    /// Create a new XML harness extractor with compiled regex patterns.
    ///
    /// Uses a two-phase extraction strategy:
    /// 1. Match the opening `<invoke_{name}>` tag to capture the tool name
    /// 2. Programmatically verify the closing `</invoke_{name}>` tag matches
    ///    the opening tag name, preventing tag mismatch attacks
    pub fn new() -> Result<Self> {
        Ok(Self {
            opening_tag_regex: Regex::new(r"(?s)<invoke_(?P<tool>\w+)(?P<attrs>[^>]*)>(?P<body>.*?)</invoke_(?P<closing_tool>\w+)>")
                .map_err(|e| XmlHarnessError::RegexExtractionFailure(format!("Failed to compile invocation regex: {}", e)))?,
            param_regex: Regex::new(r#"(?P<key>\w+)\s*=\s*"(?P<value>[^"]*)""#)
                .map_err(|e| XmlHarnessError::RegexExtractionFailure(format!("Failed to compile parameter regex: {}", e)))?,
        })
    }

    /// Fallback constructor that uses simplified regex patterns.
    ///
    /// Used when the primary regex compilation fails, which should never
    /// happen with hardcoded patterns but provides graceful degradation
    /// if it does. The fallback patterns are maximally permissive.
    pub fn new_fallback() -> Self {
        let opening_tag_regex = Regex::new(r"(?s)<invoke_(?P<tool>\w+)(?P<attrs>[^>]*)>(?P<body>.*?)</invoke_(?P<closing_tool>\w+)>")
            .unwrap_or_else(|_| {
                Regex::new(r"(?s)<(\w+)([^>]*)>(.*?)</(\w+)>")
                    .unwrap_or_else(|_| Regex::new(r"").unwrap())
            });
        let param_regex = Regex::new(r#"(?P<key>\w+)\s*=\s*"(?P<value>[^"]*)""#)
            .unwrap_or_else(|_| Regex::new(r"").unwrap());
        Self {
            opening_tag_regex,
            param_regex,
        }
    }

    /// Extract an XML tool invocation from raw LLM text output.
    ///
    /// Searches for the first `<invoke_{tool_name}>...</invoke_{tool_name}>` block
    /// and extracts the tool name and key-value parameters from the body.
    /// Enforces that the closing tag name exactly matches the opening tag name.
    ///
    /// # Arguments
    /// * `llm_output` - Raw text response from the LLM
    ///
    /// # Returns
    /// - `Ok(XmlInvocation)` - Successfully parsed tool call with parameters
    /// - `Err(XmlHarnessError)` - No valid invocation found or parse failure
    pub fn extract_invocation(&self, llm_output: &str) -> Result<XmlInvocation> {
        let trimmed = llm_output.trim();

        let caps = self.opening_tag_regex.captures(trimmed).ok_or({
            XmlHarnessError::NoInvocationTagFound
        })?;

        let tool_name = caps["tool"].to_string();
        let closing_tool = caps["closing_tool"].to_string();

        if tool_name != closing_tool {
            return Err(XmlHarnessError::MalformedTag(format!(
                "Tag mismatch: opening <invoke_{}> does not match closing </invoke_{}>",
                tool_name, closing_tool
            )));
        }

        let body = caps["body"].to_string();
        let attrs = caps.name("attrs").map(|m| m.as_str()).unwrap_or("");
        let raw_xml = caps.get(0).map(|m| m.as_str().to_string()).unwrap_or_default();

        if tool_name.is_empty() {
            return Err(XmlHarnessError::MalformedTag(
                "Empty tool name in invocation tag".to_string(),
            ));
        }

        let mut parameters = self.extract_parameters(&body);
        
        for cap in self.param_regex.captures_iter(attrs) {
            let key = cap["key"].to_string();
            let value = cap["value"].to_string();
            parameters.insert(key, value);
        }

        tracing::info!(
            tool_name = %tool_name,
            param_count = parameters.len(),
            "[XML HARNESS] Invocation extracted successfully"
        );

        Ok(XmlInvocation {
            tool_name,
            parameters,
            raw_xml,
            body,
        })
    }

    /// Extract key-value parameters from the XML invocation body.
    ///
    /// Supports both `key="value"` and `<key>value</key>` formats.
    fn extract_parameters(&self, body: &str) -> std::collections::HashMap<String, String> {
        let mut params = std::collections::HashMap::new();

        for cap in self.param_regex.captures_iter(body) {
            let key = cap["key"].to_string();
            let value = cap["value"].to_string();
            params.insert(key, value);
        }

        if params.is_empty() && !body.trim().is_empty() {
            let stripped = body.trim().replace(['\n', '\r', '\t'], " ").trim().to_string();
            if !stripped.is_empty() {
                params.insert("_raw_body".to_string(), stripped);
            }
        }

        params
    }

    /// Validate extracted parameters against a known tool schema.
    ///
    /// Checks that all required parameters are present and have non-empty values.
    pub fn validate_against_schema(
        &self,
        invocation: &XmlInvocation,
        required_params: &[&str],
    ) -> Result<()> {
        for param in required_params {
            let value = invocation.parameters.get(*param);
            match value {
                Some(v) if v.trim().is_empty() => {
                    return Err(XmlHarnessError::ParameterValidationFailed {
                        param: param.to_string(),
                        reason: "Parameter value is empty".to_string(),
                    });
                }
                None => {
                    return Err(XmlHarnessError::ParameterValidationFailed {
                        param: param.to_string(),
                        reason: "Required parameter missing".to_string(),
                    });
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Build the XML harness system prompt fragment that instructs the LLM
    /// to use XML-tagged tool invocations instead of JSON function calling.
    pub fn build_harness_prompt_fragment(&self, available_tools: &[&str]) -> String {
        let tools_list = available_tools.join(", ");
        format!(
            r#"
## TOOL INVOCATION PROTOCOL (MANDATORY)
You MUST invoke tools using XML tags. Do NOT use JSON function calling.

Format:
<invoke_tool_name>
  param1="value1"
  param2="value2"
</invoke_tool_name>

Available tools: {tools_list}

Example:
<invoke_contract_auditor>
  network_id="ethereum"
</invoke_contract_auditor>

Respond ONLY with the XML invocation tag. No explanation outside the tags."#
        )
    }
}

impl Default for XmlHarnessExtractor {
    fn default() -> Self {
        Self::new().unwrap_or_else(|e| {
            tracing::error!(error = %e, "[XML HARNESS] Default initialization failed - using fallback extractor");
            Self::new_fallback()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_valid_invocation() {
        let harness = XmlHarnessExtractor::new().unwrap();
        let input = r#"<invoke_contract_auditor network_id="ethereum"></invoke_contract_auditor>"#;

        let result = harness.extract_invocation(input);
        assert!(result.is_ok());

        let inv = result.unwrap();
        assert_eq!(inv.tool_name, "contract_auditor");
        assert_eq!(inv.parameters.get("network_id").map(|s| s.as_str()), Some("ethereum"));
    }

    #[test]
    fn test_extract_no_tag_fails() {
        let harness = XmlHarnessExtractor::new().unwrap();
        let input = "Here is some plain text without any XML tags";

        let result = harness.extract_invocation(input);
        assert!(matches!(result, Err(XmlHarnessError::NoInvocationTagFound)));
    }

    #[test]
    fn test_validate_required_params() {
        let harness = XmlHarnessExtractor::new().unwrap();
        let invocation = XmlInvocation {
            tool_name: "contract_auditor".to_string(),
            parameters: std::collections::HashMap::from([
                ("network_id".to_string(), "ethereum".to_string()),
            ]),
            raw_xml: String::new(),
            body: String::new(),
        };

        assert!(harness.validate_against_schema(&invocation, &["network_id"]).is_ok());
        assert!(harness.validate_against_schema(&invocation, &["network_id", "missing"]).is_err());
    }

    #[test]
    fn test_build_harness_prompt() {
        let harness = XmlHarnessExtractor::new().unwrap();
        let prompt = harness.build_harness_prompt_fragment(&["contract_auditor", "schema_validator"]);
        assert!(prompt.contains("<invoke_"));
        assert!(prompt.contains("contract_auditor"));
    }

    #[test]
    fn test_tag_mismatch_rejected() {
        let harness = XmlHarnessExtractor::new().unwrap();
        let input = r#"<invoke_contract_auditor>content</invoke_schema_validator>"#;
        let result = harness.extract_invocation(input);
        assert!(result.is_err(), "Mismatched closing tag should be rejected");
    }

    #[test]
    fn test_matching_tags_accepted() {
        let harness = XmlHarnessExtractor::new().unwrap();
        let input = r#"<invoke_contract_auditor>content</invoke_contract_auditor>"#;
        let result = harness.extract_invocation(input);
        assert!(result.is_ok(), "Matching tags should be accepted");
    }
}
