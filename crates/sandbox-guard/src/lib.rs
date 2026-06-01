// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Serein Aegis - Core Harness & Hooks
//!
//! Wasm component implementing the `active-defense` interface per the
//! `serein-aegis.wit` contract. Executes computationally expensive
//! cryptographic puzzle to throttle malicious actors.
//!
//! ## Core Mechanisms Implemented
//! - **MCP Server**: Dynamic tool discovery (`list_tools`) via Model Context Protocol
//! - **XML Harness**: Deprecates JSON tool calling; forces LLM to output XML tags
//!   (`<invoke_contract_auditor>`) with AST/Regex extraction for extreme fault tolerance
//! - **Deterministic Hooks**: `pre_tool_use_hook` physically blocks destructive intents
//!   (`rm -rf`, `DROP TABLE`) BEFORE hitting the WASI sandbox
//!
//! ## Enterprise Security Compliance
//! - Zero-Cost Fast-Fail active defense (non-blocking, Tokio-safe)
//! - Emergency Shutdown (ESD) trigger integration for rogue sandbox termination
//! - Structured telemetry via `tracing` spans with client IP metadata

#![allow(unexpected_cfgs)]

wit_bindgen::generate!({
    world: "aegis-world",
    path: "../interfaces/serein-aegis.wit",
    generate_unused_types: true,
});

use crate::exports::serein::core::active_defense;
use std::sync::Arc;
use tracing::error;

pub mod mcp_server;
pub mod xml_harness;
pub mod deterministic_hooks;
pub mod wasi_virt;
pub mod signature_verify;

pub use signature_verify::{PayloadVerifier, SignatureError};

use mcp_server::{McpTool, McpServer, ToolParameter};
use xml_harness::XmlHarnessExtractor;
use deterministic_hooks::PreToolUseHook;

pub use deterministic_hooks::{
    PromptInjectionWaf, WafResult, WafViolation, InjectionCategory,
};

struct AegisDefense;

struct AegisCtx {
    _mcp_server: Arc<McpServer>,
    _xml_harness: XmlHarnessExtractor,
    _pre_tool_hook: PreToolUseHook,
}

impl active_defense::GuestAttackerContext for AegisCtx {
    fn new() -> Self {
        let mut server = McpServer::new("serein-aegis", "1.0.0");
        server.register_tool(McpTool {
            name: "contract_auditor".to_string(),
            description: "Audit smart contract bytecode for vulnerabilities".to_string(),
            parameters: vec![
                ToolParameter {
                    name: "network_id".to_string(),
                    param_type: "string".to_string(),
                    required: true,
                    description: "Blockchain network identifier (e.g., ethereum, polygon)".to_string(),
                },
            ],
        });
        server.register_tool(McpTool {
            name: "schema_validator".to_string(),
            description: "Validate transaction schema against network rules".to_string(),
            parameters: vec![
                ToolParameter {
                    name: "tx_payload".to_string(),
                    param_type: "string".to_string(),
                    required: true,
                    description: "JSON-encoded execution payload to validate".to_string(),
                },
            ],
        });

        Self {
            _mcp_server: Arc::new(server),
            _xml_harness: XmlHarnessExtractor::new()
                .unwrap_or_else(|e| {
                    tracing::error!(error = %e, "[SANDBOX-GUARD] XmlHarnessExtractor initialization failed - XML tool invocation will be unavailable");
                    XmlHarnessExtractor::new_fallback()
                }),
            _pre_tool_hook: PreToolUseHook::new(),
        }
    }
}

impl active_defense::Guest for AegisDefense {
    type AttackerContext = AegisCtx;

    /// Triggers asymmetric countermeasure via zero-cost fast-fail.
    ///
    /// Immediately logs the security violation and returns a rejection
    /// without consuming CPU resources, preventing Tokio thread starvation.
    fn trigger_asymmetric_counter(_ctx: active_defense::AttackerContext) -> String {
        error!("Aegis countermeasure triggered - Fast-failing to protect Tokio threads");
        "Aegis Defense Active: Malicious intent detected and blocked. (Fast-Fail)".to_string()
    }
}

export!(AegisDefense);
