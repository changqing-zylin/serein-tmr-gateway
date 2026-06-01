// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # MCP Server - Model Context Protocol Implementation
//!
//! Implements dynamic tool discovery (`list_tools`) per the Model Context Protocol (MCP).
//! Provides a registry-based tool system that allows LLMs to discover and invoke
//! available capabilities at runtime without hardcoded tool definitions.
//!
//! ## Architecture
//! - Tool Registry: In-memory `HashMap` of registered tools with schema metadata
//! - Dynamic Discovery: `list_tools` returns all available tools with parameter schemas
//! - Thread-Safe: Uses `Arc<Mutex>` for concurrent access from async workers
//!
//! ## Safety Intent
//! Prevent tool invocation of unregistered or unknown functions by enforcing
//! a strict allowlist model where only pre-registered tools can be discovered.

use serde::{Deserialize, Serialize};
use std::sync::Mutex;

/// Parameter definition for an MCP-registered tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub param_type: String,
    pub required: bool,
    pub description: String,
}

/// A single MCP tool with its metadata and parameter schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub parameters: Vec<ToolParameter>,
}

/// Response payload for the `list_tools` MCP endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListToolsResponse {
    pub tools: Vec<McpTool>,
}

/// MCP Server implementing dynamic tool discovery.
///
/// Tools are registered at startup via `register_tool` and exposed to
/// LLM clients through the `list_tools` method. This implements the
/// server-side half of the Model Context Protocol's tool discovery flow.
pub struct McpServer {
    server_name: String,
    tools: Mutex<Vec<McpTool>>,
}

impl McpServer {
    /// Create a new MCP server instance with the given name and version.
    pub fn new(server_name: &str, _server_version: &str) -> Self {
        Self {
            server_name: server_name.to_string(),
            tools: Mutex::new(Vec::new()),
        }
    }

    /// Register a tool into the MCP server's discovery registry.
    ///
    /// Once registered, the tool will appear in `list_tools` responses
    /// and become discoverable by connected LLM clients.
    pub fn register_tool(&mut self, tool: McpTool) {
        let mut tools = self.tools.lock().unwrap_or_else(|e| e.into_inner());
        tracing::info!(
            tool_name = %tool.name,
            "[MCP SERVER] Tool registered for dynamic discovery"
        );
        tools.push(tool);
    }

    /// Execute the `list_tools` MCP method and return all registered tools.
    ///
    /// Returns a `ListToolsResponse` containing the full tool catalog
    /// with parameter schemas for each registered tool.
    pub fn list_tools(&self) -> ListToolsResponse {
        let tools = self.tools.lock().unwrap_or_else(|e| e.into_inner());
        tracing::debug!(
            tool_count = tools.len(),
            server_name = %self.server_name,
            "[MCP SERVER] list_tools invoked"
        );
        ListToolsResponse {
            tools: tools.clone(),
        }
    }

    /// Look up a specific tool by name from the registry.
    ///
    /// Returns `None` if no tool with the given name has been registered.
    pub fn get_tool(&self, name: &str) -> Option<McpTool> {
        let tools = self.tools.lock().unwrap_or_else(|e| e.into_inner());
        tools.iter().find(|t| t.name == name).cloned()
    }

    /// Return the total count of registered tools.
    pub fn tool_count(&self) -> usize {
        let tools = self.tools.lock().unwrap_or_else(|e| e.into_inner());
        tools.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_server_registration_and_discovery() {
        let mut server = McpServer::new("test-server", "0.1.0");

        assert_eq!(server.tool_count(), 0);

        server.register_tool(McpTool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            parameters: vec![],
        });

        assert_eq!(server.tool_count(), 1);

        let response = server.list_tools();
        assert_eq!(response.tools.len(), 1);
        assert_eq!(response.tools[0].name, "test_tool");
    }

    #[test]
    fn test_mcp_server_get_tool() {
        let mut server = McpServer::new("test-server", "0.1.0");

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
                ToolParameter {
                    name: "contract_address".to_string(),
                    param_type: "string".to_string(),
                    required: false,
                    description: "Smart contract address to audit".to_string(),
                },
            ],
        });

        server.register_tool(McpTool {
            name: "schema_validator".to_string(),
            description: "Validate transaction schema against network rules".to_string(),
            parameters: vec![
                ToolParameter {
                    name: "network_id".to_string(),
                    param_type: "string".to_string(),
                    required: true,
                    description: "Blockchain network identifier".to_string(),
                },
                ToolParameter {
                    name: "tx_hash".to_string(),
                    param_type: "string".to_string(),
                    required: false,
                    description: "Transaction hash to validate".to_string(),
                },
            ],
        });

        let tool = server.get_tool("contract_auditor");
        assert!(tool.is_some());
        assert_eq!(tool.unwrap().parameters.len(), 2);

        let tool2 = server.get_tool("schema_validator");
        assert!(tool2.is_some());

        assert!(server.get_tool("nonexistent").is_none());
    }

    #[test]
    fn test_mcp_list_tools_multiple() {
        let mut server = McpServer::new("multi-tool", "1.0.0");

        for i in 0..5 {
            server.register_tool(McpTool {
                name: format!("tool_{}", i),
                description: format!("Tool number {}", i),
                parameters: vec![],
            });
        }

        let response = server.list_tools();
        assert_eq!(response.tools.len(), 5);
    }
}
