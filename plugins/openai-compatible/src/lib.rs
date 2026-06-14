// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # OpenAI-Compatible WASM Adapter
//!
//! Reference implementation of the `serein-adapter.wit` contract for any
//! LLM provider that conforms to the standard OpenAI chat completions API
//! (Bearer token authentication, standard JSON request/response schema).
//!
//! ## Capability Boundary
//! This WASM component never opens sockets or performs I/O. It exclusively
//! produces `http-request-spec` records for the host to execute and parses
//! raw response bodies returned by the host. All network operations are
//! governed by the Serein microkernel's capability system.
//!
//! ## Error Handling
//! No `unwrap()` or `expect()` calls exist in this module. All serialization
//! and deserialization failures return precise `String` errors through the
//! `result<_, string>` ABI contract.

use serde::Serialize;

wit_bindgen::generate!({
    path: "../../crates/interfaces/serein-adapter.wit",
    world: "adapter-world",
});

use crate::exports::serein::adapter::api_adapter::{
    Guest, HttpRequestSpec, StandardizedRequest,
};

/// OpenAI chat completions request body - exact API contract.
#[derive(Serialize)]
struct OpenAiChatRequest<'a> {
    model: &'a str,
    messages: [OpenAiMessage<'a>; 1],
    #[serde(skip_serializing_if = "is_zero_f32")]
    temperature: f32,
    stream: bool,
}

#[derive(Serialize)]
struct OpenAiMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// OpenAI chat completions response - only the fields needed for extraction.
#[derive(serde::Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(serde::Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(serde::Deserialize)]
struct OpenAiResponseMessage {
    content: String,
}

fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}

struct OpenAiCompatibleAdapter;

impl Guest for OpenAiCompatibleAdapter {
    /// Transform a canonical TMR request into an OpenAI-compatible HTTP spec.
    ///
    /// Constructs the standard JSON body with `model`, `messages[0].content`,
    /// and optional `temperature`. Sets the `Authorization: Bearer {api_key}`
    /// header and `Content-Type: application/json`.
    ///
    /// # Errors
    /// Returns a `String` error if JSON serialization of the request body fails.
    fn transform_request(
        req: StandardizedRequest,
        api_key: String,
    ) -> Result<HttpRequestSpec, String> {
        let body = OpenAiChatRequest {
            model: &req.model,
            messages: [OpenAiMessage {
                role: "user",
                content: &req.prompt,
            }],
            temperature: req.temperature,
            stream: false,
        };

        let body_json =
            serde_json::to_string(&body).map_err(|e| format!("Failed to serialize request body: {}", e))?;

        let headers: Vec<(String, String)> = vec![
            ("Authorization".to_string(), format!("Bearer {}", api_key)),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];

        let endpoint = if req.base_url.ends_with('/') {
            format!("{}chat/completions", req.base_url)
        } else {
            format!("{}/chat/completions", req.base_url)
        };

        Ok(HttpRequestSpec {
            url: endpoint,
            headers,
            body: body_json,
        })
    }

    /// Parse the raw OpenAI chat completions response body into canonical content.
    ///
    /// Extracts `choices[0].message.content` from the JSON response.
    ///
    /// # Errors
    /// Returns a `String` error if:
    /// - The raw body is not valid JSON
    /// - The JSON structure does not match the expected OpenAI schema
    /// - The `choices` array is empty
    fn parse_response(raw_body: String) -> Result<String, String> {
        let parsed: OpenAiChatResponse = serde_json::from_str(&raw_body)
            .map_err(|e| format!("Failed to parse OpenAI response JSON: {}", e))?;

        parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| "OpenAI response contained no choices".to_string())
    }
}
