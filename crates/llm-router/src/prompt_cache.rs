// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Prompt Cache - Ephemeral Context Caching
//!
//! Injects provider-specific cache control directives into LLM API requests
//! for long contextual documents to slash token costs by ~90%.
//!
//! ## Architecture
//! - **Cache Directive Injection**: Wraps system prompts and long documents with
//!   cache control markers supported by each provider
//! - **Content Fingerprinting**: SHA-256 hash of cached content for cache invalidation
//! - **Cost Tracking**: Monitors estimated token savings from cache hits
//! - **Static/Dynamic Separation**: Static content (network rules, system prompts) is
//!   isolated from dynamic user input to maximize cache hit rates
//!
//! ## Provider Support
//! - Google Gemini: `cache_control: {"type": "ephemeral"}` in `contents` array
//! - Anthropic: `cache_control: {"type": "ephemeral"}` on message blocks +
//!   `anthropic-beta: prompt-caching-2024-07-31` header
//! - DeepSeek/OpenAI: `cache_control: {"type": "ephemeral"}` on system message +
//!   custom header support for prefix caching

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// LLM provider type for provider-specific cache directive formatting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provider {
    Gemini,
    Anthropic,
    DeepSeek,
    Groq,
}

impl Provider {
    pub fn supports_prompt_caching(&self) -> bool {
        matches!(self, Provider::Gemini | Provider::Anthropic | Provider::DeepSeek)
    }

    pub fn requires_beta_header(&self) -> bool {
        matches!(self, Provider::Anthropic)
    }
}

/// Cache control configuration for a single content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheDirective {
    #[serde(rename = "type")]
    pub cache_type: String,
}

/// A cached content block with its fingerprint and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedContent {
    pub role: String,
    pub parts: Vec<Value>,
    pub cache_control: Option<CacheDirective>,
    pub content_hash: String,
    pub estimated_tokens: u32,
}

/// Cache-aware HTTP request payload with headers and body.
#[derive(Debug, Clone)]
pub struct CacheAwarePayload {
    pub body: String,
    pub headers: Vec<(String, String)>,
    pub provider: Provider,
}

pub struct CacheAwarePayloadArgs<'a> {
    pub provider: Provider,
    pub system_prompt: &'a CachedContent,
    pub documents: &'a [CachedContent],
    pub user_query: &'a str,
    pub temperature: f32,
    pub max_tokens: u32,
    pub model: &'a str,
}

/// Statistics for prompt cache operations.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheStats {
    pub total_cached_blocks: u64,
    pub total_cache_hits: u64,
    pub estimated_token_savings: u64,
}

/// Prompt cache manager that wraps LLM requests with caching directives.
///
/// Supports provider-specific cache control injection:
/// - **Gemini**: `cache_control: {"type": "ephemeral"}` in `contents` array
/// - **Anthropic**: `cache_control` on message blocks + beta header
/// - **DeepSeek**: `cache_control` on system message for prefix caching
pub struct PromptCacheManager {
    stats: std::sync::Mutex<CacheStats>,
    enabled: bool,
}

impl PromptCacheManager {
    pub fn new(enabled: bool) -> Self {
        Self {
            stats: std::sync::Mutex::new(CacheStats::default()),
            enabled,
        }
    }

    /// Wrap a system prompt string into a cacheable content block.
    pub fn wrap_system_prompt(&self, prompt_text: &str) -> CachedContent {
        let content_hash = self.fingerprint_content(prompt_text);
        let estimated_tokens = self.estimate_tokens(prompt_text);

        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.total_cached_blocks += 1;

        tracing::debug!(
            content_hash = %content_hash[..16],
            estimated_tokens,
            "[PROMPT CACHE] System prompt wrapped with cache directive"
        );

        CachedContent {
            role: "user".to_string(),
            parts: vec![json!({ "text": prompt_text })],
            cache_control: if self.enabled {
                Some(CacheDirective {
                    cache_type: "ephemeral".to_string(),
                })
            } else {
                None
            },
            content_hash,
            estimated_tokens,
        }
    }

    /// Wrap a long document (RAG context) into a cacheable content block.
    pub fn wrap_document(&self, document_text: &str, document_title: &str) -> CachedContent {
        let full_content = format!("## {}\n\n{}", document_title, document_text);
        let content_hash = self.fingerprint_content(&full_content);
        let estimated_tokens = self.estimate_tokens(&full_content);

        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.total_cached_blocks += 1;
        stats.estimated_token_savings += estimated_tokens as u64 * 9 / 10;

        tracing::info!(
            document_title = %document_title,
            content_hash = %content_hash[..16],
            estimated_tokens,
            estimated_savings_pct = 90,
            "[PROMPT CACHE] Document wrapped - ~90% token cost reduction expected"
        );

        CachedContent {
            role: "user".to_string(),
            parts: vec![json!({
                "text": full_content
            })],
            cache_control: if self.enabled {
                Some(CacheDirective {
                    cache_type: "ephemeral".to_string(),
                })
            } else {
                None
            },
            content_hash,
            estimated_tokens,
        }
    }

    /// Build a complete Gemini-format request body with cached contents.
    pub fn build_gemini_request(
        &self,
        cached_contents: &[CachedContent],
        user_query: &str,
        temperature: f32,
        max_output_tokens: u32,
    ) -> Value {
        let mut contents: Vec<Value> = Vec::new();

        for cached in cached_contents {
            let mut content_obj = json!({
                "role": cached.role,
                "parts": cached.parts,
            });

            if let Some(ref cc) = cached.cache_control {
                content_obj["cache_control"] = json!({
                    "type": cc.cache_type
                });
            }

            contents.push(content_obj);
        }

        contents.push(json!({
            "role": "user",
            "parts": [{ "text": user_query }]
        }));

        json!({
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens
            }
        })
    }

    /// Build an Anthropic-format request body with cached system prompt and documents.
    ///
    /// Anthropic prompt caching uses `cache_control` breakpoints on message blocks.
    /// The system prompt and static documents are marked as cacheable, while
    /// the dynamic user query is not. Requires the `anthropic-beta: prompt-caching-2024-07-31`
    /// header to activate server-side caching.
    pub fn build_anthropic_request(
        &self,
        system_prompt: &CachedContent,
        documents: &[CachedContent],
        user_query: &str,
        model: &str,
        temperature: f32,
        max_tokens: u32,
    ) -> Value {
        let mut system_blocks: Vec<Value> = Vec::new();

        if let Some(text_val) = system_prompt.parts.first() {
            let text_str = text_val.get("text").and_then(|t| t.as_str()).unwrap_or("");
            let mut block = json!({
                "type": "text",
                "text": text_str,
            });
            if system_prompt.cache_control.is_some() {
                block["cache_control"] = json!({"type": "ephemeral"});
            }
            system_blocks.push(block);
        }

        for doc in documents {
            if let Some(text_val) = doc.parts.first() {
                let text_str = text_val.get("text").and_then(|t| t.as_str()).unwrap_or("");
                let mut block = json!({
                    "type": "text",
                    "text": text_str,
                });
                if doc.cache_control.is_some() {
                    block["cache_control"] = json!({"type": "ephemeral"});
                }
                system_blocks.push(block);
            }
        }

        let mut messages: Vec<Value> = Vec::new();

        let mut user_content: Vec<Value> = Vec::new();
        user_content.push(json!({
            "type": "text",
            "text": user_query,
        }));
        messages.push(json!({
            "role": "user",
            "content": user_content,
        }));

        json!({
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_blocks,
            "messages": messages,
        })
    }

    /// Build a DeepSeek/OpenAI-compatible request body with cached system prompt.
    ///
    /// DeepSeek supports prefix caching via `cache_control: {"type": "ephemeral"}`
    /// on the system message. The static system content is separated from the
    /// dynamic user message to maximize the cacheable prefix.
    pub fn build_deepseek_request(
        &self,
        system_prompt: &CachedContent,
        documents: &[CachedContent],
        user_query: &str,
        model: &str,
        temperature: f32,
        max_tokens: u32,
    ) -> Value {
        let mut messages: Vec<Value> = Vec::new();

        let mut system_text_parts: Vec<String> = Vec::new();
        if let Some(text_val) = system_prompt.parts.first() {
            if let Some(text_str) = text_val.get("text").and_then(|t| t.as_str()) {
                system_text_parts.push(text_str.to_string());
            }
        }
        for doc in documents {
            if let Some(text_val) = doc.parts.first() {
                if let Some(text_str) = text_val.get("text").and_then(|t| t.as_str()) {
                    system_text_parts.push(text_str.to_string());
                }
            }
        }

        let combined_system = system_text_parts.join("\n\n");
        let mut system_msg = json!({
            "role": "system",
            "content": combined_system,
        });
        if system_prompt.cache_control.is_some() {
            system_msg["cache_control"] = json!({"type": "ephemeral"});
        }
        messages.push(system_msg);

        messages.push(json!({
            "role": "user",
            "content": user_query,
        }));

        json!({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
    }

    /// Build a provider-specific cache-aware payload with headers and body.
    ///
    /// Separates static content (system prompt + documents) from dynamic user input
    /// to maximize cache hit rates. Returns a `CacheAwarePayload` with the
    /// serialized JSON body and any required cache-control headers.
    pub fn build_cache_aware_payload(
        &self,
        args: CacheAwarePayloadArgs,
    ) -> CacheAwarePayload {
        let mut headers: Vec<(String, String)> = Vec::new();
        headers.push(("Content-Type".to_string(), "application/json".to_string()));

        let body = match args.provider {
            Provider::Gemini => {
                let mut all_cached = vec![args.system_prompt.clone()];
                all_cached.extend(args.documents.iter().cloned());
                self.build_gemini_request(&all_cached, args.user_query, args.temperature, args.max_tokens)
            }
            Provider::Anthropic => {
                headers.push((
                    "anthropic-beta".to_string(),
                    "prompt-caching-2024-07-31".to_string(),
                ));
                self.build_anthropic_request(
                    args.system_prompt, args.documents, args.user_query, args.model, args.temperature, args.max_tokens,
                )
            }
            Provider::DeepSeek => {
                self.build_deepseek_request(
                    args.system_prompt, args.documents, args.user_query, args.model, args.temperature, args.max_tokens,
                )
            }
            Provider::Groq => {
                let mut all_cached = vec![args.system_prompt.clone()];
                all_cached.extend(args.documents.iter().cloned());
                self.build_deepseek_request(
                    args.system_prompt, args.documents, args.user_query, args.model, args.temperature, args.max_tokens,
                )
            }
        };

        CacheAwarePayload {
            body: serde_json::to_string(&body).unwrap_or_default(),
            headers,
            provider: args.provider,
        }
    }

    /// Record a cache hit for statistics tracking.
    pub fn record_cache_hit(&self) {
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        stats.total_cache_hits += 1;
    }

    /// Retrieve current cache statistics.
    pub fn get_stats(&self) -> CacheStats {
        self.stats.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn fingerprint_content(&self, content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        hex::encode(hasher.finalize())
    }

    fn estimate_tokens(&self, content: &str) -> u32 {
        content.len().div_ceil(4) as u32
    }
}

impl Default for PromptCacheManager {
    fn default() -> Self {
        Self::new(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_system_prompt() {
        let cache = PromptCacheManager::new(true);
        let result = cache.wrap_system_prompt("You are a helpful assistant.");

        assert_eq!(result.role, "user");
        assert!(result.cache_control.is_some());
        assert_eq!(result.cache_control.as_ref().unwrap().cache_type, "ephemeral");
        assert!(!result.content_hash.is_empty());
    }

    #[test]
    fn test_wrap_document() {
        let cache = PromptCacheManager::new(true);
        let doc = "This is a long document about blockchain network policies...";
        let result = cache.wrap_document(doc, "Network Policy Guide");

        assert_eq!(result.role, "user");
        assert!(result.estimated_tokens > 0);
        let stats = cache.get_stats();
        assert_eq!(stats.total_cached_blocks, 1);
    }

    #[test]
    fn test_build_gemini_request() {
        let cache = PromptCacheManager::new(true);
        let sys_prompt = cache.wrap_system_prompt("Extract execution parameters.");
        let request = cache.build_gemini_request(&[sys_prompt], "Network: ethereum", 0.1, 256);

        assert!(request["contents"].is_array());
        let contents = request["contents"].as_array().unwrap();
        assert!(contents.len() >= 2);
    }

    #[test]
    fn test_cache_disabled_omits_directive() {
        let cache = PromptCacheManager::new(false);
        let result = cache.wrap_system_prompt("Test prompt");

        assert!(result.cache_control.is_none());
    }

    #[test]
    fn test_build_anthropic_request() {
        let cache = PromptCacheManager::new(true);
        let sys_prompt = cache.wrap_system_prompt("You are a Web3 agent execution engine.");
        let request = cache.build_anthropic_request(
            &sys_prompt, &[], "Network: ethereum", "claude-3-5-sonnet-20241022", 0.1, 1024,
        );

        assert!(request["system"].is_array());
        assert!(request["messages"].is_array());
        let system = request["system"].as_array().unwrap();
        assert!(system[0]["cache_control"]["type"] == "ephemeral");
    }

    #[test]
    fn test_build_deepseek_request() {
        let cache = PromptCacheManager::new(true);
        let sys_prompt = cache.wrap_system_prompt("Extract execution parameters.");
        let request = cache.build_deepseek_request(
            &sys_prompt, &[], "Network: ethereum", "deepseek-chat", 0.1, 1024,
        );

        assert!(request["messages"].is_array());
        let messages = request["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert!(messages[0]["cache_control"]["type"] == "ephemeral");
    }

    #[test]
    fn test_build_cache_aware_payload_anthropic() {
        let cache = PromptCacheManager::new(true);
        let sys_prompt = cache.wrap_system_prompt("Adjudicate execution data.");
        let payload = cache.build_cache_aware_payload(CacheAwarePayloadArgs {
            provider: Provider::Anthropic,
            system_prompt: &sys_prompt,
            documents: &[],
            user_query: "Network: polygon",
            temperature: 0.1,
            max_tokens: 1024,
            model: "claude-3-5-sonnet-20241022",
        });

        assert_eq!(payload.provider, Provider::Anthropic);
        assert!(payload.headers.iter().any(|(k, _)| k == "anthropic-beta"));
        assert!(!payload.body.is_empty());
    }

    #[test]
    fn test_provider_supports_prompt_caching() {
        assert!(Provider::Gemini.supports_prompt_caching());
        assert!(Provider::Anthropic.supports_prompt_caching());
        assert!(Provider::DeepSeek.supports_prompt_caching());
        assert!(!Provider::Groq.supports_prompt_caching());
    }
}
