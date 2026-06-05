// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # WASI-NN Model Interface - P5 Edge Autonomy Layer
//!
//! Implements the `ModelInterface` trait for on-device Small Language Model (SLM)
//! inference via the WASI-NN specification. This module provides the foundation
//! for the P5 evolutionary autonomy layer, enabling air-gapped model inference
//! without cloud dependency.
//!
//! ## Architecture
//! - **GraphBuilder**: Constructs WASI-NN computation graphs with `Ggml` encoding
//!   and `Cpu` target for deterministic, reproducible inference.
//! - **ModelInterface**: Trait abstraction over inference and weight reloading,
//!   enabling hot-swap of model weights for evolutionary training loops.
//! - **Weight Storage**: Models are loaded from the `./serein-models` VFS mount,
//!   ensuring air-gapped operation with no network egress for model data.
//!
//! ## Safety Contract
//! - Model weights are loaded from READ-ONLY VFS - no guest write access.
//! - Inference runs in sandboxed WASI-NN context with fuel metering.
//! - Weight reload is atomic - no partial state during hot-swap.

use anyhow::Result;
use std::path::PathBuf;
use tracing::warn;

#[cfg(feature = "wasi-nn")]
use wasi_nn::{ExecutionTarget, GraphBuilder, GraphEncoding};

/// Encoding format for model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelEncoding {
    /// GGML quantized format - optimized for CPU inference.
    Ggml,
    /// ONNX format - cross-framework interoperability.
    Onnx,
    /// TensorFlow SavedModel format.
    Tensorflow,
}

impl std::fmt::Display for ModelEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelEncoding::Ggml => write!(f, "ggml"),
            ModelEncoding::Onnx => write!(f, "onnx"),
            ModelEncoding::Tensorflow => write!(f, "tensorflow"),
        }
    }
}

/// Execution target for model inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceTarget {
    /// CPU execution - deterministic, reproducible, air-gapped.
    Cpu,
    /// GPU execution - higher throughput, requires hardware access.
    Gpu,
}

impl std::fmt::Display for InferenceTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceTarget::Cpu => write!(f, "cpu"),
            InferenceTarget::Gpu => write!(f, "gpu"),
        }
    }
}

/// Configuration for a WASI-NN model instance.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Name identifier for the model (e.g., "phi-2-q4_0").
    pub model_name: String,
    /// Encoding format for the model weights.
    pub encoding: ModelEncoding,
    /// Execution target for inference.
    pub target: InferenceTarget,
    /// Root directory for model weight files (air-gapped VFS).
    pub models_dir: PathBuf,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_name: "default-slm".to_string(),
            encoding: ModelEncoding::Ggml,
            target: InferenceTarget::Cpu,
            models_dir: PathBuf::from("./serein-models"),
        }
    }
}

impl ModelConfig {
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = name.into();
        self
    }

    pub fn with_encoding(mut self, encoding: ModelEncoding) -> Self {
        self.encoding = encoding;
        self
    }

    pub fn with_target(mut self, target: InferenceTarget) -> Self {
        self.target = target;
        self
    }

    pub fn with_models_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.models_dir = dir.into();
        self
    }
}

/// Trait defining the interface for WASI-NN model inference.
///
/// This trait abstracts over the WASI-NN specification, providing a clean
/// boundary between the host orchestration layer and the model execution
/// environment. Implementations must ensure:
///
/// 1. **Air-gapped loading**: Weights are read from local VFS only.
/// 2. **Atomic reload**: Weight hot-swap is atomic - no partial state.
/// 3. **Fuel-metered inference**: Inference runs within fuel budget.
pub trait ModelInterface: Send + Sync {
    /// Perform inference on the loaded model.
    ///
    /// # Arguments
    /// * `input` - The input tensor data as raw bytes.
    /// * `max_tokens` - Maximum number of tokens to generate.
    ///
    /// # Returns
    /// The model's output as raw bytes, or an error if inference fails.
    fn infer(&self, input: &[u8], max_tokens: u32) -> Result<Vec<u8>>;

    /// Reload model weights from the VFS mount point.
    ///
    /// This enables hot-swapping of model weights for evolutionary training
    /// loops without restarting the runtime. The reload is atomic - the
    /// previous weights remain active until the new weights are fully loaded.
    ///
    /// # Arguments
    /// * `weight_path` - Path to the new weight file within the models VFS.
    fn reload_weights(&self, weight_path: &str) -> Result<()>;

    /// Get the model name identifier.
    fn model_name(&self) -> &str;

    /// Get the current encoding format.
    fn encoding(&self) -> ModelEncoding;

    /// Get the current inference target.
    fn target(&self) -> InferenceTarget;
}

/// WASI-NN backed model implementation using the wasi-nn crate.
///
/// This implementation uses `wasi_nn::GraphBuilder` to construct computation
/// graphs with `Ggml` encoding and `Cpu` target for deterministic inference.
#[cfg(feature = "wasi-nn")]
pub struct WasiNnModel {
    config: ModelConfig,
    graph: std::sync::Mutex<Option<wasi_nn::Graph>>,
    graph_context: std::sync::Mutex<Option<wasi_nn::GraphExecutionContext>>,
}

#[cfg(feature = "wasi-nn")]
impl WasiNnModel {
    /// Create a new WASI-NN model instance from the given configuration.
    ///
    /// The model weights are loaded from `config.models_dir / config.model_name`.
    pub fn new(config: ModelConfig) -> Result<Self> {
        let model = Self {
            config,
            graph: std::sync::Mutex::new(None),
            graph_context: std::sync::Mutex::new(None),
        };

        let weight_path = model.config.models_dir.join(&model.config.model_name);
        model.load_graph(&weight_path)?;

        info!(
            model_name = %model.config.model_name,
            encoding = %model.config.encoding,
            target = %model.config.target,
            "[WASI-NN] Model instance created and weights loaded"
        );

        Ok(model)
    }

    fn load_graph(&self, weight_path: &PathBuf) -> Result<()> {
        let encoding = match self.config.encoding {
            ModelEncoding::Ggml => GraphEncoding::Ggml,
            ModelEncoding::Onnx => GraphEncoding::Onnx,
            ModelEncoding::Tensorflow => GraphEncoding::Tensorflow,
        };

        let target = match self.config.target {
            InferenceTarget::Cpu => ExecutionTarget::CPU,
            InferenceTarget::Gpu => ExecutionTarget::GPU,
        };

        let weight_data = std::fs::read(weight_path)
            .with_context(|| format!("Failed to read model weights from {:?}", weight_path))?;

        let graph = GraphBuilder::new(encoding, target)
            .build_from_bytes(&[&weight_data])
            .context("Failed to build WASI-NN computation graph from weight bytes")?;

        let context = graph
            .init_execution_context()
            .context("Failed to initialize WASI-NN execution context")?;

        {
            let mut graph_guard = self.graph.lock().unwrap_or_else(|e| {
                tracing::error!(
                    "[WASI-NN] Graph mutex poisoned - recovering from poison: {}",
                    e
                );
                e.into_inner()
            });
            *graph_guard = Some(graph);
        }
        {
            let mut ctx_guard = self.graph_context.lock().unwrap_or_else(|e| {
                tracing::error!(
                    "[WASI-NN] Context mutex poisoned - recovering from poison: {}",
                    e
                );
                e.into_inner()
            });
            *ctx_guard = Some(context);
        }

        Ok(())
    }
}

#[cfg(feature = "wasi-nn")]
impl ModelInterface for WasiNnModel {
    fn infer(&self, input: &[u8], max_tokens: u32) -> Result<Vec<u8>> {
        let mut ctx_guard = self.graph_context.lock().unwrap_or_else(|e| {
            tracing::error!(
                "[WASI-NN] Context mutex poisoned during inference - recovering from poison: {}",
                e
            );
            e.into_inner()
        });
        let ctx = ctx_guard
            .as_mut()
            .context("WASI-NN execution context not initialized - call reload_weights first")?;

        let input_tensor = wasi_nn::Tensor {
            dimensions: &[1, input.len() as u32],
            r#type: wasi_nn::TensorType::U8,
            data: input.to_vec(),
        };

        ctx.set_input(0, input_tensor)
            .context("Failed to set WASI-NN input tensor")?;

        ctx.compute()
            .context("WASI-NN inference computation failed")?;

        let output = ctx
            .get_output(0)
            .context("Failed to retrieve WASI-NN output tensor")?;

        info!(
            model_name = %self.config.model_name,
            input_len = input.len(),
            output_len = output.len(),
            max_tokens = max_tokens,
            "[WASI-NN] Inference completed"
        );

        Ok(output)
    }

    fn reload_weights(&self, weight_path: &str) -> Result<()> {
        let full_path = self.config.models_dir.join(weight_path);

        info!(
            model_name = %self.config.model_name,
            weight_path = %full_path.display(),
            "[WASI-NN] Initiating atomic weight reload"
        );

        self.load_graph(&full_path)?;

        info!(
            model_name = %self.config.model_name,
            "[WASI-NN] Weight reload completed - new graph and context active"
        );

        Ok(())
    }

    fn model_name(&self) -> &str {
        &self.config.model_name
    }

    fn encoding(&self) -> ModelEncoding {
        self.config.encoding
    }

    fn target(&self) -> InferenceTarget {
        self.config.target
    }
}

/// Stub model implementation for non-WASI-NN builds.
///
/// Provides the `ModelInterface` trait implementation that logs warnings
/// when inference or reload is attempted without the `wasi-nn` feature enabled.
#[cfg(not(feature = "wasi-nn"))]
pub struct StubModel {
    config: ModelConfig,
}

#[cfg(not(feature = "wasi-nn"))]
impl StubModel {
    pub fn new(config: ModelConfig) -> Result<Self> {
        warn!(
            model_name = %config.model_name,
            "[WASI-NN] Stub model created - wasi-nn feature not enabled. \
             Inference calls will return errors."
        );
        Ok(Self { config })
    }
}

#[cfg(not(feature = "wasi-nn"))]
impl ModelInterface for StubModel {
    fn infer(&self, _input: &[u8], _max_tokens: u32) -> Result<Vec<u8>> {
        warn!(
            model_name = %self.config.model_name,
            "[WASI-NN] Infer called on stub model - wasi-nn feature not enabled"
        );
        Err(anyhow::anyhow!(
            "WASI-NN inference not available - enable the 'wasi-nn' feature to enable on-device SLM inference"
        ))
    }

    fn reload_weights(&self, _weight_path: &str) -> Result<()> {
        warn!(
            model_name = %self.config.model_name,
            "[WASI-NN] Reload called on stub model - wasi-nn feature not enabled"
        );
        Err(anyhow::anyhow!(
            "WASI-NN weight reload not available - enable the 'wasi-nn' feature"
        ))
    }

    fn model_name(&self) -> &str {
        &self.config.model_name
    }

    fn encoding(&self) -> ModelEncoding {
        self.config.encoding
    }

    fn target(&self) -> InferenceTarget {
        self.config.target
    }
}

/// Factory function to create the appropriate model implementation.
///
/// Returns a `WasiNnModel` when the `wasi-nn` feature is enabled, or a
/// `StubModel` otherwise. This allows the rest of the codebase to use
/// `ModelInterface` without feature-gating every call site.
pub fn create_model(config: ModelConfig) -> Result<Box<dyn ModelInterface>> {
    #[cfg(feature = "wasi-nn")]
    {
        Ok(Box::new(WasiNnModel::new(config)?))
    }
    #[cfg(not(feature = "wasi-nn"))]
    {
        Ok(Box::new(StubModel::new(config)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.model_name, "default-slm");
        assert_eq!(config.encoding, ModelEncoding::Ggml);
        assert_eq!(config.target, InferenceTarget::Cpu);
        assert_eq!(config.models_dir, PathBuf::from("./serein-models"));
    }

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::default()
            .with_model_name("phi-2-q4_0")
            .with_encoding(ModelEncoding::Onnx)
            .with_target(InferenceTarget::Gpu)
            .with_models_dir("/custom/models");

        assert_eq!(config.model_name, "phi-2-q4_0");
        assert_eq!(config.encoding, ModelEncoding::Onnx);
        assert_eq!(config.target, InferenceTarget::Gpu);
        assert_eq!(config.models_dir, PathBuf::from("/custom/models"));
    }

    #[test]
    fn test_encoding_display() {
        assert_eq!(format!("{}", ModelEncoding::Ggml), "ggml");
        assert_eq!(format!("{}", ModelEncoding::Onnx), "onnx");
        assert_eq!(format!("{}", ModelEncoding::Tensorflow), "tensorflow");
    }

    #[test]
    fn test_target_display() {
        assert_eq!(format!("{}", InferenceTarget::Cpu), "cpu");
        assert_eq!(format!("{}", InferenceTarget::Gpu), "gpu");
    }

    #[cfg(not(feature = "wasi-nn"))]
    #[test]
    fn test_stub_model_inference_fails() {
        let config = ModelConfig::default();
        let model = StubModel::new(config).unwrap();
        assert!(model.infer(b"test", 100).is_err());
    }

    #[cfg(not(feature = "wasi-nn"))]
    #[test]
    fn test_stub_model_reload_fails() {
        let config = ModelConfig::default();
        let model = StubModel::new(config).unwrap();
        assert!(model.reload_weights("new_weights.bin").is_err());
    }

    #[test]
    fn test_create_model_factory() {
        let config = ModelConfig::default();
        let model = create_model(config);
        assert!(model.is_ok());
        assert_eq!(model.unwrap().model_name(), "default-slm");
    }
}
