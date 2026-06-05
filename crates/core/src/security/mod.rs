// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Zero-Trust Security Module
//!
//! This module provides security primitives for the Serein Core kernel,
//! including data masking, encryption, sensitivity classification,
//! TPM 2.0 hardware measurement, and zero-leak log sanitization.

pub mod capabilities;
pub mod hmac_auth;
pub mod log_sanitizer;
pub mod masking;
pub mod mock_hsm;
pub mod tpm_measure;

pub use hmac_auth::{HmacAuthError, HmacSignature, ServiceAuthenticator};
pub use log_sanitizer::LogSanitizer;
pub use masking::{
    MaskingEngine, MaskingError, PiiField, PiiMaskConfig, PiiMaskingEngine, Sensitivity,
};
pub use tpm_measure::{
    compute_hash, request_hardware_quote, CcaPlatform, CcaPlatformDetector, EnclaveBootGuard,
    EventType, HardwareQuote, MeasurementEvent, QuoteResult, SevSnpReport, TdxReport, TpmError,
    TpmMeasurement, PCR_COUNT, SHA256_DIGEST_SIZE,
};
