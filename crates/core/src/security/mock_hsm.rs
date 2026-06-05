// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mock Hardware Security Module (HSM) Implementation
//!
//! Provides a software-stub implementation of the `HardwareSecurityModule` trait
//! for development and testing environments where physical TPM 2.0 hardware is
//! unavailable. All measurement operations are no-ops that log warnings.
//!
//! ## Production Usage
//! In production, replace this module with a real TPM 2.0 binding via `tss-esapi`
//! or a platform-specific CCA driver (AMD SEV-SNP / Intel TDX).

use super::tpm_measure::{
    compute_hash, CcaPlatform, CcaPlatformDetector, EnclaveBootGuard, HardwareQuote, QuoteResult,
    TpmError, PCR_COUNT, SHA256_DIGEST_SIZE,
};
use crate::HardwareSecurityModule;
use async_trait::async_trait;
use tracing::{info, warn};

pub struct MockHsm {
    boot_guard: EnclaveBootGuard,
}

impl MockHsm {
    pub fn new() -> Result<Self, TpmError> {
        let platform = CcaPlatformDetector::detect().unwrap_or(CcaPlatform::SoftwareStub);
        let boot_guard = EnclaveBootGuard::with_platform(platform, false);
        info!(
            platform = ?boot_guard.platform(),
            "MockHsm initialized - software-stub mode, no hardware attestation"
        );
        Ok(Self { boot_guard })
    }
}

#[async_trait]
impl HardwareSecurityModule for MockHsm {
    async fn verify_attestation(&self) -> Result<(), TpmError> {
        warn!("MockHsm: attestation verification is a no-op in software-stub mode");
        self.boot_guard.verify_attestation_before_boot().await
    }

    async fn get_verified_quote(&self) -> Option<HardwareQuote> {
        self.boot_guard.verified_quote().await
    }

    fn measure_kernel(&self, kernel_hash: &[u8; SHA256_DIGEST_SIZE]) -> Result<(), TpmError> {
        let _ = kernel_hash;
        warn!("MockHsm: kernel measurement is a no-op in software-stub mode");
        Ok(())
    }

    fn measure_security_policy(
        &self,
        policy_hash: &[u8; SHA256_DIGEST_SIZE],
    ) -> Result<(), TpmError> {
        let _ = policy_hash;
        warn!("MockHsm: security policy measurement is a no-op in software-stub mode");
        Ok(())
    }

    fn measure_module(
        &self,
        module_name: &str,
        module_hash: &[u8; SHA256_DIGEST_SIZE],
    ) -> Result<(), TpmError> {
        let _ = (module_name, module_hash);
        warn!("MockHsm: module measurement is a no-op in software-stub mode");
        Ok(())
    }

    fn measure_runtime_state(&self, state_hash: &[u8; SHA256_DIGEST_SIZE]) -> Result<(), TpmError> {
        let _ = state_hash;
        warn!("MockHsm: runtime state measurement is a no-op in software-stub mode");
        Ok(())
    }

    fn quote(&self, nonce: &[u8]) -> Result<QuoteResult, TpmError> {
        let hash = compute_hash(nonce);
        let mut pcr_values = [[0u8; SHA256_DIGEST_SIZE]; PCR_COUNT];
        pcr_values[0] = hash;
        Ok(QuoteResult {
            pcr_digest: hash,
            pcr_values,
            nonce: nonce.to_vec(),
        })
    }

    fn verify_quote(
        &self,
        quote: &QuoteResult,
        expected_pcrs: &[[u8; SHA256_DIGEST_SIZE]; PCR_COUNT],
    ) -> Result<bool, TpmError> {
        let _ = (quote, expected_pcrs);
        warn!("MockHsm: quote verification always returns true in software-stub mode");
        Ok(true)
    }
}
