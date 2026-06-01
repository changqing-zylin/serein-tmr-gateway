// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Ed25519 Payload Signature Verification
//!
//! Verifies Ed25519 digital signatures on inter-service payloads using
//! `AEGIS_PUBLIC_KEY` loaded from the environment. This ensures payload
//! integrity and authenticity across the zero-trust network boundary.

use ed25519_dalek::{VerifyingKey, Signature, Verifier};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SignatureError {
    #[error("Invalid AEGIS_PUBLIC_KEY: {0}")]
    InvalidPublicKey(String),

    #[error("Signature verification failed: {0}")]
    VerificationFailed(String),

    #[error("Missing AEGIS_PUBLIC_KEY environment variable")]
    MissingPublicKey,

    #[error("Invalid signature bytes: {0}")]
    InvalidSignature(String),
}

/// Ed25519 payload signature verifier using `AEGIS_PUBLIC_KEY`.
pub struct PayloadVerifier {
    public_key: VerifyingKey,
}

impl PayloadVerifier {
    /// Create a new verifier from the `AEGIS_PUBLIC_KEY` environment variable.
    ///
    /// The key must be a 64-character hex string representing the 32-byte
    /// Ed25519 verifying key.
    pub fn from_env() -> Result<Self, SignatureError> {
        let key_hex = std::env::var("AEGIS_PUBLIC_KEY")
            .map_err(|_| SignatureError::MissingPublicKey)?;

        Self::from_hex(&key_hex)
    }

    /// Create a new verifier from a hex-encoded public key string.
    pub fn from_hex(key_hex: &str) -> Result<Self, SignatureError> {
        if key_hex.is_empty() {
            return Err(SignatureError::MissingPublicKey);
        }

        let key_bytes = hex::decode(key_hex)
            .map_err(|e| SignatureError::InvalidPublicKey(format!("hex decode: {}", e)))?;

        let key_bytes: [u8; 32] = key_bytes.try_into()
            .map_err(|_| SignatureError::InvalidPublicKey(
                format!("expected 32 bytes (64 hex chars), got {} bytes", key_hex.len() / 2)
            ))?;

        let public_key = VerifyingKey::from_bytes(&key_bytes)
            .map_err(|e| SignatureError::InvalidPublicKey(format!("invalid Ed25519 key: {}", e)))?;

        Ok(Self { public_key })
    }

    /// Verify an Ed25519 signature against the given payload.
    ///
    /// # Arguments
    /// * `payload` - The original message bytes that were signed
    /// * `signature_hex` - The 128-character hex-encoded Ed25519 signature
    pub fn verify(&self, payload: &[u8], signature_hex: &str) -> Result<(), SignatureError> {
        let sig_bytes = hex::decode(signature_hex)
            .map_err(|e| SignatureError::InvalidSignature(format!("hex decode: {}", e)))?;

        let sig_bytes: [u8; 64] = sig_bytes.try_into()
            .map_err(|_| SignatureError::InvalidSignature(
                format!("expected 64 bytes (128 hex chars), got {} bytes", signature_hex.len() / 2)
            ))?;

        let signature = Signature::from_bytes(&sig_bytes);

        self.public_key.verify(payload, &signature)
            .map_err(|e| SignatureError::VerificationFailed(format!("{}", e)))
    }

    /// Verify a signature on a string payload.
    pub fn verify_string(&self, payload: &str, signature_hex: &str) -> Result<(), SignatureError> {
        self.verify(payload.as_bytes(), signature_hex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{SigningKey, Signer};

    #[test]
    fn test_sign_and_verify_round_trip() {
        let mut csprng = rand::thread_rng();
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();

        let message = b"test payload for Aegis verification";
        let signature = signing_key.sign(message);

        let verifier = PayloadVerifier {
            public_key: verifying_key,
        };

        let sig_hex = hex::encode(signature.to_bytes());
        assert!(verifier.verify(message, &sig_hex).is_ok());
    }

    #[test]
    fn test_verify_fails_with_wrong_key() {
        let mut csprng = rand::thread_rng();
        let signing_key = SigningKey::generate(&mut csprng);
        let wrong_signing_key = SigningKey::generate(&mut csprng);
        let wrong_verifying_key = wrong_signing_key.verifying_key();

        let message = b"test payload";
        let signature = signing_key.sign(message);

        let verifier = PayloadVerifier {
            public_key: wrong_verifying_key,
        };

        let sig_hex = hex::encode(signature.to_bytes());
        assert!(verifier.verify(message, &sig_hex).is_err());
    }

    #[test]
    fn test_verify_fails_with_tampered_payload() {
        let mut csprng = rand::thread_rng();
        let signing_key = SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();

        let message = b"original payload";
        let signature = signing_key.sign(message);

        let verifier = PayloadVerifier {
            public_key: verifying_key,
        };

        let sig_hex = hex::encode(signature.to_bytes());
        assert!(verifier.verify(b"tampered payload", &sig_hex).is_err());
    }

    #[test]
    fn test_from_hex_rejects_empty_key() {
        assert!(PayloadVerifier::from_hex("").is_err());
    }

    #[test]
    fn test_from_hex_rejects_invalid_length() {
        assert!(PayloadVerifier::from_hex("abcd1234").is_err());
    }
}
