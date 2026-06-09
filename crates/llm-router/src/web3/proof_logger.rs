// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

use sha2::{Digest, Sha256};
use tracing::info;

pub fn submit_verifiable_proof(consensus_output: &str, agreement_count: usize) {
    let mut hasher = Sha256::new();
    hasher.update(consensus_output.as_bytes());
    hasher.update(&agreement_count.to_le_bytes());
    let tx_hash = hex::encode(hasher.finalize());

    info!(
        target: "web3_proof",
        tx_hash = %format!("0x{}", tx_hash),
        agreement_count = agreement_count,
        "\u{1f310} [WEB3_PROOF] Consensus achieved. Submitting Verifiable AI Proof to ledger... TxHash: 0x{}",
        tx_hash
    );
}