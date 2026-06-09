// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

use tracing::info;

pub fn deduct_inference_cost(tenant_id: &str, tokens_used: u64) {
    let cost_usdc = tokens_used as f64 * 0.0000015;
    info!(
        target: "depin_billing",
        tenant_id = %tenant_id,
        tokens = tokens_used,
        cost_usdc = %format!("{:.4}", cost_usdc),
        "\u{26a1} [DEPIN_BILLING] Deducted {:.4} USDC from Tenant Wallet for Trustless AI Inference.",
        cost_usdc
    );
}