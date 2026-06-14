// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

use reqwest::Client;
use std::path::Path;
use std::time::Duration;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncBufReadExt, AsyncSeekExt, AsyncWriteExt, BufReader};
use tracing::{error, info, warn};

const SYNC_INTERVAL: u64 = 300;
const CORPUS_PATH: &str = "serein-telemetry/ssd_training_corpus.jsonl";
const CURSOR_PATH: &str = "serein-telemetry/sync_cursor.meta";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt().with_env_filter("info").init();

    let url = std::env::var("SUPABASE_URL").unwrap_or_default();
    let key = std::env::var("SUPABASE_ANON_KEY").unwrap_or_default();
    let client = Client::builder().timeout(Duration::from_secs(30)).build()?;

    info!("Supabase sidecar started. Streaming mode enabled with cursor tracking.");

    if let Ok(mut entries) = tokio::fs::read_dir("serein-telemetry").await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if entry.file_name().to_string_lossy().ends_with(".tmp") {
                let _ = tokio::fs::remove_file(entry.path()).await;
                info!("Cleaned up orphan temporary cursor: {:?}", entry.file_name());
            }
        }
    }

    loop {
        if !url.is_empty() && !key.is_empty() {
            if let Err(e) = sync_cycle(&client, &url, &key).await {
                error!(error = %e, "Sync cycle failed");
            }
        }
        tokio::time::sleep(Duration::from_secs(SYNC_INTERVAL)).await;
    }
}

async fn sync_cycle(client: &Client, url: &str, key: &str) -> anyhow::Result<()> {
    let path = Path::new(CORPUS_PATH);
    if !path.exists() {
        return Ok(());
    }

    let mut cursor: u64 = if Path::new(CURSOR_PATH).exists() {
        tokio::fs::read_to_string(CURSOR_PATH)
            .await?
            .trim()
            .parse()
            .unwrap_or(0)
    } else {
        0
    };

    let mut file = File::open(path).await?;
    if cursor > file.metadata().await?.len() {
        cursor = 0;
    }

    file.seek(std::io::SeekFrom::Start(cursor)).await?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut synced = 0u64;
    let mut advanced_bytes = 0u64;

    while reader.read_line(&mut line).await? != 0 {
        let line_len = line.len() as u64;

        if let Ok(payload) = serde_json::from_str::<serde_json::Value>(&line) {
            let res = client
                .post(format!("{}/rest/v1/telemetry", url))
                .header("apikey", key)
                .header("Authorization", format!("Bearer {}", key))
                .header("Content-Type", "application/json")
                .header("Prefer", "return=minimal")
                .json(&payload)
                .send()
                .await;

            match res {
                Ok(r) if r.status().is_success() => {
                    synced += 1;
                    advanced_bytes += line_len;
                }
                Ok(r) if r.status().as_u16() == 400 => {
                    warn!("Skipping malformed record (400) to prevent sync stall");
                    advanced_bytes += line_len;
                }
                Ok(r) => {
                    let status = r.status();
                    error!(
                        status = status.as_u16(),
                        "Systemic failure (HTTP {}). Aborting cycle to prevent data loss.",
                        status
                    );
                    return Err(anyhow::anyhow!(
                        "Systemic sync failure - HTTP {}",
                        status
                    ));
                }
                Err(e) => {
                    error!(
                        error = %e,
                        "Systemic failure (Network/Auth). Aborting cycle to prevent data loss."
                    );
                    return Err(anyhow::anyhow!(
                        "Systemic sync failure - network error: {}",
                        e
                    ));
                }
            }
        } else {
            advanced_bytes += line_len;
        }

        line.clear();
    }

    if advanced_bytes > 0 {
        cursor += advanced_bytes;
        let tmp_path = format!("{}.{}.tmp", CURSOR_PATH, std::process::id());
        let mut cursor_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)
            .await?;
        cursor_file.write_all(cursor.to_string().as_bytes()).await?;
        cursor_file.sync_all().await?;
        tokio::fs::rename(&tmp_path, CURSOR_PATH).await?;

        if synced > 0 {
            info!(synced, new_cursor = cursor, "Sync cycle completed successfully");
        } else {
            warn!(new_cursor = cursor, "Cursor advanced, but 0 records synced. Check Supabase credentials/schema!");
        }
    }
    Ok(())
}
