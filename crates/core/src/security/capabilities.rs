// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Capability-Based Security and Resource Management
//!
//! This module implements the core components for capability-based security,
//! resource monitoring, and threat detection within the Serein microkernel.
//! It provides the `ResourceMonitor` for tracking guest execution metrics and
//! defines the logic for the "Strangulation Mode" active defense mechanism.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Tracks resource usage for a given source IP address.
#[derive(Debug, Clone)]
struct RequestTracker {
    /// The number of requests received from this source.
    count: u64,
    /// The time of the first request in the current window.
    first_request_at: Instant,
}

/// Monitors resource consumption to detect anomalous patterns like high-frequency sniffing.
///
/// This monitor is a key component of the active defense strategy. It maintains a
/// record of incoming requests per source IP and flags any source that exceeds a
/// defined threshold within a specific time window.
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    /// A map from IP addresses to their request tracking data.
    requests: Arc<Mutex<HashMap<IpAddr, RequestTracker>>>,
    /// The maximum number of requests allowed within the `time_window`.
    request_threshold: u64,
    /// The duration of the sliding time window for tracking requests.
    time_window: Duration,
}

impl ResourceMonitor {
    /// Creates a new `ResourceMonitor` with specified detection parameters.
    ///
    /// * `request_threshold`: The number of requests from a single IP that will trigger a threat flag.
    /// * `time_window_seconds`: The sliding window duration in seconds for monitoring requests.
    pub fn new(request_threshold: u64, time_window_seconds: u64) -> Self {
        Self {
            requests: Arc::new(Mutex::new(HashMap::new())),
            request_threshold,
            time_window: Duration::from_secs(time_window_seconds),
        }
    }

    /// Records an incoming request from a source IP and checks if it exceeds the defined threshold.
    ///
    /// If a source is flagged as a threat, it enters a "Strangulation Mode," where subsequent
    /// requests from it will be funneled into a resource-intensive countermeasure hub
    /// instead of the standard execution path.
    ///
    /// # Returns
    /// * `true` if the source IP has exceeded the request threshold and is considered a threat.
    /// * `false` otherwise.
    pub fn check_and_record_request(&self, ip: IpAddr) -> Result<bool, String> {
        let mut requests = self
            .requests
            .lock()
            .map_err(|_| "ResourceMonitor mutex poisoned".to_string())?;
        let now = Instant::now();

        requests
            .retain(|_, tracker| now.duration_since(tracker.first_request_at) < self.time_window);

        let tracker = requests.entry(ip).or_insert(RequestTracker {
            count: 0,
            first_request_at: now,
        });

        tracker.count += 1;

        if tracker.count > self.request_threshold
            && now.duration_since(tracker.first_request_at) < self.time_window
        {
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

impl Default for ResourceMonitor {
    /// Creates a `ResourceMonitor` with default production-grade settings.
    /// - Threshold: 100 requests
    /// - Window: 60 seconds
    fn default() -> Self {
        Self::new(100, 60)
    }
}
