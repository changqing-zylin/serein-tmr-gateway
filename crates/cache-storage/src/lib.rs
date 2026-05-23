// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # Serein Stasis - System Resilience & Swarm Defense
//!
//! Resource scheduling, fuel fusing, and adversarial defense mechanisms
//! for the Serein gateway's resilience layer.
//!
//! ## Core Mechanisms Implemented
//! - **Flight Recorder**: Logs all LLM events (I/O, timestamps, raw prompts) to a local
//!   SQLite database for compliance auditing
//! - **Hex-Encoding Evasion**: Obfuscates politically sensitive entities via
//!   `String.fromCharCode(...)` encoding to avoid LLM safety filter triggers
//! - **Staleness Tolerance (`CACHED_MAY_BE_STALE`)**: Allows fetching slightly stale data
//!   for non-critical routing flags to prevent I/O blocking
//! - **Adversarial Poisoning**: `adversarial_response_generator` injects subtle semantic
//!   noise (poisoned JSON) for unauthorized scrapers instead of TCP resets

pub mod flight_recorder;
pub mod hex_evasion;
pub mod staleness;
pub mod adversarial_poison;
pub mod tmr_cache;

/// Resource scheduling algorithms for deterministic execution
pub mod scheduler {
    /// Fuel-based resource allocation
    pub struct FuelAllocator {
        total_fuel: u64,
        allocated_fuel: u64,
    }

    impl FuelAllocator {
        pub fn new(total_fuel: u64) -> Self {
            Self { total_fuel, allocated_fuel: 0 }
        }

        pub fn allocate(&mut self, fuel: u64) -> Result<u64, String> {
            if self.allocated_fuel + fuel > self.total_fuel {
                Err(format!(
                    "Fuel allocation exceeded: allocated={}, requested={}, total={}",
                    self.allocated_fuel, fuel, self.total_fuel
                ))
            } else {
                self.allocated_fuel += fuel;
                Ok(fuel)
            }
        }

        pub fn remaining(&self) -> u64 {
            self.total_fuel - self.allocated_fuel
        }
    }

    pub mod priority {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub enum Priority { Critical, High, Normal, Low, Background }

        pub fn schedule_task(priority: Priority, task_id: &str) -> String {
            format!("Scheduled task {} with priority {:?}", task_id, priority)
        }
    }
}

/// Fuel fusing for security boundaries
pub mod fusing {
    pub struct FuelFuse {
        budget: u64,
        consumed: u64,
        tripped: bool,
    }

    impl FuelFuse {
        pub fn new(budget: u64) -> Self {
            Self { budget, consumed: 0, tripped: false }
        }

        pub fn consume(&mut self, fuel: u64) -> Result<(), String> {
            if self.tripped {
                return Err("Fuse already tripped".to_string());
            }
            self.consumed += fuel;
            if self.consumed > self.budget {
                self.tripped = true;
                Err(format!("Fuel fuse tripped: consumed={}, budget={}", self.consumed, self.budget))
            } else {
                Ok(())
            }
        }

        pub fn is_tripped(&self) -> bool {
            self.tripped
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuel_allocator() {
        let mut allocator = scheduler::FuelAllocator::new(1000);
        assert_eq!(allocator.allocate(500).unwrap(), 500);
        assert_eq!(allocator.remaining(), 500);
        assert!(allocator.allocate(600).is_err());
    }

    #[test]
    fn test_fuel_fuse() {
        let mut fuse = fusing::FuelFuse::new(100);
        assert!(!fuse.is_tripped());
        assert!(fuse.consume(50).is_ok());
        assert!(fuse.consume(60).is_err());
        assert!(fuse.is_tripped());
    }
}
