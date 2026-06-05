// Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Engine module for WebAssembly runtime execution
//!
//! This module contains the secure WebAssembly runtime engine implementation
//! with comprehensive security controls and isolation boundaries.
//!
//! ## Components
//! - `wasmtime_rt`: Secure Wasmtime runtime engine
//! - `hot_swap`: Lock-free atomic pointer hot-swapping (Bumpless Transfer)
//! - `fuel_monitor`: Joule-level energy sensing and ESD logic

pub mod fuel_monitor;
pub mod hot_swap;
pub mod wasmtime_rt;

pub use fuel_monitor::{
    parse_to_ast, execution_payload_schema, AstNode, EnergyMetrics, EsdState, FieldConstraint, FieldType,
    FuelMonitor, PayloadViolation, Z3FormalValidator, Z3Validator, CRITICAL_THRESHOLD,
    DEFAULT_FUEL_QUOTA, JOULES_PER_FUEL, MAX_AST_NODES, MAX_AST_STRING_LENGTH, MAX_JSON_DEPTH,
    MAX_NUMERIC_VALUE, MAX_OBJECT_KEYS, MAX_STRING_LENGTH, PRE_SHUTDOWN_THRESHOLD,
};
pub use hot_swap::{HotSwapContainer, SisInterlock, SisInterlockState, GLOBAL_FUEL_QUOTA};
