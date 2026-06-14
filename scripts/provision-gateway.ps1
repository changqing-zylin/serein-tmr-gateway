<#
.SYNOPSIS
    Serein Core - Infrastructure Provisioning & Deployment Pipeline
.DESCRIPTION
    Orchestrates the end-to-end build-test-deploy lifecycle for the Serein Gateway.
.EXAMPLE
    .\provision-gateway.ps1 -FullClean
.NOTES
    Copyright (c) 2026 Changqing Zhang. All rights reserved.
#>
[CmdletBinding()]
param (
    [switch]$FullClean = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$WarningPreference = "Continue"

# --- SYSTEM PATH OVERRIDE: FORCE ROOT CONTEXT ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path -Path $ScriptDir -ChildPath ".."))
Set-Location -Path $ProjectRoot
# ------------------------------------------------

try {
    Write-Host "`n[INIT] Starting Serein Core Provisioning Sequence..." -ForegroundColor Cyan

    # --- Phase 1: Environment Sanitization ---
    if ($FullClean) {
        Write-Host "[STAGE 1/5] Performing Deep Sanitization..." -ForegroundColor Yellow
        cargo clean
        if ($LASTEXITCODE -ne 0) { throw "cargo clean failed." }
        if (Test-Path "modules") {
            Remove-Item -Path "modules" -Recurse -Force
        }
        Write-Host "[CLEAN] Target and Modules directories purged." -ForegroundColor Green
    } else {
        Write-Host "[STAGE 1/5] Skipping Clean (Incremental Build Active)..." -ForegroundColor Gray
    }

    # --- Phase 1.5: Open Source License Compliance ---
    Write-Host "[STAGE 1.5/5] Checking Open Source License Compliance..." -ForegroundColor Yellow
    try {
        $cargoDeny = Get-Command cargo-deny -ErrorAction Stop
        & cargo deny check licenses
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[WARN] License compliance check found violations. Review deny.toml for details." -ForegroundColor Yellow
        } else {
            Write-Host "[OK] License compliance check passed." -ForegroundColor Green
        }
    } catch {
        Write-Host "[WARN] cargo-deny not installed. Run 'cargo install cargo-deny' to enable license compliance checks." -ForegroundColor Yellow
        Write-Host "[INFO] Skipping license compliance stage (tool absent)." -ForegroundColor Gray
    }

    # --- Phase 2: Component Compilation (Canonical ABI) ---
    Write-Host "[STAGE 2/5] Compiling WASM Canonical ABI Components..." -ForegroundColor Yellow

    Write-Host " -> Compiling Target: intent-engine..." -ForegroundColor Gray
    cargo component build -p intent-engine --release
    if ($LASTEXITCODE -ne 0) { throw "intent-engine compilation failed." }

    Write-Host " -> Compiling Target: serein-sandbox-guard..." -ForegroundColor Gray
    cargo component build -p serein-sandbox-guard --release
    if ($LASTEXITCODE -ne 0) { throw "serein-sandbox-guard compilation failed." }

    # --- Phase 3: Artifact Deployment ---
    Write-Host "[STAGE 3/5] Synchronizing Artifacts to Runtime Modules..." -ForegroundColor Yellow
    $ModuleDir = "modules"
    if (-not (Test-Path $ModuleDir)) { New-Item -ItemType Directory -Path $ModuleDir | Out-Null }

    $ArtifactMap = @{
        "target\wasm32-wasip1\release\intent_engine.wasm" = "modules\intent_engine.wasm"
        "target\wasm32-wasip1\release\serein_sandbox_guard.wasm"  = "modules\serein_sandbox_guard.wasm"
    }

    foreach ($Source in $ArtifactMap.Keys) {
        if (Test-Path $Source) {
            Copy-Item -Path $Source -Destination $ArtifactMap[$Source] -Force
        } else {
            throw "Missing build artifact: $Source"
        }
    }
    Write-Host "[OK] Artifact synchronization verified." -ForegroundColor Green

    # --- Phase 4: Automated Regression Testing ---
    Write-Host "[STAGE 4/5] Executing TCB Regression Suite..." -ForegroundColor Yellow
    cargo test --workspace --release
    if ($LASTEXITCODE -ne 0) { throw "Regression tests failed. Aborting deployment." }

    # --- Phase 5: Secure Provisioning & Service Ignition ---
    Write-Host "[STAGE 5/5] Injecting Secrets & Initializing Host..." -ForegroundColor Green

    # Load Environment Variables from .env (Security Compliance)
    if (Test-Path ".env") {
        Get-Content ".env" | Where-Object { $_ -match "^[^#].*=" } | ForEach-Object {
            $key, $value = $_ -split '=', 2
            [System.Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim(), "Process")
        }
        Write-Host "[OK] Environment variables loaded." -ForegroundColor Green
    } else {
        Write-Host "[WARN] .env file not found. Falling back to system environment." -ForegroundColor Yellow
    }

    # Critical Telemetry Defaults
    if (-not $env:RUST_LOG) { $env:RUST_LOG = "info,serein_core=debug,serein_server=debug" }

    # Provider Configuration Validation
    $ProvidersConfig = if ($env:PROVIDERS_CONFIG) { $env:PROVIDERS_CONFIG } else { "config/providers.toml" }
    if (-not (Test-Path $ProvidersConfig)) {
        Write-Host "[WARN] Provider configuration not found at $ProvidersConfig - falling back to environment variables." -ForegroundColor Yellow
        Write-Host "[INFO] Set PROVIDERS_CONFIG env var or create config/providers.toml for dynamic provider loading." -ForegroundColor Gray
    } else {
        Write-Host "[OK] Provider configuration loaded from $ProvidersConfig" -ForegroundColor Green
    }

    if (-not (Test-Path "modules\intent_engine.wasm")) {
        throw "Post-build integrity check failed: modules\intent_engine.wasm not found."
    }
    Write-Host "[OK] Post-build WASM integrity verified: modules\intent_engine.wasm present." -ForegroundColor Green

    Write-Host "==============================================================================" -ForegroundColor Cyan
    Write-Host "[SUCCESS] DAEMON ACTIVE: Serein Gateway Ingress ready." -ForegroundColor Green
    
    cargo run --release -p serein-gateway --bin serein-gateway
    if ($LASTEXITCODE -ne 0) { throw "Gateway daemon exited abnormally." }

} catch {
    Write-Host "`n[FATAL] Provisioning Pipeline Halted." -ForegroundColor Red
    Write-Host "Details: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}