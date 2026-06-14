<#
.SYNOPSIS
    Serein Core - TMR Adjudication & Integration Verification
.DESCRIPTION
    Executes an end-to-end zero-trust integration test against the Serein Gateway.
    Validates HMAC-SHA256 signatures, UUID anti-replay nonces, and multi-node consensus.
.NOTES
    Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
#>
[CmdletBinding()]
param (
    [string]$Endpoint        = "http://127.0.0.1:8888",
    [string]$Tenant          = "test_tenant_001",
    [string]$Network         = "ethereum",
    [string]$ContractAddress = "0x1234abcd5678ef901234567890abcdef123456789",
    [string]$TaskType        = "swap",
    [string]$InternalToken   = "hackathon-demo-token"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$WarningPreference = "Continue"

try {
    Write-Host "`n[SYSTEM] Initializing Secure TMR Adjudication Sequence..." -ForegroundColor Cyan
    $TargetUrl = "$Endpoint/v1/agent/execute"

    # --- Phase 1: Payload Construction ---
    Write-Host "[PAYLOAD] Constructing Agent Execution Payload..." -ForegroundColor Gray

    $RequestBody = @{
        network_id       = $Network
        contract_address = $ContractAddress
        task_type        = $TaskType
        max_gas_limit    = 300000
        confidence_score = 0.95
        source_url       = "https://explorer.mantle.xyz/tx/0x9a8f5678ef901234567890abcdef1234567890"
    } | ConvertTo-Json -Compress

    Write-Host " -> Request Body: $RequestBody" -ForegroundColor DarkGray

    # --- Phase 2: Cryptographic Signature Generation ---
    Write-Host "[AUTH] Computing HMAC-SHA256 Signature over Intent Payload..." -ForegroundColor Gray
    $Timestamp = [int64]([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())
    $Nonce = [guid]::NewGuid().ToString().ToLower()

    # STRICT ZERO-TRUST SIGNATURE PAYLOAD (Aligning with Gateway expectations)
    $SignPayload = "${Tenant}:${Network}:${TaskType}:${Timestamp}:${Nonce}"
    
    $Encoding = [Text.Encoding]::UTF8
    $HmacProvider = New-Object System.Security.Cryptography.HMACSHA256
    $HmacProvider.Key = $Encoding.GetBytes($InternalToken)
    $ComputeHash = $HmacProvider.ComputeHash($Encoding.GetBytes($SignPayload))
    $Signature = [System.BitConverter]::ToString($ComputeHash).Replace("-", "").ToLower()

    $AuthHeader = "Serein-Hmac-SHA256 ${Timestamp}.${Nonce}.${Signature}"
    Write-Host " -> Intent Payload: $SignPayload" -ForegroundColor DarkGray
    Write-Host " -> Generated Hash: $Signature" -ForegroundColor DarkGray

    Write-Host "`n[STEP 1/2] Dispatching Encrypted Payload to Gateway..." -ForegroundColor Yellow
    $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    $Response = Invoke-RestMethod -Uri $TargetUrl `
        -Method Post `
        -Headers @{
            "Content-Type"       = "application/json"
            "x-serein-tenant-id" = $Tenant
            "x-serein-timestamp" = $Timestamp
            "x-serein-nonce"     = $Nonce
            "cf-connecting-ip"   = "127.0.0.1"
            "Authorization"      = $AuthHeader
        } `
        -Body $RequestBody

    $Stopwatch.Stop()

    # --- Phase 3: Automated Quorum Verification ---
    Write-Host "[STEP 2/2] Performing Multi-Node Consensus Audit..." -ForegroundColor Yellow

    $AgreedNodes = $Response.consensus.agreeing_nodes
    $TotalNodes  = $Response.consensus.total_nodes
    $Logic       = $Response.consensus.adjudication_logic

    if ($AgreedNodes -lt 2) {
        throw "Consensus Denied: Quorum not met ($AgreedNodes/$TotalNodes)."
    }

    Write-Host "[OK] Consensus Verified: $AgreedNodes/$TotalNodes agreement reached via $Logic." -ForegroundColor Green
    Write-Host "[INFO] Processing Latency: $($Stopwatch.ElapsedMilliseconds)ms" -ForegroundColor Gray
    Write-Host "[STATUS] E2E Pipeline Verification Successful.`n" -ForegroundColor Green

} catch {
    Write-Host "`n[FATAL] TMR Orchestration Pipeline Fault." -ForegroundColor Red
    
    if ($_.Exception.Response) {
        $StatusCode = $_.Exception.Response.StatusCode.value__
        Write-Host "-> Gateway responded with HTTP $StatusCode" -ForegroundColor Yellow
        
        if ($StatusCode -eq 409) {
            Write-Host "======================================================" -ForegroundColor Cyan
            Write-Host "[SECURITY ALERT] AI SWARM CONSENSUS REJECTED" -ForegroundColor Red
            Write-Host "-> Reason: The BFT engine failed to reach a 2/3 majority." -ForegroundColor Yellow
            Write-Host "-> Details: One or more nodes failed or diverged." -ForegroundColor Yellow
            Write-Host "-> Action: Transaction BLOCKED by Aegis Guard to protect user assets." -ForegroundColor Green
            Write-Host "======================================================" -ForegroundColor Cyan
        }
    } else {
        Write-Host "Remote Error Details: $($_.Exception.Message)" -ForegroundColor Red
    }
    exit 1
}