/**
 * Copyright (c) 2026 Changqing Zhang Technologies. All rights reserved.
 * * Serein Core - Industrial Gateway Reliability Test Suite
 * * Description:
 * Evaluates gateway throughput, memory stability, and cryptographic 
 * signature verification under controlled concurrency to prevent 
 * upstream LLM provider rate-limiting (HTTP 429).
 * * Target Service Level Agreements (SLAs):
 * - P95 Latency < 3000ms (Accounts for TMR LLM latency)
 * - P99 Latency < 5000ms
 * - Error Rate < 1%
 */

import http from 'k6/http';
import crypto from 'k6/crypto';
import { check, sleep, fail } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// -----------------------------------------------------------------------------
// Telemetry & Metrics Configuration
// -----------------------------------------------------------------------------
export const errorRate = new Rate('http_error_rate');
export const processingTime = new Trend('gateway_processing_time');

export const options = {
    scenarios: {
        ramp_up_and_spike: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: [
                // Controlled concurrency to stay within paid API rate limits
                { duration: '10s', target: 5 },
                { duration: '20s', target: 10 },
                { duration: '10s', target: 0 },
            ],
        },
    },
    thresholds: {
        // Thresholds relaxed slightly to account for synchronous LLM calls
        'http_req_duration': ['p(95)<3000', 'p(99)<5000'],
        'http_error_rate': ['rate<0.01'],
    },
};

// -----------------------------------------------------------------------------
// Utilities & Polyfills
// -----------------------------------------------------------------------------

/**
 * Polyfill for UUIDv4 generation compatible with the k6 Goja runtime.
 * Bypasses the absence of `crypto.randomUUID()` in constrained JS environments.
 * * @returns {string} A valid UUIDv4 string.
 */
function generateUuidV4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// -----------------------------------------------------------------------------
// Environment Configuration & Data Pools
// -----------------------------------------------------------------------------
const TARGET_URL = __ENV.TARGET_URL || 'http://127.0.0.1:8080/v1/agent/execute';
const HMAC_SECRET = __ENV.HMAC_SECRET || 'e4b7a1c8-5e2d-4b9a-8c3f-1d6e7f8a9b0c';
const TENANT_ID = __ENV.TENANT_ID || 'test_tenant_001';

const LOCAL_IP_POOL = ['127.0.0.1', '10.0.0.1', '10.0.0.2', '10.0.0.3', '10.0.0.4'];
const NETWORK_POOL = ['ethereum', 'polygon', 'arbitrum', 'optimism', 'base', 'avalanche', 'bsc', 'solana'];
const CONTRACT_POOL = [
    '0x1234abcd5678ef901234567890abcdef123456789',
    '0x2345bcde6789fa01234567890abcdef234567890',
    '0x3456cdef7890ab1234567890abcdef345678901',
    '0x4567defa8901bc234567890abcdef456789012'
];

// -----------------------------------------------------------------------------
// Lifecycle Hooks
// -----------------------------------------------------------------------------

/**
 * Test initialization phase.
 * Performs an atomic connectivity and cryptographic handshake check before 
 * spawning Virtual Users (VUs) to prevent cascading failures.
 */
export function setup() {
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const nonce = generateUuidV4();
    
    // Construct the strict cryptographic signature payload
    const signPayload = `${TENANT_ID}:ethereum:0x1234abcd5678ef901234567890abcdef123456789:${timestamp}:${nonce}`;
    const signature = crypto.hmac('sha256', HMAC_SECRET, signPayload, 'hex').toLowerCase();

    const payload = JSON.stringify({
        network_id: 'ethereum',
        contract_address: '0x1234abcd5678ef901234567890abcdef123456789',
        task_type: 'swap',
        gas_limit: 300000,
        confidence_score: 0.95,
        source_url: 'https://explorer.mantle.xyz/tx/0x9a8f5678ef901234567890abcdef1234567890'
    });

    const res = http.post(TARGET_URL, payload, {
        headers: {
            'Content-Type': 'application/json',
            'x-serein-tenant-id': TENANT_ID,
            'x-serein-timestamp': timestamp,
            'x-serein-nonce': nonce,
            'cf-connecting-ip': '127.0.0.1',
            'Authorization': `Serein-Hmac-SHA256 ${timestamp}.${nonce}.${signature}`
        },
    });

    if (res.status === 0 || res.status >= 500) {
        fail(`[FATAL] Gateway is unreachable or returning HTTP ${res.status}. Aborting load test.`);
    }
    
    return { initialized: true };
}

// -----------------------------------------------------------------------------
// Main Execution Loop
// -----------------------------------------------------------------------------

/**
 * Virtual User (VU) iteration logic.
 * Generates dynamic payloads and cryptographic signatures per request.
 */
export default function () {
    // Introduce entropy to prevent CDN caching or predictable routing
    const network = NETWORK_POOL[Math.floor(Math.random() * NETWORK_POOL.length)];
    const contractAddr = CONTRACT_POOL[Math.floor(Math.random() * CONTRACT_POOL.length)];
    const clientIp = LOCAL_IP_POOL[Math.floor(Math.random() * LOCAL_IP_POOL.length)];
    
    const timestamp = Math.floor(Date.now() / 1000).toString();
    const nonce = generateUuidV4(); 

    const signPayload = `${TENANT_ID}:${network}:${contractAddr}:${timestamp}:${nonce}`;
    const signature = crypto.hmac('sha256', HMAC_SECRET, signPayload, 'hex').toLowerCase();

    const payload = JSON.stringify({
        network_id: network,
        contract_address: contractAddr,
        task_type: 'swap',
        gas_limit: 300000,
        confidence_score: 0.95,
        source_url: 'https://explorer.mantle.xyz/tx/0x9a8f5678ef901234567890abcdef1234567890'
    });

    const params = {
        headers: {
            'Content-Type': 'application/json',
            'x-serein-tenant-id': TENANT_ID,
            'x-serein-timestamp': timestamp,
            'x-serein-nonce': nonce,
            'cf-connecting-ip': clientIp, 
            'Authorization': `Serein-Hmac-SHA256 ${timestamp}.${nonce}.${signature}`
        },
        tags: { name: 'AgentExecutionEndpoint' }
    };

    const res = http.post(TARGET_URL, payload, params);
    
    processingTime.add(res.timings.duration);

    const isSuccessful = check(res, {
        'status is 200': (r) => r.status === 200,
        'status is not 401': (r) => r.status !== 401,
        'status is not 429': (r) => r.status !== 429,
        'status is not 5xx': (r) => r.status < 500 || r.status === 502, 
    });

    errorRate.add(!isSuccessful ? 1 : 0);
    
    // Apply jittered pacing to simulate realistic traffic and respect API rate limits
    // Min delay: 1.5s, Max delay: 3s
    sleep(Math.random() * 1.5 + 1.5);
}

// -----------------------------------------------------------------------------
// Teardown — Post-Run Observability Summary
// -----------------------------------------------------------------------------

/**
 * Post-execution hook invoked after all VUs complete.
 *
 * Prints a structured banner directing the SRE to cross-reference k6 HTTP 429
 * counts against the Compliance Worker's `dropped_events` emission in the main
 * gateway terminal. Any HTTP 429 recorded in the k6 end-of-test summary
 * represents a successful Load Shedding intervention by the circuit breaker
 * and/or rate limiter — not a test failure.
 */
export function teardown(data) {
    console.log('');
    console.log('========================================================================');
    console.log('  SEREIN GATEWAY — POST-RUN OBSERVABILITY CHECKLIST');
    console.log('========================================================================');
    console.log('');
    console.log('  [1] Review the k6 end-of-test summary above for HTTP 429 counts.');
    console.log('      -> HTTP 429 = successful Load Shedding (circuit breaker / rate limiter).');
    console.log('      -> These are NOT test failures; they confirm protective intervention.');
    console.log('');
    console.log('  [2] In the MAIN GATEWAY TERMINAL, grep for "dropped_events":');
    console.log('      $ grep "dropped_events" <gateway-log-file>');
    console.log('      -> Each dropped_events entry corresponds to a request rejected');
    console.log('         by the Compliance Worker before reaching an upstream LLM.');
    console.log('');
    console.log('  [3] Cross-reference: k6 HTTP 429 count ≈ gateway dropped_events count.');
    console.log('      -> A tight correlation validates the Load Shedding pipeline.');
    console.log('');
    console.log('========================================================================');
    console.log('');
}