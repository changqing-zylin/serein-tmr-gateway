# ── Builder Stage ───────────────────────────────────────────────────────────
FROM rustlang/rust:nightly-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Cargo.toml Cargo.lock* ./
COPY apps/ ./apps/
COPY crates/ ./crates/
COPY plugins/ ./plugins/

RUN cargo build --release -p serein-gateway

# ── Runtime Stage ───────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash --uid 1000 serein
WORKDIR /home/serein

COPY config/providers.toml ./providers.toml
COPY --from=builder /app/target/release/serein-gateway /usr/local/bin/serein-gateway

USER serein

EXPOSE 8080

ENTRYPOINT ["serein-gateway"]