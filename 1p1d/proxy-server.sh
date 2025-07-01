#!/bin/bash

: "${PREFILL_PORT:?Error: PREFILL_PORT must be set (e.g. export PREFILL_PORT=8000)}"
: "${DECODE_PORT:?Error: DECODE_PORT must be set (e.g. export DECODE_PORT=8001)}"

DECODE_IP=$(getent ahosts decode-0.decode-headless.default.svc.cluster.local | awk '{print $1}' | uniq | head -n1 || true)
PREFILL_IP=$(getent ahosts prefill-0.prefill-headless.default.svc.cluster.local | awk '{print $1}' | uniq | head -n1 || true)

neuron-proxy-server \
    --prefill-ip "$PREFILL_IP" \
    --decode-ip "$DECODE_IP" \
    --prefill-port "$PREFILL_PORT" \
    --decode-port "$DECODE_PORT"
