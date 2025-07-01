#!/bin/bash
set -euo pipefail

: "${MAX_SEQ_LEN:?ERROR: MAX_SEQ_LEN must be set (e.g. 1024)}"
: "${VLLM_BATCH:?ERROR: VLLM_BATCH must be set (e.g. 1)}"
: "${TP_DEGREE:?ERROR: TP_DEGREE must be set (e.g. 8)}"
: "${MODEL_ID:?ERROR: MODEL_ID must be set (e.g. meta‐llama/Llama‐3.1‐8B‐Instruct)}"
: "${COMPILED_MODEL_ID:?ERROR: COMPILED_MODEL_ID must be set (e.g. meta‐llama/Llama‐3.1‐8B‐Instruct)}"

echo "Starting vLLM server with:"
echo "  MAX_SEQ_LEN              = $MAX_SEQ_LEN"
echo "  VLLM_BATCH               = $VLLM_BATCH"
echo "  TP_DEGREE                = $TP_DEGREE"
echo "  MODEL_ID               = $MODEL_ID"
echo "  MODEL_ID               = $COMPILED_MODEL_ID"
echo

# Resolve IP addresses
send_ip=$(getent ahosts prefill-0.prefill-headless.default.svc.cluster.local | awk '{print $1}' | uniq | head -n1 || true)
recv_ip=$(getent ahosts decode-0.decode-headless.default.svc.cluster.local | awk '{print $1}' | uniq | head -n1 || true)
echo "running send: $send_ip receive: $recv_ip"

# Export all environment variables with their values
export OUTLINES_CACHE_DIR=/tmp/.outlines
export NEURON_RT_INSPECT_ENABLE=0
export XLA_HANDLE_SPECIAL_SCALAR=1
export UNSAFE_FP8FNCAST=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=2
export MAX_SEQ_LEN="$MAX_SEQ_LEN"
export VLLM_BATCH="$VLLM_BATCH"
export TP_DEGREE="$TP_DEGREE"
export MODEL_ID="$MODEL_ID"
export NEURON_COMPILED_ARTIFACTS="/$COMPILED_MODEL_ID"

export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"
export VLLM_RPC_TIMEOUT=100000
timestamp=$(date '+%Y%m%d_%H%M%S')
mkdir -p logs/vllm_${SEND}/

# DI specific
export KV_IP=$send_ip
export NEURON_SEND_IP=$send_ip
export NEURON_RECV_IP=$recv_ip
export NEURON_RT_ASYNC_SENDRECV_BOOTSTRAP_PORT=45645
export NEURON_RT_ASYNC_SENDRECV_EXPERIMENTAL_ENABLED=1

if [ "$SEND" = "1" ]; then
    PORT=8100
    TRANSFER_CONFIG='{"kv_connector":"NeuronConnector","kv_buffer_device":"cpu","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":2e11,"kv_ip":"'"$KV_IP"'"}'
else
    PORT=8200
    TRANSFER_CONFIG='{"kv_connector":"NeuronConnector","kv_buffer_device":"cpu","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_buffer_size":2e11,"kv_ip":"'"$KV_IP"'"}'
fi

echo "running di server"
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_ID \
    --max-num-seqs $VLLM_BATCH \
    --max-model-len $MAX_SEQ_LEN \
    --tensor-parallel-size $TP_DEGREE \
    --device neuron \
    --use-v2-block-manager \
    --override-neuron-config "{}" \
    --kv-transfer-config $TRANSFER_CONFIG \
    --port ${PORT} 2>&1 | tee logs/vllm_${SEND}/$timestamp.log
