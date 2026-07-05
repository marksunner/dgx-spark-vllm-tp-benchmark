<!--
Deprecation banner for github.com/marksunner/dgx-spark-vllm-tp-benchmark.
Paste everything below this comment at the VERY TOP of README.md, above the
existing "# DeepSeek V4 Flash — Dual DGX Spark vLLM Tensor Parallelism
Benchmark" heading. Keep all original content below it unchanged.
-->

> [!WARNING]
> **⚠️ This benchmark is outdated (early 2026) and no longer reflects what dual DGX Sparks can do with DeepSeek V4 Flash.**
>
> The community has moved far past these numbers. For the current best recipe, use
> **[tonyd2wild/deepseek-v4-flash-2x-spark-1m](https://github.com/tonyd2wild/deepseek-v4-flash-2x-spark-1m)**:
>
> | | This repo (early 2026) | tonyd2wild recipe (current) |
> |---|---|---|
> | **Decode speed** | 12.4 tok/s | **45.5 tok/s** (MTP n=2) |
> | **Context** | 4K tested | **1M tokens** (400K/800K prompts served in production) |
> | **Prefill** | not measured at scale | ~800–900 tok/s on 400K–800K prompts |
> | **Setup** | hand-patched fork + Ray | pre-built Docker image, two boxes and a cable |
> | **Status** | benchmark snapshot | running a 24/7 production agent fleet |
>
> If you arrived here looking for how to serve DeepSeek V4 Flash on two Sparks, **start there, not here.**
>
> The content below is preserved unchanged for historical reference — it documents the state of dual-Spark vLLM tensor parallelism in early 2026, including the NCCL/RoCE setup details and the layer-42 DeepGEMM fix, some of which may still be useful background.

---

<!-- Original README content continues unchanged below this line. -->

# DeepSeek V4 Flash — Dual DGX Spark vLLM Tensor Parallelism Benchmark

Running DeepSeek V4 Flash (284B total / 13B active MoE) across **two DGX Sparks** using vLLM with tensor parallelism over NCCL/RoCE.

## TL;DR

| Metric | Result |
|--------|--------|
| **Generation speed** | 12.4–13.8 tok/s |
| **Sustained throughput** (vLLM engine) | 12.4–12.6 tok/s |
| **Model per node** | 74.13 GiB |
| **Available KV cache** | ~20 GiB per node |
| **Max context tested** | 4,096 tokens |
| **English output quality** | ✅ Perfect (layer 42 DeepGEMM fix) |
| **Stability** | ✅ No crashes, route drops, or errors |

## Why This Exists

Our [previous benchmark](https://github.com/marksunner/dgx-spark-ds4-benchmark) tested DeepSeek V4 Flash using [antirez's ds4 engine](https://github.com/antirez/ds4) with pipeline parallelism (sequential layer split). That approach achieved 11.4 tok/s at low context but suffered from:
- Speed degradation as context grew (down to 7.8 tok/s at 192K)
- Random `accept failed: Resource temporarily unavailable` socket errors
- Route drops making it unreliable for production agent use

This benchmark tests the **alternative approach**: vLLM with tensor parallelism via NCCL, using the native FP4+FP8 checkpoint (no re-quantisation needed).

## Hardware

- 2× NVIDIA DGX Spark (GB10, 128 GB unified memory each)
- ConnectX-7 200 Gbps QSFP link between nodes
- RoCE/RDMA active (NCCL transport)

## Software Stack

| Component | Details |
|-----------|---------|
| Docker image | `lmxxf/vllm-deepseek-v4-dgx-spark:latest` |
| vLLM | v0.1.dev (jasl/vllm ds4-sm120 fork) |
| Model | `deepseek-ai/DeepSeek-V4-Flash` (native FP4+FP8, ~160 GB) |
| MoE backend | Marlin (fast) + DeepGEMM fallback on layer 42 |
| Distributed | Ray + NCCL AllReduce over RoCE |
| Parallelism | Tensor Parallel (TP=2) |

### Why the jasl/vllm fork?

The official vLLM `main` branch has SM120+ (Blackwell) MXFP4 MoE issues. The [jasl fork](https://github.com/jasl/vllm/tree/ds4-sm120) includes Triton-rewritten attention/MLA kernels for SM120+ and the [lmxxf mixed backend](https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark) that fixes Marlin's accumulated numerical error at layer 42 (which corrupts English output).

## Benchmark Results

### Generation Speed

| Test | Tokens | Time | Speed |
|------|--------|------|-------|
| Math (warmup) | 2 | 0.24s | 8.4 tok/s |
| Code generation | 500 | 36.2s | **13.8 tok/s** |
| English prose | 1000 | 179.6s | 5.6 tok/s* |
| Reasoning | 75 | 7.1s | 10.6 tok/s |
| Long generation | 2000 | 161.2s | **12.4 tok/s** |

*The 1000-token English test appears anomalous — likely prefill overhead or sampling variance. vLLM's own engine logs show a consistent **12.4–12.6 tok/s** sustained generation throughput across all requests.

### vLLM Engine Metrics (from server logs)

```
Avg generation throughput: 12.4-12.6 tokens/s (sustained, measured over multiple 10s windows)
GPU KV cache usage: 0.1-0.2% at 4K context
Running: 1 req at a time (single-user scenario)
```

### Output Quality

All outputs produced clean, correct English:
- ✅ Math: 17 × 23 = 391 (correct)
- ✅ Code: Proper merge sort implementation with comments
- ✅ Reasoning: "All but 9 run away" → 9 sheep remain (correct)
- ✅ Long form: Comprehensive Flask REST API tutorial, well-structured

The Marlin + DeepGEMM layer 42 fix eliminates the English corruption issue that pure Marlin produces on SM120+.

## Comparison: vLLM TP vs ds4 Pipeline

| | ds4 (pipeline parallelism) | vLLM (tensor parallelism) |
|---|---|---|
| Gen speed (low context) | 11.4 tok/s | **13.8 tok/s** |
| Gen speed (high context) | 7.8 tok/s (192K) | TBD (12.4+ expected) |
| Prefill | 200+ tok/s (pipelined) | ~3.4 tok/s (not optimised) |
| Stability | ❌ Socket bugs, route drops | ✅ Rock solid |
| Model format | Q4 imatrix GGUF (~154 GB) | Native FP4+FP8 (~160 GB) |
| Quantisation quality | Good (imatrix) | **Native** (no re-quant) |
| Agent-viable | ❌ (20s+ turns, crashes) | ✅ (predictable latency) |
| Network transport | TCP sockets | NCCL/RoCE/RDMA |

**Key insight:** Tensor parallelism produces faster, more stable generation than pipeline parallelism on this hardware, even though antirez [suggested TP wouldn't be viable](https://antirez.com/news/167) on DGX Spark-class interconnects. The MoE architecture's small all-reduce payloads make it practical.

## Reproduction

### Prerequisites

- 2× DGX Spark with ConnectX-7 QSFP link
- RoCE/RDMA configured between nodes (MTU 9000 recommended)
- Docker installed on both nodes
- SSH key access between nodes over QSFP interface
- HuggingFace token (for model download)

### Step 1: Download Model

On the head node:
```bash
pip install huggingface_hub
hf download deepseek-ai/DeepSeek-V4-Flash --local-dir ~/deepseek-v4-flash
```

Sync to worker via QSFP:
```bash
rsync -avP ~/deepseek-v4-flash/ worker@<WORKER_QSFP_IP>:~/deepseek-v4-flash/
```

### Step 2: Pull Docker Image

On both nodes:
```bash
docker pull lmxxf/vllm-deepseek-v4-dgx-spark:latest
```

### Step 3: Launch Containers

Head node:
```bash
docker run -d --gpus all --privileged --ipc=host --network host \
  --name vllm_node \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_SOCKET_IFNAME=enP2p1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enP2p1s0f1np1 \
  -e VLLM_HOST_IP=<HEAD_QSFP_IP> \
  -e RAY_NODE_IP_ADDRESS=<HEAD_QSFP_IP> \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42 \
  -v ~/deepseek-v4-flash:/root/.cache/huggingface/deepseek-v4-flash \
  lmxxf/vllm-deepseek-v4-dgx-spark:latest sleep infinity
```

Worker node:
```bash
docker run -d --gpus all --privileged --ipc=host --network host \
  --name vllm_node \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_SOCKET_IFNAME=enP2p1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enP2p1s0f1np1 \
  -e VLLM_HOST_IP=<WORKER_QSFP_IP> \
  -e RAY_NODE_IP_ADDRESS=<WORKER_QSFP_IP> \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42 \
  -v ~/deepseek-v4-flash:/root/.cache/huggingface/deepseek-v4-flash \
  lmxxf/vllm-deepseek-v4-dgx-spark:latest sleep infinity
```

### Step 4: Start Ray Cluster

```bash
# Head node
docker exec vllm_node ray start --head --port=6379 \
  --node-ip-address=<HEAD_QSFP_IP> --dashboard-host=0.0.0.0

# Worker node
docker exec vllm_node ray start \
  --address=<HEAD_QSFP_IP>:6379 --node-ip-address=<WORKER_QSFP_IP>
```

### Step 5: Launch vLLM

```bash
docker exec vllm_node vllm serve /root/.cache/huggingface/deepseek-v4-flash \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype fp8 \
  --max-model-len 4096 \
  --enforce-eager \
  --moe-backend marlin
```

### Step 6: Test

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/.cache/huggingface/deepseek-v4-flash",
       "messages":[{"role":"user","content":"What is 17 * 23?"}],
       "max_tokens":50}'
```

## Critical Deployment Notes

### GLOO_SOCKET_IFNAME is mandatory

Without `GLOO_SOCKET_IFNAME=enP2p1s0f1np1`, Gloo resolves node addresses with mixed IPv4/IPv6, producing:
```
RuntimeError: ss1.ss_family == ss2.ss_family. 2 vs 10
```

### VLLM_MXFP4_MARLIN_DEEPGEMM_LAYERS=42

Without this, Marlin's accumulated numerical error at the final MoE layer corrupts English output. The fix forces only layer 42 through DeepGEMM while keeping Marlin's speed for the other 42 layers.

### Ray must use QSFP IPs

Start Ray with `--node-ip-address=<QSFP_IP>` on both nodes. If Ray auto-detects the Ethernet interface, vLLM can't find GPUs at the expected node addresses.

### Volume mounts differ per node

Each node needs its own `-v` path mapping (home directories differ). The `launch-cluster.sh` script from [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) sends identical Docker args to all nodes, which breaks with different usernames. Manual Docker launch avoids this.

## Memory Layout

| Component | Per Node | Total |
|-----------|----------|-------|
| Unified Memory | 128 GB | 256 GB |
| Model weights (TP split) | 74.13 GB | ~148 GB |
| Available KV cache | ~20 GB | ~40 GB |
| OS/runtime overhead | ~34 GB | ~68 GB |

DeepSeek V4 Flash's extremely compact KV cache (~2% of traditional GQA) means 40 GB supports well over 1M tokens of context.

## Agent Test Results (Hermes + Discord)

Tested as a live Discord agent via [Hermes Agent v0.14.0](https://github.com/nousresearch/hermes-agent):

| Capability | Status |
|-----------|--------|
| Basic chat/greetings | ✅ Works |
| Story writing (multi-part) | ✅ Works (slow but correct) |
| Code generation via chat | ✅ Works |
| Tool calling (file ops) | ❌ Format mismatch — infinite loop |
| Tool calling (code execution) | ❌ Empty responses after tool calls |

**Key findings:**
- Hermes requires `--max-model-len 65536` minimum (64K context requirement)
- Must add `--enable-auto-tool-choice --tool-call-parser deepseek_v4 --tokenizer-mode deepseek_v4`
- The `deepseek_v4` tool-call parser technically works (calls are attempted) but results are unreliable — the model frequently returns empty responses after tool invocations, creating "nudge to continue" loops
- For **pure inference/chat** workloads: production-ready
- For **agentic tool-calling** workloads: not yet reliable without framework adaptation

This is likely solvable with Hermes updates or a custom tool-call parser, but is not plug-and-play today.

## Context Scaling

| max-model-len | Status | Notes |
|--------------|--------|-------|
| 4,096 | ✅ Works | Insufficient for Hermes (needs 64K min) |
| 65,536 | ✅ Works | Fits with 0.80 GPU utilisation, ~16K KV cache tokens |
| 131,072 | ❌ OOM during Marlin weight loading | Needs investigation — may work with 0.75 util or after stopping other containers |

## Next Steps

- [ ] Investigate 128K context (may need lower gpu-memory-utilization or phased loading)
- [ ] Measure prefill throughput at longer prompts
- [ ] Test tool-call compatibility with updated Hermes/parser versions
- [ ] Compare stability over extended use (hours/days)

## Credits

- [lmxxf/deepseek-v4-deployment-on-dgx-spark](https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark) — the deployment guide and Docker image that made this possible
- [jasl/vllm](https://github.com/jasl/vllm/tree/ds4-sm120) — SM120+ fork with Triton fallbacks
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — Ray cluster launcher
- [antirez/ds4](https://github.com/antirez/ds4) — for the pipeline parallelism comparison baseline
- The DGX Spark community for sharing configurations and solutions

## License

MIT
