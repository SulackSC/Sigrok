# Model Runtime Notes

## Active Profile (Qwen3.5-9B · single GPU)

Model: Qwen 3.5 · 9B · multimodal (vision)

Weights: Q4_K_M (~5.3 GiB)
`Qwen3.5-9B-Q4_K_M.gguf`

Vision: mmproj F16 (~876 MiB)
`mmproj-F16.gguf`

Runtime · Server: llama.cpp `llama-server` (build c46758d)
· alias `qwen3.5-9b` (matches `settings.toml` → `llamacpp/qwen3.5-9b`)

· Context: 16384 (`-c 16384`) · batch `-b 512` · ubatch `-ub 256` · flash-attn on · parallel 1
· GPU: layers 999 · full GPU offload · no tensor-split
· KV cache: Q4_1 (`--cache-type-k q4_1`, `--cache-type-v q4_1`)
· Thinking: off (`--reasoning off`, `--reasoning-budget 0`)

Hardware: RTX 4060 Ti (~8 GiB VRAM, ~7805 MiB reported by llama-server)

Systemd unit: `deploy/systemd/llama-server.qwen35-9b.service` · cutover script: `switch-to-qwen35-9b.sh`

### Files

- Text model: `/home/sulack/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf`
- Vision projector: `/home/sulack/models/Qwen3.5-9B-GGUF/mmproj-F16.gguf`

### Runtime

```bash
/home/sulack/llama.cpp/build/bin/llama-server \
  -m /home/sulack/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf \
  --mmproj /home/sulack/models/Qwen3.5-9B-GGUF/mmproj-F16.gguf \
  -a qwen3.5-9b \
  --n-gpu-layers 999 \
  -c 16384 \
  -b 512 \
  -ub 256 \
  --flash-attn on \
  --cache-type-k q4_1 \
  --cache-type-v q4_1 \
  --parallel 1 \
  --host 127.0.0.1 \
  --port 8081 \
  --api-key sk-no-key-required \
  --reasoning off \
  --reasoning-budget 0
```

### Sigrok Settings

```toml
[genai]
model = "llamacpp/qwen3.5-9b"
base_url = "http://127.0.0.1:8081"
temperature = 0.7
repeat_penalty = 1.15
request_timeout = 180

[genai.tokens]
limit = 7000
overhead_max = 20
prompt_max = 100
output_max = 1024
```

### Notes

- Rolled back from Qwen3.6 35B-A3B / 27B after removing the second GPU (5060); dense 27B and MoE 35B no longer fit on a single 8 GiB card.
- At idle after load, expect ~6.5–6.6 GiB VRAM used with ~1.2 GiB headroom on the 4060 Ti.
- Previous dual-GPU 27B profile used partial CPU spill (`--n-gpu-layers 54`, tensor-split 0.52/0.48, 8192 context); see git history / `switch-to-qwen36-27b-safe.sh` if restoring two GPUs.

## Known-Good 35B-A3B Profile (archived · dual GPU)

This is the original Qwen3.6 35B-A3B setup that ran reliably and responded quickly before the 27B, IQ1, long-context, and thinking experiments.

### Identity

- Model: Qwen3.6 35B-A3B, MoE, multimodal
- Alias: `qwen3.6-35b-a3b`
- Sigrok config model: `llamacpp/qwen3.6-35b-a3b`
- Endpoint: `http://127.0.0.1:8081`

### Files

- Text model: `/home/sulack/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`
- Vision projector: `/home/sulack/models/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf`

### Runtime

```bash
/home/sulack/llama.cpp/build/bin/llama-server \
  -m /home/sulack/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  --mmproj /home/sulack/models/Qwen3.6-35B-A3B-GGUF/mmproj-F16.gguf \
  -a qwen3.6-35b-a3b \
  --n-gpu-layers 999 \
  --tensor-split 0.48,0.52 \
  -c 8192 \
  -b 512 \
  -ub 256 \
  --flash-attn on \
  --parallel 1 \
  --host 127.0.0.1 \
  --port 8081 \
  --api-key sk-no-key-required \
  --reasoning off \
  --reasoning-budget 0
```

### Sigrok Settings

```toml
[genai]
model = "llamacpp/qwen3.6-35b-a3b"
base_url = "http://127.0.0.1:8081"
temperature = 0.7
repeat_penalty = 1.15
request_timeout = 180

[genai.tokens]
limit = 7000
overhead_max = 20
prompt_max = 100
output_max = 1024
```

### Notes

- No KV cache quant flags were used in the original stable profile.
- Thinking was explicitly disabled with `--reasoning off --reasoning-budget 0`.
- Context was `8192`, not `32768` or `65536`.
- The model was fully offloaded across the two GPUs with `--n-gpu-layers 999`.
- This profile was much faster than the dense 27B partial-CPU setup and more reliable than the IQ1 thinking experiments.

