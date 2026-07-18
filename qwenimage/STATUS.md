# Krea 2 Turbo Pure-Go Spike — STATUS

**Date:** 2026-07-18 · **Branch:** `feat/krea2-image` · **Machine:** 16-core Apple Silicon, 64 GB

## Verdict

(finalized after THE RUN — see bottom)

## Parity ladder (all gates green)

| Component | Method | Result |
| --- | --- | --- |
| GGUF loader (`internal/gguf`) | vs gguf-py dequant, real Q4_K_M/F32/F16 tensors | **≤ 1e-6** (no loader changes needed — upstream code was already generic) |
| Qwen2 BPE tokenizer | vs transformers, 8 cases (whitespace lookahead, emoji, CJK, chat template) | **exact match** |
| Qwen3-VL-4B text encoder (36L, f32) | vs torch fixture, same Q8_0-dequant weights, fixed 546-token layout | **maxAbs 2.5e-4** |
| Krea2 DiT, tiny config | vs random-weight diffusers Krea2Transformer2DModel | **1.2e-6** (f32) / 4.2e-4 (deliberate f16 storage) |
| Krea2 DiT, full 12B single step | vs torch fixture, same Q4_K_M-dequant weights, seq 1536 | **maxAbs 2.3e-4** (f32) |
| Wan 2.1 VAE decoder | vs diffusers AutoencoderKLWan, seeded latents 512² | **maxAbs 3.4e-6** (2D reduction exact; initial 0.5 divergence was the missing [-1,1] clamp) |
| Flow-match Euler scheduler | vs FlowMatchEulerDiscreteScheduler, 9 sigmas | **≤ 1e-7 each** |
| Python harness self-validation | e2e fox from the same GGUF weights (CPU f32) | recognizable fox ✔ |
| sd.cpp oracle | Metal, cfg-scale 1.0 | recognizable fox ✔ |

## Performance (unoptimized correctness-first CPU code)

| Stage | Time | Notes |
| --- | --- | --- |
| Text encoder load (Q8→f32) | 17.7 s | 4B params |
| Text encoder forward (seq 546) | 76.5 s | parallel naive matmul, ~62 GFLOPS effective |
| DiT load (Q4→f32) | 83 s | 12B params |
| DiT forward f32 (seq 1536) | 11 m 09 s | **swap-bound at 69 GB heap** — motivated f16 storage |
| VAE decode 512² | 34 s | naive conv loops |
| THE RUN (f16 DiT, sequential loads) | (recorded below) | |

Optimization headroom (per the Supertonic playbook, none applied yet beyond
row-parallelism): register-tiled/blocked matmul, im2col conv, on-the-fly
dequant (skip the f32/f16 intermediate entirely and matmul straight from
Q4_K blocks — halves load time and another 3× memory), WebGPU backend.

## Ops/kernels added (all pure Go, `qwenimage` package)

Byte-level BPE with hand-rolled HF-regex pre-tokenizer (incl. `(?!\S)`
emulation); rotate-half RoPE with explicit positions; FLUX interleaved-pair
3-axis RoPE; zero-centered RMSNorm; WanRMS channel-L2 norm; GQA attention
(causal + non-causal, key-padding masks, sigmoid output gate); SwiGLU;
gelu-tanh; cos-first sinusoidal timestep embedding; shared-modulation AdaLN;
text-fusion transformer (layer-axis attention + projector); 2D conv (same-pad,
1×1, 3×3) with 3D-causal→2D load-time reduction; nearest-exact 2× upsample;
single-head spatial attention; flow-match Euler with constant-μ time shift;
f16 weight storage with 64K decode table.

## Key discoveries (vs pre-spike assumptions)

1. Krea 2 = **its own single-stream DiT** (28 blocks, shared modulation,
   sigmoid-gated attention, 12-layer text-stack fusion) — not qwen_image
   dual-stream. sd.cpp `krea2.hpp` (783 lines) was the decisive blueprint.
2. Text encoder is **Qwen3-VL 4B** (not Qwen2.5-VL 7B).
3. Conditioning is a **fixed 546-token middle-padded block** with
   cumulative-valid position ids; hidden states tapped from layers
   {2,5,…,35}; first 34 dropped.
4. Distilled Turbo uses **constant shift μ=1.15** (`is_distilled`); the
   resolution-dynamic shift is only for Krea-2-Raw.
5. sd.cpp `--cfg-scale 0.0` = unconditioned (prompt ignored); Turbo mode
   there is `1.0`. In diffusers, guidance-off is literally `0.0`.
6. For single-frame t2i the Wan video VAE **reduces exactly to 2D**
   (causal left-pad is all zeros; upsample3d skips time_conv on first chunk).
7. mRoPE with equal per-axis positions reduces to standard 1D RoPE.
8. 12B f32 does NOT fit a 64 GB machine alongside activations (69 GB heap,
   swap-bound) — f16 weight storage (24 GB) is the floor for this hardware;
   Q4 on-the-fly matmul is the next step down.

## Model files (`~/.cache/manifold/krea2-models/`, SHAs in reference/ORACLE.md)

DiT `Krea-2-Turbo-Q4_K_M.gguf` (7.2 GB, realrebelai/KREA-2_GGUFs) ·
encoder `Qwen3VL-4B-Instruct-Q8_0.gguf` (4.3 GB, Qwen official) ·
VAE `wan-vae-decoder.gguf` (converted from Comfy-Org wan_2.1_vae.safetensors
by `reference/convert_vae.py`) · `tokenizer/tokenizer.json` (Qwen3-VL-4B).

## Deviations from plan

- T4/T5 order swapped opportunistically (T4 needed no code).
- The injected-noise full-pipeline test was skipped: every component is
  individually parity-gated and an extra e2e run costs ~an hour of compute;
  THE RUN plus the visual oracle comparison serves as the end gate.
- MPS/fp16 harness path abandoned (Metal mixed-dtype matmul asserts);
  CPU f32 sequential harness with a cached state dict replaced it.

## Go/no-go recommendation for Manifold productization

(finalized after THE RUN)
