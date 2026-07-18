# Krea 2 Turbo — Oracle Inventory & Architecture Facts

Pinned 2026-07-18 from stable-diffusion.cpp (master, `src/model/diffusion/krea2.hpp`,
`src/conditioning/conditioner.hpp`, `src/stable-diffusion.cpp`) and Hugging Face.
These facts supersede the spec's pre-Phase-0 assumptions (Qwen2.5-VL 7B → actually
Qwen3-VL 4B; dual-stream qwen_image → actually Krea2's own single-stream DiT).

## Model files (`~/.cache/manifold/krea2-models/`)

| File | Size | Source |
| --- | --- | --- |
| `Krea-2-Turbo-Q4_K_M.gguf` | 7.22 GB | `realrebelai/KREA-2_GGUFs` `TURBO/` (repo referenced by sd.cpp docs/krea2.md) |
| `Qwen3VL-4B-Instruct-Q8_0.gguf` | 4.28 GB | `Qwen/Qwen3-VL-4B-Instruct-GGUF` (text encoder; mmproj vision file NOT needed for t2i) |
| `wan_2.1_vae.safetensors` | ~254 MB | `Comfy-Org/Wan_2.1_ComfyUI_repackaged` `split_files/vae/` |
| `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json` | few MB | `Qwen/Qwen3-VL-4B-Instruct` |

SHA-256 sums: recorded after download completes (see below).

## sd.cpp oracle command

Built at `../stable-diffusion.cpp/build/bin/sd-cli` (Metal, Release).

```
sd-cli --diffusion-model ~/.cache/manifold/krea2-models/Krea-2-Turbo-Q4_K_M.gguf \
       --llm ~/.cache/manifold/krea2-models/Qwen3VL-4B-Instruct-Q8_0.gguf \
       --vae ~/.cache/manifold/krea2-models/wan_2.1_vae.safetensors \
       -p "a red fox sitting in fresh snow, golden hour, photorealistic" \
       --steps 8 --cfg-scale 0.0 -W 512 -H 512 -s 42 --diffusion-fa -v \
       -o oracle-512-seed42.png
```

(Exact flags verified against `sd-cli --help` at run time; record final command + timing here.)

## Krea 2 architecture (from `krea2.hpp` — the transliteration blueprint)

**Config (Krea2Config defaults, confirmed by weight detection):** patch_size 2,
in/out_channels 16, features 6144, timestep_dim 256, text_dim 2560, text_layers 12,
layers 28, heads 48 (head_dim 128), kv_heads 12 (GQA), text_heads/text_kv_heads 20,
mlp_multiplier 4, theta 1000, norm_eps 1e-5, RoPE axes_dim {32,48,48} (sum 128).

**Blocks:**
- `KreaRMSNorm`: RMS norm then multiply by `(1 + scale)` — scale stored zero-centered.
- `KreaSwiGLU`: mlp_dim = ceil_to_multiple((2*features/3)*multiplier, 128) = 16384;
  gate/up/down linears, NO bias; silu(gate(x)) * up(x) → down.
- `KreaAttention`: wq/wk/wv/gate/wo linears NO bias; per-head QK RMSNorm (1+scale);
  GQA 48q/12kv; RoPE applied via precomputed `pe`; **output gated:
  attn_out * sigmoid(gate(x))** before wo.
- `KreaDoubleSharedModulation` (per block): learned 6*6144 vector added to shared
  tvec, chunked into 6 mods. `Flux::modulate(ctx, x, shift, scale)` = x*(1+scale)+shift
  (VERIFIED flux.hpp:413); krea2 passes (mods[1], mods[0]) so:
  **mods[0]=scale1, mods[1]=shift1, mods[2]=gate1 (attn); mods[3]=scale2,
  mods[4]=shift2, mods[5]=gate2 (mlp)**.
  Block: x += attn(modulate(prenorm(x), shift1, scale1)) * gate1;
  x += mlp(modulate(postnorm(x), shift2, scale2)) * gate2.
- Timestep: `timestep_embedding(t, 256, max_period=10000, scale=1000)` → TimeMLP
  (Linear 256→6144, gelu, Linear 6144→6144, WITH bias) → `t`;
  tvec = TProj: gelu(t) → Linear 6144→36864 (bias).
- Text path: context = **stack of 12 hidden-state layers** from Qwen3-VL →
  `txtfusion`: reshape to (2560, 12, tokens*B) → 2 layerwise KreaTextFusionBlocks
  (attention over the 12-layer axis) → permute → projector Linear 12→1 →
  (2560, tokens, B) → 2 refiner blocks (attention over token axis) →
  `txtmlp`: RMSNorm → Linear 2560→6144 → gelu(tanh) → Linear 6144→6144 (bias).
- Sequence: hidden = concat(txt, img) along tokens; 28 single-stream blocks;
  slice img part; `last`: RMSNorm + FinalModulation (2×6144 learned + t, chunk 2)
  → modulate → Linear 6144→64 (= 2*2*16); unpatchify.
- Patchify: `DiT::pad_and_patchify(x, 2, 2)`; 512×512 → latent 64×64×16 → 1024 tokens.
- RoPE ids: FLUX-style — txt ids zeros (3 axes), img ids (0, y, x); theta 1000;
  `Rope::embed_nd` with axes {32,48,48}. (t2i only; ref_latents/image-edit path skipped.)

**GELU:** the bool arg of `ggml_ext_gelu` is just `inplace` (VERIFIED
ggml_extend.hpp:980) — TimeMLP/TProj/TextMLP all use ggml's default `ggml_gelu`,
which is the **tanh approximation**. One GELU kernel needed.

**RoPE img ids** (rope.hpp gen_flux_img_ids): h_len=(h+p/2)/p; per patch position
ids = (0, y, x) over 3 axes; txt ids all zeros; `Rope::embed_nd(ids, bs, theta=1000,
axes_dim={32,48,48})` builds the pe tensor consumed by `Rope::attention`.

## Text encoder / conditioner (from `conditioner.hpp` krea2 branch)

- Encoder: **Qwen3-VL 4B Instruct** (36 text layers, hidden 2560), TEXT path only.
- Prompt template (t2i):
  `<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n`
- `prompt_template_encode_start_idx = 34` — first 34 tokens dropped from outputs.
- `out_layers = {2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35}` — the 12 hidden-state
  layers stacked as DiT context (matches text_layers=12, text_dim=2560).
  (Verify indexing convention — embedding-is-layer-0 vs first-block-output — against
  LLMEmbedder in sd.cpp during T6.)

## Sampler

- Prediction: `FLUX_FLOW_PRED` (flow matching, velocity prediction).
- Flow shift: **1.15 constant** for krea2 (`default_flow_shift = 1.15f`) — NOT
  resolution-dynamic. Turbo: 8 steps, cfg 0.0 (no negative pass).

## VAE

- **Wan 2.1 VAE** (literal wan file), 16-channel latents, 8× spatial downsample.
  Decoder port target; encode path not needed for t2i.

## Perf/memory notes

(filled in as measured)
