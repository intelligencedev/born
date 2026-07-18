"""Krea 2 Turbo parity harness.

Generates numeric fixtures for the born Go port from the SAME GGUF weights the
Go side loads, so parity failures always indicate born-side logic bugs, never
quantization differences.

Subcommands are added as the spike progresses:
  dump-gguf-tensors   representative dequantized tensors from the DiT GGUF
"""

import argparse
import os
import pathlib

import numpy as np

from dump import dump, dump_json

MODELS = pathlib.Path(os.path.expanduser("~/.cache/manifold/krea2-models"))
DIT_GGUF = MODELS / "Krea-2-Turbo-Q4_K_M.gguf"

# Representative tensors spanning every quant type present in the DiT GGUF
# (histogram: Q4_K x262, F32 x166, F16 x2 — see gguf_probe_test.go).
GGUF_PARITY_TENSORS = [
    "blocks.0.attn.wq.weight",   # Q4_K, 2D large
    "blocks.0.mlp.down.weight",  # Q4_K, 2D transposed-ish shape
    "blocks.27.attn.wo.weight",  # Q4_K, last block
    "blocks.0.prenorm.scale",    # F32, 1D
    "blocks.0.mod.lin",          # F32, 1D large (6*6144)
    "blocks.0.attn.qknorm.qnorm.scale",  # F32, head_dim
]


def cmd_dump_gguf_tensors(_args) -> None:
    from gguf import GGUFReader
    from gguf.quants import dequantize

    reader = GGUFReader(str(DIT_GGUF))
    by_name = {t.name: t for t in reader.tensors}

    # Also cover the two F16 tensors, whatever they are.
    names = list(GGUF_PARITY_TENSORS)
    f16 = [t.name for t in reader.tensors if t.tensor_type.name == "F16"]
    names += f16

    manifest = []
    for name in names:
        t = by_name[name]
        data = dequantize(t.data, t.tensor_type)
        # gguf-py returns data already in logical (row-major, reversed-GGUF)
        # order matching torch convention; born's Convert() reverses dims the
        # same way.
        shape = list(data.shape)
        dump("gguf." + name, np.asarray(data, dtype=np.float32))
        manifest.append(
            {"name": name, "shape": shape, "quant": t.tensor_type.name}
        )
    dump_json("gguf-manifest", manifest)


KREA2_SYSTEM_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
FOX_PROMPT = "a red fox sitting in fresh snow, golden hour, photorealistic"


def krea2_prompt(text: str) -> str:
    return KREA2_SYSTEM_TEMPLATE + text + "<|im_end|>\n<|im_start|>assistant\n"


def cmd_dump_tokenizer_cases(_args) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODELS / "tokenizer"))
    cases = [
        "hello world",
        "a red fox sitting in fresh snow, golden hour, photorealistic",
        "Ünïcödé — em-dash, naïve café ☕",
        "emoji 🦊🎨 and CJK 你好世界",
        "  leading spaces and\nnewlines\t tabs",
        "numbers 12345 67.89 and CamelCaseTokens",
        krea2_prompt(FOX_PROMPT),
        krea2_prompt(""),
    ]
    out = [{"text": c, "ids": tok(c, add_special_tokens=False)["input_ids"]} for c in cases]
    dump_json("tokenizer-cases", out)


def load_text_encoder(dtype=None):
    """Qwen3-VL text encoder built from the SAME Q8_0 GGUF born loads.

    Config values pinned from GGUF metadata (see ORACLE.md): 36 layers,
    hidden 2560, heads 32/8 (head_dim 128), ffn 9728, rms_eps 1e-6,
    rope_theta 5e6, mrope sections [24, 20, 20].
    """
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    dtype = dtype or torch.float32
    cfg = Qwen3VLTextConfig(
        vocab_size=151936,
        hidden_size=2560,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=9728,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20]},
        attention_bias=False,
        tie_word_embeddings=True,
    )
    model = Qwen3VLTextModel(cfg)

    reader = GGUFReader(str(MODELS / "Qwen3VL-4B-Instruct-Q8_0.gguf"))
    sd = {}
    for t in reader.tensors:
        name = t.name
        w = torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type))).to(dtype)
        if name == "token_embd.weight":
            sd["embed_tokens.weight"] = w
        elif name == "output_norm.weight":
            sd["norm.weight"] = w
        elif name.startswith("blk."):
            _, n, rest = name.split(".", 2)
            base = f"layers.{n}."
            m = {
                "attn_q.weight": "self_attn.q_proj.weight",
                "attn_k.weight": "self_attn.k_proj.weight",
                "attn_v.weight": "self_attn.v_proj.weight",
                "attn_output.weight": "self_attn.o_proj.weight",
                "attn_q_norm.weight": "self_attn.q_norm.weight",
                "attn_k_norm.weight": "self_attn.k_norm.weight",
                "attn_norm.weight": "input_layernorm.weight",
                "ffn_norm.weight": "post_attention_layernorm.weight",
                "ffn_gate.weight": "mlp.gate_proj.weight",
                "ffn_up.weight": "mlp.up_proj.weight",
                "ffn_down.weight": "mlp.down_proj.weight",
            }[rest]
            sd[base + m] = w
        else:
            raise ValueError(f"unmapped GGUF tensor {name}")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # rotary_emb buffers etc. may be "missing"; real weights must all match.
    real_missing = [k for k in missing if not k.endswith("inv_freq")]
    if real_missing or unexpected:
        raise ValueError(f"state dict mismatch: missing={real_missing} unexpected={unexpected}")
    return model.eval()


def krea2_text_inputs(prompt: str, max_sequence_length: int = 512):
    """Replicates Krea2Pipeline.get_text_hidden_states tokenization exactly:
    [prefix | prompt | PAD -> (msl + 34 - 5) | suffix(5)], bool mask,
    cumulative-valid-token position ids broadcast over 3 mRoPE axes."""
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODELS / "tokenizer"))
    prefix_idx = 34
    text_tokens = tok(
        [KREA2_SYSTEM_TEMPLATE + prompt],
        truncation=True,
        padding="max_length",
        max_length=max_sequence_length + prefix_idx - 5,
        return_tensors="pt",
    )
    suffix_tokens = tok(["<|im_end|>\n<|im_start|>assistant\n"], return_tensors="pt")
    input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
    attention_mask = torch.cat(
        [text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1
    ).bool()
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    return input_ids, attention_mask, position_ids


SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


def cmd_dump_text_encoder(_args) -> None:
    """Full-weight text-encoder fixture: fox prompt -> (1, 512, 12, 2560)."""
    import torch

    model = load_text_encoder(torch.float32)
    input_ids, attention_mask, position_ids = krea2_text_inputs(FOX_PROMPT)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
    hidden = torch.stack([out.hidden_states[i] for i in SELECT_LAYERS], dim=2)
    hidden = hidden[:, 34:]
    mask = attention_mask[:, 34:]
    dump("textenc.input_ids", input_ids.to(torch.float32))
    dump("textenc.mask", mask.to(torch.float32))
    dump("textenc.hidden", hidden)


def cmd_dump_tiny_text_encoder(_args) -> None:
    """Tiny random Qwen3 text model + IO fixture for the fast Go dev loop.
    Weights are dumped under GGUF (llama.cpp) tensor names so the Go encoder
    exercises the same loading path as the real Q8_0 file."""
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

    torch.manual_seed(7)
    cfg = Qwen3VLTextConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [6, 5, 5]},
        attention_bias=False,
        tie_word_embeddings=True,
    )
    model = Qwen3VLTextModel(cfg).eval()

    hf2gguf = {
        "embed_tokens.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight",
    }
    per_layer = {
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    for k, v in model.state_dict().items():
        if k in hf2gguf:
            dump("tinytextenc.w." + hf2gguf[k], v)
        else:
            parts = k.split(".")
            assert parts[0] == "layers", k
            dump(f"tinytextenc.w.blk.{parts[1]}." + per_layer[".".join(parts[2:])], v)

    # IO: 12 tokens with 3 pads in the middle (exercises cumulative positions),
    # taking hidden states from layers 1 and 2 (embeddings excluded).
    input_ids = torch.tensor([[5, 9, 200, 13, 42, 7, 0, 0, 0, 99, 100, 101]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]]).bool()
    position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
    dump("tinytextenc.input_ids", input_ids.to(torch.float32))
    dump("tinytextenc.mask", attention_mask.to(torch.float32))
    dump("tinytextenc.hidden.l1", out.hidden_states[1])
    dump("tinytextenc.hidden.l2", out.hidden_states[2])


def load_transformer(dtype=None):
    """Krea2Transformer2DModel with weights from the SAME Q4_K_M GGUF born
    loads (Krea2 has no from_single_file GGUF path yet). Both sides store
    zero-centered RMSNorm scales, so the mapping is 1:1 with no offsets."""
    import torch
    from diffusers import Krea2Transformer2DModel
    from gguf import GGUFReader
    from gguf.quants import dequantize

    dtype = dtype or torch.float32

    def attn(rest):
        return {
            "attn.wq.weight": "attn.to_q.weight",
            "attn.wk.weight": "attn.to_k.weight",
            "attn.wv.weight": "attn.to_v.weight",
            "attn.gate.weight": "attn.to_gate.weight",
            "attn.qknorm.qnorm.scale": "attn.norm_q.weight",
            "attn.qknorm.knorm.scale": "attn.norm_k.weight",
            "attn.wo.weight": "attn.to_out.0.weight",
            "mlp.gate.weight": "ff.gate.weight",
            "mlp.up.weight": "ff.up.weight",
            "mlp.down.weight": "ff.down.weight",
            "prenorm.scale": "norm1.weight",
            "postnorm.scale": "norm2.weight",
        }[rest]

    sd = {}
    reader = GGUFReader(str(DIT_GGUF))
    for t in reader.tensors:
        name = t.name
        w = torch.from_numpy(np.ascontiguousarray(dequantize(t.data, t.tensor_type))).to(dtype)
        if name == "first.weight":
            sd["img_in.weight"] = w
        elif name == "first.bias":
            sd["img_in.bias"] = w
        elif name.startswith("tmlp."):
            idx = {"0": "1", "2": "2"}[name.split(".")[1]]
            sd[f"time_embed.linear_{idx}.{name.rsplit('.', 1)[1]}"] = w
        elif name.startswith("tproj.1."):
            sd["time_mod_proj." + name.rsplit(".", 1)[1]] = w
        elif name.startswith("txtmlp."):
            idx, kind = name.split(".")[1], name.rsplit(".", 1)[1]
            if idx == "0":
                sd["txt_in.norm.weight"] = w
            else:
                sd[f"txt_in.linear_{'1' if idx == '1' else '2'}.{kind}"] = w
        elif name == "txtfusion.projector.weight":
            # GGUF stores the layer projector 1-D; diffusers Linear wants (1, 12).
            sd["text_fusion.projector.weight"] = w.reshape(1, -1)
        elif name.startswith("txtfusion."):
            _, group, n, rest = name.split(".", 3)
            sd[f"text_fusion.{group}.{n}.{attn(rest)}"] = w
        elif name.startswith("blocks."):
            _, n, rest = name.split(".", 2)
            if rest == "mod.lin":
                sd[f"transformer_blocks.{n}.scale_shift_table"] = w.reshape(6, -1)
            else:
                sd[f"transformer_blocks.{n}.{attn(rest)}"] = w
        elif name == "last.norm.scale":
            sd["final_layer.norm.weight"] = w
        elif name == "last.modulation.lin":
            sd["final_layer.scale_shift_table"] = w
        elif name.startswith("last.linear."):
            sd["final_layer.linear." + name.rsplit(".", 1)[1]] = w
        else:
            raise ValueError(f"unmapped GGUF tensor {name}")

    model = Krea2Transformer2DModel()
    model.load_state_dict(sd, strict=True)  # raises on any mismatch
    return model.to(dtype).eval()


def build_pipeline(device="mps", dtype=None):
    import torch
    from diffusers import (
        AutoencoderKLWan,
        FlowMatchEulerDiscreteScheduler,
        Krea2Pipeline,
    )
    from transformers import AutoTokenizer

    dtype = dtype or torch.float32
    transformer = load_transformer(dtype)
    vae = AutoencoderKLWan.from_single_file(
        str(MODELS / "wan_2.1_vae.safetensors"), torch_dtype=dtype
    )
    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=256,
        max_image_seq_len=6400,
    )
    pipe = Krea2Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=load_text_encoder(dtype),
        tokenizer=AutoTokenizer.from_pretrained(str(MODELS / "tokenizer")),
        transformer=transformer,
    )
    return pipe.to(device)


def _patch_rope_dtype_for_mps() -> None:
    """MPS workaround: diffusers' Krea2 rope tables are f32, promoting fp16
    q/k to f32 while v stays fp16 — MPS matmul asserts on mixed dtypes.
    Cast the rope output back to the input dtype (numerics unchanged: rope is
    still computed in f32)."""
    import diffusers.models.transformers.transformer_krea2 as tk

    orig = tk.apply_rotary_emb

    def rope_same_dtype(x, *a, **kw):
        return orig(x, *a, **kw).to(x.dtype)

    tk.apply_rotary_emb = rope_same_dtype


def cmd_e2e(args) -> None:
    import torch

    device = args.device
    if device == "mps":
        _patch_rope_dtype_for_mps()
    # fp16: the 12B DiT in f32 would be ~48 GB (over budget with the encoder
    # resident), and MPS rejects bf16 in some matmul kernels. Self-validation
    # only needs a recognizable image; parity fixtures are per-component f32.
    pipe = build_pipeline(device=device, dtype=torch.float16)
    gen = torch.Generator("cpu").manual_seed(42)
    image = pipe(
        FOX_PROMPT,
        width=512,
        height=512,
        num_inference_steps=8,
        generator=gen,
    ).images[0]
    image.save(args.out)
    print(f"saved {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dump-gguf-tensors").set_defaults(fn=cmd_dump_gguf_tensors)
    sub.add_parser("dump-tokenizer-cases").set_defaults(fn=cmd_dump_tokenizer_cases)
    sub.add_parser("dump-text-encoder").set_defaults(fn=cmd_dump_text_encoder)
    sub.add_parser("dump-tiny-text-encoder").set_defaults(fn=cmd_dump_tiny_text_encoder)
    e2e = sub.add_parser("e2e")
    e2e.add_argument("--out", default="harness-512-seed42.png")
    e2e.add_argument("--device", default="mps")
    e2e.set_defaults(fn=cmd_e2e)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
