"""Convert the Wan 2.1 VAE (decoder side) to a GGUF file born can load with
its generic reader, plus dump a decode parity fixture.

Usage:
  .venv/bin/python convert_vae.py            # writes wan-vae-decoder.gguf + fixtures

The GGUF carries the diffusers state-dict tensor names verbatim (decoder.* and
post_quant_conv.*) in F16, plus latents_mean/std and the structural config as
metadata. For t2i the 3D causal convs reduce exactly to 2D convs using the
LAST temporal kernel slice (causal left-padding is all zeros for a single
frame; upsample3d's time_conv is skipped on the first chunk) — the Go side
performs that reduction at load time; weights here stay unmodified 3D.
"""

import json
import os
import pathlib

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from gguf import GGUFWriter

from dump import dump, dump_json

MODELS = pathlib.Path(os.path.expanduser("~/.cache/manifold/krea2-models"))
OUT = MODELS / "wan-vae-decoder.gguf"


def main() -> None:
    vae = AutoencoderKLWan.from_single_file(
        str(MODELS / "wan_2.1_vae.safetensors"), torch_dtype=torch.float32
    ).eval()
    cfg = dict(vae.config)
    print(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, default=str, indent=1))

    w = GGUFWriter(str(OUT), "wan-vae-decoder")
    w.add_string("wanvae.config", json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, default=str))
    sd = vae.state_dict()
    kept = 0
    for name, t in sd.items():
        if not (name.startswith("decoder.") or name.startswith("post_quant_conv")):
            continue
        a = t.detach().to(torch.float32).numpy()
        w.add_tensor(name, a.astype(np.float16))
        kept += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {OUT} with {kept} tensors")

    # Parity fixture: seeded packed-latent decode. The latents here are
    # ALREADY denormalized (x*std+mean applied), matching what the Go VAE
    # receives after the pipeline denorm step.
    torch.manual_seed(21)
    z = torch.randn(1, 16, 1, 64, 64, dtype=torch.float32)
    with torch.no_grad():
        img = vae.decode(z, return_dict=False)[0][:, :, 0]  # (1, 3, 512, 512)
    dump("vae.z", z[:, :, 0])  # (1, 16, 64, 64)
    dump("vae.out", img)
    dump("vae.latents_mean", np.array(cfg["latents_mean"], dtype=np.float32))
    dump("vae.latents_std", np.array(cfg["latents_std"], dtype=np.float32))
    dump_json(
        "vae-structure",
        {
            "base_dim": cfg.get("base_dim"),
            "z_dim": cfg.get("z_dim"),
            "dim_mult": cfg.get("dim_mult"),
            "num_res_blocks": cfg.get("num_res_blocks"),
            "attn_scales": cfg.get("attn_scales"),
            "temperal_downsample": cfg.get("temperal_downsample"),
        },
    )


if __name__ == "__main__":
    main()
