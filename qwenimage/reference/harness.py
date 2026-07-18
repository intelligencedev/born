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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dump-gguf-tensors").set_defaults(fn=cmd_dump_gguf_tensors)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
