"""Fixture writer shared by harness subcommands.

Fixtures are raw little-endian float32 .bin files with a JSON sidecar
{"shape": [...], "dtype": "float32"} — trivially readable from Go tests.
"""

import json
import pathlib

import numpy as np

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def dump(name: str, t) -> None:
    """Write tensor-like t (torch.Tensor or ndarray) as fixtures/<name>.bin/.json."""
    FIXTURES.mkdir(exist_ok=True)
    try:
        import torch

        if isinstance(t, torch.Tensor):
            t = t.detach().to("cpu", torch.float32).numpy()
    except ImportError:
        pass
    a = np.ascontiguousarray(t, dtype=np.float32)
    (FIXTURES / f"{name}.bin").write_bytes(a.tobytes())
    (FIXTURES / f"{name}.json").write_text(
        json.dumps({"shape": list(a.shape), "dtype": "float32"})
    )
    print(f"dumped {name}: shape={list(a.shape)}")


def dump_json(name: str, obj) -> None:
    FIXTURES.mkdir(exist_ok=True)
    (FIXTURES / f"{name}.json").write_text(json.dumps(obj, indent=1))
    print(f"dumped {name}.json")
