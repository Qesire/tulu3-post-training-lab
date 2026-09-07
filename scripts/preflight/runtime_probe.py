#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def module_version(name: str):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def command(args: list[str]):
    try:
        cp = subprocess.run(args, check=False, capture_output=True, text=True, timeout=30)
        return {"returncode": cp.returncode, "stdout": cp.stdout.strip(), "stderr": cp.stderr.strip()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    out = Path(os.environ.get("RUNTIME_PROBE_OUT", "runtime.json"))
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                    "total_memory": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}

    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "executable": sys.executable,
        "slurm": {
            k: os.environ.get(k)
            for k in [
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_NODELIST",
                "SLURM_JOB_PARTITION",
                "SLURM_GPUS",
                "SLURM_CPUS_PER_TASK",
            ]
        },
        "torch": torch_info,
        "packages": {
            "flash_attn": module_version("flash_attn"),
            "transformers": module_version("transformers"),
            "accelerate": module_version("accelerate"),
            "datasets": module_version("datasets"),
            "peft": module_version("peft"),
            "bitsandbytes": module_version("bitsandbytes"),
            "vllm": module_version("vllm"),
        },
        "nvidia_smi": command(["nvidia-smi"]),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
