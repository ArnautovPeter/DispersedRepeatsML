"""Run with .venv/bin/python -m src.check_gpu (outside the agent sandbox)."""

import json
import sys

import torch


def main():
    info = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not info["cuda_available"]:
        print(json.dumps(info, indent=2))
        print(
            "GPU unavailable: check nvidia-smi outside the sandbox and driver/build compatibility.",
            file=sys.stderr,
        )
        return 1

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    layer = torch.nn.Linear(512, 128).to(device)
    x = torch.randn(32, 512, device=device)
    loss = layer(x).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss) and all(
        torch.isfinite(p.grad).all() for p in layer.parameters()
    )

    info.update(
        device=torch.cuda.get_device_name(0),
        total_memory_mib=torch.cuda.get_device_properties(0).total_memory // 2**20,
        peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 2),
        loss=loss.item(),
        forward_backward="passed",
    )

    print(json.dumps(info, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
