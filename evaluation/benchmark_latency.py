"""Wall-clock inference latency of all model types on the local GPU (random weights; timing only).

Usage: python -m evaluation.benchmark_latency
Reports median ms per image for batch size 1 and throughput (images/s) for batch size 32, FP32,
after warm-up, with CUDA synchronization. Results: results/latency/latency.csv
"""
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.sc_model_unified import EfficientNetBaseline, EfficientNetHeadless, SCEfficientNet  # noqa: E402


def timed(model, x, iters, device):
    times = []
    with torch.no_grad():
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter(); model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
    return sorted(times)[len(times) // 2]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    rows = []
    for bb, size in [("b0", 224), ("b4", 384)]:
        for name, cls in [("baseline", EfficientNetBaseline), ("headless", EfficientNetHeadless), ("sc", SCEfficientNet)]:
            m = cls(num_classes=7, pretrained=False, backbone=bb).to(device).eval()
            r = {"model": name, "backbone": bb, "input": size, "device": gpu}
            for bs, iters in [(1, 200), (32, 50)]:
                x = torch.randn(bs, 3, size, size, device=device)
                timed(m, x, 30, device)  # warm-up
                t = timed(m, x, iters, device)
                if bs == 1:
                    r["ms_per_image_bs1"] = 1000 * t
                    r["ms_per_image_bs1_tta5"] = 5000 * t
                else:
                    r["images_per_s_bs32"] = bs / t
            rows.append(r); print(r)
            del m; torch.cuda.empty_cache() if device.type == "cuda" else None
    out = ROOT / "results" / "latency"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "latency.csv", index=False)
    print("Saved", out / "latency.csv")


if __name__ == "__main__":
    main()
