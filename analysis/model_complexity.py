"""Hardware-independent model complexity (parameters and multiply-accumulate FLOPs).

Models are instantiated with random weights (no checkpoint or image data needed)
and a single forward pass is traced with fvcore. FLOPs are reported as fvcore
counts fused multiply-adds (1 MAC = 1 FLOP in fvcore's convention).
TTA cost is reported as 5x the single-view inference FLOPs.
"""
import json
import sys
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.sc_model_unified import EfficientNetBaseline, SCEfficientNet  # noqa: E402


def count(model, size):
    model.eval()
    x = torch.zeros(1, 3, size, size)
    flops = FlopCountAnalysis(model, x)
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    params = sum(p.numel() for p in model.parameters())
    return params, flops.total()


def main():
    rows = []
    for name, cls, bb, size in [
        ("EfficientNet-B0 baseline", EfficientNetBaseline, "b0", 224),
        ("SC-EfficientNet-B0", SCEfficientNet, "b0", 224),
        ("EfficientNet-B4 (reference)", EfficientNetBaseline, "b4", 384),
        ("SC-EfficientNet-B4", SCEfficientNet, "b4", 384),
    ]:
        m = cls(num_classes=7, pretrained=False, backbone=bb)
        p, f = count(m, size)
        extra = {}
        if cls is SCEfficientNet:
            extra["attention_params"] = sum(q.numel() for q in m.attention.parameters())
            extra["head_params"] = sum(q.numel() for q in m.classifier_org.parameters()) + \
                sum(q.numel() for q in m.classifier_masked.parameters())
            extra["final_feature_channels"] = m.classifier_org.in_features
        rows.append({"model": name, "input": size, "params_M": p / 1e6,
                     "GFLOPs_single_view": f / 1e9, "GFLOPs_5view_TTA": 5 * f / 1e9, **extra})
    out = ROOT / "results" / "statistics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_complexity.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
