"""Build the correct model (baseline or SC, B0 or B4) from a saved checkpoint."""
import torch

from .sc_model_unified import EfficientNetBaseline, EfficientNetHeadless, SCEfficientNet


def build_from_checkpoint(checkpoint_path, device, num_classes=7, default_model="sc", default_backbone="b4"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_type = cfg.get("model", default_model)
    backbone = cfg.get("backbone", default_backbone)
    cls = {"baseline": EfficientNetBaseline, "headless": EfficientNetHeadless}.get(model_type, SCEfficientNet)
    model = cls(num_classes=num_classes, pretrained=False, backbone=backbone)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, model_type, backbone
