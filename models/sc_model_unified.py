import torch
import torch.nn as nn
import timm
from .attention import SpatialAttention


BACKBONE_NAMES = {
    "b0": "efficientnet_b0",
    "b4": "efficientnet_b4",
}


class EfficientNetBaseline(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, backbone="b0"):
        super().__init__()
        if backbone not in BACKBONE_NAMES:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            BACKBONE_NAMES[backbone],
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.backbone(x)


class EfficientNetHeadless(nn.Module):
    """Control model with the SC feature path but no attention and no consistency:
    final-stage features (features_only, no 1x1 head convolution) -> GAP -> linear.
    Isolates the effect of removing the standard head convolution."""

    def __init__(self, num_classes=7, pretrained=True, backbone="b0"):
        super().__init__()
        if backbone not in BACKBONE_NAMES:
            raise ValueError(f"Unsupported backbone: {backbone}")
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            BACKBONE_NAMES[backbone], pretrained=pretrained, features_only=True)
        channels = self.backbone.feature_info.channels()[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        f = self.backbone(x)[-1]
        return self.classifier(self.pool(f).flatten(1))


class SCEfficientNet(nn.Module):
    """Dual-path SC-EfficientNet supporting EfficientNet-B0/B4."""

    def __init__(self, num_classes=7, pretrained=True, backbone="b0"):
        super().__init__()
        if backbone not in BACKBONE_NAMES:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone_name = backbone
        self.backbone = timm.create_model(
            BACKBONE_NAMES[backbone],
            pretrained=pretrained,
            features_only=True,
        )

        channels = self.backbone.feature_info.channels()[-1]
        self.attention = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_org = nn.Linear(channels, num_classes)
        self.classifier_masked = nn.Linear(channels, num_classes)
        # When True, eval-mode forward also returns (p_org, p_masked, mask);
        # used only to record mask statistics. Inference still uses p_masked.
        self.return_mask = False

    def forward(self, x):
        feat = self.backbone(x)[-1]
        mask = self.attention(feat)
        masked_feat = feat * mask

        p_org = self.classifier_org(self.pool(feat).flatten(1))
        p_masked = self.classifier_masked(
            self.pool(masked_feat).flatten(1)
        )

        if self.training or self.return_mask:
            return p_org, p_masked, mask
        return p_masked
