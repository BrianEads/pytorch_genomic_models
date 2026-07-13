"""Cell Painting ResNet tower for morphological phenotype embeddings."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models


class CellPaintingResNetTower(nn.Module):
    """ResNet-50 backbone adapted for 5-channel Cell Painting images.

    Args:
        d_model: Output embedding dimension after the projection head.
        pretrained: Whether to initialise RGB weights from ImageNet (first
            three channels only; remaining channels are mean-initialized).

    Input shape:
        images: ``(B, 5, 224, 224)`` five-channel fluorescence crops.

    Output:
        Tuple of embedding ``(B, d_model)`` and ``None`` (no token-level
        attention in this tower stub).
    """

    def __init__(self, d_model: int = 256, pretrained: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        original_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            5,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )
        if pretrained and original_conv.weight is not None:
            with torch.no_grad():
                backbone.conv1.weight[:, :3, :, :] = original_conv.weight
                backbone.conv1.weight[:, 3:, :, :] = original_conv.weight.mean(
                    dim=1, keepdim=True
                ).expand(-1, 2, -1, -1)

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.fc.in_features, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode Cell Painting images into a CLS-style embedding.

        Args:
            images: Batch of 5-channel images, shape ``(B, 5, 224, 224)``.

        Returns:
            Projected embedding and ``None`` for attention weights.
        """
        features = self.backbone(images)
        embedding = self.projection(features)
        return embedding, None
