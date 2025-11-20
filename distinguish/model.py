"""Model building utilities for the distinguish package.

Provides a function ``build_model`` that constructs a binary classifier
based on a torchvision backbone (default ResNet18) and adapts the final
layer for a single-logit binary output (real vs AI-generated).
"""

from __future__ import annotations
from typing import Literal

from torch import nn
from torchvision import models

BackboneName = Literal["resnet18", "resnet34", "efficientnet_b0"]


def build_model(
    backbone: BackboneName = "resnet18",
    pretrained: bool = True,
) -> nn.Module:
    """Build a binary classification model.

    Parameters
    ----------
    backbone: str
        Backbone architecture to use. Supported: 'resnet18', 'resnet34', 'efficientnet_b0'.
    pretrained: bool
        If True, load ImageNet pretrained weights.

    Returns
    -------
    torch.nn.Module
        Model with a single output logit suitable for BCEWithLogitsLoss.
    """
    if backbone == "resnet18":
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    elif backbone == "resnet34":
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            model = models.resnet34(weights=weights)
        else:
            model = models.resnet34(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    elif backbone == "efficientnet_b0":
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            model = models.efficientnet_b0(weights=weights)
        else:
            model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model
