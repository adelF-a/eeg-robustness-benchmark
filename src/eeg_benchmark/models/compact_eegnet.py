"""Compact EEGNet-like convolutional network.

Inspired by EEGNet (Lawhern et al., 2018) but not a byte-exact reproduction: it follows
the temporal-conv -> depthwise spatial-conv -> separable-conv structure at reduced width
to stay well under 100k parameters. Input shape is ``[batch, 1, channels, time]``.
"""

from __future__ import annotations

import torch
from torch import nn


class CompactEEGNet(nn.Module):
    """A small EEGNet-style classifier for 2-class motor imagery.

    Parameters
    ----------
    n_channels, n_times : input spatial/temporal dimensions.
    n_classes : output classes (2).
    F1 : number of temporal filters. D : depthwise spatial multiplier.
    kernel_length : temporal-convolution length (~half the sampling rate captures mu/beta).
    dropout : dropout probability after each pooling stage.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        kernel_length: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        F2 = F1 * D
        self.n_channels = n_channels
        self.n_times = n_times

        # Block 1: temporal conv (learn frequency filters) + depthwise spatial conv.
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        # Block 2: separable conv (depthwise temporal + pointwise mixing).
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(F2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, 1, channels, time] -> logits [batch, n_classes]."""
        if x.dim() == 3:  # [batch, channels, time] -> add conv channel dim
            x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
