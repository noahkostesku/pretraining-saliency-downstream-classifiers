"""Frozen encoder plus linear classifier head."""

import torch
from torch import nn

from cv.encoders import EncoderWrapper


class LinearProbeModel(nn.Module):
    """Frozen encoder with a trainable linear classifier head."""

    def __init__(self, encoder: EncoderWrapper, num_classes: int = 10) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(self.encoder.feature_dim, num_classes)
        self.encoder.freeze(set_eval=True)

    def train(self, mode: bool = True):
        """Keep the encoder in eval mode while training the classifier head."""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits for STL-10 classes."""
        features = self.encoder(images)
        return self.classifier(features)
