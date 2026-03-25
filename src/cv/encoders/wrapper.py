"""Unified encoder wrapper exposing pooled 2048-D features."""


class EncoderWrapper:
    """Wrap a pretrained encoder behind a common interface."""

    def __init__(self, encoder) -> None:
        self.encoder = encoder
