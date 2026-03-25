"""Frozen encoder plus linear classifier head."""


class LinearProbeModel:
    """Placeholder linear probe model definition."""

    def __init__(self, encoder, num_classes: int = 10) -> None:
        self.encoder = encoder
        self.num_classes = num_classes
