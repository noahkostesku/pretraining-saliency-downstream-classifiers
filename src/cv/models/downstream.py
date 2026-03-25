"""Shared downstream wrapper with freeze and unfreeze controls."""


class DownstreamModel:
    """Common downstream model wrapper for all encoder conditions."""

    def __init__(self, encoder, classifier) -> None:
        self.encoder = encoder
        self.classifier = classifier
