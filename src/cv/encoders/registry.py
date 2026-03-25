"""Map encoder condition names to concrete loader functions."""

from .moco import load_moco_encoder
from .supervised import load_supervised_encoder
from .swav import load_swav_encoder


def load_encoder(name: str):
    """Load an encoder by condition name."""
    loaders = {
        "supervised": load_supervised_encoder,
        "moco": load_moco_encoder,
        "swav": load_swav_encoder,
    }
    return loaders[name]()
