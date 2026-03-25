"""Map encoder condition names to concrete loader functions."""

from .moco import load_moco_encoder
from .supervised import load_supervised_encoder
from .swav import load_swav_encoder
from .wrapper import LoadedEncoder


def load_encoder(name: str, **kwargs) -> LoadedEncoder:
    """Load an encoder by condition name."""
    loaders = {
        "supervised": load_supervised_encoder,
        "moco": load_moco_encoder,
        "swav": load_swav_encoder,
    }
    if name not in loaders:
        valid_names = ", ".join(sorted(loaders))
        raise KeyError(
            f"Unknown encoder condition '{name}'. Valid values: {valid_names}"
        )
    return loaders[name](**kwargs)
