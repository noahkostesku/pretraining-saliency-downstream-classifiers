# Commands 

1. `uv run python -m compilell src scripts` for an overall script check
2. `uv run python scripts/inspect_encoders.py --conditions supervised --batch-size 1 --device cpu` to test whether the MoCo, SwAV and ResNet50 encoders have been loaded properly.

On UNIX-like systems, the models are downloaded to `/Users/<your_username>/.cache/torch/hub/checkpoints/<modelname>.pth`.
