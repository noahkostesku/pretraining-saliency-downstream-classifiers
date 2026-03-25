# Commands 

1. `uv run python -m compilell src scripts` for an overall script check
2. `uv run python scripts/inspect_encoders.py --conditions supervised --batch-size 1 --device cpu` to test whether the MoCo, SwAV and ResNet50 encoders have been loaded properly.

On UNIX-like systems, the models are downloaded to `/Users/<your_username>/.cache/torch/hub/checkpoints/<modelname>.pth`.

`scripts/prepare_encoders.py` prepares encoders and tests them using `scripts/inspect_encoders.py`, and downloads the encoder ready models after preparation into `data/external/` which is not tracked by Git. 

3. `uv run python scripts/prepare_encoders.py --conditions swav --skip-inspect --device cpu`: Prepares and saves the SWaV encoder.
4. For MoCo: ` uv run python scripts/prepare_encoders.py --conditions moco --skip-inspect --device cpu`
5. Full preparation: `uv run python scripts/inspect_encoders.py --conditions supervised moco swav --device cpu`
