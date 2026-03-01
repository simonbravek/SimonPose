# Configuration

SimonPose uses a single configuration file: `config.py`.

It is intentionally simple: it defines filesystem paths to datasets, external dependencies, and model files.

## Paths

Defaults (relative to the repo root):

- `data/`: datasets and annotations
- `external/`: third-party repositories (Detectron2)
- `models/`: downloaded model weights (DensePose, SMPL)
- `output/`: generated results

Concrete paths used by the current projects:

- `SMPL_MODEL`: the SMPL `.pkl` file used by `smplx.SMPL`
- `DENSEPOSE_CONFIG`: the DensePose config YAML inside `external/detectron2/`
- `DENSEPOSE_WEIGHTS`: the DensePose checkpoint `.pkl`

## Changing the layout

If you keep data/models/external elsewhere (common on clusters), you have two options:

1) Use symlinks so the default paths still exist under the repo root.
2) Edit `config.py` to point to your actual locations.

Quick sanity check:

```bash
python -c "from config import *; print(SMPL_MODEL); print(DENSEPOSE_CONFIG); print(DENSEPOSE_WEIGHTS)"
```
