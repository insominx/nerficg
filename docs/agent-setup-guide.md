# Agent Setup Guide — nerficg

Dense, rule-like guidance for agents working with this framework for the first time.

---

## 1. Conda Environment — Invocation Pattern

**Never** use `conda activate <env> && python ...` in PowerShell (`pwsh`).  
`conda activate` does not propagate into the subprocess in pwsh; the base interpreter runs instead, and imports fail silently or with misleading errors.

**Always** invoke as:
```
conda run -n nerficg python scripts/<script>.py [args]
```

The conda environment for this project is named `nerficg`. It is created from `environments/py311_cu128.yaml` (Python 3.11, PyTorch 2.8, CUDA 12.8). If you need COLMAP data preparation (running `scripts/colmap.py`), use the `_with_colmap` variant instead.

---

## 2. Method CUDA Extensions — Install Before First Run

Every method ships with one or more compiled CUDA extensions that are **not installed by the conda env creation**. Training will start (GPU init succeeds, dataset loads), then die mid-setup with a pip/build error.

**Symptom:** framework logs GPU info, then exits with a message containing: *"use `./scripts/install.py -m <METHOD_NAME>` to automatically install all dependencies"*.

**Fix — always run this first, once per method:**
```
conda run -n nerficg python scripts/install.py -m <METHOD_NAME>
```

How it works: `install.py` iteratively imports the method module, catches `Framework.ExtensionError` for each missing extension (one at a time), builds+installs it via pip, then retries — so a single invocation installs all required extensions, even if there are multiple. Re-running it is safe (pip skips already-installed packages).

---

## 3. Identifying Method Name and Dataset Type

### Method names
Available methods are subdirectories of `src/Methods/` (excluding `Base/`):
- `GaussianSplatting` — standard 3DGS
- `FasterGS` — faster variant (has its own example config at `src/Methods/FasterGS/fastergs_garden.yaml`)
- `InstantNGP`
- `NeRF`

### Dataset type names
Dataset types are Python file stems in `src/Datasets/` (excluding `Base.py` and `utils.py`).  
Each file defines exactly one `CustomDataset` class; the framework registry maps the filename stem → `CustomDataset`. 

| File in `src/Datasets/` | `DATASET_TYPE` value |
|---|---|
| `Colmap.py` | `Colmap` |
| `MipNeRF360.py` | `MipNeRF360` |
| `NeRF.py` | `NeRF` |
| `TanksAndTemples.py` | `TanksAndTemples` |
| (others) | (filename stem) |

---

## 4. Choosing the Right Dataset Class

Inspect the dataset directory structure before choosing `DATASET_TYPE`.

| What you see in the dataset dir | Use this type |
|---|---|
| `sparse/0/` with binary COLMAP files + `images/` | `Colmap` |
| `transforms_train.json` (NeRF synthetic format) | `NeRF` |
| MipNeRF360 structure (colmap + `images_N/` subfolders) | `MipNeRF360` |
| Tanks & Temples structure | `TanksAndTemples` |

**Critical detail for `Colmap`:** the loader uses `pycolmap.Reconstruction(path / 'sparse' / '0')`, which reads **binary** COLMAP format. If your dataset only has text-format files (e.g., in a `colmap_text/` directory), they will be ignored — you must have binary files in `sparse/0/`. The `scripts/colmap.py` pipeline produces this automatically.

**Optional subdirectories** the `Colmap` loader will use if present:
- `sfm_masks/` — per-image segmentation masks (note: NOT `masks/`)
- `flow/` — optical flow (forward/backward `.flo` files)
- `monoc_depth/` — monocular depth (`.npy` files)

---

## 5. Config Creation — Use the Script, Don't Handcraft

Prefer generating configs over writing them by hand:
```
conda run -n nerficg python scripts/create_config.py -m <METHOD_NAME> -d <DATASET_TYPE> -o <config_name>
```

This writes a correctly-defaulted `configs/<config_name>.yaml` for the exact method×dataset combination. Then only edit the fields you need to change (typically `DATASET.PATH`, `TRAINING.MODEL_NAME`, `IMAGE_SCALE_FACTOR`).

**When adapting an existing config from a different dataset type** (e.g., copying `gs_garden.yaml` which uses `MipNeRF360`), watch out: you will inherit dataset-specific parameters that may not exist on the target dataset class. This won't necessarily crash (unknown YAML keys are often tolerated), but it's noise. Use `create_config.py` to get a clean baseline.

---

## 6. Key Config Fields to Always Verify

| Field | Rule |
|---|---|
| `GLOBAL.METHOD_TYPE` | Must match a directory name in `src/Methods/` |
| `GLOBAL.DATASET_TYPE` | Must match a filename stem in `src/Datasets/` |
| `DATASET.PATH` | Relative to the **project root** (e.g., `dataset/train`) |
| `TRAINING.MODEL_NAME` | Used as the output run directory prefix |
| `TRAINING.GUI.ACTIVATE` | **Set to `false` for headless/CLI runs.** The default in GS configs is `true`; this will fail without a display server. |
| `DATASET.TEST_STEP` | `0` = all images go to train split (no held-out test). Positive integer N = every Nth image goes to test. |
| `DATASET.IMAGE_SCALE_FACTOR` | `1.0` = full resolution. Reduce (e.g., `0.5`, `0.25`) to reduce VRAM and speed up training. |

---

## 7. Complete First-Time Training Workflow

```
# 1. Inspect dataset structure → pick DATASET_TYPE
# 2. Generate config
conda run -n nerficg python scripts/create_config.py -m GaussianSplatting -d Colmap -o my_scene

# 3. Edit configs/my_scene.yaml:
#    - DATASET.PATH: dataset/<your_scene>
#    - TRAINING.GUI.ACTIVATE: false
#    - TRAINING.MODEL_NAME: my_scene

# 4. Install method extensions (once per method)
conda run -n nerficg python scripts/install.py -m GaussianSplatting

# 5. Train
conda run -n nerficg python scripts/train.py -c configs/my_scene.yaml

# 6. Export to PLY (for external viewers)
conda run -n nerficg python scripts/convert_to_ply.py -d output/GaussianSplatting/<run_dir>
```

---

## 8. Output Structure

All outputs land in `output/<METHOD_NAME>/<MODEL_NAME>_<YYYY-MM-DD-HH-MM-SS>/`:

```
output/
  GaussianSplatting/
    my_scene_2026-02-27-13-45-17/
      training_config.yaml    ← full config snapshot used for this run
      checkpoints/
        final.pt              ← framework checkpoint (reload for inference/GUI)
      final.ply               ← exported after running convert_to_ply.py
```

The `.ply` file is a standard Gaussian splat file viewable in SuperSplat, Luma AI viewer, or similar tools.

---

## 9. Performance Reference

On a mid-range NVIDIA GPU (RTX class), 30,000 iterations of standard GaussianSplatting on ~120 images at full resolution takes approximately:

- **~4–5 minutes** at ~107 iterations/second
- **~3.3–3.8 GiB VRAM** (allocated/reserved)
- **~200K Gaussian primitives** typical for an indoor/outdoor scene of this size

`FasterGS` is a faster alternative method with the same `Colmap`-compatible dataset interface — use it if iteration speed is the priority.
