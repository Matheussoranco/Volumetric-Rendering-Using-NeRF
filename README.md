# Volumetric Rendering with NeRF (Minimal Keras Port)

> Study / prototype — frozen. Minimal MLP + Fourier encoding NeRF on `tiny_nerf_data.npz` (32 samples/ray, 20 epochs).

## 1. Overview

Port of the Keras NeRF example (itself a tiny version of Mildenhall et al. 2020). Learns a volumetric scene function `(x, y, z, θ, φ) → (RGB, σ)` and renders novel views by alpha-compositing samples along each camera ray.

## 2. Architecture

- `encode_position`: Fourier features (`POS_ENCODE_DIMS=16`, sin/cos).
- `get_rays`: ray origins/directions from pose + focal.
- MLP: Dense-ReLU × N → density + color heads; `NUM_SAMPLES=32`, `BATCH_SIZE=5`, `EPOCHS=20`, Adam. Data: `tiny_nerf_data.npz` (images + poses + focal, auto-downloaded).

## 3. Repo layout

```
3D_volumetric_rendering_with_NeRF.ipynb  # original notebook
3D_volumetric_rendering_with_NeRF.py     # 444 lines (encode_position, get_rays, render, train)
requirements.txt                         # tensorflow>=2.16,<2.20, keras>=3,<4, numpy, imageio, tqdm, matplotlib
README.md
```

## 4. Install

```
pip install -r requirements.txt
```

GPU strongly recommended; CPU run ≈ hours.

## 5. Usage

```bash
python 3D_volumetric_rendering_with_NeRF.py
```

Trains and renders test views with `plt.show()`. Runs on import — see §8.

## 6. Expected output

Training PSNR/loss + side-by-side ground-truth vs rendered novel views. No saved weights.

## 7. Limitations / what this is not

- Tiny scene, 32 samples, no hierarchical sampling, no pose refinement — quality far from full NeRF / mip-NeRF / Nerfstudio.
- Version-fragile TF/Keras pins; `imageio.v2` API assumed.
- No tests, no CLI, no checkpointing.

## 8. Tests

None. Smoke check = full run.

## 9. References

- Keras example: https://keras.io/examples/vision/nerf/
- bmild/nerf: https://github.com/bmild/nerf
- Paper: https://arxiv.org/abs/2003.08934

## 10. License

None declared. Study code.
