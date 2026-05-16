# Gaussian Mixture Flow Matching Models (GMFlow)

Official PyTorch implementation of the paper:

**Gaussian Mixture Flow Matching Models [[arXiv](https://arxiv.org/abs/2504.05304)]**
<br>
In ICML 2025
<br>
[Hansheng Chen](https://lakonik.github.io/)<sup>1</sup>, 
[Kai Zhang](https://kai-46.github.io/website/)<sup>2</sup>,
[Hao Tan](https://research.adobe.com/person/hao-tan/)<sup>2</sup>,
[Zexiang Xu](https://zexiangxu.github.io/)<sup>3</sup>, 
[Fujun Luan](https://research.adobe.com/person/fujun/)<sup>2</sup>,
[Leonidas Guibas](https://geometry.stanford.edu/?member=guibas)<sup>1</sup>,
[Gordon Wetzstein](http://web.stanford.edu/~gordonwz/)<sup>1</sup>, 
[Sai Bi](https://sai-bi.github.io/)<sup>2</sup><br>
<sup>1</sup>Stanford University, <sup>2</sup>Adobe Research, <sup>3</sup>Hillbot
<br>

<img src="gmdit.png" width="600"  alt=""/>

<img src="gmdit_results.png" width="1000"  alt=""/>

## 🔥News

- [Nov 7, 2025] GM-Qwen-Image and GM-FLUX is now available for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) via the [ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow) extension. Supports 4-step sampling of Qwen-Image and Flux.1 dev using 8-bit models on a single consumer-grade GPU.

- [Oct 16, 2025] GMFlow is now fully merged into the **pi-Flow** codebase (see the new GMFlow code [here](https://github.com/Lakonik/piFlow/tree/main/configs/gmflow)). [pi-Flow](https://github.com/Lakonik/piFlow) introduces a novel imitation distillation method based on GMFlow for 1~4-step generation. 

## Highlights

GMFlow is an extension of diffusion/flow matching models.

- **Gaussian Mixture Output**: GMFlow expands the network's output layer to predict a Gaussian mixture (GM) distribution of flow velocity. Standard diffusion/flow matching models are special cases of GMFlow with a single Gaussian component.

- **Precise Few-Step Sampling**: GMFlow introduces novel **GM-SDE** and **GM-ODE** solvers that leverage analytic denoising distributions and velocity fields for precise few-step sampling.

- **Improved Classifier-Free Guidance (CFG)**: GMFlow introduces a **probabilistic guidance** scheme that mitigates the over-saturation issues of CFG and improves image generation quality.

- **Efficiency**: GMFlow maintains similar training and inference costs to standard diffusion/flow matching models.

## Installation

The code has been tested in the environment described as follows:

- Linux (tested on Ubuntu 20 and above)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) 11.8 and above
- [PyTorch](https://pytorch.org/get-started/previous-versions/) 2.1 and above

Other dependencies can be installed via `pip install -r requirements.txt`. 

An example of installation commands is shown below (assuming you have already installed CUDA Toolkit and configured the environment variables):

```bash
# Create conda environment
conda create -y -n gmflow python=3.10 numpy=1.26 ninja
conda activate gmflow

# Goto https://pytorch.org/ to select the appropriate version
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

This codebase may work on Windows systems, but it has not been tested extensively.

## GM-DiT ImageNet 256x256

### Inference

We provide a [Diffusers pipeline](lib/pipelines/gmdit_pipeline.py) for easy inference. The following code demonstrates how to sample images from the pretrained GM-DiT model using the GM-ODE 2 solver and the GM-SDE 2 solver.

```python
import torch
from huggingface_hub import snapshot_download
from lib.models.diffusions.schedulers import FlowEulerODEScheduler, GMFlowSDEScheduler
from lib.pipelines.gmdit_pipeline import GMDiTPipeline

# Currently the pipeline can only load local checkpoints, so we need to download the checkpoint first
ckpt = snapshot_download(repo_id='Lakonik/gmflow_imagenet_k8_ema')
pipe = GMDiTPipeline.from_pretrained(ckpt, variant='bf16', torch_dtype=torch.bfloat16)
pipe = pipe.to('cuda')

# Pick words that exist in ImageNet
words = ['jay', 'magpie']
class_ids = pipe.get_label_ids(words)

# Sample using GM-ODE 2 solver
pipe.scheduler = FlowEulerODEScheduler.from_config(pipe.scheduler.config)
generator = torch.manual_seed(42)
output = pipe(
    class_labels=class_ids,
    guidance_scale=0.45,
    num_inference_steps=32,
    num_inference_substeps=4,
    output_mode='mean',
    order=2,
    generator=generator)
for i, (word, image) in enumerate(zip(words, output.images)):
    image.save(f'{i:03d}_{word}_gmode2_step32.png')

# Sample using GM-SDE 2 solver (the first run may be slow due to CUDA compilation)
pipe.scheduler = GMFlowSDEScheduler.from_config(pipe.scheduler.config)
generator = torch.manual_seed(42)
output = pipe(
    class_labels=class_ids,
    guidance_scale=0.45,
    num_inference_steps=32,
    num_inference_substeps=1,
    output_mode='sample',
    order=2,
    generator=generator)
for i, (word, image) in enumerate(zip(words, output.images)):
    image.save(f'{i:03d}_{word}_gmsde2_step32.png')
```

The results will be saved under the current directory.

<img src="example_results.png" width="800"  alt=""/>

### Before Training: Data Preparation

Download [ILSVRC2012_img_train.tar](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php) and the [metadata](http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz). Extract the downloaded archives according to the following folder tree (or use symlinks).
```
./
├── configs/
├── data/
│   └── imagenet/
│       ├── train/
│       │   ├── n01440764/
│       │   │   ├── n01440764_10026.JPEG
│       │   │   ├── n01440764_10027.JPEG
│       │   │   …
│       │   ├── n01443537/
│       │   …
│       ├── imagenet1000_clsidx_to_labels.txt
│       ├── train.txt
|       …
├── lib/
├── tools/
…
```

Run the following command to prepare the ImageNet dataset using DDP on 1 node with 8 GPUs

```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/prepare_imagenet_dit.py
```

### Training

Run the following command to train the model using DDP on 1 node with 8 GPUs:

```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/train.py configs/gmflow_imagenet_k8_8gpus.py --launcher pytorch --diff_seed
```

Alternatively, you can start single-node DDP training from a Python script:

```bash
python train.py configs/gmflow_imagenet_k8_8gpus.py --gpu-ids 0 1 2 3 4 5 6 7
```

The config in [gmflow_imagenet_k8_8gpus.py](configs/gmflow_imagenet_k8_8gpus.py) specifies a training batch size of 512 images per GPU and an inference batch size of 125 images per GPU. Training requires 32GB of VRAM per GPU, and the validation step requires an additional 8GB of VRAM per GPU. If you are using 32GB GPUs, you can disable the validation step by adding the `--no-validate` flag to the training command. Alternatively, you can also edit the config file to adjust the batch sizes.

By default, checkpoints will be saved into [checkpoints/](checkpoints/), logs will be saved into [work_dirs/](work_dirs/), and sampled images will be saved into [viz/](viz/).

#### Resuming Training

If existing checkpoints are found, the training will automatically resume from the latest checkpoint.

#### Tensorboard

The logs can be plotted using Tensorboard. Run the following command to start Tensorboard:

```bash
tensorboard --logdir work_dirs/
```

### Evaluation

After training, to conduct a complete evaluation of the model under varying guidance scales, run the following command to start DDP evaluation on 1 node with 8 GPUs:

```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/test.py configs/gmflow_imagenet_k8_test.py checkpoints/gmflow_imagenet_k8_8gpus/latest.pth --launcher pytorch --diff_seed
```

Alternatively, you can start single-node DDP evaluation from a Python script:
```bash
python test.py configs/gmflow_imagenet_k8_test.py checkpoints/gmflow_imagenet_k8_8gpus/latest.pth --gpu-ids 0 1 2 3 4 5 6 7
```

The config in [gmflow_imagenet_k8_test.py](configs/gmflow_imagenet_k8_test.py) specifies an inference batch size of 125 images per GPU, which requires 35GB of VRAM per GPU. You can edit the config file to adjust the batch size.

The evaluation results will be saved to where the checkpoint is located, and the sampled images will be saved into [viz/](viz/).

## Toy Model on 2D Checkerboard

We provide a minimal GMFlow trainer in [train_toymodel.py](train_toymodel.py) for the toy model on the 2D checkerboard dataset. Run the following command to train the model:

```bash
python train_toymodel.py -k 64
```

This minimal trainer does not support transition loss and EMA. To reproduce the results in the paper, you can use the following command to start the full trainer:

```bash
python train.py configs/gmflow_checkerboard_k64.py --gpu-ids 0
```

This full trainer is not optimized for the simple 2D checkerboard dataset, so GPU usage may be inefficient.

## GMFlow 3D: Brain MRI Generation Conditioned on Age

This repository extends GMFlow to **3D volumetric brain MRI generation** using the [OpenBHB](https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset-age-prediction-and-debiasing) dataset, conditioned on continuous age (6–86 years).

### Key Differences from 2D GMFlow

- Operates on 5D tensors `(bs, C, D, H, W)` instead of 4D `(bs, C, H, W)`
- `PatchEmbed3D` using `Conv3d` → 4096 tokens for 64³ volumes with patch_size=4
- `AgeEmbedding` for continuous age conditioning (replaces discrete class labels)
- No VAE — works directly in voxel space at 64×64×64 resolution
- No spectral loss component (SpectrumMLP uses FFT2D)
- GM gaussian dimension at `-5` instead of `-4`

### Data Preparation

1. Download the OpenBHB dataset and place raw `.npy` volumes under `data/openbhb/train/quasiraw_3d/` along with `metadata.tsv`.

2. Preprocess volumes (downsample to 64³, normalize to [-1, 1]):

```bash
python tools/prepare_openbhb.py \
    --data_root data/openbhb/train/quasiraw_3d \
    --metadata data/openbhb/train/quasiraw_3d/metadata.tsv \
    --output_dir data/openbhb/train_cache_64 \
    --target_shape 64 64 64
```

### Training

The standalone training script requires **no mmcv/mmgen** — only PyTorch and diffusers:

```bash
python tools/train_standalone.py \
    --cache_dir data/openbhb/train_cache_64 \
    --metadata data/openbhb/train/quasiraw_3d/metadata.tsv \
    --work_dir work_dirs/gmflow3d_openbhb
```

For GPUs with limited VRAM (e.g., 12GB RTX 3060):

```bash
python tools/train_standalone.py \
    --cache_dir data/openbhb/train_cache_64 \
    --metadata data/openbhb/train/quasiraw_3d/metadata.tsv \
    --num_layers 8 --num_heads 8 --head_dim 64 \
    --batch_size 1 --grad_accum 8 \
    --work_dir work_dirs/gmflow3d_openbhb
```

Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `--volume_size` | 64 | Volume resolution (D=H=W) |
| `--patch_size` | 4 | Patch size for tokenization |
| `--num_layers` | 12 | Transformer depth |
| `--num_heads` | 12 | Attention heads |
| `--head_dim` | 64 | Head dimension (inner_dim = heads × head_dim) |
| `--batch_size` | 2 | Per-GPU batch size |
| `--grad_accum` | 4 | Gradient accumulation steps |
| `--autocast_dtype` | bfloat16 | AMP dtype (`--no_amp` to disable) |
| `--use_ema` / `--no_ema` | enabled | Exponential moving average |
| `--resume` | — | Resume from checkpoint |

### Training at Higher Resolutions (128³, 256³)

The model supports arbitrary volume resolutions. The sequence length scales as `(volume_size / patch_size)³`, and attention memory scales quadratically with sequence length, so higher resolutions require adjusting `patch_size`, model size, and batch size.

| Resolution | patch_size | Sequence length | Approx. VRAM (12 heads, 64 head_dim) |
|------------|------------|-----------------|---------------------------------------|
| 64³        | 4          | 16³ = 4,096     | ~12–16 GB                             |
| 128³       | 8          | 16³ = 4,096     | ~12–16 GB                             |
| 128³       | 4          | 32³ = 32,768    | ~100+ GB                              |
| 256³       | 8          | 32³ = 32,768    | ~100+ GB                              |
| 256³       | 16         | 16³ = 4,096     | ~12–16 GB                             |

**Step 1: Preprocess at the target resolution**

```bash
# 128³
python tools/prepare_openbhb.py \
    --data_root data/openbhb/train/quasiraw_3d \
    --metadata data/openbhb/train/quasiraw_3d/metadata.tsv \
    --output_dir data/openbhb/train_cache_128 \
    --target_shape 128 128 128

# 256³
python tools/prepare_openbhb.py \
    --data_root data/openbhb/train/quasiraw_3d \
    --metadata data/openbhb/train/quasiraw_3d/metadata.tsv \
    --output_dir data/openbhb/train_cache_256 \
    --target_shape 256 256 256
```

**Step 2: Train with matching `--volume_size` and `--patch_size`**

To keep the sequence length manageable on a single 24 GB GPU, increase `patch_size` proportionally:

```bash
# 128³ on a 24 GB GPU (RTX 3090/4090) — patch_size=8 keeps seq_len = 4096
python tools/train_standalone.py \
    --cache_dir data/openbhb/train_cache_128 \
    --metadata data/openbhb/train/quasiraw_3d/metadata.tsv \
    --volume_size 128 --patch_size 8 \
    --batch_size 1 --grad_accum 8 \
    --work_dir work_dirs/gmflow3d_openbhb_128

# 256³ on a 24 GB GPU — patch_size=16 keeps seq_len = 4096
python tools/train_standalone.py \
    --cache_dir data/openbhb/train_cache_256 \
    --metadata data/openbhb/train/quasiraw_3d/metadata.tsv \
    --volume_size 256 --patch_size 16 \
    --batch_size 1 --grad_accum 8 \
    --work_dir work_dirs/gmflow3d_openbhb_256
```

For multi-GPU setups with more VRAM (e.g., A100 80 GB), you can use smaller `patch_size` for finer-grained tokenization:

```bash
# 128³ with patch_size=4 (seq_len = 32768) — requires ~100 GB VRAM
python tools/train_standalone.py \
    --cache_dir data/openbhb/train_cache_128 \
    --metadata data/openbhb/train/quasiraw_3d/metadata.tsv \
    --volume_size 128 --patch_size 4 \
    --batch_size 1 --grad_accum 8 \
    --work_dir work_dirs/gmflow3d_openbhb_128_p4
```

> **Note:** Larger `patch_size` reduces the number of tokens but each patch covers a bigger spatial region. This trades spatial granularity for memory efficiency. For brain MRI, `patch_size=8` at 128³ or `patch_size=16` at 256³ are good starting points.

### Logging

By default, the standalone trainer logs to **TensorBoard**. You can also use **Weights & Biases (W&B)**, or both simultaneously.

**TensorBoard only (default):**

```bash
tensorboard --logdir work_dirs/gmflow3d_openbhb/tb
```

**W&B only:**

```bash
# First time: authenticate with your API key
pip install wandb
wandb login

# Train with W&B logging
python tools/train_standalone.py \
    --cache_dir data/openbhb/train_cache_64 \
    --metadata data/openbhb/train/quasiraw_3d/metadata.tsv \
    --logger wandb \
    --wandb_project gmflow3d \
    --wandb_name my_experiment \
    --work_dir work_dirs/gmflow3d_openbhb
```

**Both TensorBoard and W&B:**

```bash
python tools/train_standalone.py \
    --cache_dir data/openbhb/train_cache_64 \
    --metadata data/openbhb/train/quasiraw_3d/metadata.tsv \
    --logger tensorboard wandb \
    --wandb_project gmflow3d \
    --wandb_name my_experiment \
    --work_dir work_dirs/gmflow3d_openbhb
```

W&B logging options:

| Option | Default | Description |
|--------|---------|-------------|
| `--logger` | `tensorboard` | Logging backends (`tensorboard`, `wandb`, or both) |
| `--wandb_project` | `gmflow3d` | W&B project name |
| `--wandb_name` | auto-generated | W&B run name |
| `--wandb_entity` | — | W&B team/entity (for shared projects) |

W&B logs training loss, learning rate, gradient norm, and sample brain slices (axial, coronal, sagittal) at each sample interval. Runs can be resumed automatically if the `--work_dir` is the same.

### Inference

Generate brain MRI volumes for specific ages:

```bash
python tools/inference.py \
    --checkpoint work_dirs/gmflow3d_openbhb/checkpoints/latest.pt \
    --ages 10 25 50 75 \
    --output_dir output/samples
```

With classifier-free guidance and EMA weights:

```bash
python tools/inference.py \
    --checkpoint work_dirs/gmflow3d_openbhb/checkpoints/latest.pt \
    --ages 20 40 60 80 \
    --guidance_scale 0.04 \
    --use_ema \
    --num_samples 5 \
    --output_dir output/samples_cfg
```

Output formats: `.npy` (default), `.nii.gz` (NIfTI, requires nibabel), `.png` (center slices).

### Visualization

```bash
# Grid of center slices (axial, coronal, sagittal) for all generated volumes:
python tools/visualize.py output/samples/

# Interactive 3D slice explorer with sliders:
python tools/visualize.py output/samples/age050.0_seed42.npy --interactive

# Axial slice montage:
python tools/visualize.py output/samples/age050.0_seed42.npy --montage

# Compare real vs generated:
python tools/visualize.py \
    --real data/openbhb/train_cache_64/100053248969.pt \
    --generated output/samples/age025.0_seed42.npy

# Voxel intensity histograms:
python tools/visualize.py output/samples/ --histogram

# Save without displaying:
python tools/visualize.py output/samples/ --save figure.png --no_show
```

### 3D Architecture

| Component | File |
|-----------|------|
| PatchEmbed3D, AgeEmbedding, GMOutput3D, GMDiTTransformer3D | [lib/models/architecture/gmflow3d.py](lib/models/architecture/gmflow3d.py) |
| GMFlow3D diffusion (training + sampling) | [lib/models/diffusions/gmflow3d.py](lib/models/diffusions/gmflow3d.py) |
| 3D GM operations (gm_to_mean, gm_to_sample, etc.) | [lib/ops/gmflow_ops/gmflow_ops_3d.py](lib/ops/gmflow_ops/gmflow_ops_3d.py) |
| GMFlowNLLLoss3D | [lib/models/losses/diffusion_loss.py](lib/models/losses/diffusion_loss.py) |
| OpenBHB dataset | [lib/datasets/openbhb.py](lib/datasets/openbhb.py) |
| Standalone training (no mmcv) | [tools/train_standalone.py](tools/train_standalone.py) |
| Inference | [tools/inference.py](tools/inference.py) |
| Visualization | [tools/visualize.py](tools/visualize.py) |
| Data preprocessing | [tools/prepare_openbhb.py](tools/prepare_openbhb.py) |
| Smoke tests | [test_3d_smoke.py](test_3d_smoke.py) |

---

## GM-DiT ImageNet 256×256 (Original 2D)

## Essential Code

- Training
    - [train_toymodel.py](train_toymodel.py): A simplified training script for the 2D checkerboard experiment.
    - [gmflow.py](lib/models/diffusions/gmflow.py): The `forward_train` method contains the full training loop.
- Inference
    - [gmdit_pipeline.py](lib/pipelines/gmdit_pipeline.py): Full sampling code in the style of Diffusers.
    - [gmflow.py](lib/models/diffusions/gmflow.py): The `forward_test` method contains the same full sampling loop.
- Network
    - [gmflow.py](lib/models/architecture/gmflow.py): GMDiT and SpectrumMLP
    - [toymodels.py](lib/models/architecture/toymodels.py): MLP toy model for the 2D checkerboard experiment.
- GM math operations
    - [gmflow_ops](lib/ops/gmflow_ops/): A complete library of analytic operations for GM and Gaussian distributions.

## Citation
```
@inproceedings{gmflow,
  title={Gaussian Mixture Flow Matching Models},
  author={Hansheng Chen and Kai Zhang and Hao Tan and Zexiang Xu and Fujun Luan and Leonidas Guibas and Gordon Wetzstein and Sai Bi},
  booktitle={ICML},
  year={2025},
}
```
