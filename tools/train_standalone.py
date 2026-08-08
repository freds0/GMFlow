"""Standalone training script for 3D GMFlow - no mmcv/mmgen dependency.

Usage:
    python tools/train_standalone.py \
        --cache_dir /home/fred/Projetos/Einstein/openbhb_train_sample/train_cache_64 \
        --metadata /home/fred/Projetos/Einstein/openbhb_train_sample/train/quasiraw_3d/metadata.tsv

    # Resume from checkpoint:
    python tools/train_standalone.py \
        --cache_dir data/openbhb/train_cache_64 \
        --metadata data/openbhb/train/quasiraw_3d/metadata.tsv \
        --resume checkpoints/gmflow3d/latest.pt
"""

import os
import sys
import types
import argparse
import importlib.util
import time
import json
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Mock framework (bypass mmcv/mmgen)
# ---------------------------------------------------------------------------

def mock_framework():
    """Mock mmcv/mmgen so we can import lib modules without installing them."""
    mmcv_mocks = ['mmcv', 'mmcv.runner', 'mmcv.runner.fp16_utils', 'mmcv.cnn',
                  'mmcv.cnn.utils', 'mmcv.parallel', 'mmcv.runner.checkpoint',
                  'mmcv.runner.hooks', 'mmcv.utils']
    for name in mmcv_mocks:
        m = types.ModuleType(name)
        m.__path__ = ['/tmp/fake_' + name.replace('.', '_')]
        m.__package__ = name
        sys.modules[name] = m

    sys.modules['mmcv.runner.fp16_utils'].force_fp32 = lambda: (lambda fn: fn)

    def load_checkpoint(*a, **kw): pass
    def _load_checkpoint(*a, **kw): return {}
    def load_state_dict(*a, **kw): pass
    sys.modules['mmcv.runner'].load_checkpoint = load_checkpoint
    sys.modules['mmcv.runner']._load_checkpoint = _load_checkpoint
    sys.modules['mmcv.runner'].load_state_dict = load_state_dict

    def constant_init(m, val=0):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.constant_(m.weight, val)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, val)

    def xavier_init(m, distribution='uniform'):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def kaiming_init(m, **kw):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.kaiming_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    sys.modules['mmcv.cnn'].constant_init = constant_init
    sys.modules['mmcv.cnn'].xavier_init = xavier_init
    sys.modules['mmcv.cnn'].kaiming_init = kaiming_init
    sys.modules['mmcv.cnn.utils'].constant_init = constant_init
    sys.modules['mmcv.cnn.utils'].xavier_init = xavier_init
    sys.modules['mmcv.cnn.utils'].kaiming_init = kaiming_init

    for name in ['mmgen', 'mmgen.models', 'mmgen.models.builder',
                 'mmgen.models.architectures', 'mmgen.models.architectures.common',
                 'mmgen.models.diffusions', 'mmgen.models.diffusions.utils',
                 'mmgen.models.losses', 'mmgen.models.losses.ddpm_loss',
                 'mmgen.models.losses.utils',
                 'mmgen.datasets', 'mmgen.datasets.builder', 'mmgen.utils']:
        sys.modules[name] = types.ModuleType(name)

    class FakeRegistry:
        def register_module(self):
            return lambda cls: cls

    sys.modules['mmgen.models.builder'].MODULES = FakeRegistry()
    sys.modules['mmgen.models.builder'].MODELS = FakeRegistry()
    sys.modules['mmgen.models.builder'].build_module = lambda cfg, **kw: None
    sys.modules['mmgen.models'].MODULES = FakeRegistry()
    sys.modules['mmgen.utils'].get_root_logger = lambda: None
    def _get_module_device(m):
        try:
            return next(m.parameters()).device
        except StopIteration:
            return torch.device('cpu')
    sys.modules['mmgen.models.architectures.common'].get_module_device = _get_module_device

    def _get_noise_batch(noise, shape, num_timesteps=1000, num_batches=1, timesteps_noise=False):
        return torch.randn(num_batches, *shape)
    sys.modules['mmgen.models.diffusions.utils']._get_noise_batch = _get_noise_batch

    def weighted_loss(fn):
        import functools
        @functools.wraps(fn)
        def wrapper(*args, reduction='mean', **kwargs):
            loss = fn(*args, **kwargs)
            if reduction == 'mean':
                return loss.mean()
            elif reduction == 'flatmean':
                return loss.flatten(1).mean(dim=1)
            return loss
        return wrapper
    sys.modules['mmgen.models.losses.utils'].weighted_loss = weighted_loss

    class FakeDDPMLoss(nn.Module):
        def __init__(self, log_cfgs=None, reduction='mean', loss_name='loss',
                     rescale_mode=None, rescale_cfg=None, sampler=None, weight=None, **kwargs):
            super().__init__()
            self.reduction = reduction
            self.loss_name = loss_name
            self.log_vars = {}
            self.log_fn_list = []
            self.rescale_fn = lambda loss, t: loss
            if log_cfgs is not None and log_cfgs.get('type', None) == 'quartile':
                self.log_fn_list.append(
                    partial(self.quartile_log_collect,
                            total_timesteps=log_cfgs.get('total_timesteps', 1000),
                            prefix_name=log_cfgs.get('prefix_name', 'loss')))
    sys.modules['mmgen.models.losses.ddpm_loss'].DDPMLoss = FakeDDPMLoss

    def _reduce_loss(loss, reduction):
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'flatmean':
            return loss.flatten(1).mean(dim=1)
        elif reduction == 'sum':
            return loss.sum()
        return loss
    sys.modules['mmgen.models.losses.ddpm_loss'].reduce_loss = _reduce_loss
    sys.modules['mmgen.models.losses.ddpm_loss'].mse_loss = lambda **kw: torch.tensor(0.0)

    sys.modules['mmgen.datasets.builder'].DATASETS = FakeRegistry()
    sys.modules['mmcv.parallel'].DataContainer = lambda x, cpu_only=True: x

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    mock_packages = {
        'lib': os.path.join(base, 'lib'),
        'lib.core': os.path.join(base, 'lib', 'core'),
        'lib.core.utils': os.path.join(base, 'lib', 'core', 'utils'),
        'lib.core.evaluation': os.path.join(base, 'lib', 'core', 'evaluation'),
        'lib.apis': os.path.join(base, 'lib', 'apis'),
        'lib.datasets': os.path.join(base, 'lib', 'datasets'),
        'lib.models': os.path.join(base, 'lib', 'models'),
        'lib.models.losses': os.path.join(base, 'lib', 'models', 'losses'),
        'lib.models.architecture': os.path.join(base, 'lib', 'models', 'architecture'),
        'lib.models.diffusions': os.path.join(base, 'lib', 'models', 'diffusions'),
        'lib.models.diffusions.schedulers': os.path.join(base, 'lib', 'models', 'diffusions', 'schedulers'),
        'lib.ops': os.path.join(base, 'lib', 'ops'),
        'lib.ops.gmflow_ops': os.path.join(base, 'lib', 'ops', 'gmflow_ops'),
        'lib.parallel': os.path.join(base, 'lib', 'parallel'),
        'lib.runner': os.path.join(base, 'lib', 'runner'),
        'lib.pipelines': os.path.join(base, 'lib', 'pipelines'),
    }

    for name, path in mock_packages.items():
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [path]
            m.__package__ = name
            sys.modules[name] = m

    sys.modules['lib.core'].reduce_mean = lambda x: x
    sys.modules['lib.core'].rgetattr = lambda obj, attr: getattr(obj, attr)

    _gf_spec = importlib.util.spec_from_file_location(
        'lib.models.diffusions.gaussian_flow',
        os.path.join(base, 'lib', 'models', 'diffusions', 'gaussian_flow.py'))
    _gf_mod = importlib.util.module_from_spec(_gf_spec)
    sys.modules['lib.models.diffusions.gaussian_flow'] = _gf_mod
    _gf_spec.loader.exec_module(_gf_mod)
    sys.modules['lib.models.diffusions'].GaussianFlow = _gf_mod.GaussianFlow

    _sched_init = os.path.join(base, 'lib', 'models', 'diffusions', 'schedulers', '__init__.py')
    _sched_spec = importlib.util.spec_from_file_location(
        'lib.models.diffusions.schedulers', _sched_init,
        submodule_search_locations=[os.path.join(base, 'lib', 'models', 'diffusions', 'schedulers')])
    _sched_mod = importlib.util.module_from_spec(_sched_spec)
    sys.modules['lib.models.diffusions.schedulers'] = _sched_mod
    _sched_spec.loader.exec_module(_sched_mod)
    sys.modules['lib.models.diffusions'].schedulers = _sched_mod


# ---------------------------------------------------------------------------
# Simple dataset (no mmcv dependency)
# ---------------------------------------------------------------------------

class OpenBHBStandaloneDataset(Dataset):
    """Load OpenBHB from a prepared .pt cache or directly from .npy volumes."""

    def __init__(
            self,
            metadata_path,
            cache_dir=None,
            data_root=None,
            volume_size=64,
            npy_suffix='_quasiraw_3d',
            age_min=None,
            age_max=None,
            random_flip=True,
            negative_age=-1.0,
            split=None,
            clip_range=3.0):
        self.cache_dir = cache_dir
        self.data_root = data_root
        self.volume_size = int(volume_size)
        self.npy_suffix = npy_suffix
        self.random_flip = random_flip
        self.negative_age = negative_age
        self.clip_range = float(clip_range)

        metadata = pd.read_csv(metadata_path, sep='\t')
        if split and 'split' in metadata.columns:
            metadata = metadata[metadata['split'] == split]
            print(f'Filtered to split={split}: {len(metadata)} rows')
        if 'age' not in metadata.columns:
            raise ValueError(f'Metadata has no age column: {metadata_path}')

        metadata = metadata.copy()
        metadata['age'] = pd.to_numeric(metadata['age'], errors='coerce')
        metadata = metadata[np.isfinite(metadata['age'])]

        self.subjects = []
        for _, row in metadata.iterrows():
            participant_id = str(
                row.get('participant_id', row.get('scan_id', 'unknown')))
            scan_value = row.get('scan_id', participant_id)
            scan_id = participant_id if pd.isna(scan_value) else str(scan_value)

            cache_path = None
            cache_value = row.get('cache_file', None)
            if cache_dir is not None:
                cache_name = (
                    f'{participant_id}.pt'
                    if cache_value is None or pd.isna(cache_value)
                    else str(cache_value))
                cache_path = (
                    cache_name if os.path.isabs(cache_name)
                    else os.path.join(cache_dir, cache_name))

            raw_path = None
            if data_root is not None:
                raw_path = os.path.join(
                    data_root, f'{participant_id}{npy_suffix}.npy')

            if cache_path is not None and os.path.exists(cache_path):
                path, source = cache_path, 'cache'
            elif raw_path is not None and os.path.exists(raw_path):
                path, source = raw_path, 'raw'
            else:
                continue

            self.subjects.append(dict(
                participant_id=participant_id,
                scan_id=scan_id,
                age=float(row['age']),
                path=path,
                source=source))

        if not self.subjects:
            raise RuntimeError(
                f'No OpenBHB volumes were found using metadata={metadata_path}, '
                f'cache_dir={cache_dir}, data_root={data_root}')

        ages = np.asarray(
            [subject['age'] for subject in self.subjects], dtype=np.float64)
        self.age_min = float(ages.min()) if age_min is None else float(age_min)
        self.age_max = float(ages.max()) if age_max is None else float(age_max)
        if self.age_max <= self.age_min:
            raise ValueError(
                f'age_max ({self.age_max}) must be greater than '
                f'age_min ({self.age_min})')
        if ages.min() < self.age_min or ages.max() > self.age_max:
            raise ValueError(
                f'Metadata ages [{ages.min()}, {ages.max()}] are outside '
                f'configured range [{self.age_min}, {self.age_max}]')
        self.age_range = self.age_max - self.age_min
        self.volume_shape = (self.volume_size,) * 3

        cached = sum(subject['source'] == 'cache' for subject in self.subjects)
        raw = len(self.subjects) - cached
        print(
            f'OpenBHBStandaloneDataset: loaded {len(self.subjects)} scans '
            f'({cached} cached, {raw} raw)')
        print(f'  Age range: [{self.age_min:g}, {self.age_max:g}] years')
        print(f'  Output volume shape: {self.volume_shape}')

    def __len__(self):
        return len(self.subjects)

    def _load_volume(self, subject):
        if subject['source'] == 'cache':
            volume = torch.load(
                subject['path'], map_location='cpu', weights_only=True)
            if not isinstance(volume, torch.Tensor):
                raise TypeError(
                    f'Expected tensor cache at {subject["path"]}, '
                    f'got {type(volume).__name__}')
        else:
            volume = torch.from_numpy(
                np.load(subject['path']).astype(np.float32))

        volume = volume.float().squeeze()
        if volume.dim() != 3:
            raise ValueError(
                f'Expected a 3D volume at {subject["path"]}, '
                f'got shape {tuple(volume.shape)}')
        volume = volume.unsqueeze(0).unsqueeze(0)
        if tuple(volume.shape[-3:]) != self.volume_shape:
            volume = torch.nn.functional.interpolate(
                volume,
                size=self.volume_shape,
                mode='trilinear',
                align_corners=False)
        volume = volume.squeeze(0)

        volume = torch.nan_to_num(volume)
        mean = volume.mean()
        std = volume.std().clamp(min=1e-6)
        volume = ((volume - mean) / std).clamp(
            -self.clip_range, self.clip_range)
        return volume / self.clip_range

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        volume = self._load_volume(subject)

        if self.random_flip and torch.rand(()).item() < 0.5:
            volume = volume.flip(-1)

        age_norm = (subject['age'] - self.age_min) / self.age_range
        return dict(
            volumes=volume,
            age=torch.tensor(age_norm, dtype=torch.float32),
            negative_age=torch.tensor(self.negative_age, dtype=torch.float32),
            participant_id=subject['participant_id'],
            scan_id=subject['scan_id'])


# Backwards-compatible name used by existing scripts.
OpenBHBSimple = OpenBHBStandaloneDataset

def collate_fn(batch):
    """Simple collate that stacks tensors and collects strings."""
    result = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[key] = torch.stack(vals)
        else:
            result[key] = vals
    return result


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters()}
        self.backup = {}

    def update(self, model):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        """Apply EMA weights (for inference)."""
        self.backup = {name: p.data.clone() for name, p in model.named_parameters()}
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights."""
        for name, p in model.named_parameters():
            p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        loaded = 0
        skipped = []
        for key, value in state_dict.items():
            if (
                    key in self.shadow
                    and self.shadow[key].shape == value.shape):
                self.shadow[key].copy_(value)
                loaded += 1
            else:
                skipped.append(key)
        if skipped:
            print(
                f'EMA: loaded {loaded} tensors and skipped '
                f'{len(skipped)} incompatible tensors')


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(args, device):
    """Build the 3D GMFlow model used by the standalone trainer."""
    from lib.models.architecture.gmflow3d import _GMDiTTransformer3DModel
    from lib.models.diffusions.gmflow3d import GMFlow3D
    from lib.models.diffusions.sampler import ContinuousTimeStepSampler
    from lib.models.losses.diffusion_loss import GMFlowNLLLoss3D

    model_volume_size = (
        args.volume_size // 2 if args.use_wavelet else args.volume_size)
    model_channels = 8 if args.use_wavelet else 1

    denoising = _GMDiTTransformer3DModel(
        num_gaussians=args.num_gaussians,
        num_attention_heads=args.num_heads,
        attention_head_dim=args.head_dim,
        in_channels=model_channels,
        out_channels=model_channels,
        num_layers=args.num_layers,
        sample_size=model_volume_size,
        patch_size=args.patch_size,
        overlap_patch_embed=args.overlap_patch_embed,
        local_refinement=args.local_refinement,
        refinement_hidden_channels=args.refinement_hidden_channels,
        refinement_num_layers=args.refinement_num_layers,
        gm_per_channel_logstd=(
            args.per_channel_logstd and args.use_wavelet),
        age_dropout_prob=args.age_dropout_prob,
        age_fourier_frequencies=args.age_fourier_frequencies)
    denoising.init_weights()
    if args.checkpointing:
        denoising.gradient_checkpointing = True

    band_weights = (
        [1.0, 2.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0]
        if args.use_wavelet else None)
    boundary_period = args.boundary_period
    if boundary_period is None:
        boundary_period = args.patch_size * (2 if args.use_wavelet else 1)

    flow_loss = GMFlowNLLLoss3D(
        weight_scale=2.0,
        band_weights=band_weights,
        mixture_mean_weight=args.mixture_mean_weight,
        voxel_gradient_weight=args.voxel_gradient_weight,
        reconstruct_wavelet=args.use_wavelet,
        boundary_period=boundary_period,
        boundary_weight=args.boundary_weight,
        data_info=dict(
            pred_means='means',
            target='x_t_low',
            pred_logstds='logstds',
            pred_logweights='logweights'),
        log_cfgs=dict(
            type='quartile',
            prefix_name='loss_trans',
            total_timesteps=1000))

    timestep_sampler = ContinuousTimeStepSampler(
        num_timesteps=1000, shift=1.0, logit_normal_enable=True)

    # Assemble GMFlow3D without relying on mmcv/mmgen builders.
    diffusion = GMFlow3D.__new__(GMFlow3D)
    torch.nn.Module.__init__(diffusion)
    diffusion.num_timesteps = 1000
    diffusion.denoising = denoising
    diffusion.denoising_mean_mode = 'U'
    diffusion.timestep_sampler = timestep_sampler
    diffusion.flow_loss = flow_loss
    diffusion.train_cfg = dict(
        trans_ratio=0.5,
        eps=1e-4,
        randomize_trans_ratio=args.randomize_trans_ratio,
        trans_ratio_min=args.trans_ratio_min,
        trans_ratio_max=args.trans_ratio_max)
    diffusion.test_cfg = dict(
        sampler='FlowEulerODE',
        output_mode='mean',
        num_timesteps=args.sample_timesteps,
        num_substeps=args.sample_substeps,
        order=args.sample_order)
    diffusion.use_wavelet = args.use_wavelet
    diffusion.randomize_trans_ratio = args.randomize_trans_ratio
    diffusion.trans_ratio_min = args.trans_ratio_min
    diffusion.trans_ratio_max = args.trans_ratio_max
    diffusion.intermediate_x_t = []
    diffusion.intermediate_x_0 = []
    diffusion = diffusion.to(device)

    num_params = sum(p.numel() for p in diffusion.parameters())
    print(f'Model parameters: {num_params:,}')
    return diffusion


# ---------------------------------------------------------------------------
# Save / Load checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, ema, optimizer, scaler, iteration, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        iteration=iteration,
        args=vars(args))
    if ema is not None:
        state['ema'] = ema.state_dict()
    if scaler is not None:
        state['scaler'] = scaler.state_dict()
    torch.save(state, path)

    # Also save as 'latest.pt'
    latest_path = os.path.join(os.path.dirname(path), 'latest.pt')
    torch.save(state, latest_path)
    print(f'Saved checkpoint: {path}')


def _load_compatible_state(module, state_dict, label):
    current = module.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    result = module.load_state_dict(compatible, strict=False)
    print(
        f'{label}: loaded {len(compatible)} tensors, '
        f'skipped {len(skipped)}, missing {len(result.missing_keys)}')
    if skipped:
        print(
            f'  Incompatible keys (first 8): '
            f'{", ".join(skipped[:8])}')
    return skipped


def load_checkpoint(
        path,
        model,
        ema=None,
        optimizer=None,
        scaler=None,
        resume_weights_only=False):
    state = torch.load(path, map_location='cpu', weights_only=False)
    _load_compatible_state(model, state['model'], 'Model')

    if ema is not None and 'ema' in state:
        ema.load_state_dict(state['ema'])

    if not resume_weights_only:
        if optimizer is not None and 'optimizer' in state:
            try:
                optimizer.load_state_dict(state['optimizer'])
            except ValueError as error:
                print(
                    'WARNING: optimizer state is incompatible with the '
                    f'updated architecture and was not loaded: {error}')
        if scaler is not None and 'scaler' in state:
            scaler.load_state_dict(state['scaler'])

    iteration = 0 if resume_weights_only else state.get('iteration', 0)
    mode = 'weights only' if resume_weights_only else 'full state'
    print(f'Resumed {mode} from {path}, iteration {iteration}')
    return iteration


# ---------------------------------------------------------------------------
# Inference / sampling
# ---------------------------------------------------------------------------

def _normalize_slice(img):
    """Normalize a 2D numpy slice to [0, 1] for image logging."""
    vmin, vmax = img.min(), img.max()
    if vmax - vmin > 1e-8:
        return (img - vmin) / (vmax - vmin)
    return np.zeros_like(img)


def _periodic_boundary_ratio(volume, period):
    """Measure gradient energy at periodic patch borders versus elsewhere."""
    ratios = []
    for axis in range(3):
        gradient = np.abs(np.diff(volume, axis=axis))
        indices = np.arange(gradient.shape[axis])
        boundary = (indices + 1) % period == 0
        if not boundary.any() or boundary.all():
            continue
        boundary_mean = np.take(
            gradient, indices[boundary], axis=axis).mean()
        interior_mean = np.take(
            gradient, indices[~boundary], axis=axis).mean()
        ratios.append(float(boundary_mean / max(interior_mean, 1e-12)))
    return float(np.mean(ratios)) if ratios else 1.0


class TrainLogger:
    """Unified logger that dispatches to TensorBoard, W&B, or both."""

    def __init__(self, work_dir, backends, wandb_project=None, wandb_name=None,
                 wandb_entity=None, config=None):
        self.tb_writer = None
        self.wandb_run = None

        if 'tensorboard' in backends:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(os.path.join(work_dir, 'tb'))
                print(f'TensorBoard: {os.path.join(work_dir, "tb")}')
            except ImportError:
                print('WARNING: tensorboard not installed, skipping TensorBoard logging')

        if 'wandb' in backends:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project or 'gmflow3d',
                    name=wandb_name,
                    entity=wandb_entity,
                    dir=work_dir,
                    config=config,
                    resume='allow')
                print(f'W&B run: {self.wandb_run.url}')
            except ImportError:
                print('WARNING: wandb not installed, skipping W&B logging')

    @property
    def enabled(self):
        return self.tb_writer is not None or self.wandb_run is not None

    def log_scalars(self, scalars, step):
        if self.tb_writer is not None:
            for k, v in scalars.items():
                self.tb_writer.add_scalar(k, v, step)
        if self.wandb_run is not None:
            import wandb
            wandb.log({k: v for k, v in scalars.items()}, step=step)

    def log_image(self, tag, img_array, step, caption=None):
        """Log a 2D numpy array (H, W) or (C, H, W) as an image."""
        if self.tb_writer is not None:
            if img_array.ndim == 2:
                img_array_chw = img_array[None]  # (1, H, W)
            else:
                img_array_chw = img_array
            self.tb_writer.add_image(tag, img_array_chw, step, dataformats='CHW')
        if self.wandb_run is not None:
            import wandb
            img = img_array if img_array.ndim == 2 else img_array.transpose(1, 2, 0)
            wandb.log({tag: wandb.Image(img, caption=caption)}, step=step)

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()


@torch.no_grad()
def run_inference(model, device, args, iteration, out_dir, logger=None):
    """Generate sample volumes at fixed ages for monitoring."""
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    ages = [0.0, 0.25, 0.5, 0.75, 1.0]  # normalized
    age_real = [
        a * (args.resolved_age_max - args.resolved_age_min)
        + args.resolved_age_min
        for a in ages]

    vol_shape = (1, args.volume_size, args.volume_size, args.volume_size)
    seed = 42

    for age_norm, age_yr in zip(ages, age_real):
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(1, *vol_shape, device=device, generator=generator)
        age_t = torch.tensor([age_norm], device=device)

        # Without CFG
        output = model(
            noise=noise,
            age=age_t,
            guidance_scale=0.0,
            test_cfg_override=dict(
                sampler='FlowEulerODE',
                output_mode='mean',
                num_timesteps=args.sample_timesteps,
                num_substeps=args.sample_substeps,
                order=args.sample_order))

        vol_np = output[0].cpu().numpy()  # (1, D, H, W)
        fname = f'iter{iteration:07d}_age{age_yr:.0f}.npy'
        np.save(os.path.join(out_dir, fname), vol_np)

        # Log mid-slices
        if logger is not None and logger.enabled:
            vol = vol_np[0]  # (D, H, W)
            d, h, w = vol.shape
            axial = _normalize_slice(vol[d // 2, :, :])
            coronal = _normalize_slice(vol[:, h // 2, :])
            sagittal = _normalize_slice(vol[:, :, w // 2])

            tag = f'samples/age_{age_yr:.0f}yr'
            boundary_period = args.patch_size * (
                2 if args.use_wavelet else 1)
            boundary_ratio = _periodic_boundary_ratio(
                vol, boundary_period)
            logger.log_scalars(
                {f'{tag}/patch_boundary_ratio': boundary_ratio},
                iteration)
            logger.log_image(f'{tag}/axial', axial, iteration,
                             caption=f'age={age_yr:.0f} axial')
            logger.log_image(f'{tag}/coronal', coronal, iteration,
                             caption=f'age={age_yr:.0f} coronal')
            logger.log_image(f'{tag}/sagittal', sagittal, iteration,
                             caption=f'age={age_yr:.0f} sagittal')

    print(f'Saved {len(ages)} sample volumes to {out_dir}')


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name()}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision('high')

    # Dataset
    dataset = OpenBHBStandaloneDataset(
        cache_dir=args.cache_dir,
        data_root=args.data_root,
        metadata_path=args.metadata,
        volume_size=args.volume_size,
        npy_suffix=args.npy_suffix,
        age_min=args.age_min,
        age_max=args.age_max,
        random_flip=True,
        split=args.split)
    args.resolved_age_min = dataset.age_min
    args.resolved_age_max = dataset.age_max
    if args.batch_size > len(dataset):
        raise ValueError(
            f'Batch size {args.batch_size} exceeds dataset size '
            f'{len(dataset)} while drop_last=True')

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0)

    # Model
    model = build_model(args, device)

    # EMA
    ema = EMAModel(model, decay=args.ema_decay) if args.use_ema else None

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay)

    # AMP scaler
    use_amp = args.autocast_dtype is not None and device.type == 'cuda'
    autocast_dtype = getattr(torch, args.autocast_dtype) if args.autocast_dtype else None
    scaler = torch.cuda.amp.GradScaler() if (use_amp and autocast_dtype == torch.float16) else None

    # Resume
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(
            args.resume,
            model,
            ema,
            optimizer,
            scaler,
            resume_weights_only=args.resume_weights_only)

    # LR warmup
    def get_lr(iteration):
        if iteration < args.warmup_iters:
            return args.lr * (iteration + 1) / args.warmup_iters
        return args.lr

    # Logging
    os.makedirs(args.work_dir, exist_ok=True)
    log_path = os.path.join(args.work_dir, 'train_log.jsonl')
    logger = TrainLogger(
        work_dir=args.work_dir,
        backends=args.logger,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
        config=vars(args))

    # Training
    model.train()
    data_iter = iter(dataloader)
    accum_loss = 0.0
    accum_steps = 0
    t0 = time.time()

    print(f'\nStarting training from iteration {start_iter}')
    print(f'  Total iterations: {args.total_iters}')
    print(f'  Batch size: {args.batch_size} x {args.grad_accum} (accum) = {args.batch_size * args.grad_accum}')
    print(f'  LR: {args.lr}, warmup: {args.warmup_iters} iters')
    print(f'  AMP: {args.autocast_dtype or "disabled"}')
    print(f'  EMA: {"enabled" if args.use_ema else "disabled"}')
    print(f'  Checkpointing: {"enabled" if args.checkpointing else "disabled"}')
    print(f'  Wavelet: {"enabled" if args.use_wavelet else "disabled"}')
    print(
        f'  Sampler: order {args.sample_order}, '
        f'{args.sample_timesteps} x {args.sample_substeps} steps')
    print(
        f'  Age range: [{args.resolved_age_min:g}, '
        f'{args.resolved_age_max:g}] years')
    print(f'  Loggers: {", ".join(args.logger)}')
    print()

    for iteration in range(start_iter, args.total_iters):
        # Get batch (cycle through dataloader)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        volumes = batch['volumes'].to(device, non_blocking=True)
        age = batch['age'].to(device, non_blocking=True)

        # Age dropout for CFG
        if args.prob_age < 1.0:
            neg_age = batch['negative_age'].to(device, non_blocking=True)
            mask = torch.rand_like(age) >= args.prob_age
            age = torch.where(mask, neg_age, age)

        # LR scheduling
        lr = get_lr(iteration)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward
        with torch.autocast(device_type='cuda', enabled=use_amp, dtype=autocast_dtype):
            loss, log_vars = model(volumes, return_loss=True, age=age)
            loss = loss / args.grad_accum

        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_loss += float(loss.detach()) * args.grad_accum
        accum_steps += 1

        # Optimizer step (after gradient accumulation)
        if accum_steps >= args.grad_accum:
            # Gradient clipping
            if args.grad_clip > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip)
            else:
                grad_norm = torch.tensor(0.0)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            # EMA update
            if ema is not None:
                ema.update(model)

            # Logging
            if (iteration + 1) % args.log_interval == 0:
                elapsed = time.time() - t0
                avg_loss = accum_loss / accum_steps
                it_per_sec = args.log_interval / elapsed

                log_entry = dict(
                    iter=iteration + 1,
                    loss=round(avg_loss, 5),
                    lr=round(lr, 7),
                    grad_norm=round(float(grad_norm), 3),
                    it_s=round(it_per_sec, 2))

                print(f'[iter {iteration+1:>7d}] loss={avg_loss:.4f}  '
                      f'lr={lr:.1e}  grad_norm={float(grad_norm):.2f}  '
                      f'{it_per_sec:.2f} it/s')

                with open(log_path, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

                if logger.enabled:
                    scalars = {
                        'train/loss': avg_loss,
                        'train/lr': lr,
                        'train/grad_norm': float(grad_norm),
                    }
                    for k, v in log_vars.items():
                        scalars[f'train/{k}'] = float(v)
                    logger.log_scalars(scalars, iteration + 1)

                t0 = time.time()

            accum_loss = 0.0
            accum_steps = 0

        # Save checkpoint
        if (iteration + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(args.work_dir, 'checkpoints', f'iter_{iteration+1:07d}.pt')
            save_checkpoint(ckpt_path, model, ema, optimizer, scaler, iteration + 1, args)

        # Sample
        if (iteration + 1) % args.sample_interval == 0:
            sample_dir = os.path.join(args.work_dir, 'samples')
            if ema is not None:
                ema.apply(model)
                run_inference(model, device, args, iteration + 1, sample_dir, logger)
                ema.restore(model)
            else:
                run_inference(model, device, args, iteration + 1, sample_dir, logger)
            model.train()

    # Final save
    ckpt_path = os.path.join(args.work_dir, 'checkpoints', f'iter_{args.total_iters:07d}.pt')
    save_checkpoint(ckpt_path, model, ema, optimizer, scaler, args.total_iters, args)
    logger.close()
    print('\nTraining complete!')


def main():
    parser = argparse.ArgumentParser(description='Standalone GMFlow3D training')

    default_root = os.environ.get(
        'OPENBHB_ROOT',
        '/media/fred/FRED5TB/Einstein/Open_BHB_processado')

    # Data
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=os.environ.get('OPENBHB_CACHE_DIR'),
        help='Optional directory with prepared .pt volumes')
    parser.add_argument(
        '--data_root',
        type=str,
        default=os.environ.get(
            'OPENBHB_DATA_ROOT',
            os.path.join(default_root, 'train', 'quasiraw_3d')),
        help='Directory with raw/preprocessed OpenBHB .npy volumes')
    parser.add_argument(
        '--metadata',
        type=str,
        default=os.environ.get(
            'OPENBHB_METADATA',
            os.path.join(default_root, 'train.tsv')),
        help='Path to metadata.tsv or train.tsv')
    parser.add_argument('--npy_suffix', type=str, default='_quasiraw_3d')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--age_min', type=float, default=None)
    parser.add_argument('--age_max', type=float, default=None)

    # Model architecture
    parser.add_argument('--num_gaussians', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--head_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--volume_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--age_dropout_prob', type=float, default=0.0)
    parser.add_argument('--age_fourier_frequencies', type=int, default=8)

    parser.add_argument(
        '--use_wavelet', action='store_true', default=True,
        help='Train in one-level 3D Haar space (default)')
    parser.add_argument(
        '--no_wavelet', action='store_false', dest='use_wavelet')

    parser.add_argument(
        '--overlap_patch_embed', action='store_true', default=True,
        help='Use overlapping Conv3d patch embedding (default)')
    parser.add_argument(
        '--no_overlap_patch_embed',
        action='store_false',
        dest='overlap_patch_embed')

    parser.add_argument(
        '--local_refinement', action='store_true', default=True,
        help='Enable zero-initialized local mixture-mean refinement (default)')
    parser.add_argument(
        '--no_local_refinement',
        action='store_false',
        dest='local_refinement')
    parser.add_argument('--refinement_hidden_channels', type=int, default=64)
    parser.add_argument('--refinement_num_layers', type=int, default=2)

    parser.add_argument(
        '--per_channel_logstd', action='store_true', default=True,
        help='Use independent variance per wavelet subband (default)')
    parser.add_argument(
        '--global_logstd',
        action='store_false',
        dest='per_channel_logstd')

    parser.add_argument(
        '--checkpointing', action='store_true', default=True)
    parser.add_argument(
        '--no_checkpointing',
        action='store_false',
        dest='checkpointing')

    # Loss and flow path
    parser.add_argument('--mixture_mean_weight', type=float, default=0.05)
    parser.add_argument('--voxel_gradient_weight', type=float, default=0.25)
    parser.add_argument('--boundary_period', type=int, default=None)
    parser.add_argument('--boundary_weight', type=float, default=2.0)
    parser.add_argument(
        '--randomize_trans_ratio', action='store_true', default=True)
    parser.add_argument(
        '--fixed_trans_ratio',
        action='store_false',
        dest='randomize_trans_ratio')
    parser.add_argument('--trans_ratio_min', type=float, default=0.05)
    parser.add_argument('--trans_ratio_max', type=float, default=1.0)

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--warmup_iters', type=int, default=1000)
    parser.add_argument('--total_iters', type=int, default=100000)
    parser.add_argument('--prob_age', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--autocast_dtype',
        type=str,
        default='bfloat16',
        choices=['bfloat16', 'float16'])
    parser.add_argument(
        '--no_amp',
        action='store_const',
        const=None,
        dest='autocast_dtype')

    # EMA and sampling
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--no_ema', action='store_false', dest='use_ema')
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--sample_timesteps', type=int, default=25)
    parser.add_argument('--sample_substeps', type=int, default=4)
    parser.add_argument('--sample_order', type=int, choices=[1, 2], default=2)

    # I/O
    parser.add_argument(
        '--work_dir',
        type=str,
        default='work_dirs/gmflow3d_standalone')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--sample_interval', type=int, default=5000)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument(
        '--resume_weights_only',
        action='store_true',
        help='Load compatible model/EMA tensors and restart optimizer/iteration')

    # Logging
    parser.add_argument(
        '--logger',
        type=str,
        nargs='+',
        default=['tensorboard'],
        choices=['tensorboard', 'wandb'])
    parser.add_argument('--wandb_project', type=str, default='gmflow3d')
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)

    args = parser.parse_args()

    if not os.path.isfile(args.metadata):
        parser.error(f'Metadata file does not exist: {args.metadata}')
    if args.cache_dir is not None and not os.path.isdir(args.cache_dir):
        print(
            f'WARNING: cache directory does not exist; using .npy files: '
            f'{args.cache_dir}')
        args.cache_dir = None
    if args.data_root is not None and not os.path.isdir(args.data_root):
        if args.cache_dir is None:
            parser.error(f'Data root does not exist: {args.data_root}')
        args.data_root = None
    if args.use_wavelet and args.volume_size % 2 != 0:
        parser.error('--use_wavelet requires an even --volume_size')

    model_volume_size = (
        args.volume_size // 2 if args.use_wavelet else args.volume_size)
    if model_volume_size % args.patch_size != 0:
        parser.error(
            'Model-space volume size must be divisible by --patch_size')
    if not 0 <= args.prob_age <= 1:
        parser.error('--prob_age must be in [0, 1]')
    if not 0 < args.trans_ratio_min <= args.trans_ratio_max <= 1:
        parser.error(
            'Transition ratio bounds must satisfy 0 < min <= max <= 1')
    if args.mixture_mean_weight < 0 or args.voxel_gradient_weight < 0:
        parser.error('Auxiliary loss weights must be non-negative')

    train(args)


if __name__ == '__main__':
    mock_framework()
    main()
