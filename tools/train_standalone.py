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

class OpenBHBSimple(Dataset):
    """OpenBHB dataset that loads cached .pt volumes + age from metadata."""

    def __init__(self, cache_dir, metadata_path, age_min=6.0, age_max=86.0,
                 random_flip=True, negative_age=-1.0):
        self.cache_dir = cache_dir
        self.age_min = age_min
        self.age_range = age_max - age_min
        self.random_flip = random_flip
        self.negative_age = negative_age

        metadata = pd.read_csv(metadata_path, sep='\t')
        self.subjects = []
        for _, row in metadata.iterrows():
            pid = str(row['participant_id'])
            pt_path = os.path.join(cache_dir, f'{pid}.pt')
            if os.path.exists(pt_path):
                self.subjects.append(dict(
                    participant_id=pid,
                    age=float(row['age']),
                    path=pt_path))
        print(f'OpenBHBSimple: loaded {len(self.subjects)} subjects from {cache_dir}')

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        vol = torch.load(subject['path'], weights_only=True)  # (1, D, H, W)

        if self.random_flip and torch.rand(()).item() < 0.5:
            vol = vol.flip(-1)

        age_norm = (subject['age'] - self.age_min) / self.age_range

        return dict(
            volumes=vol,
            age=torch.tensor(age_norm, dtype=torch.float32),
            negative_age=torch.tensor(self.negative_age, dtype=torch.float32),
            participant_id=subject['participant_id'])


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
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(args, device):
    """Build GMFlow3D model from components."""
    from lib.models.architecture.gmflow3d import _GMDiTTransformer3DModel
    from lib.models.diffusions.gmflow3d import GMFlow3D
    from lib.models.diffusions.sampler import ContinuousTimeStepSampler
    from lib.models.losses.diffusion_loss import GMFlowNLLLoss3D

    # Denoising network
    denoising = _GMDiTTransformer3DModel(
        num_gaussians=args.num_gaussians,
        num_attention_heads=args.num_heads,
        attention_head_dim=args.head_dim,
        in_channels=1,
        num_layers=args.num_layers,
        sample_size=args.volume_size,
        patch_size=args.patch_size,
        age_dropout_prob=args.age_dropout_prob)
    denoising.init_weights()
    if args.checkpointing:
        denoising.gradient_checkpointing = True

    # Loss
    flow_loss = GMFlowNLLLoss3D(
        weight_scale=2.0,
        data_info=dict(
            pred_means='means',
            target='x_t_low',
            pred_logstds='logstds',
            pred_logweights='logweights'),
        log_cfgs=dict(type='quartile', prefix_name='loss_trans', total_timesteps=1000))

    # Timestep sampler
    timestep_sampler = ContinuousTimeStepSampler(
        num_timesteps=1000, shift=1.0, logit_normal_enable=True)

    # Assemble GMFlow3D (bypass build_module)
    diffusion = GMFlow3D.__new__(GMFlow3D)
    torch.nn.Module.__init__(diffusion)
    diffusion.num_timesteps = 1000
    diffusion.denoising = denoising
    diffusion.denoising_mean_mode = 'U'
    diffusion.timestep_sampler = timestep_sampler
    diffusion.flow_loss = flow_loss
    diffusion.train_cfg = dict(trans_ratio=0.5, eps=1e-4)
    diffusion.test_cfg = dict(
        sampler='FlowEulerODE', output_mode='mean',
        num_timesteps=16, num_substeps=4)
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


def load_checkpoint(path, model, ema=None, optimizer=None, scaler=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    if ema is not None and 'ema' in state:
        ema.load_state_dict(state['ema'])
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    if scaler is not None and 'scaler' in state:
        scaler.load_state_dict(state['scaler'])
    iteration = state.get('iteration', 0)
    print(f'Resumed from {path}, iteration {iteration}')
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
    age_real = [a * 80 + 6 for a in ages]

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
                num_timesteps=16,
                num_substeps=4))

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

    # Dataset
    dataset = OpenBHBSimple(
        cache_dir=args.cache_dir,
        metadata_path=args.metadata,
        random_flip=True)

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
    scaler = torch.amp.GradScaler('cuda') if (use_amp and autocast_dtype == torch.float16) else None

    # Resume
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, ema, optimizer, scaler)

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

    # Data
    parser.add_argument('--cache_dir', type=str, required=True,
                        help='Directory with preprocessed .pt volumes')
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to metadata.tsv')

    # Model architecture
    parser.add_argument('--num_gaussians', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--head_dim', type=int, default=64,
                        help='inner_dim = num_heads * head_dim (default: 768)')
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--volume_size', type=int, default=64)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--age_dropout_prob', type=float, default=0.1)
    parser.add_argument('--checkpointing', action='store_true', default=True,
                        help='Enable gradient checkpointing')
    parser.add_argument('--no_checkpointing', action='store_false', dest='checkpointing')

    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--grad_clip', type=float, default=10.0)
    parser.add_argument('--warmup_iters', type=int, default=1000)
    parser.add_argument('--total_iters', type=int, default=100000)
    parser.add_argument('--prob_age', type=float, default=0.9,
                        help='Probability of using real age (1-p = unconditional for CFG)')
    parser.add_argument('--autocast_dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16', None],
                        help='AMP dtype (None to disable)')
    parser.add_argument('--no_amp', action='store_const', const=None, dest='autocast_dtype')

    # EMA
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--no_ema', action='store_false', dest='use_ema')
    parser.add_argument('--ema_decay', type=float, default=0.9999)

    # I/O
    parser.add_argument('--work_dir', type=str, default='work_dirs/gmflow3d_standalone')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--sample_interval', type=int, default=5000)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--logger', type=str, nargs='+',
                        default=['tensorboard'],
                        choices=['tensorboard', 'wandb'],
                        help='Logging backends (default: tensorboard). '
                             'Use --logger tensorboard wandb for both.')
    parser.add_argument('--wandb_project', type=str, default='gmflow3d',
                        help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='W&B run name (auto-generated if not set)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B team/entity name')

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    mock_framework()
    main()
