"""Generate 3D brain MRI volumes from a trained GMFlow3D checkpoint.

Usage:
    # Generate volumes for specific ages:
    python tools/inference.py \
        --checkpoint work_dirs/gmflow3d_openbhb/checkpoints/latest.pt \
        --ages 10 25 40 55 70 85 \
        --output_dir output/samples

    # Generate with CFG guidance:
    python tools/inference.py \
        --checkpoint work_dirs/gmflow3d_openbhb/checkpoints/latest.pt \
        --ages 20 50 80 \
        --guidance_scale 0.04 \
        --output_dir output/samples_cfg

    # Generate multiple samples per age (different seeds):
    python tools/inference.py \
        --checkpoint work_dirs/gmflow3d_openbhb/checkpoints/latest.pt \
        --ages 30 60 \
        --num_samples 5 \
        --output_dir output/multi_samples

    # Use EMA weights (if available in checkpoint):
    python tools/inference.py \
        --checkpoint work_dirs/gmflow3d_openbhb/checkpoints/latest.pt \
        --ages 25 50 75 \
        --use_ema \
        --output_dir output/samples_ema

    # Higher quality (more sampling steps):
    python tools/inference.py \
        --checkpoint work_dirs/gmflow3d_openbhb/checkpoints/latest.pt \
        --ages 25 50 75 \
        --num_timesteps 32 --num_substeps 8 \
        --output_dir output/samples_hq
"""

import os
import sys
import types
import argparse
import importlib.util
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn


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
# Build model
# ---------------------------------------------------------------------------

def build_model(model_args, device):
    """Build GMFlow3D model from architecture parameters."""
    from lib.models.architecture.gmflow3d import _GMDiTTransformer3DModel
    from lib.models.diffusions.gmflow3d import GMFlow3D
    from lib.models.diffusions.sampler import ContinuousTimeStepSampler
    from lib.models.losses.diffusion_loss import GMFlowNLLLoss3D

    denoising = _GMDiTTransformer3DModel(
        num_gaussians=model_args['num_gaussians'],
        num_attention_heads=model_args['num_heads'],
        attention_head_dim=model_args['head_dim'],
        in_channels=1,
        num_layers=model_args['num_layers'],
        sample_size=model_args['volume_size'],
        patch_size=model_args['patch_size'],
        age_dropout_prob=0.0)  # no dropout at inference
    denoising.init_weights()

    flow_loss = GMFlowNLLLoss3D(
        weight_scale=2.0,
        data_info=dict(
            pred_means='means', target='x_t_low',
            pred_logstds='logstds', pred_logweights='logweights'),
        log_cfgs=dict(type='quartile', prefix_name='loss_trans', total_timesteps=1000))

    timestep_sampler = ContinuousTimeStepSampler(
        num_timesteps=1000, shift=1.0, logit_normal_enable=True)

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
    return diffusion


def load_checkpoint(path, model, use_ema=False):
    """Load checkpoint, optionally applying EMA weights."""
    state = torch.load(path, map_location='cpu', weights_only=False)

    if use_ema and 'ema' in state:
        # Apply EMA weights directly to model parameters
        ema_state = state['ema']
        model_state = model.state_dict()
        for name in model_state:
            if name in ema_state:
                model_state[name].copy_(ema_state[name])
        model.load_state_dict(model_state)
        print(f'Loaded EMA weights from {path}')
    else:
        model.load_state_dict(state['model'])
        if use_ema and 'ema' not in state:
            print(f'Warning: --use_ema requested but no EMA weights in checkpoint, using model weights')
        print(f'Loaded model weights from {path}')

    iteration = state.get('iteration', 0)
    print(f'  Checkpoint iteration: {iteration}')

    # Return saved args for model architecture reconstruction
    return state.get('args', {})


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, device, ages_years, args):
    """Generate volumes for a list of ages (in years).

    Returns:
        list of (age_years, volume_numpy) tuples
    """
    age_min, age_max = 6.0, 86.0
    age_range = age_max - age_min
    vol_size = args.volume_size

    results = []

    for sample_idx in range(args.num_samples):
        seed = args.seed + sample_idx
        generator = torch.Generator(device=device).manual_seed(seed)

        for age_yr in ages_years:
            age_norm = (age_yr - age_min) / age_range
            age_norm = max(0.0, min(1.0, age_norm))

            noise = torch.randn(
                1, 1, vol_size, vol_size, vol_size,
                device=device, generator=generator)

            # Build age tensor
            if args.guidance_scale > 0:
                age_t = torch.tensor([-1.0, age_norm], device=device)
            else:
                age_t = torch.tensor([age_norm], device=device)

            t0 = time.time()

            with torch.autocast(
                    device_type='cuda',
                    enabled=args.autocast_dtype is not None and device.type == 'cuda',
                    dtype=getattr(torch, args.autocast_dtype) if args.autocast_dtype else None):
                vol = model(
                    noise=noise,
                    age=age_t,
                    guidance_scale=args.guidance_scale,
                    show_pbar=args.show_pbar,
                    test_cfg_override=dict(
                        sampler='FlowEulerODE',
                        output_mode=args.output_mode,
                        num_timesteps=args.num_timesteps,
                        num_substeps=args.num_substeps))

            elapsed = time.time() - t0
            vol_np = vol[0].cpu().float().numpy()  # (1, D, H, W)

            results.append(dict(
                age_years=age_yr,
                age_norm=age_norm,
                seed=seed,
                sample_idx=sample_idx,
                volume=vol_np,
                elapsed=elapsed))

            print(f'  age={age_yr:.1f}yr  seed={seed}  '
                  f'range=[{vol_np.min():.3f}, {vol_np.max():.3f}]  '
                  f'{elapsed:.1f}s')

    return results


def save_results(results, output_dir, save_format):
    """Save generated volumes."""
    os.makedirs(output_dir, exist_ok=True)

    for r in results:
        age_yr = r['age_years']
        seed = r['seed']
        vol = r['volume']

        base_name = f'age{age_yr:05.1f}_seed{seed}'

        if 'npy' in save_format:
            np.save(os.path.join(output_dir, f'{base_name}.npy'), vol)

        if 'nifti' in save_format:
            try:
                import nibabel as nib
                # vol shape: (1, D, H, W) -> (D, H, W) for NIfTI
                img = nib.Nifti1Image(vol[0], affine=np.eye(4))
                nib.save(img, os.path.join(output_dir, f'{base_name}.nii.gz'))
            except ImportError:
                print('Warning: nibabel not installed, skipping NIfTI output. '
                      'Install with: pip install nibabel')

        if 'png' in save_format:
            try:
                from PIL import Image
                # Save 3 orthogonal slices (axial, coronal, sagittal) at center
                v = vol[0]  # (D, H, W)
                d, h, w = v.shape
                slices = [
                    v[d // 2, :, :],   # axial
                    v[:, h // 2, :],   # coronal
                    v[:, :, w // 2],   # sagittal
                ]
                # Normalize to [0, 255]
                for i, s in enumerate(slices):
                    s = (s - s.min()) / (s.max() - s.min() + 1e-8) * 255
                    s = s.astype(np.uint8)
                    view = ['axial', 'coronal', 'sagittal'][i]
                    Image.fromarray(s).save(
                        os.path.join(output_dir, f'{base_name}_{view}.png'))
            except ImportError:
                print('Warning: Pillow not installed, skipping PNG output. '
                      'Install with: pip install Pillow')

    total = len(results)
    print(f'\nSaved {total} volumes to {output_dir}/')
    if 'npy' in save_format:
        print(f'  .npy files: shape=(1, {results[0]["volume"].shape[1]}, '
              f'{results[0]["volume"].shape[2]}, {results[0]["volume"].shape[3]}), float32')
    if 'nifti' in save_format:
        print(f'  .nii.gz files: NIfTI format, viewable in FreeSurfer/FSLeyes/3D Slicer')
    if 'png' in save_format:
        print(f'  .png files: center slices (axial, coronal, sagittal)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D brain MRI volumes from a trained GMFlow3D checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint (.pt)')
    parser.add_argument('--ages', type=float, nargs='+', required=True,
                        help='Ages in years to generate (e.g., 10 25 50 75)')

    # Model architecture (auto-detected from checkpoint if available)
    parser.add_argument('--num_gaussians', type=int, default=None)
    parser.add_argument('--num_heads', type=int, default=None)
    parser.add_argument('--head_dim', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--volume_size', type=int, default=None)
    parser.add_argument('--patch_size', type=int, default=None)

    # Generation
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples per age (different seeds)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (incremented per sample)')
    parser.add_argument('--guidance_scale', type=float, default=0.0,
                        help='CFG guidance scale (0.0=no guidance, 0.04=recommended)')
    parser.add_argument('--num_timesteps', type=int, default=16,
                        help='Number of sampling timesteps')
    parser.add_argument('--num_substeps', type=int, default=4,
                        help='ODE substeps per timestep')
    parser.add_argument('--output_mode', type=str, default='mean',
                        choices=['mean', 'sample'],
                        help='GM output mode: mean (deterministic) or sample (stochastic)')

    # Weights
    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='Use EMA weights from checkpoint')

    # Output
    parser.add_argument('--output_dir', type=str, default='output/samples',
                        help='Output directory')
    parser.add_argument('--save_format', type=str, nargs='+',
                        default=['npy', 'png'],
                        choices=['npy', 'nifti', 'png'],
                        help='Output formats (default: npy png)')

    # Performance
    parser.add_argument('--autocast_dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16'],
                        help='AMP dtype for inference')
    parser.add_argument('--no_amp', action='store_const', const=None,
                        dest='autocast_dtype')
    parser.add_argument('--show_pbar', action='store_true', default=False,
                        help='Show progress bar during sampling')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name()}')

    # Peek at checkpoint to get model architecture args
    print(f'\nLoading checkpoint: {args.checkpoint}')
    ckpt_state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_args = ckpt_state.get('args', {})

    # Model architecture: use CLI args if provided, else fall back to checkpoint args, else defaults
    DEFAULTS = dict(num_gaussians=4, num_heads=12, head_dim=64,
                    num_layers=12, volume_size=64, patch_size=4)
    model_args = {}
    for key in DEFAULTS:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            model_args[key] = cli_val
        elif key in ckpt_args:
            model_args[key] = ckpt_args[key]
        else:
            model_args[key] = DEFAULTS[key]

    print(f'Model config: {model_args}')

    # Also set on args for generate() to use volume_size
    args.volume_size = model_args['volume_size']

    # Build model
    model = build_model(model_args, device)
    del ckpt_state  # free memory before loading weights

    # Load weights
    load_checkpoint(args.checkpoint, model, use_ema=args.use_ema)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params:,}')

    # Generate
    print(f'\nGenerating {len(args.ages)} age(s) x {args.num_samples} sample(s) = '
          f'{len(args.ages) * args.num_samples} volume(s)')
    print(f'  Ages: {args.ages}')
    print(f'  Guidance: {args.guidance_scale}')
    print(f'  Steps: {args.num_timesteps} x {args.num_substeps} substeps')
    print(f'  Output mode: {args.output_mode}')
    print()

    results = generate(model, device, args.ages, args)

    # Save
    save_results(results, args.output_dir, args.save_format)


if __name__ == '__main__':
    mock_framework()
    main()
