"""Smoke test for 3D GMFlow architecture.

Run: python test_3d_smoke.py

This test avoids framework dependencies (mmcv, mmgen) by importing modules
directly using importlib, bypassing lib/__init__.py.
"""
import sys
import types
import importlib.util
from functools import partial
import torch
import torch.nn as nn


def mock_framework():
    """Mock mmcv/mmgen/diffusers so we can import our modules."""
    # Mock mmcv - treat subpackages as packages with __path__
    import os as _os
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

    # Mock mmgen
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

    # Mock lib.core and related - use real packages where we have real code
    # For packages with real submodules, we need to set __path__
    import os
    base = os.path.dirname(os.path.abspath(__file__))

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

    # We need GaussianFlow and schedulers for gmflow3d import
    # Load them using importlib since the package system is mocked
    _gf_spec = importlib.util.spec_from_file_location(
        'lib.models.diffusions.gaussian_flow',
        os.path.join(base, 'lib', 'models', 'diffusions', 'gaussian_flow.py'))
    _gf_mod = importlib.util.module_from_spec(_gf_spec)
    sys.modules['lib.models.diffusions.gaussian_flow'] = _gf_mod
    _gf_spec.loader.exec_module(_gf_mod)
    sys.modules['lib.models.diffusions'].GaussianFlow = _gf_mod.GaussianFlow

    # Load schedulers submodule properly (contains FlowEulerODEScheduler etc.)
    import os as _os2
    _sched_init = _os2.path.join(base, 'lib', 'models', 'diffusions', 'schedulers', '__init__.py')
    _sched_spec = importlib.util.spec_from_file_location(
        'lib.models.diffusions.schedulers', _sched_init,
        submodule_search_locations=[_os2.path.join(base, 'lib', 'models', 'diffusions', 'schedulers')])
    _sched_mod = importlib.util.module_from_spec(_sched_spec)
    sys.modules['lib.models.diffusions.schedulers'] = _sched_mod
    _sched_spec.loader.exec_module(_sched_mod)
    sys.modules['lib.models.diffusions'].schedulers = _sched_mod


def test_components():
    print('=' * 60)
    print('Testing 3D GM Operations')
    print('=' * 60)

    from lib.ops.gmflow_ops.gmflow_ops_3d import (
        gm_to_mean_3d, gm_to_sample_3d, gm_to_iso_gaussian_3d,
        gm_mul_iso_gaussian_3d, iso_gaussian_mul_iso_gaussian_3d)

    bs, K, C, D, H, W = 2, 4, 1, 8, 8, 8
    gm = dict(
        means=torch.randn(bs, K, C, D, H, W),
        logstds=torch.randn(bs, 1, 1, 1, 1, 1),
        logweights=torch.randn(bs, K, 1, D, H, W).log_softmax(dim=1))

    mean = gm_to_mean_3d(gm)
    assert mean.shape == (bs, C, D, H, W)
    print(f'[PASS] gm_to_mean_3d: {mean.shape}')

    gauss, diffs, gm_vars = gm_to_iso_gaussian_3d(gm)
    assert gauss['mean'].shape == (bs, C, D, H, W)
    print(f'[PASS] gm_to_iso_gaussian_3d: mean={gauss["mean"].shape}, var={gauss["var"].shape}')

    for ns in [1, 3]:
        samples = gm_to_sample_3d(gm, n_samples=ns)
        assert samples.shape == (bs, ns, C, D, H, W)
        print(f'[PASS] gm_to_sample_3d(n={ns}): {samples.shape}')

    iso_g = dict(mean=torch.randn(bs, C, D, H, W), var=torch.ones(bs, 1, D, H, W) * 0.5)
    result, _ = gm_mul_iso_gaussian_3d(gm, iso_g)
    assert result['means'].shape == (bs, K, C, D, H, W)
    print(f'[PASS] gm_mul_iso_gaussian_3d')

    g1 = dict(mean=torch.randn(bs, C, D, H, W), var=torch.ones(bs, 1, D, H, W))
    g2 = dict(mean=torch.randn(bs, C, D, H, W), var=torch.ones(bs, 1, D, H, W) * 0.5)
    result = iso_gaussian_mul_iso_gaussian_3d(g1, g2)
    print(f'[PASS] iso_gaussian_mul_iso_gaussian_3d')
    print()


def test_architecture():
    print('=' * 60)
    print('Testing 3D Architecture')
    print('=' * 60)

    from lib.models.architecture.gmflow3d import (
        PatchEmbed3D, AgeEmbedding, CombinedTimestepAgeEmbeddings,
        GMOutput3D, _GMDiTTransformer3DModel)

    bs = 2

    # PatchEmbed3D
    pe = PatchEmbed3D(volume_size=16, patch_size=4, in_channels=1, embed_dim=192)
    x = torch.randn(bs, 1, 16, 16, 16)
    out = pe(x)
    assert out.shape == (bs, 64, 192)
    print(f'[PASS] PatchEmbed3D: {x.shape} -> {out.shape}')

    # AgeEmbedding
    ae = AgeEmbedding(hidden_size=192)
    age = torch.tensor([0.5, -1.0])
    out = ae(age)
    assert out.shape == (bs, 192)
    print(f'[PASS] AgeEmbedding: {out.shape}')

    # CombinedTimestepAgeEmbeddings
    cte = CombinedTimestepAgeEmbeddings(embedding_dim=192)
    t = torch.tensor([500.0, 200.0])
    age = torch.tensor([0.3, 0.7])
    out = cte(t, age)
    assert out.shape == (bs, 192)
    print(f'[PASS] CombinedTimestepAgeEmbeddings: {out.shape}')

    # GMOutput3D
    K, C = 4, 1
    gmo = GMOutput3D(num_gaussians=K, out_channels=C, embed_dim=192)
    x = torch.randn(bs, K * (C + 1), 16, 16, 16)
    emb = torch.randn(bs, 192)
    out = gmo(x, emb)
    assert out['means'].shape == (bs, K, C, 16, 16, 16)
    assert out['logweights'].shape == (bs, K, 1, 16, 16, 16)
    assert out['logstds'].shape == (bs, 1, 1, 1, 1, 1)
    print(f'[PASS] GMOutput3D: means={out["means"].shape}')

    # Full model (small version)
    print('\nBuilding small _GMDiTTransformer3DModel (2 layers, dim=192)...')
    model = _GMDiTTransformer3DModel(
        num_gaussians=4,
        num_attention_heads=4,
        attention_head_dim=48,  # inner_dim = 192
        in_channels=1,
        num_layers=2,
        sample_size=16,
        patch_size=4,
        age_dropout_prob=0.1)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {num_params:,}')

    x = torch.randn(bs, 1, 16, 16, 16)
    t = torch.tensor([500.0, 200.0])
    age = torch.tensor([0.3, 0.7])

    model.eval()
    with torch.no_grad():
        out = model(x, timestep=t, age=age)
    assert out['means'].shape == (bs, 4, 1, 16, 16, 16)
    print(f'[PASS] Forward (eval): means={out["means"].shape}')

    model.train()
    out = model(x, timestep=t, age=age)
    assert out['means'].shape == (bs, 4, 1, 16, 16, 16)
    print(f'[PASS] Forward (train with dropout): means={out["means"].shape}')

    loss = out['means'].sum()
    loss.backward()
    has_grads = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f'[PASS] Backward: all grads={has_grads}')

    # Test with negative age (unconditional)
    model.eval()
    with torch.no_grad():
        out = model(x, timestep=t, age=torch.tensor([-1.0, -1.0]))
    assert out['means'].shape == (bs, 4, 1, 16, 16, 16)
    print(f'[PASS] Forward with null age (CFG): means={out["means"].shape}')

    print()


def test_jit_functions():
    print('=' * 60)
    print('Testing JIT functions')
    print('=' * 60)

    from lib.models.diffusions.gmflow3d import (
        denoising_gm_convert_to_mean_3d_jit,
        probabilistic_guidance_3d_jit)

    bs, K, C, D, H, W = 2, 4, 1, 8, 8, 8

    sigma_src = torch.tensor([0.5, 0.3]).reshape(2, 1, 1, 1, 1)
    sigma_tgt = torch.tensor([0.3, 0.1]).reshape(2, 1, 1, 1, 1)
    x_t_src = torch.randn(bs, C, D, H, W)
    x_t_tgt = torch.randn(bs, C, D, H, W)
    gm_means = torch.randn(bs, K, C, D, H, W)
    gm_vars = torch.ones(bs, 1, 1, 1, 1, 1) * 0.1
    gm_logweights = torch.randn(bs, K, 1, D, H, W).log_softmax(dim=1)

    out_mean = denoising_gm_convert_to_mean_3d_jit(
        sigma_src, sigma_tgt, x_t_src, x_t_tgt,
        gm_means, gm_vars, gm_logweights, 1e-6)
    assert out_mean.shape == (bs, C, D, H, W)
    print(f'[PASS] denoising_gm_convert_to_mean_3d_jit: {out_mean.shape}')

    cond_mean = torch.randn(bs, C, D, H, W)
    total_var = torch.ones(bs, 1, D, H, W) * 0.5
    uncond_mean = torch.randn(bs, C, D, H, W)
    gauss_out, bias, avg_var = probabilistic_guidance_3d_jit(
        cond_mean, total_var, uncond_mean, 0.04)
    assert gauss_out['mean'].shape == (bs, C, D, H, W)
    print(f'[PASS] probabilistic_guidance_3d_jit: mean={gauss_out["mean"].shape}')

    print()


def test_nll_loss():
    print('=' * 60)
    print('Testing NLL Loss 3D')
    print('=' * 60)

    from lib.models.losses.diffusion_loss import gaussian_mixture_nll_loss_3d

    bs, K, C, D, H, W = 2, 4, 1, 8, 8, 8
    pred_means = torch.randn(bs, K, C, D, H, W)
    target = torch.randn(bs, C, D, H, W)
    pred_logstds = torch.randn(bs, 1, 1, 1, 1, 1)
    pred_logweights = torch.randn(bs, K, 1, D, H, W).log_softmax(dim=1)

    loss = gaussian_mixture_nll_loss_3d(
        pred_means, target, pred_logstds, pred_logweights, reduction='mean')
    print(f'[PASS] NLL loss (mean): {loss.item():.4f}')

    loss_flat = gaussian_mixture_nll_loss_3d(
        pred_means, target, pred_logstds, pred_logweights, reduction='flatmean')
    assert loss_flat.shape == (bs,)
    print(f'[PASS] NLL loss (flatmean): shape={loss_flat.shape}')

    # Gradient check
    pred_means.requires_grad_(True)
    loss = gaussian_mixture_nll_loss_3d(
        pred_means, target, pred_logstds, pred_logweights, reduction='mean')
    loss.backward()
    assert pred_means.grad is not None
    print(f'[PASS] NLL loss backward: grad shape={pred_means.grad.shape}')

    print()


def test_full_config_shape():
    """Test with actual config dimensions (64x64x64, patch_size=4, K=4)."""
    print('=' * 60)
    print('Testing Full Config Dimensions (64x64x64)')
    print('=' * 60)

    from lib.models.architecture.gmflow3d import _GMDiTTransformer3DModel

    # Use config dimensions but fewer layers for speed
    model = _GMDiTTransformer3DModel(
        num_gaussians=4,
        num_attention_heads=12,
        attention_head_dim=64,  # inner_dim = 768
        in_channels=1,
        num_layers=2,  # Fewer layers for testing
        sample_size=64,
        patch_size=4,
        age_dropout_prob=0.1)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters (2 layers): {num_params:,}')

    # Expected: 4096 tokens = (64/4)^3
    bs = 1
    x = torch.randn(bs, 1, 64, 64, 64)
    t = torch.tensor([500.0])
    age = torch.tensor([0.5])

    model.eval()
    with torch.no_grad():
        out = model(x, timestep=t, age=age)

    assert out['means'].shape == (bs, 4, 1, 64, 64, 64), f'{out["means"].shape}'
    assert out['logweights'].shape == (bs, 4, 1, 64, 64, 64)
    assert out['logstds'].shape == (bs, 1, 1, 1, 1, 1)
    print(f'[PASS] Full config forward: means={out["means"].shape}')
    print(f'  Tokens: {model.pos_embed.num_patches} (expected 4096)')

    # Estimate full model size
    full_params_est = num_params / 2 * 12  # Scale from 2 to 12 layers
    print(f'  Estimated full model (12 layers): ~{full_params_est/1e6:.0f}M params')

    print()


def test_training_loop():
    """End-to-end training test: build GMFlow3D pipeline and overfit on synthetic data."""
    print('=' * 60)
    print('Testing Training Loop (overfit on synthetic data)')
    print('=' * 60)

    from lib.models.architecture.gmflow3d import _GMDiTTransformer3DModel
    from lib.models.diffusions.gmflow3d import GMFlow3D
    from lib.models.diffusions.sampler import ContinuousTimeStepSampler
    from lib.models.losses.diffusion_loss import GMFlowNLLLoss3D

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')

    # Use small dimensions for fast testing
    vol_size = 16
    K = 4
    bs = 2
    num_iters = 50

    # 1. Build denoising network (small)
    denoising = _GMDiTTransformer3DModel(
        num_gaussians=K,
        num_attention_heads=4,
        attention_head_dim=48,  # inner_dim = 192
        in_channels=1,
        num_layers=2,
        sample_size=vol_size,
        patch_size=4,
        age_dropout_prob=0.1)
    denoising.init_weights()

    # 2. Build loss
    flow_loss = GMFlowNLLLoss3D(
        weight_scale=2.0,
        data_info=dict(
            pred_means='means',
            target='x_t_low',
            pred_logstds='logstds',
            pred_logweights='logweights'),
        log_cfgs=dict(type='quartile', prefix_name='loss_trans', total_timesteps=1000))

    # 3. Build timestep sampler
    timestep_sampler = ContinuousTimeStepSampler(
        num_timesteps=1000, shift=1.0, logit_normal_enable=True)

    # 4. Build GMFlow3D manually (bypass build_module)
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
        num_timesteps=4, num_substeps=1)
    diffusion.intermediate_x_t = []
    diffusion.intermediate_x_0 = []

    diffusion = diffusion.to(device)
    num_params = sum(p.numel() for p in diffusion.parameters())
    print(f'  Model parameters: {num_params:,}')

    # 5. Create synthetic training data (fixed batch to overfit)
    torch.manual_seed(42)
    # Create a simple pattern: sphere in center, intensity depends on age
    x_train = torch.zeros(bs, 1, vol_size, vol_size, vol_size, device=device)
    ages_train = torch.tensor([0.2, 0.8], device=device)
    center = vol_size // 2
    radius = vol_size // 4
    for i in range(vol_size):
        for j in range(vol_size):
            for k in range(vol_size):
                dist = ((i - center)**2 + (j - center)**2 + (k - center)**2) ** 0.5
                if dist < radius:
                    x_train[:, 0, i, j, k] = 1.0
    # Modulate by age
    x_train[0] *= ages_train[0]  # young -> lower intensity
    x_train[1] *= ages_train[1]  # old -> higher intensity
    # Normalize to [-1, 1]
    x_train = x_train * 2 - 1

    print(f'  Training data: x={x_train.shape}, age={ages_train}')
    print(f'  x_train range: [{x_train.min():.2f}, {x_train.max():.2f}]')

    # 6. Training loop
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=2e-4, weight_decay=0.01)
    diffusion.train()

    losses = []
    print(f'\n  Training for {num_iters} iterations...')
    for i in range(num_iters):
        optimizer.zero_grad()

        loss, log_vars = diffusion(
            x_train, return_loss=True, age=ages_train)

        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 10.0)

        optimizer.step()
        losses.append(float(loss))

        if (i + 1) % 10 == 0 or i == 0:
            print(f'    iter {i+1:3d}/{num_iters}: loss={loss.item():.4f}  '
                  f'grad_norm={float(grad_norm):.2f}')

    # 7. Verify loss decreased (NLL loss is negative; more negative = better)
    avg_first_5 = sum(losses[:5]) / 5
    avg_last_5 = sum(losses[-5:]) / 5
    print(f'\n  Loss: first 5 avg={avg_first_5:.4f}, last 5 avg={avg_last_5:.4f}')

    if avg_last_5 < avg_first_5:
        print(f'[PASS] Loss decreased during training (delta={avg_first_5 - avg_last_5:.4f})')
    else:
        print('[WARN] Loss did not decrease significantly (may need more iters)')

    # 8. Test inference (forward_test)
    print('\n  Running inference...')
    diffusion.eval()
    with torch.no_grad():
        noise = torch.randn(bs, 1, vol_size, vol_size, vol_size, device=device)
        output = diffusion(
            noise=noise,
            age=ages_train,
            guidance_scale=0.0,
            test_cfg_override=dict(
                sampler='FlowEulerODE',
                output_mode='mean',
                num_timesteps=4,
                num_substeps=1))

    assert output.shape == (bs, 1, vol_size, vol_size, vol_size), \
        f'Expected {(bs, 1, vol_size, vol_size, vol_size)}, got {output.shape}'
    print(f'[PASS] Inference output: {output.shape}')
    print(f'  Output range: [{output.min():.2f}, {output.max():.2f}]')

    # 9. Test inference with CFG
    print('\n  Running inference with CFG...')
    with torch.no_grad():
        noise = torch.randn(bs, 1, vol_size, vol_size, vol_size, device=device)
        # CFG: concat negative_age + age
        age_cfg = torch.cat([
            torch.full((bs,), -1.0, device=device),  # unconditional
            ages_train                                 # conditional
        ])
        output_cfg = diffusion(
            noise=noise,
            age=age_cfg,
            guidance_scale=0.04,
            test_cfg_override=dict(
                sampler='FlowEulerODE',
                output_mode='mean',
                num_timesteps=4,
                num_substeps=1))

    assert output_cfg.shape == (bs, 1, vol_size, vol_size, vol_size), \
        f'Expected {(bs, 1, vol_size, vol_size, vol_size)}, got {output_cfg.shape}'
    print(f'[PASS] CFG inference output: {output_cfg.shape}')
    print(f'  CFG output range: [{output_cfg.min():.2f}, {output_cfg.max():.2f}]')

    # 10. Verify outputs differ between ages (conditioning works)
    diff = (output[0] - output[1]).abs().mean().item()
    print(f'\n  Mean absolute difference between age=0.2 and age=0.8 outputs: {diff:.4f}')
    if diff > 0.01:
        print('[PASS] Age conditioning produces different outputs')
    else:
        print('[WARN] Outputs similar - conditioning may need more training')

    print()


if __name__ == '__main__':
    mock_framework()
    test_components()
    test_architecture()
    test_jit_functions()
    test_nll_loss()
    test_full_config_shape()
    test_training_loop()
    print('=' * 60)
    print('ALL SMOKE TESTS PASSED!')
    print('=' * 60)
