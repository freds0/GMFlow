"""Smoke test for the audit fixes (run inside the gmflow env).

Validates the changed code paths with a tiny model:
  1. Gradient accumulation: optimizer steps only at cycle ends.
  2. Inference determinism: same seed -> identical volume.
  3. Output clamp: generated volume within [-1, 1].
  4. Resolution assert: PatchEmbed3D rejects sample_size %% patch_size != 0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import lib  # noqa: F401  registers modules
from mmgen.models import build_model

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={DEV}')

VOX = (1, 16, 16, 16)  # voxel volume -> wavelet 8x8^3 -> patch2 -> 4^3 tokens

train_cfg = dict(
    trans_ratio=0.5, randomize_trans_ratio=True, trans_ratio_min=0.05, trans_ratio_max=1.0,
    prob_age=0.9, diffusion_grad_clip=10.0, diffusion_grad_clip_begin_iter=0,
    gradient_accumulation_steps=4)
test_cfg = dict(volume_size=VOX, sampler='FlowEulerODE', output_mode='mean',
                num_timesteps=4, num_substeps=1, order=1, seed=123)

model_cfg = dict(
    type='Diffusion3DAge',
    diffusion=dict(
        type='GMFlow3D', use_wavelet=True, randomize_trans_ratio=True,
        denoising=dict(
            type='GMDiTTransformer3DModel', num_gaussians=2, gm_per_channel_logstd=True,
            logstd_inner_dim=64, gm_num_logstd_layers=2, age_dropout_prob=0.0,
            age_fourier_frequencies=4, num_attention_heads=2, attention_head_dim=16,
            in_channels=8, out_channels=8, num_layers=2, sample_size=8, patch_size=2,
            torch_dtype='float32', checkpointing=False),
        flow_loss=dict(
            type='GMFlowNLLLoss3D',
            data_info=dict(pred_means='means', target='x_t_low',
                           pred_logstds='logstds', pred_logweights='logweights'),
            band_weights=[1.0, 2.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0], weight_scale=2.0),
        num_timesteps=1000,
        timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=1.0, logit_normal_enable=True),
        denoising_mean_mode='U'),
    diffusion_use_ema=False, autocast_dtype=None)

model = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg).to(DEV)
model.train()
opt = {'diffusion': torch.optim.SGD(model.diffusion.parameters(), lr=0.1)}

# a reference trainable parameter to watch
ref = dict(model.diffusion.named_parameters())['denoising.pos_embed.proj.weight']

def batch():
    return dict(
        volumes=torch.randn((1,) + VOX, device=DEV),
        age=torch.rand(1, device=DEV),
        negative_age=torch.full((1,), -1.0, device=DEV))

print('\n[1] gradient accumulation (accum=4)')
prev = ref.detach().clone()
changed = []
for it in range(8):
    model.train_step(batch(), opt, running_status={'iteration': it})
    now = ref.detach().clone()
    changed.append(not torch.equal(prev, now))
    prev = now
# expect param to change only on cycle-end iters: indices 3 and 7
exp = [i in (3, 7) for i in range(8)]
print('  changed-per-iter:', changed)
print('  expected        :', exp)
assert changed == exp, 'grad accumulation did not step on the right iters'
print('  PASS')

print('\n[2] inference determinism (same seed) + [3] output clamp')
model.eval()
data = dict(age=torch.rand(2, device=DEV),
           negative_age=torch.full((2,), -1.0, device=DEV),
           ids=[0, 1])
out1 = model.val_step(data)['pred_volumes']
out2 = model.val_step(data)['pred_volumes']
print('  shape', tuple(out1.shape), 'range [%.3f, %.3f]' % (float(out1.min()), float(out1.max())))
assert torch.allclose(out1, out2), 'seeded val_step is not deterministic'
print('  determinism PASS')
assert float(out1.min()) >= -1.0 - 1e-5 and float(out1.max()) <= 1.0 + 1e-5, 'output not clamped to [-1,1]'
print('  clamp PASS')

print('\n[4] resolution assert')
from lib.models.architecture.gmflow3d import PatchEmbed3D
try:
    PatchEmbed3D(volume_size=8, patch_size=3)
    raise SystemExit('FAIL: assert did not fire')
except AssertionError:
    print('  PASS (rejected sample_size %% patch_size != 0)')

print('\nALL SMOKE CHECKS PASSED')
