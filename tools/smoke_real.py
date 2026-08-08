"""Real-config smoke: build the actual model + dataset, run train_step.

Validates the cache-path + normalization fixes (loads real cached volumes) and
the gradient-accumulation path on the real 135M model; reports peak GPU memory.
Run inside the gmflow env.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, mmcv, lib  # noqa
from mmcv import Config
from mmgen.models import build_model

cfg = Config.fromfile('configs/gmflow3d_openbhb_k4.py')

# Force single-sample to fit the 12GB RTX 3060.
cfg.data.train.setdefault('use_cache', True)
ds = mmcv.build_from_cfg(cfg.data.train, __import__('mmgen').datasets.builder.DATASETS)
print('dataset size:', len(ds))
s0 = ds[0]
print('sample volume', tuple(s0['volumes'].shape),
      'range [%.3f, %.3f]' % (float(s0['volumes'].min()), float(s0['volumes'].max())),
      '| source endswith .pt:', ds.subjects[0]['path'].endswith('.pt'))

model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()
model.train()
opt = {'diffusion': torch.optim.AdamW(model.diffusion.parameters(), lr=1e-4)}

def collate(idxs):
    vols = torch.stack([ds[i]['volumes'] for i in idxs]).cuda()
    ages = torch.stack([ds[i]['age'] for i in idxs]).cuda()
    neg = torch.stack([ds[i]['negative_age'] for i in idxs]).cuda()
    return dict(volumes=vols, age=ages, negative_age=neg)

torch.cuda.reset_peak_memory_stats()
accum = cfg.train_cfg['gradient_accumulation_steps']
print(f'\nrunning {accum} micro-iters (1 full accumulation cycle), samples_per_gpu=1')
for it in range(accum):
    out = model.train_step(collate([it % len(ds)]), opt, running_status={'iteration': it})
    print(f'  iter {it}: loss={out["log_vars"].get("loss_transition", float("nan")):.4f}')
peak = torch.cuda.max_memory_allocated() / 1e9
print(f'\npeak GPU mem (bs=1, 64^3, checkpointing): {peak:.2f} GB')
print('REAL-CONFIG SMOKE OK')
