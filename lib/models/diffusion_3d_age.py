"""Model wrapper for 3D brain MRI generation conditioned on age.

Follows the pattern of LatentDiffusionClassImage but:
- No VAE (operates directly in voxel space)
- Conditioning on continuous age instead of discrete class labels
- Saves output volumes as .npy instead of .png
"""

import os

import numpy as np
import torch

from copy import deepcopy
from mmgen.models.builder import MODELS, build_module
from mmgen.utils import get_root_logger

from .base import BaseModel
from ..core import link_untrained_params


@MODELS.register_module()
class Diffusion3DAge(BaseModel):

    def __init__(self,
                 diffusion=dict(type='GMFlow3D'),
                 diffusion_use_ema=False,
                 autocast_dtype=None,
                 pretrained=None,
                 inference_only=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        diffusion.update(train_cfg=train_cfg, test_cfg=test_cfg)
        self.diffusion = build_module(diffusion)
        self.diffusion_use_ema = diffusion_use_ema
        if self.diffusion_use_ema:
            if inference_only:
                self.diffusion_ema = self.diffusion
            else:
                self.diffusion_ema = build_module(diffusion)
                link_untrained_params(self.diffusion_ema, self.diffusion)

        self.autocast_dtype = autocast_dtype
        self.pretrained = pretrained

        if self.pretrained is not None:
            self.load_checkpoint(
                self.pretrained, map_location='cpu', strict=False,
                logger=get_root_logger())

        self.train_cfg = dict() if train_cfg is None else deepcopy(train_cfg)
        self.test_cfg = dict() if test_cfg is None else deepcopy(test_cfg)

    def train_step(self, data, optimizer, loss_scaler=None, running_status=None):
        bs = data['volumes'].size(0)
        age = data['age']

        # Age dropout for CFG: randomly replace age with negative_age
        prob_age = self.train_cfg.get('prob_age', 1.0)
        if prob_age < 1.0:
            age = torch.where(
                torch.rand_like(age) < prob_age,
                age, data['negative_age'])

        # Gradient accumulation: zero grads at the start of an accumulation
        # cycle, scale the loss by 1/accum, and only step the optimizer at the
        # end of the cycle. accum == 1 reproduces the original per-iter behavior.
        accum = self.train_cfg.get('gradient_accumulation_steps', 1)
        if accum > 1:
            cur_iter = running_status['iteration']
            is_cycle_start = cur_iter % accum == 0
            is_cycle_end = cur_iter % accum == accum - 1
        else:
            is_cycle_start = is_cycle_end = True

        if is_cycle_start:
            for v in optimizer.values():
                v.zero_grad()

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
            loss, log_vars = self.diffusion(
                data['volumes'],
                return_loss=True,
                age=age)

        loss_bwd = loss / accum
        loss_bwd.backward() if loss_scaler is None else loss_scaler.scale(loss_bwd).backward()

        if is_cycle_end:
            log_vars = self.step_optimizer(optimizer, loss_scaler, running_status, log_vars)
        log_vars = {k: float(v) for k, v in log_vars.items()}
        outputs_dict = dict(log_vars=log_vars, num_samples=bs)

        return outputs_dict

    def eval_and_viz(self, out_volumes, data, viz_dir=None, cfg=dict()):
        if viz_dir is None:
            viz_dir = cfg.get('viz_dir', None)

        if viz_dir is not None:
            os.makedirs(viz_dir, exist_ok=True)
            volumes_np = out_volumes.cpu().numpy()
            for i in range(volumes_np.shape[0]):
                name = f'{data["ids"][i]:09d}_age{data["age"][i].item():.2f}.npy'
                np.save(os.path.join(viz_dir, name), volumes_np[i])

        return dict()

    def val_step(self, data, viz_dir=None, test_cfg_override=dict(), **kwargs):
        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        volume_size = cfg.get('volume_size', (1, 64, 64, 64))
        guidance_scale = cfg.get('guidance_scale', 0.0)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        bs = len(data['age'])

        with torch.no_grad():
            age = data['age']

            if guidance_scale != 0.0:
                age = torch.cat([data['negative_age'], age], dim=0)

            with torch.autocast(
                    device_type='cuda',
                    enabled=self.autocast_dtype is not None,
                    dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
                if 'noise' in data:
                    noise = data['noise']
                else:
                    seed = cfg.get('seed', None)
                    if seed is not None:
                        # Deterministic sampling: with output_mode='mean' the only
                        # stochasticity is the initial noise, so a seeded generator
                        # makes generation fully reproducible.
                        generator = torch.Generator(
                            device=data['age'].device).manual_seed(int(seed))
                        noise = torch.randn(
                            (bs,) + tuple(volume_size),
                            device=data['age'].device, generator=generator)
                    else:
                        noise = torch.randn(
                            (bs,) + tuple(volume_size),
                            device=data['age'].device)
                volumes_out = diffusion(
                    noise=noise,
                    age=age,
                    guidance_scale=guidance_scale,
                    test_cfg_override=test_cfg_override)

            log_vars = self.eval_and_viz(
                volumes_out, data, viz_dir=viz_dir, cfg=cfg)

            return dict(log_vars=log_vars, num_samples=bs, pred_volumes=volumes_out)
