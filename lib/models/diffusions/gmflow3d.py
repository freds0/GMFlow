"""GMFlow adapted for 3D volumetric data.

Key differences from 2D GMFlow:
- Operates on 5D tensors (bs, C, D, H, W) instead of 4D (bs, C, H, W)
- GM gaussian dim is -5 instead of -4 for 3D
- Noise shape uses x_0.shape[-4:] to include channel dim
- No spectrum_net (spectral loss uses FFT2D, complex to adapt to 3D)
- probabilistic_guidance uses .mean(dim=(-4, -3, -2, -1)) for 3D+channel
"""

import sys
import inspect
import torch
import diffusers
import mmcv

from typing import Optional
from copy import deepcopy
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES, build_module
from mmgen.models.diffusions.utils import _get_noise_batch

from . import GaussianFlow, schedulers
from lib.ops.gmflow_ops.gmflow_ops_3d import (
    gm_to_mean_3d, gm_to_sample_3d, gm_to_iso_gaussian_3d,
    gm_mul_iso_gaussian_3d, iso_gaussian_mul_iso_gaussian_3d)


@torch.jit.script
def probabilistic_guidance_3d_jit(
        cond_mean, total_var, uncond_mean, guidance_scale: float,
        orthogonal: float = 1.0, orthogonal_axis: Optional[torch.Tensor] = None):
    bias = cond_mean - uncond_mean
    if orthogonal > 0.0:
        if orthogonal_axis is None:
            orthogonal_axis = cond_mean
        # 3D: mean over (C, D, H, W) = last 4 dims
        bias = bias - ((bias * orthogonal_axis).mean(
            dim=(-4, -3, -2, -1), keepdim=True
        ) / (orthogonal_axis * orthogonal_axis).mean(
            dim=(-4, -3, -2, -1), keepdim=True
        ).clamp(min=1e-6) * orthogonal_axis).mul(orthogonal)
    bias_power = (bias * bias).mean(dim=(-4, -3, -2, -1), keepdim=True)
    avg_var = total_var.mean(dim=(-4, -3, -2, -1), keepdim=True)
    bias = bias * ((avg_var / bias_power.clamp(min=1e-6)).sqrt() * guidance_scale)
    gaussian_output = dict(
        mean=cond_mean + bias,
        var=total_var * (1 - (guidance_scale * guidance_scale)))
    return gaussian_output, bias, avg_var


@torch.jit.script
def denoising_gm_convert_to_mean_3d_jit(
        sigma_src, sigma_tgt, x_t_src, x_t_tgt,
        gm_means, gm_vars, gm_logweights,
        eps: float):
    """Fused denoising + GM-to-mean for 3D. Gaussian dim at -5."""
    alpha_src = 1 - sigma_src
    alpha_tgt = 1 - sigma_tgt

    alpha_tgt_sigma_src = alpha_tgt * sigma_src
    alpha_src_sigma_tgt = alpha_src * sigma_tgt
    denom = (alpha_tgt_sigma_src.square() - alpha_src_sigma_tgt.square()).clamp(min=eps)
    g_mean = (alpha_tgt_sigma_src * sigma_src * x_t_tgt - alpha_src_sigma_tgt * sigma_tgt * x_t_src) / denom
    g_var = (sigma_tgt * sigma_src).square() / denom

    # 3D: gaussian dim is -5
    g_mean = g_mean.unsqueeze(-5)  # (bs, *, 1, out_channels, d, h, w)
    g_var = g_var.unsqueeze(-5)    # (bs, *, 1, 1, 1, 1, 1)

    gm_diffs = gm_means - g_mean  # (bs, *, num_gaussians, out_channels, d, h, w)
    norm_factor = (g_var + gm_vars).clamp(min=eps)

    out_means = (g_var * gm_means + gm_vars * g_mean) / norm_factor
    # 3D: sum over channels at dim=-4
    logweights_delta = gm_diffs.square().sum(dim=-4, keepdim=True) * (-0.5 / norm_factor)
    out_weights = (gm_logweights + logweights_delta).softmax(dim=-5)

    out_mean = (out_means * out_weights).sum(dim=-5)

    return out_mean


class GMFlow3DMixin:
    """3D-specific GM flow operations."""

    def sample_forward_transition_3d(self, x_t_low, t_low, t_high, noise):
        bs = x_t_low.size(0)
        if t_low.dim() == 0:
            t_low = t_low.expand(bs)
        if t_high.dim() == 0:
            t_high = t_high.expand(bs)

        std_low = t_low.reshape(*t_low.size(), *((x_t_low.dim() - t_low.dim()) * [1])) / self.num_timesteps
        mean_low = 1 - std_low
        std_high = t_high.reshape(*t_high.size(), *((x_t_low.dim() - t_high.dim()) * [1])) / self.num_timesteps
        mean_high = 1 - std_high

        mean_trans = mean_high / mean_low
        std_trans = (std_high ** 2 - (mean_trans * std_low) ** 2).sqrt()
        return x_t_low * mean_trans + noise * std_trans

    def u_to_x_0_3d(self, denoising_output, x_t, t=None, sigma=None, eps=1e-6):
        if isinstance(denoising_output, dict) and 'logweights' in denoising_output:
            x_t = x_t.unsqueeze(-5)

        if sigma is None:
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=x_t.device)
            t = t.reshape(*t.size(), *((x_t.dim() - t.dim()) * [1]))
            sigma = t / self.num_timesteps

        if isinstance(denoising_output, dict):
            if 'logweights' in denoising_output:
                means_x_0 = x_t - sigma * denoising_output['means']
                logstds_x_0 = denoising_output['logstds'] + torch.log(sigma.clamp(min=eps))
                return dict(
                    means=means_x_0,
                    logstds=logstds_x_0,
                    logweights=denoising_output['logweights'])
            elif 'var' in denoising_output:
                mean = x_t - sigma * denoising_output['mean']
                var = denoising_output['var'] * sigma.square()
                return dict(mean=mean, var=var)
            else:
                raise ValueError('Invalid denoising_output.')
        else:
            x_0 = x_t - sigma * denoising_output
            return x_0

    def denoising_gm_convert_to_mean_3d(
            self, gm_src, x_t_tgt, x_t_src, t_tgt=None, t_src=None,
            sigma_src=None, sigma_tgt=None, eps=1e-6, prediction_type='x0'):
        assert isinstance(gm_src, dict)

        if sigma_src is None:
            if not isinstance(t_src, torch.Tensor):
                t_src = torch.tensor(t_src, device=x_t_src.device)
            t_src = t_src.reshape(*t_src.size(), *((x_t_src.dim() - t_src.dim()) * [1]))
            sigma_src = t_src / self.num_timesteps

        if sigma_tgt is None:
            if not isinstance(t_tgt, torch.Tensor):
                t_tgt = torch.tensor(t_tgt, device=x_t_src.device)
            t_tgt = t_tgt.reshape(*t_tgt.size(), *((x_t_src.dim() - t_tgt.dim()) * [1]))
            sigma_tgt = t_tgt / self.num_timesteps

        if prediction_type == 'u':
            gm_src = self.u_to_x_0_3d(gm_src, x_t_src, sigma=sigma_src)
        else:
            assert prediction_type == 'x0'

        gm_means = gm_src['means']
        gm_logweights = gm_src['logweights']
        if 'gm_vars' in gm_src:
            gm_vars = gm_src['gm_vars']
        else:
            gm_vars = (gm_src['logstds'] * 2).exp()
            gm_src['gm_vars'] = gm_vars

        return denoising_gm_convert_to_mean_3d_jit(
            sigma_src, sigma_tgt, x_t_src, x_t_tgt,
            gm_means, gm_vars, gm_logweights, eps)

    def reverse_transition_3d(self, denoising_output, x_t_high, t_low, t_high, eps=1e-6, prediction_type='u'):
        if isinstance(denoising_output, dict):
            x_t_high = x_t_high.unsqueeze(-5)

        bs = x_t_high.size(0)
        if not isinstance(t_low, torch.Tensor):
            t_low = torch.tensor(t_low, device=x_t_high.device)
        if not isinstance(t_high, torch.Tensor):
            t_high = torch.tensor(t_high, device=x_t_high.device)
        if t_low.dim() == 0:
            t_low = t_low.expand(bs)
        if t_high.dim() == 0:
            t_high = t_high.expand(bs)
        t_low = t_low.reshape(*t_low.size(), *((x_t_high.dim() - t_low.dim()) * [1]))
        t_high = t_high.reshape(*t_high.size(), *((x_t_high.dim() - t_high.dim()) * [1]))

        sigma = t_high / self.num_timesteps
        sigma_to = t_low / self.num_timesteps
        alpha = 1 - sigma
        alpha_to = 1 - sigma_to

        sigma_to_over_sigma = sigma_to / sigma.clamp(min=eps)
        alpha_over_alpha_to = alpha / alpha_to.clamp(min=eps)
        beta_over_sigma_sq = 1 - (sigma_to_over_sigma * alpha_over_alpha_to) ** 2

        c1 = sigma_to_over_sigma ** 2 * alpha_over_alpha_to
        c2 = beta_over_sigma_sq * alpha_to

        if isinstance(denoising_output, dict):
            c3 = beta_over_sigma_sq * sigma_to ** 2
            if prediction_type == 'u':
                means_x_0 = x_t_high - sigma * denoising_output['means']
                logstds_x_t_low = torch.logaddexp(
                    (denoising_output['logstds'] + torch.log((sigma * c2).clamp(min=eps))) * 2,
                    torch.log(c3.clamp(min=eps))
                ) / 2
            elif prediction_type == 'x0':
                means_x_0 = denoising_output['means']
                logstds_x_t_low = torch.logaddexp(
                    (denoising_output['logstds'] + torch.log(c2.clamp(min=eps))) * 2,
                    torch.log(c3.clamp(min=eps))
                ) / 2
            else:
                raise ValueError('Invalid prediction_type.')
            means_x_t_low = c1 * x_t_high + c2 * means_x_0
            return dict(
                means=means_x_t_low,
                logstds=logstds_x_t_low,
                logweights=denoising_output['logweights'])

        else:
            c3_sqrt = beta_over_sigma_sq ** 0.5 * sigma_to
            noise = torch.randn_like(denoising_output)
            if prediction_type == 'u':
                x_0 = x_t_high - sigma * denoising_output
            elif prediction_type == 'x0':
                x_0 = denoising_output
            else:
                raise ValueError('Invalid prediction_type.')
            x_t_low = c1 * x_t_high + c2 * x_0 + c3_sqrt * noise
            return x_t_low

    @staticmethod
    def gm_sample_3d(gm, n_samples=1, generator=None):
        samples = gm_to_sample_3d(gm, n_samples=n_samples)
        return samples, None

    def gm_to_model_output_3d(self, gm, output_mode, generator=None):
        assert output_mode in ['mean', 'sample']
        if output_mode == 'mean':
            output = gm_to_mean_3d(gm)
        else:
            output = self.gm_sample_3d(gm, generator=generator)[0].squeeze(-5)
        return output

    def init_gm_cache_3d(self):
        self.prev_gm = None
        self.prev_x_t = None
        self.prev_t = None
        self.prev_h = None


@MODULES.register_module()
class GMFlow3D(GaussianFlow, GMFlow3DMixin):
    """3D GMFlow for volumetric data. No spectrum_net."""

    def __init__(self, *args, **kwargs):
        # Remove spectrum_net if passed (not supported in 3D)
        kwargs.pop('spectrum_net', None)
        super().__init__(*args, **kwargs)
        self.intermediate_x_t = []
        self.intermediate_x_0 = []

    def pred(self, x_t, t, **kwargs):
        """Override pred to pass age instead of class_labels."""
        ori_dtype = x_t.dtype
        if hasattr(self.denoising, 'dtype'):
            denoising_dtype = self.denoising.dtype
        else:
            denoising_dtype = next(self.denoising.parameters()).dtype
        x_t = x_t.to(denoising_dtype)
        num_batches = x_t.size(0)
        if t.dim() == 0 or len(t) != num_batches:
            t = t.expand(num_batches)
        output = self.denoising(x_t, t, **kwargs)
        if isinstance(output, dict):
            output = {k: v.to(ori_dtype) for k, v in output.items()}
        else:
            output = output.to(ori_dtype)
        return output

    def loss(self, denoising_output, x_t_low, x_t_high, t_low, t_high):
        """GMFlow 3D transition loss."""
        with torch.autocast(device_type='cuda', enabled=False):
            x_t_low = x_t_low.float()
            x_t_high = x_t_high.float()
            t_low = t_low.float()
            t_high = t_high.float()

            x_t_low_gm = self.reverse_transition_3d(denoising_output, x_t_high, t_low, t_high)
            loss_kwargs = {k: v for k, v in x_t_low_gm.items()}
            loss_kwargs.update(x_t_low=x_t_low, timesteps=t_high)
            return self.flow_loss(loss_kwargs)

    def forward_train(self, x_0, *args, **kwargs):
        device = get_module_device(self)

        assert x_0.dim() == 5, f'Expected 5D input (bs, C, D, H, W), got {x_0.dim()}D'
        num_batches = x_0.size(0)
        trans_ratio = self.train_cfg.get('trans_ratio', 1.0)
        eps = self.train_cfg.get('eps', 1e-4)

        with torch.autocast(device_type='cuda', enabled=False):
            t_high = self.timestep_sampler(num_batches).to(device).clamp(min=eps, max=self.num_timesteps)
            t_low = t_high * (1 - trans_ratio)
            t_low = torch.minimum(t_low, t_high - eps).clamp(min=0)

            # 3D: noise shape is (C, D, H, W) = x_0.shape[-4:]
            noise = torch.randn(
                (2 * num_batches,) + x_0.shape[1:],
                device=x_0.device, dtype=x_0.dtype)
            noise_0, noise_1 = torch.chunk(noise, 2, dim=0)

            x_t_low, _, _ = self.sample_forward_diffusion(x_0, t_low, noise_0)
            x_t_high = self.sample_forward_transition_3d(x_t_low, t_low, t_high, noise_1)

        denoising_output = self.pred(x_t_high, t_high, **kwargs)
        loss = self.loss(denoising_output, x_t_low, x_t_high, t_low, t_high)
        log_vars = self.flow_loss.log_vars
        log_vars.update(loss_transition=float(loss.detach()))

        return loss, log_vars

    def forward_test(
            self, x_0=None, noise=None, guidance_scale=0.0,
            test_cfg_override=dict(), show_pbar=False, **kwargs):
        x_t = torch.randn_like(x_0) if noise is None else noise
        num_batches = x_t.size(0)
        ori_dtype = x_t.dtype
        x_t = x_t.float()

        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)

        output_mode = cfg.get('output_mode', 'mean')
        assert output_mode in ['mean', 'sample']

        sampler = cfg['sampler']
        sampler_class = getattr(diffusers.schedulers, sampler + 'Scheduler', None)
        if sampler_class is None:
            sampler_class = getattr(schedulers, sampler + 'Scheduler', None)
        if sampler_class is None:
            raise AttributeError(f'Cannot find sampler [{sampler}].')

        sampler_kwargs = cfg.get('sampler_kwargs', {})
        signatures = inspect.signature(sampler_class).parameters.keys()
        if 'shift' in signatures and 'shift' not in sampler_kwargs:
            sampler_kwargs['shift'] = cfg.get('shift', self.timestep_sampler.shift)
        if 'output_mode' in signatures:
            sampler_kwargs['output_mode'] = output_mode
        sampler = sampler_class(self.num_timesteps, **sampler_kwargs)

        num_timesteps = cfg.get('num_timesteps', self.num_timesteps)
        num_substeps = cfg.get('num_substeps', 1)
        orthogonal_guidance = cfg.get('orthogonal_guidance', 1.0)

        sampler.set_timesteps(num_timesteps * num_substeps, device=x_t.device)
        timesteps = sampler.timesteps
        self.intermediate_x_t = []
        self.intermediate_x_0 = []
        use_guidance = guidance_scale > 0.0

        if show_pbar:
            pbar = mmcv.ProgressBar(num_timesteps)

        self.init_gm_cache_3d()

        for timestep_id in range(num_timesteps):
            t = timesteps[timestep_id * num_substeps]

            x_t_input = x_t
            if use_guidance:
                x_t_input = torch.cat([x_t_input, x_t_input], dim=0)

            gm_output = self.pred(x_t_input, t, **kwargs)
            assert isinstance(gm_output, dict)
            gm_output = self.u_to_x_0_3d(gm_output, x_t_input, t)

            # Probabilistic CFG (3D)
            if use_guidance:
                gm_cond = {k: v[num_batches:] for k, v in gm_output.items()}
                gm_uncond = {k: v[:num_batches] for k, v in gm_output.items()}
                uncond_mean = gm_to_mean_3d(gm_uncond)
                gaussian_cond = gm_to_iso_gaussian_3d(gm_cond)[0]
                gaussian_cond['var'] = gaussian_cond['var'].mean(dim=(-1, -2, -3), keepdim=True)
                gaussian_output, cfg_bias, avg_var = probabilistic_guidance_3d_jit(
                    gaussian_cond['mean'], gaussian_cond['var'], uncond_mean, guidance_scale,
                    orthogonal=orthogonal_guidance)
                gm_output = gm_mul_iso_gaussian_3d(
                    gm_cond,
                    iso_gaussian_mul_iso_gaussian_3d(gaussian_output, gaussian_cond, 1, -1),
                    1, 1)[0]
            else:
                gaussian_output = gm_to_iso_gaussian_3d(gm_output)[0]

            # GM ODE substeps
            x_t_base = x_t
            t_base = t
            for substep_id in range(num_substeps):
                if substep_id == 0:
                    model_output = self.gm_to_model_output_3d(gm_output, output_mode)
                else:
                    assert output_mode == 'mean'
                    t = timesteps[timestep_id * num_substeps + substep_id]
                    model_output = self.denoising_gm_convert_to_mean_3d(
                        gm_output, x_t, x_t_base, t, t_base, prediction_type='x0')
                x_t = sampler.step(model_output, t, x_t, return_dict=False, prediction_type='x0')[0]

            if show_pbar:
                pbar.update()

        if show_pbar:
            sys.stdout.write('\n')

        return x_t.to(ori_dtype)

    def forward(self, x_0=None, return_loss=False, **kwargs):
        if return_loss:
            return self.forward_train(x_0, **kwargs)
        return self.forward_test(x_0, **kwargs)
