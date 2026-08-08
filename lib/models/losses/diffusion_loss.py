from functools import partial

import torch
import torch.distributed as dist

from mmgen.models import MODULES
from mmgen.models.losses.ddpm_loss import DDPMLoss, mse_loss, reduce_loss
from mmgen.models.losses.utils import weighted_loss

from ...core import reduce_mean


@weighted_loss
def gaussian_nll_loss(pred, target, logstd, eps=1e-4):
    inverse_std = torch.exp(-logstd).clamp(max=1 / eps)
    diff_weighted = (pred - target) * inverse_std
    return 0.5 * diff_weighted.square() + logstd


@weighted_loss
def gaussian_mixture_nll_loss(
        pred_means, target, pred_logstds, pred_logweights, eps=1e-4):
    """
    Args:
        pred_means (torch.Tensor): Shape (bs, *, num_gaussians, c, h, w)
        target (torch.Tensor): Shape (bs, *, c, h, w)
        pred_logstds (torch.Tensor): Shape (bs, *, 1 or num_gaussians, 1 or c, 1 or h, 1 or w)
        pred_logweights (torch.Tensor): Shape (bs, *, num_gaussians, 1, h, w)

    Returns:
        torch.Tensor: Shape (bs, *, h, w)
    """
    inverse_std = torch.exp(-pred_logstds).clamp(max=1 / eps)
    diff_weighted = (pred_means - target.unsqueeze(-4)) * inverse_std
    gaussian_ll = (-0.5 * diff_weighted.square() - pred_logstds).sum(dim=-3)  # (bs, *, num_gaussians, h, w)
    loss = -torch.logsumexp(gaussian_ll + pred_logweights.squeeze(-3), dim=-3)
    return loss


@weighted_loss
def gaussian_mixture_nll_loss_3d(
        pred_means, target, pred_logstds, pred_logweights, eps=1e-4,
        band_weights=None):
    """GM NLL loss for 3D volumes.

    Args:
        pred_means (torch.Tensor): Shape (bs, *, num_gaussians, c, d, h, w)
        target (torch.Tensor): Shape (bs, *, c, d, h, w)
        pred_logstds (torch.Tensor): Shape (bs, *, 1, 1, 1, 1, 1) broadcastable
        pred_logweights (torch.Tensor): Shape (bs, *, num_gaussians, 1, d, h, w)

    Returns:
        torch.Tensor: Shape (bs, *, d, h, w)
    """
    inverse_std = torch.exp(-pred_logstds).clamp(max=1 / eps)
    diff_weighted = (pred_means - target.unsqueeze(-5)) * inverse_std
    channel_ll = -0.5 * diff_weighted.square() - pred_logstds
    if band_weights is not None:
        weights = torch.as_tensor(
            band_weights, dtype=channel_ll.dtype, device=channel_ll.device)
        if weights.numel() != channel_ll.shape[-4]:
            raise ValueError(
                f'band_weights has {weights.numel()} entries, '
                f'but prediction has {channel_ll.shape[-4]} channels')
        weights = weights / weights.mean().clamp(min=1e-6)
        weights = weights.view(*([1] * (channel_ll.dim() - 4)), -1, 1, 1, 1)
        channel_ll = channel_ll * weights
    # sum over channels (dim=-4 for 3D: K, C, d, h, w -> sum C)
    gaussian_ll = channel_ll.sum(dim=-4)
    # gaussian_ll: (bs, *, num_gaussians, d, h, w)
    loss = -torch.logsumexp(gaussian_ll + pred_logweights.squeeze(-4), dim=-4)
    return loss


class DDPMLossMod(DDPMLoss):

    def __init__(self,
                 *args,
                 weight_scale=1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_scale = weight_scale

    def timestep_weight_rescale(self, loss, timesteps, weight):
        return loss * weight.to(timesteps.device)[timesteps] * self.weight_scale

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'You should offer a dictionary containing network outputs '
                'for building up computational graph of this loss module.')
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0, (
                'If the outputs dict is given in keyworded arguments, no'
                ' further non-keyworded arguments should be offered.')
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # check keys in output_dict
        assert 'timesteps' in output_dict, (
            '\'timesteps\' is must for DDPM-based losses, but found'
            f'{output_dict.keys()} in \'output_dict\'')

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        loss_rescaled = self.rescale_fn(loss, timesteps)

        # update log_vars of this class
        self.collect_log(loss_rescaled, timesteps=timesteps)  # Mod: log after rescaling

        return reduce_loss(loss_rescaled, self.reduction)


@MODULES.register_module()
class DDPMMSELossMod(DDPMLossMod):
    _default_data_info = dict(pred='eps_t_pred', target='noise')

    def __init__(self,
                 rescale_mode=None,
                 rescale_cfg=None,
                 sampler=None,
                 weight=None,
                 weight_scale=1.0,
                 log_cfgs=None,
                 reduction='mean',
                 data_info=None,
                 loss_name='loss_ddpm_mse',
                 scale_norm=False,
                 momentum=0.001):
        super().__init__(rescale_mode=rescale_mode,
                         rescale_cfg=rescale_cfg,
                         log_cfgs=log_cfgs,
                         weight=weight,
                         weight_scale=weight_scale,
                         sampler=sampler,
                         reduction=reduction,
                         loss_name=loss_name)

        self.data_info = self._default_data_info \
            if data_info is None else data_info

        self.loss_fn = partial(mse_loss, reduction='flatmean')
        self.scale_norm = scale_norm
        self.freeze_norm = False
        if scale_norm:
            self.register_buffer('norm_factor', torch.ones(1, dtype=torch.float))
        self.momentum = momentum

    def forward(self, *args, **kwargs):
        loss = super().forward(*args, **kwargs)
        if self.scale_norm:
            if self.training and not self.freeze_norm:
                if len(args) == 1:
                    assert isinstance(args[0], dict), (
                        'You should offer a dictionary containing network outputs '
                        'for building up computational graph of this loss module.')
                    output_dict = args[0]
                elif 'output_dict' in kwargs:
                    assert len(args) == 0, (
                        'If the outputs dict is given in keyworded arguments, no'
                        ' further non-keyworded arguments should be offered.')
                    output_dict = kwargs.pop('outputs_dict')
                else:
                    raise NotImplementedError(
                        'Cannot parsing your arguments passed to this loss module.'
                        ' Please check the usage of this module')
                norm_factor = output_dict['x_0'].detach().square().mean()
                norm_factor = reduce_mean(norm_factor)
                self.norm_factor[:] = (1 - self.momentum) * self.norm_factor \
                                      + self.momentum * norm_factor
            loss = loss / self.norm_factor
        return loss

    def _forward_loss(self, outputs_dict):
        """Forward function for loss calculation.
        Args:
            outputs_dict (dict): Outputs of the model used to calculate losses.

        Returns:
            torch.Tensor: Calculated loss.
        """
        loss_input_dict = {
            k: outputs_dict[v]
            for k, v in self.data_info.items()
        }
        loss = self.loss_fn(**loss_input_dict) * 0.5
        return loss


@MODULES.register_module()
class FlowNLLLoss(DDPMLossMod):
    _default_data_info = dict(pred='u_t_pred', target='u_t', logstd='logstd')

    def __init__(self,
                 weight_scale=1.0,
                 log_cfgs=None,
                 data_info=None,
                 reduction='mean',
                 loss_name='loss_ddpm_nll',
                 band_weights=None):
        super().__init__(
            weight_scale=weight_scale,
            log_cfgs=log_cfgs,
            reduction=reduction,
            loss_name=loss_name)
        self.data_info = self._default_data_info \
            if data_info is None else data_info
        self.loss_fn = partial(gaussian_nll_loss, reduction='flatmean')
        if log_cfgs is not None and log_cfgs.get('type', None) == 'quartile':
            for i in range(4):
                self.register_buffer(f'loss_quartile_{i}', torch.zeros((1, ), dtype=torch.float))
                self.register_buffer(f'var_quartile_{i}', torch.ones((1, ), dtype=torch.float))
                self.register_buffer(f'count_quartile_{i}', torch.zeros((1, ), dtype=torch.long))

    @torch.no_grad()
    def collect_log(self, loss, var, timesteps):
        if not self.log_fn_list:
            return

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder_l = [torch.zeros_like(loss) for _ in range(ws)]
            placeholder_v = [torch.zeros_like(var) for _ in range(ws)]
            placeholder_t = [torch.zeros_like(timesteps) for _ in range(ws)]
            dist.all_gather(placeholder_l, loss)
            dist.all_gather(placeholder_v, var)
            dist.all_gather(placeholder_t, timesteps)
            loss = torch.cat(placeholder_l, dim=0)
            var = torch.cat(placeholder_v, dim=0)
            timesteps = torch.cat(placeholder_t, dim=0)
        log_vars = dict()

        if (dist.is_initialized()
                and dist.get_rank() == 0) or not dist.is_initialized():
            for log_fn in self.log_fn_list:
                log_vars.update(log_fn(loss, var, timesteps))
        self.log_vars = log_vars

    @torch.no_grad()
    def quartile_log_collect(self,
                             loss,
                             var,
                             timesteps,
                             total_timesteps,
                             prefix_name,
                             reduction='mean',
                             momentum=0.1):
        quartile = (timesteps / total_timesteps * 4)
        quartile = quartile.to(torch.long).clamp(min=0, max=3)

        log_vars = dict()

        for idx in range(4):
            quartile_mask = quartile == idx
            quartile_count = torch.count_nonzero(quartile_mask).reshape(1)
            if quartile_count > 0:
                loss_quartile = reduce_loss(loss[quartile_mask], reduction).reshape(1)
                var_quartile = reduce_loss(var[quartile_mask], reduction).reshape(1)

                cur_weight = 1 - torch.exp(-momentum * quartile_count)
                getattr(self, f'count_quartile_{idx}').add_(quartile_count)
                total_weight = 1 - torch.exp(-momentum * getattr(self, f'count_quartile_{idx}'))
                cur_weight /= total_weight.clamp(min=1e-4)
                getattr(self, f'loss_quartile_{idx}').mul_(1 - cur_weight).add_(loss_quartile * cur_weight)
                getattr(self, f'var_quartile_{idx}').mul_(1 - cur_weight).add_(var_quartile * cur_weight)

            log_vars[f'{prefix_name}_quartile_{idx}'] = getattr(self, f'loss_quartile_{idx}').item()
            log_vars[f'{prefix_name}_var_quartile_{idx}'] = getattr(self, f'var_quartile_{idx}').item()

        return log_vars

    def _forward_loss(self, outputs_dict):
        loss_input_dict = {
            k: outputs_dict[v]
            for k, v in self.data_info.items()
        }
        loss = self.loss_fn(**loss_input_dict)
        return loss

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'You should offer a dictionary containing network outputs '
                'for building up computational graph of this loss module.')
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0, (
                'If the outputs dict is given in keyworded arguments, no'
                ' further non-keyworded arguments should be offered.')
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # check keys in output_dict
        assert 'timesteps' in output_dict, (
            '\'timesteps\' is must for DDPM-based losses, but found'
            f'{output_dict.keys()} in \'output_dict\'')

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        loss_rescaled = loss * self.weight_scale

        with torch.no_grad():
            var = torch.exp(output_dict['logstd'] * 2)  # (bs, *)
            if 'weight' in self.data_info:
                weight = output_dict[self.data_info['weight']]  # (bs, *)
                weight_norm_factor = weight.flatten(1).mean(dim=1).clamp(min=1e-6)
                _var = (var * weight).flatten(1).mean(dim=1) / weight_norm_factor
                _loss = loss / weight_norm_factor
            else:
                _var = var.flatten(1).mean(dim=1)
                _loss = loss

            # update log_vars of this class
            self.collect_log(_loss, _var, timesteps=timesteps)  # Mod: log after rescaling

        return reduce_loss(loss_rescaled, self.reduction)


@MODULES.register_module()
class GMFlowNLLLoss(FlowNLLLoss):
    _default_data_info = dict(
        pred_means='means',
        target='u_t',
        pred_logstds='logstds',
        pred_logweights='logweights')

    def __init__(self,
                 weight_scale=1.0,
                 log_cfgs=None,
                 data_info=None,
                 reduction='mean',
                 loss_name='loss_ddpm_nll',
                 band_weights=None):
        super().__init__(
            weight_scale=weight_scale,
            log_cfgs=log_cfgs,
            reduction=reduction,
            loss_name=loss_name)
        self.data_info = self._default_data_info \
            if data_info is None else data_info
        self.loss_fn = partial(gaussian_mixture_nll_loss, reduction='flatmean')
        if log_cfgs is not None and log_cfgs.get('type', None) == 'quartile':
            for i in range(4):
                self.register_buffer(f'loss_quartile_{i}', torch.zeros((1,), dtype=torch.float))
                self.register_buffer(f'var_quartile_{i}', torch.ones((1,), dtype=torch.float))
                self.register_buffer(f'count_quartile_{i}', torch.zeros((1,), dtype=torch.long))

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'You should offer a dictionary containing network outputs '
                'for building up computational graph of this loss module.')
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0, (
                'If the outputs dict is given in keyworded arguments, no'
                ' further non-keyworded arguments should be offered.')
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # check keys in output_dict
        assert 'timesteps' in output_dict, (
            '\'timesteps\' is must for DDPM-based losses, but found'
            f'{output_dict.keys()} in \'output_dict\'')

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        loss_rescaled = loss * self.weight_scale

        with torch.no_grad():
            weights = output_dict['logweights'].exp()
            mean = (weights * output_dict['means']).sum(-4, keepdim=True)  # (bs, *, 1, c, h, w)
            var = (weights * ((output_dict['means'] - mean).square()
                              + (output_dict['logstds'] * 2).exp())).sum(-4)  # (bs, *, c, h, w)
            if 'weight' in self.data_info:
                weight = output_dict[self.data_info['weight']].unsqueeze(-3)  # (bs, *, 1, h, w)
                weight_norm_factor = weight.flatten(1).mean(dim=1).clamp(min=1e-6)
                _var = (var * weight).flatten(1).mean(dim=1) / weight_norm_factor
                _loss = loss / weight_norm_factor
            else:
                _var = var.flatten(1).mean(dim=1)
                _loss = loss

            # update log_vars of this class
            self.collect_log(_loss, _var, timesteps=timesteps)  # Mod: log after rescaling

        return reduce_loss(loss_rescaled, self.reduction)


@MODULES.register_module()
class GMFlowNLLLoss3D(FlowNLLLoss):
    """GM NLL loss adapted for 3D volumes (bs, K, C, D, H, W)."""

    _default_data_info = dict(
        pred_means='means',
        target='u_t',
        pred_logstds='logstds',
        pred_logweights='logweights')

    def __init__(self,
                 weight_scale=1.0,
                 log_cfgs=None,
                 data_info=None,
                 reduction='mean',
                 loss_name='loss_ddpm_nll',
                 band_weights=None,
                 mixture_mean_weight=0.0,
                 voxel_gradient_weight=0.0,
                 reconstruct_wavelet=False,
                 boundary_period=None,
                 boundary_weight=1.0):
        super().__init__(
            weight_scale=weight_scale,
            log_cfgs=log_cfgs,
            reduction=reduction,
            loss_name=loss_name)
        self.data_info = self._default_data_info \
            if data_info is None else data_info
        self.band_weights = band_weights
        self.mixture_mean_weight = float(mixture_mean_weight)
        self.voxel_gradient_weight = float(voxel_gradient_weight)
        self.reconstruct_wavelet = bool(reconstruct_wavelet)
        self.boundary_period = boundary_period
        self.boundary_weight = float(boundary_weight)
        if self.mixture_mean_weight < 0 or self.voxel_gradient_weight < 0:
            raise ValueError('Auxiliary loss weights must be non-negative')
        if self.boundary_period is not None and self.boundary_period < 2:
            raise ValueError('boundary_period must be at least 2')
        if self.boundary_weight < 1:
            raise ValueError('boundary_weight must be at least 1')
        self.loss_fn = partial(
            gaussian_mixture_nll_loss_3d,
            reduction='flatmean',
            band_weights=band_weights)
        if log_cfgs is not None and log_cfgs.get('type', None) == 'quartile':
            for i in range(4):
                self.register_buffer(f'loss_quartile_{i}', torch.zeros((1,), dtype=torch.float))
                self.register_buffer(f'var_quartile_{i}', torch.ones((1,), dtype=torch.float))
                self.register_buffer(f'count_quartile_{i}', torch.zeros((1,), dtype=torch.long))

    def _channel_weighted_l1(self, prediction, target):
        error = (prediction - target).abs()
        if self.band_weights is None:
            return error.mean()

        weights = torch.as_tensor(
            self.band_weights, dtype=error.dtype, device=error.device)
        if weights.numel() != error.shape[1]:
            raise ValueError(
                f'band_weights has {weights.numel()} entries, '
                f'but prediction has {error.shape[1]} channels')
        weights = weights / weights.mean().clamp(min=1e-6)
        return (error * weights.view(1, -1, 1, 1, 1)).mean()

    def _gradient_matching_loss(self, prediction, target):
        losses = []
        for axis in (-3, -2, -1):
            pred_gradient = torch.diff(prediction, dim=axis)
            target_gradient = torch.diff(target, dim=axis)
            error = (pred_gradient - target_gradient).abs()

            if self.boundary_period is not None and self.boundary_weight > 1:
                length = error.shape[axis]
                indices = torch.arange(length, device=error.device)
                boundary = (indices + 1).remainder(self.boundary_period) == 0
                axis_weights = torch.where(
                    boundary,
                    error.new_tensor(self.boundary_weight),
                    error.new_tensor(1.0))
                view_shape = [1] * error.dim()
                view_shape[axis] = length
                axis_weights = axis_weights.view(view_shape)
                losses.append(
                    (error * axis_weights).mean()
                    / axis_weights.mean().clamp(min=1e-6))
            else:
                losses.append(error.mean())
        return sum(losses) / len(losses)

    def _auxiliary_losses(self, output_dict):
        zero = output_dict['means'].new_zeros(())
        if self.mixture_mean_weight == 0 and self.voxel_gradient_weight == 0:
            return zero, zero

        weights = output_dict['logweights'].exp()
        prediction = (weights * output_dict['means']).sum(dim=-5)
        target = output_dict[self.data_info['target']]

        mean_loss = (
            self._channel_weighted_l1(prediction, target)
            if self.mixture_mean_weight > 0 else zero)

        if self.voxel_gradient_weight > 0:
            if self.reconstruct_wavelet:
                from ..diffusions.wavelet import haar_idwt3d
                prediction = haar_idwt3d(prediction)
                target = haar_idwt3d(target)
            gradient_loss = self._gradient_matching_loss(prediction, target)
        else:
            gradient_loss = zero
        return mean_loss, gradient_loss

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            assert isinstance(args[0], dict)
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.')

        assert 'timesteps' in output_dict

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        loss_rescaled = loss * self.weight_scale
        mean_aux, gradient_aux = self._auxiliary_losses(output_dict)
        auxiliary_loss = (
            self.mixture_mean_weight * mean_aux
            + self.voxel_gradient_weight * gradient_aux)

        with torch.no_grad():
            weights = output_dict['logweights'].exp()
            # 3D: sum over gaussian dim -5
            mean = (weights * output_dict['means']).sum(-5, keepdim=True)
            var = (weights * ((output_dict['means'] - mean).square()
                              + (output_dict['logstds'] * 2).exp())).sum(-5)
            _var = var.flatten(1).mean(dim=1)
            _loss = loss

            self.collect_log(_loss, _var, timesteps=timesteps)
            self.log_vars.update(
                loss_gm_mean=float(mean_aux.detach()),
                loss_voxel_gradient=float(gradient_aux.detach()),
                loss_auxiliary=float(auxiliary_loss.detach()))

        return reduce_loss(loss_rescaled, self.reduction) + auxiliary_loss
