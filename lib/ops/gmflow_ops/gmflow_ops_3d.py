"""Gaussian mixture operations adapted for 3D volumes.

Tensor layout for 3D GM:
    means:      (bs, *, num_gaussians, out_channels, d, h, w)
    logstds:    (bs, *, 1, 1, 1, 1, 1)
    logweights: (bs, *, num_gaussians, 1, d, h, w)

Compared to 2D where spatial dims are (h, w), here we have (d, h, w).
The gaussian dim index shifts from -4 to -5.
"""

import torch


@torch.jit.script
def gm_to_iso_gaussian_3d_jit(gm_weights, gm_means, gm_vars):
    """Convert 3D GM to isotropic Gaussian.

    Args:
        gm_weights: (bs, *, num_gaussians, 1, d, h, w)
        gm_means:   (bs, *, num_gaussians, out_channels, d, h, w)
        gm_vars:    (bs, *, 1, 1, 1, 1, 1)
    """
    g_mean = (gm_weights * gm_means).sum(-5, keepdim=True)  # (bs, *, 1, C, d, h, w)
    gm_diffs = gm_means - g_mean  # (bs, *, K, C, d, h, w)
    g_var = (gm_weights * (gm_diffs * gm_diffs)).sum(-5, keepdim=True).mean(-4, keepdim=True) + gm_vars
    # g_var: (bs, *, 1, 1, d, h, w)
    return g_mean, g_var, gm_diffs


def gm_to_iso_gaussian_3d(gm):
    """Convert 3D GM dict to isotropic Gaussian.

    Args:
        gm (dict):
            means:      (bs, *, num_gaussians, out_channels, d, h, w)
            logstds:    (bs, *, 1, 1, 1, 1, 1)
            logweights: (bs, *, num_gaussians, 1, d, h, w)

    Returns:
        (gaussian_dict, gm_diffs, gm_vars)
    """
    gm_means = gm['means']
    gm_logweights = gm['logweights']

    if 'gm_weights' in gm:
        gm_weights = gm['gm_weights']
    else:
        gm_weights = gm_logweights.exp()
        gm['gm_weights'] = gm_weights

    gm_logstds = gm['logstds']
    if 'gm_vars' in gm:
        gm_vars = gm['gm_vars']
    else:
        gm_vars = (gm_logstds * 2).exp()
        gm['gm_vars'] = gm_vars

    g_mean, g_var, gm_diffs = gm_to_iso_gaussian_3d_jit(gm_weights, gm_means, gm_vars)
    gaussian = dict(
        mean=g_mean.squeeze(-5),  # (bs, *, C, d, h, w)
        var=g_var.squeeze(-5))    # (bs, *, 1, d, h, w)

    return gaussian, gm_diffs, gm_vars


def gm_to_mean_3d(gm):
    """Get mean of 3D Gaussian mixture.

    Args:
        gm (dict):
            means:      (bs, *, num_gaussians, out_channels, d, h, w)
            logweights: (bs, *, num_gaussians, 1, d, h, w)

    Returns:
        torch.Tensor: (bs, *, out_channels, d, h, w)
    """
    if 'gm_weights' in gm:
        weights = gm['gm_weights']
    else:
        weights = gm['logweights'].exp()
        gm['gm_weights'] = weights
    return (weights * gm['means']).sum(dim=-5)


def gm_to_sample_3d(gm, n_samples=1):
    """Sample from 3D Gaussian mixture.

    For basic case without extra batch dims (*):
        means:      (bs, K, C, d, h, w) - 6D
        logstds:    (bs, 1, 1, 1, 1, 1) - 6D
        logweights: (bs, K, 1, d, h, w) - 6D

    Returns:
        torch.Tensor: (bs, n_samples, C, d, h, w)
    """
    means = gm['means']       # (bs, K, C, d, h, w)
    logstds = gm['logstds']   # (bs, 1, 1, 1, 1, 1)
    logweights = gm['logweights']  # (bs, K, 1, d, h, w)

    batch_shape = means.shape[:-5]  # (bs,) or (bs, *)
    K, C = means.shape[-5], means.shape[-4]
    d, h, w = means.shape[-3:]

    # Flatten batch dims for simplicity
    flat_means = means.reshape(-1, K, C, d, h, w)
    flat_lw = logweights.reshape(-1, K, 1, d, h, w)
    flat_bs = flat_means.shape[0]

    # Gumbel-max trick to sample component indices
    # (flat_bs, n_samples, K, 1, d, h, w)
    gumbel = -torch.empty(
        flat_bs, n_samples, K, 1, d, h, w,
        device=means.device, dtype=means.dtype).exponential_().log()
    indices = (flat_lw.unsqueeze(1) + gumbel).argmax(dim=2, keepdim=True)
    # indices: (flat_bs, n_samples, 1, 1, d, h, w)

    # Gather means: (flat_bs, n_samples, K, C, d, h, w) -> gather on dim=2
    means_exp = flat_means.unsqueeze(1).expand(flat_bs, n_samples, K, C, d, h, w)
    idx_exp = indices.expand(flat_bs, n_samples, 1, C, d, h, w)
    selected = means_exp.gather(2, idx_exp).squeeze(2)  # (flat_bs, n_samples, C, d, h, w)

    # Add Gaussian noise - logstds is scalar-like, reshape to (flat_bs, 1, 1, 1, 1, 1) for broadcast
    std = logstds.reshape(-1, *([1] * (selected.dim() - 1))).exp()
    noise = torch.randn_like(selected) * std
    samples = selected + noise

    # Restore batch shape
    return samples.reshape(*batch_shape, n_samples, C, d, h, w)


@torch.jit.script
def gm_mul_iso_gaussian_3d_jit(
        gm_means, gm_vars, gm_logweights,
        g_mean, g_var, eps: float = 1e-6):
    """Multiply 3D GM by isotropic Gaussian.

    gm_means:      (bs, *, K, C, d, h, w)
    gm_vars:       (bs, *, 1, 1, 1, 1, 1)
    gm_logweights: (bs, *, K, 1, d, h, w)
    g_mean:        (bs, *, 1, C, d, h, w) or broadcastable
    g_var:         (bs, *, 1, 1, d, h, w) or broadcastable
    """
    norm_factor = (g_var + gm_vars).clamp(min=eps)
    out_means = (g_var * gm_means + gm_vars * g_mean) / norm_factor
    out_vars = g_var * gm_vars / norm_factor

    gm_diffs = gm_means - g_mean
    logweights_delta = gm_diffs.square().sum(dim=-4, keepdim=True) * (-0.5 / norm_factor)
    out_logweights = (gm_logweights + logweights_delta).log_softmax(dim=-5)

    return out_means, out_vars, out_logweights


def gm_mul_iso_gaussian_3d(gm, iso_gaussian, gm_sign=1, iso_sign=1):
    """Multiply 3D GM by isotropic Gaussian (with optional sign for division).

    Args:
        gm (dict): GM distribution.
        iso_gaussian (dict): Isotropic Gaussian with 'mean' and 'var'.
        gm_sign (int): 1 for multiply, -1 for divide.
        iso_sign (int): 1 for multiply, -1 for divide.

    Returns:
        (result_gm, result_gm_diffs)
    """
    gm_means = gm['means']
    gm_logweights = gm['logweights']

    if 'gm_vars' in gm:
        gm_vars = gm['gm_vars']
    else:
        gm_vars = (gm['logstds'] * 2).exp()
        gm['gm_vars'] = gm_vars

    g_mean = iso_gaussian['mean'].unsqueeze(-5)
    g_var = iso_gaussian['var'].unsqueeze(-5)

    out_means, out_vars, out_logweights = gm_mul_iso_gaussian_3d_jit(
        gm_means, gm_vars * gm_sign, gm_logweights,
        g_mean, g_var * iso_sign)

    result = dict(
        means=out_means,
        logweights=out_logweights,
        logstds=(out_vars.clamp(min=1e-12).log() / 2),
        gm_vars=out_vars,
        gm_weights=out_logweights.exp())

    return result, None


def iso_gaussian_mul_iso_gaussian_3d(g1, g2, sign1=1, sign2=1, eps=1e-6):
    """Multiply (or divide) two isotropic Gaussians for 3D.

    Works identically to 2D version since it operates on scalar var.
    """
    v1 = g1['var'] * sign1
    v2 = g2['var'] * sign2
    norm = (v1 + v2).clamp(min=eps)
    out_var = v1 * v2 / norm
    out_mean = (v2 * g1['mean'] + v1 * g2['mean']) / norm
    return dict(mean=out_mean, var=out_var)


def gaussian_samples_to_gm_samples_3d(gm, gaussian_samples, axis_aligned=False):
    """Convert standard Gaussian samples to GM samples for 3D.

    Args:
        gm (dict): GM distribution.
        gaussian_samples: (bs, *, n_samples, C, d, h, w)

    Returns:
        torch.Tensor: (bs, *, n_samples, C, d, h, w)
    """
    gaussian, gm_diffs, gm_vars = gm_to_iso_gaussian_3d(gm)
    g_std = gaussian['var'].unsqueeze(-5).clamp(min=1e-12).sqrt()
    gm_std = gm_vars.clamp(min=1e-12).sqrt()

    if axis_aligned:
        gm_std_ratio = gm_std / g_std.clamp(min=1e-6)
        weights = gm['gm_weights']  # (bs, *, K, 1, d, h, w)
        correction = (weights * gm_diffs * (1 - gm_std_ratio)).sum(dim=-5)
        samples = gaussian_samples * (
            (weights * gm_std_ratio).sum(dim=-5)
        ) + correction
    else:
        samples = gaussian_samples  # Fallback
    return samples


def gm_samples_to_gaussian_samples_3d(gm, samples, axis_aligned=False):
    """Convert GM samples to standard Gaussian samples for 3D.

    Args:
        gm (dict): GM distribution.
        samples: (bs, *, n_samples, C, d, h, w)

    Returns:
        torch.Tensor: (bs, *, n_samples, C, d, h, w)
    """
    gaussian, gm_diffs, gm_vars = gm_to_iso_gaussian_3d(gm)
    g_std = gaussian['var'].unsqueeze(-5).clamp(min=1e-12).sqrt()
    gm_std = gm_vars.clamp(min=1e-12).sqrt()

    if axis_aligned:
        gm_std_ratio = g_std / gm_std.clamp(min=1e-6)
        weights = gm['gm_weights']
        correction = (weights * gm_diffs * (1 - gm_std_ratio)).sum(dim=-5)
        gaussian_samples = (samples - correction) / (
            (weights * gm_std_ratio.reciprocal()).sum(dim=-5).clamp(min=1e-6)
        )
    else:
        gaussian_samples = samples
    return gaussian_samples
