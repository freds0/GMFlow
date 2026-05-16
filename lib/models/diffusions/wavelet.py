import math

import torch


def haar_dwt3d(x):
    """One-level orthonormal 3D Haar DWT.

    Converts (B, C, D, H, W) to (B, 8*C, D/2, H/2, W/2) with subbands ordered:
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH.
    """
    if x.dim() != 5:
        raise ValueError(f'Expected 5D tensor, got {x.dim()}D')
    if any(size % 2 != 0 for size in x.shape[-3:]):
        raise ValueError(f'DWT requires even spatial dims, got {tuple(x.shape[-3:])}')

    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    x_even = x[:, :, 0::2, :, :]
    x_odd = x[:, :, 1::2, :, :]
    l_d = (x_even + x_odd) * inv_sqrt2
    h_d = (-x_even + x_odd) * inv_sqrt2

    bands_d = []
    for band in (l_d, h_d):
        x_even = band[:, :, :, 0::2, :]
        x_odd = band[:, :, :, 1::2, :]
        bands_d.append((
            (x_even + x_odd) * inv_sqrt2,
            (-x_even + x_odd) * inv_sqrt2,
        ))

    subbands = []
    for l_h in bands_d:
        for band in l_h:
            x_even = band[:, :, :, :, 0::2]
            x_odd = band[:, :, :, :, 1::2]
            subbands.extend([
                (x_even + x_odd) * inv_sqrt2,
                (-x_even + x_odd) * inv_sqrt2,
            ])

    return torch.cat(subbands, dim=1)


def haar_idwt3d(x):
    """Inverse of :func:`haar_dwt3d`.

    Converts (B, 8*C, D, H, W) to (B, C, 2D, 2H, 2W).
    """
    if x.dim() != 5:
        raise ValueError(f'Expected 5D tensor, got {x.dim()}D')
    if x.size(1) % 8 != 0:
        raise ValueError(f'IDWT channel count must be divisible by 8, got {x.size(1)}')

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    b, channels8, d, h, w = x.shape
    channels = channels8 // 8
    bands = x.reshape(b, 8, channels, d, h, w)

    restored_d_bands = []
    idx = 0
    for _ in range(2):
        restored_h_bands = []
        for _ in range(2):
            low_w = bands[:, idx]
            high_w = bands[:, idx + 1]
            idx += 2
            merged_w = x.new_empty(b, channels, d, h, w * 2)
            merged_w[:, :, :, :, 0::2] = (low_w - high_w) * inv_sqrt2
            merged_w[:, :, :, :, 1::2] = (low_w + high_w) * inv_sqrt2
            restored_h_bands.append(merged_w)

        low_h, high_h = restored_h_bands
        merged_h = x.new_empty(b, channels, d, h * 2, w * 2)
        merged_h[:, :, :, 0::2, :] = (low_h - high_h) * inv_sqrt2
        merged_h[:, :, :, 1::2, :] = (low_h + high_h) * inv_sqrt2
        restored_d_bands.append(merged_h)

    low_d, high_d = restored_d_bands
    out = x.new_empty(b, channels, d * 2, h * 2, w * 2)
    out[:, :, 0::2, :, :] = (low_d - high_d) * inv_sqrt2
    out[:, :, 1::2, :, :] = (low_d + high_d) * inv_sqrt2
    return out
