"""3D GMFlow architecture for volumetric generation conditioned on age.

Key differences from 2D:
- PatchEmbed3D uses Conv3d instead of Conv2d
- AgeEmbedding replaces LabelEmbedding (continuous age vs discrete class)
- GMOutput3D handles 6D tensors (bs, K, C, D, H, W)
- Unpatchify reshapes (bs, D', H', W', p, p, p, gm_channels) -> (bs, gm_channels, D, H, W)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional
from functools import partial

from diffusers.models.attention import BasicTransformerBlock, _chunked_feed_forward, Attention, FeedForward
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.models.normalization import AdaLayerNormZero
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import register_to_config, ConfigMixin

from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, xavier_init
from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger

from .diffusers import autocast_patch
from .gmflow import BasicTransformerBlockMod
from ...core import rgetattr


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding: Conv3d projection + learned positional embedding."""

    def __init__(self, volume_size=64, patch_size=4, in_channels=1, embed_dim=768):
        super().__init__()
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.num_patches_per_dim = volume_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 3

        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim))

    def forward(self, x):
        """
        Args:
            x: (bs, C, D, H, W)
        Returns:
            (bs, num_patches, embed_dim)
        """
        x = self.proj(x)  # (bs, embed_dim, D', H', W')
        x = x.flatten(2).transpose(1, 2)  # (bs, num_patches, embed_dim)
        x = x + self.pos_embed
        return x


class AgeEmbedding(nn.Module):
    """Embeds continuous age value into a vector.

    Uses a small MLP to map scalar age to embedding dim.
    Includes a learnable null embedding for CFG dropout.
    """

    def __init__(self, hidden_size, dropout_prob=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size))
        self.dropout_prob = dropout_prob
        self.null_embedding = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, age, force_drop=False):
        """
        Args:
            age: (bs,) normalized age in [0, 1] or negative for null
        Returns:
            (bs, hidden_size)
        """
        # Detect null condition (negative age = unconditional)
        is_null = age < 0  # (bs,)

        age_input = age.clamp(min=0).unsqueeze(-1)  # (bs, 1)
        emb = self.mlp(age_input)  # (bs, hidden_size)

        # Replace with null embedding where age is negative
        null = self.null_embedding.unsqueeze(0).expand_as(emb)
        emb = torch.where(is_null.unsqueeze(-1), null, emb)

        return emb


class CombinedTimestepAgeEmbeddings(nn.Module):
    """Combines sinusoidal timestep embedding with age embedding (summed)."""

    def __init__(self, embedding_dim, age_dropout_prob=0.0):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim)
        self.age_embedder = AgeEmbedding(
            embedding_dim, dropout_prob=age_dropout_prob)

    def forward(self, timestep, age, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        if hidden_dtype is not None:
            timesteps_proj = timesteps_proj.to(dtype=hidden_dtype)
        timestep_emb = self.timestep_embedder(timesteps_proj)
        age_emb = self.age_embedder(age)
        return timestep_emb + age_emb


class GMOutput3D(nn.Module):
    """Splits raw output into GM parameters for 3D volumes.

    Output layout:
        means:      (bs, num_gaussians, out_channels, D, H, W)
        logweights: (bs, num_gaussians, 1, D, H, W)
        logstds:    (bs, 1, 1, 1, 1, 1)
    """

    def __init__(self,
                 num_gaussians,
                 out_channels,
                 embed_dim,
                 constant_logstd=None,
                 logstd_inner_dim=1024,
                 num_logstd_layers=2,
                 activation_fn='silu'):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.constant_logstd = constant_logstd

        if constant_logstd is None:
            if activation_fn == 'gelu-approximate':
                act = partial(nn.GELU, approximate='tanh')
            elif activation_fn == 'silu':
                act = nn.SiLU
            else:
                raise ValueError(f'Unsupported activation function: {activation_fn}')

            assert num_logstd_layers >= 1
            in_dim = self.embed_dim
            logstd_layers = []
            for _ in range(num_logstd_layers - 1):
                logstd_layers.extend([
                    act(),
                    nn.Linear(in_dim, logstd_inner_dim)])
                in_dim = logstd_inner_dim
            self.logstd_layers = nn.Sequential(
                *logstd_layers,
                act(),
                nn.Linear(in_dim, 1))

        self.init_weights()

    def init_weights(self):
        if self.constant_logstd is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    xavier_init(m, distribution='uniform')
            constant_init(self.logstd_layers[-1], val=0)

    def forward(self, x, emb):
        """
        Args:
            x: (bs, gm_channels, D, H, W) where gm_channels = K*(C+1)
            emb: (bs, embed_dim) conditioning embedding
        Returns:
            dict with means, logweights, logstds
        """
        bs, c, d, h, w = x.size()
        means, logweights = x.split(
            [self.num_gaussians * self.out_channels, self.num_gaussians], dim=1)
        means = means.view(bs, self.num_gaussians, self.out_channels, d, h, w)
        logweights = logweights.view(bs, self.num_gaussians, 1, d, h, w).log_softmax(dim=1)
        if self.constant_logstd is None:
            logstds = self.logstd_layers(emb).view(bs, 1, 1, 1, 1, 1)
        else:
            logstds = torch.full(
                (bs, 1, 1, 1, 1, 1), self.constant_logstd,
                dtype=x.dtype, device=x.device)
        return dict(
            means=means,
            logweights=logweights,
            logstds=logstds)


class _GMDiTTransformer3DModel(ModelMixin, ConfigMixin):
    """3D DiT backbone for GM flow matching on volumetric data.

    Uses the same BasicTransformerBlockMod (dimension-agnostic) as 2D,
    but with 3D patch embedding and 3D unpatchify.
    """

    @register_to_config
    def __init__(
            self,
            num_gaussians=4,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            age_dropout_prob=0.0,
            num_attention_heads: int = 12,
            attention_head_dim: int = 64,
            in_channels: int = 1,
            out_channels: Optional[int] = None,
            num_layers: int = 12,
            dropout: float = 0.0,
            attention_bias: bool = True,
            sample_size: int = 64,
            patch_size: int = 4,
            activation_fn: str = 'gelu-approximate',
            norm_type: str = 'ada_norm_zero',
            norm_elementwise_affine: bool = False,
            norm_eps: float = 1e-5,
            upcast_attention: bool = False):

        super().__init__()

        if norm_type != 'ada_norm_zero':
            raise NotImplementedError(f'norm_type={norm_type} not supported.')

        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gm_channels = num_gaussians * (self.out_channels + 1)
        self.gradient_checkpointing = False

        self.volume_size = sample_size
        self.patch_size = patch_size
        self.num_patches_per_dim = sample_size // patch_size

        # 1. Patch embedding (3D)
        self.pos_embed = PatchEmbed3D(
            volume_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim)

        # 2. Timestep + Age conditioning
        self.emb = CombinedTimestepAgeEmbeddings(
            self.inner_dim, age_dropout_prob=0.0)

        # 3. Transformer blocks (same as 2D, sequence-based)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlockMod(
                self.inner_dim,
                num_attention_heads,
                attention_head_dim,
                dropout=dropout,
                activation_fn=activation_fn,
                num_embeds_ada_norm=None,
                attention_bias=attention_bias,
                upcast_attention=upcast_attention,
                norm_type=norm_type,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps)
            for _ in range(num_layers)])

        # 4. Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(
            self.inner_dim,
            patch_size * patch_size * patch_size * self.gm_channels)

        # 5. GM output head
        self.gm_out = GMOutput3D(
            num_gaussians,
            self.out_channels,
            self.inner_dim,
            constant_logstd=constant_logstd,
            logstd_inner_dim=logstd_inner_dim,
            num_logstd_layers=gm_num_logstd_layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv3d)
        w = self.pos_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.pos_embed.proj.bias, 0)

        # Initialize positional embedding
        nn.init.normal_(self.pos_embed.pos_embed, std=0.02)

        # Zero-out adaLN modulation layers
        for m in self.modules():
            if isinstance(m, AdaLayerNormZero):
                constant_init(m.linear, val=0)

        # Zero-out output layers
        constant_init(self.proj_out_1, val=0)

        self.gm_out.init_weights()

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            age: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None):
        """
        Args:
            hidden_states: (bs, C, D, H, W) input volume
            timestep: (bs,) diffusion timestep
            age: (bs,) normalized age (negative = unconditional)
        Returns:
            dict with means, logweights, logstds (GM parameters)
        """
        bs = hidden_states.size(0)
        p = self.patch_size
        npd = self.num_patches_per_dim  # patches per spatial dim

        # 1. Patch embed
        hidden_states = self.pos_embed(hidden_states)  # (bs, num_patches, inner_dim)

        # 2. Conditioning
        cond_emb = self.emb(timestep, age, hidden_dtype=hidden_states.dtype)

        dropout_enabled = self.config.age_dropout_prob > 0 and self.training
        if dropout_enabled:
            # Create unconditional embedding (negative age triggers null embedding)
            uncond_emb = self.emb(
                timestep,
                torch.full_like(age, -1.0),
                hidden_dtype=hidden_states.dtype)

        # 3. Transformer blocks
        for block in self.transformer_blocks:
            if dropout_enabled:
                dropout_mask = torch.rand((bs, 1), device=hidden_states.device) < self.config.age_dropout_prob
                emb = torch.where(dropout_mask, uncond_emb, cond_emb)
            else:
                emb = cond_emb

            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None, None, None, timestep,
                    cross_attention_kwargs, None, emb,
                    use_reentrant=False)
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=None,
                    emb=emb)

        # 4. Output: adaLN + project + unpatchify 3D
        if dropout_enabled:
            dropout_mask = torch.rand((bs, 1), device=hidden_states.device) < self.config.age_dropout_prob
            emb = torch.where(dropout_mask, uncond_emb, cond_emb)
        else:
            emb = cond_emb

        shift, scale = self.proj_out_1(F.silu(emb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states)
        # hidden_states: (bs, npd^3, p^3 * gm_channels)

        # Unpatchify 3D: reshape to volume
        hidden_states = hidden_states.reshape(
            bs, npd, npd, npd, p, p, p, self.gm_channels
        ).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
            bs, self.gm_channels,
            npd * p, npd * p, npd * p)
        # hidden_states: (bs, gm_channels, D, H, W)

        return self.gm_out(hidden_states, cond_emb.detach())


@MODULES.register_module()
class GMDiTTransformer3DModel(_GMDiTTransformer3DModel):
    """Registered wrapper with freeze/pretrained/checkpointing support."""

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            torch_dtype='float32',
            freeze_exclude_fp32=True,
            checkpointing=True,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)
            for attr in freeze_exclude:
                rgetattr(self, attr).requires_grad_(True)

        self.init_weights(pretrained)
        if torch_dtype is not None:
            self.to(getattr(torch, torch_dtype))

        self.freeze_exclude_fp32 = freeze_exclude_fp32
        if self.freeze_exclude_fp32:
            for attr in freeze_exclude:
                m = rgetattr(self, attr)
                assert isinstance(m, nn.Module)
                m.to(torch.float32)
                autocast_patch(m, enabled=False)

        if checkpointing:
            self.enable_gradient_checkpointing()

    def init_weights(self, pretrained=None):
        super().init_weights()
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
