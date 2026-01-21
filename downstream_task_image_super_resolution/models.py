# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp
from model_utils import PatchEmbed_1d

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class NumericalEmbedder(nn.Module):
    """
    Numerical embedding for continuous values.
    """
    def __init__(self, data_info, hidden_size, dropout_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.input_mean = data_info["mean"]
        self.input_std = data_info["std"]
        self.normalizer = lambda x: (x - self.input_mean) / self.input_std
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size,bias=True),
            nn.Tanh()  # Ensure outputs are bounded
        )
        self.dropout = nn.Dropout(dropout_prob)

    def value_drop(self, values, force_drop_mask=None):
        """
        Applies dropout to continuous values by zeroing out certain inputs.
        """
        if force_drop_mask is None:
            drop_mask = torch.rand(values.shape[0], device=values.device) < self.dropout_prob
        else:
            drop_mask = force_drop_mask == 1
        values = torch.where(drop_mask.unsqueeze(1), torch.zeros_like(values), values)
        return values

    def forward(self, values, train=True, force_drop_mask=None):
        if values.ndim == 1:
            values = values.unsqueeze(1)  # Ensure shape is (batch_size, 1)
        values = self.normalizer(values)
        if (train and self.dropout_prob > 0) or (force_drop_mask is not None):
            values = self.value_drop(values, force_drop_mask)
        embeddings = self.mlp(values)
        return embeddings

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x



class FinalLayer_1d(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT_UKB(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=228453,
        patch_size=256,
        padding_size=155,
        in_channels=1,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        condition_info = None,
        learn_sigma=True
    ):
        super().__init__()
        self.learn_sigma  = learn_sigma
        self.in_channels  = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size   = patch_size
        self.padding_size = padding_size
        self.num_heads    = num_heads
        self.condition_info = condition_info
        self.class_dropout_prob =class_dropout_prob

        self.x_embedder  = PatchEmbed_1d(input_size=input_size, patch_size=patch_size, padding_size=padding_size,in_chans=in_channels, embed_dim=hidden_size)
        self.t_embedder  = TimestepEmbedder(hidden_size)
        self.y1_embedder = NumericalEmbedder(self.condition_info['age_info'], hidden_size, class_dropout_prob)   # age
        self.y2_embedder = LabelEmbedder(self.condition_info['sex_num'], hidden_size, class_dropout_prob) # sex
        self.y3_embedder = LabelEmbedder(self.condition_info['modality_mum'], hidden_size, class_dropout_prob)   # modality

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer_1d(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)+1)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed[:self.x_embedder.num_patches]).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y1_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y1_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y2_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.y3_embedder.embedding_table.weight, std=0.02)
        
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size * C)
        imgs: (N, C,L)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size
        t = x.shape[1]
        x = x.reshape(shape=(x.shape[0], t, p, c))
        x = torch.einsum('ntpc->nctp', x)
        imgs = x.reshape(shape=(x.shape[0], c, t * p))
        return imgs[:,:,:-self.padding_size]
    
        
    def forward(self, x, t, y1, y2, y3):
        """
        Forward pass of DiT.
        x: (N, C, L) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y1: (N,) tensor of age labels
        y2: (N,) tensor of sex labels
        y3: (N,) tensor of modality
        """
        x = self.x_embedder(x) + self.pos_embed      # (N, T, D), where T = L / patch_size 
        t = self.t_embedder(t)                       # (N, D)

        drop_mask = torch.rand(x.shape[0], device=x.device) < self.class_dropout_prob

        y1 = self.y1_embedder(y1, self.training,drop_mask)     # (N, D)  age
        y2 = self.y2_embedder(y2, self.training,drop_mask)     # (N, D)  sex
        y3 = self.y3_embedder(y3, self.training,drop_mask)     # (N, D)  modality

        c = t + y1 + y2 + y3                        
        for block in self.blocks:
            x = block(x, c)                       # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size  * out_channels)
        x = self.unpatchify(x)                    # (N, out_channels, L)
        return x

    def forward_with_cfg(self, x, t, y1, y2, y3, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y1,y2,y3)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

#################################################################################
#                                   DiT Configs                                 #
#################################################################################


def DiT_UKB_B_256(**kwargs):
    return DiT_UKB(depth=12, hidden_size=768, patch_size=256,  num_heads=12, **kwargs)

def DiT_UKB_L_256(**kwargs):
    return DiT_UKB(depth=24, hidden_size=1024, patch_size=256,  num_heads=16, **kwargs)


DiT_models = {
        'DiT-UKB-B/256': DiT_UKB_B_256,
        'DiT-UKB-L/256': DiT_UKB_L_256
}
