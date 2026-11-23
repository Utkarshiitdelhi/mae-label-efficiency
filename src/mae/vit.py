"""Vision Transformer written from scratch (no timm / HF / nn.MultiheadAttention).

ViT-Tiny (DeiT-Ti): 12 pre-norm blocks, width 192, 3 heads, MLP 768.
CIFAR-10 at 32 px with 4x4 patches -> 8x8 = 64 tokens of 48 values each.
No class token: the classifier mean-pools patch tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos_embed import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    """Linear 48 -> D per 4x4x3 patch, written as a stride-4 conv."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                      # (N, 3, H, W)
        x = self.proj(x)                       # (N, D, 8, 8)
        return x.flatten(2).transpose(1, 2)    # (N, 64, D), row-major tokens


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):                      # (N, T, D)
        N, T, D = x.shape
        qkv = self.qkv(x).reshape(N, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)   # each (N, H, T, hd)
        # softmax(q k^T / sqrt(hd)) v, fused kernel for speed
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(N, T, D)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def drop_path(x, p: float, training: bool):
    """Stochastic depth: drop the whole residual branch per sample."""
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty(x.shape[0], *([1] * (x.ndim - 1))).bernoulli_(keep)
    return x * mask / keep


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)); x + MLP(LN(x))."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0, eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.drop_path = drop_path

    def forward(self, x):
        x = x + drop_path(self.attn(self.norm1(x)), self.drop_path, self.training)
        x = x + drop_path(self.mlp(self.norm2(x)), self.drop_path, self.training)
        return x


def init_weights(module):
    """MAE-style init: xavier-uniform linears, unit LayerNorm."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ViTEncoder(nn.Module):
    """ViT encoder. `forward(x, ids_keep)` runs only on the kept tokens."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192, depth=12,
                 num_heads=3, mlp_ratio=4.0, drop_path_rate=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        pe = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size)
        self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)  # fixed
        dpr = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]   # linear ramp
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.apply(init_weights)
        # patch embedding initialised like a linear layer (as in MAE)
        w = self.patch_embed.proj.weight
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.proj.bias)

    def forward(self, x, ids_keep=None):
        """x: images (N,3,H,W). ids_keep: (N, K) indices of visible tokens, or None
        for all 64. Masked tokens are *dropped*, never replaced by a mask token."""
        x = self.patch_embed(x) + self.pos_embed
        if ids_keep is not None:
            x = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class ViTClassifier(nn.Module):
    """Encoder on all 64 tokens -> mean-pool -> LayerNorm -> Linear. Same head for both arms."""

    def __init__(self, encoder: ViTEncoder, num_classes=10, head_init_std=0.02):
        super().__init__()
        self.encoder = encoder
        self.fc_norm = nn.LayerNorm(encoder.embed_dim, eps=1e-6)
        self.head = nn.Linear(encoder.embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=head_init_std)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        z = self.encoder(x).mean(dim=1)
        return self.head(self.fc_norm(z))


def vit_tiny(**kw) -> ViTEncoder:
    cfg = dict(img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0)
    cfg.update(kw)
    return ViTEncoder(**cfg)


def param_groups(model: nn.Module, weight_decay: float):
    """No weight decay on biases, LayerNorm weights and other 1-D params (MAE convention)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith("mask_token") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]
