"""Masked Autoencoder (He et al., 2021) written from scratch.

Pipeline for one image batch:
  1. random_masking: per image, noise -> argsort -> keep first K tokens.
  2. encoder runs on the K visible tokens only (no mask tokens).
  3. decoder: project 192 -> 128, append mask tokens, unshuffle with ids_restore,
     add position embedding, 4 blocks, predict 48 pixel values per patch.
  4. loss: MSE against per-patch-normalised targets, on masked patches only.
"""
import torch
import torch.nn as nn

from .pos_embed import get_2d_sincos_pos_embed
from .vit import Block, ViTEncoder, init_weights, vit_tiny


def patchify(imgs: torch.Tensor, p: int = 4) -> torch.Tensor:
    """(N, 3, H, W) -> (N, L, p*p*3). Token order is row-major; inside a token the
    48 values are ordered (row-in-patch, col-in-patch, channel)."""
    N, C, H, W = imgs.shape
    assert H % p == 0 and W % p == 0
    h, w = H // p, W // p
    x = imgs.reshape(N, C, h, p, w, p)
    x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(N, h * w, p * p * C)


def unpatchify(x: torch.Tensor, p: int = 4, c: int = 3) -> torch.Tensor:
    """Inverse of patchify: (N, L, p*p*c) -> (N, c, H, W)."""
    N, L, _ = x.shape
    h = w = int(L ** 0.5)
    assert h * w == L
    x = x.reshape(N, h, w, p, p, c)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(N, c, h * p, w * p)


def num_visible(num_patches: int, mask_ratio: float) -> int:
    return int(num_patches * (1 - mask_ratio) + 1e-6)   # 64 -> 32 / 16 / 6


def random_masking(N: int, L: int, mask_ratio: float, device=None, noise=None):
    """Uniform random masking, independently per image, with no Python loops.

    Returns
      ids_keep    (N, K)  indices of visible tokens (in shuffled order)
      mask        (N, L)  1 = masked (hidden), 0 = visible, in original token order
      ids_restore (N, L)  gather index that undoes the shuffle
    """
    K = num_visible(L, mask_ratio)
    if noise is None:
        noise = torch.rand(N, L, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)          # ascending: small noise is kept
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :K]
    mask = torch.ones(N, L, device=noise.device)
    mask[:, :K] = 0
    mask = torch.gather(mask, 1, ids_restore)          # back to original order
    return ids_keep, mask, ids_restore


def normalized_targets(imgs: torch.Tensor, p: int = 4, eps: float = 1e-6) -> torch.Tensor:
    """Per-patch normalised pixels: (p - mean(p)) / sqrt(var(p) + eps)."""
    target = patchify(imgs, p)
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)            # unbiased, as in the MAE reference code
    return (target - mean) / (var + eps) ** 0.5


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L = 1/|M| * sum_{i in M} 1/48 * ||pred_i - target_i||^2. Visible patches get weight 0."""
    per_patch = ((pred - target) ** 2).mean(dim=-1)   # (N, L), mean over 48 values
    return (per_patch * mask).sum() / mask.sum()


class MAEDecoder(nn.Module):
    def __init__(self, num_patches=64, enc_dim=192, dec_dim=128, depth=4, num_heads=4,
                 mlp_ratio=4.0, patch_dim=48):
        super().__init__()
        self.embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        grid = int(num_patches ** 0.5)
        pe = get_2d_sincos_pos_embed(dec_dim, grid)
        self.register_buffer("pos_embed", pe.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList([Block(dec_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dec_dim, eps=1e-6)
        self.pred = nn.Linear(dec_dim, patch_dim)
        self.apply(init_weights)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, latent, ids_restore):
        x = self.embed(latent)                                     # (N, K, 128)
        N, K, D = x.shape
        L = ids_restore.shape[1]
        x = torch.cat([x, self.mask_token.expand(N, L - K, D)], dim=1)   # (N, L, 128) shuffled
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))  # unshuffle
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.pred(self.norm(x))                             # (N, L, 48)


class MAE(nn.Module):
    def __init__(self, encoder: ViTEncoder = None, mask_ratio=0.75, dec_dim=128, dec_depth=4,
                 dec_heads=4, norm_pix_loss=True):
        super().__init__()
        self.encoder = encoder if encoder is not None else vit_tiny()
        self.patch_size = self.encoder.patch_embed.patch_size
        L = self.encoder.patch_embed.num_patches
        self.decoder = MAEDecoder(L, self.encoder.embed_dim, dec_dim, dec_depth, dec_heads,
                                  patch_dim=self.patch_size ** 2 * 3)
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

    def targets(self, imgs):
        if self.norm_pix_loss:
            return normalized_targets(imgs, self.patch_size)
        return patchify(imgs, self.patch_size)

    def forward(self, imgs, mask_ratio=None, noise=None):
        """Returns (loss, pred, mask). pred is (N, L, 48) in normalised-target space."""
        mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio
        N = imgs.shape[0]
        L = self.encoder.patch_embed.num_patches
        ids_keep, mask, ids_restore = random_masking(N, L, mask_ratio, imgs.device, noise)
        latent = self.encoder(imgs, ids_keep)
        pred = self.decoder(latent, ids_restore)
        loss = masked_mse_loss(pred.float(), self.targets(imgs).float(), mask)
        return loss, pred, mask


@torch.no_grad()
def reconstruct(model: MAE, imgs, mask_ratio=None, noise=None):
    """For visualisation: returns (masked_input, prediction_pasted, mask_img).

    The prediction is in normalised space, so it is mapped back using each target
    patch's own mean/std, then visible patches are pasted from the input."""
    loss, pred, mask = model(imgs, mask_ratio, noise)
    p = model.patch_size
    target = patchify(imgs, p)
    if model.norm_pix_loss:
        mean = target.mean(-1, keepdim=True)
        std = (target.var(-1, keepdim=True) + 1e-6) ** 0.5
        pred = pred.float() * std + mean
    m = mask.unsqueeze(-1)
    pasted = unpatchify(target * (1 - m) + pred * m, p)
    masked = unpatchify(target * (1 - m), p)
    return masked, pasted, loss
