"""Fixed 2D sine-cosine position embeddings (as in MAE / MoCo v3).

Half of the channels encode the row index, half the column index. Within
each half, positions are encoded with sin/cos at geometrically spaced
frequencies, exactly like the Transformer's 1D encoding.
"""
import torch


def _sincos_1d(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """pos: (M,) float positions -> (M, embed_dim)."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64) / (embed_dim / 2.0)
    omega = 1.0 / (10000 ** omega)                     # (D/2,)
    out = pos.double()[:, None] * omega[None, :]       # (M, D/2)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Returns (grid_size * grid_size, embed_dim) float32, row-major order.

    Token k corresponds to row k // grid_size, column k % grid_size, which is
    the same order produced by the strided-conv patch embedding and by
    `patchify` in mae.py.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos"
    rows = torch.arange(grid_size).repeat_interleave(grid_size)   # 0,0,..,1,1,..
    cols = torch.arange(grid_size).repeat(grid_size)              # 0,1,..,0,1,..
    emb_h = _sincos_1d(embed_dim // 2, rows)
    emb_w = _sincos_1d(embed_dim // 2, cols)
    return torch.cat([emb_h, emb_w], dim=1).float()
