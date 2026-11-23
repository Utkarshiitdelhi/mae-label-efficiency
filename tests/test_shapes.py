import torch

from mae.data import gpu_random_resized_crop_flip
from mae.mae import MAE, patchify, unpatchify
from mae.pos_embed import get_2d_sincos_pos_embed
from mae.vit import ViTClassifier, vit_tiny


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def test_patchify_roundtrip():
    x = torch.randn(5, 3, 32, 32)
    p = patchify(x, 4)
    assert p.shape == (5, 64, 48)
    assert torch.equal(unpatchify(p, 4), x)


def test_patchify_matches_patch_embed_order():
    """Token k of patchify must be the same 4x4 region as token k of the conv embedding."""
    x = torch.zeros(1, 3, 32, 32)
    x[0, :, 4:8, 8:12] = 1.0                    # row 1, column 2 -> token 1*8 + 2 = 10
    assert patchify(x, 4)[0].abs().sum(-1).nonzero().flatten().tolist() == [10]
    enc = vit_tiny()
    with torch.no_grad():
        enc.patch_embed.proj.bias.zero_()
        tok = enc.patch_embed(x)[0].abs().sum(-1)
    assert tok.nonzero().flatten().tolist() == [10]


def test_param_counts_match_design_doc():
    m = MAE()
    enc, dec = n_params(m.encoder), n_params(m.decoder)
    assert round(enc / 1e6, 2) == 5.35        # ViT-Tiny, 12 blocks, width 192
    assert round(dec / 1e6, 2) == 0.82        # 4 blocks, width 128
    assert 0.14 < dec / enc < 0.16            # "about 15%"
    assert len(m.encoder.blocks) == 12 and len(m.decoder.blocks) == 4


def test_forward_shapes():
    m = MAE(mask_ratio=0.75)
    x = torch.randn(3, 3, 32, 32)
    loss, pred, mask = m(x)
    assert loss.ndim == 0 and pred.shape == (3, 64, 48) and mask.shape == (3, 64)
    clf = ViTClassifier(vit_tiny())
    assert clf(x).shape == (3, 10)


def test_encoder_sees_only_visible_tokens():
    enc = vit_tiny()
    ids_keep = torch.stack([torch.randperm(64)[:16] for _ in range(2)])
    assert enc(torch.randn(2, 3, 32, 32), ids_keep).shape == (2, 16, 192)


def test_pos_embed_fixed_and_distinct():
    pe = get_2d_sincos_pos_embed(192, 8)
    assert pe.shape == (64, 192)
    assert torch.unique(pe, dim=0).shape[0] == 64
    enc = vit_tiny()
    assert "pos_embed" not in dict(enc.named_parameters())   # not learned


def test_gpu_augment_identity_and_flip():
    x = torch.randn(4, 3, 32, 32)
    same = gpu_random_resized_crop_flip(x, scale=(1.0, 1.0), ratio=(1.0, 1.0), flip_p=0.0)
    assert torch.allclose(same, x, atol=1e-5)
    flipped = gpu_random_resized_crop_flip(x, scale=(1.0, 1.0), ratio=(1.0, 1.0), flip_p=1.0)
    assert torch.allclose(flipped, x.flip(-1), atol=1e-5)
    out = gpu_random_resized_crop_flip(x)
    assert out.shape == x.shape and not torch.allclose(out, x)
