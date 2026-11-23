import pytest
import torch

from mae.mae import MAE, masked_mse_loss, normalized_targets, patchify, unpatchify
from mae.vit import vit_tiny


def test_visible_target_pixels_do_not_change_loss():
    """CV bullet 1: the loss covers masked patches only."""
    torch.manual_seed(0)
    m = MAE().eval()
    imgs = torch.randn(4, 3, 32, 32)
    noise = torch.rand(4, 64)
    with torch.no_grad():
        _, pred, mask = m(imgs, noise=noise)
        base = masked_mse_loss(pred, normalized_targets(imgs), mask)
        # overwrite the *target* pixels of visible patches only (prediction is held fixed)
        patches = patchify(imgs)
        vis = (mask == 0).unsqueeze(-1)
        perturbed = unpatchify(torch.where(vis, torch.randn_like(patches) * 10, patches))
        same = masked_mse_loss(pred, normalized_targets(perturbed), mask)
        # sanity: changing a masked patch's target DOES change the loss
        masked_only = unpatchify(torch.where(~vis, torch.randn_like(patches) * 10, patches))
        diff = masked_mse_loss(pred, normalized_targets(masked_only), mask)
    assert torch.allclose(base, same)
    assert not torch.allclose(base, diff)


def test_no_gradient_from_visible_predictions():
    pred = torch.randn(2, 64, 48, requires_grad=True)
    target = torch.randn(2, 64, 48)
    mask = (torch.rand(2, 64) < 0.75).float()
    masked_mse_loss(pred, target, mask).backward()
    g = pred.grad.abs().sum(-1)
    assert torch.all(g[mask == 0] == 0) and torch.all(g[mask == 1] > 0)


def test_loss_formula():
    pred, target = torch.randn(2, 64, 48), torch.randn(2, 64, 48)
    mask = torch.zeros(2, 64)
    mask[:, :48] = 1
    expected = ((pred - target) ** 2)[:, :48].mean()
    assert torch.allclose(masked_mse_loss(pred, target, mask), expected)


def test_normalized_targets_per_patch():
    t = normalized_targets(torch.rand(3, 3, 32, 32) * 5 + 2)
    assert torch.allclose(t.mean(-1), torch.zeros(3, 64), atol=1e-4)
    assert torch.allclose(t.std(-1), torch.ones(3, 64), atol=1e-3)


def _natural_batch(n=64):
    """64 CIFAR training images if the dataset is on disk, else smooth synthetic images
    (pure per-pixel noise has no structure and is a poor stand-in for photos)."""
    import os
    import numpy as np
    from mae.data import CIFAR_MEAN, CIFAR_STD
    root = os.environ.get("MAE_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
    try:
        from mae.data import cifar_arrays
        x = torch.from_numpy(cifar_arrays(root, True, download=False)[0][:n]).permute(0, 3, 1, 2)
        x = x.float() / 255
    except Exception:
        g = torch.Generator().manual_seed(0)
        x = torch.nn.functional.interpolate(torch.rand(n, 3, 8, 8, generator=g), size=32,
                                            mode="bilinear", align_corners=False)
    m, s = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1), torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
    return (x - m) / s


@pytest.mark.slow
def test_overfit_one_batch():
    """Model, loss and optimiser are wired correctly: with a fixed mask, one batch of 64
    images is memorised (loss falls >10x, towards zero). ~4 min on 2 CPU cores, seconds on GPU."""
    torch.manual_seed(0)
    m = MAE(vit_tiny())
    imgs = _natural_batch(64)
    noise = torch.rand(64, 64)
    lr = 1e-3
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0)
    first = None
    for step in range(500):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / 20)
        loss, _, _ = m(imgs, noise=noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.1 * first, (first, loss.item())
