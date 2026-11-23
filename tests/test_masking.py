import pytest
import torch

from mae.mae import random_masking


@pytest.mark.parametrize("ratio,visible", [(0.5, 32), (0.75, 16), (0.9, 6)])
def test_keeps_exact_number_of_tokens(ratio, visible):
    ids_keep, mask, ids_restore = random_masking(8, 64, ratio)
    assert ids_keep.shape == (8, visible)
    assert torch.all(mask.sum(1) == 64 - visible)
    # kept ids are exactly the unmasked positions
    for i in range(8):
        assert set(ids_keep[i].tolist()) == set((mask[i] == 0).nonzero().flatten().tolist())


def test_restore_undoes_shuffle():
    N, L = 4, 64
    tokens = torch.arange(L).float().repeat(N, 1).unsqueeze(-1)        # token value = position
    noise = torch.rand(N, L)
    ids_shuffle = torch.argsort(noise, dim=1)
    _, _, ids_restore = random_masking(N, L, 0.75, noise=noise)
    shuffled = torch.gather(tokens, 1, ids_shuffle.unsqueeze(-1))
    restored = torch.gather(shuffled, 1, ids_restore.unsqueeze(-1))
    assert torch.equal(restored, tokens)


def test_masks_differ_across_images_and_calls():
    _, m1, _ = random_masking(16, 64, 0.75)
    _, m2, _ = random_masking(16, 64, 0.75)
    assert torch.unique(m1, dim=0).shape[0] == 16       # per image, not shared
    assert not torch.equal(m1, m2)                      # fresh each call


def test_masking_is_roughly_uniform():
    _, m, _ = random_masking(4000, 64, 0.75)
    freq = m.mean(0)                                    # P(position masked)
    assert torch.all((freq - 0.75).abs() < 0.04)        # no centre / position bias
