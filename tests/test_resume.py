"""Save at epoch 3, resume, and compare to an uninterrupted run (tiny model, CPU)."""
import numpy as np
import torch

from mae.engine import pretrain_mae

TINY = dict(run_name="tiny", mask_ratio=0.75, epochs=6, batch_size=32, base_lr=1.5e-4,
            warmup_epochs=1, weight_decay=0.05, ckpt_every=3, encoder_save_every=3,
            recon_every=0, amp=False, seed=0, dec_dim=32, dec_depth=1, dec_heads=2,
            encoder=dict(embed_dim=48, depth=2, num_heads=3))


def _images(n=128):
    return np.random.default_rng(0).integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)


def test_resume_is_exact(tmp_path):
    dev = torch.device("cpu")
    imgs = _images()
    full = pretrain_mae(dict(TINY), str(tmp_path / "a"), str(tmp_path / "ra"), dev,
                        train_images=imgs, log_every_epoch=False)

    pretrain_mae(dict(TINY), str(tmp_path / "b"), str(tmp_path / "rb"), dev, train_images=imgs,
                 stop_after_epoch=3, log_every_epoch=False)              # "session killed"
    torch.manual_seed(12345)                                             # disturb global RNG
    resumed = pretrain_mae(dict(TINY), str(tmp_path / "b"), str(tmp_path / "rb"), dev,
                           train_images=imgs, log_every_epoch=False)

    a, b = full.state_dict(), resumed.state_dict()
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k
    # the run is logged once, and the per-epoch log has 6 rows in both cases
    for d in ("ra", "rb"):
        assert len(open(tmp_path / d / "runs.csv").read().strip().splitlines()) == 2
        assert len(open(tmp_path / d / "logs" / "pt_tiny.csv").read().strip().splitlines()) == 7


def test_rng_state_survives_device_move():
    """Regression (found on a Kaggle T4): loading a checkpoint with map_location=cuda moved the
    saved RNG ByteTensor off the CPU and set_rng_state raised TypeError. Simulate the move by
    giving set_rng_state a non-CPU-contiguous/converted tensor: it must call .cpu() itself."""
    from mae.utils import rng_state, set_rng_state

    class Moved:                      # stands in for a CUDA tensor on a CPU-only machine
        def __init__(self, t):
            self.t = t

        def cpu(self):
            return self.t

    s = rng_state()
    a = torch.rand(3)
    s["torch"] = Moved(s["torch"])
    set_rng_state(s)
    assert torch.equal(torch.rand(3), a)
