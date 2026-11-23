"""CIFAR-10 loading, label-subset splits and transforms.

Pretraining keeps the whole uint8 training set on the GPU and augments there
(random-resized-crop + flip via one batched grid_sample), so a T4 is never
waiting on 2-4 Kaggle CPU cores. Fine-tuning uses torchvision transforms
(RandAugment) in DataLoader workers; the labelled subsets are small.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def load_cifar(root: str, train: bool, download: bool = True):
    """Returns the torchvision dataset object (PIL images, .data uint8 NHWC, .targets)."""
    return torchvision.datasets.CIFAR10(root, train=train, download=download)


def cifar_arrays(root: str, train: bool, download: bool = True):
    ds = load_cifar(root, train, download)
    return ds.data, np.asarray(ds.targets, dtype=np.int64)     # (N,32,32,3) uint8, (N,)


# ---------------------------------------------------------------- splits
def split_path(splits_dir: str, fraction: int, seed: int) -> str:
    return os.path.join(splits_dir, f"frac{fraction}_seed{seed}.json")


def load_split(splits_dir: str, fraction: int, seed: int):
    with open(split_path(splits_dir, fraction, seed)) as f:
        s = json.load(f)
    return np.asarray(s["train"], dtype=np.int64), np.asarray(s["val"], dtype=np.int64)


def make_split(labels: np.ndarray, fraction: int, seed: int, num_classes: int = 10,
               val_frac: float = 0.1, min_val_per_class: int = 10):
    """Class-balanced subset of `fraction`% of the training set, then a class-balanced
    val split carved out of it (10% of the subset, at least 10 per class)."""
    rng = np.random.default_rng([seed, fraction])
    n_total = len(labels)
    per_class = int(round(n_total * fraction / 100 / num_classes))
    n_val = max(min_val_per_class, int(round(per_class * val_frac)))
    train, val = [], []
    for c in range(num_classes):
        idx = np.flatnonzero(labels == c)
        chosen = rng.permutation(idx)[:per_class]
        val.extend(chosen[:n_val].tolist())
        train.extend(chosen[n_val:].tolist())
    return sorted(int(i) for i in train), sorted(int(i) for i in val)


# ---------------------------------------------------------------- GPU augment (pretraining)
def gpu_random_resized_crop_flip(x: torch.Tensor, scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3),
                                 flip_p: float = 0.5, generator=None) -> torch.Tensor:
    """Batched RandomResizedCrop(32) + horizontal flip with one bilinear grid_sample.

    x: (N, C, H, W) float. Each image gets its own crop: area fraction ~ U(scale),
    log aspect ratio ~ U(log ratio), centre uniform over valid positions. The crop is
    expressed as an affine map in normalised [-1, 1] coordinates.
    """
    N, _, H, W = x.shape
    dev = x.device
    rand = lambda *s: torch.rand(*s, device=dev, generator=generator)
    area = scale[0] + (scale[1] - scale[0]) * rand(N)
    logr = math.log(ratio[0]) + (math.log(ratio[1]) - math.log(ratio[0])) * rand(N)
    r = torch.exp(logr)
    w = torch.sqrt(area * r).clamp(max=1.0)        # crop width as a fraction of W
    h = torch.sqrt(area / r).clamp(max=1.0)
    cx = (rand(N) * 2 - 1) * (1 - w)               # centre so the crop stays inside
    cy = (rand(N) * 2 - 1) * (1 - h)
    flip = torch.where(rand(N) < flip_p, -1.0, 1.0)
    theta = torch.zeros(N, 2, 3, device=dev)
    theta[:, 0, 0] = w * flip
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = h
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, (N, x.shape[1], H, W), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


class GPUImageBank:
    """Whole uint8 image set on device; yields normalised, augmented float batches."""

    def __init__(self, images_nhwc: np.ndarray, device, mean=CIFAR_MEAN, std=CIFAR_STD):
        self.x = torch.from_numpy(images_nhwc).permute(0, 3, 1, 2).contiguous().to(device)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.n = self.x.shape[0]

    def normalise(self, xb_uint8):
        return (xb_uint8.float() / 255.0 - self.mean) / self.std

    def epoch_batches(self, batch_size: int, epoch_seed: int, augment: bool = True, drop_last=True):
        """Deterministic per-epoch order (seeded by epoch) so resuming reproduces data order."""
        g = torch.Generator().manual_seed(epoch_seed)
        perm = torch.randperm(self.n, generator=g).to(self.x.device)
        stop = self.n - self.n % batch_size if drop_last else self.n
        for i in range(0, stop, batch_size):
            xb = self.normalise(self.x[perm[i:i + batch_size]])
            if augment:
                xb = gpu_random_resized_crop_flip(xb)
            yield xb


# ---------------------------------------------------------------- torchvision transforms
def finetune_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def eval_transform():
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


class IndexedCIFAR(torch.utils.data.Dataset):
    """A subset of a CIFAR split given by indices, with a transform."""

    def __init__(self, base, indices, transform):
        self.base, self.indices, self.transform = base, np.asarray(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, y = self.base[int(self.indices[i])]
        return self.transform(img), y


def infinite_loader(loader):
    """Cycle a DataLoader forever (step-based training)."""
    while True:
        for batch in loader:
            yield batch
