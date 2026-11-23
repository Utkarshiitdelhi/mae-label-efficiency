"""The committed split files are class-balanced, disjoint and the sizes in the design doc."""
import json
import os

import pytest

SPLITS = os.path.join(os.path.dirname(__file__), "..", "splits")
EXPECTED = {(1, 0): (400, 100), (1, 1): (400, 100), (1, 2): (400, 100),
            (10, 0): (4500, 500), (100, 0): (45000, 5000)}


@pytest.mark.parametrize("frac,seed", list(EXPECTED))
def test_split(frac, seed):
    with open(os.path.join(SPLITS, f"frac{frac}_seed{seed}.json")) as f:
        s = json.load(f)
    n_tr, n_va = EXPECTED[(frac, seed)]
    assert len(s["train"]) == n_tr and len(s["val"]) == n_va
    assert len(set(s["train"]) | set(s["val"])) == n_tr + n_va          # disjoint, no dupes
    assert all(0 <= i < 50000 for i in s["train"] + s["val"])            # training set only
    assert len(set(s["train_per_class"])) == 1 and len(set(s["val_per_class"])) == 1  # balanced
    assert s["val_per_class"][0] >= 10


def test_one_percent_seeds_differ():
    sets = []
    for seed in range(3):
        with open(os.path.join(SPLITS, f"frac1_seed{seed}.json")) as f:
            sets.append(frozenset(json.load(f)["train"]))
    assert len(set(sets)) == 3


def test_config_numbers_parse_as_floats(tmp_path):
    """`--set lr=1e-3` must give a float (PyYAML alone returns the string '1e-3')."""
    from mae.utils import load_config
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "pretrain_m75.yaml"),
                      ["lr=1e-3", "run_name=m75_lr1e-3", "epochs=20"])
    assert cfg["lr"] == 1e-3 and isinstance(cfg["lr"], float)
    assert cfg["run_name"] == "m75_lr1e-3" and cfg["epochs"] == 20 and cfg["base_lr"] == 1.5e-4
    ft = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "finetune.yaml"))
    assert ft["lr_sweep"] == [5e-4, 1e-3, 2e-3] and ft["budgets"][1]["steps"] == 3000
