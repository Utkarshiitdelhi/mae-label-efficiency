import json, sys
def nb(cells):
    out=[]
    for kind, src in cells:
        c={"cell_type":kind,"metadata":{},"source":src.strip("\n").splitlines(True)}
        if kind=="code": c.update(execution_count=None, outputs=[])
        out.append(c)
    return {"cells":out,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
            "language_info":{"name":"python"},"kaggle":{"accelerator":"nvidiaTeslaT4","isInternetEnabled":True}},
            "nbformat":4,"nbformat_minor":5}

SETUP = r'''
# --- Code: attach this repo as a Kaggle Dataset named "mae-code" (zip of the repo), OR set REPO_URL.
import glob, os, shutil, subprocess
REPO_URL = ""   # e.g. "https://github.com/<you>/mae-label-efficiency.git"
CODE = "/tmp/mae"   # outside /kaggle/working so code + smoke files are not saved as notebook output
shutil.rmtree(CODE, ignore_errors=True)
hits = glob.glob("/kaggle/input/**/scripts/run_block.sh", recursive=True)
if hits:                                   # repo attached as a dataset (any folder depth)
    shutil.copytree(os.path.dirname(os.path.dirname(hits[0])), CODE)
elif REPO_URL:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CODE], check=True)
else:
    raise SystemExit("Attach the code dataset or set REPO_URL")
os.chdir(CODE)
os.environ.update(DATA="/kaggle/working/data", RESULTS="/kaggle/working/results",
                  CKPT="/kaggle/working/checkpoints",
                  SEARCH="/kaggle/working/checkpoints /kaggle/input")
!nvidia-smi --query-gpu=name,memory.total --format=csv
!python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
'''

pre = nb([
 ("markdown", """
# MAE pretraining (thin wrapper)
All logic lives in `scripts/`. Settings: **Accelerator = GPU T4**, **Internet = On** (CIFAR-10 download).
Run with **Save Version → Save & Run All (Commit)** so it keeps going with the browser closed.
Outputs (`/kaggle/working/checkpoints`, `/kaggle/working/results`) become this notebook's output; attach that
output as an input to the next session.

| Session | `BLOCKS` |
| --- | --- |
| Day 1a (interactive is fine, ~20 min) | `test timing lrcheck` |
| Day 1b (background) | `pretrain_m75` (set `LR_PT` if lrcheck picked 1e-3) |
| Day 2 (background) | `pretrain_abl` (optionally add `pretrain_m75e200`) |

If a session stops early, attach its output and rerun the same block: it resumes from `ckpt_latest.pt`.
"""),
 ("code", 'BLOCKS = "pretrain_m75"\nLR_PT = ""   # "1e-3" only if the lrcheck block says so'),
 ("code", SETUP),
 ("code", 'if LR_PT: os.environ["LR_PT"] = LR_PT\n!bash scripts/run_block.sh {BLOCKS}'),
 ("code", '!ls -la /kaggle/working/checkpoints/* /kaggle/working/results/figures 2>/dev/null | head -50\n!tail -3 /kaggle/working/results/logs/pt_*.csv 2>/dev/null'),
])
ft = nb([
 ("markdown", """
# Fine-tuning, probes and figures (thin wrapper)
Attach as inputs: the code dataset (or set `REPO_URL`) and the **output of every earlier session**
(pretrain checkpoints and earlier `results/runs.csv`, which are merged automatically so LR choices carry over).
Settings: GPU T4, Internet On.

| Session | `BLOCKS` | Needs |
| --- | --- | --- |
| Day 2 gate (quick) | `probe_ep100` | `encoder_m75_ep100.pt` (exists from epoch 100 on) |
| Day 2 | `ft_low` | MAE-75 at 400 epochs |
| Day 3 | `ft_100` | MAE-75 at 400 epochs |
| Day 3 | `ft_ablation probes figures` | MAE-50/90 pretrains + the `ft_low` output |

If time runs short at 100%: set `LRS_100 = "5e-4 1e-3"` (design doc risk table).
"""),
 ("code", 'BLOCKS = "ft_low"\nLRS_100 = ""   # e.g. "5e-4 1e-3" to sweep only two LRs at 100%'),
 ("code", SETUP),
 ("code", 'if LRS_100: os.environ["LRS_100"] = LRS_100\n!bash scripts/run_block.sh {BLOCKS}'),
 ("code", '!tail -n 30 /kaggle/working/results/runs.csv\n!ls /kaggle/working/results/figures /kaggle/working/results/tables 2>/dev/null'),
])
json.dump(pre, open("notebooks/kaggle_pretrain.ipynb","w"), indent=1)
json.dump(ft, open("notebooks/kaggle_finetune.ipynb","w"), indent=1)
