"""
InfoBatch reproduction (Qin et al., ICLR 2024), ResNet-18 on CIFAR-10/100.

Paper targets (Tab. 3, R-18): CIFAR-10 full 95.6 / InfoBatch 95.5; CIFAR-100 full 78.2 / InfoBatch 78.2.
Where the paper is silent we follow the official repo (examples/cifar_example.py, infobatch/infobatch.py).

Our deviations from the official code:
  fp16 autocast + channels_last; torchvision R-18 init; GPU crop-then-flip;
  OneCycle rebuilt with the exact current epoch length (repo uses previous epoch length);
  the unreleased "r=0.75 on lowest 20% loss" rule for CIFAR-100 (Sec. 3) is not used.

Outputs: outputs/infobatch_repro/<run>[_e<epochs>].pt + summary.json. Complete runs are skipped.

Usage: python repro.py [--runs c10_full c10_ib ...] [--epochs 200]
"""
import os
import sys
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.data import load, batches
from utils.results import completed_run, output_dir, save_indexed_run

# name: (dataset, InfoBatch on/off, OneCycle max lr)
RUNS = {
    'c10_full': ('cifar10', False, 0.2),          # Appendix A: SGD max lr 0.2, bs 128
    'c10_ib': ('cifar10', True, 0.2),
    'c100_full': ('cifar100', False, 0.03),       # Appendix A: baseline max lr 0.03
    'c100_ib': ('cifar100', True, 0.05),          # Appendix A: InfoBatch max lr 0.05
    'c100_full_lr05': ('cifar100', False, 0.05),  # ours: same-lr control for c100_ib
}


def seed_all(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def resnet18(K):
    # CIFAR stem: 3x3 stride-1 conv1, no maxpool (official examples/model.py)
    m = torchvision.models.resnet18(num_classes=K)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def ib_prune(H, r):
    """
    Soft pruning Eq. (3): P_t(z) = r if H_t(z) < mean(H_t), else 0.
    Expectation rescaling Eq. (7): gamma_t(z) = 1/(1-r) on kept small-score samples.
    Kept count is int((1-r)|D_small|) without replacement, as official infobatch.py prune().
    """
    w = torch.ones_like(H)
    m = H < H.mean()
    small = m.nonzero().squeeze(1)
    keep = small[torch.randperm(len(small), device=H.device)[:int((1 - r) * len(small))]]
    w[keep] = 1 / (1 - r)
    idx = torch.cat([(~m).nonzero().squeeze(1), keep])
    return idx[torch.randperm(len(idx), device=H.device)], w


@torch.no_grad()
def evaluate(net, D, dev):
    net.eval()
    amp = dev.type == 'cuda'
    c = 0
    for x, y, _ in batches(D, torch.arange(len(D['Xte']), device=dev), 512, train=False):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            c += (net(x.contiguous(memory_format=torch.channels_last)).argmax(1) == y).sum()
    return 100 * c.item() / len(D['Xte'])


def run(D, ib, max_lr, dev, C=200, r=0.5, delta=0.875, ls=0.1, bs=128, seed=0):
    """
    r, delta defaults from Sec. 3 implementation details; ls, OneCycle args from official cifar_example.py.
    """
    seed_all(seed)
    N = len(D['Xtr'])
    amp = dev.type == 'cuda'
    net = resnet18(D['K']).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(net.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)

    # Score init 3 as official infobatch.py (paper Sec. 2.3: 1); epoch 0 prunes nothing either way
    H = torch.full((N,), 3.0, device=dev)
    log, seen, tt = [], 0, 0.0

    for e in range(C):
        # Annealing Sec. 2.4, Eq. (10): prune only in the first delta*C epochs, full data after
        if ib and e + 1 <= delta * C:
            idx, w = ib_prune(H, r)
        else:
            idx, w = torch.randperm(N, device=dev), torch.ones(N, device=dev)
        S = math.ceil(len(idx) / bs)

        # OneCycle rebuilt each epoch so LR position tracks epochs, not steps (official cifar_example.py)
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr, epochs=C, steps_per_epoch=S, div_factor=25,
            final_div_factor=1e4, pct_start=0.3, last_epoch=e * S - 1)

        net.train()
        tl = torch.zeros((), device=dev)
        if amp:
            torch.cuda.synchronize()
        t1 = time.time()
        for x, y, i in batches(D, idx, bs):
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                out = net(x.contiguous(memory_format=torch.channels_last))
                l = F.cross_entropy(out, y, reduction='none', label_smoothing=ls)

            # Eq. (4): H_{t+1}(z) = L(z) for z in S_t; pruned samples keep H_t(z)
            H[i] = l.detach().float()

            # Eq. (7) via per-sample weights; batch mean as official update(): values.mul_(weights).mean()
            loss = (l * w[i]).mean()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            tl += loss.detach() * len(i)

        # Train-only wall clock, as total_time in official cifar_example.py
        if amp:
            torch.cuda.synchronize()
        tt += time.time() - t1

        seen += len(idx)
        acc = evaluate(net, D, dev)
        log.append(dict(e=e, n=len(idx), loss=tl.item() / len(idx), acc=acc, t=tt))
        print(f'{e:3d}  |S|/N {len(idx)/N:.3f}  lr {sch.get_last_lr()[0]:.4f}  '
              f'loss {log[-1]["loss"]:.4f}  acc {acc:.2f}  {tt:.0f}s', flush=True)

    saved = 1 - seen / (C * N)
    return log, saved


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', choices=RUNS, default=list(RUNS))
    p.add_argument('--epochs', type=int, default=200)
    a = p.parse_args()
    if a.epochs < 1:
        p.error('--epochs must be positive')

    out = output_dir() / 'infobatch_repro'
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Official repo leaves cudnn.benchmark off; on, it re-tunes for every new last-batch shape
    torch.backends.cudnn.benchmark = False

    data = {}

    for method in a.runs:
        ds, ib, lr = RUNS[method]
        name = method if a.epochs == 200 else f'{method}_e{a.epochs}'
        path = out / f'{name}.pt'
        settings = dict(dataset=ds, infobatch=ib, max_lr=lr, epochs=a.epochs)
        if completed_run(path, settings):
            print(f'{name}: matching log and summary found, skipped', flush=True)
            continue
        if ds not in data:
            data[ds] = load(ds, dev)
        print(f'\n{name}: {ds}  infobatch={ib}  max_lr={lr}  epochs={a.epochs}', flush=True)

        log, saved = run(data[ds], ib, lr, dev, C=a.epochs)
        record = dict(
            **settings,
            final=log[-1]['acc'], best=max(d['acc'] for d in log),
            saved_fwd=round(100 * saved, 2), train_s=round(log[-1]['t'], 1))
        save_indexed_run(path, log, record)
        print(f'{name}: final {record["final"]:.2f}  best {record["best"]:.2f}  '
              f'training examples saved {record["saved_fwd"]:.1f}%  train {record["train_s"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
