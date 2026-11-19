"""
Noisy/clean separability of per-sample signals during InfoBatch training.
CIFAR-100, symmetric or asymmetric label noise (default sym 40%), ResNet-18, max lr 0.05.

Every --every epochs, all 50k train images are scored on the clean (unaugmented) view and
AUROC(noisy vs clean) is computed on all samples and on the high-loss half (hard-vs-noisy regime).

Signals (higher = more suspect):
  loss      CE with given label                                    (InfoBatch score, label-aware)
  MSP       -max softmax              [Hendrycks & Gimpel, ICLR 2017]  (label-free)
  energy    -logsumexp(z)             [Liu et al., NeurIPS 2020]       (label-free)
  margin    max_{k!=y} p_k - p_y                                       (label-aware)
  knn       kNN label disagreement in feature space [Bahri et al., ICML 2020] (label-aware)
  centroid  cos to nearest other class centroid - cos to own [FINE-style, Kim et al., NeurIPS 2021]
  knn_dist  k-th NN distance in feature space [Sun et al., ICML 2022]  (label-free)
  cen_norm       centroid margin / std of the 5 largest centroid similarities (confidence-normalized)
  cen_iter       centroid margin with iteratively cleaned centroids (recomputed on samples with margin < 0)
  cen_iter_norm  cen_iter margin / std of its 5 largest centroid similarities

Under symmetric noise the flip is independent of x, so a score of x alone from a model that never
saw the labels has expected AUROC 0.5. Deviations come only from training on the noisy labels.

Outputs: outputs/uncertainty_signals/<noise><eta>_<ib|full>[_settings]_s<seed>/.
Matching runs with all four result files are skipped. --tag appends a suffix to the output folder.

Usage: python uncertainty_signals.py [--noise sym|asym] [--eta 0.4] [--epochs 200] [--every 10] [--full] [--seed 0] [--tag v2]
"""
import os
import sys
import csv
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from filelock import FileLock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.data import load, batches, norm, add_noise
from utils.results import atomic_write, output_dir, percent_slug, read_summary, write_summary

NAMES = ['loss', 'MSP', 'energy', 'margin', 'knn', 'centroid', 'knn_dist', 'cen_norm', 'cen_iter', 'cen_iter_norm']


def seed_all(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def resnet18(K):
    # CIFAR stem: 3x3 stride-1 conv1, no maxpool (official InfoBatch examples/model.py)
    m = torchvision.models.resnet18(num_classes=K)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def ib_prune(H, r):
    """
    InfoBatch soft pruning Eq. (3) + expectation rescaling Eq. (7), as official infobatch.py prune().
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


def cen_margin(Fe, y, K, it=0, top=5):
    """
    Centroid margin m = max_{k!=y} cos(f, c_k) - cos(f, c_y) and its confidence-normalized form m / std(top-5 cos).
    it > 0: centroids recomputed it times from samples with m < 0 (classes left empty keep the full-data centroid).
    """
    C0 = F.normalize(torch.zeros(K, Fe.size(1), device=Fe.device).index_add_(0, y, Fe), dim=1)
    C, w = C0, None
    for i in range(it + 1):
        if w is not None:
            Cw = torch.zeros(K, Fe.size(1), device=Fe.device).index_add_(0, y[w], Fe[w])
            ok = torch.bincount(y[w], minlength=K) > 0
            C = torch.where(ok[:, None], F.normalize(Cw, dim=1), C0)
        sc = Fe @ C.T
        m = sc.scatter(1, y[:, None], -2).max(1).values - sc.gather(1, y[:, None]).squeeze(1)
        w = m < 0
    return m, m / (sc.topk(top, 1).values.std(1) + 1e-6)


@torch.no_grad()
def signals(net, D, dev, k=20):
    """
    (N, 10) scores in NAMES order on the clean train view, against the given (noisy) labels.
    """
    net.eval()
    amp = dev.type == 'cuda'
    body = nn.Sequential(*list(net.children())[:-1])
    Z, Fe = [], []
    for s in range(0, len(D['Xtr']), 1024):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            f = body(norm(D, D['Xtr'][s:s + 1024]).contiguous(memory_format=torch.channels_last)).flatten(1)
            z = net.fc(f)
        Z.append(z.float())
        Fe.append(F.normalize(f.float(), dim=1))
    Z, Fe, y = torch.cat(Z), torch.cat(Fe), D['Ytr']

    p = Z.softmax(1)
    py = p.gather(1, y[:, None]).squeeze(1)

    # Largest competitor prob, given label excluded
    po = p.scatter(1, y[:, None], 0).max(1).values

    # Class centroids from given labels: plain, confidence-normalized, iteratively cleaned (5 passes)
    cen, cen_n = cen_margin(Fe, y, D['K'])
    cen_i, cen_in = cen_margin(Fe, y, D['K'], it=4)

    # kNN on cosine similarity, self excluded
    knn, kd = torch.empty(len(y), device=dev), torch.empty(len(y), device=dev)
    for s in range(0, len(y), 1024):
        sim = Fe[s:s + 1024] @ Fe.T
        n = sim.size(0)
        sim[torch.arange(n), torch.arange(s, s + n)] = -2
        v, nb = sim.topk(k, 1)
        knn[s:s + n] = 1 - (y[nb] == y[s:s + n, None]).float().mean(1)
        kd[s:s + n] = -v[:, -1]

    return torch.stack([
        F.cross_entropy(Z, y, reduction='none'),
        -p.max(1).values,
        -torch.logsumexp(Z, 1),
        po - py,
        knn, cen, kd, cen_n, cen_i, cen_in
    ], 1).cpu()


def auroc(s, t):
    # Mann-Whitney U: P(score_noisy > score_clean); tiny jitter breaks ties in discrete kNN scores
    s = s + 1e-7 * torch.rand_like(s)
    r = s.argsort().argsort().double() + 1
    n1, n0 = t.sum().double(), (~t).sum().double()
    return ((r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


def run(D, ib, dev, snap, C=200, max_lr=0.05, r=0.5, delta=0.875, ls=0.1, bs=128, seed=0):
    """
    InfoBatch (or full-data) training; signals() snapshot after each epoch in snap.
    """
    seed_all(seed)
    N = len(D['Xtr'])
    amp = dev.type == 'cuda'
    net = resnet18(D['K']).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(net.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)

    # Score init 3 as official infobatch.py
    H = torch.full((N,), 3.0, device=dev)
    log, sig, seen, tt = [], {}, 0, 0.0

    for e in range(C):
        # Annealing, InfoBatch Sec. 2.4 Eq. (10)
        if ib and e + 1 <= delta * C:
            idx, w = ib_prune(H, r)
        else:
            idx, w = torch.randperm(N, device=dev), torch.ones(N, device=dev)
        S = math.ceil(len(idx) / bs)

        # OneCycle rebuilt each epoch (official cifar_example.py)
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

            # InfoBatch Eq. (4) score update
            H[i] = l.detach().float()

            # InfoBatch Eq. (7) rescaling, batch mean
            loss = (l * w[i]).mean()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            tl += loss.detach() * len(i)

        if amp:
            torch.cuda.synchronize()
        tt += time.time() - t1

        seen += len(idx)
        acc = evaluate(net, D, dev)
        log.append(dict(e=e, n=len(idx), loss=tl.item() / len(idx), acc=acc, t=tt))

        # Snapshot outside timed region
        if e in snap:
            sig[e] = signals(net, D, dev)

        print(f'{e:3d}  |S|/N {len(idx)/N:.3f}  lr {sch.get_last_lr()[0]:.4f}  '
              f'loss {log[-1]["loss"]:.4f}  acc {acc:.2f}  {tt:.0f}s', flush=True)

    return log, sig, 1 - seen / (C * N)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eta', type=float, default=0.4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--every', type=int, default=10)
    p.add_argument('--full', action='store_true', help='full-data training instead of InfoBatch')
    p.add_argument('--noise', choices=['sym', 'asym'], default='sym')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tag', default='')
    a = p.parse_args()
    if not 0 <= a.eta <= 1 or min(a.epochs, a.every) < 1:
        p.error('--eta must be in [0, 1], and --epochs and --every must be positive')
    if '/' in a.tag or '\\' in a.tag:
        p.error('--tag must be a filename suffix')

    mode = 'full' if a.full else 'infobatch'
    epochs_tag = f'_e{a.epochs}' if a.epochs != 200 else ''
    every_tag = f'_every{a.every}' if a.every != 10 else ''
    tag = f'{a.noise}{percent_slug(a.eta)}_{"full" if a.full else "ib"}{epochs_tag}{every_tag}_s{a.seed}'
    if a.tag:
        tag += f'_{a.tag}'
    out = output_dir() / 'uncertainty_signals' / tag
    summary_path = out / 'summary.json'
    settings = dict(noise=a.noise, eta=a.eta, seed=a.seed, method=mode, epochs=a.epochs, every=a.every)
    log_path, signals_path, csv_path = (out / name for name in ('log.pt', 'signals.pt', 'auroc.csv'))
    summary = {'every': 10, **read_summary(summary_path)}
    if all(summary.get(key) == value for key, value in settings.items()) and all(
        path.is_file() and path.stat().st_size for path in (log_path, signals_path, csv_path)
    ):
        print(f'{tag}: matching diagnostic outputs found, skipped', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False

    D = load('cifar100', dev)
    t = add_noise(D, a.noise, a.eta, seed=0).cpu()
    print(f'{tag}: noisy fraction {t.float().mean():.4f}', flush=True)

    # Snapshots every --every epochs, plus last pruned epoch and final epoch
    snap = set(range(0, a.epochs, a.every)) | {int(0.875 * a.epochs) - 1, a.epochs - 1}
    log, sig, saved = run(D, not a.full, dev, snap, C=a.epochs, seed=a.seed)

    # AUROC table: all samples, and high-loss half (loss >= median)
    rows = []
    for e in sorted(sig):
        s = sig[e]
        h = s[:, 0] >= s[:, 0].median()
        row = dict(epoch=e)
        for j, n in enumerate(NAMES):
            row[f'{n}_all'] = round(auroc(s[:, j], t), 4)
            row[f'{n}_high'] = round(auroc(s[h, j], t[h]), 4)
        rows.append(row)
        print(f'epoch {e}: ' + '  '.join(f'{n} {row[n + "_all"]:.3f}/{row[n + "_high"]:.3f}' for n in NAMES), flush=True)

    def write_csv(path):
        with path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    signals_data = dict(sig={e: s.half() for e, s in sig.items()}, noisy=t, names=NAMES)
    record = dict(**settings, final=log[-1]['acc'], best=max(d['acc'] for d in log),
                  saved_fwd=round(100 * saved, 2), train_s=round(log[-1]['t'], 1))
    with FileLock(str(summary_path) + '.lock'):
        summary_path.unlink(missing_ok=True)
        atomic_write(log_path, lambda temp: torch.save(log, temp))
        atomic_write(signals_path, lambda temp: torch.save(signals_data, temp))
        atomic_write(csv_path, write_csv)
        write_summary(summary_path, record)


if __name__ == '__main__':
    main()
