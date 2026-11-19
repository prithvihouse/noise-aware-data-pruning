"""
InfoBatch with a noisy-label filter from the model's own penultimate features.
CIFAR-100, symmetric label noise (default 40%), ResNet-18, max lr 0.05.

From epoch --f0, every --fe epochs, all train samples are re-scored on the clean view and
flagged samples are excluded from InfoBatch pruning, rescaling and annealing until the next refresh.
Label noise is fixed by seed 0 for every training seed, so seeds vary init/order only.
Thresholds need no noise-rate estimate:
  centroid > 0     features closer to another class centroid than to the given class [FINE-style]
  knn      > 0.5   majority of 20 feature-space neighbours carry a different label [Bahri et al.]
  oracle           drop exactly the flipped samples (uses privileged labels)
  clip             CLIP zero-shot top-1 disagrees with the given label (external-model filter)
  cen_iter         centroid > 0 with iteratively cleaned centroids: recomputed 4 times from samples with
                   margin < 0 (empty classes keep the full-data centroid). Targets centroid contamination
                   under class-dependent (asymmetric) noise.

Baseline yang: adaptation of the selection rule of Yang et al., ICML 2025:
  p_sel = p_rho * p_con [their Eq. 3]; p_rho = min-max mean distance to k NN in our model's features
  [their Eq. 1], p_con = min-max CLIP cos(image, "A photo of a {c}") for the given label [their Eq. 2,
  prompt and ViT-B/32 from their ICLR 2025 code]. Top --keep fraction per epoch, InfoBatch annealing.
  Adaptation: frozen CLIP, the shared training augmentation, and deterministic top-k selection.
Baseline yang_sel: as yang, but selection stays active in the annealing epochs (full data is never
  reintroduced), matching how the feature filters keep excluding flagged samples during annealing.

Baseline ib_align: AlignPrune + InfoBatch (Qin et al., CVPR 2026 Findings), ported from the official code
  (github.com/leonqin430/AlignPrune, alignprune/pruners.py, scores.py). The InfoBatch loss score is replaced by
  the Dynamic Alignment Score: Pearson correlation between a sample's training losses over the last --win
  epochs and the mean loss of a clean reference set over the same epochs. Pruning rule, 1/(1-r) rescaling and
  full-data annealing are InfoBatch's, unchanged. As in their main_cifar.py, --ref training images (seed 0,
  fixed like the noise) are held out with their clean labels and never trained on. Ring buffers start at 6.0,
  DAS at 1.0, only samples trained in an epoch get a new DAS. Trace logs the low-DAS candidate set
  (pruned with prob r) every --fe epochs, not hard drops.

Training, online filtering, and CLIP setup are timed separately. saved_fwd records the percentage
reduction in training examples processed relative to --full (200) epochs on the eligible pool.

Outputs: outputs/feature_filters/<noise><eta>/<run>[_settings]_s<seed>.pt + summary.json.
Set INFOBATCH_OUTPUT_DIR for separate runs. Matching, complete runs are skipped.

Usage: python feature_filters.py [--runs full ib ib_oracle ib_centroid ib_knn ib_align] [--noise sym|asym]
                                 [--eta 0.4] [--epochs 200] [--f0 40] [--fe 5] [--seed 0] [--ref 1000] [--win 25]
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
from filelock import FileLock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from utils.data import load, batches, norm, add_noise
from utils.results import atomic_write, completed_run, output_dir, percent_slug, save_indexed_run


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


def ib_prune(H, r, keep):
    """
    InfoBatch Eq. (3) + Eq. (7) restricted to the kept set; threshold = mean of H over kept samples.
    With keep all True this is identical to official infobatch.py prune().
    """
    w = torch.ones_like(H)
    m = (H < H[keep].mean()) & keep
    small = m.nonzero().squeeze(1)
    sel = small[torch.randperm(len(small), device=H.device)[:int((1 - r) * len(small))]]
    w[sel] = 1 / (1 - r)
    idx = torch.cat([(keep & ~m).nonzero().squeeze(1), sel])
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


@torch.no_grad()
def embed(net, D, dev):
    # L2-normalized penultimate features on the clean train view
    net.eval()
    amp = dev.type == 'cuda'
    body = nn.Sequential(*list(net.children())[:-1])
    Fe = []
    for s in range(0, len(D['Xtr']), 1024):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            f = body(norm(D, D['Xtr'][s:s + 1024]).contiguous(memory_format=torch.channels_last)).flatten(1)
        Fe.append(F.normalize(f.float(), dim=1))
    return torch.cat(Fe)


@torch.no_grad()
def density(net, D, dev, k=20):
    """
    Yang et al. Eq. (1): rho(x) = mean distance to k NN (higher = sparser), min-max normalized -> p_rho.
    """
    Fe = embed(net, D, dev)
    rho = torch.empty(len(Fe), device=dev)
    for s in range(0, len(Fe), 1024):
        sim = Fe[s:s + 1024] @ Fe.T
        n = sim.size(0)
        sim[torch.arange(n), torch.arange(s, s + n)] = -2
        v = sim.topk(k, 1).values
        rho[s:s + n] = (2 - 2 * v).clamp(min=0).sqrt().mean(1)
    return (rho - rho.min()) / (rho.max() - rho.min() + 1e-12)


@torch.no_grad()
def clip_sims(D, dev, cache):
    """
    (N, K) cosine between CLIP ViT-B/32 image embedding and "A photo of a {c}." text embedding.
    Label-independent, computed once on clean images and cached. Model/prompt as Yang et al. ICLR 2025 code.
    """
    if cache.is_file():
        return torch.load(cache, map_location='cpu', weights_only=True).float().to(dev)
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    m = m.to(dev).eval()
    tok = open_clip.get_tokenizer('ViT-B-32')
    T = F.normalize(m.encode_text(tok([f'A photo of a {c.replace("_", " ")}.' for c in D['names']]).to(dev)).float(), dim=1)

    # CLIP preprocessing on GPU: bicubic 32 -> 224, CLIP mean/std
    cm = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    cs = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
    S = []
    for s in range(0, len(D['Xtr']), 512):
        x = F.interpolate(D['Xtr'][s:s + 512].float() / 255, size=224, mode='bicubic', align_corners=False).clamp(0, 1)
        S.append(F.normalize(m.encode_image((x - cm) / cs).float(), dim=1) @ T.T)
    S = torch.cat(S).half().cpu()
    atomic_write(cache, lambda temp: torch.save(S, temp))
    return S.float().to(dev)


@torch.no_grad()
def cen_iter(net, D, dev, it=4):
    """
    Centroid margin max_{k!=y} cos(f, c_k) - cos(f, c_y) with centroids re-estimated it times on samples with margin < 0.
    """
    Fe, y, K = embed(net, D, dev), D['Ytr'], D['K']
    C0 = F.normalize(torch.zeros(K, Fe.size(1), device=dev).index_add_(0, y, Fe), dim=1)
    C, w = C0, None
    for i in range(it + 1):
        if w is not None:
            Cw = torch.zeros(K, Fe.size(1), device=dev).index_add_(0, y[w], Fe[w])
            ok = torch.bincount(y[w], minlength=K) > 0
            C = torch.where(ok[:, None], F.normalize(Cw, dim=1), C0)
        sc = Fe @ C.T
        m = sc.scatter(1, y[:, None], -2).max(1).values - sc.gather(1, y[:, None]).squeeze(1)
        w = m < 0
    return m


def pearson(traj, ref, eps=1e-8):
    """
    DAS rho, verbatim from AlignPrune scores.py. traj (W, B) oldest epoch first, ref (W,). Returns (B,).
    """
    traj = traj - traj.mean(dim=0, keepdim=True)
    ref = (ref - ref.mean()).view(-1, 1)
    num = (traj * ref).sum(dim=0)
    den = torch.sqrt((traj ** 2).sum(dim=0) * (ref ** 2).sum()) + eps
    return num / den


@torch.no_grad()
def ref_loss(net, D, dev):
    # Mean CE (no smoothing) of the clean reference set on the clean view, as their evaluate()
    net.eval()
    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == 'cuda'):
        out = net(norm(D, D['Xtr'][D['ref']]).contiguous(memory_format=torch.channels_last))
    return F.cross_entropy(out.float(), D['Y_clean'][D['ref']])


@torch.no_grad()
def knn_disagreement(net, D, dev, k=20):
    """Fraction of cosine nearest neighbors whose given labels disagree, excluding self."""
    Fe, y = embed(net, D, dev), D['Ytr']
    knn = torch.empty(len(y), device=dev)
    for s in range(0, len(y), 1024):
        sim = Fe[s:s + 1024] @ Fe.T
        n = sim.size(0)
        sim[torch.arange(n), torch.arange(s, s + n)] = -2
        nb = sim.topk(k, 1).indices
        knn[s:s + n] = 1 - (y[nb] == y[s:s + n, None]).float().mean(1)
    return knn


# Drop rules: True = exclude sample until next refresh
FILTERS = {
    'oracle': lambda net, D, dev: D['noisy'].clone(),
    'centroid': lambda net, D, dev: cen_iter(net, D, dev, it=0) > 0,
    'knn': lambda net, D, dev: knn_disagreement(net, D, dev) > 0.5,
    'cen_iter': lambda net, D, dev: cen_iter(net, D, dev) > 0,
    'clip': lambda net, D, dev: D['clip'].argmax(1) != D['Ytr'],       # same top-1 rule as centroid, in CLIP space
}

# name: (InfoBatch on/off, filter)
RUNS = {
    'full': (False, None),
    'ib': (True, None),
    'ib_oracle': (True, 'oracle'),
    'ib_centroid': (True, 'centroid'),
    'ib_knn': (True, 'knn'),
    'ib_cen_iter': (True, 'cen_iter'),
    'ib_clip': (True, 'clip'),
    'yang': ('yang', None),
    'yang_sel': ('yang_sel', None),
    'ib_align': ('align', None),
}


def run_identity(method, args):
    ib, filt = RUNS[method]
    tags = [method]
    settings = dict(run=method, infobatch=ib, filter=filt, noise=args.noise, eta=args.eta,
                    epochs=args.epochs, full_epochs=args.full, seed=args.seed,
                    f0=args.f0 if filt is not None else None)
    if filt is not None:
        tags.append(f'f{args.f0}')
    if ib in ('yang', 'yang_sel'):
        settings['keep'] = args.keep
        tags.append(f'k{percent_slug(args.keep)}')
    if filt is not None or ib in ('yang', 'yang_sel', 'align'):
        settings['fe'] = args.fe
    if ib == 'align':
        settings.update(ref=args.ref, win=args.win)
    for field, prefix, default in [('fe', 'fe', 5), ('ref', 'ref', 1000), ('win', 'win', 25),
                                   ('full_epochs', 'full', 200), ('epochs', 'e', 200)]:
        value = settings.get(field, default)
        if value != default:
            tags.append(f'{prefix}{value}')
    tags.append(f's{args.seed}')
    return '_'.join(tags), settings


def run(D, ib, filt, dev, C=200, max_lr=0.05, r=0.5, delta=0.875, ls=0.1, bs=128, seed=0, f0=10, fe=5, kp=0.35,
        W=25, Cf=200):
    seed_all(seed)
    N = len(D['Xtr'])
    amp = dev.type == 'cuda'
    net = resnet18(D['K']).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(net.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)

    # Score init 3 as official infobatch.py
    H = torch.full((N,), 3.0, device=dev)
    keep = torch.ones(N, dtype=torch.bool, device=dev)
    log, trace, seen, tt, ft = [], [], 0, 0.0, 0.0

    # AlignPrune: reference set out of training; loss ring buffer (W, N), reference buffer (W,), DAS init 1
    if ib == 'align':
        keep[D['ref']] = False
        T = torch.full((W, N), 6.0, device=dev)
        R = torch.full((W,), 6.0, device=dev)
        vis = torch.zeros(N, dtype=torch.bool, device=dev)
        A = torch.ones(N, device=dev)
    pool = keep.sum().item()

    for e in range(C):
        # Filter refresh; dropped samples are re-scored each refresh and may return
        if filt is not None and e >= f0 and (e - f0) % fe == 0:
            if amp:
                torch.cuda.synchronize()
            t0 = time.time()
            d = FILTERS[filt](net, D, dev)
            if amp:
                torch.cuda.synchronize()
            ft += time.time() - t0

            # Degenerate filter (drops nearly everything): skip this refresh
            if (~d).sum() < bs:
                print(f'     filter e{e}: drops {d.float().mean():.3f}, ignored', flush=True)
                d = torch.zeros_like(d)
            keep = ~d
            nd = D['noisy']
            tr = dict(e=e, drop=d.float().mean().item(),
                      prec=(nd[d].float().mean().item() if d.any() else float('nan')),
                      rec=((d & nd).sum() / nd.sum().clamp(min=1)).item())
            trace.append(tr)
            print(f'     filter e{e}: drop {tr["drop"]:.3f}  prec {tr["prec"]:.3f}  rec {tr["rec"]:.3f}', flush=True)

        # Yang-style: top-kp by p_sel = p_rho * p_con; p_rho refreshed every fe epochs (outside train time)
        if ib in ('yang', 'yang_sel'):
            if e % fe == 0:
                if amp:
                    torch.cuda.synchronize()
                t0 = time.time()
                prho = density(net, D, dev)
                if amp:
                    torch.cuda.synchronize()
                ft += time.time() - t0
            if e + 1 <= delta * C or ib == 'yang_sel':
                idx = (prho * D['pcon']).topk(int(kp * N)).indices
            else:
                idx = torch.arange(N, device=dev)
            idx, w = idx[torch.randperm(len(idx), device=dev)], torch.ones(N, device=dev)
            if e % fe == 0:
                d = torch.ones(N, dtype=torch.bool, device=dev)
                d[idx] = False
                nd = D['noisy']
                trace.append(dict(e=e, drop=d.float().mean().item(), prec=nd[d].float().mean().item(),
                                  rec=((d & nd).sum() / nd.sum().clamp(min=1)).item()))

        # Annealing, InfoBatch Sec. 2.4 Eq. (10); applied to the kept set. AlignPrune ranks by DAS instead of loss
        elif ib and e + 1 <= delta * C:
            idx, w = ib_prune(A if ib == 'align' else H, r, keep)
        else:
            idx = keep.nonzero().squeeze(1)
            idx, w = idx[torch.randperm(len(idx), device=dev)], torch.ones(N, device=dev)
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
            if ib == 'align':
                T[e % W, i] = H[i]
                vis[i] = True

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

        # DAS of the samples trained this epoch, after the reference loss of this epoch (their update_reference)
        if ib == 'align':
            t0 = time.time()
            R[e % W] = ref_loss(net, D, dev)
            o = (torch.arange(W, device=dev) + e % W + 1) % W
            v = vis.nonzero().squeeze(1)
            A[v] = pearson(T[o][:, v], R[o])
            vis[:] = False
            if amp:
                torch.cuda.synchronize()
            ft += time.time() - t0
            if (e + 1) % fe == 0:
                d = (A < A[keep].mean()) & keep
                nd = D['noisy'] & keep
                trace.append(dict(e=e, drop=d.sum().item() / pool, prec=nd[d].float().mean().item() if d.any() else float('nan'),
                                  rec=((d & nd).sum() / nd.sum().clamp(min=1)).item()))
                print(f'     DAS e{e}: low {trace[-1]["drop"]:.3f}  prec {trace[-1]["prec"]:.3f}  rec {trace[-1]["rec"]:.3f}', flush=True)

        seen += len(idx)
        acc = evaluate(net, D, dev)
        log.append(dict(e=e, n=len(idx), loss=tl.item() / len(idx), acc=acc, t=tt, ft=ft))
        print(f'{e:3d}  |S|/N {len(idx)/N:.3f}  lr {sch.get_last_lr()[0]:.4f}  '
              f'loss {log[-1]["loss"]:.4f}  acc {acc:.2f}  {tt:.0f}s', flush=True)

    return log, trace, 1 - seen / (Cf * pool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', choices=RUNS, default=list(RUNS))
    p.add_argument('--noise', choices=['sym', 'asym'], default='sym')
    p.add_argument('--eta', type=float, default=0.4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--f0', type=int, default=40)
    p.add_argument('--fe', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--keep', type=float, default=0.35, help='yang: kept fraction per pruned epoch')
    p.add_argument('--ref', type=int, default=1000, help='ib_align: clean reference images held out')
    p.add_argument('--win', type=int, default=25, help='ib_align: trajectory window N')
    p.add_argument('--full', type=int, default=200, help='full-training epoch budget for the sample reduction metric')
    a = p.parse_args()
    if not 0 <= a.eta <= 1 or not 0 < a.keep <= 1:
        p.error('--eta must be in [0, 1] and --keep must be in (0, 1]')
    if min(a.epochs, a.full, a.fe, a.ref, a.win) < 1 or a.f0 < 0:
        p.error('--epochs, --full, --fe, --ref, and --win must be positive, and --f0 must be nonnegative')

    cfg = f'{a.noise}{percent_slug(a.eta)}'
    out = output_dir() / 'feature_filters' / cfg
    out.mkdir(parents=True, exist_ok=True)

    # The archived logs used these defaults for settings absent from their summaries.
    legacy_defaults = dict(keep=0.35, fe=5, ref=1000, win=25, full_epochs=200)
    pending = []
    for method in dict.fromkeys(a.runs):
        name, settings = run_identity(method, a)
        path = out / f'{name}.pt'
        if completed_run(path, settings, legacy_defaults):
            print(f'{name}: matching log and summary found, skipped', flush=True)
        else:
            pending.append((path, settings))
    if not pending:
        return

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False

    D = load('cifar100', dev)
    add_noise(D, a.noise, a.eta, seed=0)
    print(f'{cfg}: noisy fraction {D["noisy"].float().mean():.4f}', flush=True)

    methods = {settings['run'] for _, settings in pending}
    clip_methods = {'ib_clip', 'yang', 'yang_sel'}
    clip_timing = {}
    if clip_methods & methods:
        cache = output_dir() / 'clip_vitb32_cifar100.pt'
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with FileLock(str(cache) + '.lock'):
            cache_hit = cache.is_file()
            D['clip'] = clip_sims(D, dev, cache)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        clip_timing = dict(clip_setup_s=round(time.perf_counter() - start, 1), clip_cache_hit=cache_hit)
        con = D['clip'].gather(1, D['Ytr'][:, None]).squeeze(1)
        D['pcon'] = (con - con.min()) / (con.max() - con.min() + 1e-12)
        top1 = (D['clip'].argmax(1) == D['Y_clean']).float().mean()
        print(f'CLIP zero-shot train top-1 vs clean labels: {top1:.4f}', flush=True)

    # AlignPrune reference set: fixed by seed 0 like the noise, independent of the training seed
    if 'ib_align' in methods:
        if a.ref >= len(D['Ytr']):
            p.error('--ref must leave training examples outside the clean reference set')
        D['ref'] = torch.randperm(len(D['Ytr']), generator=torch.Generator().manual_seed(0))[:a.ref].to(dev)
        print(f'reference set: {a.ref} clean images, held out from ib_align training', flush=True)

    for path, settings in pending:
        name = path.stem
        ib, filt = settings['infobatch'], settings['filter']
        print(f'\n{name}: infobatch={ib}  filter={filt}  {cfg}  epochs={a.epochs}  seed={a.seed}', flush=True)
        log, trace, saved = run(D, ib, filt, dev, C=a.epochs, seed=a.seed, f0=a.f0, fe=a.fe, kp=a.keep,
                                W=a.win, Cf=a.full)
        last_filter = trace[-1].copy() if trace else None
        if last_filter is not None and math.isnan(last_filter['prec']):
            last_filter['prec'] = None
        record = dict(
            **settings,
            final=log[-1]['acc'], best=max(d['acc'] for d in log),
            saved_fwd=round(100 * saved, 2), train_s=round(log[-1]['t'], 1),
            filter_s=round(log[-1]['ft'], 1), last_filter=last_filter)
        if settings['run'] in clip_methods:
            record.update(clip_timing)
        save_indexed_run(path, dict(log=log, trace=trace), record)
        print(f'{name}: final {record["final"]:.2f}  best {record["best"]:.2f}  '
              f'training examples saved {record["saved_fwd"]:.1f}%  filter {record["filter_s"]:.0f}s', flush=True)


if __name__ == '__main__':
    main()
