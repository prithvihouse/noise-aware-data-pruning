"""
CIFAR-10/100 from Hugging Face, held on GPU as uint8.
Augmentation and normalization run per batch on GPU (no DataLoader workers).
All functions take the data dict D returned by load().
"""
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

# (HF path, label column, K)
HF = {
    'cifar10': ('uoft-cs/cifar10', 'label', 10),
    'cifar100': ('uoft-cs/cifar100', 'fine_label', 100),
}

# Channel mean/std. CIFAR-100 values from official InfoBatch examples/cifar_example.py
STATS = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    'cifar100': ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)),
}


def load(name, dev):
    """
    Returns D with Xtr, Xte (N,3,32,32) uint8; Ytr, Yte long; Y_clean; noisy mask; K; M, S in pixel units.
    """
    path, key, K = HF[name]
    ds = load_dataset(path)

    def split(s):
        # [:] needed: datasets>=4 returns lazy Column on column access
        d = ds[s].with_format('numpy')[:]
        X = torch.from_numpy(np.stack(d['img'])).permute(0, 3, 1, 2).contiguous()
        return X.to(dev), torch.from_numpy(np.asarray(d[key])).long().to(dev)

    Xtr, Ytr = split('train')
    Xte, Yte = split('test')
    m, s = STATS[name]
    D = dict(
        name=name, K=K, Xtr=Xtr, Ytr=Ytr, Xte=Xte, Yte=Yte, names=ds['train'].features[key].names,
        Y_clean=Ytr.clone(), noisy=torch.zeros(len(Ytr), dtype=torch.bool, device=dev),
        M=torch.tensor(m, device=dev).view(1, 3, 1, 1) * 255,
        S=torch.tensor(s, device=dev).view(1, 3, 1, 1) * 255,
    )

    # CIFAR-100 superclass per train sample, for asymmetric noise
    if name == 'cifar100':
        D['coarse'] = torch.from_numpy(np.asarray(ds['train'].with_format('numpy')[:]['coarse_label'])).long().to(dev)
    return D


def norm(D, x):
    return (x.float() - D['M']) / D['S']


def augment(D, x):
    # Reflect pad 4 + random 32x32 crop + hflip (InfoBatch Sec. 3: crop, flip, normalize)
    B, d = x.size(0), x.device
    x = F.pad(x.float(), (4, 4, 4, 4), mode='reflect')
    r = torch.randint(0, 9, (B,), device=d)[:, None] + torch.arange(32, device=d)
    c = torch.randint(0, 9, (B,), device=d)[:, None] + torch.arange(32, device=d)
    b = torch.arange(B, device=d)[:, None, None, None]
    ch = torch.arange(3, device=d)[None, :, None, None]
    x = x[b, ch, r[:, None, :, None], c[:, None, None, :]]
    f = torch.rand(B, device=d) < 0.5
    x = torch.where(f[:, None, None, None], x.flip(3), x)
    return (x - D['M']) / D['S']


def batches(D, idx, bs=128, train=True):
    """
    Yields (x, y, i) over idx; i keys per-sample tables. No drop_last (official repo default).
    """
    X, Y = (D['Xtr'], D['Ytr']) if train else (D['Xte'], D['Yte'])
    for k in range(0, len(idx), bs):
        i = idx[k:k + bs]
        yield (augment(D, X[i]) if train else norm(D, X[i])), Y[i], i


def sym_noise(D, eta, seed=0):
    """
    Symmetric noise: eta-fraction of train labels flipped uniformly to one of the K-1 other classes.
    Writes D['Ytr'] (noisy) and D['noisy'] (flip mask). Test labels untouched.
    """
    Y, K = D['Y_clean'].cpu(), D['K']
    g = torch.Generator().manual_seed(seed)
    f = torch.rand(len(Y), generator=g) < eta
    off = torch.randint(1, K, (len(Y),), generator=g)
    Yn = Y.clone()
    Yn[f] = (Yn[f] + off[f]) % K
    dev = D['Y_clean'].device
    D['Ytr'], D['noisy'] = Yn.to(dev), f.to(dev)
    return D['noisy']


def asym_noise(D, eta, seed=0):
    """
    CIFAR-100 asymmetric noise: eta-fraction flipped to the next fine class within the same
    superclass, circularly (Patrini et al., CVPR 2017 protocol). Concentrated, class-dependent noise.
    """
    Y, Cc, K = D['Y_clean'].cpu(), D['coarse'].cpu(), D['K']

    # Fine -> superclass, then next fine class inside each superclass (sorted, cyclic)
    sup = torch.zeros(K, dtype=torch.long)
    sup[Y] = Cc
    nxt = torch.empty(K, dtype=torch.long)
    for c in sup.unique():
        g = (sup == c).nonzero().squeeze(1).sort().values
        nxt[g] = g.roll(-1)

    gen = torch.Generator().manual_seed(seed)
    f = torch.rand(len(Y), generator=gen) < eta
    Yn = Y.clone()
    Yn[f] = nxt[Y[f]]
    dev = D['Y_clean'].device
    D['Ytr'], D['noisy'] = Yn.to(dev), f.to(dev)
    return D['noisy']


def add_noise(D, kind, eta, seed=0):
    # kind in {'sym', 'asym'}
    return (sym_noise if kind == 'sym' else asym_noise)(D, eta, seed)
