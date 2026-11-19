"""
All figures used in report/report.pdf, built only from outputs/. Writes outputs/report/pdf_figs/*.pdf.

  signals_auroc     AUROC(noisy vs clean) of per-sample signals over training, no filtering
  acc_curves        test accuracy per epoch, seed 0
  acc_vs_compute    final accuracy (seed mean, sample std bars) vs training examples saved
  filter_quality    precision and recall of the drop set at every filter refresh
  das_precision     share of noisy samples in AlignPrune's low-DAS set vs the noise rate

Usage: python make_figures.py
"""
import os
import csv
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.results import output_dir

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = str(output_dir())
FIG = os.path.join(OUT, 'report', 'pdf_figs')
SET = [('sym20', 'Symmetric 20%', 0.1992), ('sym40', 'Symmetric 40%', 0.3975), ('asym40', 'Asymmetric 40%', 0.3975)]

# Fixed colour per method (Okabe-Ito), never cycled
M = {
    'full': ('Full data', '#999999', ':'),
    'ib': ('InfoBatch', '#000000', '-'),
    'ib_oracle_f40': ('Oracle filter', '#555555', '--'),
    'ib_clip_f40': ('CLIP filter', '#0072B2', '-'),
    'yang_sel_k35': ('Yang-style adaptation', '#CC79A7', '-'),
    'ib_align': ('AlignPrune port', '#E69F00', '-'),
    'ib_centroid_f40': ('Centroid', '#56B4E9', '-'),
    'ib_cen_iter_f40': ('Cleaned centroid', '#D55E00', '-'),
}
SIG = {'loss': '#000000', 'MSP': '#E69F00', 'energy': '#CC79A7', 'knn': '#009E73',
       'centroid': '#56B4E9', 'cen_iter': '#D55E00'}

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6, 'lines.linewidth': 1.6,
                     'legend.frameon': False, 'savefig.bbox': 'tight'})


def run(s, name):
    p = os.path.join(OUT, 'feature_filters', s, name + '.pt')
    return torch.load(p, map_location='cpu', weights_only=True) if os.path.exists(p) else None


def summary(s):
    # Seed groups: {method: [final, ...]}, {method: [saved, ...]}
    path = os.path.join(OUT, 'feature_filters', s, 'summary.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    else:
        d = {}
    G = {}
    for k, v in d.items():
        G.setdefault(k.rsplit('_s', 1)[0], []).append(v)
    return G


def save(f, n):
    os.makedirs(FIG, exist_ok=True)
    f.savefig(os.path.join(FIG, n + '.pdf'))
    f.savefig(os.path.join(FIG, n + '.png'), dpi=200)
    plt.close(f)


def signals_auroc():
    f, ax = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
    for a, (s, t, _) in zip(ax, SET):
        base = os.path.join(OUT, 'uncertainty_signals', s + '_ib_s0')
        path = base + '_v2/auroc.csv'
        if not os.path.exists(path):
            path = base + '/auroc.csv'
        if os.path.exists(path):
            with open(path, encoding='utf-8') as csv_file:
                R = list(csv.DictReader(csv_file))
        else:
            R = []
        if not R:
            a.text(0.5, 0.5, 'Run pending', ha='center', transform=a.transAxes)
            a.set_title(t)
            a.set_xlabel('epoch')
            continue
        e = [int(r['epoch']) for r in R]
        for k, c in SIG.items():
            if k + '_all' in R[0]:
                a.plot(e, [float(r[k + '_all']) for r in R], color=c, label=k)
        a.axhline(0.5, color='#999999', lw=0.8, ls='--')
        a.set_title(t)
        a.set_xlabel('epoch')
    ax[0].set_ylabel('AUROC, noisy vs clean')
    handles = {}
    for a in ax:
        h, l = a.get_legend_handles_labels()
        handles.update(zip(l, h))
    if handles:
        ax[-1].legend(handles.values(), handles.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5))
    save(f, 'signals_auroc')


def acc_curves():
    f, ax = plt.subplots(1, 3, figsize=(10, 2.9))
    for a, (s, t, _) in zip(ax, SET):
        for k, (lab, c, ls) in M.items():
            r = run(s, k + '_s0')
            if r is not None and k not in ('full', 'yang_sel_k35'):
                a.plot([l['e'] for l in r['log']], [l['acc'] for l in r['log']], color=c, ls=ls, lw=1.2, label=lab)
        a.set_title(t)
        a.set_xlabel('epoch')
    ax[0].set_ylabel('test accuracy (%)')
    ax[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    save(f, 'acc_curves')


def acc_vs_compute():
    f, ax = plt.subplots(1, 3, figsize=(10, 3.1))
    for a, (s, t, _) in zip(ax, SET):
        G = summary(s)
        for k, (lab, c, _) in M.items():
            if k not in G:
                continue
            y = np.array([g['final'] for g in G[k]])
            x = np.mean([g['saved_fwd'] for g in G[k]])
            a.errorbar(x, y.mean(), yerr=y.std(ddof=1) if len(y) > 1 else None, fmt='o' if k != 'ib_oracle_f40' else 's',
                       color=c, ms=6, capsize=3, mec='white', mew=0.8, label=lab)
        a.set_title(t)
        a.set_xlabel('training examples saved (%)')
    ax[0].set_ylabel('final test accuracy (%)')
    h, l = ax[0].get_legend_handles_labels()
    ax[-1].legend(h, l, loc='center left', bbox_to_anchor=(1.02, 0.5))
    save(f, 'acc_vs_compute')


def filter_quality():
    f, ax = plt.subplots(2, 3, figsize=(10, 4.8), sharex=True)
    for j, (s, t, eta) in enumerate(SET):
        for k in ('ib_clip_f40', 'ib_centroid_f40', 'ib_cen_iter_f40'):
            r = run(s, k + '_s0')
            if r is None:
                continue
            lab, c, _ = M[k]
            e = [q['e'] for q in r['trace']]
            ax[0, j].plot(e, [q['prec'] for q in r['trace']], color=c, label=lab)
            ax[1, j].plot(e, [q['rec'] for q in r['trace']], color=c, label=lab)
        ax[0, j].axhline(eta, color='#999999', lw=0.8, ls='--')
        ax[0, j].set_title(t)
        ax[1, j].set_xlabel('epoch')
    ax[0, 0].set_ylabel('precision of drop set')
    ax[1, 0].set_ylabel('recall of noisy samples')
    ax[0, -1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    save(f, 'filter_quality')


def das_precision():
    f, ax = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
    for a, (s, t, eta) in zip(ax, SET):
        r = run(s, 'ib_align_s0')
        if r is None:
            continue
        a.plot([q['e'] for q in r['trace']], [q['prec'] for q in r['trace']], color=M['ib_align'][1],
               label='noisy share of low-DAS set')
        a.axhline(eta, color='#999999', lw=0.8, ls='--', label='noise rate (random)')
        a.axvline(24, color='#cccccc', lw=0.8)
        a.set_title(t)
        a.set_xlabel('epoch')
    ax[0].set_ylabel('precision')
    ax[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    save(f, 'das_precision')


if __name__ == '__main__':
    for fn in (signals_auroc, acc_curves, acc_vs_compute, filter_quality, das_precision):
        fn()
        print('wrote', fn.__name__)
