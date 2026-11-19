"""
Tables and figures from outputs/. Safe to run at any point; missing runs are skipped.
Writes outputs/report/results.md and outputs/report/*.png.
"""
import os
import re
import csv
import json
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.results import output_dir

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = str(output_dir())
REP = os.path.join(OUT, 'report')

# Categorical slots in fixed order (reference palette, light mode)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'lines.linewidth': 1.8, 'savefig.dpi': 160, 'savefig.bbox': 'tight',
})

# Paper targets, InfoBatch Tab. 3 (R-18)
PAPER = {'c10_full': 95.6, 'c10_ib': 95.5, 'c100_full': 78.2, 'c100_ib': 78.2}


def jload(p):
    if not os.path.exists(p):
        return {}
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def ms(v):
    # mean ± std over seeds
    v = np.asarray(v, float)
    return f'{v.mean():.2f}' if len(v) == 1 else f'{v.mean():.2f} ± {v.std(ddof=1):.2f}'


def repro_table():
    s = jload(os.path.join(OUT, 'infobatch_repro', 'summary.json'))
    if not s:
        return []
    L = ['## InfoBatch reproduction (ResNet-18)', '',
         '| Run | Dataset | max lr | Paper | Final | Best | Training examples saved % | Train s |', '|---|---|---|---|---|---|---|---|']
    for k, d in s.items():
        L.append(f'| {k} | {d["dataset"]} | {d["max_lr"]} | {PAPER.get(k, "")} | {d["final"]:.2f} | '
                 f'{d["best"]:.2f} | {d["saved_fwd"]:.1f} | {d["train_s"]:.0f} |')
    return L + ['']


def ff_tables():
    L = []
    for cfg_dir in sorted(glob.glob(os.path.join(OUT, 'feature_filters', '*'))):
        s = jload(os.path.join(cfg_dir, 'summary.json'))
        if not s:
            continue
        cfg = os.path.basename(cfg_dir)

        # Group seeds: key = name without _s<seed>
        G = {}
        for k, d in s.items():
            G.setdefault(k.rsplit('_s', 1)[0], []).append(d)

        L += [f'## Feature filters, CIFAR-100 {cfg}', '',
              '| Method | Seeds | Final acc | Best acc | Training examples saved % | Train + filter s | Filter s | CLIP setup s | Drop / prec / rec (seed 0 end) |',
              '|---|---|---|---|---|---|---|---|---|']
        order = ['full', 'ib', 'ib_oracle', 'ib_clip', 'yang', 'yang_sel', 'ib_align', 'ib_centroid', 'ib_cen_iter', 'ib_knn']
        base = lambda x: re.sub(r'(_[fke]\d+)+$', '', x)
        rank = lambda x: (order.index(base(x)) if base(x) in order else 99, x)
        for g in sorted(G, key=rank):
            ds = G[g]
            first = min(ds, key=lambda d: d.get('seed', 0))
            lf = first.get('last_filter')
            pr = ''
            if lf:
                precision = f'{lf["prec"]:.3f}' if lf['prec'] is not None and np.isfinite(lf['prec']) else 'undefined'
                pr = f'{lf["drop"]:.3f} / {precision} / {lf["rec"]:.3f}'
            setup = [d.get('clip_setup_s') for d in ds]
            clip = f'{np.mean(setup):.1f}' if all(value is not None for value in setup) else 'unrecorded'
            if first.get('run') not in ('ib_clip', 'yang', 'yang_sel'):
                clip = '-'
            L.append(f'| {g} | {len(ds)} | {ms([d["final"] for d in ds])} | {ms([d["best"] for d in ds])} | '
                     f'{np.mean([d["saved_fwd"] for d in ds]):.1f} | '
                     f'{np.mean([d["train_s"] + d.get("filter_s", 0) for d in ds]):.1f} | '
                     f'{np.mean([d.get("filter_s", 0) for d in ds]):.1f} | {clip} | {pr} |')

        # Share of the InfoBatch -> oracle gap closed (seed means)
        m = {g: np.mean([d['final'] for d in G[g]]) for g in G}
        if 'ib' in m and 'ib_oracle_f40' in m:
            L.append('')
            for k, lab in [('ib_clip_f40', 'CLIP'), ('ib_align', 'AlignPrune'), ('ib_centroid_f40', 'Centroid'),
                           ('ib_cen_iter_f40', 'Cleaned centroid')]:
                if k in m:
                    gap = (m[k] - m['ib']) / max(m['ib_oracle_f40'] - m['ib'], 1e-9)
                    L.append(f'{lab} closes {100 * gap:.0f}% of the InfoBatch to oracle gap.  ')
        L.append('')
    return L


def fig_auroc():
    free = {'MSP', 'energy', 'knn_dist'}
    out = []
    for p in sorted(glob.glob(os.path.join(OUT, 'uncertainty_signals', '*', 'auroc.csv'))):
        tag = os.path.basename(os.path.dirname(p))
        with open(p, encoding='utf-8') as csv_file:
            R = list(csv.DictReader(csv_file))
        allnames = [c[:-4] for c in R[0] if c.endswith('_all')]

        # v2 runs: centroid variants only (fixed slots), else the original 7 signals
        names = ['loss', 'knn', 'centroid', 'cen_norm', 'cen_iter', 'cen_iter_norm'] if 'cen_iter' in allnames else allnames
        e = [int(r['epoch']) for r in R]
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        for a, part, title in [(ax[0], 'all', 'All samples'), (ax[1], 'high', 'High-loss half')]:
            a.axhline(0.5, color=MUTED, lw=1, ls=':')
            for j, n in enumerate(names):
                a.plot(e, [float(r[f'{n}_{part}']) for r in R], color=C[j], ls='--' if n in free else '-', label=n)
            a.set_title(title, color=INK, loc='left')
            a.set_xlabel('epoch')
        ax[0].set_ylabel('AUROC noisy vs clean')
        ax[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='dashed = label-free' if free & set(names) else None)
        fig.suptitle(f'Noise separability per signal, {tag}', x=0.01, y=1.06, ha='left', color=INK)
        f = os.path.join(REP, f'auroc_{tag}.png')
        fig.savefig(f)
        plt.close(fig)
        out.append(f)
    return out


def auroc_table():
    # Mean AUROC over the filtering window (epochs >= 40), all / high-loss half
    L = []
    for p in sorted(glob.glob(os.path.join(OUT, 'uncertainty_signals', '*', 'auroc.csv'))):
        tag = os.path.basename(os.path.dirname(p))
        with open(p, encoding='utf-8') as csv_file:
            R = [r for r in csv.DictReader(csv_file) if int(r['epoch']) >= 40]
        if not R:
            continue
        names = [c[:-4] for c in R[0] if c.endswith('_all')]
        L += [f'## Mean AUROC, epochs >= 40, {tag}', '', '| Signal | All | High-loss half |', '|---|---|---|']
        for n in names:
            L.append(f'| {n} | {np.mean([float(r[n + "_all"]) for r in R]):.3f} | {np.mean([float(r[n + "_high"]) for r in R]):.3f} |')
        L.append('')
    return L


def fig_acc_and_filter():
    out = []
    for cfg_dir in sorted(glob.glob(os.path.join(OUT, 'feature_filters', '*'))):
        cfg = os.path.basename(cfg_dir)
        runs = [('full_s0', 'full data'), ('ib_s0', 'InfoBatch'),
                ('ib_oracle_f40_s0', 'InfoBatch + oracle'), ('ib_centroid_f40_s0', 'InfoBatch + centroid'),
                ('ib_clip_f40_s0', 'InfoBatch + CLIP'), ('yang_sel_k35_s0', 'Density x CLIP (Yang-style)'),
                ('ib_cen_iter_f40_s0', 'InfoBatch + cleaned centroid'), ('ib_align_s0', 'AlignPrune (1k clean ref)')]
        R = {k: torch.load(os.path.join(cfg_dir, f'{k}.pt'), map_location='cpu', weights_only=True)
             for k, _ in runs if os.path.exists(os.path.join(cfg_dir, f'{k}.pt'))}
        if not R:
            continue

        fig, a = plt.subplots(figsize=(7, 3.8))
        for j, (k, lab) in enumerate(runs):
            if k in R:
                L = R[k]['log']
                a.plot([d['e'] for d in L], [d['acc'] for d in L], color=C[j], label=f'{lab} ({L[-1]["acc"]:.1f})')
        a.set_xlabel('epoch')
        a.set_ylabel('test accuracy (%)')
        a.legend(loc='lower right')
        a.set_title(f'Test accuracy, CIFAR-100 {cfg}', color=INK, loc='left')
        f = os.path.join(REP, f'acc_{cfg}.png')
        fig.savefig(f)
        plt.close(fig)
        out.append(f)

        k = 'ib_centroid_f40_s0'
        if k in R and R[k]['trace']:
            T = R[k]['trace']
            summary = jload(os.path.join(cfg_dir, 'summary.json'))
            eta = next(iter(summary.values()))['eta']
            fig, a = plt.subplots(figsize=(7, 3.4))
            a.axhline(eta, color=MUTED, lw=1, ls=':', label=f'true noise rate {eta:.2f}')
            for j, (m, lab) in enumerate([('drop', 'dropped fraction'), ('prec', 'precision'), ('rec', 'recall')]):
                a.plot([t['e'] for t in T], [t[m] for t in T], color=C[j], label=lab)
            a.set_ylim(0, 1.02)
            a.set_xlabel('epoch')
            a.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
            a.set_title(f'Centroid filter (f0=40), CIFAR-100 {cfg}', color=INK, loc='left')
            f = os.path.join(REP, f'filter_{cfg}.png')
            fig.savefig(f)
            plt.close(fig)
            out.append(f)
    return out


def main():
    os.makedirs(REP, exist_ok=True)
    L = ['# Results', '',
         'Accuracy is in percent. Error bars use sample standard deviation across training seeds. '
         'Synthetic label corruption is fixed at seed 0. Best accuracy is descriptive only.', '',
         'Training examples saved measures the reduction in training examples processed. '
         'Filter refreshes, evaluation, data loading, and external-model preparation are outside this count. '
         'Train + filter time includes the two recorded stages. Archived CLIP setup times are unrecorded. '
         'Archived plain-centroid timings include an unused kNN calculation removed from the current code. '
         'AlignPrune uses a 49,000-image training pool, and other archived methods use 50,000.', '',
         'Baselines use this project\'s training recipe. Yang-style selection is an adaptation, '
         'and AlignPrune is a port. The oracle is a privileged-label reference. '
         'The asymmetric full-training baseline is pending.', ''] + repro_table() + ff_tables() + auroc_table()

    figs = fig_auroc() + fig_acc_and_filter()
    if figs:
        L += ['## Figures', ''] + [f'![]({os.path.basename(f)})' for f in figs]

    with open(os.path.join(REP, 'results.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {REP}')


if __name__ == '__main__':
    main()
