"""Regression checks for filter decisions, saved results, and experiment entry points."""

import argparse
import contextlib
import io
import json
import os
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from experiments import feature_filters as filters
from experiments import uncertainty_signals as uncertainty
from infobatch_repro import repro
from utils.results import completed_run, percent_slug, read_summary, save_indexed_run


ROOT = Path(__file__).resolve().parent.parent


class FilterChecks(unittest.TestCase):
    def test_centroid_decisions_match_original_without_knn(self):
        torch.manual_seed(3)
        features = F.normalize(torch.randn(30, 8), dim=1)
        labels = torch.arange(30) % 3
        data = dict(Ytr=labels, K=3)
        centroids = F.normalize(torch.zeros(3, 8).index_add_(0, labels, features), dim=1)
        scores = features @ centroids.T
        expected = scores.scatter(1, labels[:, None], -2).max(1).values - scores.gather(
            1, labels[:, None]
        ).squeeze(1)
        with patch.object(filters, 'embed', return_value=features), patch.object(
            torch.Tensor, 'topk', side_effect=AssertionError('centroid invoked kNN')
        ):
            actual = filters.FILTERS['centroid'](None, data, torch.device('cpu'))
        torch.testing.assert_close(actual, expected > 0)

    def test_cleaned_centroids_handle_empty_classes(self):
        features = F.normalize(torch.randn(20, 8), dim=1)
        labels = torch.arange(20) % 2
        with patch.object(filters, 'embed', return_value=features):
            margins = filters.cen_iter(None, dict(Ytr=labels, K=3), torch.device('cpu'))
        self.assertTrue(torch.isfinite(margins).all())

    def test_first_and_cached_clip_scores_are_identical(self):
        class ClipStub(torch.nn.Module):
            def encode_text(self, tokens):
                return tokens

            def encode_image(self, images):
                return images.mean(dim=(2, 3))

        fake_clip = types.SimpleNamespace(
            create_model_and_transforms=lambda *args, **kwargs: (ClipStub(), None, None),
            get_tokenizer=lambda *args: lambda texts: torch.eye(len(texts)),
        )
        data = dict(Xtr=torch.randint(0, 256, (5, 3, 32, 32), dtype=torch.uint8),
                    names=['one', 'two', 'three'])
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', open_clip=fake_clip):
            cache = Path(directory) / 'clip.pt'
            first = filters.clip_sims(data, torch.device('cpu'), cache)
            cached = filters.clip_sims(data, torch.device('cpu'), cache)
            torch.testing.assert_close(first, cached, rtol=0, atol=0)


class OutputChecks(unittest.TestCase):
    def test_names_preserve_defaults_and_distinguish_settings(self):
        args = argparse.Namespace(noise='sym', eta=0.4, epochs=200, full=200, seed=0,
                                  f0=40, fe=5, keep=0.35, ref=1000, win=25)
        self.assertEqual(filters.run_identity('ib_cen_iter', args)[0], 'ib_cen_iter_f40_s0')
        self.assertEqual(filters.run_identity('yang_sel', args)[0], 'yang_sel_k35_s0')
        for field, value, method in [('fe', 2, 'ib_cen_iter'), ('ref', 500, 'ib_align'),
                                     ('win', 10, 'ib_align'), ('full', 150, 'ib'),
                                     ('epochs', 2, 'full'), ('keep', 0.351, 'yang')]:
            changed = argparse.Namespace(**{**vars(args), field: value})
            self.assertNotEqual(filters.run_identity(method, args), filters.run_identity(method, changed))
        self.assertEqual(percent_slug(0.401), '40.1')

    def test_cache_requires_matching_complete_record_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ib_s0.pt'
            settings = dict(epochs=2, fe=5)
            log = [dict(e=0, acc=10.0, t=0.1), dict(e=1, acc=20.0, t=0.2)]
            record = dict(**settings, final=20.0, train_s=0.2)
            torch.save(dict(log=log), path)
            self.assertFalse(completed_run(path, settings))
            save_indexed_run(path, dict(log=log), record)
            self.assertTrue(completed_run(path, settings))
            self.assertFalse(completed_run(path, dict(epochs=2, fe=2)))
            save_indexed_run(path, dict(log=log[:1]), record)
            self.assertFalse(completed_run(path, settings))
            path.write_bytes(b'interrupted output')
            self.assertFalse(completed_run(path, settings))

    def test_concurrent_writes_preserve_all_summary_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            def save(index):
                path = Path(directory) / f'run_s{index}.pt'
                save_indexed_run(path, dict(log=[]), dict(seed=index))

            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(save, range(12)))
            self.assertEqual(len(read_summary(Path(directory) / 'summary.json')), 12)

    def test_archived_summaries_match_epoch_logs(self):
        paths = list((ROOT / 'outputs/feature_filters').glob('*/summary.json'))
        paths += list((ROOT / 'outputs/infobatch_repro').glob('summary.json'))
        if not paths:
            self.skipTest('Archived metrics are optional in a source-only checkout.')
        for path in paths:
            for name, record in read_summary(path).items():
                with self.subTest(run=name, setting=path.parent.name):
                    result = torch.load(path.with_name(name + '.pt'), map_location='cpu', weights_only=True)
                    log = result['log'] if isinstance(result, dict) else result
                    self.assertEqual(len(log), record['epochs'])
                    self.assertEqual(record['final'], log[-1]['acc'])
                    self.assertEqual(record['best'], max(row['acc'] for row in log))
                    self.assertEqual(record['train_s'], round(log[-1]['t'], 1))
                    self.assertEqual(record.get('filter_s', 0), round(log[-1].get('ft', 0), 1))
                    pool = 49000 if record.get('run') == 'ib_align' else 50000
                    expected = round(100 * (1 - sum(row['n'] for row in log) / (200 * pool)), 2)
                    self.assertAlmostEqual(record['saved_fwd'], expected, places=2)


class EntryPointChecks(unittest.TestCase):
    def test_training_entry_point_writes_and_then_skips_matching_run(self):
        torch.set_num_threads(2)
        images = torch.randint(0, 256, (256, 3, 32, 32), dtype=torch.uint8)
        labels = torch.arange(256) % 3
        data = dict(Xtr=images, Ytr=labels.clone(), Y_clean=labels.clone(), K=3,
                    Xte=images[:8], Yte=labels[:8],
                    M=torch.zeros(1, 3, 1, 1), S=torch.full((1, 3, 1, 1), 255.0))

        def tiny_net(classes):
            return torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, 1), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, classes)
            )

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, INFOBATCH_OUTPUT_DIR=directory), \
                patch.object(filters.torch.cuda, 'is_available', return_value=False), \
                patch.object(filters, 'load', return_value=data) as loader, \
                patch.object(filters, 'resnet18', side_effect=tiny_net), \
                patch('sys.argv', ['feature_filters.py', '--epochs', '2', '--f0', '1', '--fe', '1',
                                   '--runs', 'ib', 'ib_cen_iter']), contextlib.redirect_stdout(io.StringIO()):
            filters.main()
            summaries = list(Path(directory).glob('feature_filters/*/summary.json'))
            self.assertEqual(len(summaries), 1)
            self.assertEqual(len(read_summary(summaries[0])), 2)
            loader.reset_mock()
            filters.main()
            loader.assert_not_called()

    def test_reproduction_short_epoch_run_has_separate_cache(self):
        log = [dict(e=0, acc=20.0, t=0.1, n=50000)]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, INFOBATCH_OUTPUT_DIR=directory), \
                patch.object(repro, 'load', return_value={}), \
                patch.object(repro, 'run', return_value=(log, 0.0)) as trainer, \
                patch('sys.argv', ['repro.py', '--runs', 'c10_full', '--epochs', '1']), \
                contextlib.redirect_stdout(io.StringIO()):
            repro.main()
            self.assertTrue((Path(directory) / 'infobatch_repro/c10_full_e1.pt').is_file())
            trainer.reset_mock()
            repro.main()
            trainer.assert_not_called()

    def test_signal_settings_have_separate_cache(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, INFOBATCH_OUTPUT_DIR=directory), \
                patch.object(uncertainty, 'load', side_effect=RuntimeError('stop before download')), \
                patch('sys.argv', ['uncertainty_signals.py', '--epochs', '2', '--every', '1']):
            with self.assertRaisesRegex(RuntimeError, 'stop before download'):
                uncertainty.main()
            self.assertTrue((Path(directory) / 'uncertainty_signals/sym40_ib_e2_every1_s0').is_dir())

    def test_diagnostic_entry_point_rebuilds_missing_csv(self):
        labels = torch.arange(100) % 3
        data = dict(Y_clean=labels, Ytr=labels.clone(), K=3)
        log = [dict(e=0, acc=25.0, t=0.1)]
        signals = {0: torch.randn(100, len(uncertainty.NAMES))}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, INFOBATCH_OUTPUT_DIR=directory), \
                patch.object(uncertainty.torch.cuda, 'is_available', return_value=False), \
                patch.object(uncertainty, 'load', return_value=data), \
                patch.object(uncertainty, 'run', return_value=(log, signals, 0.0)) as trainer, \
                patch('sys.argv', ['uncertainty_signals.py', '--epochs', '1']), \
                contextlib.redirect_stdout(io.StringIO()):
            uncertainty.main()
            trainer.reset_mock()
            uncertainty.main()
            trainer.assert_not_called()
            csv = next(Path(directory).glob('uncertainty_signals/*/auroc.csv'))
            csv.unlink()
            uncertainty.main()
            trainer.assert_called_once()


if __name__ == '__main__':
    unittest.main()
