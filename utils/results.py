"""Shared paths and persistence for experiment outputs."""

import json
import os
import pickle
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from filelock import FileLock


def output_dir():
    default = Path(__file__).resolve().parent.parent / 'outputs'
    return Path(os.environ.get('INFOBATCH_OUTPUT_DIR', default)).expanduser().resolve()


def percent_slug(rate):
    """Keep integer names such as sym40 and retain fractional percentages."""
    return format(Decimal(str(rate)).scaleb(2).normalize(), 'f')


def atomic_write(path, write):
    with NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', delete=False) as file:
        temp = Path(file.name)
    try:
        write(temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def read_summary(path):
    try:
        with path.open(encoding='utf-8') as file:
            # Older traces used NaN for precision when no examples were dropped.
            return json.load(file, parse_constant=lambda _: None)
    except FileNotFoundError:
        return {}


def write_summary(path, record):
    content = json.dumps(record, indent=2, allow_nan=False) + '\n'
    atomic_write(path, lambda temp: temp.write_text(content, encoding='utf-8'))


def completed_run(path, settings, legacy_defaults=None):
    """Check both the summary and the epoch log before resuming an indexed run."""
    if not path.is_file():
        return False

    summary_path = path.with_name('summary.json')
    legacy_defaults = legacy_defaults or {}
    with FileLock(str(summary_path) + '.lock'):
        record = read_summary(summary_path).get(path.stem, {})
        if any(record.get(key, legacy_defaults.get(key)) != value for key, value in settings.items()):
            return False

        try:
            data = torch.load(path, map_location='cpu', weights_only=True)
            log = data['log'] if isinstance(data, dict) else data
            last = log[-1]
            return (
                len(log) == settings['epochs']
                and last['e'] == settings['epochs'] - 1
                and last['acc'] == record['final']
                and round(last['t'], 1) == record['train_s']
                and round(last.get('ft', 0.0), 1) == record.get('filter_s', 0.0)
            )
        except (EOFError, OSError, RuntimeError, pickle.UnpicklingError, IndexError, KeyError, TypeError):
            return False


def save_indexed_run(path, data, record):
    summary_path = path.with_name('summary.json')
    with FileLock(str(summary_path) + '.lock'):
        summary = read_summary(summary_path)
        summary[path.stem] = record
        atomic_write(path, lambda temp: torch.save(data, temp))
        write_summary(summary_path, summary)
