"""Parse FGAT ablation results and print a comparison table.

Usage:
    python parse_fgat_ablations.py eval_results/fgat_ablations
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ABLATIONS = ['fgat_full', 'fgat_no_graph', 'fgat_single_graph', 'fgat_no_reref', 'fgat_mean_pool']
ABLATION_LABELS = {
    'fgat_full':         'full',
    'fgat_no_graph':     'no_graph',
    'fgat_single_graph': 'single_graph',
    'fgat_no_reref':     'no_reref',
    'fgat_mean_pool':    'mean_pool',
}


def parse_dir(base):
    """Returns dict: ablation -> task -> [test_aucs]"""
    results = {a: defaultdict(list) for a in ABLATIONS}

    for ablation in ABLATIONS:
        pattern = os.path.join(base, ablation, '**', '*.json')
        for fpath in glob.glob(pattern, recursive=True):
            fname = os.path.basename(fpath)
            # extract task name from filename: population_btbankX_Y_<task>.json
            parts = fname.replace('.json', '').split('_')
            # find index after subject identifier (btbankX_Y)
            task = '_'.join(parts[3:])  # everything after population_btbankX_Y

            with open(fpath) as f:
                d = json.load(f)

            pop = list(d['evaluation_results'].values())[0]['population']
            entry = pop.get('one_second_after_onset') or next(
                (b for b in pop.get('time_bins', [])
                 if b['time_bin_start'] == 0 and b['time_bin_end'] == 1), None)
            if entry is None:
                continue

            aucs = [fold['test_roc_auc'] for fold in entry['folds'] if 'test_roc_auc' in fold]
            if aucs:
                results[ablation][task].extend(aucs)

    return results


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else 'eval_results/fgat_ablations'

    results = parse_dir(base)
    all_tasks = sorted(set(t for a in ABLATIONS for t in results[a]))

    if not all_tasks:
        print(f"No results found in {base}")
        sys.exit(1)

    # Header
    col_w = 13
    header = f"{'Task':<22}" + ''.join(f"{ABLATION_LABELS[a]:>{col_w}}" for a in ABLATIONS)
    print(header)
    print('-' * len(header))

    task_means = {a: [] for a in ABLATIONS}
    for task in all_tasks:
        row = f"{task:<22}"
        for a in ABLATIONS:
            aucs = results[a].get(task, [])
            if aucs:
                mean = np.mean(aucs)
                task_means[a].append(mean)
                row += f"{mean:>{col_w}.4f}"
            else:
                row += f"{'n/a':>{col_w}}"
        print(row)

    print('-' * len(header))
    row = f"{'MEAN':<22}"
    for a in ABLATIONS:
        if task_means[a]:
            row += f"{np.mean(task_means[a]):>{col_w}.4f}"
        else:
            row += f"{'n/a':>{col_w}}"
    print(row)

    # Component contribution table
    print()
    print("Component contribution (vs full model):")
    print(f"  {'Component':<20}  {'Mean AUC':>9}  {'vs full':>8}  {'Conclusion'}")
    print('  ' + '-'*65)
    full_mean = np.mean(task_means['fgat_full']) if task_means['fgat_full'] else None
    labels = {
        'fgat_no_graph':     'Graph bias',
        'fgat_single_graph': 'Multiband (vs single)',
        'fgat_no_reref':     'Laplacian reref',
        'fgat_mean_pool':    'Gated pooling',
    }
    for a, label in labels.items():
        if not task_means[a] or full_mean is None:
            continue
        mean = np.mean(task_means[a])
        diff = mean - full_mean
        if diff < -0.005:
            conclusion = 'HELPS significantly'
        elif diff < -0.001:
            conclusion = 'helps slightly'
        elif diff > 0.005:
            conclusion = 'HURTS significantly'
        else:
            conclusion = 'negligible effect'
        print(f"  {label:<20}  {mean:>9.4f}  {diff:>+8.4f}  {conclusion}")


if __name__ == '__main__':
    main()
