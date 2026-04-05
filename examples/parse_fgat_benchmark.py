"""Parse FGAT benchmark results and print a comparison table.

Usage (from examples/):
    python parse_fgat_benchmark.py [--base eval_results/fgat_benchmark]
"""
import json, glob, os, sys
from collections import defaultdict

base = sys.argv[1] if len(sys.argv) > 1 else 'eval_results/fgat_benchmark'

# {config: {task: [auc, ...]}}
data = defaultdict(lambda: defaultdict(list))
train_data = defaultdict(lambda: defaultdict(list))

for jf in glob.glob(os.path.join(base, '**', '*.json'), recursive=True):
    parts = jf.replace(os.sep, '/').split('/')
    # path: base/config/sub_trial/save_dir_name/population_*.json
    config = parts[len(base.split('/'))].rstrip('/')
    filename = os.path.basename(jf)
    task = filename.replace('population_', '').split('_', 2)[-1].replace('.json', '')

    try:
        d = json.load(open(jf))
        for subj, sdata in d['evaluation_results'].items():
            pop = sdata['population']
            window = pop.get('one_second_after_onset') or (pop['time_bins'][0] if pop.get('time_bins') else None)
            if window is None:
                continue
            for fold in window['folds']:
                data[config][task].append(fold['test_roc_auc'])
                train_data[config][task].append(fold['train_roc_auc'])
    except Exception as e:
        print(f"Warning: could not parse {jf}: {e}")

if not data:
    print(f"No results found in {base}")
    sys.exit(1)

configs = sorted(data.keys())
all_tasks = sorted({t for c in data for t in data[c]})

# Per-task table
print(f"\n{'Task':<20}", end='')
for c in configs:
    print(f"  {c:<18}", end='')
print()
print('-' * (20 + 20 * len(configs)))

task_means = defaultdict(dict)
for task in all_tasks:
    print(f"{task:<20}", end='')
    for c in configs:
        aucs = data[c].get(task, [])
        if aucs:
            m = sum(aucs) / len(aucs)
            task_means[c][task] = m
            print(f"  {m:.4f}{'':12}", end='')
        else:
            print(f"  {'—':18}", end='')
    print()

# Overall means
print('-' * (20 + 20 * len(configs)))
print(f"{'MEAN':<20}", end='')
for c in configs:
    vals = list(task_means[c].values())
    m = sum(vals) / len(vals) if vals else 0
    print(f"  {m:.4f}{'':12}", end='')
print()

# Train vs test summary (overfitting check)
print(f"\n{'Overfitting check (train - test AUC):'}")
print(f"{'Config':<22} {'Mean Train':>12} {'Mean Test':>12} {'Gap':>8}")
print('-' * 56)
for c in configs:
    all_test  = [v for vs in data[c].values() for v in vs]
    all_train = [v for vs in train_data[c].values() for v in vs]
    if all_test:
        mt = sum(all_train)/len(all_train)
        ms = sum(all_test)/len(all_test)
        print(f"{c:<22} {mt:>12.4f} {ms:>12.4f} {mt-ms:>8.4f}")

print(f"\nReference: linear baseline ~0.6600 | GNN ~0.5490")
