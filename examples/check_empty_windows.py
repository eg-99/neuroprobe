"""Check whether any dataset items have empty time windows for the 0-1s bin.

Usage (from examples/):
    python check_empty_windows.py --subject_id 3 --trial_id 1
    python check_empty_windows.py  # scans all 12 Lite subject/trial pairs
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neuroprobe.braintreebank_subject import BrainTreebankSubject
import neuroprobe.train_test_splits as neuroprobe_train_test_splits
import neuroprobe.config as neuroprobe_config
from eval_utils import subset_electrodes

SR = 2048
DATA_IDX_FROM = 0
DATA_IDX_TO = SR

LITE_PAIRS = [
    (1, 1), (1, 2),
    (2, 0), (2, 4),
    (3, 0), (3, 1),
    (4, 0), (4, 1),
    (7, 0), (7, 1),
    (10, 0), (10, 1),
]


def check_pair(subject_id, trial_id, task='onset'):
    subject = BrainTreebankSubject(subject_id, cache=False, dtype=torch.float32)
    subset_electrodes(subject, lite=True, nano=False)
    subject.load_neural_data(trial_id)

    issues = []
    for split in ['WithinSession', 'CrossSession']:
        try:
            folds = neuroprobe_train_test_splits.generate_splits_within_session(
                subject, trial_id, task,
                dtype=torch.float32, output_indices=False,
                start_neural_data_before_word_onset=0,
                end_neural_data_after_word_onset=SR,
                lite=True, nano=False, binary_tasks=True,
            ) if split == 'WithinSession' else \
            neuroprobe_train_test_splits.generate_splits_cross_session(
                subject, trial_id, task,
                dtype=torch.float32, output_indices=False,
                start_neural_data_before_word_onset=0,
                end_neural_data_after_word_onset=SR,
                lite=True, binary_tasks=True,
            )
        except Exception as e:
            issues.append(f"  {split}: FOLD GENERATION ERROR — {e}")
            continue

        for fold_idx, fold in enumerate(folds):
            for ds_name, dataset in [('train', fold['train_dataset']), ('test', fold['test_dataset'])]:
                empty = 0
                total = 0
                for item in dataset:
                    x = item['data'] if isinstance(item, dict) else item[0]
                    total += 1
                    if x[:, DATA_IDX_FROM:DATA_IDX_TO].shape[-1] == 0:
                        empty += 1
                if empty:
                    issues.append(f"  {split} fold{fold_idx} {ds_name}: {empty}/{total} empty windows")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=int, default=None)
    parser.add_argument('--trial_id', type=int, default=None)
    parser.add_argument('--task', type=str, default='onset')
    args = parser.parse_args()

    if args.subject_id is not None:
        pairs = [(args.subject_id, args.trial_id)]
    else:
        pairs = LITE_PAIRS

    any_issue = False
    for sub, trial in pairs:
        print(f"sub{sub}_trial{trial} ... ", end='', flush=True)
        issues = check_pair(sub, trial, args.task)
        if issues:
            print("ISSUES FOUND:")
            for msg in issues:
                print(msg)
            any_issue = True
        else:
            print("OK")

    if not any_issue:
        print("\nNo empty windows found — crash was likely just the machine restart.")
    else:
        print("\nEmpty windows found — code fix needed before restarting.")


if __name__ == '__main__':
    main()
