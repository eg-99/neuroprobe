"""Fast feature probe — compare alternative feature types using logistic regression CV.

Loads real iEEG data for one subject/task, extracts 4 feature types, runs
StratifiedKFold logistic regression on each, and prints an AUC table.
Runtime: ~2-5 minutes on Devon. Use this to validate new feature hypotheses
before investing in a full FGAT benchmark run.

Usage (from examples/):
    python probe_features.py --subject_id 1 --trial_id 1 --task onset
    python probe_features.py --subject_id 1 --trial_id 1 --task onset --features stft,coherence
    python probe_features.py --subject_id 3 --trial_id 0 --task speech

Feature types:
    stft       — log1p STFT magnitude (reference, mirrors linear baseline)
    bandpower  — mean STFT power per frequency band (5 bands × E electrodes)
    envelope   — Hilbert amplitude envelope per band
    coherence  — inter-electrode magnitude-squared coherence per band
"""

import argparse
import os
import sys
import time

import numpy as np
import scipy.signal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from neuroprobe.braintreebank_subject import BrainTreebankSubject
import neuroprobe.train_test_splits as neuroprobe_train_test_splits
import neuroprobe.config as neuroprobe_config
from eval_utils import laplacian_rereference_neural_data, preprocess_stft

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SR = 2048  # Hz
DATA_IDX_FROM = 0                # 0s after word onset
DATA_IDX_TO = SR                 # 1s after word onset

# Frequency bands — same as MultibandGraphBuilder in eval_utils.py
BANDS = {
    'theta':      (4,   8),
    'alpha':      (8,   13),
    'beta':       (13,  30),
    'low_gamma':  (30,  70),
    'high_gamma': (70,  150),
}
MAX_FREQ = 150  # Hz

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _laplacian_reref(X_raw, electrode_labels):
    """X_raw: (N, E, T) numpy → (N, E_lap, T) numpy, electrode_labels_lap."""
    X_tensor = torch.from_numpy(X_raw).float()
    X_ref, labels_ref, _ = laplacian_rereference_neural_data(X_tensor, electrode_labels)
    if isinstance(X_ref, torch.Tensor):
        X_ref = X_ref.numpy()
    return X_ref.astype(np.float32), labels_ref


def _band_bin_mask(F, max_freq=MAX_FREQ):
    """Return dict of boolean masks over F freq bins for each band."""
    freqs = np.linspace(0, max_freq, F)
    return {name: (freqs >= lo) & (freqs < hi) for name, (lo, hi) in BANDS.items()}


# ---------------------------------------------------------------------------
# Feature extractors
# All take X_raw: (N, E, T) numpy + electrode_labels list
# All return (N, n_features) float32
# ---------------------------------------------------------------------------

def feat_stft(X_raw, electrode_labels):
    """Reference: log1p STFT magnitude, mirrors FGAT/linear baseline pipeline."""
    X_ref, _ = _laplacian_reref(X_raw, electrode_labels)
    X_tensor = torch.from_numpy(X_ref).float()
    stft = preprocess_stft(X_tensor, sampling_rate=SR, preprocess='stft_abs',
                           preprocess_parameters={'stft': {
                               'nperseg': 512, 'poverlap': 0.75,
                               'window': 'hann', 'max_frequency': MAX_FREQ, 'min_frequency': 0
                           }})
    if isinstance(stft, torch.Tensor):
        stft = stft.numpy()
    stft_log = np.log1p(stft.astype(np.float32))  # (N, E_lap, TT, F)
    return stft_log.reshape(len(X_raw), -1)


def feat_bandpower(X_raw, electrode_labels):
    """Mean STFT power per frequency band: (N, E_lap * 5)."""
    X_ref, _ = _laplacian_reref(X_raw, electrode_labels)
    X_tensor = torch.from_numpy(X_ref).float()
    stft = preprocess_stft(X_tensor, sampling_rate=SR, preprocess='stft_abs',
                           preprocess_parameters={'stft': {
                               'nperseg': 512, 'poverlap': 0.75,
                               'window': 'hann', 'max_frequency': MAX_FREQ, 'min_frequency': 0
                           }})
    if isinstance(stft, torch.Tensor):
        stft = stft.numpy()
    # stft: (N, E, TT, F) — average over time, then per band
    stft_mean = stft.mean(axis=2)  # (N, E, F)
    F = stft_mean.shape[-1]
    masks = _band_bin_mask(F)
    bands_out = [stft_mean[:, :, mask].mean(axis=-1) for mask in masks.values()]  # 5 × (N, E)
    return np.stack(bands_out, axis=-1).reshape(len(X_raw), -1).astype(np.float32)  # (N, E*5)


def feat_envelope(X_raw, electrode_labels):
    """Hilbert amplitude envelope per frequency band: (N, E_lap * 5)."""
    X_ref, _ = _laplacian_reref(X_raw, electrode_labels)
    N, E, T = X_ref.shape
    nyq = SR / 2

    bands_out = []
    for name, (lo, hi) in BANDS.items():
        lo_safe = max(lo, 1.0)           # avoid DC
        hi_safe = min(hi, nyq * 0.99)   # avoid Nyquist
        b, a = scipy.signal.butter(4, [lo_safe / nyq, hi_safe / nyq], btype='band')
        env = np.empty((N, E), dtype=np.float32)
        for i in range(N):
            filtered = scipy.signal.filtfilt(b, a, X_ref[i], axis=-1)  # (E, T)
            env[i] = np.abs(scipy.signal.hilbert(filtered, axis=-1)).mean(axis=-1)  # (E,)
        bands_out.append(env)

    return np.stack(bands_out, axis=-1).reshape(N, -1).astype(np.float32)  # (N, E*5)


def feat_coherence(X_raw, electrode_labels, max_pairs=500):
    """Magnitude-squared coherence between electrode pairs per band: (N, n_pairs * 5).

    Inter-electrode coherence captures phase synchrony — orthogonal to STFT magnitude.
    n_pairs = E*(E-1)/2; capped at max_pairs (ranked by mean coherence on full data).
    """
    X_ref, _ = _laplacian_reref(X_raw, electrode_labels)
    N, E, T = X_ref.shape
    all_pairs = list(zip(*np.triu_indices(E, k=1)))  # (E*(E-1)/2, 2)
    n_all_pairs = len(all_pairs)

    # Compute coherence for all pairs on all samples (band-averaged)
    # Shape: (N, n_pairs, 5)
    print(f"    coherence: {n_all_pairs} electrode pairs × {N} samples × 5 bands...", flush=True)

    # Pre-select top-max_pairs by mean coherence across samples to control dimensionality
    if n_all_pairs > max_pairs:
        # Quick pass on a subsample to rank pairs
        sub_n = min(50, N)
        sub_idx = np.random.choice(N, sub_n, replace=False)
        pair_means = np.zeros(n_all_pairs, dtype=np.float32)
        for k, (i, j) in enumerate(all_pairs):
            f, coh = scipy.signal.coherence(X_ref[sub_idx, i, :].T, X_ref[sub_idx, j, :].T,
                                             fs=SR, nperseg=512)
            # coh: (n_freqs, sub_n) — average over freq and samples
            pair_means[k] = coh.mean()
        top_idx = np.argsort(pair_means)[-max_pairs:]
        selected_pairs = [all_pairs[k] for k in top_idx]
        print(f"    coherence: limited to top-{max_pairs} pairs (of {n_all_pairs})", flush=True)
    else:
        selected_pairs = all_pairs

    n_pairs = len(selected_pairs)
    result = np.empty((N, n_pairs, len(BANDS)), dtype=np.float32)

    for k, (i, j) in enumerate(selected_pairs):
        f, coh = scipy.signal.coherence(X_ref[:, i, :], X_ref[:, j, :], fs=SR, nperseg=512)
        # coh: (N, n_freqs); f: (n_freqs,)
        F = len(f)
        # Derive band masks from actual freq axis
        for b_idx, (lo, hi) in enumerate(BANDS.values()):
            mask = (f >= lo) & (f < hi)
            if mask.sum() == 0:
                result[:, k, b_idx] = 0.0
            else:
                result[:, k, b_idx] = coh[:, mask].mean(axis=-1)

    return result.reshape(N, -1).astype(np.float32)  # (N, n_pairs * 5)


# ---------------------------------------------------------------------------
# CV probe
# ---------------------------------------------------------------------------

EXTRACTORS = {
    'stft':      feat_stft,
    'bandpower': feat_bandpower,
    'envelope':  feat_envelope,
    'coherence': feat_coherence,
}


def probe(X_feat, y, n_folds=5, random_state=42):
    """StratifiedKFold logistic regression. Returns (mean_auc, std_auc)."""
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=10000, tol=1e-3, random_state=random_state)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    aucs = []
    for tr, te in skf.split(X_feat, y):
        X_tr = scaler.fit_transform(X_feat[tr])
        X_te = scaler.transform(X_feat[te])
        clf.fit(X_tr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X_te)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(subject_id, trial_id, task):
    """Load raw iEEG windows for one subject/trial/task (WithinSession, 0-1s).

    Returns X_raw: (N, E, T) numpy float32, y: (N,) int, electrode_labels: list.
    Pools fold-0 train + test sets for maximum sample count.
    """
    print(f"Loading subject {subject_id} trial {trial_id}...", flush=True)
    subject = BrainTreebankSubject(subject_id, cache=True, dtype=torch.float32)
    from eval_utils import subset_electrodes
    subset_electrodes(subject, lite=True, nano=False)
    subject.load_neural_data(trial_id)

    folds = neuroprobe_train_test_splits.generate_splits_within_session(
        subject, trial_id, task,
        dtype=torch.float32,
        output_indices=False,
        start_neural_data_before_word_onset=0,
        end_neural_data_after_word_onset=SR,
        lite=True, nano=False, binary_tasks=True,
    )

    fold = folds[0]  # use fold 0 — pool train + test for CV

    def get_item(item):
        if isinstance(item, dict):
            return item['data'], int(item['label'])
        return item[0], int(item[1])

    Xs, ys = [], []
    for dataset in [fold['train_dataset'], fold['test_dataset']]:
        for item in dataset:
            x, y = get_item(item)
            Xs.append(x[:, DATA_IDX_FROM:DATA_IDX_TO].numpy())
            ys.append(y)

    X_raw = np.stack(Xs, axis=0).astype(np.float32)  # (N, E, T)
    y_arr = np.array(ys, dtype=np.int32)
    print(f"  Loaded {len(y_arr)} samples, {X_raw.shape[1]} electrodes, {X_raw.shape[2]} timepoints", flush=True)
    print(f"  Class balance: {y_arr.mean():.2f} (1s fraction)", flush=True)
    return X_raw, y_arr, subject.electrode_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Feature probe — compare feature types with logistic regression CV')
    parser.add_argument('--subject_id', type=int, default=1)
    parser.add_argument('--trial_id', type=int, default=1)
    parser.add_argument('--task', type=str, default='onset')
    parser.add_argument('--features', type=str, default='stft,bandpower,envelope,coherence',
                        help='Comma-separated list of feature types to test')
    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()

    requested = [f.strip() for f in args.features.split(',')]
    unknown = [f for f in requested if f not in EXTRACTORS]
    if unknown:
        parser.error(f"Unknown feature types: {unknown}. Choose from: {list(EXTRACTORS)}")

    X_raw, y, electrode_labels = load_data(args.subject_id, args.trial_id, args.task)

    print(f"\nProbing {len(requested)} feature type(s) with {args.n_folds}-fold CV...\n")
    header = f"{'Feature':<12}  {'AUC':>6}  {'±std':>6}  {'n_feat':>8}  {'time':>7}"
    print(header)
    print('-' * len(header))

    for name in requested:
        extractor = EXTRACTORS[name]
        t0 = time.time()
        print(f"  [{name}] extracting...", end='', flush=True)
        X_feat = extractor(X_raw, electrode_labels)
        elapsed_extract = time.time() - t0

        print(f" probing...", end='', flush=True)
        mean_auc, std_auc = probe(X_feat, y, n_folds=args.n_folds)
        elapsed_total = time.time() - t0

        print(f"\r{name:<12}  {mean_auc:.4f}  {std_auc:.4f}  {X_feat.shape[1]:>8}  {elapsed_total:>6.1f}s")

    print()
    print(f"Reference: linear baseline (full benchmark) ~0.660")
    print(f"           FGAT-large onset (benchmark_v2)  ~0.766")
    print()
    print("Next step: if any feature beats stft → integrate into eval_utils.py")
    print("           and run a proper benchmark via eval_population.py")


if __name__ == '__main__':
    main()
