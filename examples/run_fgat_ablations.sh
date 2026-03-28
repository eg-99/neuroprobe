#!/bin/bash
# FGAT ablation study — run 4 ablation variants on sub1_trial1, WithinSession, onset only.
#
# Ablations (in priority order):
#   1. No graph bias    (key contribution test — if this matches full model, rethink)
#   2. Single Pearson   (multiband vs wideband graph)
#   3. No Laplacian     (is reref needed?)
#   4. Mean pooling     (gated vs mean pool)
#
# Usage:
#   cd examples/
#   ROOT_DIR_BRAINTREEBANK=/storage/eg99/braintreebank_data \
#     conda run -n gait_fixed bash run_fgat_ablations.sh
#
# After running, compare with parse_competition.py on each ablation's SAVE_BASE.

set -e

export ROOT_DIR_BRAINTREEBANK="${ROOT_DIR_BRAINTREEBANK:-/storage/eg99/braintreebank_data}"
export NEUROPROBE_FEATURES_FILE="${NEUROPROBE_FEATURES_FILE:-features.csv}"

# Fast eval: sub1 trial1, onset only, WithinSession
SUBJECT_ID=1
TRIAL_ID=1
SPLIT="WithinSession"
EVAL_TASK="onset"

run_ablation() {
    local NAME=$1
    local SAVE_DIR="eval_results/fgat_ablations/${NAME}/sub${SUBJECT_ID}_trial${TRIAL_ID}"
    shift
    echo "--- Ablation: $NAME ---"
    python eval_population.py \
        --classifier_type fgat \
        --subject_id "$SUBJECT_ID" \
        --trial_id "$TRIAL_ID" \
        --eval_name "$EVAL_TASK" \
        --split_type "$SPLIT" \
        --only_1second \
        --verbose \
        --if_exists skip \
        --save_dir "$SAVE_DIR" \
        "$@"
    echo ""
}

echo "========================================================"
echo "FGAT Ablation Study"
echo "  Subject: sub${SUBJECT_ID} trial${TRIAL_ID}"
echo "  Task: $EVAL_TASK  Split: $SPLIT"
echo "========================================================"
echo ""

# Full model (baseline for comparison)
run_ablation "fgat_full" \
    --fgat_graph_bias true --fgat_graph_type multiband \
    --fgat_reref fixed --fgat_pooling gated

# Ablation 1: No graph bias
run_ablation "fgat_no_graph_bias" \
    --fgat_graph_bias false --fgat_graph_type multiband \
    --fgat_reref fixed --fgat_pooling gated

# Ablation 2: Single Pearson graph
run_ablation "fgat_single_graph" \
    --fgat_graph_bias true --fgat_graph_type single \
    --fgat_reref fixed --fgat_pooling gated

# Ablation 3: No Laplacian reref
run_ablation "fgat_no_reref" \
    --fgat_graph_bias true --fgat_graph_type multiband \
    --fgat_reref none --fgat_pooling gated

# Ablation 4: Mean pooling
run_ablation "fgat_mean_pool" \
    --fgat_graph_bias true --fgat_graph_type multiband \
    --fgat_reref fixed --fgat_pooling mean

echo "========================================================"
echo "Ablations complete. Results in eval_results/fgat_ablations/"
echo ""
echo "To parse results, run parse_competition.py pointing at each subdir."
echo "========================================================"
