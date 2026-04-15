#!/bin/bash
# FGAT ablation study — isolate which components drive FGAT's advantage over GNN.
#
# Baseline: fgat_medium (D=96, layers=2) — faster than large, same conclusions
#
# Ablations:
#   no_graph_bias  — remove graph bias from spatial attention
#                    (tests: is the graph structure contributing at all?)
#   single_graph   — single wideband Pearson graph instead of 5-band
#                    (tests: does multiband specificity matter?)
#   no_reref       — skip Laplacian rereferencing
#                    (tests: how much does reref help?)
#   mean_pool      — mean pooling instead of gated attention pooling
#                    (tests: does gated pooling matter?)
#
# Scope: 3 subjects × 5 tasks × WithinSession — same as benchmark_v2.
# Expected runtime: ~6-8 hours on Devon GPU.
#
# Usage (from examples/):
#   export ROOT_DIR_BRAINTREEBANK=/storage/eg99/braintreebank_data
#   export NEUROPROBE_FEATURES_FILE=features.csv
#   HOME=/storage/eg99 nohup bash run_fgat_ablations.sh \
#     > /storage/eg99/neuroprobe/logs/fgat_ablations.log 2>&1 &

set -e

export ROOT_DIR_BRAINTREEBANK="${ROOT_DIR_BRAINTREEBANK:-/storage/eg99/braintreebank_data}"
export NEUROPROBE_FEATURES_FILE="${NEUROPROBE_FEATURES_FILE:-features.csv}"

TASKS="onset,speech,volume,gpt2_surprisal,word_length"
SPLIT="WithinSession"
SAVE_BASE="eval_results/fgat_ablations"

# Same 3 subjects as benchmark_v2
SUBJECT_TRIALS=("1 1" "3 0" "7 0")

run_ablation() {
    local NAME=$1
    local SUBJECT_ID=$2
    local TRIAL_ID=$3
    shift 3
    local SAVE_DIR="${SAVE_BASE}/${NAME}/sub${SUBJECT_ID}_trial${TRIAL_ID}"
    echo "  [${NAME}] sub${SUBJECT_ID} trial${TRIAL_ID}"
    python eval_population.py \
        --classifier_type fgat \
        --subject_id "$SUBJECT_ID" \
        --trial_id "$TRIAL_ID" \
        --eval_name "$TASKS" \
        --split_type "$SPLIT" \
        --only_1second \
        --verbose \
        --if_exists skip \
        --save_dir "$SAVE_DIR" \
        --fgat_D 96 \
        --fgat_n_layers 2 \
        --fgat_dropout 0.1 \
        --fgat_weight_decay 1e-3 \
        "$@"
}

echo "========================================================"
echo "FGAT Ablation Study"
echo "  Baseline: fgat_medium (D=96, layers=2)"
echo "  Variants: no_graph_bias / single_graph / no_reref / mean_pool"
echo "  Subjects: sub1_trial1, sub3_trial0, sub7_trial0"
echo "  Tasks   : $TASKS"
echo "  Output  : $SAVE_BASE"
echo "========================================================"
echo ""

for ST in "${SUBJECT_TRIALS[@]}"; do
    S=$(echo "$ST" | cut -d' ' -f1)
    T=$(echo "$ST" | cut -d' ' -f2)

    # Baseline — full model
    run_ablation "fgat_full"         "$S" "$T" \
        --fgat_graph_bias true  --fgat_graph_type multiband \
        --fgat_reref fixed      --fgat_pooling gated

    # Ablation 1: remove graph bias entirely
    # If this matches fgat_full → graph contributes nothing
    run_ablation "fgat_no_graph"     "$S" "$T" \
        --fgat_graph_bias false --fgat_graph_type multiband \
        --fgat_reref fixed      --fgat_pooling gated

    # Ablation 2: single wideband graph instead of 5-band
    # If this matches fgat_full → band specificity adds nothing
    run_ablation "fgat_single_graph" "$S" "$T" \
        --fgat_graph_bias true  --fgat_graph_type single \
        --fgat_reref fixed      --fgat_pooling gated

    # Ablation 3: no Laplacian rereferencing
    # If this drops → reref is important
    run_ablation "fgat_no_reref"     "$S" "$T" \
        --fgat_graph_bias true  --fgat_graph_type multiband \
        --fgat_reref none       --fgat_pooling gated

    # Ablation 4: mean pooling instead of gated attention
    # If this drops → gated pooling is important
    run_ablation "fgat_mean_pool"    "$S" "$T" \
        --fgat_graph_bias true  --fgat_graph_type multiband \
        --fgat_reref fixed      --fgat_pooling mean

    echo ""
done

echo "========================================================"
echo "Ablations complete. Results in: $SAVE_BASE"
echo ""
echo "Parse with:"
echo "  python parse_fgat_ablations.py $SAVE_BASE"
echo "========================================================"
