#!/bin/bash
# FGAT benchmark — compare 4 model configs across 3 subjects × 5 tasks.
#
# Configs tested:
#   fgat_small  — D=64  layers=2  (fast, lower capacity)
#   fgat_medium — D=96  layers=2  (default)
#   fgat_large  — D=128 layers=3  (higher capacity)
#   fgat_no_graph — D=96 layers=2, no graph bias (pure transformer ablation)
#
# Subjects: sub1_trial1, sub3_trial0, sub7_trial0
# Tasks: onset, speech, volume, gpt2_surprisal, word_length
# Split: WithinSession only (fastest, cleanest signal)
#
# Expected runtime: ~6-12 hours on Devon GPU.
#
# Usage:
#   source /storage/eg99/devon_init.sh
#   HOME=/storage/eg99 ROOT_DIR_BRAINTREEBANK=/storage/eg99/braintreebank_data \
#     nohup bash run_fgat_benchmark.sh > /storage/eg99/neuroprobe/logs/fgat_benchmark.log 2>&1 &

set -e

export ROOT_DIR_BRAINTREEBANK="${ROOT_DIR_BRAINTREEBANK:-/storage/eg99/braintreebank_data}"
export NEUROPROBE_FEATURES_FILE="${NEUROPROBE_FEATURES_FILE:-features.csv}"

TASKS="onset,speech,volume,gpt2_surprisal,word_length"
SPLIT="WithinSession"
SAVE_BASE="eval_results/fgat_benchmark"

SUBJECT_TRIALS=(
    "1 1"
    "3 0"
    "7 0"
)

run_config() {
    local CONFIG_NAME=$1
    local SUBJECT_ID=$2
    local TRIAL_ID=$3
    shift 3   # remaining args are fgat flags

    local SAVE_DIR="${SAVE_BASE}/${CONFIG_NAME}/sub${SUBJECT_ID}_trial${TRIAL_ID}"
    echo "  [${CONFIG_NAME}] sub${SUBJECT_ID} trial${TRIAL_ID}"

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
        "$@"
}

echo "========================================================"
echo "FGAT Benchmark"
echo "  Configs : small / medium / large / no_graph"
echo "  Subjects: sub1_trial1, sub3_trial0, sub7_trial0"
echo "  Tasks   : $TASKS"
echo "  Output  : $SAVE_BASE"
echo "========================================================"
echo ""

for ST in "${SUBJECT_TRIALS[@]}"; do
    S=$(echo "$ST" | cut -d' ' -f1)
    T=$(echo "$ST" | cut -d' ' -f2)

    run_config "fgat_small"    "$S" "$T" --fgat_D 64  --fgat_n_layers 2 --fgat_dropout 0.1 --fgat_weight_decay 1e-3
    run_config "fgat_medium"   "$S" "$T" --fgat_D 96  --fgat_n_layers 2 --fgat_dropout 0.1 --fgat_weight_decay 1e-3
    run_config "fgat_large"    "$S" "$T" --fgat_D 128 --fgat_n_layers 3 --fgat_dropout 0.1 --fgat_weight_decay 1e-3
    run_config "fgat_no_graph" "$S" "$T" --fgat_D 96  --fgat_n_layers 2 --fgat_dropout 0.1 --fgat_weight_decay 1e-3 --fgat_graph_bias false

    echo ""
done

echo "========================================================"
echo "Benchmark complete. Results in: $SAVE_BASE"
echo ""
echo "Parse results:"
echo "  python parse_fgat_benchmark.py"
echo "========================================================"
