#!/bin/bash
# Bandpower benchmark — linear classifier with laplacian-bandpower preprocessing
# Covers all 12 Lite subject/trial pairs × 15 tasks × WithinSession + CrossSession + CrossSubject
#
# Usage:
#   cd examples/
#   ROOT_DIR_BRAINTREEBANK=/storage/eg99/braintreebank_data \
#     conda run -n neuroprobe bash run_bandpower_benchmark.sh

set -e

SAVE_BASE="${SAVE_BASE:-eval_results/bandpower_benchmark}"
ALL_TASKS="onset,speech,volume,delta_volume,pitch,word_index,word_gap,gpt2_surprisal,word_head_pos,word_part_speech,word_length,global_flow,local_flow,frame_brightness,face_num"

export ROOT_DIR_BRAINTREEBANK="${ROOT_DIR_BRAINTREEBANK:-/storage/eg99/braintreebank_data}"
export NEUROPROBE_FEATURES_FILE="${NEUROPROBE_FEATURES_FILE:-features.csv}"

SUBJECT_TRIALS=(
    "1 1" "1 2"
    "2 0" "2 4"
    "3 0" "3 1"
    "4 0" "4 1"
    "7 0" "7 1"
    "10 0" "10 1"
)

run_eval() {
    local SUBJECT_ID=$1
    local TRIAL_ID=$2
    local SPLIT=$3
    local SAVE_DIR="${SAVE_BASE}/${SPLIT}/sub${SUBJECT_ID}_trial${TRIAL_ID}"
    echo "  sub${SUBJECT_ID} trial${TRIAL_ID} ${SPLIT} → ${SAVE_DIR}"
    python eval_population.py \
        --classifier_type linear \
        --preprocess.type laplacian-bandpower \
        --subject_id "$SUBJECT_ID" \
        --trial_id "$TRIAL_ID" \
        --eval_name "$ALL_TASKS" \
        --split_type "$SPLIT" \
        --only_1second \
        --verbose \
        --if_exists skip \
        --save_dir "$SAVE_DIR"
}

echo "========================================================"
echo "Bandpower Benchmark"
echo "  Preprocess : laplacian-bandpower"
echo "  Output     : $SAVE_BASE"
echo "========================================================"

echo "--- WithinSession ---"
for ST in "${SUBJECT_TRIALS[@]}"; do
    run_eval $(echo $ST) "WithinSession"
done

echo "--- CrossSession ---"
for ST in "${SUBJECT_TRIALS[@]}"; do
    run_eval $(echo $ST) "CrossSession"
done

echo "--- CrossSubject ---"
for ST in "${SUBJECT_TRIALS[@]}"; do
    SUBJECT_ID=$(echo "$ST" | cut -d' ' -f1)
    TRIAL_ID=$(echo "$ST" | cut -d' ' -f2)
    if [ "$SUBJECT_ID" = "2" ]; then
        echo "  Skipping sub2 (training subject)"
        continue
    fi
    run_eval "$SUBJECT_ID" "$TRIAL_ID" "CrossSubject"
done

echo "========================================================"
echo "Done. Results in: $SAVE_BASE"
echo "========================================================"
