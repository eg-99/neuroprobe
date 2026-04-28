#!/bin/bash
# Extended FGAT ablation study — paper-quality, all 12 pairs × 15 tasks × WS + CS
#
# Variants:
#   fgat_full      — full model (D=96, layers=2, multiband graph, gated pooling)
#   fgat_no_graph  — no graph bias (tests: does graph contribute anything?)
#   fgat_mean_pool — mean pooling instead of gated attention (tests: does attention pooling matter?)
#   linear         — linear STFT baseline (direct comparison in same table)
#
# Parallelization: 3 GPUs, 4 subject/trial pairs each (sequential within GPU).
# Expected runtime: ~18-24 hours on Devon (3 GPUs).
#
# Usage (from examples/):
#   export ROOT_DIR_BRAINTREEBANK=/storage/eg99/braintreebank_data
#   export NEUROPROBE_FEATURES_FILE=features.csv
#   HOME=/storage/eg99 nohup bash run_fgat_ablations_full.sh \
#     > eval_results/fgat_ablations_full/main.log 2>&1 &

set -e

export ROOT_DIR_BRAINTREEBANK="${ROOT_DIR_BRAINTREEBANK:-/storage/eg99/braintreebank_data}"
export NEUROPROBE_FEATURES_FILE="${NEUROPROBE_FEATURES_FILE:-features.csv}"

ALL_TASKS="onset,speech,volume,delta_volume,pitch,word_index,word_gap,gpt2_surprisal,word_head_pos,word_part_speech,word_length,global_flow,local_flow,frame_brightness,face_num"
SAVE_BASE="eval_results/fgat_ablations_full"
mkdir -p "$SAVE_BASE"

# All 12 Lite subject/trial pairs split across 3 GPUs (4 pairs each)
GPU0_PAIRS=("1 1" "1 2" "2 0" "2 4")
GPU1_PAIRS=("3 0" "3 1" "4 0" "4 1")
GPU2_PAIRS=("7 0" "7 1" "10 0" "10 1")

run_fgat() {
    local NAME=$1
    local SUBJECT_ID=$2
    local TRIAL_ID=$3
    local SPLIT=$4
    shift 4
    local SAVE_DIR="${SAVE_BASE}/${NAME}/${SPLIT}/sub${SUBJECT_ID}_trial${TRIAL_ID}"
    echo "  [${NAME}] sub${SUBJECT_ID} trial${TRIAL_ID} ${SPLIT}"
    python eval_population.py \
        --classifier_type fgat \
        --subject_id "$SUBJECT_ID" \
        --trial_id "$TRIAL_ID" \
        --eval_name "$ALL_TASKS" \
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

run_linear() {
    local SUBJECT_ID=$1
    local TRIAL_ID=$2
    local SPLIT=$3
    local SAVE_DIR="${SAVE_BASE}/linear/${SPLIT}/sub${SUBJECT_ID}_trial${TRIAL_ID}"
    echo "  [linear] sub${SUBJECT_ID} trial${TRIAL_ID} ${SPLIT}"

    # CrossSubject: skip subject 2 (training subject)
    if [ "$SPLIT" = "CrossSubject" ] && [ "$SUBJECT_ID" = "2" ]; then
        echo "  Skipping sub2 for CrossSubject"
        return
    fi

    python eval_population.py \
        --classifier_type linear \
        --preprocess.type laplacian-stft_abs \
        --subject_id "$SUBJECT_ID" \
        --trial_id "$TRIAL_ID" \
        --eval_name "$ALL_TASKS" \
        --split_type "$SPLIT" \
        --only_1second \
        --verbose \
        --if_exists skip \
        --save_dir "$SAVE_DIR"
}

run_pair() {
    local SUBJECT_ID=$1
    local TRIAL_ID=$2

    for SPLIT in WithinSession CrossSession; do
        run_fgat "fgat_full" "$SUBJECT_ID" "$TRIAL_ID" "$SPLIT" \
            --fgat_graph_bias true --fgat_graph_type multiband \
            --fgat_reref fixed     --fgat_pooling gated

        run_fgat "fgat_no_graph" "$SUBJECT_ID" "$TRIAL_ID" "$SPLIT" \
            --fgat_graph_bias false --fgat_graph_type multiband \
            --fgat_reref fixed      --fgat_pooling gated

        run_fgat "fgat_mean_pool" "$SUBJECT_ID" "$TRIAL_ID" "$SPLIT" \
            --fgat_graph_bias true --fgat_graph_type multiband \
            --fgat_reref fixed     --fgat_pooling mean

        run_linear "$SUBJECT_ID" "$TRIAL_ID" "$SPLIT"
    done
}

run_gpu_worker() {
    local GPU=$1
    shift
    local PAIRS=("$@")
    export CUDA_VISIBLE_DEVICES=$GPU
    local LOG="${SAVE_BASE}/log_gpu${GPU}.txt"

    for ST in "${PAIRS[@]}"; do
        S=$(echo "$ST" | cut -d' ' -f1)
        T=$(echo "$ST" | cut -d' ' -f2)
        run_pair "$S" "$T"
    done
}

echo "========================================================"
echo "FGAT Extended Ablation Study (paper-quality)"
echo "  Variants : fgat_full / fgat_no_graph / fgat_mean_pool / linear"
echo "  Pairs    : all 12 Lite subject/trial pairs"
echo "  Tasks    : all 15"
echo "  Splits   : WithinSession + CrossSession"
echo "  Output   : $SAVE_BASE"
echo "========================================================"

run_gpu_worker 0 "${GPU0_PAIRS[@]}" > "${SAVE_BASE}/log_gpu0.txt" 2>&1 &
run_gpu_worker 1 "${GPU1_PAIRS[@]}" > "${SAVE_BASE}/log_gpu1.txt" 2>&1 &
run_gpu_worker 2 "${GPU2_PAIRS[@]}" > "${SAVE_BASE}/log_gpu2.txt" 2>&1 &

echo "3 GPU workers running in background."
echo "Monitor with:"
echo "  tail -f ${SAVE_BASE}/log_gpu0.txt"
echo "  tail -f ${SAVE_BASE}/log_gpu1.txt"
echo "  tail -f ${SAVE_BASE}/log_gpu2.txt"

wait

echo "========================================================"
echo "Done. Results in: $SAVE_BASE"
echo "========================================================"
