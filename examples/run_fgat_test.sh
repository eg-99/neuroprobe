#!/bin/bash
# Quick integration test for FGAT — runs one subject/task/bin to verify the pipeline.
# Expected runtime: ~3-5 minutes on Devon GPU.
#
# Usage (from examples/ dir on Devon):
#   source /storage/eg99/devon_init.sh
#   HOME=/storage/eg99 bash run_fgat_test.sh

set -e

export ROOT_DIR_BRAINTREEBANK="${ROOT_DIR_BRAINTREEBANK:-/storage/eg99/braintreebank_data}"
export NEUROPROBE_FEATURES_FILE="${NEUROPROBE_FEATURES_FILE:-features.csv}"

echo "========================================================"
echo "FGAT Integration Test"
echo "  Subject: sub1 trial1 | Task: onset | Split: WithinSession"
echo "  Time bin: 0-1s only"
echo "========================================================"

python eval_population.py \
    --classifier_type fgat \
    --subject_id 1 \
    --trial_id 1 \
    --eval_name onset \
    --split_type WithinSession \
    --only_1second \
    --only_bin_start 0 \
    --only_bin_end 1 \
    --verbose \
    --save_dir eval_results/fgat_test

echo ""
echo "Done. Check eval_results/fgat_test/ for output JSON."
echo "Verify: test_roc_auc is present and no errors above."
