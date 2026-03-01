#!/bin/bash

DEVICE=${1:-cuda:2}
dataset="PEMS04"

EVAL_MISS_RATES=(0.8)
TRAIN_MISS_TYPES=(SR-TC)
# CFG
CFG_SCALES=(1.0)
# FBG
FBG_MODES=("cluster" "global" "spatial")
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

RESULTS_FILE="./results/eval_${dataset}_run_${CURRENT_TIME}.csv"
mkdir -p "$(dirname "$RESULTS_FILE")"

LOGFILE="./logs/log_${dataset}_${CURRENT_TIME}.log"
mkdir -p "$(dirname "$LOGFILE")"

echo "========================================="
echo "      Starting batch model evaluation, using DEVICE=$DEVICE"
echo "========================================="
for type in "${TRAIN_MISS_TYPES[@]}"; do
    for rate in "${EVAL_MISS_RATES[@]}"; do
        echo "All evaluation results will be saved to: ${RESULTS_FILE}"
        echo ""
        SAVENAME="${dataset}_${type}_${rate}"
        UNCOND="./params/${SAVENAME}_uncond.pth"
        COND="./params/${SAVENAME}_cond.pth"

        echo "###############################################################"
        echo "  Loading model name: ${SAVENAME}, missing type: ${type}, missing rate: ${rate}"
        echo "###############################################################"

        echo ""
        echo "--- [1/2] Starting evaluation of CFG guidance method ---"
        for cfg in "${CFG_SCALES[@]}"; do
            echo "  --> Evaluating CFG Scale: ${cfg}"
            python run.py \
                --config config/SR-TC/PEMS04.conf \
                --mode eval \
                --dataset ${dataset} \
                --miss_type ${type} \
                --miss_rate ${rate} \
                --cond_path ${COND} \
                --uncond_path ${UNCOND} \
                --guidance cfg \
                --cfg_scale ${cfg} \
                --results_file ${RESULTS_FILE} \
                --device ${DEVICE} \
                --savename ${SAVENAME} \
                --logfile  ${LOGFILE} \
                --seed 1
            echo "-----------------------------------------"
        done

        echo ""
        echo "--- [2/2] Starting evaluation of FBG guidance method ---"
        for fbg_mode in "${FBG_MODES[@]}"; do
            echo "  --> Evaluating FBG Mode: ${fbg_mode}"
            python run.py \
                --config config/SR-TC/PEMS04.conf \
                --mode eval \
                --dataset ${dataset} \
                --miss_type ${type} \
                --miss_rate ${rate} \
                --cond_path ${COND} \
                --uncond_path ${UNCOND} \
                --guidance fbg \
                --fbg_mode ${fbg_mode} \
                --results_file ${RESULTS_FILE} \
                --device ${DEVICE} \
                --savename ${SAVENAME} \
                --logfile  ${LOGFILE} \
                --seed 1
            echo "-----------------------------------------"
        done

    done
done
echo ""
echo "========================================="
echo "      All evaluations completed!         "
echo "      Results have been saved to ${RESULTS_FILE}"
echo "========================================="

echo ""
echo "Evaluation results summary:"
cat ${RESULTS_FILE}