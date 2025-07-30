#!/bin/bash

# =================================================================
#                 CSDI 批量训练脚本
# =================================================================

# --- 脚本配置 ---
CONFIG_FILE="config/PEMS08.conf"
DEVICE=${1:-cuda:6}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# RESULTS_FILE="./results/train_summary_${TIMESTAMP}.csv"
RESULTS_FILE="./results/train_summary_loss.csv"

TRAIN_MISS_RATES=(0.9)

echo "========================================="
echo "  开始批量训练模型 ,使用 DEVICE=$DEVICE"
echo "========================================="

for rate in "${TRAIN_MISS_RATES[@]}"; do
    
    SAVE_NAME="pems08_100_100_SCTC_depart_${rate}"
    
    echo "#########################################"
    echo "  开始训练：缺失率 ${rate}"
    echo "  模型将保存为: ${SAVE_NAME}.pth"
    echo "#########################################"
    
    python run.py \
        --config ${CONFIG_FILE} \
        --device ${DEVICE} \
        --mode train \
        --miss_rate ${rate} \
        --savename ${SAVE_NAME} \
        --results_file ${RESULTS_FILE}
    echo ""
    echo "--- 缺失率 ${rate} 的模型训练完成 ---"
    echo "#########################################"
    echo ""

done

echo ""
echo "========================================="
echo "      所有训练任务完成！                "
echo "========================================="