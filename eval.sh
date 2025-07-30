#!/bin/bash

# =================================================================
#                 CSDI 批量评估脚本 (CFG & FBG)
# =================================================================

# --- 脚本配置 ---
CONFIG_FILE="config/PEMS08.conf"
DEVICE=${1:-cuda:7}
MODEL_LOAD_DIR="./params/"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="./results/eval_summary_${TIMESTAMP}.csv"
EVAL_MISS_RATES=(0.9)

# CFG
CFG_SCALES=(1.2 1.0 1.4 1.6)

# FBG
FBG_MODES=("cluster", "global" "spatial")

# --- 准备工作 ---
mkdir -p "$(dirname "$RESULTS_FILE")"
echo "所有评估结果将保存在: ${RESULTS_FILE}"
echo ""


echo "========================================="
echo "      开始批量评估模型 , 使用 DEVICE=$DEVICE"
echo "========================================="

for rate in "${EVAL_MISS_RATES[@]}"; do
    MODEL_NAME="pems08_100_100_depart_${rate}.pth"
    MODEL_PATH="${MODEL_LOAD_DIR}${MODEL_NAME}"
    
#    if [ ! -f "${MODEL_PATH}" ]; then
#        echo "警告：找不到模型文件 ${MODEL_PATH}，跳过对缺失率 ${rate} 的评估。"
#        continue
#    fi
#
#    echo "#########################################"
#    echo "  加载模型: ${MODEL_PATH} (缺失率: ${rate})"
#    echo "#########################################"
    
    # --- (A) 评估 CFG ---
    echo ""
    echo "--- [1/2] 开始评估 CFG 指导方法 ---"
    for cfg in "${CFG_SCALES[@]}"; do
        echo "  --> 评估 CFG Scale: ${cfg}"
        python run_convert.py \
            --config ${CONFIG_FILE} \
            --device ${DEVICE} \
            --mode eval \
            --miss_rate ${rate} \
            --guidance cfg \
            --cfg_scale ${cfg} \
            --results_file ${RESULTS_FILE}
        echo "-----------------------------------------"
    done
    
    #--- (B) 评估 FBG ---
    echo ""
    echo "--- [2/2] 开始评估 FBG 指导方法 ---"
    for fbg_mode in "${FBG_MODES[@]}"; do
        echo "  --> 评估 FBG Mode: ${fbg_mode}"
        python run.py \
            --config ${CONFIG_FILE} \
            --device ${DEVICE} \
            --mode eval \
            --miss_rate ${rate} \
            --guidance fbg \
            --fbg_mode ${fbg_mode} \
            --results_file ${RESULTS_FILE}
        echo "-----------------------------------------"
    done

done

echo ""
echo "========================================="
echo "      所有评估完成！                "
echo "      结果已保存在 ${RESULTS_FILE}"
echo "========================================="

echo ""
echo "评估结果摘要:"
cat ${RESULTS_FILE}