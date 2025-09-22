#!/bin/bash

MAX_PARALLEL=6  # 3 jobs per GPU * 3 GPUs
count=0
GPUS=(0 1)

SIGMA_MIN_VALUES=(1.0 1.1 1.2 1.3 1.4 1.5)  # put your desired sigma_min values here

for sigma_min in "${SIGMA_MIN_VALUES[@]}"
do
    gpu_idx=$((count % ${#GPUS[@]}))
    gpu_id=${GPUS[$gpu_idx]}

    echo "Running with sigma_min=$sigma_min on cuda:$gpu_id"
    python train_dikl.py \
        --sigma_min "$sigma_min" \
        --sigma_max 40 \
        --device "cuda:$gpu_id" \
        --weight sigma2 \
        --seed 0 \
        --power 2.0 &

    ((count++))

    if (( count % MAX_PARALLEL == 0 )); then
        wait
    fi
done

wait

