#!/bin/bash

MAX_PARALLEL=9
count=0
GPUS=(0 1 2)  # Devices to use

for seed in {0..8}
do
    gpu_idx=$((count % ${#GPUS[@]}))
    gpu_id=${GPUS[$gpu_idx]}

    echo "Launching job with seed $seed on cuda:$gpu_id"
    python train_upper_diffusion.py \
        --true_score \
        --sigma_min 0.1 \
        --sigma_max 20 \
        --power 2.0 \
        --device "cuda:$gpu_id" \
        --weight sigma2 \
        --seed "$seed" &

    ((count++))
    # Wait whenever we have launched MAX_PARALLEL background jobs
    if (( count % MAX_PARALLEL == 0 )); then
        wait
    fi
done
# do
#     gpu_idx=$((count % ${#GPUS[@]}))
#     gpu_id=${GPUS[$gpu_idx]}

#     echo "Launching job with seed $seed on cuda:$gpu_id"
#     python train_dikl.py \
#         --sigma_min 1.1 \
#         --sigma_max 40 \
#         --score_step 1 \
#         --device "cuda:$gpu_id" \
#         --weight sigma2 \
#         --seed "$seed" &

#     ((count++))
#     # Wait whenever we have launched MAX_PARALLEL background jobs
#     if (( count % MAX_PARALLEL == 0 )); then
#         wait
#     fi
# done

# wait  # Wait for any remaining jobs to finish


