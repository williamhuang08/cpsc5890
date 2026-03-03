#!/bin/bash

CONFIG_TEMPLATE="config/lift_cube.yaml"
TMP_CONFIG="config/tmp_lift_cube.yaml"
BASE_SAVE_DIR="asset/policy"

mkdir -p ${BASE_SAVE_DIR}

SCHEDULERS=("linear" "squaredcos_cap_v2" "scaled_linear")
STEPS=(50 100 200)

for scheduler in "${SCHEDULERS[@]}"; do
  for steps in "${STEPS[@]}"; do

    FOLDER_NAME="${scheduler}_${steps}"
    SAVE_NAME="${FOLDER_NAME}_model"

    echo "============================================="
    echo "Running scheduler=${scheduler}, steps=${steps}"
    echo "Checkpoints → asset/policy/${FOLDER_NAME}/"
    echo "CSV log     → asset/training/DiT__state_${SAVE_NAME}_loss_log.csv"
    echo "============================================="

    # Copy original config
    cp ${CONFIG_TEMPLATE} ${TMP_CONFIG}

    # Update diffusion steps
    sed -i "s/^num_diffusion_steps:.*/num_diffusion_steps: ${steps}/" ${TMP_CONFIG}

    # Update save model name (NO slash to avoid CSV issues)
    sed -i "s|^save_model_name:.*|save_model_name: ${SAVE_NAME}|" ${TMP_CONFIG}

    # Run training
    python -m scripts.ddpm \
      --mode train \
      --config ${TMP_CONFIG} \
      --scheduler ${scheduler}

  done
done

rm ${TMP_CONFIG}

echo "✅ All experiments completed."