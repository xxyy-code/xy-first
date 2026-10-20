#!/usr/bin/env bash
# CW-TTA on a target domain (PED / MET / ...) — edit variables below.
set -euo pipefail
cd "$(dirname "$0")/.."

########## configurable paths / hyper-params ##########
GPU="${GPU:-0}"

# target domain (to adapt / evaluate)
DATA_ROOT="${DATA_ROOT:-/path/to/Brast2023_PED_trainingdata}"
TRAIN_FILE="${TRAIN_FILE:-/path/to/target_list.txt}"
TEST_FILE="${TEST_FILE:-/path/to/target_list.txt}"

# source domain (for class-prior estimation)
SOURCE_ROOT="${SOURCE_ROOT:-/path/to/BraTS2020_TrainingData}"
SOURCE_TRAIN_FILE="${SOURCE_TRAIN_FILE:-/path/to/BraTS2020_train.txt}"

# pretrained CSD-MM checkpoint
PRETRAIN="${PRETRAIN:-./output/pth/model_full_moegw_500.pth}"
RESUME="${RESUME:-}"

SAVE_PATH="${SAVE_PATH:-./output/ada_pth}"
CSV_PATH="${CSV_PATH:-./output/csv/cw_tta.csv}"

# optimization
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
ITER_PER_EPOCH="${ITER_PER_EPOCH:-99}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_CLS="${NUM_CLS:-4}"
SEED="${SEED:-1024}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# schedule / TTA-specific
REGION_FUSION_START_EPOCH="${REGION_FUSION_START_EPOCH:-0}"
ALIGN_TUMOR_WEIGHT="${ALIGN_TUMOR_WEIGHT:-10.0}"
MASS_DELTA_RATIO="${MASS_DELTA_RATIO:-0.3}"
EVAL_EVERY="${EVAL_EVERY:-2}"

# augmentation
CROP_SIZE="${CROP_SIZE:-80}"
ROTATE_ANGLE="${ROTATE_ANGLE:-10}"
INTENSITY_SHIFT="${INTENSITY_SHIFT:-0.1}"
#######################################################

CMD=(
  python test_time_ada_distribution_align.py
  --dataname "${DATA_ROOT}"
  --train_file "${TRAIN_FILE}"
  --test_file "${TEST_FILE}"
  --source_root "${SOURCE_ROOT}"
  --source_train_file "${SOURCE_TRAIN_FILE}"
  --pretrain "${PRETRAIN}"
  --savepath "${SAVE_PATH}"
  --csv_path "${CSV_PATH}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --num_epochs "${NUM_EPOCHS}"
  --iter_per_epoch "${ITER_PER_EPOCH}"
  --batch_size "${BATCH_SIZE}"
  --num_cls "${NUM_CLS}"
  --seed "${SEED}"
  --num_workers "${NUM_WORKERS}"
  --region_fusion_start_epoch "${REGION_FUSION_START_EPOCH}"
  --align_tumor_weight "${ALIGN_TUMOR_WEIGHT}"
  --mass_delta_ratio "${MASS_DELTA_RATIO}"
  --eval_every "${EVAL_EVERY}"
  --crop_size "${CROP_SIZE}"
  --rotate_angle "${ROTATE_ANGLE}"
  --intensity_shift "${INTENSITY_SHIFT}"
)

if [[ -n "${RESUME}" ]]; then
  CMD+=(--resume "${RESUME}")
fi

echo "Running: CUDA_VISIBLE_DEVICES=${GPU} ${CMD[*]}"
CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}"
