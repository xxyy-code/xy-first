#!/usr/bin/env bash
# Pretrain CSD-MM — edit variables below (or export them before calling).
set -euo pipefail
cd "$(dirname "$0")/.."

########## configurable paths / hyper-params ##########
GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-/path/to/BraTS2020_TrainingData}"          # contains vol/ seg/
TRAIN_FILE="${TRAIN_FILE:-/path/to/BraTS2020_train.txt}"
TEST_FILE="${TEST_FILE:-/path/to/BraTS2020_test.txt}"
# optional OOD eval during training (leave empty to skip)
OOD_ROOT="${OOD_ROOT:-}"
OOD_TEST_FILE="${OOD_TEST_FILE:-}"

SAVE_PATH="${SAVE_PATH:-./output/pth}"
CSV_PATH="${CSV_PATH:-./output/csv/moegw.csv}"
RESUME="${RESUME:-}"                 # e.g. ./output/pth/model_last.pth
PRETRAIN="${PRETRAIN:-}"             # optional init weights
EVAL_CKPT="${EVAL_CKPT:-}"           # optional final eval ckpt

# optimization
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
ITER_PER_EPOCH="${ITER_PER_EPOCH:-110}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_CLS="${NUM_CLS:-4}"
SEED="${SEED:-1024}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# schedule / losses
REGION_FUSION_START_EPOCH="${REGION_FUSION_START_EPOCH:-34}"
ROU_VAR_WEIGHT="${ROU_VAR_WEIGHT:-0.01}"
ROU_ENTRO_WEIGHT="${ROU_ENTRO_WEIGHT:-0.005}"

# augmentation
CROP_SIZE="${CROP_SIZE:-80}"
ROTATE_ANGLE="${ROTATE_ANGLE:-10}"
INTENSITY_SHIFT="${INTENSITY_SHIFT:-0.1}"
#######################################################

CMD=(
  python train_gw.py
  --dataname "${DATA_ROOT}"
  --train_file "${TRAIN_FILE}"
  --test_file "${TEST_FILE}"
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
  --rou_var_weight "${ROU_VAR_WEIGHT}"
  --rou_entro_weight "${ROU_ENTRO_WEIGHT}"
  --crop_size "${CROP_SIZE}"
  --rotate_angle "${ROTATE_ANGLE}"
  --intensity_shift "${INTENSITY_SHIFT}"
)

if [[ -n "${RESUME}" ]]; then
  CMD+=(--resume "${RESUME}")
fi
if [[ -n "${PRETRAIN}" ]]; then
  CMD+=(--pretrain "${PRETRAIN}")
fi
if [[ -n "${EVAL_CKPT}" ]]; then
  CMD+=(--eval_ckpt "${EVAL_CKPT}")
fi
if [[ -n "${OOD_ROOT}" && -n "${OOD_TEST_FILE}" ]]; then
  CMD+=(--ood_root "${OOD_ROOT}" --ood_test_file "${OOD_TEST_FILE}")
fi

echo "Running: CUDA_VISIBLE_DEVICES=${GPU} ${CMD[*]}"
CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}"
