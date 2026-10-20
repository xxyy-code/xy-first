# CSD-MM + CW-TTA

Class-Specific Disentangled Multi-Modal Feature Representation Learning for Test-Time Adaptation in Brain Tumor Segmentation.

This repo provides:
1. **Pretrain** CSD-MM on BraTS2020 (`train_gw.py`)
2. **Adapt** at test time with CW-TTA on OOD targets (e.g. BraTS-MET / BraTS-PED) (`test_time_ada_distribution_align.py`)

---

## 1. Environment

```bash
conda create -n csdmm python=3.11 -y
conda activate csdmm

# Install PyTorch matching your CUDA (example: CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

Recommended: NVIDIA GPU with ≥11GB memory.

---

## 2. Data preparation

### 2.1 Expected layout after preprocessing

```text
DATA_ROOT/
  vol/
    <case>_vol.npy   # float32, shape [H, W, D, 4]  (FLAIR, T1ce, T1, T2)
  seg/
    <case>_seg.npy   # uint8,  labels {0,1,2,3}
```

Split lists are plain text files, one case id per line (without `_vol.npy` / `_seg.npy` suffix), e.g.:

```text
HG_BraTS20_Training_001
LG_BraTS20_Training_002
```

### 2.2 Preprocess BraTS (optional helper)

Edit paths in `scripts/run_preprocess.sh`, then:

```bash
bash scripts/run_preprocess.sh
```

Or:

```bash
python preprocess.py \
  --src_path /path/to/BraTS/Training \
  --tar_path /path/to/output_npy_root
```

`src_path` should contain `HGG/` and `LGG/` folders.

---

## 3. Train CSD-MM (source domain)

Edit the variables at the top of `scripts/run_train.sh`:

| Variable | Meaning | Default |
|----------|---------|---------|
| `DATA_ROOT` | Source data root with `vol/` and `seg/` | — |
| `TRAIN_FILE` / `TEST_FILE` | Split lists | — |
| `SAVE_PATH` / `CSV_PATH` | Checkpoint & log paths | `./output/...` |
| `GPU` | GPU id | `0` |
| `LR` / `WEIGHT_DECAY` | Optimizer | `2e-4` / `1e-4` |
| `NUM_EPOCHS` / `ITER_PER_EPOCH` / `BATCH_SIZE` | Schedule | `1000` / `110` / `2` |
| `REGION_FUSION_START_EPOCH` | Start using fuse/gw loss | `34` |
| `CROP_SIZE` / `ROTATE_ANGLE` / `INTENSITY_SHIFT` | Augmentation | `80` / `10` / `0.1` |
| `ROU_VAR_WEIGHT` / `ROU_ENTRO_WEIGHT` | Router loss weights | `0.01` / `0.005` |
| `SEED` / `NUM_WORKERS` / `NUM_CLS` | Misc | `1024` / `8` / `4` |

Then run:

```bash
bash scripts/run_train.sh
```

Equivalent Python call:

```bash
CUDA_VISIBLE_DEVICES=0 python train_gw.py \
  --dataname /path/to/BraTS2020_TrainingData \
  --train_file /path/to/BraTS2020_train.txt \
  --test_file /path/to/BraTS2020_test.txt \
  --savepath ./output/pth \
  --csv_path ./output/csv/moegw.csv \
  --lr 2e-4 \
  --batch_size 2 \
  --num_epochs 1000
```

Default training crop is `80×80×80`. Checkpoints are written under `--savepath`.

Optional OOD eval during training:

```bash
OOD_ROOT=/path/to/OOD_data OOD_TEST_FILE=/path/to/ood_test.txt bash scripts/run_train.sh
```

---

## 4. CW-TTA (target domain)

Edit `scripts/run_tta.sh`:

| Variable | Meaning | Default |
|----------|---------|---------|
| `DATA_ROOT` / `TRAIN_FILE` / `TEST_FILE` | Target domain data & lists | — |
| `SOURCE_ROOT` / `SOURCE_TRAIN_FILE` | Source priors | — |
| `PRETRAIN` | Pretrained CSD-MM `.pth` | — |
| `LR` / `NUM_EPOCHS` / `ITER_PER_EPOCH` | Adaptation schedule | `2e-5` / `10` / `99` |
| `ALIGN_TUMOR_WEIGHT` | Tumor moment-align weight | `10.0` |
| `MASS_DELTA_RATIO` | Bad-class reject threshold | `0.3` |
| `EVAL_EVERY` | Eval every N epochs | `2` |
| `CROP_SIZE` / `ROTATE_ANGLE` / `INTENSITY_SHIFT` | Augmentation | `80` / `10` / `0.1` |

```bash
bash scripts/run_tta.sh
```

Or:

```bash
CUDA_VISIBLE_DEVICES=0 python test_time_ada_distribution_align.py \
  --dataname /path/to/target_data \
  --train_file /path/to/target_list.txt \
  --test_file /path/to/target_list.txt \
  --source_root /path/to/BraTS2020_TrainingData \
  --source_train_file /path/to/BraTS2020_train.txt \
  --pretrain ./output/pth/your_checkpoint.pth \
  --savepath ./output/ada_pth \
  --lr 2e-5 \
  --num_epochs 10 \
  --batch_size 1
```

---

## 5. Project layout

```text
├── train_gw.py                          # CSD-MM pretraining
├── test_time_ada_distribution_align.py  # CW-TTA
├── preprocess.py                        # BraTS -> npy
├── mmformer_moe2.py                     # model
├── predict.py                           # evaluation helpers
├── layers.py
├── data/                                # datasets & transforms
├── utils/                               # losses, scheduler, ...
├── scripts/
│   ├── run_preprocess.sh
│   ├── run_train.sh
│   └── run_tta.sh
└── requirements.txt
```

All paths are passed via CLI / shell variables; nothing is hard-coded to a local machine path.
