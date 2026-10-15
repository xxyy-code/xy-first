#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

SRC_PATH="${SRC_PATH:-/path/to/BraTS/Training}"   # contains HGG/ LGG/
TAR_PATH="${TAR_PATH:-/path/to/BraTS_none_npy}"  # output root

python preprocess.py --src_path "${SRC_PATH}" --tar_path "${TAR_PATH}"
