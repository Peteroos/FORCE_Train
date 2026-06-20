#!/bin/bash
# =============================================================================
# Unconditional PFGM++ training on Mayo full-dose CT (FORCE Interior prior).
#
# This re-train uses a FIXED HU normalization that covers the full diagnostic
# range incl. negative HU (see dataset/mayo.py: HU_MIN/HU_MAX), so the prior can
# represent fat / lung / air and display correctly in the [-160,240] window.
#
# Prereqs:
#   1. conda env with PyTorch (matching your CUDA), see requirements.txt
#   2. Training slices as .npy (HU, float32, 512x512) under DATA_DIR, e.g.:
#        python convert_dicom_to_npy.py --input_dir <DICOM dir> --output_dir ./data/mayo_npy
#      (each *.npy is a single CT slice in *standard* HU, water=0, air=-1024.)
#
# VRAM (512x512, fp16):  batch_size=2 ~13GB, batch_size=1 ~9GB.
# =============================================================================
set -e
cd "$(dirname "$0")"

# --- EDIT THESE for your machine ------------------------------------------
NUM_GPUS=1
DATA_DIR="./data/mayo_npy"          # folder of .npy/.npz OR raw DICOM (.IMA/.dcm)
CKPT=""                             # empty = from scratch; or path to resume
NCPUS=4                             # DataLoader workers per process. Raise this
                                    # (e.g. 8-16) when reading DICOM directly so
                                    # GPU does not wait on per-slice decoding.
BATCH_SIZE=2                        # 512^2 fp16: 2~13GB, 1~9GB
# --------------------------------------------------------------------------

if [ "${NUM_GPUS}" -gt 1 ]; then
    LAUNCH="torchrun --standalone --nproc_per_node=${NUM_GPUS}"
else
    LAUNCH="python"
fi

# Architecture flags below MUST match the ones used at inference time.
${LAUNCH} train_pfgm.py \
    --data_dir "${DATA_DIR}" \
    --image_size 512 \
    --in_channels 1 \
    --out_channels 1 \
    --channel_mult 1,2,4,8,16 \
    --num_res_blocks 1 \
    --dims 2 \
    --batch_size ${BATCH_SIZE} \
    --ncpus ${NCPUS} \
    --use_fp16 True \
    --lr 1e-4 \
    --lr_anneal_steps 180000 \
    --max_norm 1.0 \
    --save_interval 5000 \
    --log_interval 100 \
    --checkpointdir checkpoints_pfgm \
    --resume_checkpoint "${CKPT}"
