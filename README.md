# FORCE Interior — Prior Re-training

Self-contained code to **re-train the unconditional PFGM++ CT prior** used by
FORCE Interior, with a **fixed HU normalization** so the prior can represent the
full diagnostic range (including negative HU: fat, lung, bowel gas, air).

自包含的代码,用于**重训 FORCE Interior 的无条件 PFGM++ CT 先验**,修正了 HU 归一化,使先验能表示**完整诊断 HU 范围(含负 HU:脂肪、肺、肠气、空气)**。

---

## Why re-train? / 为什么重训

The original prior normalized HU as `hu/2000*2-1` (standard HU, water=0) and
clamped to `[-1,1]`. Every **HU ≤ 0 was clamped onto the −1 floor**, so the
trained model could not produce negative HU and rendered fat/air as 0 HU. In the
standard soft-tissue window `[-160,240]` this makes reconstructions look
"washed". Confirmed from the prior's own samples (min ≈ 0, no negatives).

原先验用 `hu/2000*2-1` 归一化并 clamp 到 `[-1,1]`,导致**所有 HU≤0 被压到 −1 地板**,模型生成不了负 HU、把脂肪/空气显示成 0 HU,在 `[-160,240]` 软组织窗下"发灰"。

**The fix (`dataset/mayo.py`)**: map the full range `[HU_MIN, HU_MAX] = [-1024, 3071]`
linearly to `[-1,1]`:

```python
pix = (hu - HU_MIN) / (HU_MAX - HU_MIN) * 2 - 1     # hu  -> [-1,1]
hu  = (pix + 1) / 2 * (HU_MAX - HU_MIN) + HU_MIN     # pix -> hu
```

You can change `HU_MIN/HU_MAX` in `dataset/mayo.py` (e.g. `[-1000, 2000]` for a
tighter range). Larger range = full fidelity but slightly less soft-tissue
resolution per code level.

---

## Setup / 环境

```bash
conda create -n force_train python=3.10 -y && conda activate force_train
# install PyTorch for your CUDA, then:
pip install -r requirements.txt
```

## 1. Prepare data / 准备数据

The dataloader reads **raw DICOM (`.IMA`/`.dcm`) directly** as well as
`.npy`/`.npz`, recursing into sub-folders. Files must be `512×512` slices in
**standard HU** (water=0). Use **all training patients** (exclude your test
patients, e.g. L333 / L506).

dataloader **可直读 DICOM(`.IMA`/`.dcm`)**,也支持 `.npy`/`.npz`,会递归子目录。要求 512×512、标准 HU(水=0);放**所有训练患者**(排除测试患者 L333/L506)。

**Option A — no conversion (recommended): point `DATA_DIR` straight at the DICOM folder.**
直接把 `DATA_DIR` 指向 DICOM 目录即可,无需转换。DICOM 在加载时现场做 HU 换算
(`pixel * RescaleSlope + RescaleIntercept`)。需要 `pip install pydicom`。

**Option B — pre-convert to `.npy` (faster I/O for repeated epochs):**
若想要更快的反复读取,可先转成 `.npy`:

```bash
python convert_dicom_to_npy.py \
    --input_dir  /path/to/Mayo/full_dose_dicom \
    --output_dir ./data/mayo_npy
```

Then set `DATA_DIR=./data/mayo_npy`. / 然后把 `DATA_DIR` 指向它。

## 2. Train / 训练

```bash
bash train.sh        # edit NUM_GPUS / DATA_DIR / CKPT at the top
```

- Architecture flags (`--channel_mult 1,2,4,8,16 --num_res_blocks 1 --image_size 512`)
  **must match your inference config**. Checkpoints (with EMA) are written to
  `checkpoints_pfgm/model_XXX.pth` every `--save_interval` steps.
- VRAM @512² fp16: batch 2 ≈ 13 GB, batch 1 ≈ 9 GB. Multi-GPU via `torchrun`
  (set `NUM_GPUS>1`).

---

## 3. After training: matching inference changes (IMPORTANT)

The inference repo MUST use the **same** normalization, or the new checkpoint
will be decoded wrong. Update these in your inference code:

**(a) HU ↔ model domain** (e.g. `test_interior_force.py` `hu_to_norm`/`norm_to_hu`,
`prepare_interior_data.py` `hu_to_pix`/`pix_to_hu`):

```python
HU_MIN, HU_MAX = -1024.0, 3071.0
def hu_to_norm(hu):  return np.clip((hu - HU_MIN)/(HU_MAX-HU_MIN)*2-1, -1, 1)
def norm_to_hu(x):   return (np.clip(x,-1,1)+1)/2*(HU_MAX-HU_MIN)+HU_MIN
```

**(b) pixel ↔ attenuation** for OS-SART data consistency (`diffusion/edm.py`
`pix2atten`/`atten2pix`). These must stay consistent with (a). With water μ =
0.0192 /px and standard HU (`μ = 0.0192*(1+HU/1000)`):

```python
def pix2atten(x):  return 0.0192*(1 + norm_to_hu(x)/1000.0)
def atten2pix(mu): return hu_to_norm((mu/0.0192 - 1)*1000.0)
```

**(c)** Re-evaluate the metric window. With negative HU now representable, you
can report PSNR/SSIM in the standard soft-tissue window `[-160,240]` (no longer
unfair), and the qualitative figures no longer need the de-floor post-processing.

训练后,推理端必须用**同一套**归一化((a)/(b)),否则新 checkpoint 解码会错;之后即可在标准 `[-160,240]` 窗下公平地算指标、出图,无需 de-floor 后处理。

---

## Files / 文件

```
train_pfgm.py            training entry
train.sh                 launcher (edit paths)
convert_dicom_to_npy.py  DICOM -> HU .npy
dataset/mayo.py          dataloader + FIXED HU normalization (the change)
model/                   UNet (EDM backbone)
diffusion/               PFGM++ / EDM SDE, losses, schedulers
utils/                   training loop, model factory, logging, dist
```
