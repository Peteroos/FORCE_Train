#!/usr/bin/env python3
"""Convert Mayo LDCT DICOM (.IMA) files to .npy (HU, float32, 512x512).

Usage:
    python convert_dicom_to_npy.py \
        --input_dir "/mnt/e/Data/MayoLDCT/Training_Image_Data/Training_Image_Data/1mm B30/FD_1mm/full_1mm" \
        --output_dir ./data/mayo_npy
"""
import argparse
import glob
import os

import numpy as np
import pydicom


def dicom_to_hu(ds):
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return ds.pixel_array.astype(np.float32) * slope + intercept


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ima_files = sorted(
        glob.glob(os.path.join(args.input_dir, "**", "*.IMA"), recursive=True)
    )
    if not ima_files:
        ima_files = sorted(
            glob.glob(os.path.join(args.input_dir, "**", "*.dcm"), recursive=True)
        )
    print(f"Found {len(ima_files)} DICOM files in {args.input_dir}")

    saved = 0
    for i, fpath in enumerate(ima_files):
        try:
            ds = pydicom.dcmread(fpath, force=True)
            hu = dicom_to_hu(ds)
        except Exception as e:
            print(f"  [SKIP] {fpath}: {e}")
            continue

        if hu.shape != (512, 512):
            print(f"  [SKIP] {fpath}: unexpected shape {hu.shape}")
            continue

        rel = os.path.relpath(fpath, args.input_dir)
        parts = rel.replace("\\", "/").split("/")
        patient = parts[0] if len(parts) > 1 else "unknown"
        basename = os.path.splitext(os.path.basename(fpath))[0]

        patient_dir = os.path.join(args.output_dir, patient)
        os.makedirs(patient_dir, exist_ok=True)
        out_path = os.path.join(patient_dir, f"{basename}.npy")
        np.save(out_path, hu)
        saved += 1

        if (i + 1) % 200 == 0 or i == 0:
            print(f"  [{i+1}/{len(ima_files)}] {patient}/{basename}")

    print(f"\nDone. Saved {saved} .npy files to {args.output_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_dir",
        default="/mnt/e/Data/MayoLDCT/Training_Image_Data/Training_Image_Data/1mm B30/FD_1mm/full_1mm",
    )
    p.add_argument("--output_dir", default="./data/mayo_npy")
    return p.parse_args()


if __name__ == "__main__":
    main()
