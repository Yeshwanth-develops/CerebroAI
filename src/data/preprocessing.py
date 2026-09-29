"""
Robust Medical Imaging Preprocessing Pipeline for ADNI 3D T1-weighted MRI volumes.
Provides subject-level train/val/test splitting, slice extraction, and normalization.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm


def subject_level_split(
    raw_mri_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    classes: Tuple[str, ...] = ("ad", "nor"),
    random_seed: int = 42,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Performs patient/subject-level dataset splitting to guarantee zero data leakage
    across train, validation, and test subsets.
    """
    np.random.seed(random_seed)
    raw_path = Path(raw_mri_dir)
    out_path = Path(output_dir)

    split_records: Dict[str, Dict[str, List[str]]] = {
        "train": {cls: [] for cls in classes},
        "val": {cls: [] for cls in classes},
        "test": {cls: [] for cls in classes},
    }

    for cls in classes:
        cls_dir = raw_path / cls
        if not cls_dir.exists():
            print(f"[Warning] Class directory not found: {cls_dir}")
            continue

        subjects = [p.name for p in cls_dir.iterdir() if p.is_dir()]
        np.random.shuffle(subjects)

        n_total = len(subjects)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_subs = subjects[:n_train]
        val_subs = subjects[n_train : n_train + n_val]
        test_subs = subjects[n_train + n_val :]

        for sub in train_subs:
            src = cls_dir / sub
            dst = out_path / "train" / cls / sub
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() and src.exists():
                shutil.copytree(src, dst)
            split_records["train"][cls].append(sub)

        for sub in val_subs:
            src = cls_dir / sub
            dst = out_path / "val" / cls / sub
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() and src.exists():
                shutil.copytree(src, dst)
            split_records["val"][cls].append(sub)

        for sub in test_subs:
            src = cls_dir / sub
            dst = out_path / "test" / cls / sub
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() and src.exists():
                shutil.copytree(src, dst)
            split_records["test"][cls].append(sub)

        print(
            f"Class '{cls}' - Split {n_total} subjects -> Train: {len(train_subs)}, Val: {len(val_subs)}, Test: {len(test_subs)}"
        )

    return split_records


def nifti_to_slices(
    nii_file_path: str,
    target_size: Tuple[int, int] = (192, 160),
    slice_range_percent: Tuple[float, float] = (0.45, 0.80),
) -> np.ndarray:
    """
    Loads a 3D NIfTI volume, extracts axial slices, filters middle informative brain
    regions (e.g. hippocampus, ventricles), normalizes intensities to [0, 1], and resizes.
    """
    nii = nib.load(nii_file_path)
    volume = np.asanyarray(nii.dataobj, dtype=np.float32)

    total_slices = volume.shape[0]
    start_idx = int(total_slices * slice_range_percent[0])
    end_idx = int(total_slices * slice_range_percent[1])

    extracted = []
    for i in range(start_idx, end_idx):
        slice_2d = volume[i, :, :]
        min_val, max_val = np.amin(slice_2d), np.amax(slice_2d)
        if max_val > min_val:
            norm_slice = (slice_2d - min_val) / (max_val - min_val)
        else:
            norm_slice = slice_2d

        img = Image.fromarray((norm_slice * 255.0).astype(np.uint8))
        if img.mode != "L":
            img = img.convert("L")
        img_resized = img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)

        norm_arr = np.array(img_resized, dtype=np.float32) / 255.0
        extracted.append(norm_arr)

    if not extracted:
        return np.empty((0, target_size[0], target_size[1], 1), dtype=np.float32)

    return np.expand_dims(np.stack(extracted, axis=0), axis=-1)


def batch_convert_dataset(
    split_dir: str,
    output_png_dir: str,
    target_size: Tuple[int, int] = (192, 160),
) -> None:
    """
    Scans a split directory containing .nii files and exports normalized, informative
    PNG axial slices with patient and sequence IDs preserved.
    """
    split_path = Path(split_dir)
    out_path = Path(output_png_dir)

    nii_files = list(split_path.glob("**/*.nii")) + list(split_path.glob("**/*.nii.gz"))
    print(f"Found {len(nii_files)} NIfTI files in {split_dir}")

    for file_path in tqdm(nii_files, desc="Converting NIfTI to PNG"):
        rel_path = file_path.relative_to(split_path)
        slices = nifti_to_slices(str(file_path), target_size=target_size)

        dest_folder = out_path / rel_path.parent
        dest_folder.mkdir(parents=True, exist_ok=True)

        base_name = file_path.stem
        for idx, slice_img in enumerate(slices):
            img_2d = (slice_img.squeeze() * 255.0).astype(np.uint8)
            im = Image.fromarray(img_2d, mode="L")
            im.save(dest_folder / f"{base_name}_slice_{idx:03d}.png")
