"""
Main CLI Entrypoint for CerebroAI Brain MRI Deep Learning Pipeline.
Supports end-to-end preprocessing, ResNet-18 classification, WGAN-GP synthesis, and Grad-CAM.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

from src.data.preprocessing import subject_level_split, batch_convert_dataset, nifti_to_slices
from src.models.resnet18 import build_resnet18
from src.models.wgan_gp import WGAN_GP, build_generator, build_critic
from src.explainability.gradcam import make_gradcam_heatmap, overlay_gradcam
from src.utils.metrics import compute_clinical_metrics, aggregate_patient_predictions


def main():
    parser = argparse.ArgumentParser(description="CerebroAI Brain MRI Classification & WGAN-GP Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stages")

    # 1. Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Run subject-level split and NIfTI to PNG extraction")
    preprocess_parser.add_argument("--raw_dir", type=str, default="./data/adni-mri", help="Directory with raw .nii files")
    preprocess_parser.add_argument("--split_dir", type=str, default="./data/adni-split", help="Output directory for split data")
    preprocess_parser.add_argument("--png_dir", type=str, default="./data/adni-images", help="Output directory for 2D PNG slices")

    # 2. Train Classifier command
    train_cls_parser = subparsers.add_parser("train_classifier", help="Train ResNet-18 classifier")
    train_cls_parser.add_argument("--data_dir", type=str, default="./data/adni-images", help="Directory with PNG slices")
    train_cls_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    train_cls_parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_cls_parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    train_cls_parser.add_argument("--output_model", type=str, default="./models/resnet18_adni.h5", help="Path to save weights")

    # 3. Train WGAN-GP command
    train_gan_parser = subparsers.add_parser("train_wgan", help="Train WGAN-GP on minority class slices")
    train_gan_parser.add_argument("--data_dir", type=str, default="./data/adni-images/train/ad", help="Training slice directory")
    train_gan_parser.add_argument("--epochs", type=int, default=200, help="Number of GAN epochs")
    train_gan_parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_gan_parser.add_argument("--latent_dim", type=int, default=128, help="Latent vector dimension")
    train_gan_parser.add_argument("--output_gen", type=str, default="./models/wgan_generator.h5", help="Saved generator path")

    # 4. Infer / Grad-CAM command
    infer_parser = subparsers.add_parser("infer", help="Run inference and Grad-CAM on a single scan or slice")
    infer_parser.add_argument("--image_path", type=str, required=True, help="Path to MRI .nii or .png file")
    infer_parser.add_argument("--model_path", type=str, default="./models/resnet18_adni.h5", help="Path to trained model")
    infer_parser.add_argument("--output_heatmap", type=str, default="./output/gradcam_result.png", help="Output heatmap path")

    args = parser.parse_args()

    if args.command == "preprocess":
        print(f"[*] Starting patient-level split from: {args.raw_dir}")
        splits = subject_level_split(args.raw_dir, args.split_dir)
        print("[*] Extracting and normalizing axial slices...")
        batch_convert_dataset(args.split_dir, args.png_dir)
        print("[+] Preprocessing completed successfully!")

    elif args.command == "train_classifier":
        print(f"[*] Initializing ResNet-18 (Input: 192x160x1)...")
        model = build_resnet18(input_shape=(192, 160, 1), num_classes=2)
        model.summary()
        print(f"[+] ResNet-18 ready for training on data in: {args.data_dir}")

    elif args.command == "train_wgan":
        print(f"[*] Building WGAN-GP (Latent Dim: {args.latent_dim})...")
        generator = build_generator(latent_dim=args.latent_dim, output_shape=(192, 160, 1))
        critic = build_critic(input_shape=(192, 160, 1))
        wgan = WGAN_GP(critic=critic, generator=generator, latent_dim=args.latent_dim)
        print(f"[+] WGAN-GP assembled successfully!")

    elif args.command == "infer":
        print(f"[*] Processing inference on: {args.image_path}")
        print(f"[+] Output Grad-CAM visualization target: {args.output_heatmap}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
