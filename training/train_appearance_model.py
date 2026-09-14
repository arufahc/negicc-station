#!/usr/bin/env python3
"""Appearance Model Training & Export Script for DINOv3.

Trains a Photographic Intent Adapter from:
1. DINOv3-small vision backbone (downloaded locally from HuggingFace)
2. Image corpus of authentic film scans
3. Configuration from metadata.json or CLI arguments
"""

import os
import sys
import json
import time
import argparse
import torch
from torch.utils.data import TensorDataset, DataLoader

# Ensure training directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PhotographicIntentAdapter, BACKBONE_CONFIGS, load_backbone
from loss import HeteroscedasticHuberLoss
from extract import extract_native_aspect_features
from export import export_appearance_model


def train_intent_adapter(
    features,
    targets,
    profile_name,
    backbone_name,
    output_checkpoint_path,
    epochs=60,
    batch_size=64,
    lr=1e-3,
    val_split=0.1,
    random_seed=42,
    device="cpu",
):
    """Trains the PhotographicIntentAdapter on multi-layer multi-aspect features."""
    torch.set_num_threads(16)
    torch.manual_seed(random_seed)

    if features.dim() == 3:
        N_imgs, seq_len, in_dim = features.shape
        features = features.reshape(N_imgs * seq_len, in_dim)
        targets = targets.reshape(N_imgs * seq_len, 3)

    N_tokens = len(features)
    in_dim = features.shape[-1]
    hidden_dim = in_dim // 2

    val_size = max(1, int(val_split * N_tokens))
    indices = torch.randperm(N_tokens)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    train_feat, train_tgt = features[train_indices], targets[train_indices]
    val_feat, val_tgt = features[val_indices], targets[val_indices]

    token_batch_size = batch_size * 1024 if batch_size < 1000 else batch_size

    print(f"\n[{backbone_name} | {profile_name}] Training Tokens: {len(train_feat):,} | Validation Tokens: {len(val_feat):,}", flush=True)
    print(f"  Batch Size (Tokens): {token_batch_size:,} | In Dim: {in_dim} | Hidden Dim: {hidden_dim} | Epochs: {epochs}", flush=True)

    train_dataset = TensorDataset(train_feat, train_tgt)
    train_loader = DataLoader(train_dataset, batch_size=token_batch_size, shuffle=True)

    adapter = PhotographicIntentAdapter(in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    loss_fn = HeteroscedasticHuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_metrics = {}

    t_train_start = time.time()
    for epoch in range(1, epochs + 1):
        adapter.train()
        train_loss_total = 0.0
        train_mae_total = 0.0

        for z_batch, tgt_batch in train_loader:
            z_batch = z_batch.to(device).float()
            tgt_batch = tgt_batch.to(device).float()

            optimizer.zero_grad()
            mu, sigma, log_var = adapter(z_batch)
            loss, metrics = loss_fn(mu, sigma, log_var, tgt_batch)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(z_batch)
            train_mae_total += metrics["mae_total"] * len(z_batch)

        scheduler.step()
        train_loss = train_loss_total / len(train_dataset)
        train_mae = train_mae_total / len(train_dataset)

        # Validation loop
        adapter.eval()
        with torch.no_grad():
            v_loss_tot = 0.0
            v_res_tot = 0.0
            v_res_L_tot = 0.0
            v_res_ab_tot = 0.0
            v_sig_tot = 0.0

            val_loader = DataLoader(TensorDataset(val_feat, val_tgt), batch_size=token_batch_size, shuffle=False)
            for z_v, tgt_v in val_loader:
                z_v = z_v.to(device).float()
                tgt_v = tgt_v.to(device).float()
                mu_v, sig_v, log_var_v = adapter(z_v)
                v_loss, v_met = loss_fn(mu_v, sig_v, log_var_v, tgt_v)
                N_v = len(z_v)
                v_loss_tot += v_loss.item() * N_v
                v_res_tot += v_met["mae_total"] * N_v
                v_res_L_tot += v_met["mae_L"] * N_v
                v_res_ab_tot += v_met["mae_ab"] * N_v
                v_sig_tot += sig_v.mean().item() * N_v

            val_loss = v_loss_tot / len(val_feat)
            val_mae = v_res_tot / len(val_feat)
            val_mae_L = v_res_L_tot / len(val_feat)
            val_mae_ab = v_res_ab_tot / len(val_feat)
            mean_sig = v_sig_tot / len(val_feat)

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} (MAE: {train_mae:.2f}) | "
                f"Val Loss: {val_loss:.4f} (MAE: {val_mae:.2f}, L*: {val_mae_L:.2f}, ab*: {val_mae_ab:.2f}) | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_mae_L": val_mae_L,
                "val_mae_ab": val_mae_ab,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "mean_sigma": mean_sig,
                "profile": profile_name,
                "backbone": backbone_name,
                "in_dim": in_dim,
                "hidden_dim": hidden_dim,
            }

            os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint_path)), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": adapter.state_dict(),
                    "backbone": backbone_name,
                    "profile": profile_name,
                    "in_dim": in_dim,
                    "hidden_dim": hidden_dim,
                    "metrics": best_metrics,
                },
                output_checkpoint_path,
            )

    train_duration = time.time() - t_train_start
    print(f"Training completed in {train_duration:.2f}s! Best Val Loss: {best_val_loss:.4f}")
    return adapter, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train DINOv3 Photographic Intent Appearance Model")
    parser.add_argument("--metadata", type=str, default=None, help="Path to reference metadata.json to copy config from")
    parser.add_argument("--profile", type=str, default="portra400", help="Film profile name (e.g. portra400, gold200)")
    parser.add_argument("--backbone", type=str, default="dinov3-small", help="Vision backbone (default: dinov3-small)")
    parser.add_argument("--corpus-dir", type=str, default="/home/alpha/Projects/FilmCorpus/training", help="Corpus image directory")
    parser.add_argument("--output-dir", type=str, default="../models/dinov3-small_portra400", help="Output directory to save exported model")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Directory to cache extracted features")
    parser.add_argument("--max-samples", type=int, default=400, help="Max images to extract features from")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size in 1024 token units")
    parser.add_argument("--extract-batch-size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of features")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    # If metadata.json was provided, parse profile and backbone from it
    if args.metadata and os.path.isfile(args.metadata):
        print(f"Loading configuration from metadata JSON: {args.metadata}")
        with open(args.metadata, "r", encoding="utf-8") as f:
            meta = json.load(f)
        profile_name = meta.get("profile", args.profile)
        backbone_name = meta.get("backbone", args.backbone)
    else:
        profile_name = args.profile
        backbone_name = args.backbone

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "model.pt")

    print("=" * 80)
    print(f" TRAINING APPEARANCE MODEL: {profile_name.upper()} ({backbone_name})")
    print(f" Output Target: {output_dir}")
    print(f" Corpus Directory: {args.corpus_dir}")
    print(f" Epochs: {args.epochs} | Max Samples: {args.max_samples} | Device: {args.device}")
    print("=" * 80)

    # 1. Extract / Cache Features
    features, targets, _ = extract_native_aspect_features(
        profile_name=profile_name,
        backbone_name=backbone_name,
        corpus_dir=args.corpus_dir,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        batch_size=args.extract_batch_size,
        force_extract=args.force_extract,
        device=args.device,
    )

    # 2. Train Adapter
    adapter, metrics = train_intent_adapter(
        features=features,
        targets=targets,
        profile_name=profile_name,
        backbone_name=backbone_name,
        output_checkpoint_path=checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    # 3. Export PyTorch, TorchScript, ONNX, and metadata.json
    export_appearance_model(
        checkpoint_path=checkpoint_path,
        export_dir=output_dir,
    )

    print("\n" + "=" * 80)
    print(f" MODEL SUCCESSFULLY TRAINED AND EXPORTED TO: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
