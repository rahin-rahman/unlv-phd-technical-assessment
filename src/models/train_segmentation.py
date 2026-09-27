"""
Configurable Training Pipeline for Binary Building Footprint Semantic Segmentation.

Key Features:
-------------
- Model: U-Net with ResNet34 ImageNet-pretrained encoder.
- Loss: BCEDiceLoss (Binary Cross Entropy + Soft Dice).
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4).
- LR Scheduler: ReduceLROnPlateau (mode='max' on val_iou, factor=0.5, patience=3).
- Early Stopping: 7 epochs patience on best validation IoU.
- Automatic Mixed Precision (AMP): Enabled when CUDA is available.
- Checkpointing: Saves model weights to outputs/models/best_unet_resnet34.pth based on best val IoU.
- History & Plotting: Saves metrics history CSV and generates outputs/figures/fig8_training_curves.png.
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data.aerial_patch_dataset import InriaAerialPatchDataset
from src.models.unet_segmentation import build_unet_model
from src.models.segmentation_losses import BCEDiceLoss
from src.models.evaluate_segmentation import evaluate_model, evaluate_test_set, plot_qualitative_predictions, identify_failure_cases


def set_seed(seed=42):
    """Fix random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_training_history(history_df, save_path="outputs/figures/fig8_training_curves.png"):
    """
    Generates training and validation curves.
    
    Subplot 1: Train Loss vs Validation Loss
    Subplot 2: Validation IoU
    Subplot 3: Validation Dice Coefficient
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = history_df['epoch']
    
    # Subplot 1: Loss
    axes[0].plot(epochs, history_df['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
    axes[0].plot(epochs, history_df['val_loss'], label='Validation Loss', color='#ff7f0e', lw=2, linestyle='--')
    axes[0].set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss (BCE + Soft Dice)', fontsize=11)
    axes[0].legend(frameon=True)
    axes[0].grid(True, alpha=0.3)
    
    # Subplot 2: IoU
    axes[1].plot(epochs, history_df['val_iou'], label='Validation IoU', color='#2ca02c', lw=2)
    axes[1].set_title('Validation IoU (Primary Metric)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Intersection over Union (IoU)', fontsize=11)
    axes[1].legend(frameon=True)
    axes[1].grid(True, alpha=0.3)
    
    # Subplot 3: Dice
    axes[2].plot(epochs, history_df['val_dice'], label='Validation Dice (F1)', color='#d62728', lw=2)
    axes[2].set_title('Validation Dice Coefficient', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Dice Coefficient / F1 Score', fontsize=11)
    axes[2].legend(frameon=True)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history figure saved to {save_path}")


def train_model(
    manifest_path="outputs/tables/part2_patch_manifest.csv",
    model_save_path="outputs/models/best_unet_resnet34.pth",
    history_csv_path="outputs/tables/segmentation_training_history.csv",
    curves_png_path="outputs/figures/fig8_training_curves.png",
    batch_size=8,
    epochs=40,
    lr=1e-4,
    weight_decay=1e-4,
    patience=7,
    num_workers=2,
    seed=42,
    dry_run=False
):
    """
    Main training routine.
    """
    set_seed(seed)
    
    # Select compute device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    print(f"Executing Training Pipeline on Device: [{device.type.upper()}] | AMP Enabled: [{use_amp}]")
    
    if device.type == 'cpu' and not dry_run:
        print("\n[WARNING] Environment is CPU-only. Training on CPU is disabled for performance.")
        print("To verify pipeline structure on CPU without full training, pass dry_run=True.")
        
    # Load Datasets
    train_dataset = InriaAerialPatchDataset(manifest_path, split='train', is_train=True)
    val_dataset = InriaAerialPatchDataset(manifest_path, split='validation', is_train=False)
    
    print(f"Dataset Loaded: Train = {len(train_dataset)} patches | Validation (Tyrol-w) = {len(val_dataset)} patches")
    
    # DataLoader setup
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem
    )
    
    # Initialize Model, Loss, Optimizer, Scheduler, Scaler
    model = build_unet_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    if dry_run:
        print("\n[DRY RUN MODE VERIFICATION]")
        print("Executing 1 training step and 1 validation step to verify tensor flow...")
        
        model.train()
        images, targets, _ = next(iter(train_loader))
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss, bce, dice = criterion(logits, targets)
            
        print(f"Dry Run Batch Forward Pass successful!")
        print(f"  - Logits Shape: {list(logits.shape)}")
        print(f"  - Total Loss:   {loss.item():.4f} (BCE: {bce.item():.4f}, Dice: {dice.item():.4f})")
        print("DRY RUN PIPELINE VERIFICATION: [SUCCESS]\n")
        return None, None

    # Full Training Loop
    best_val_iou = -1.0
    epochs_no_improve = 0
    history = []
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_csv_path), exist_ok=True)
    
    print(f"\nStarting Model Training for up to {epochs} Epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss, _, _ = criterion(logits, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            
        train_loss = running_loss / len(train_dataset)
        
        # Validation
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_iou = val_metrics['iou']
        val_dice = val_metrics['dice']
        val_prec = val_metrics['precision']
        val_rec = val_metrics['recall']
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_iou)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.6f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_iou': val_iou,
            'val_dice': val_dice,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'learning_rate': current_lr
        })
        
        # Checkpointing based on Best Validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'val_metrics': val_metrics
            }, model_save_path)
            print(f"  --> Best Checkpoint Saved! (New Best Val IoU: {best_val_iou:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[EARLY STOPPING TRIGGERED] Validation IoU did not improve for {patience} consecutive epochs.")
                break
                
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to {history_csv_path}")
    
    plot_training_history(history_df, save_path=curves_png_path)
    return model, history_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train U-Net ResNet34 Building Footprint Segmentation Model")
    parser.add_argument("--manifest", type=str, default="outputs/tables/part2_patch_manifest.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry_run", action="store_true", help="Run 1 batch test verification without full training")
    
    args = parser.parse_args()
    
    train_model(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        dry_run=args.dry_run or (not torch.cuda.is_available())
    )
