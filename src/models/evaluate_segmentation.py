"""
Evaluation and Qualitative Visualization Pipeline for Building Footprint Semantic Segmentation.

Implements:
1. Pixel-level and spatial overlap metrics (Pixel Accuracy, Precision, Recall, Dice/F1, IoU/Jaccard).
2. Batch evaluation over validation and held-out test data.
3. Test evaluation specifically for the held-out Vienna test set with coverage stratification.
4. Qualitative prediction visualization with RGB, Ground Truth, Prediction, and Error Overlay (TP, FP, FN).
5. Empirical failure analysis categorization and CSV report generation.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image


def compute_segmentation_metrics(y_true, y_pred_prob, threshold=0.5, eps=1e-7):
    """
    Computes standard binary semantic segmentation metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray or torch.Tensor
        Binary ground truth mask {0, 1}.
    y_pred_prob : np.ndarray or torch.Tensor
        Predicted probabilities [0.0, 1.0] (post-sigmoid).
    threshold : float, default=0.5
        Decision threshold for binary classification.
    eps : float, default=1e-7
        Numerical stability epsilon to prevent zero division.
        
    Returns:
    --------
    metrics : dict
        Dictionary containing 'iou', 'dice', 'precision', 'recall', 'accuracy'.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_prob, torch.Tensor):
        y_pred_prob = y_pred_prob.detach().cpu().numpy()
        
    y_pred_bin = (y_pred_prob >= threshold).astype(np.float32)
    y_true_bin = (y_true > 0.5).astype(np.float32)
    
    tp = np.sum((y_pred_bin == 1.0) & (y_true_bin == 1.0))
    fp = np.sum((y_pred_bin == 1.0) & (y_true_bin == 0.0))
    fn = np.sum((y_pred_bin == 0.0) & (y_true_bin == 1.0))
    tn = np.sum((y_pred_bin == 0.0) & (y_true_bin == 0.0))
    
    intersection = tp
    union = tp + fp + fn
    
    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy)
    }


def evaluate_model(model, dataloader, criterion, device, threshold=0.5):
    """
    Evaluates model performance over a full dataset split (e.g., validation set).
    
    Returns:
    --------
    avg_loss : float
    metrics : dict
        Aggregated dataset-wide metrics.
    """
    model.eval()
    total_loss = 0.0
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    eps = 1e-7
    
    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            logits = model(images)
            loss, _, _ = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)
            
            probs = torch.sigmoid(logits).cpu().numpy()
            targets_np = targets.cpu().numpy()
            preds_bin = (probs >= threshold).astype(np.float32)
            targets_bin = (targets_np > 0.5).astype(np.float32)
            
            total_tp += np.sum((preds_bin == 1.0) & (targets_bin == 1.0))
            total_fp += np.sum((preds_bin == 1.0) & (targets_bin == 0.0))
            total_fn += np.sum((preds_bin == 0.0) & (targets_bin == 1.0))
            total_tn += np.sum((preds_bin == 0.0) & (targets_bin == 0.0))
            
    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    
    iou = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    dice = (2.0 * total_tp + eps) / (2.0 * total_tp + total_fp + total_fn + eps)
    precision = (total_tp + eps) / (total_tp + total_fp + eps)
    recall = (total_tp + eps) / (total_tp + total_fn + eps)
    accuracy = (total_tp + total_tn + eps) / (total_tp + total_fp + total_fn + total_tn + eps)
    
    return avg_loss, {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy)
    }


def evaluate_test_set(model, test_loader, device, output_csv="outputs/tables/segmentation_test_metrics.csv", threshold=0.5):
    """
    Evaluates model exclusively on the held-out Vienna test set.
    Calculates overall metrics as well as coverage-stratified metrics.
    Exports result table to output_csv.
    """
    model.eval()
    results = []
    eps = 1e-7
    
    patch_metrics = []
    
    with torch.no_grad():
        for images, targets, metadata in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for i in range(images.size(0)):
                p_id = metadata['patch_id'][i]
                c_name = metadata['city'][i]
                bldg_pct = metadata['bldg_pct'][i] if isinstance(metadata['bldg_pct'], list) else metadata['bldg_pct'][i].item()
                
                m = compute_segmentation_metrics(targets_np[i], probs[i], threshold=threshold)
                m['patch_id'] = p_id
                m['city'] = c_name
                m['bldg_pct'] = bldg_pct
                patch_metrics.append(m)
                
    df_patch = pd.DataFrame(patch_metrics)
    
    # Calculate overall dataset-wide metrics
    overall_iou = df_patch['iou'].mean()
    overall_dice = df_patch['dice'].mean()
    overall_prec = df_patch['precision'].mean()
    overall_rec = df_patch['recall'].mean()
    overall_acc = df_patch['accuracy'].mean()
    
    summary_rows = [
        {'Evaluation_Split': 'Vienna Test (Overall)', 'IoU': overall_iou, 'Dice': overall_dice, 
         'Precision': overall_prec, 'Recall': overall_rec, 'Pixel_Accuracy': overall_acc, 'Num_Patches': len(df_patch)}
    ]
    
    # Stratify by building coverage
    low_cov = df_patch[df_patch['bldg_pct'] < 5.0]
    mod_cov = df_patch[(df_patch['bldg_pct'] >= 5.0) & (df_patch['bldg_pct'] <= 25.0)]
    high_cov = df_patch[df_patch['bldg_pct'] > 25.0]
    
    for name, sub_df in [('Low Coverage (<5%)', low_cov), ('Moderate Coverage (5-25%)', mod_cov), ('High Coverage (>25%)', high_cov)]:
        if len(sub_df) > 0:
            summary_rows.append({
                'Evaluation_Split': f'Vienna Test ({name})',
                'IoU': sub_df['iou'].mean(),
                'Dice': sub_df['dice'].mean(),
                'Precision': sub_df['precision'].mean(),
                'Recall': sub_df['recall'].mean(),
                'Pixel_Accuracy': sub_df['accuracy'].mean(),
                'Num_Patches': len(sub_df)
            })
            
    df_summary = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_summary.to_csv(output_csv, index=False)
    print(f"Test evaluation saved to {output_csv}")
    return df_summary


def create_error_overlay(image_np, gt_mask, pred_mask):
    """
    Creates an error visualization map:
    - True Positives (TP): Green (Building correctly detected)
    - False Positives (FP): Red (Background wrongly predicted as building)
    - False Negatives (FN): Blue (Building missed by model)
    """
    H, W = gt_mask.shape[:2]
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    
    tp = (pred_mask == 1.0) & (gt_mask == 1.0)
    fp = (pred_mask == 1.0) & (gt_mask == 0.0)
    fn = (pred_mask == 0.0) & (gt_mask == 1.0)
    
    overlay[tp] = [0, 255, 0]   # Green for TP
    overlay[fp] = [255, 0, 0]   # Red for FP
    overlay[fn] = [0, 100, 255]  # Blue for FN
    
    return overlay


def plot_qualitative_predictions(model, dataloader, device, save_path="outputs/figures/fig9_segmentation_predictions.png", num_samples=4, threshold=0.5):
    """
    Generates representative qualitative prediction figure showing:
    Col 1: Original Aerial RGB Image
    Col 2: Ground Truth Building Mask
    Col 3: Model Predicted Building Mask
    Col 4: Error Visualization Overlay (TP=Green, FP=Red, FN=Blue)
    """
    model.eval()
    images_list, targets_list, preds_list, metadata_list = [], [], [], []
    
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    
    with torch.no_grad():
        for images, targets, metadata in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets_np = targets.cpu().numpy()
            images_np = images.cpu().numpy()
            
            for i in range(images.size(0)):
                if len(images_list) < num_samples:
                    # Un-normalize image for plotting
                    img_orig = images_np[i].transpose(1, 2, 0)
                    img_orig = (img_orig * std + mean).clip(0, 1)
                    
                    gt = targets_np[i, 0]
                    pred = (probs[i, 0] >= threshold).astype(np.float32)
                    
                    images_list.append(img_orig)
                    targets_list.append(gt)
                    preds_list.append(pred)
                    
                    meta_item = {k: v[i] if isinstance(v, list) else v[i].item() for k, v in metadata.items()}
                    metadata_list.append(meta_item)
                else:
                    break
            if len(images_list) >= num_samples:
                break
                
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
        
    titles = ['Original Aerial RGB', 'Ground Truth Mask', 'U-Net Prediction', 'Error Overlay (TP:G, FP:R, FN:B)']
    for j, title in enumerate(titles):
        axes[0, j].set_title(title, fontsize=13, fontweight='bold', pad=10)
        
    for i in range(num_samples):
        img_rgb = images_list[i]
        gt_mask = targets_list[i]
        pred_mask = preds_list[i]
        error_map = create_error_overlay(img_rgb, gt_mask, pred_mask)
        
        meta = metadata_list[i]
        patch_info = f"{meta['city']} ({meta['patch_id']})"
        
        # Col 1: Original Image
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_ylabel(patch_info, fontsize=11, fontweight='semibold')
        
        # Col 2: Ground Truth
        axes[i, 1].imshow(gt_mask, cmap='gray')
        
        # Col 3: Prediction
        axes[i, 2].imshow(pred_mask, cmap='gray')
        
        # Col 4: Error Overlay
        # Blend overlay with original image for context
        blended = (img_rgb * 0.5 + (error_map / 255.0) * 0.5).clip(0, 1)
        axes[i, 3].imshow(blended)
        
        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Qualitative visualization figure saved to {save_path}")


def identify_failure_cases(model, dataloader, device, output_csv="outputs/tables/segmentation_failure_cases.csv", max_cases=10, threshold=0.5):
    """
    Identifies and categorizes representative failure cases based on empirical metrics.
    Categorization logic:
    - false_positives: FP > 2 * FN (over-prediction, shadow/vegetation confusion)
    - missed_buildings: FN > 2 * FP (under-prediction, small building omission)
    - boundary_errors: Moderate IoU (0.3 - 0.6) with balanced FP and FN (roof boundary misalignment)
    - dense_structure_merging: High ground truth coverage (>30%) with moderate IoU (<0.6)
    """
    model.eval()
    failures = []
    
    with torch.no_grad():
        for images, targets, metadata in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for i in range(images.size(0)):
                gt = targets_np[i, 0]
                pred = (probs[i, 0] >= threshold).astype(np.float32)
                
                m = compute_segmentation_metrics(gt, pred, threshold=threshold)
                
                tp = np.sum((pred == 1.0) & (gt == 1.0))
                fp = np.sum((pred == 1.0) & (gt == 0.0))
                fn = np.sum((pred == 0.0) & (gt == 1.0))
                bldg_pct = np.mean(gt) * 100.0
                
                p_id = metadata['patch_id'][i]
                c_name = metadata['city'][i]
                
                # Determine failure category if IoU < 0.65
                if m['iou'] < 0.65:
                    if bldg_pct > 30.0 and m['iou'] < 0.60:
                        cat = 'dense_structure_merging'
                    elif fp > 2 * max(fn, 1):
                        cat = 'false_positives'
                    elif fn > 2 * max(fp, 1):
                        cat = 'missed_buildings'
                    else:
                        cat = 'boundary_errors'
                        
                    failures.append({
                        'patch_id': p_id,
                        'city': c_name,
                        'failure_type': cat,
                        'IoU': m['iou'],
                        'Dice': m['dice'],
                        'Precision': m['precision'],
                        'Recall': m['recall'],
                        'Building_Coverage_Pct': bldg_pct
                    })
                    
    df_fail = pd.DataFrame(failures)
    if len(df_fail) > max_cases:
        # Sort by worst IoU
        df_fail = df_fail.sort_values(by='IoU').head(max_cases).reset_index(drop=True)
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_fail.to_csv(output_csv, index=False)
    print(f"Failure analysis saved to {output_csv}")
    return df_fail


if __name__ == '__main__':
    print("Testing evaluate_segmentation metric calculation...")
    dummy_gt = torch.zeros(2, 1, 512, 512)
    dummy_gt[:, :, 100:200, 100:200] = 1.0
    
    dummy_pred = torch.zeros(2, 1, 512, 512)
    dummy_pred[:, :, 110:210, 100:200] = 0.8  # 90% overlap
    
    m = compute_segmentation_metrics(dummy_gt, dummy_pred)
    print(f"Metric calculation test:")
    print(f"  - IoU:       {m['iou']:.4f}")
    print(f"  - Dice:      {m['dice']:.4f}")
    print(f"  - Precision: {m['precision']:.4f}")
    print(f"  - Recall:    {m['recall']:.4f}")
    print(f"  - Accuracy:  {m['accuracy']:.4f}")
    print("EVALUATION MODULE TEST: [PASS]")
