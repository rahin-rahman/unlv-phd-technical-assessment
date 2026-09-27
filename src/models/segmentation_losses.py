"""
Loss Functions for Binary Building Footprint Semantic Segmentation.
Implements BCEDiceLoss (Binary Cross-Entropy with Logits + Soft Dice Loss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Directly optimizes overlap quality (Dice Coefficient / F1 score) to combat class imbalance.
    
    Formula:
    L_Dice = 1 - (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth)
    where p = sigmoid(logits), y = binary target {0, 1}.
    """
    def __init__(self, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """
    Composite Loss Function: Binary Cross-Entropy with Logits + Soft Dice Loss.
    
    - BCE Loss provides smooth, pixel-wise classification gradient supervision.
    - Soft Dice Loss directly optimizes boundary alignment and overcomes severe background/building imbalance.
    
    Total Loss = bce_weight * BCE_Loss + dice_weight * Dice_Loss
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0, pos_weight=None):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = SoftDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        l_bce = self.bce_loss(logits, targets)
        l_dice = self.dice_loss(logits, targets)
        
        total_loss = self.bce_weight * l_bce + self.dice_weight * l_dice
        return total_loss, l_bce, l_dice


if __name__ == '__main__':
    # Unit Test for Loss Function
    logits = torch.randn(2, 1, 512, 512, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 512, 512)).float()
    
    criterion = BCEDiceLoss()
    total_l, bce_l, dice_l = criterion(logits, targets)
    
    print("BCEDiceLoss Unit Test:")
    print(f"  - Total Loss: {total_l.item():.4f}")
    print(f"  - BCE Loss:   {bce_l.item():.4f}")
    print(f"  - Dice Loss:  {dice_l.item():.4f}")
