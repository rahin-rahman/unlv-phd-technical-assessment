"""
U-Net Model Architecture for Binary Building Footprint Semantic Segmentation.

Architecture Overview:
----------------------
- Model Framework: U-Net (Ronneberger et al., 2015)
- Encoder Backbone: ResNet-34 (He et al., 2016)
- Pretrained Weights: ImageNet (dramatically speeds up convergence on spatial features)
- Input Shape: (Batch_Size, 3, 512, 512) -- 3 RGB aerial channels
- Output Shape: (Batch_Size, 1, 512, 512) -- 1 channel unnormalized binary classification logits

Key Architectural Features:
- Encoder extracts multi-scale spatial representations (64, 128, 256, 512 channels across residual layers).
- Skip connections transmit high-resolution local spatial details directly from encoder feature maps to decoder blocks.
- Decoder performs successive transposed convolution / upsampling with double 3x3 convolutions to refine building boundaries.
- Final 1x1 convolution produces raw pixel-wise logits for binary building prediction.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_unet_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
    """
    Instantiates a U-Net semantic segmentation model with a pretrained ResNet encoder.
    
    Parameters:
    -----------
    encoder_name : str, default='resnet34'
        Encoder backbone name from segmentation_models_pytorch (e.g., 'resnet34', 'resnet50').
    encoder_weights : str or None, default='imagenet'
        Pretrained weight initialization. Uses ImageNet transfer learning weights if specified.
    in_channels : int, default=3
        Number of input channels (RGB imagery = 3).
    classes : int, default=1
        Number of output prediction channels (Binary segmentation = 1).
        
    Returns:
    --------
    model : nn.Module
        PyTorch U-Net model instance returning unnormalized logits.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None  # Returns unnormalized logits for numerical stability in BCEWithLogitsLoss
    )
    return model


if __name__ == '__main__':
    # Standalone verification unit test
    print("==================================================")
    print("U-Net ResNet34 Model Verification Test")
    print("==================================================")
    
    model = build_unet_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Architecture:       U-Net")
    print(f"Encoder Backbone:         ResNet-34 (Pretrained: ImageNet)")
    print(f"Total Parameters:         {total_params:,}")
    print(f"Trainable Parameters:     {trainable_params:,}")
    
    # Run dummy forward pass
    dummy_input = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        dummy_output = model(dummy_input)
        
    print(f"Dummy Input Shape:        {list(dummy_input.shape)}")
    print(f"Dummy Output Logits Shape: {list(dummy_output.shape)}")
    
    assert dummy_output.shape == (2, 1, 512, 512), "Shape mismatch!"
    print("FORWARD PASS VERIFICATION: [PASS]")
    print("==================================================")
