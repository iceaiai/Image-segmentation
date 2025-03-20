"""
Mask R-CNN model implementation for electron microscopy particle segmentation.
This module provides functions to create and configure the Mask R-CNN model.
"""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_model_instance_segmentation(num_classes, backbone="resnet50", pretrained=True, trainable_backbone_layers=3):
    """
    Create a Mask R-CNN model with the specified backbone.
    
    Args:
        num_classes (int): Number of classes (including background)
        backbone (str): Backbone network architecture ("resnet50" or "resnet101")
        pretrained (bool): Whether to use pretrained weights for the backbone
        trainable_backbone_layers (int): Number of backbone layers to train
        
    Returns:
        model (MaskRCNN): Configured Mask R-CNN model
    """
    # Check if backbone is supported
    if backbone not in ["resnet50", "resnet101"]:
        raise ValueError(f"Backbone {backbone} not supported. Use 'resnet50' or 'resnet101'.")
    
    # Load a pre-trained model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers
    )
    
    # Replace the backbone if needed
    if backbone == "resnet101":
        backbone_net = resnet_fpn_backbone(
            'resnet101', 
            pretrained=pretrained, 
            trainable_layers=trainable_backbone_layers
        )
        model.backbone = backbone_net
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
    return model


def configure_model_for_grayscale(model):
    """
    Configure the model to accept grayscale images (1 channel) instead of RGB (3 channels).
    
    Args:
        model (MaskRCNN): The Mask R-CNN model to configure
        
    Returns:
        model (MaskRCNN): Configured model for grayscale input
    """
    print("Configuring model for grayscale input...")
    
    # Get the first conv layer
    first_conv_layer = model.backbone.body.conv1
    
    # Save the weights and bias
    original_weight = first_conv_layer.weight.clone()
    original_bias = first_conv_layer.bias.clone() if first_conv_layer.bias is not None else None
    
    # Create a new conv layer with 1 input channel
    new_conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=first_conv_layer.out_channels,
        kernel_size=first_conv_layer.kernel_size,
        stride=first_conv_layer.stride,
        padding=first_conv_layer.padding,
        bias=first_conv_layer.bias is not None
    )
    
    # Average the weights across the RGB channels and set for the new layer
    new_weight = original_weight.sum(dim=1, keepdim=True) / 3.0
    new_conv.weight.data = new_weight
    
    if original_bias is not None:
        new_conv.bias.data = original_bias
    
    # Replace the first conv layer
    model.backbone.body.conv1 = new_conv
    
    # Print confirmation
    print(f"First conv layer input channels: {new_conv.in_channels}")
    print(f"First conv layer output channels: {new_conv.out_channels}")
    
    return model


class CombinedLoss(torch.nn.Module):
    """
    Combined loss function for Mask R-CNN training.
    Combines the default Mask R-CNN losses with additional losses like Dice loss.
    """
    def __init__(self, dice_weight=0.5):
        """
        Initialize the combined loss.
        
        Args:
            dice_weight (float): Weight for the Dice loss component
        """
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        
    def forward(self, loss_dict, masks_pred, masks_gt):
        """
        Compute the combined loss.
        
        Args:
            loss_dict (dict): Dictionary of losses from Mask R-CNN
            masks_pred (torch.Tensor): Predicted masks
            masks_gt (torch.Tensor): Ground truth masks
            
        Returns:
            total_loss (torch.Tensor): Combined loss value
        """
        # Sum the standard Mask R-CNN losses
        mask_rcnn_loss = sum(loss for loss in loss_dict.values())
        
        # If we have mask predictions and ground truth, add Dice loss
        if masks_pred is not None and masks_gt is not None:
            dice_loss = self._dice_loss(masks_pred, masks_gt)
            total_loss = mask_rcnn_loss + self.dice_weight * dice_loss
        else:
            total_loss = mask_rcnn_loss
            
        return total_loss
    
    def _dice_loss(self, pred, target, smooth=1.0):
        """
        Compute the Dice loss.
        
        Args:
            pred (torch.Tensor): Predicted masks
            target (torch.Tensor): Ground truth masks
            smooth (float): Smoothing factor to avoid division by zero
            
        Returns:
            dice_loss (torch.Tensor): Dice loss value
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice 