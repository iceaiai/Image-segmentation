"""
Evaluation script for Mask R-CNN for electron microscopy particle segmentation.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torchvision
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

# Import local modules
from config import InferenceConfig
import utils
from datasets.emps_dataset import EMPSDataset
from models.mask_rcnn import get_model_instance_segmentation, configure_model_for_grayscale


# Custom transform to ensure single-channel output
class EnsureSingleChannel:
    def __call__(self, image, target=None):
        if isinstance(image, torch.Tensor) and image.dim() == 3 and image.shape[0] == 3:
            # Convert 3-channel to 1-channel by taking the first channel
            image = image[0:1, :, :]
        return (image, target) if target is not None else image


# Patch the GeneralizedRCNN forward method to handle grayscale images
# This should be the same patch as in train.py
original_forward = GeneralizedRCNN.forward

def patched_forward(self, images, targets=None):
    """
    Patched forward method for GeneralizedRCNN to handle grayscale images.
    """
    if self.training and targets is None:
        raise ValueError("In training mode, targets should be passed")
    
    # Check if images is a list of tensors
    if isinstance(images, list):
        # Check if images are grayscale (1 channel)
        if all(img.shape[0] == 1 for img in images):
            # Create ImageList directly without converting to RGB
            from torchvision.models.detection.image_list import ImageList
            
            # Get image sizes
            image_sizes = [img.shape[-2:] for img in images]
            
            # We can't use torch.stack because images might have different sizes
            # Instead, we'll create a batched tensor with padding
            max_height = max(img.shape[1] for img in images)
            max_width = max(img.shape[2] for img in images)
            
            # Create a padded batch tensor
            batched_imgs = torch.zeros(len(images), 1, max_height, max_width, device=images[0].device)
            
            # Fill the batch tensor with images
            for i, img in enumerate(images):
                h, w = img.shape[1:]
                batched_imgs[i, :, :h, :w] = img
            
            # Create ImageList
            images = ImageList(batched_imgs, image_sizes)
            
            # Forward pass through backbone
            features = self.backbone(images.tensors)
            
            # Continue with the rest of the forward pass
            if isinstance(features, torch.Tensor):
                features = {"0": features}
            
            proposals, proposal_losses = self.rpn(images, features, targets)
            detections, detector_losses = self.roi_heads(features, proposals, image_sizes, targets)
            
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            
            if self.training:
                return losses
            
            return detections
    
    # If not grayscale or not a list, use the original forward method
    return original_forward(self, images, targets)

# Apply the patch
GeneralizedRCNN.forward = patched_forward


def get_transform():
    """
    Get transformations for the dataset.
    
    Returns:
        transform (callable): Transformation function
    """
    transforms = []
    # Add ToTensor transform
    transforms.append(utils.ToTensor())
    
    # Add the custom transform
    transforms.append(EnsureSingleChannel())
    
    return utils.Compose(transforms)


def evaluate_model(config):
    """
    Evaluate the Mask R-CNN model.
    
    Args:
        config: Configuration object with evaluation parameters
    """
    print(f"Loading COCO annotations from {config.COCO_ANNOTATIONS_PATH}")
    
    # Load COCO annotations
    coco_data = utils.load_coco_annotations(config.COCO_ANNOTATIONS_PATH)
    
    # Check if test set exists
    if not config.TEST_IMAGES:
        print("No test set defined. Using all images for evaluation.")
        config.TEST_IMAGES = [img['id'] for img in coco_data['images']]
    
    print(f"Evaluating on {len(config.TEST_IMAGES)} test images")
    
    # Create test dataset
    test_dataset = EMPSDataset(
        config=config,
        coco_data=coco_data,
        image_ids=config.TEST_IMAGES,
        transforms=get_transform()
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Use main process only to avoid multiprocessing issues
        collate_fn=utils.collate_fn
    )
    
    # Initialize model
    print(f"Initializing Mask R-CNN with {config.BACKBONE} backbone...")
    model = get_model_instance_segmentation(
        num_classes=config.NUM_CLASSES,
        backbone=config.BACKBONE,
        pretrained=False,  # We'll load weights from checkpoint
        trainable_backbone_layers=0  # No need to train for inference
    )
    
    # Configure model for grayscale input
    if config.IMAGE_CHANNEL_COUNT == 1:
        print("Configuring model for grayscale input...")
        model = configure_model_for_grayscale(model)
    
    # Move model to device
    model.to(config.DEVICE)
    
    # Load model checkpoint
    if not config.CHECKPOINT_PATH:
        # Try to find the best model checkpoint
        checkpoint_dir = os.path.join(config.ROOT_DIR, "models", "checkpoints")
        best_model_path = os.path.join(checkpoint_dir, "mask_rcnn_best.pth")
        if os.path.exists(best_model_path):
            config.CHECKPOINT_PATH = best_model_path
            print(f"Using best model checkpoint: {best_model_path}")
        else:
            print("No checkpoint found. Please specify a checkpoint path.")
            return
    
    print(f"Loading model from checkpoint: {config.CHECKPOINT_PATH}")
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    # Create output directories
    predictions_dir = os.path.join(config.ROOT_DIR, "outputs", "predictions")
    visualizations_dir = os.path.join(config.ROOT_DIR, "outputs", "visualizations")
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Initialize metrics
    metrics = {
        'image_id': [],
        'dice_coefficient': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'pixel_error': [],
        'rand_error': []
    }
    
    # Evaluate model
    print("Evaluating model...")
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            # Move images to device
            images = [image.to(config.DEVICE) for image in images]
            
            # Run inference
            outputs = model(images)
            
            # Process each image in the batch
            for i, (image, target, output) in enumerate(zip(images, targets, outputs)):
                image_id = target['image_id'].item()
                metrics['image_id'].append(image_id)
                
                # Get ground truth mask
                gt_mask = target['masks'].cpu().numpy()
                
                # Get predicted mask
                if len(output['masks']) > 0:
                    # Get mask with highest score
                    scores = output['scores']
                    if len(scores) > 0:
                        # Filter predictions by score threshold
                        keep = scores > 0.5
                        if keep.sum() > 0:
                            masks = output['masks'][keep]
                            scores = scores[keep]
                            
                            # Get mask with highest score
                            best_idx = torch.argmax(scores)
                            pred_mask = masks[best_idx, 0].cpu().numpy()
                            
                            # Convert to binary mask
                            pred_mask = (pred_mask > 0.5).astype(np.float32)
                        else:
                            # No predictions above threshold
                            pred_mask = np.zeros_like(gt_mask)
                    else:
                        # No predictions
                        pred_mask = np.zeros_like(gt_mask)
                else:
                    # No predictions
                    pred_mask = np.zeros_like(gt_mask)
                
                # Calculate metrics
                dice = utils.compute_dice_coefficient(gt_mask, pred_mask)
                iou = utils.compute_iou(gt_mask, pred_mask)
                precision, recall, f1 = utils.compute_precision_recall_f1(gt_mask, pred_mask)
                pixel_error = utils.compute_pixel_error(gt_mask, pred_mask)
                rand_error = utils.compute_rand_error(gt_mask, pred_mask)
                
                # Store metrics
                metrics['dice_coefficient'].append(dice)
                metrics['iou'].append(iou)
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1_score'].append(f1)
                metrics['pixel_error'].append(pixel_error)
                metrics['rand_error'].append(rand_error)
                
                # Save visualization
                original_image = image.cpu().permute(1, 2, 0).numpy()
                if original_image.shape[2] == 1:
                    original_image = original_image[:, :, 0]
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(original_image, cmap='gray')
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Ground truth mask
                axes[1].imshow(original_image, cmap='gray')
                axes[1].imshow(gt_mask, alpha=0.5, cmap='jet')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # Predicted mask
                axes[2].imshow(original_image, cmap='gray')
                axes[2].imshow(pred_mask, alpha=0.5, cmap='jet')
                axes[2].set_title(f'Prediction (Dice={dice:.4f}, IoU={iou:.4f})')
                axes[2].axis('off')
                
                # Save visualization
                vis_path = os.path.join(visualizations_dir, f"image_{image_id}.png")
                plt.tight_layout()
                plt.savefig(vis_path, dpi=150)
                plt.close()
                
                # Save prediction as numpy array
                pred_path = os.path.join(predictions_dir, f"pred_{image_id}.npy")
                np.save(pred_path, pred_mask)
    
    # Calculate average metrics
    avg_metrics = {
        'dice_coefficient': np.mean(metrics['dice_coefficient']),
        'iou': np.mean(metrics['iou']),
        'precision': np.mean(metrics['precision']),
        'recall': np.mean(metrics['recall']),
        'f1_score': np.mean(metrics['f1_score']),
        'pixel_error': np.mean(metrics['pixel_error']),
        'rand_error': np.mean(metrics['rand_error'])
    }
    
    # Print average metrics
    print("\nEvaluation Results:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(predictions_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    
    # Save average metrics to JSON
    avg_metrics_path = os.path.join(predictions_dir, "avg_metrics.json")
    with open(avg_metrics_path, 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    print(f"Average metrics saved to {avg_metrics_path}")
    
    # Plot metrics distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Dice and IoU
    axes[0, 0].hist(metrics['dice_coefficient'], bins=20, alpha=0.7)
    axes[0, 0].set_title(f'Dice Coefficient (Avg: {avg_metrics["dice_coefficient"]:.4f})')
    axes[0, 0].set_xlabel('Dice Coefficient')
    axes[0, 0].set_ylabel('Count')
    
    axes[0, 1].hist(metrics['iou'], bins=20, alpha=0.7)
    axes[0, 1].set_title(f'IoU (Avg: {avg_metrics["iou"]:.4f})')
    axes[0, 1].set_xlabel('IoU')
    axes[0, 1].set_ylabel('Count')
    
    # Precision and Recall
    axes[1, 0].hist(metrics['precision'], bins=20, alpha=0.7)
    axes[1, 0].set_title(f'Precision (Avg: {avg_metrics["precision"]:.4f})')
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_ylabel('Count')
    
    axes[1, 1].hist(metrics['recall'], bins=20, alpha=0.7)
    axes[1, 1].set_title(f'Recall (Avg: {avg_metrics["recall"]:.4f})')
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    metrics_plot_path = os.path.join(visualizations_dir, "metrics_distribution.png")
    plt.savefig(metrics_plot_path, dpi=150)
    plt.close()
    print(f"Metrics distribution plot saved to {metrics_plot_path}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    # Load configuration
    config = InferenceConfig()
    config.display()
    
    # Evaluate model
    evaluate_model(config) 