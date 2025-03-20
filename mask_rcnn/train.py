"""
Training script for Mask R-CNN for electron microscopy particle segmentation.
"""

import os
import sys
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.ops import box_iou
import cv2

# Import local modules
from config import TrainingConfig
import utils
from datasets.emps_dataset import EMPSDataset
from models.mask_rcnn import get_model_instance_segmentation, configure_model_for_grayscale, CombinedLoss


# Custom transform to ensure single-channel output
class EnsureSingleChannel:
    def __call__(self, image, target=None):
        if isinstance(image, torch.Tensor) and image.dim() == 3 and image.shape[0] == 3:
            # Convert 3-channel to 1-channel by taking the first channel
            image = image[0:1, :, :]
        return (image, target) if target is not None else image


# Patch the GeneralizedRCNN forward method to handle grayscale images
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





# import json
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import pycocotools.mask as mask_util
# from PIL import Image
# import torch  # Import torch to check tensor type

# # Draw bounding boxes
# def draw_bboxes(image, target):
#     boxes = target['boxes']
#     if isinstance(boxes, torch.Tensor):  # Convert tensor to numpy
#         boxes = boxes.cpu().numpy()
    
#     for box in boxes:
#         x1, y1, x2, y2 = box  # Using (x_min, y_min, x_max, y_max)
#         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

# # Draw segmentation masks
# def draw_masks(image, target):
#     masks = target['masks']
    
#     if isinstance(masks, torch.Tensor):  # Convert tensor to numpy
#         masks = masks.cpu().numpy()
    
#     for mask in masks:
#         if isinstance(mask, list):  # Polygon segmentation
#             for seg in mask:
#                 points = np.array(seg, dtype=np.int32).reshape((-1, 2))
#                 cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=1)

# # Show image with annotations
# def show_image_with_annotations(image, target):
#     image_with_bboxes = image.cpu().numpy().copy()
#     image_with_bboxes = np.transpose(image_with_bboxes, (1, 2, 0))
#     if image_with_bboxes.max() <= 1.0:
#         image_with_bboxes = (image_with_bboxes * 255).astype(np.uint8)
#     if len(image_with_bboxes.shape) == 2 or image_with_bboxes.shape[2] == 1:
#         image_with_bboxes = cv2.cvtColor(image_with_bboxes, cv2.COLOR_GRAY2RGB)
#     vis_image = image_with_bboxes.copy()
    
#     # Create figure and axes
#     fig, ax = plt.subplots(1, figsize=(12, 12))
#     ax.imshow(vis_image)

#     # draw_bboxes(vis_image, target)
#     # image_with_annotations = draw_masks(image_with_bboxes, target)
#     draw_bboxes(vis_image, target)
#     draw_masks(vis_image, target)

#     # Show the image
#     plt.figure(figsize=(10, 10))
#     plt.imshow(vis_image)
#     plt.axis('off')
#     plt.show()





def get_transform(train, config=None):
    """
    Get the transforms to apply to the dataset.
    
    Args:
        train: Whether to get transforms for training or validation
        config: Configuration object
        
    Returns:
        Compose object with transforms
    """
    transforms = []
    
    # Always add EnsureSingleChannel transform
    transforms.append(EnsureSingleChannel())
    
    # Add data augmentation transforms for training
    if train and config and config.USE_AUGMENTATION:
        # Add random horizontal flip
        if config.AUGMENTATION.get("horizontal_flip", False):
            transforms.append(utils.RandomHorizontalFlip(0.5))
        
        # Add random vertical flip
        if config.AUGMENTATION.get("vertical_flip", False):
            transforms.append(utils.RandomVerticalFlip(0.5))
        
        # Add random rotation
        if config.AUGMENTATION.get("rotation_range", 0) > 0:
            transforms.append(utils.RandomRotation(config.AUGMENTATION["rotation_range"]))
    
    # Always add ToTensor transform
    transforms.append(utils.ToTensor())
    
    return utils.Compose(transforms)


def train_model(config):
    """
    Train the Mask R-CNN model.
    
    Args:
        config: Configuration object with training parameters
    """
    print(f"Loading COCO annotations from {config.COCO_ANNOTATIONS_PATH}")
    
    # Load COCO annotations
    coco_data = utils.load_coco_annotations(config.COCO_ANNOTATIONS_PATH)
    
    # Split dataset if not already split
    if not config.TRAIN_IMAGES and not config.VAL_IMAGES and not config.TEST_IMAGES:
        print("Splitting dataset into train, validation, and test sets...")
        image_ids = [img['id'] for img in coco_data['images']]
        train_ids, val_ids, test_ids = utils.split_dataset(
            image_ids, 
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            test_ratio=config.TEST_RATIO
        )
        config.TRAIN_IMAGES = train_ids
        config.VAL_IMAGES = val_ids
        config.TEST_IMAGES = test_ids
        print(f"Dataset split: {len(train_ids)} training, {len(val_ids)} validation, {len(test_ids)} test images")
    
    # Create datasets and data loaders
    print("Creating datasets and data loaders...")
    
    # Create datasets
    train_dataset = EMPSDataset(
        config=config,
        coco_data=coco_data,
        image_ids=train_ids,
        transforms=get_transform(train=True, config=config)
    )
    
    val_dataset = EMPSDataset(
        config=config,
        coco_data=coco_data,
        image_ids=val_ids,
        transforms=get_transform(train=False, config=config)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use main process only to avoid multiprocessing issues
        collate_fn=utils.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
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
        pretrained=True,
        trainable_backbone_layers=3
    )
    
    # Configure model for grayscale input
    if config.IMAGE_CHANNEL_COUNT == 1:
        print("Configuring model for grayscale input...")
        model = configure_model_for_grayscale(model)
    
    # Move model to device
    model.to(config.DEVICE)
    
    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=3, 
        verbose=True
    )
    
    # Initialize loss function
    criterion = CombinedLoss(dice_weight=0.5)
    
    # Create TensorBoard writer
    log_dir = os.path.join(config.MODEL_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create directory for model checkpoints
    checkpoint_dir = os.path.join(config.ROOT_DIR, "models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for images, targets in train_progress:
            # Move data to device
            images = [image.to(config.DEVICE) for image in images]
            targets = [{k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Calculate total loss
            if isinstance(loss_dict, dict):
                losses = sum(loss for loss in loss_dict.values())
            elif isinstance(loss_dict, list):
                # Handle list of dictionaries
                total_loss = 0.0
                for item in loss_dict:
                    if isinstance(item, dict):
                        # Extract loss values individually
                        for key, value in item.items():
                            if isinstance(value, torch.Tensor) and value.numel() == 1:
                                total_loss += value.item()
                    elif isinstance(item, (int, float)):
                        total_loss += item
                    elif isinstance(item, torch.Tensor) and item.numel() == 1:
                        total_loss += item.item()
                losses = torch.tensor(total_loss, device=config.DEVICE, requires_grad=True)
            else:
                losses = loss_dict  # In case it's already a tensor
            
            # Backward pass and optimize
            losses.backward()
            optimizer.step()
            
            # Update progress bar
            train_loss += losses.item() if isinstance(losses, torch.Tensor) else losses
            train_progress.set_postfix({"Loss": losses.item() if isinstance(losses, torch.Tensor) else losses})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mask_iou = 0.0
        val_progress = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_progress):
                # Move data to device
                images = [image.to(config.DEVICE) for image in images]
                targets = [{k: v.to(config.DEVICE) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # show_image_with_annotations(images[0], targets[0])
                
                # Get predictions
                predictions = model(images)
                
                # Compute metrics using utility functions
                batch_metrics = utils.evaluate_batch(predictions, targets)
                
                # Extract box loss (1 - IoU) and mask IoU
                batch_loss = 1.0 - batch_metrics.get('precision', 0)
                batch_mask_iou = batch_metrics.get('mask_iou', 0)
                
                # Update progress
                val_loss += batch_loss
                val_mask_iou += batch_mask_iou
                val_progress.set_postfix({"Box Loss": batch_loss, "Mask IoU": batch_mask_iou})
                
                # Visualize validation images if enabled
                if config.VISUALIZE_VAL_IMAGES > 0 and batch_idx == 0:
                    # Create visualization directory
                    vis_dir = os.path.join(config.VISUALIZATION_DIR, f"epoch_{epoch+1}")
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # Get image IDs or filenames for the current batch
                    batch_image_ids = []
                    for i, target in enumerate(targets):
                        if 'file_name' in target:
                            # Use filename (without extension) for more readable visualization names
                            file_name = os.path.splitext(target['file_name'])[0]
                            batch_image_ids.append(file_name)
                        elif 'image_id' in target:
                            batch_image_ids.append(target['image_id'])
                        else:
                            batch_image_ids.append(f"img_{batch_idx}_{i}")
                    
                    # Visualize and save batch
                    utils.visualize_and_save_batch(
                        images, 
                        predictions, 
                        targets, 
                        output_dir=vis_dir, 
                        prefix=f"val_batch_{batch_idx}", 
                        max_images=config.VISUALIZE_VAL_IMAGES,
                        image_ids=batch_image_ids
                    )
        
        # Calculate averages
        avg_val_loss = val_loss / len(val_loader)
        avg_mask_iou = val_mask_iou / len(val_loader)
        
        print(f"Validation Loss: {avg_val_loss:.4f}, Mask IoU: {avg_mask_iou:.4f}")
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('IoU/mask', avg_mask_iou, epoch)
        
        # Update learning rate
        lr_scheduler.step(avg_val_loss)
        
        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"mask_rcnn_epoch_{epoch+1}.pth")
            utils.save_checkpoint(model, optimizer, epoch, checkpoint_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    print("Training complete!")
    writer.close()


if __name__ == "__main__":
    # Create configuration
    config = TrainingConfig()
    
    # Enable visualization and disable augmentation for testing
    config.VISUALIZE_VAL_IMAGES = 4  # Visualize 4 validation images
    config.USE_AUGMENTATION = False  # Disable data augmentation
    
    # Display configuration
    config.display()
    
    # Train the model
    train_model(config)


