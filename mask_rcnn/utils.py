"""
Utility functions for Mask R-CNN for electron microscopy particle segmentation.
"""

import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from pycocotools import mask as maskUtils


# Dataset Utilities
def load_coco_annotations(annotations_path):
    """
    Load COCO format annotations.
    
    Args:
        annotations_path: Path to the COCO annotations JSON file.
        
    Returns:
        Dictionary containing parsed COCO annotations.
    """
    with open(annotations_path, 'r') as f:
        return json.load(f)


def split_dataset(image_ids, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_seed=42):
    """
    Split dataset into training, validation, and testing sets.
    
    Args:
        image_ids: List of image IDs.
        train_ratio: Ratio of images for training.
        val_ratio: Ratio of images for validation.
        test_ratio: Ratio of images for testing.
        random_seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    # Shuffle image IDs
    random.seed(random_seed)
    image_ids_shuffled = image_ids.copy()
    random.shuffle(image_ids_shuffled)
    
    # Calculate split indices
    n_images = len(image_ids_shuffled)
    n_train = int(n_images * train_ratio)
    n_val = int(n_images * val_ratio)
    
    # Split dataset
    train_ids = image_ids_shuffled[:n_train]
    val_ids = image_ids_shuffled[n_train:n_train + n_val]
    test_ids = image_ids_shuffled[n_train + n_val:]
    
    return train_ids, val_ids, test_ids


def resize_image(image, min_dim=512, max_dim=512, maintain_aspect_ratio=False):
    """
    Resize image to target dimensions.
    
    Args:
        image: Input image (numpy array).
        min_dim: Minimum dimension after resizing.
        max_dim: Maximum dimension after resizing.
        maintain_aspect_ratio: If True, maintain aspect ratio and pad if needed.
        
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    # Get original dimensions
    h, w = image.shape[:2]
    
    if maintain_aspect_ratio:
        raise ValueError("This function is not implemented yet")
        # Calculate scale factor to maintain aspect ratio
        scale = min(min_dim / min(h, w), max_dim / max(h, w))
        
        # Calculate new dimensions
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to ensure dimensions are exactly min_dim x max_dim
        if min_dim == max_dim:
            # If min_dim equals max_dim, pad to make a square image
            padded_image = np.zeros((min_dim, min_dim), dtype=resized_image.dtype)
            # Center the resized image in the padded image
            y_offset = (min_dim - new_h) // 2
            x_offset = (min_dim - new_w) // 2
            padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image
            return padded_image, scale
        
        return resized_image, scale
    else:
        # Simple resize to target dimensions without maintaining aspect ratio
        resized_image = cv2.resize(image, (max_dim, min_dim), interpolation=cv2.INTER_AREA)
        
        # Calculate scale factors for width and height
        scale_w = max_dim / w
        scale_h = min_dim / h
        
        # Return average scale factor
        return resized_image, (scale_w + scale_h) / 2


# Data Transformations
class Compose:
    """Compose several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target=None):
        if target is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target


class ToTensor:
    """Convert image and mask to PyTorch tensors."""
    def __call__(self, image, target=None):
        # If image is already a tensor, don't convert it again
        if not isinstance(image, torch.Tensor):
            # Convert image to tensor
            if isinstance(image, np.ndarray):
                # Handle numpy array
                if image.ndim == 2:
                    image = image[:, :, np.newaxis]
                image = torch.from_numpy(image.transpose((2, 0, 1)))
                if isinstance(image, torch.ByteTensor):
                    image = image.float().div(255)
            else:
                # Handle PIL Image
                image = TF.to_tensor(image)
        
        if target is None:
            return image
        
        # Convert target to tensor if it's not already
        if isinstance(target, dict):
            for k, v in target.items():
                if isinstance(v, np.ndarray):
                    target[k] = torch.from_numpy(v)
        
        return image, target


class RandomHorizontalFlip:
    """Randomly flip image and mask horizontally."""
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            # Get image dimensions
            if isinstance(image, torch.Tensor):
                _, height, width = image.shape
            else:
                height, width = image.shape[:2]
            
            # Flip image
            if isinstance(image, torch.Tensor):
                image = image.flip(-1)
            else:
                image = np.fliplr(image).copy()
            
            # Flip masks in target
            if 'masks' in target and len(target['masks']) > 0:
                if isinstance(target['masks'], torch.Tensor):
                    target['masks'] = target['masks'].flip(-1)
                else:
                    target['masks'] = np.fliplr(target['masks']).copy()
            
            # Flip boxes in target
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                if isinstance(boxes, torch.Tensor):
                    flipped_boxes = boxes.clone()
                    flipped_boxes[:, 0] = width - boxes[:, 2]
                    flipped_boxes[:, 2] = width - boxes[:, 0]
                    target['boxes'] = flipped_boxes
                else:
                    flipped_boxes = boxes.copy()
                    flipped_boxes[:, 0] = width - boxes[:, 2]
                    flipped_boxes[:, 2] = width - boxes[:, 0]
                    target['boxes'] = flipped_boxes
        
        return image, target


class RandomVerticalFlip:
    """Randomly flip image and mask vertically."""
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if random.random() < self.prob:
            # Get image dimensions
            if isinstance(image, torch.Tensor):
                _, height, width = image.shape
            else:
                height, width = image.shape[:2]
            
            # Flip image
            if isinstance(image, torch.Tensor):
                image = image.flip(-2)
            else:
                image = np.flipud(image).copy()
            
            # Flip masks in target
            if 'masks' in target and len(target['masks']) > 0:
                if isinstance(target['masks'], torch.Tensor):
                    target['masks'] = target['masks'].flip(-2)
                else:
                    target['masks'] = np.flipud(target['masks']).copy()
            
            # Flip boxes in target
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                if isinstance(boxes, torch.Tensor):
                    flipped_boxes = boxes.clone()
                    flipped_boxes[:, 1] = height - boxes[:, 3]
                    flipped_boxes[:, 3] = height - boxes[:, 1]
                    target['boxes'] = flipped_boxes
                else:
                    flipped_boxes = boxes.copy()
                    flipped_boxes[:, 1] = height - boxes[:, 3]
                    flipped_boxes[:, 3] = height - boxes[:, 1]
                    target['boxes'] = flipped_boxes
        
        return image, target


class RandomRotation:
    """Randomly rotate image and mask."""
    def __init__(self, degrees):
        self.degrees = degrees
        
    def __call__(self, image, target):
        # Choose random angle
        angle = random.choice([0, 90, 180, 270]) if self.degrees == 90 else random.uniform(-self.degrees, self.degrees)
        
        if angle == 0:
            return image, target
        
        # Get image dimensions
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2], image.shape[-1]
        else:
            height, width = image.shape[:2]
        
        # Rotate image
        if isinstance(image, torch.Tensor):
            # Convert to PIL for rotation
            image_pil = TF.to_pil_image(image)
            image_pil = TF.rotate(image_pil, angle)
            image = TF.to_tensor(image_pil)
        else:
            # OpenCV rotation
            center = (width // 2, height // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_LINEAR)
        
        # Rotate masks in target
        if 'masks' in target and len(target['masks']) > 0:
            if isinstance(target['masks'], torch.Tensor):
                # Convert to numpy for rotation
                masks = target['masks'].cpu().numpy()
                rotated_masks = []
                for mask in masks:
                    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST)
                    rotated_mask = np.array(mask_pil) / 255.0
                    rotated_masks.append(rotated_mask)
                target['masks'] = torch.tensor(np.array(rotated_masks), dtype=torch.uint8)
            else:
                # OpenCV rotation
                rotated_masks = []
                for mask in target['masks']:
                    mask_img = (mask * 255).astype(np.uint8)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated_mask = cv2.warpAffine(mask_img, M, (width, height), flags=cv2.INTER_NEAREST)
                    rotated_masks.append(rotated_mask > 0)
                target['masks'] = np.array(rotated_masks)
        
        # Update boxes in target
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            
            # Convert boxes to corners representation
            corners = self._get_corners(boxes)
            
            # Rotate corners
            corners = self._rotate_corners(corners, angle, height, width)
            
            # Get axis-aligned bounding boxes
            rotated_boxes = self._get_enclosing_box(corners)
            
            # Update boxes in target
            if isinstance(boxes, torch.Tensor):
                target['boxes'] = torch.tensor(rotated_boxes, dtype=torch.float32)
            else:
                target['boxes'] = rotated_boxes
        
        return image, target
    
    def _get_corners(self, boxes):
        """
        Get the four corners of the bounding boxes.
        
        Args:
            boxes: Bounding boxes in [x1, y1, x2, y2] format
            
        Returns:
            corners: Array of shape (N, 4, 2) with corners coordinates
        """
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + width
        y2 = boxes[:, 1]
        x3 = boxes[:, 0] + width
        y3 = boxes[:, 1] + height
        x4 = boxes[:, 0]
        y4 = boxes[:, 1] + height
        
        corners = np.stack([
            np.stack([x1, y1], axis=1),
            np.stack([x2, y2], axis=1),
            np.stack([x3, y3], axis=1),
            np.stack([x4, y4], axis=1)
        ], axis=1)
        
        return corners
    
    def _rotate_corners(self, corners, angle, height, width):
        """
        Rotate the corners of the bounding boxes.
        
        Args:
            corners: Array of shape (N, 4, 2) with corners coordinates
            angle: Rotation angle in degrees
            height: Image height
            width: Image width
            
        Returns:
            rotated_corners: Array of shape (N, 4, 2) with rotated corners
        """
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate center of the image
        cx, cy = width / 2, height / 2
        
        # Create rotation matrix
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Flatten corners for vectorized operations
        corners_flat = corners.reshape(-1, 2)
        
        # Translate corners to origin
        corners_flat[:, 0] -= cx
        corners_flat[:, 1] -= cy
        
        # Rotate corners
        rotated_x = corners_flat[:, 0] * cos_angle - corners_flat[:, 1] * sin_angle
        rotated_y = corners_flat[:, 0] * sin_angle + corners_flat[:, 1] * cos_angle
        
        # Translate corners back
        rotated_x += cx
        rotated_y += cy
        
        # Reshape back to (N, 4, 2)
        rotated_corners = np.stack([rotated_x, rotated_y], axis=1).reshape(corners.shape)
        
        return rotated_corners
    
    def _get_enclosing_box(self, corners):
        """
        Get the axis-aligned bounding box that encloses the rotated box.
        
        Args:
            corners: Array of shape (N, 4, 2) with corners coordinates
            
        Returns:
            boxes: Array of shape (N, 4) with [x1, y1, x2, y2] coordinates
        """
        # Get min and max coordinates
        min_x = np.min(corners[:, :, 0], axis=1)
        min_y = np.min(corners[:, :, 1], axis=1)
        max_x = np.max(corners[:, :, 0], axis=1)
        max_y = np.max(corners[:, :, 1], axis=1)
        
        # Create boxes
        boxes = np.stack([min_x, min_y, max_x, max_y], axis=1)
        
        return boxes


# Collate function for data loader
def collate_fn(batch):
    """
    Custom collate function for data loader.
    
    Args:
        batch: List of tuples (image, target)
        
    Returns:
        Tuple of (images, targets)
    """
    return tuple(zip(*batch))


# Evaluation Metrics
def compute_iou(gt_mask, pred_mask, smooth=1e-6):
    """
    Compute Intersection over Union (IoU) between two binary masks.
    
    Args:
        gt_mask: Ground truth binary mask.
        pred_mask: Predicted binary mask.
        smooth: Smoothing factor to avoid division by zero.
        
    Returns:
        IoU score.
    """
    # Convert to binary masks
    gt_mask = gt_mask > 0.5
    pred_mask = pred_mask > 0.5
    
    # Calculate intersection and union
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def compute_dice_coefficient(gt_mask, pred_mask, smooth=1e-6):
    """
    Compute Dice coefficient between two binary masks.
    
    Args:
        gt_mask: Ground truth binary mask.
        pred_mask: Predicted binary mask.
        smooth: Smoothing factor to avoid division by zero.
        
    Returns:
        Dice coefficient.
    """
    # Convert to binary masks
    gt_mask = gt_mask > 0.5
    pred_mask = pred_mask > 0.5
    
    # Calculate intersection and sum of areas
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    gt_area = gt_mask.sum()
    pred_area = pred_mask.sum()
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (gt_area + pred_area + smooth)
    
    return dice


def compute_precision_recall_f1(gt_mask, pred_mask, smooth=1e-6):
    """
    Compute precision, recall, and F1 score between two binary masks.
    
    Args:
        gt_mask: Ground truth binary mask.
        pred_mask: Predicted binary mask.
        smooth: Smoothing factor to avoid division by zero.
        
    Returns:
        Tuple of (precision, recall, f1_score).
    """
    # Convert to binary masks
    gt_mask = gt_mask > 0.5
    pred_mask = pred_mask > 0.5
    
    # Calculate true positives, false positives, and false negatives
    tp = np.logical_and(gt_mask, pred_mask).sum()
    fp = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()
    fn = np.logical_and(gt_mask, np.logical_not(pred_mask)).sum()
    
    # Calculate precision, recall, and F1 score
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1_score = (2.0 * precision * recall + smooth) / (precision + recall + smooth)
    
    return precision, recall, f1_score


def compute_pixel_error(gt_mask, pred_mask):
    """
    Compute pixel error (percentage of misclassified pixels).
    
    Args:
        gt_mask: Ground truth binary mask.
        pred_mask: Predicted binary mask.
        
    Returns:
        Pixel error.
    """
    # Convert to binary masks
    gt_mask = gt_mask > 0.5
    pred_mask = pred_mask > 0.5
    
    # Calculate pixel error
    pixel_error = np.mean(np.abs(gt_mask.astype(float) - pred_mask.astype(float)))
    
    return pixel_error


def compute_rand_error(gt_mask, pred_mask):
    """
    Compute Rand error (simplified version).
    
    Args:
        gt_mask: Ground truth binary mask.
        pred_mask: Predicted binary mask.
        
    Returns:
        Rand error.
    """
    # Convert to binary masks
    gt_mask = gt_mask > 0.5
    pred_mask = pred_mask > 0.5
    
    # Calculate Rand error
    rand_error = 1.0 - np.mean((gt_mask == pred_mask).astype(float))
    
    return rand_error


# Image Processing Utilities
def apply_augmentation(image, mask, config):
    """
    Apply augmentation to image and mask.
    
    Args:
        image: Image to augment.
        mask: Mask to augment.
        config: Configuration object with augmentation settings.
        
    Returns:
        Augmented image and mask.
    """
    # Apply horizontal flip
    if config.AUGMENTATION.get("horizontal_flip", False) and random.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    
    # Apply vertical flip
    if config.AUGMENTATION.get("vertical_flip", False) and random.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    
    # Apply rotation
    if config.AUGMENTATION.get("rotation_range", 0) > 0:
        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            # Rotate image
            M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
            image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    
    # Apply brightness adjustment
    if config.AUGMENTATION.get("brightness_range", None):
        brightness_range = config.AUGMENTATION["brightness_range"]
        brightness_factor = random.uniform(brightness_range[0], brightness_range[1])
        image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    
    # Apply contrast adjustment
    if config.AUGMENTATION.get("contrast_range", None):
        contrast_range = config.AUGMENTATION["contrast_range"]
        contrast_factor = random.uniform(contrast_range[0], contrast_range[1])
        image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    
    # Apply shear
    if config.AUGMENTATION.get("shear_range", 0) > 0:
        shear_range = config.AUGMENTATION["shear_range"]
        shear_factor = random.uniform(-shear_range, shear_range)
        
        # Create shear matrix
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        
        # Apply shear
        image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
    
    return image, mask


# Visualization Utilities
def random_colors(N, bright=True):
    """
    Generate random colors.
    
    Args:
        N: Number of colors to generate.
        bright: Whether to generate bright colors.
        
    Returns:
        List of colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: plt.cm.hsv(c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply a mask to an image.
    
    Args:
        image: Image to apply mask to.
        mask: Mask to apply.
        color: Color for the mask.
        alpha: Transparency of the mask.
        
    Returns:
        Image with mask applied.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, masks, boxes=None, class_ids=None, class_names=None,
                      scores=None, title="", figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True, colors=None, captions=None):
    """
    Display instance segmentation results.
    
    Args:
        image: Image to display.
        masks: Instance masks.
        boxes: Bounding boxes.
        class_ids: Class IDs for each instance.
        class_names: Class names for each ID.
        scores: Detection scores.
        title: Figure title.
        figsize: Figure size.
        ax: Matplotlib axis.
        show_mask: Whether to show masks.
        show_bbox: Whether to show bounding boxes.
        colors: Colors for each instance.
        captions: Captions for each instance.
        
    Returns:
        Matplotlib axis with the visualization.
    """
    # Number of instances
    N = masks.shape[2] if masks.ndim == 3 else 1
    if N == 0:
        print("No instances to display")
        return
    
    # If no axis is passed, create a new figure and axis
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
    
    # Generate random colors if not provided
    if colors is None:
        colors = random_colors(N)
    
    # Show the image
    if image.ndim == 2:
        # Convert grayscale to RGB
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    elif image.shape[2] == 1:
        # Convert single channel to RGB
        image = np.repeat(image, 3, axis=2)
    
    # Normalize image if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    ax.imshow(image)
    
    # Show title
    if title:
        ax.set_title(title, fontsize=18)
    
    # Show masks and bounding boxes
    for i in range(N):
        color = colors[i]
        
        # Mask
        if show_mask:
            mask = masks[:, :, i] if masks.ndim == 3 else masks
            masked_image = apply_mask(image.copy(), mask, color)
            ax.imshow(masked_image)
        
        # Bounding box
        if show_bbox and boxes is not None:
            y1, x1, y2, x2 = boxes[i]
            p = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                         alpha=0.7, linestyle="dashed",
                         edgecolor=color, facecolor='none')
            ax.add_patch(p)
        
        # Caption
        if captions is not None:
            caption = captions[i]
        elif class_names is not None and class_ids is not None:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = f"{label} {score:.3f}" if score else label
        else:
            caption = f"Mask {i}"
        
        # Add caption
        if boxes is not None:
            y1, x1, y2, x2 = boxes[i]
            ax.text(x1, y1 - 8, caption, color='w', size=12, backgroundcolor="none")
    
    # Remove axes and set tight layout
    ax.axis('off')
    plt.tight_layout()
    
    return ax


def display_differences(image, gt_mask, pred_mask, title="", figsize=(16, 5)):
    """
    Display differences between ground truth and predicted masks.
    
    Args:
        image: Original image.
        gt_mask: Ground truth mask.
        pred_mask: Predicted mask.
        title: Figure title.
        figsize: Figure size.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Original image
    if image.ndim == 2:
        # Convert grayscale to RGB
        image_rgb = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    elif image.shape[2] == 1:
        # Convert single channel to RGB
        image_rgb = np.repeat(image, 3, axis=2)
    else:
        image_rgb = image.copy()
    
    # Normalize image if needed
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Ground truth mask
    axes[1].imshow(gt_mask, cmap='inferno')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    # Predicted mask
    axes[2].imshow(pred_mask, cmap='inferno')
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    
    # Difference
    diff = np.zeros_like(image_rgb)
    diff[:, :, 0] = np.logical_and(gt_mask, np.logical_not(pred_mask)) * 255  # False negatives (red)
    diff[:, :, 1] = np.logical_and(gt_mask, pred_mask) * 255  # True positives (green)
    diff[:, :, 2] = np.logical_and(np.logical_not(gt_mask), pred_mask) * 255  # False positives (blue)
    
    axes[3].imshow(diff)
    axes[3].set_title("Differences")
    axes[3].axis('off')
    
    # Add a legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='r', label='False Negative'),
        plt.Rectangle((0, 0), 1, 1, color='g', label='True Positive'),
        plt.Rectangle((0, 0), 1, 1, color='b', label='False Positive')
    ]
    axes[3].legend(handles=legend_elements, loc='lower right')
    
    # Set title for the figure
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return fig


def plot_training_metrics(metrics_df, save_path=None):
    """
    Plot training metrics.
    
    Args:
        metrics_df: DataFrame with training metrics.
        save_path: Path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss plot
    axes[0, 0].plot(metrics_df['epoch'], metrics_df['train_loss'], label='Training Loss')
    axes[0, 0].plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss over Epochs')
    axes[0, 0].legend()
    
    # Precision, Recall, F1 plot
    axes[0, 1].plot(metrics_df['epoch'], metrics_df['precision'], label='Precision')
    axes[0, 1].plot(metrics_df['epoch'], metrics_df['recall'], label='Recall')
    axes[0, 1].plot(metrics_df['epoch'], metrics_df['f1'], label='F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Metric')
    axes[0, 1].set_title('Precision, Recall, F1 over Epochs')
    axes[0, 1].legend()
    
    # IoU and Dice plot
    axes[1, 0].plot(metrics_df['epoch'], metrics_df['iou'], label='IoU')
    axes[1, 0].plot(metrics_df['epoch'], metrics_df['dice_coeff'], label='Dice Coefficient')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Metric')
    axes[1, 0].set_title('IoU and Dice Coefficient over Epochs')
    axes[1, 0].legend()
    
    # Error metrics plot
    axes[1, 1].plot(metrics_df['epoch'], metrics_df['pixel_error'], label='Pixel Error')
    axes[1, 1].plot(metrics_df['epoch'], metrics_df['rand_error'], label='Rand Error')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].set_title('Error Metrics over Epochs')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    return fig


# TensorBoard Utilities
def log_to_tensorboard(writer, tag, value, step):
    """
    Log a scalar value to TensorBoard.
    
    Args:
        writer: TensorBoard writer.
        tag: Data identifier.
        value: Value to log.
        step: Global step value.
    """
    writer.add_scalar(tag, value, step)


def log_images_to_tensorboard(writer, tag, images, step):
    """
    Log images to TensorBoard.
    
    Args:
        writer: TensorBoard writer.
        tag: Data identifier.
        images: Images to log.
        step: Global step value.
    """
    writer.add_images(tag, images, step)


# Model Checkpoint Utilities
def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save.
        optimizer: Optimizer state.
        epoch: Current epoch.
        filepath: Path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        filepath: Path to the checkpoint.
        
    Returns:
        Epoch number of the loaded checkpoint.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
    return epoch


# Evaluation Metrics for Object Detection and Instance Segmentation
def compute_box_iou_matrix(pred_boxes, target_boxes):
    """
    Compute IoU matrix between predicted and target boxes.
    
    Args:
        pred_boxes: Predicted boxes of shape (N, 4)
        target_boxes: Target boxes of shape (M, 4)
        
    Returns:
        IoU matrix of shape (N, M)
    """
    from torchvision.ops import box_iou
    return box_iou(pred_boxes, target_boxes)

def compute_mask_iou(pred_mask, target_mask, resize=True):
    """
    Compute IoU between predicted and target masks.
    
    Args:
        pred_mask: Predicted mask
        target_mask: Target mask
        resize: Whether to resize pred_mask to match target_mask size
        
    Returns:
        IoU score
    """
    # Convert to numpy if tensors
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.squeeze().cpu().numpy()
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.squeeze().cpu().numpy()
    
    # Resize predicted mask to match target mask size if needed
    if resize and pred_mask.shape != target_mask.shape:
        import cv2
        target_h, target_w = target_mask.shape
        pred_mask = cv2.resize(pred_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Convert to binary masks
    pred_mask = pred_mask > 0.5
    target_mask = target_mask > 0.5
    
    # Calculate intersection and union
    intersection = np.logical_and(target_mask, pred_mask).sum()
    union = np.logical_or(target_mask, pred_mask).sum()
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    
    return iou

def match_predictions_to_targets(predictions, targets, iou_threshold=0.5):
    """
    Match predicted boxes to target boxes based on IoU.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for considering a match
        
    Returns:
        List of (image_idx, pred_idx, target_idx, iou) tuples
    """
    matches = []
    
    for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
        if len(pred['boxes']) == 0 or len(target['boxes']) == 0:
            continue
            
        # Compute IoU matrix
        iou_matrix = compute_box_iou_matrix(pred['boxes'], target['boxes'])
        
        # For each target, find best matching prediction
        best_ious, best_idx = iou_matrix.max(dim=0)
        
        # Filter matches by IoU threshold
        valid_matches = best_ious > iou_threshold
        
        # Create list of matches
        for i, is_valid in enumerate(valid_matches):
            if is_valid:
                pred_idx = best_idx[i].item()
                target_idx = i
                iou = best_ious[i].item()
                matches.append((img_idx, pred_idx, target_idx, iou))
    
    return matches

def compute_detection_metrics(predictions, targets, iou_threshold=0.5):
    """
    Compute detection metrics (precision, recall, F1) for a batch.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for considering a match
        
    Returns:
        Dictionary of metrics
    """
    total_predictions = sum(len(p['boxes']) for p in predictions)
    total_targets = sum(len(t['boxes']) for t in targets)
    
    # Match predictions to targets
    matches = match_predictions_to_targets(predictions, targets, iou_threshold)
    true_positives = len(matches)
    
    # Compute metrics
    false_positives = total_predictions - true_positives
    false_negatives = total_targets - true_positives
    
    precision = true_positives / total_predictions if total_predictions > 0 else 0
    recall = true_positives / total_targets if total_targets > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def compute_segmentation_metrics(predictions, targets, iou_threshold=0.5):
    """
    Compute segmentation metrics for a batch.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        iou_threshold: IoU threshold for considering a match
        
    Returns:
        Dictionary of metrics
    """
    # Match predictions to targets
    matches = match_predictions_to_targets(predictions, targets, iou_threshold)
    
    # Compute mask IoU for matched pairs

    mask_ious = []
    for match in matches:
        img_idx, pred_idx, target_idx, _ = match
        pred = predictions[img_idx]
        target = targets[img_idx]

        # print('pred mask size: ', pred['masks'][pred_idx].shape)
        # print('target mask size: ', target['masks'][target_idx].shape)
        
        # resize the predicted mask to the size of the target mask
        mask_small = pred['masks'][pred_idx]
        mask_small = mask_small.squeeze().cpu().numpy()
        bbox = pred['boxes'][pred_idx]
        x1, y1, x2, y2 = bbox
        mask = np.zeros((512, 512))
        mask_resized = cv2.resize(mask_small, (int(x2)-int(x1), int(y2)-int(y1)),interpolation=cv2.INTER_LINEAR)
        mask[int(y1):int(y2), int(x1):int(x2)] = mask_resized
        
        target_mask = target['masks'][target_idx]
        mask_iou = compute_mask_iou(mask, target_mask)
        mask_ious.append(mask_iou)
    
    # Compute average mask IoU
    avg_mask_iou = sum(mask_ious) / len(mask_ious) if mask_ious else 0
    
    return {
        'mask_iou': avg_mask_iou,
        'num_matches': len(mask_ious)
    }

def evaluate_batch(predictions, targets, box_iou_threshold=0.5):
    """
    Evaluate predictions against targets for a batch.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        box_iou_threshold: IoU threshold for box matching
        
    Returns:
        Dictionary of metrics
    """
    # Compute detection metrics
    detection_metrics = compute_detection_metrics(predictions, targets, box_iou_threshold)
    
    # Compute segmentation metrics
    segmentation_metrics = compute_segmentation_metrics(predictions, targets, box_iou_threshold)
    
    # Combine metrics
    metrics = {**detection_metrics, **segmentation_metrics}
    
    return metrics


import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from PIL import Image
import torch  # Import torch to check tensor type

# Draw bounding boxes
def draw_bboxes(image, target):
    boxes = target['boxes']
    if isinstance(boxes, torch.Tensor):  # Convert tensor to numpy
        boxes = boxes.cpu().numpy()
    
    for box in boxes:
        x1, y1, x2, y2 = box  # Using (x_min, y_min, x_max, y_max)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

# Draw segmentation masks
def draw_masks(image, target):
    masks = target['masks']
    
    if isinstance(masks, torch.Tensor):  # Convert tensor to numpy
        masks = masks.cpu().numpy()
    
    for mask in masks:
        if isinstance(mask, list):  # Polygon segmentation
            for seg in mask:
                points = np.array(seg, dtype=np.int32).reshape((-1, 2))
                cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=1)

# Show image with annotations
def show_image_with_annotations(image, target):
    image_with_bboxes = image.copy()
    draw_bboxes(image_with_bboxes, target)
    draw_masks(image_with_bboxes, target)

    # Show the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image_with_bboxes)
    plt.axis('off')
    plt.show()







# Visualization Functions for Model Predictions
def visualize_prediction(image, prediction, target=None, score_threshold=0.5, image_id=None):
    """
    Visualize model prediction on an image, optionally comparing with ground truth.
    
    Args:
        image: The input image (numpy array, HxWxC)
        prediction: Dictionary with keys 'boxes', 'labels', 'scores', 'masks'
        target: Optional ground truth dictionary with keys 'boxes', 'labels', 'masks'
        score_threshold: Minimum score to display a prediction
        image_id: Optional image ID or filename to display in the title
        
    Returns:
        vis_image: Visualization image with predictions and optionally ground truth
    """
    # Convert image to RGB if it's grayscale
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(vis_image)
    
    # Set title if image_id is provided
    if image_id is not None:
        # Check if image_id looks like a filename (contains dot or common extensions)
        if isinstance(image_id, str) and ('.' in image_id or any(ext in image_id for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff'])):
            ax.set_title(f"File: {image_id}", fontsize=14)
        else:
            ax.set_title(f"Image ID: {image_id}", fontsize=14)
    
    # Define colors for predictions and ground truth
    pred_color = 'red'
    gt_color = 'green'

    # show_image_with_annotations(vis_image, target)
    
    # Draw ground truth if provided
    if target is not None and len(target['boxes']) > 0:
        boxes = target['boxes'].cpu().numpy()
        masks = target['masks'].cpu().numpy() if 'masks' in target else None
        
        for i, box in enumerate(boxes):
            # Draw bounding box
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=gt_color, facecolor='none')
            ax.add_patch(rect)
            
            # Draw mask if available
            if masks is not None:
                mask = masks[i].squeeze()
                
                # Process mask if it has valid dimensions
                if mask.shape[0] > 0 and mask.shape[1] > 0:
                    # Ensure mask is binary and convert to uint8 for OpenCV
                    binary_mask = (mask > 0.5).astype(np.uint8) * 255
                    
                    # Find contours using OpenCV
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw each contour as a polygon
                    for contour in contours:
                        # Convert OpenCV contour format to a simple array of points
                        # OpenCV contours are in format [[[x1,y1]], [[x2,y2]], ...] so we need to reshape
                        polygon = contour.reshape(-1, 2)
                        
                        # Create a polygon patch
                        poly_patch = patches.Polygon(
                            polygon, 
                            closed=True, 
                            fill=True, 
                            facecolor=gt_color,
                            alpha=0.3,
                            edgecolor=gt_color,
                            linewidth=1
                        )
                        ax.add_patch(poly_patch)
    
    # Draw predictions
    if len(prediction['boxes']) > 0:
        boxes = prediction['boxes'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        masks = prediction['masks'].cpu().numpy() if 'masks' in prediction else None
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score < score_threshold:
                continue
                
            # Draw bounding box
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=pred_color, facecolor='none')
            ax.add_patch(rect)
            
            # Add score text
            ax.text(x1, y1, f'{score:.2f}', bbox=dict(facecolor=pred_color, alpha=0.5))
            
            # Draw mask if available
            if masks is not None:
                mask_small = masks[i].squeeze()

                # resize mask to the size of the bounding box
                mask = np.zeros((512, 512))
                mask_resized = cv2.resize(mask_small, (int(x2)-int(x1), int(y2)-int(y1)),interpolation=cv2.INTER_LINEAR)
                mask[int(y1):int(y2), int(x1):int(x2)] = mask_resized
                
                # Process mask if it has valid dimensions
                if mask.shape[0] > 0 and mask.shape[1] > 0:
                    # Ensure mask is binary and convert to uint8 for OpenCV
                    binary_mask = (mask > 0.5).astype(np.uint8) * 255
                    
                    # Find contours using OpenCV
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Draw each contour as a polygon
                    for contour in contours:
                        # Convert OpenCV contour format to a simple array of points
                        polygon = contour.reshape(-1, 2)
                        
                        # Create a polygon patch
                        poly_patch = patches.Polygon(
                            polygon, 
                            closed=True, 
                            fill=True, 
                            facecolor=pred_color,
                            alpha=0.3,
                            edgecolor=pred_color,
                            linewidth=1
                        )
                        ax.add_patch(poly_patch)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Convert figure to numpy array
    fig.canvas.draw()
    vis_image = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    return vis_image

def visualize_batch(images, predictions, targets=None, score_threshold=0.5, max_images=4, image_ids=None):
    """
    Visualize predictions for a batch of images.
    
    Args:
        images: List of images
        predictions: List of prediction dictionaries
        targets: Optional list of target dictionaries
        score_threshold: Minimum score to display a prediction
        max_images: Maximum number of images to visualize
        image_ids: Optional list of image IDs or filenames
        
    Returns:
        List of visualization images
    """
    vis_images = []
    
    # Limit the number of images to visualize
    num_images = min(len(images), max_images)
    
    for i in range(num_images):
        # Get image
        image = images[i].cpu().numpy()
        
        # Convert from CxHxW to HxWxC
        image = np.transpose(image, (1, 2, 0))
        
        # Denormalize if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Get prediction and target
        prediction = predictions[i]
        target = targets[i] if targets is not None else None
        
        # Get image ID or filename
        display_id = None
        if image_ids is not None:
            display_id = image_ids[i]
        elif target is not None and 'file_name' in target:
            display_id = target['file_name']
        elif target is not None and 'image_id' in target:
            display_id = target['image_id']
        else:
            display_id = f"Image {i}"
        
        # Visualize
        vis_image = visualize_prediction(image, prediction, target, score_threshold, display_id)
        vis_images.append(vis_image)
    
    return vis_images

def save_visualization(vis_image, filepath):
    """
    Save visualization image to file.
    
    Args:
        vis_image: Visualization image
        filepath: Path to save the image
    """
    # Convert RGBA to RGB
    if vis_image.shape[2] == 4:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGBA2RGB)
    
    # Save image
    cv2.imwrite(filepath, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

def visualize_and_save_batch(images, predictions, targets=None, output_dir=None, 
                            prefix='pred', score_threshold=0.5, max_images=4, image_ids=None):
    """
    Visualize and save predictions for a batch of images.
    
    Args:
        images: List of images
        predictions: List of prediction dictionaries
        targets: Optional list of target dictionaries
        output_dir: Directory to save visualizations
        prefix: Prefix for saved files
        score_threshold: Minimum score to display a prediction
        max_images: Maximum number of images to visualize
        image_ids: Optional list of image IDs or filenames
        
    Returns:
        List of visualization images
    """
    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Visualize batch
    vis_images = visualize_batch(images, predictions, targets, score_threshold, max_images, image_ids)
    
    # Save visualizations
    if output_dir is not None:
        for i, vis_image in enumerate(vis_images):
            # Use image ID in filename if available
            img_id = "unknown"
            if image_ids is not None and i < len(image_ids):
                img_id = image_ids[i]
            
            # Clean up img_id for use in filename
            if isinstance(img_id, str):
                # If it looks like a filename, remove the extension
                if '.' in img_id:
                    img_id = os.path.splitext(img_id)[0]
                
                # Replace any characters that might cause issues in filenames
                img_id = img_id.replace('/', '_').replace('\\', '_').replace(':', '_')
            
            filepath = os.path.join(output_dir, f'{prefix}_{img_id}.png')
            save_visualization(vis_image, filepath)
    
    return vis_images 