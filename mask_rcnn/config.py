"""
Configuration for Mask R-CNN for electron microscopy particle segmentation.
"""

import os
import json
import torch

# Base Configuration Class
class Config:
    """Base configuration class for Mask R-CNN."""
    
    # NAME
    NAME = "emps"  # Electron Microscopy Particle Segmentation
    
    # PATHS
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(ROOT_DIR, "datasets", "electron-microscopy-particle-segmentation")
    COCO_ANNOTATIONS_PATH = os.path.join(DATASET_DIR, "coco_annotations_fixed.json")
    
    # Model Checkpoint and Logs
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    CHECKPOINT_PATH = None  # Path to pre-trained weights (if any)
    
    # Dataset splits (to be populated by train.py)
    TRAIN_IMAGES = []
    VAL_IMAGES = []
    TEST_IMAGES = []
    
    # Dataset split ratios
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    
    # Input image size
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_CHANNEL_COUNT = 1  # Grayscale
    
    # Image resizing parameters
    MAINTAIN_ASPECT_RATIO = False  # Whether to maintain aspect ratio
    
    # Backbone network architecture
    BACKBONE = "resnet50"  # Options: "resnet50", "resnet101"
    
    # RPN and Detection parameters
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_NMS_THRESHOLD = 0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    
    # ROI parameters
    ROI_POSITIVE_RATIO = 0.33
    TRAIN_ROIS_PER_IMAGE = 200
    
    # Mask parameters
    MASK_SHAPE = [28, 28]
    
    # Training parameters
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1.5e-4
    BATCH_SIZE = 16
    EPOCHS = 20
    
    # Data augmentation parameters
    AUGMENTATION = {
        "horizontal_flip": False,
        "vertical_flip": False,
        "rotation_range": 90,
        "brightness_range": [0.8, 1.2],
        "contrast_range": [0.8, 1.2],
        "shear_range": 20
    }
    
    # Visualization parameters
    VISUALIZE_VAL_IMAGES = 0  # Number of validation images to visualize (0 to disable)
    VISUALIZATION_DIR = os.path.join(ROOT_DIR, "outputs", "visualizations")
    
    # Enable/disable data augmentation
    USE_AUGMENTATION = False
    
    # Device configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self):
        """Initialize the configuration."""
        # Load COCO annotations to get class information
        if os.path.exists(self.COCO_ANNOTATIONS_PATH):
            try:
                with open(self.COCO_ANNOTATIONS_PATH, 'r') as f:
                    coco_data = json.load(f)
                # Extract categories from COCO annotations
                self.CATEGORIES = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
                self.NUM_CLASSES = len(self.CATEGORIES) + 1  # +1 for background
            except Exception as e:
                print(f"Error loading COCO annotations: {e}")
                self.CATEGORIES = {}
                self.NUM_CLASSES = 1 + 1  # Default: 1 class + background
        else:
            print(f"COCO annotations file not found at {self.COCO_ANNOTATIONS_PATH}")
            self.CATEGORIES = {}
            self.NUM_CLASSES = 1 + 1  # Default: 1 class + background
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for key, val in sorted(self.__dict__.items()):
            if not key.startswith("__") and not callable(val):
                print(f"{key:30} {val}")
        print("\n")


# Configuration for training
class TrainingConfig(Config):
    """Configuration for training on the electron microscopy dataset."""
    # You can override base configuration here
    pass


# Configuration for inference
class InferenceConfig(Config):
    """Configuration for inference on the electron microscopy dataset."""
    # Override base configuration for inference
    BATCH_SIZE = 1
    # Don't resize images for inference
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals
    RPN_NMS_THRESHOLD = 0.7 