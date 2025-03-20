"""
Script to display a specific image with its bounding boxes and masks.
This is useful for debugging annotation issues.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.path import Path
import json
import sys
import argparse

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config

def display_image_with_annotations(image_id):
    """
    Display an image with its bounding boxes and masks.
    
    Args:
        image_id: ID of the image to display
    """
    # Load configuration
    config = Config()
    
    # Load COCO annotations
    print(f"Loading COCO annotations from {config.COCO_ANNOTATIONS_PATH}")
    with open(config.COCO_ANNOTATIONS_PATH, 'r') as f:
        coco_data = json.load(f)
    
    # Find image info
    image_info = None
    for img in coco_data['images']:
        if img['id'] == image_id:
            image_info = img
            break
    
    if image_info is None:
        print(f"Image ID {image_id} not found in the dataset")
        return
    
    # Load image
    image_path = os.path.join(config.DATASET_DIR, 'images', image_info['file_name'])
    print(f"Loading image from {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Convert grayscale to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Find annotations for this image
    annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            annotations.append(ann)
    
    print(f"Found {len(annotations)} annotations for image ID {image_id}")
    
    # Generate colors for annotations
    # Use standard matplotlib colors
    colors = list(mcolors.TABLEAU_COLORS.values())
    # If we need more colors, add some
    if len(annotations) > len(colors):
        colors.extend(['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'magenta'] * 10)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # First subplot: Image with bounding boxes and masks
    ax1.imshow(image_rgb)
    ax1.set_title(f"Image ID: {image_id}, Filename: {image_info['file_name']}\nBounding Boxes & Masks")
    
    # Second subplot: Image with masks only
    ax2.imshow(image_rgb)
    ax2.set_title(f"Image ID: {image_id}, Filename: {image_info['file_name']}\nMasks Only")
    
    # Draw annotations
    for i, ann in enumerate(annotations):
        # Get color for this annotation (cycle through colors if needed)
        color = colors[i % len(colors)]
        
        # Draw bounding box on first subplot
        x, y, w, h = ann['bbox']
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor='none')
        ax1.add_patch(rect)
        
        # Draw category label on first subplot
        category_id = ann['category_id']
        category_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id), 'unknown')
        ax1.text(x, y, f"{category_name}", fontsize=8, bbox=dict(facecolor=color, alpha=0.5))
        
        # Draw mask on both subplots
        if 'segmentation' in ann:
            # Get segmentation polygons
            segmentation = ann['segmentation']
            
            # Draw each polygon
            for polygon in segmentation:
                # Convert flat list to points
                points = np.array(polygon).reshape(-1, 2)
                
                # Create a polygon patch for the first subplot (with bboxes)
                poly_patch1 = patches.Polygon(points, closed=True, fill=True, 
                                             facecolor=color, alpha=0.3, edgecolor=color, linewidth=1)
                ax1.add_patch(poly_patch1)
                
                # Create a polygon patch for the second subplot (masks only)
                poly_patch2 = patches.Polygon(points, closed=True, fill=True, 
                                             facecolor=color, alpha=0.5, edgecolor=color, linewidth=1)
                ax2.add_patch(poly_patch2)
    
    # Remove axis ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Save figure
    output_dir = os.path.join(config.ROOT_DIR, "outputs", "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"image_{image_id}.png")
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
    
    # Show figure
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Get image ID from command line argument, default to 157
    parser = argparse.ArgumentParser(description='Display an image with its annotations')
    parser.add_argument('--image_id', type=int, default=157, help='ID of the image to display')
    args = parser.parse_args()
    
    # Display image with the specified ID
    display_image_with_annotations(args.image_id) 