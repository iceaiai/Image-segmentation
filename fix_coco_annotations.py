"""
Script to fix COCO annotations file by replacing multiple categories with a single particle category.
"""

import os
import json

# Path to original COCO annotations file
ORIGINAL_COCO_PATH = "mask_rcnn/datasets/electron-microscopy-particle-segmentation/coco_annotations.json"
# Path to save the fixed COCO annotations file
FIXED_COCO_PATH = "mask_rcnn/datasets/electron-microscopy-particle-segmentation/coco_annotations_fixed.json"

def load_coco_annotations(annotations_path):
    """Load COCO format annotations."""
    print(f"Loading COCO annotations from {annotations_path}")
    with open(annotations_path, 'r') as f:
        return json.load(f)

def fix_coco_data(coco_data):
    """Fix the COCO data by replacing categories with a single particle category."""
    # Create a new categories list with just one category for particles
    new_categories = [{"id": 1, "name": "particle", "supercategory": "none"}]
    
    # Update all annotations to use category_id 1
    for ann in coco_data['annotations']:
        if 'category_id' in ann:
            ann['category_id'] = 1
    
    # Update the categories in the COCO data
    coco_data['categories'] = new_categories
    
    return coco_data

def save_coco_annotations(coco_data, output_path):
    """Save COCO annotations to a file."""
    print(f"Saving fixed COCO annotations to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(coco_data, f)

def main():
    """Main function to fix COCO annotations."""
    # Check if original file exists
    if not os.path.exists(ORIGINAL_COCO_PATH):
        print(f"Error: Original COCO annotations file not found at {ORIGINAL_COCO_PATH}")
        return
    
    # Load original COCO annotations
    coco_data = load_coco_annotations(ORIGINAL_COCO_PATH)
    
    # Print original categories count
    if 'categories' in coco_data:
        print(f"Original categories count: {len(coco_data['categories'])}")
    
    # Fix COCO data
    fixed_coco_data = fix_coco_data(coco_data)
    
    # Print fixed categories count
    if 'categories' in fixed_coco_data:
        print(f"Fixed categories count: {len(fixed_coco_data['categories'])}")
        for cat in fixed_coco_data['categories']:
            print(f"Category: ID={cat['id']}, Name={cat['name']}")
    
    # Save fixed COCO annotations
    save_coco_annotations(fixed_coco_data, FIXED_COCO_PATH)
    
    print("\nFix completed successfully!")
    print(f"Original file: {ORIGINAL_COCO_PATH}")
    print(f"Fixed file: {FIXED_COCO_PATH}")
    print("\nNow update the config.py file to use the fixed annotations file:")
    print("COCO_ANNOTATIONS_PATH = os.path.join(DATASET_DIR, \"coco_annotations_fixed.json\")")

if __name__ == "__main__":
    main() 