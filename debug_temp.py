"""
Debug script to investigate COCO annotations and category issues.
"""

import os
import json
import numpy as np
from collections import Counter

# Path to COCO annotations file
ORIGINAL_COCO_PATH = "mask_rcnn/datasets/electron-microscopy-particle-segmentation/coco_annotations.json"
FIXED_COCO_PATH = "mask_rcnn/datasets/electron-microscopy-particle-segmentation/coco_annotations_fixed.json"

def load_coco_annotations(annotations_path):
    """Load COCO format annotations."""
    print(f"Loading COCO annotations from {annotations_path}")
    try:
        with open(annotations_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {annotations_path}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON in file: {annotations_path}")
        return None

def analyze_coco_data(coco_data, title="COCO Data Analysis"):
    """Analyze COCO data structure and content."""
    if coco_data is None:
        print("No COCO data to analyze")
        return
    
    print(f"\n=== {title} ===")
    print("COCO Data Structure:")
    for key in coco_data.keys():
        if isinstance(coco_data[key], list):
            print(f"  {key}: {len(coco_data[key])} items")
        else:
            print(f"  {key}: {coco_data[key]}")
    
    # Analyze categories
    if 'categories' in coco_data:
        categories = coco_data['categories']
        print(f"\nCategories ({len(categories)}):")
        if len(categories) < 20:  # Only print if not too many
            for cat in categories:
                print(f"  ID: {cat['id']}, Name: {cat.get('name', 'N/A')}, Supercategory: {cat.get('supercategory', 'N/A')}")
        else:
            print(f"  Too many categories to display. First 5:")
            for cat in categories[:5]:
                print(f"  ID: {cat['id']}, Name: {cat.get('name', 'N/A')}, Supercategory: {cat.get('supercategory', 'N/A')}")
            print(f"  ... and {len(categories) - 5} more")
            
            # Check for unusual category IDs
            cat_ids = [cat['id'] for cat in categories]
            min_id = min(cat_ids)
            max_id = max(cat_ids)
            print(f"  Category ID range: {min_id} to {max_id}")
            
            # Check if IDs are sequential
            if max_id - min_id + 1 != len(cat_ids):
                print("  Warning: Category IDs are not sequential!")
    
    # Analyze annotations
    if 'annotations' in coco_data:
        annotations = coco_data['annotations']
        print(f"\nAnnotations ({len(annotations)}):")
        
        # Check category IDs in annotations
        cat_ids_in_annotations = [ann.get('category_id', None) for ann in annotations]
        cat_ids_in_annotations = [cat_id for cat_id in cat_ids_in_annotations if cat_id is not None]
        
        if cat_ids_in_annotations:
            cat_counter = Counter(cat_ids_in_annotations)
            print(f"  Number of unique category IDs in annotations: {len(cat_counter)}")
            print(f"  Most common category IDs: {cat_counter.most_common(5)}")
            
            # Check for category IDs not in the categories list
            if 'categories' in coco_data:
                valid_cat_ids = {cat['id'] for cat in coco_data['categories']}
                invalid_cat_ids = set(cat_ids_in_annotations) - valid_cat_ids
                if invalid_cat_ids:
                    print(f"  Warning: Found {len(invalid_cat_ids)} category IDs in annotations that are not in the categories list!")
                    print(f"  Examples: {list(invalid_cat_ids)[:5]}")
        else:
            print("  No category IDs found in annotations!")
        
        # Check for other issues
        print("\nAnnotation Structure:")
        if annotations:
            sample_ann = annotations[0]
            print(f"  Sample annotation keys: {list(sample_ann.keys())}")
            
            # Check segmentation format
            if 'segmentation' in sample_ann:
                seg_type = type(sample_ann['segmentation'])
                print(f"  Segmentation type: {seg_type}")
                if seg_type == list:
                    print(f"  Segmentation format: {type(sample_ann['segmentation'][0])}")
                    if len(sample_ann['segmentation']) > 0 and isinstance(sample_ann['segmentation'][0], list):
                        print(f"  Polygon points: {len(sample_ann['segmentation'][0])} coordinates")
    
    # Analyze images
    if 'images' in coco_data:
        images = coco_data['images']
        print(f"\nImages ({len(images)}):")
        if images:
            sample_img = images[0]
            print(f"  Sample image keys: {list(sample_img.keys())}")
            print(f"  Sample image ID: {sample_img.get('id', 'N/A')}")
            print(f"  Sample image file name: {sample_img.get('file_name', 'N/A')}")
            print(f"  Sample image dimensions: {sample_img.get('width', 'N/A')}x{sample_img.get('height', 'N/A')}")

def fix_coco_data(coco_data):
    """Generate a fixed version of the COCO data with corrected categories."""
    if coco_data is None:
        return None
    
    # Create a new categories list with just one category for particles
    new_categories = [{"id": 1, "name": "particle", "supercategory": "none"}]
    
    # Update all annotations to use category_id 1
    for ann in coco_data['annotations']:
        if 'category_id' in ann:
            ann['category_id'] = 1
    
    # Update the categories in the COCO data
    coco_data['categories'] = new_categories
    
    return coco_data

def check_config_compatibility(coco_data):
    """Check if the COCO data is compatible with the config.py implementation."""
    if coco_data is None:
        return
    
    print("\n=== Config Compatibility Check ===")
    
    # Simulate what config.py does
    if 'categories' in coco_data:
        categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        num_classes = len(categories) + 1  # +1 for background
        
        print(f"Categories dict: {categories}")
        print(f"NUM_CLASSES: {num_classes}")
        
        # Check if this matches expectations
        if num_classes == 2:  # 1 class + background
            print("✓ NUM_CLASSES is correct (2 = 1 class + background)")
        else:
            print(f"✗ NUM_CLASSES is incorrect. Expected 2, got {num_classes}")

def main():
    """Main function to debug COCO annotations."""
    # Check original COCO annotations
    original_coco_data = load_coco_annotations(ORIGINAL_COCO_PATH)
    if original_coco_data:
        analyze_coco_data(original_coco_data, "Original COCO Data")
    
    # Check if fixed file exists and analyze it
    fixed_coco_data = load_coco_annotations(FIXED_COCO_PATH)
    if fixed_coco_data:
        analyze_coco_data(fixed_coco_data, "Fixed COCO Data")
        check_config_compatibility(fixed_coco_data)
    else:
        # Generate fixed data
        print("\n=== Generating Fixed COCO Data ===")
        fixed_coco_data = fix_coco_data(original_coco_data)
        if fixed_coco_data:
            analyze_coco_data(fixed_coco_data, "Generated Fixed COCO Data")
            check_config_compatibility(fixed_coco_data)
    
    # Provide instructions
    print("\n=== Instructions ===")
    if not os.path.exists(FIXED_COCO_PATH):
        print("To fix the issue, run the fix_coco_annotations.py script:")
        print("python fix_coco_annotations.py")
    else:
        print("The fixed COCO annotations file already exists.")
    
    print("\nMake sure config.py is updated to use the fixed annotations file:")
    print("COCO_ANNOTATIONS_PATH = os.path.join(DATASET_DIR, \"coco_annotations_fixed.json\")")

if __name__ == "__main__":
    main() 