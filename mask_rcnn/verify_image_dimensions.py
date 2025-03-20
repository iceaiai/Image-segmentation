import os
import cv2
import json
import numpy as np

def load_coco_annotations(annotations_path):
    """Load COCO annotations from file."""
    with open(annotations_path, 'r') as f:
        return json.load(f)

def verify_image_dimensions(coco_data, image_dir):
    """
    Verify that the width and height in image_info matches the actual image dimensions.
    
    Args:
        coco_data: COCO annotation data
        image_dir: Directory containing the images
        
    Returns:
        List of mismatches (image_id, filename, annotation dims, actual dims)
    """
    mismatches = []
    matched = 0
    errors = 0
    
    print(f"Verifying dimensions for {len(coco_data['images'])} images...")
    
    for img_info in coco_data['images']:
        image_id = img_info['id']
        filename = img_info['file_name']
        anno_width = img_info['width']
        anno_height = img_info['height']
        
        # Load image
        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            errors += 1
            continue
        
        try:
            # Read image and get actual dimensions
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Failed to load image: {image_path}")
                errors += 1
                continue
                
            actual_height, actual_width = image.shape[:2]
            
            # Check if dimensions match
            if anno_width != actual_width or anno_height != actual_height:
                mismatch = {
                    'image_id': image_id,
                    'filename': filename,
                    'annotation_dims': (anno_width, anno_height),
                    'actual_dims': (actual_width, actual_height)
                }
                mismatches.append(mismatch)
                print(f"Mismatch: {filename} (ID: {image_id})")
                print(f"  - Annotation: {anno_width}x{anno_height}")
                print(f"  - Actual: {actual_width}x{actual_height}")
            else:
                matched += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            errors += 1
    
    # Print summary
    print("\nSummary:")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Matches: {matched}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Errors: {errors}")
    
    return mismatches

def main():
    # Path to COCO annotations and images
    config_dir = os.path.dirname(os.path.abspath(__file__))
    coco_path = os.path.join(config_dir, 'datasets', 'electron-microscopy-particle-segmentation', 'coco_annotations_fixed.json')
    image_dir = os.path.join(config_dir, 'datasets', 'electron-microscopy-particle-segmentation', 'images')
    
    # Load annotations
    coco_data = load_coco_annotations(coco_path)
    
    # Verify image dimensions
    mismatches = verify_image_dimensions(coco_data, image_dir)
    
    # Save mismatches to file if any
    if mismatches:
        mismatch_file = "dimension_mismatches.json"
        with open(mismatch_file, 'w') as f:
            json.dump(mismatches, f, indent=2)
        print(f"Saved mismatches to {mismatch_file}")

if __name__ == "__main__":
    main() 