"""
Dataset class for electron microscopy particle segmentation.
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pycocotools import mask as maskUtils
import utils


import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from PIL import Image
import torch  # Import torch to check tensor type



class EMPSDataset(Dataset):
    """Dataset class for Electron Microscopy Particle Segmentation."""
    
    def __init__(self, config, coco_data, image_ids, transforms=None):
        """
        Initialize the dataset.
        
        Args:
            config: Configuration object.
            coco_data: COCO annotations data.
            image_ids: List of image IDs to include in this dataset.
            transforms: Optional transforms to apply to images and targets.
        """
        self.config = config
        self.image_ids = image_ids
        self.coco_data = coco_data
        self.transforms = transforms
        
        # Create image ID to image info mapping
        self.image_info = {}
        for img in coco_data['images']:
            if img['id'] in image_ids:
                self.image_info[img['id']] = img
        
        # Create image ID to annotations mapping
        self.annotations = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id in image_ids:
                if image_id not in self.annotations:
                    self.annotations[image_id] = []
                self.annotations[image_id].append(ann)
    
    def __len__(self):
        """Get dataset length."""
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Get dataset item.
        
        Args:
            idx: Index.
            
        Returns:
            Tuple of (image, target) where target is a dictionary containing:
            - boxes: Bounding boxes in [x1, y1, x2, y2] format
            - labels: Class labels
            - masks: Instance masks
            - image_id: Image ID
            - file_name: Image filename
        """
        # Get image ID
        image_id = self.image_ids[idx]
        
        # Load image
        image_info = self.image_info[image_id]
        image_path = os.path.join(self.config.DATASET_DIR, 'images', image_info['file_name'])
        image_1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # if image is None:
        #     raise ValueError(f"Failed to load image: {image_path}")
        
        # # Resize image
        # image, scale = utils.resize_image(
        #     image_1, 
        #     min_dim=self.config.IMAGE_MIN_DIM, 
        #     max_dim=self.config.IMAGE_MAX_DIM,
        #     maintain_aspect_ratio=self.config.MAINTAIN_ASPECT_RATIO
        # )
        image = cv2.resize(image_1, (512, 512), interpolation=cv2.INTER_AREA)
        # print(image_1.shape)
        # image = image_1
        # resized_image = cv2.resize(image, (max_dim, min_dim), interpolation=cv2.INTER_AREA)

        # Initialize target
        target = {
            'boxes': [],
            'labels': [],
            'masks': [],
            'image_id': torch.tensor(image_id),
            'file_name': image_info['file_name']
        }
        
        # plot image with the resized annotations
        image_with_annotations = image.copy()
        
        # Add annotations to target
        if image_id in self.annotations:
            for ann in self.annotations[image_id]:
                # Convert COCO polygon to mask
                segmentation = ann['segmentation']
                # put the segmentation polygon in a zero matrix where size is the same as the image
                mask = np.zeros((image_1.shape[0], image_1.shape[1]), dtype=np.uint8)
                for polygon in segmentation:
                    points = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(mask, [points], 1)

                # print(mask.shape)
                mask = cv2.resize(mask, (512,512), interpolation=cv2.INTER_AREA)
                # print(mask.shape)
                # fig, ax = plt.subplots()
                # ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
                # ax.imshow(mask, cmap='jet', alpha=0.5)  # Overlay mask with transparency
                # ax.axis("off")
                # plt.show()

                binary_mask = (mask[:, :, 0] if mask.ndim == 3 else mask) > 0


                # points = np.array(segmentation, dtype=np.int32).reshape((-1, 2))
                # cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=1)

                # # show image with the annotations
                # plt.figure(figsize=(10, 10))
                # plt.imshow(image)
                # plt.show()

                # rle = maskUtils.frPyObjects(segmentation, image_info['height'], image_info['width'])
                # m = maskUtils.decode(rle)
                # m_pycoco_binary = (m[:, :, 0] if m.ndim == 3 else m) > 0

                # m_opencv = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

                # for polygon in segmentation:
                #     if len(polygon) >= 6:  # At least 3 points
                #         points = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                #         cv2.fillPoly(m_opencv, [points], 1)
                # m_opencv_binary = m_opencv > 0

                # # plot image with the resized annotations
                # # image_with_annotations = image.copy()
                # # draw_masks(image_with_annotations, 
                # m_opencv_binary = cv2.resize(m_opencv, (512,512), interpolation=cv2.INTER_AREA)
                # m_opencv_binary = m_opencv_binary > 0
                # assert image.shape[:2] == m_opencv_binary.shape[:2], "Image and mask dimensions must match"

                # fig, ax = plt.subplots()
                # ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
                # ax.imshow(m_opencv_binary, cmap='jet', alpha=0.5)  # Overlay mask with transparency
                # ax.axis("off")
                # plt.show()


                # # are_equal = np.array_equal(m_pycoco_binary, m_opencv_binary)
                # # diff_percentage = np.sum(np.logical_xor(m_pycoco_binary, m_opencv_binary)) / (image_info['height'] * image_info['width']) * 100
                # # print(f"Difference percentage: {diff_percentage}%")

                # # Resize mask to match image
                # if self.config.MAINTAIN_ASPECT_RATIO:
                #     raise NotImplementedError("Resizing masks while maintaining aspect ratio is not implemented.")
                # else:
                #     if m.shape[:2] != image.shape[:2]:
                #         m = cv2.resize(m_opencv, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

                # # Get binary mask
                # binary_mask = (m[:, :, 0] if m.ndim == 3 else m) > 0

                # Skip if mask is empty
                if not np.any(binary_mask):
                    continue

                # Get bounding box from mask
                pos = np.where(binary_mask)
                if len(pos[0]) > 0:  # If mask is not empty
                    xmin = np.min(pos[1])
                    ymin = np.min(pos[0])
                    xmax = np.max(pos[1])
                    ymax = np.max(pos[0])
                    
                    # Skip if box is too small
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    
                    # Add box to target
                    target['boxes'].append([xmin, ymin, xmax, ymax])
                    
                    # Add label to target (category_id or 1 if not specified)
                    category_id = ann.get('category_id', 1)
                    target['labels'].append(category_id)
                    
                    # Add mask to target
                    target['masks'].append(binary_mask)
        
        # Convert lists to tensors
        if len(target['boxes']) > 0:
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
            target['masks'] = torch.tensor(np.array(target['masks']), dtype=torch.uint8)
        else:
            # Empty target
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['masks'] = torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.uint8)
        
        # Apply transforms if specified
        if self.transforms is not None:
            # Keep image as numpy array for transforms
            image_for_transform = image.copy()
            image, target = self.transforms(image_for_transform, target)
            
            # Validate boxes after transforms
            if len(target['boxes']) > 0:
                # Ensure boxes have positive width and height
                boxes = target['boxes']
                
                # Calculate width and height
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                
                # Find valid boxes (positive width and height)
                valid_boxes = (widths > 0) & (heights > 0)
                
                if valid_boxes.sum() > 0:
                    # Keep only valid boxes
                    target['boxes'] = boxes[valid_boxes]
                    target['labels'] = target['labels'][valid_boxes]
                    target['masks'] = target['masks'][valid_boxes]
                else:
                    # No valid boxes, create empty target
                    target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    target['labels'] = torch.zeros((0,), dtype=torch.int64)
                    target['masks'] = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
        else:
            # Convert image to tensor (add channel dimension for grayscale)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        
        return image, target
    
    def get_image_info(self, image_id):
        """
        Get image info for a specific image ID.
        
        Args:
            image_id: Image ID.
            
        Returns:
            Image info dictionary.
        """
        return self.image_info.get(image_id, None)
    
    def get_annotations(self, image_id):
        """
        Get annotations for a specific image ID.
        
        Args:
            image_id: Image ID.
            
        Returns:
            List of annotations.
        """
        return self.annotations.get(image_id, []) 