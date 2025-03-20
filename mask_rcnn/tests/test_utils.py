"""
Tests for utility functions.
"""

import os
import sys
import unittest
import numpy as np
import torch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_split_dataset(self):
        """Test the dataset splitting function."""
        image_ids = list(range(100))
        train_ids, val_ids, test_ids = utils.split_dataset(image_ids)
        
        # Check that the split sizes are correct
        self.assertEqual(len(train_ids), 60)
        self.assertEqual(len(val_ids), 20)
        self.assertEqual(len(test_ids), 20)
        
        # Check that all IDs are used and no duplicates
        all_ids = train_ids + val_ids + test_ids
        self.assertEqual(len(all_ids), 100)
        self.assertEqual(len(set(all_ids)), 100)
    
    def test_resize_image(self):
        """Test the image resizing function."""
        # Create a test image
        image = np.zeros((100, 200), dtype=np.uint8)
        
        # Test resizing with min_dim
        resized_image, scale = utils.resize_image(image, min_dim=300)
        self.assertEqual(resized_image.shape[0], 300)
        self.assertEqual(resized_image.shape[1], 600)
        
        # Test resizing with max_dim
        resized_image, scale = utils.resize_image(image, max_dim=150)
        self.assertEqual(resized_image.shape[0], 75)
        self.assertEqual(resized_image.shape[1], 150)
    
    def test_compute_iou(self):
        """Test the IoU computation function."""
        # Create test masks
        mask1 = np.zeros((10, 10), dtype=np.bool_)
        mask1[2:8, 2:8] = True
        
        mask2 = np.zeros((10, 10), dtype=np.bool_)
        mask2[4:10, 4:10] = True
        
        # Compute IoU
        iou = utils.compute_iou(mask1, mask2)
        
        # Expected IoU: intersection = 12, union = 72
        expected_iou = 12 / 72
        self.assertAlmostEqual(iou, expected_iou)
    
    def test_compute_dice_coefficient(self):
        """Test the Dice coefficient computation function."""
        # Create test masks
        mask1 = np.zeros((10, 10), dtype=np.bool_)
        mask1[2:8, 2:8] = True
        
        mask2 = np.zeros((10, 10), dtype=np.bool_)
        mask2[4:10, 4:10] = True
        
        # Compute Dice coefficient
        dice = utils.compute_dice_coefficient(mask1, mask2)
        
        # Expected Dice: 2 * intersection / (sum1 + sum2) = 2 * 12 / (36 + 36)
        expected_dice = 2 * 12 / 72
        self.assertAlmostEqual(dice, expected_dice)
    
    def test_compute_precision_recall_f1(self):
        """Test the precision, recall, and F1 computation function."""
        # Create test masks
        gt_mask = np.zeros((10, 10), dtype=np.bool_)
        gt_mask[2:8, 2:8] = True
        
        pred_mask = np.zeros((10, 10), dtype=np.bool_)
        pred_mask[4:10, 4:10] = True
        
        # Compute precision, recall, and F1
        precision, recall, f1 = utils.compute_precision_recall_f1(gt_mask, pred_mask)
        
        # Expected values
        # True positives = 12
        # False positives = 24
        # False negatives = 24
        expected_precision = 12 / 36
        expected_recall = 12 / 36
        expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
        
        self.assertAlmostEqual(precision, expected_precision)
        self.assertAlmostEqual(recall, expected_recall)
        self.assertAlmostEqual(f1, expected_f1)


if __name__ == "__main__":
    unittest.main() 