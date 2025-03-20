"""
Unit tests for evaluation metrics in utils.py.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

class TestEvaluationMetrics(unittest.TestCase):
    """Test cases for evaluation metrics in utils.py."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample boxes
        self.pred_boxes = torch.tensor([
            [10, 10, 50, 50],  # Box 1: perfect match with target 1
            [100, 100, 150, 150],  # Box 2: perfect match with target 2
            [200, 200, 250, 250],  # Box 3: no match
        ], dtype=torch.float32)
        
        self.target_boxes = torch.tensor([
            [10, 10, 50, 50],  # Target 1: perfect match with box 1
            [100, 100, 150, 150],  # Target 2: perfect match with box 2
            [300, 300, 350, 350],  # Target 3: no match
        ], dtype=torch.float32)
        
        # Create sample masks
        self.pred_mask = np.zeros((100, 100))
        self.pred_mask[20:60, 20:60] = 1  # 40x40 square
        
        self.target_mask = np.zeros((100, 100))
        self.target_mask[30:70, 30:70] = 1  # 40x40 square, overlapping with pred_mask
        
        # Create sample predictions and targets
        self.predictions = [
            {
                'boxes': self.pred_boxes,
                'labels': torch.tensor([1, 1, 1]),
                'scores': torch.tensor([0.9, 0.8, 0.7]),
                'masks': torch.tensor([
                    self.pred_mask[None, :, :],
                    self.pred_mask[None, :, :],
                    self.pred_mask[None, :, :]
                ])
            }
        ]
        
        self.targets = [
            {
                'boxes': self.target_boxes,
                'labels': torch.tensor([1, 1, 1]),
                'masks': torch.tensor([
                    self.target_mask[None, :, :],
                    self.target_mask[None, :, :],
                    self.target_mask[None, :, :]
                ])
            }
        ]
    
    def test_compute_box_iou_matrix(self):
        """Test compute_box_iou_matrix function."""
        iou_matrix = utils.compute_box_iou_matrix(self.pred_boxes, self.target_boxes)
        
        # Check shape
        self.assertEqual(iou_matrix.shape, (3, 3))
        
        # Check values
        self.assertAlmostEqual(iou_matrix[0, 0].item(), 1.0, places=5)  # Perfect match
        self.assertAlmostEqual(iou_matrix[1, 1].item(), 1.0, places=5)  # Perfect match
        self.assertAlmostEqual(iou_matrix[2, 2].item(), 0.0, places=5)  # No overlap
    
    def test_compute_mask_iou(self):
        """Test compute_mask_iou function."""
        # Test with numpy arrays
        iou = utils.compute_mask_iou(self.pred_mask, self.target_mask)
        
        # Expected IoU: intersection area / union area
        # Intersection: 30x30 square from (30,30) to (60,60) = 900
        # Union: two 40x40 squares minus intersection = 2*1600 - 900 = 2300
        expected_iou = 900 / 2300
        self.assertAlmostEqual(iou, expected_iou, places=5)
        
        # Test with tensors
        pred_mask_tensor = torch.tensor(self.pred_mask)
        target_mask_tensor = torch.tensor(self.target_mask)
        iou = utils.compute_mask_iou(pred_mask_tensor, target_mask_tensor)
        self.assertAlmostEqual(iou, expected_iou, places=5)
        
        # Test with different sizes
        small_pred_mask = np.zeros((28, 28))
        small_pred_mask[5:15, 5:15] = 1
        iou = utils.compute_mask_iou(small_pred_mask, self.target_mask)
        self.assertTrue(0 <= iou <= 1)  # IoU should be between 0 and 1
    
    def test_match_predictions_to_targets(self):
        """Test match_predictions_to_targets function."""
        matches = utils.match_predictions_to_targets(self.predictions, self.targets)
        
        # Should have 2 matches (box 1 with target 1, box 2 with target 2)
        self.assertEqual(len(matches), 2)
        
        # Check match format
        for match in matches:
            self.assertEqual(len(match), 4)  # (img_idx, pred_idx, target_idx, iou)
            img_idx, pred_idx, target_idx, iou = match
            self.assertEqual(img_idx, 0)  # Only one image in batch
            self.assertTrue(0 <= pred_idx < 3)  # Valid pred index
            self.assertTrue(0 <= target_idx < 3)  # Valid target index
            self.assertAlmostEqual(iou, 1.0, places=5)  # Perfect matches
    
    def test_compute_detection_metrics(self):
        """Test compute_detection_metrics function."""
        metrics = utils.compute_detection_metrics(self.predictions, self.targets)
        
        # Check metrics
        self.assertEqual(metrics['true_positives'], 2)  # 2 matches
        self.assertEqual(metrics['false_positives'], 1)  # 1 unmatched prediction
        self.assertEqual(metrics['false_negatives'], 1)  # 1 unmatched target
        
        # Check precision, recall, F1
        self.assertAlmostEqual(metrics['precision'], 2/3, places=5)  # TP / (TP + FP)
        self.assertAlmostEqual(metrics['recall'], 2/3, places=5)  # TP / (TP + FN)
        self.assertAlmostEqual(metrics['f1'], 2/3, places=5)  # 2 * (P * R) / (P + R)
    
    def test_compute_segmentation_metrics(self):
        """Test compute_segmentation_metrics function."""
        metrics = utils.compute_segmentation_metrics(self.predictions, self.targets)
        
        # Check metrics
        self.assertTrue('mask_iou' in metrics)
        self.assertTrue('num_matches' in metrics)
        self.assertEqual(metrics['num_matches'], 2)  # 2 matches
        self.assertTrue(0 <= metrics['mask_iou'] <= 1)  # IoU should be between 0 and 1
    
    def test_evaluate_batch(self):
        """Test evaluate_batch function."""
        metrics = utils.evaluate_batch(self.predictions, self.targets)
        
        # Check that all metrics are present
        expected_keys = ['precision', 'recall', 'f1', 'true_positives', 
                         'false_positives', 'false_negatives', 'mask_iou', 'num_matches']
        for key in expected_keys:
            self.assertTrue(key in metrics)
        
        # Check combined metrics
        self.assertEqual(metrics['true_positives'], 2)
        self.assertEqual(metrics['num_matches'], 2)
        self.assertTrue(0 <= metrics['mask_iou'] <= 1)

if __name__ == '__main__':
    unittest.main() 