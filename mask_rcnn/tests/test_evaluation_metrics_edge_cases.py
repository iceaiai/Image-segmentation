"""
Unit tests for edge cases in evaluation metrics in utils.py.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils

class TestEvaluationMetricsEdgeCases(unittest.TestCase):
    """Test cases for edge cases in evaluation metrics in utils.py."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample boxes
        self.pred_boxes = torch.tensor([
            [10, 10, 50, 50],
            [100, 100, 150, 150],
        ], dtype=torch.float32)
        
        self.target_boxes = torch.tensor([
            [10, 10, 50, 50],
            [100, 100, 150, 150],
        ], dtype=torch.float32)
        
        # Create sample masks
        self.pred_mask = np.zeros((100, 100))
        self.pred_mask[20:60, 20:60] = 1
        
        self.target_mask = np.zeros((100, 100))
        self.target_mask[30:70, 30:70] = 1
    
    def test_empty_predictions(self):
        """Test with empty predictions."""
        empty_predictions = [{'boxes': torch.tensor([]), 'labels': torch.tensor([]), 
                             'scores': torch.tensor([]), 'masks': torch.tensor([])}]
        
        targets = [{'boxes': self.target_boxes, 
                   'labels': torch.tensor([1, 1]), 
                   'masks': torch.tensor([self.target_mask[None, :, :], self.target_mask[None, :, :]])}]
        
        # Test match_predictions_to_targets
        matches = utils.match_predictions_to_targets(empty_predictions, targets)
        self.assertEqual(len(matches), 0)
        
        # Test compute_detection_metrics
        metrics = utils.compute_detection_metrics(empty_predictions, targets)
        self.assertEqual(metrics['true_positives'], 0)
        self.assertEqual(metrics['false_positives'], 0)
        self.assertEqual(metrics['false_negatives'], 2)
        self.assertEqual(metrics['precision'], 0)
        self.assertEqual(metrics['recall'], 0)
        self.assertEqual(metrics['f1'], 0)
        
        # Test compute_segmentation_metrics
        metrics = utils.compute_segmentation_metrics(empty_predictions, targets)
        self.assertEqual(metrics['num_matches'], 0)
        self.assertEqual(metrics['mask_iou'], 0)
        
        # Test evaluate_batch
        metrics = utils.evaluate_batch(empty_predictions, targets)
        self.assertEqual(metrics['true_positives'], 0)
        self.assertEqual(metrics['num_matches'], 0)
    
    def test_empty_targets(self):
        """Test with empty targets."""
        predictions = [{'boxes': self.pred_boxes, 
                       'labels': torch.tensor([1, 1]), 
                       'scores': torch.tensor([0.9, 0.8]), 
                       'masks': torch.tensor([self.pred_mask[None, :, :], self.pred_mask[None, :, :]])}]
        
        empty_targets = [{'boxes': torch.tensor([]), 'labels': torch.tensor([]), 
                         'masks': torch.tensor([])}]
        
        # Test match_predictions_to_targets
        matches = utils.match_predictions_to_targets(predictions, empty_targets)
        self.assertEqual(len(matches), 0)
        
        # Test compute_detection_metrics
        metrics = utils.compute_detection_metrics(predictions, empty_targets)
        self.assertEqual(metrics['true_positives'], 0)
        self.assertEqual(metrics['false_positives'], 2)
        self.assertEqual(metrics['false_negatives'], 0)
        self.assertEqual(metrics['precision'], 0)
        self.assertEqual(metrics['recall'], 0)  # 0/0 is defined as 0
        self.assertEqual(metrics['f1'], 0)
        
        # Test compute_segmentation_metrics
        metrics = utils.compute_segmentation_metrics(predictions, empty_targets)
        self.assertEqual(metrics['num_matches'], 0)
        self.assertEqual(metrics['mask_iou'], 0)
        
        # Test evaluate_batch
        metrics = utils.evaluate_batch(predictions, empty_targets)
        self.assertEqual(metrics['true_positives'], 0)
        self.assertEqual(metrics['num_matches'], 0)
    
    def test_no_matches(self):
        """Test with predictions and targets that don't match."""
        predictions = [{'boxes': torch.tensor([[200, 200, 250, 250]]), 
                       'labels': torch.tensor([1]), 
                       'scores': torch.tensor([0.9]), 
                       'masks': torch.tensor([self.pred_mask[None, :, :]])}]
        
        targets = [{'boxes': torch.tensor([[300, 300, 350, 350]]), 
                   'labels': torch.tensor([1]), 
                   'masks': torch.tensor([self.target_mask[None, :, :]])}]
        
        # Test match_predictions_to_targets
        matches = utils.match_predictions_to_targets(predictions, targets)
        self.assertEqual(len(matches), 0)
        
        # Test compute_detection_metrics
        metrics = utils.compute_detection_metrics(predictions, targets)
        self.assertEqual(metrics['true_positives'], 0)
        self.assertEqual(metrics['false_positives'], 1)
        self.assertEqual(metrics['false_negatives'], 1)
        self.assertEqual(metrics['precision'], 0)
        self.assertEqual(metrics['recall'], 0)
        self.assertEqual(metrics['f1'], 0)
        
        # Test compute_segmentation_metrics
        metrics = utils.compute_segmentation_metrics(predictions, targets)
        self.assertEqual(metrics['num_matches'], 0)
        self.assertEqual(metrics['mask_iou'], 0)
        
        # Test evaluate_batch
        metrics = utils.evaluate_batch(predictions, targets)
        self.assertEqual(metrics['true_positives'], 0)
        self.assertEqual(metrics['num_matches'], 0)
    
    def test_multiple_images_in_batch(self):
        """Test with multiple images in a batch."""
        predictions = [
            {'boxes': torch.tensor([[10, 10, 50, 50]]), 
             'labels': torch.tensor([1]), 
             'scores': torch.tensor([0.9]), 
             'masks': torch.tensor([self.pred_mask[None, :, :]])},
            {'boxes': torch.tensor([[100, 100, 150, 150]]), 
             'labels': torch.tensor([1]), 
             'scores': torch.tensor([0.8]), 
             'masks': torch.tensor([self.pred_mask[None, :, :]])}
        ]
        
        targets = [
            {'boxes': torch.tensor([[10, 10, 50, 50]]), 
             'labels': torch.tensor([1]), 
             'masks': torch.tensor([self.target_mask[None, :, :]])},
            {'boxes': torch.tensor([[100, 100, 150, 150]]), 
             'labels': torch.tensor([1]), 
             'masks': torch.tensor([self.target_mask[None, :, :]])}
        ]
        
        # Test match_predictions_to_targets
        matches = utils.match_predictions_to_targets(predictions, targets)
        self.assertEqual(len(matches), 2)
        
        # Check that matches are from different images
        img_indices = [match[0] for match in matches]
        self.assertEqual(set(img_indices), {0, 1})
        
        # Test compute_detection_metrics
        metrics = utils.compute_detection_metrics(predictions, targets)
        self.assertEqual(metrics['true_positives'], 2)
        self.assertEqual(metrics['false_positives'], 0)
        self.assertEqual(metrics['false_negatives'], 0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)
        
        # Test compute_segmentation_metrics
        metrics = utils.compute_segmentation_metrics(predictions, targets)
        self.assertEqual(metrics['num_matches'], 2)
        self.assertTrue(0 < metrics['mask_iou'] < 1)
        
        # Test evaluate_batch
        metrics = utils.evaluate_batch(predictions, targets)
        self.assertEqual(metrics['true_positives'], 2)
        self.assertEqual(metrics['num_matches'], 2)

if __name__ == '__main__':
    unittest.main() 