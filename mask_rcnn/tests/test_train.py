"""
Tests for training pipeline.
"""

import os
import sys
import unittest
import torch
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TrainingConfig
import train


class TestTrain(unittest.TestCase):
    """Test cases for training pipeline."""
    
    def test_get_model_instance_segmentation(self):
        """Test the model creation function."""
        config = TrainingConfig()
        model = train.get_model_instance_segmentation(config)
        
        # Check that the model is a PyTorch model
        self.assertIsInstance(model, torch.nn.Module)
        
        # Check that the model has the expected number of classes
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, config.NUM_CLASSES)
        self.assertEqual(model.roi_heads.mask_predictor.mask_fcn_logits.out_channels, config.NUM_CLASSES)
    
    def test_combined_loss(self):
        """Test the combined loss function."""
        loss_fn = train.CombinedLoss()
        
        # Create dummy outputs and targets
        batch_size = 2
        height, width = 10, 10
        outputs = torch.zeros(batch_size, 1, height, width)
        targets = torch.zeros(batch_size, 1, height, width)
        
        # Set some values to create a simple test case
        outputs[0, 0, 2:5, 2:5] = 10.0  # High confidence for positive
        targets[0, 0, 3:6, 3:6] = 1.0   # Ground truth
        
        # Compute loss
        loss = loss_fn(outputs, targets)
        
        # Check that loss is a scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        
        # Check that loss is positive
        self.assertGreater(loss.item(), 0)


if __name__ == "__main__":
    unittest.main() 