"""
Tests for evaluation pipeline.
"""

import os
import sys
import unittest
import torch
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import InferenceConfig
import test as test_module  # Renamed to avoid conflict with Python's test module


class TestEvaluate(unittest.TestCase):
    """Test cases for evaluation pipeline."""
    
    def test_get_model_instance_segmentation(self):
        """Test the model creation function."""
        config = InferenceConfig()
        model = test_module.get_model_instance_segmentation(config)
        
        # Check that the model is a PyTorch model
        self.assertIsInstance(model, torch.nn.Module)
        
        # Check that the model has the expected number of classes
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, config.NUM_CLASSES)
        self.assertEqual(model.roi_heads.mask_predictor.mask_fcn_logits.out_channels, config.NUM_CLASSES)
    
    def test_model_inference_mode(self):
        """Test that the model can be put in inference mode."""
        config = InferenceConfig()
        model = test_module.get_model_instance_segmentation(config)
        
        # Set model to eval mode
        model.eval()
        
        # Create a dummy input
        batch_size = 1
        channels = 3
        height, width = 224, 224
        dummy_input = [torch.zeros(channels, height, width)]
        
        # Run inference (this will fail if the model can't be put in inference mode)
        with torch.no_grad():
            try:
                outputs = model(dummy_input)
                # If we get here, the model can be put in inference mode
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Model inference failed with error: {e}")


if __name__ == "__main__":
    unittest.main() 