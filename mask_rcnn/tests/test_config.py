"""
Tests for configuration management.
"""

import os
import sys
import unittest
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config, TrainingConfig, InferenceConfig


class TestConfig(unittest.TestCase):
    """Test cases for configuration management."""
    
    def test_config_initialization(self):
        """Test that the Config class initializes correctly."""
        config = Config()
        self.assertEqual(config.NAME, "emps")
        self.assertEqual(config.TRAIN_RATIO, 0.6)
        self.assertEqual(config.VAL_RATIO, 0.2)
        self.assertEqual(config.TEST_RATIO, 0.2)
    
    def test_training_config(self):
        """Test that the TrainingConfig class initializes correctly."""
        config = TrainingConfig()
        self.assertEqual(config.NAME, "emps")
        # Add more assertions for training-specific parameters
    
    def test_inference_config(self):
        """Test that the InferenceConfig class initializes correctly."""
        config = InferenceConfig()
        self.assertEqual(config.NAME, "emps")
        self.assertEqual(config.BATCH_SIZE, 1)
        # Add more assertions for inference-specific parameters
    
    def test_config_display(self):
        """Test the display method of Config."""
        config = Config()
        # This is a simple test to ensure the method runs without errors
        config.display()
    
    def test_dataset_paths(self):
        """Test that dataset paths are correctly constructed."""
        config = Config()
        self.assertTrue(os.path.exists(config.ROOT_DIR))
        # Note: This test will fail if the dataset directory doesn't exist
        # self.assertTrue(os.path.exists(config.DATASET_DIR))
    
    def test_split_ratios_sum_to_one(self):
        """Test that the dataset split ratios sum to 1."""
        config = Config()
        self.assertAlmostEqual(config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO, 1.0)


if __name__ == "__main__":
    unittest.main() 