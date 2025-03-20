# Mask RCNN Instance Segmentation Project

> **Note**: This document serves as a design planner to guide implementation.

## Project Overview

This project implements Mask R-CNN for instance segmentation on electron microscopy particle images. The implementation uses the electron-microscopy-particle-segmentation dataset with COCO format annotations.

## Project Structure

```
mask_rcnn/
│── config.py         # Configuration settings
│── utils.py          # Utility functions including visualization
│── train.py          # Training script
│── evaluate.py       # Evaluation script
│── models/           # Directory for model architectures and checkpoints
│   │── checkpoints/  # Saved model checkpoints
│── datasets/         # Directory for dataset management
│   │── emps_dataset.py # Dataset class for electron microscopy particles
│   │── electron-microscopy-particle-segmentation/ # Dataset directory
│       │── images/   # Original images
│       │── segmaps/  # Segmentation maps
│       │── coco_annotations.json # COCO format annotations
│── logs/             # Logging and TensorBoard outputs
│── outputs/          # Directory for saving predictions and models
│   │── visualizations/ # Segmentation visualization results
│   │── predictions/    # Model output files (JSON, etc.)
│── tests/            # Directory for unit tests
│   │── test_config.py  # Tests for configuration management
│   │── test_utils.py   # Tests for utility functions
│   │── test_train.py   # Tests for training pipeline
│   │── test_evaluate.py # Tests for evaluation pipeline
```

## Implementation Steps (Based on Notebook Analysis)

### 1. Setup and Environment
- Required libraries: PyTorch, OpenCV, NumPy, Matplotlib, pycocotools
- GPU acceleration when available

### 2. Dataset Preparation
- Load images from `datasets/electron-microscopy-particle-segmentation/images` and masks from `segmaps`
- Convert RGB to grayscale
- Implement data augmentation:
  - Rotations (0°, 90°, 180°, 270°)
  - Translations (random within ±50 pixels)
  - Shears (+20° and -20°)
- Resize images to consistent dimensions (512×512)
- Split dataset into training (60%), validation (20%), and testing (20%)

### 3. Data Loading
- Create custom dataset class for electron microscopy particles
- Implement data loaders with appropriate batch sizes
- Apply transformations during loading

### 4. Model Architecture
- Mask R-CNN with ResNet50 backbone and Feature Pyramid Network (FPN)
- Configure for single-class segmentation
- Trainable backbone layers for transfer learning

### 5. Training Process
- Combined loss function (BCE + Dice + Focal Loss)
- Adam optimizer with weight decay
- Learning rate scheduling
- Checkpoint saving for best models
- TensorBoard logging for metrics

### 6. Evaluation Metrics
- Precision, Recall, F1 Score
- IoU (Intersection over Union)
- Dice Coefficient
- Pixel Error and Rand Error

---

## **1. config.py**

This file contains all the configurations required for training and testing.

### **Key Features:**

- **Dataset Path:** Path to `datasets/electron-microscopy-particle-segmentation` where COCO annotation resides.
- **Model Parameters:** Hyperparameters such as learning rate, batch size, number of epochs, etc.
- **Data Splitting:**
  - List of image names for training, validation, and testing.
  - The first run of `train.py` generates and saves these splits unless already specified.
  - If splits exist, training script does not overwrite unless explicitly allowed by the user.
- **Augmentation Settings:** Enable/disable data augmentation techniques.
- **Logging Settings:** Set paths for logs, checkpoints, and TensorBoard outputs.
- **Checkpoint Management:** Paths for saving and loading model checkpoints.
- **Model Architecture Settings:**
  - Backbone network (ResNet50/101)
  - RPN anchor scales and ratios
  - ROI parameters
  - Mask shape parameters

---

## **2. utils.py**

Contains commonly used functions shared across training and testing.

### **Key Functionalities:**

- **Dataset Loader:**
  - Load images and annotations from COCO format.
  - Convert them into a format suitable for Mask R-CNN.
- **Data Augmentation:**
  - Implement rotations, translations, shears, and flips
  - Apply brightness and contrast adjustments
- **Metrics Calculation:**
  - Compute standard segmentation metrics (IoU, mAP, Dice coefficient, etc.).
  - Calculate precision, recall, F1 score
- **Visualization:**
  - Display instance segmentation results
  - Compare ground truth and predicted masks
  - Visualize evaluation metrics
  - Save visualization results
- **TensorBoard Logger:**
  - Functions to log training statistics.
- **Model Saving and Loading:**
  - Standardized method for handling checkpoints.

---

## **3. train.py**

Handles the training process in a simple and structured way.

### **Workflow:**

1. **Load Configuration:** Read settings from `config.py`.
2. **Data Preparation:**
   - Check if dataset split exists, otherwise generate it.
   - Apply augmentations.
3. **Model Initialization:**
   - Load Mask R-CNN with pre-trained ResNet50 backbone.
   - Configure for single-class segmentation.
4. **Loss Function:**
   - Implement combined loss (BCE + Dice + Focal Loss).
5. **Optimizer Setup:**
   - Adam optimizer with weight decay.
   - Learning rate scheduler.
6. **Training Loop:**
   - Train on the dataset.
   - Save checkpoints periodically.
   - Log training progress to TensorBoard.
7. **Validation Step:**
   - Evaluate on the validation set.
   - Save best-performing model.

---

## **4. evaluate.py**

Handles model evaluation on the test set.

### **Workflow:**

1. **Load Configuration:** Load settings from `config.py`.
2. **Load Trained Model:** Use the best checkpoint.
3. **Inference:**
   - Run segmentation on test images.
   - Compute evaluation metrics:
     - Precision, Recall, F1 Score
     - IoU (Intersection over Union)
     - Dice Coefficient
     - Pixel Error and Rand Error
   - Save visualized results.
4. **Export Predictions:** Save results in a structured format.

---

## **5. datasets/emps_dataset.py**

Custom dataset class for electron microscopy particle segmentation.

### **Key Features:**

- Load images and masks from the dataset
- Apply transformations and augmentations
- Convert to PyTorch tensors
- Implement the PyTorch Dataset interface

---

## **6. tests/**

A directory for unit tests to ensure code reliability and correctness.

### **Unit Test Considerations:**

- **test\_config.py** - Ensure configurations load correctly and data splitting logic functions as expected.
- **test\_utils.py** - Test dataset loading, augmentation, and metric computations.
- **test\_train.py** - Validate training workflow on a small dataset.
- **test\_evaluate.py** - Verify that model inference runs smoothly and outputs correct predictions.

---

## **Next Steps**

- Implement dataset loading and augmentation in `utils.py`.
- Develop `train.py` and `evaluate.py` with a modular, clean structure.
- Test the configuration management and dataset split logic.
- Implement unit tests in `tests/` to ensure functionality across components.

This setup ensures a scalable and user-friendly pipeline for instance segmentation using Mask R-CNN.

