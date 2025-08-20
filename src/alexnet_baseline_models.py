"""
alexnet_models.py - WHO Grade Classification for Brain Tumors

This module contains deep learning models for classifying brain tumor WHO grades
from 2D MRI slices. We implement AlexNet architecture and compare it against
a simple fully-connected baseline model.

Authors: Brain Tumor Classification Project Team
Purpose: Predict WHO grades (1-4) from T1c MRI tumor slices
Data: UCSF-PDGM dataset with glioblastoma patient data

Usage:
    from alexnet_models import AlexNetWHO, SimpleBaseline, train_model
    
    # Create models
    alexnet = AlexNetWHO(num_classes=4)
    baseline = SimpleBaseline(num_classes=4)
    
    # Train models
    trained_model, train_accs, val_accs = train_model(alexnet, train_loader, val_loader)
    
    # Test performance
    accuracy, predictions, targets = test_model(trained_model, test_loader)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class AlexNetWHO(nn.Module):
    """
    Modified AlexNet for WHO Grade Classification
    
    This is a CNN architecture based on the original AlexNet paper (Krizhevsky et al., 2012)
    but adapted for medical imaging:
    - Input: Single-channel 224x224 MRI slices (grayscale medical images)
    - Output: 4 classes representing WHO grades 1-4
    - Architecture: 5 convolutional layers + 3 fully connected layers
    
    Key modifications from original AlexNet:
    1. Single input channel (medical images are grayscale)
    2. Smaller classifier to prevent overfitting on medical data
    3. Adaptive pooling for flexible input sizes
    
    Args:
        num_classes (int): Number of WHO grade classes (default: 4 for grades 1-4)
    """
    def __init__(self, num_classes=4):
        super(AlexNetWHO, self).__init__()
        
        # Feature Extraction Layers
        # These convolutional layers learn spatial patterns in tumor images
        self.features = nn.Sequential(
            # Layer 1: Large receptive field to capture global tumor patterns
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),  # Non-linear activation
            nn.MaxPool2d(kernel_size=3, stride=2),  # Downsample to reduce computation
            
            # Layer 2: Detect more complex features
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 3: Higher-level feature detection
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4: Further feature refinement
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 5: Final convolutional features
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling ensures consistent output size regardless of input variations
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classification Layers
        # These fully connected layers map visual features to WHO grade predictions
        self.classifier = nn.Sequential(
            # Regularization to prevent overfitting on small medical datasets
            nn.Dropout(0.5),
            
            # First FC layer: High-dimensional feature representation
            nn.Linear(256 * 6 * 6, 1024),  # 256 channels * 6*6 spatial = 9216 input features
            nn.ReLU(inplace=True),
            
            # Second FC layer: Intermediate feature compression
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            
            # Output layer: Final WHO grade prediction
            nn.Linear(256, num_classes),  # Maps to 4 WHO grades
        )
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 224, 224)
                             representing MRI tumor slices
        
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, num_classes)
                         Higher values indicate higher confidence for that WHO grade
        """
        # Extract spatial features using convolutional layers
        x = self.features(x)
        
        # Pool to fixed size and flatten for fully connected layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        
        # Classify based on extracted features
        x = self.classifier(x)
        
        return x

class SimpleBaseline(nn.Module):
    """
    Simple Fully-Connected Baseline Model
    
    This is a basic neural network that serves as a baseline comparison for AlexNet.
    It directly maps flattened pixel values to WHO grade predictions without 
    considering spatial relationships in the image.
    
    Purpose: 
    - Establish a simple baseline for comparison
    - If AlexNet can't beat this simple model, there might be issues with our approach
    - Common practice in deep learning to compare against simple baselines
    
    Architecture:
    - Flatten input image to 1D vector
    - Two fully connected layers with ReLU activation
    - Dropout for regularization
    
    Limitations:
    - Ignores spatial structure of medical images
    - Cannot capture local patterns or textures
    - Treats each pixel independently
    
    Args:
        num_classes (int): Number of WHO grade classes (default: 4)
    """
    def __init__(self, num_classes=4):
        super(SimpleBaseline, self).__init__()
        
        self.classifier = nn.Sequential(
            # Flatten 2D image to 1D vector
            nn.Flatten(),  # 224*224 = 50,176 features
            
            # Hidden layer: Learn basic feature combinations
            nn.Linear(224*224, 512),  # Map 50K pixels to 512 features
            nn.ReLU(inplace=True),
            
            # Regularization to prevent memorization
            nn.Dropout(0.5),
            
            # Output layer: Direct mapping to WHO grades
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        """
        Forward pass: Direct pixel-to-class mapping
        
        Args:
            x (torch.Tensor): Input MRI slice (batch_size, 1, 224, 224)
        
        Returns:
            torch.Tensor: WHO grade predictions (batch_size, num_classes)
        """
        return self.classifier(x)

def train_model(model, train_loader, val_loader, num_epochs=20, lr=0.001):
    """
    Train a neural network model for WHO grade classification
    
    This function implements a standard supervised learning training loop:
    1. Forward pass: Compute predictions from input images
    2. Loss calculation: Compare predictions with true WHO grades
    3. Backward pass: Compute gradients to update model parameters
    4. Validation: Monitor performance on unseen data to prevent overfitting
    
    Args:
        model (nn.Module): The neural network to train (AlexNet or Baseline)
        train_loader (DataLoader): Training data (MRI slices + WHO grade labels)
        val_loader (DataLoader): Validation data for monitoring overfitting
        num_epochs (int): Number of complete passes through training data
        lr (float): Learning rate - controls how much to update weights each step
    
    Returns:
        tuple: (trained_model, training_accuracies, validation_accuracies)
    
    Training Strategy:
    - Adam optimizer: Adaptive learning rate for stable convergence
    - Cross-entropy loss: Standard for multi-class classification
    - Early stopping: Save best model based on validation accuracy
    - Progress tracking: Monitor training progress with accuracy metrics
    """
    # Setup training environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    model.to(device)
    
    # Loss function: Cross-entropy for multi-class classification
    # Penalizes confident wrong predictions more than uncertain ones
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Adam with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Track performance over time
    train_accs = []
    val_accs = []
    best_val_acc = 0
    best_model_state = None
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training Phase: Update model parameters
        model.train()  # Enable dropout and batch norm training mode
        train_correct = 0
        train_total = 0
        
        # Progress bar for user feedback
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]')
        
        for batch_idx, (data, target) in enumerate(train_bar):
            # Move data to GPU if available
            data, target = data.to(device), target.to(device)
            
            # Forward pass: Get model predictions
            optimizer.zero_grad()  # Clear previous gradients
            output = model(data)   # Get WHO grade predictions
            
            # Calculate loss: How wrong are our predictions?
            loss = criterion(output, target)
            
            # Backward pass: Calculate gradients
            loss.backward()
            
            # Update model parameters based on gradients
            optimizer.step()
            
            # Track accuracy for monitoring
            _, predicted = torch.max(output, 1)  # Get predicted class
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Update progress bar with current accuracy
            current_acc = 100. * train_correct / train_total
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.2f}%'})
        
        # Validation Phase: Check performance on unseen data
        model.eval()  # Disable dropout for consistent evaluation
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # Don't compute gradients during validation
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                # Get predictions without updating weights
                output = model(data)
                _, predicted = torch.max(output, 1)
                
                # Track validation accuracy
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate epoch performance metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Training Accuracy: {train_acc:.2f}%')
        print(f'  Validation Accuracy: {val_acc:.2f}%')
        
        # Early stopping: Save best model to prevent overfitting
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f'  → New best validation accuracy: {best_val_acc:.2f}%')
        
        print('-' * 50)
    
    # Load the best performing model
    model.load_state_dict(best_model_state)
    print(f'\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%')
    
    return model, train_accs, val_accs

def test_model(model, test_loader):
    """
    Evaluate trained model on held-out test data
    
    This function provides an unbiased estimate of model performance by testing
    on data that was never seen during training or validation.
    
    Args:
        model (nn.Module): Trained neural network
        test_loader (DataLoader): Test dataset with ground truth WHO grades
    
    Returns:
        tuple: (accuracy, predictions, true_labels)
            - accuracy: Overall classification accuracy (0-1)
            - predictions: Model predictions for each test sample
            - true_labels: Ground truth WHO grades for comparison
    
    Why we need separate test data:
    - Training data: Used to learn model parameters
    - Validation data: Used to tune hyperparameters and prevent overfitting
    - Test data: Provides final, unbiased performance estimate
    """
    # Setup evaluation environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set to evaluation mode (disable dropout)
    
    # Collect all predictions and ground truth labels
    all_predictions = []
    all_targets = []
    
    print("Evaluating model on test data...")
    
    # Evaluate without updating model weights
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="Testing")):
            # Move data to appropriate device
            data, target = data.to(device), target.to(device)
            
            # Get model predictions
            output = model(data)
            
            # Convert logits to predicted classes
            _, predicted = torch.max(output, 1)
            
            # Store results for final analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate overall test accuracy
    accuracy = accuracy_score(all_targets, all_predictions)
    
    print(f'Test Results:')
    print(f'  Total test samples: {len(all_targets)}')
    print(f'  Correct predictions: {int(accuracy * len(all_targets))}')
    print(f'  Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    
    return accuracy, all_predictions, all_targets

def create_alexnet(num_classes=4):
    """
    Factory function to create AlexNet model
    
    Args:
        num_classes (int): Number of WHO grade classes
    
    Returns:
        AlexNetWHO: Initialized AlexNet model for brain tumor classification
    """
    model = AlexNetWHO(num_classes=num_classes)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"AlexNet Model Created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model

def create_baseline(num_classes=4):
    """
    Factory function to create simple baseline model
    
    Args:
        num_classes (int): Number of WHO grade classes
    
    Returns:
        SimpleBaseline: Initialized baseline model for comparison
    """
    model = SimpleBaseline(num_classes=num_classes)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Baseline Model Created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model