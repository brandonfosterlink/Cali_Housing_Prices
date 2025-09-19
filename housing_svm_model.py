"""
Housing Price Prediction using Support Vector Machine (SVM)
===========================================================

This script implements a complete machine learning pipeline for predicting house prices
using a Support Vector Machine model built with PyTorch.

Sections:
1. Data Import
2. Preprocessing
3. Model Setup (PyTorch SVM)
4. Training
5. Testing
6. Saving Model
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("HOUSING PRICE PREDICTION WITH SUPPORT VECTOR MACHINE")
print("=" * 60)

# ============================================================================
# 1. DATA IMPORT
# ============================================================================
print("\n1. DATA IMPORT")
print("-" * 30)

# Load the California Housing dataset
print("Loading California Housing dataset...")
housing_data = fetch_california_housing()
X = housing_data.data
y = housing_data.target

# Create DataFrame for better visualization
feature_names = housing_data.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['MedHouseVal'] = y

print(f"Dataset shape: {df.shape}")
print(f"Features: {feature_names}")
print(f"Target variable: MedHouseVal (median house value in $100,000s)")
print("\nFirst few rows of the dataset:")
print(df.head())

print("\nDataset statistics:")
print(df.describe())

# ============================================================================
# 2. PREPROCESSING
# ============================================================================
print("\n\n2. PREPROCESSING")
print("-" * 30)

# Check for missing values
print("Checking for missing values...")
missing_values = df.isnull().sum()
print(f"Missing values per column:\n{missing_values}")

# Handle any missing values (if any)
if missing_values.sum() > 0:
    print("Handling missing values...")
    df = df.dropna()
    print("Missing values handled by dropping rows with NaN values.")

# Feature scaling
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Number of features: {X_train.shape[1]}")

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

print("Data successfully converted to PyTorch tensors.")

# ============================================================================
# 3. MODEL SETUP (PyTorch SVM)
# ============================================================================
print("\n\n3. MODEL SETUP")
print("-" * 30)

class SVMRegressor(nn.Module):
    """
    Support Vector Machine for Regression implemented in PyTorch.
    
    This implementation uses the epsilon-SVM approach where we minimize:
    L = 0.5 * ||w||² + C * Σ(ξ + ξ*)
    subject to:
    -ε - ξ* ≤ y - (w·x + b) ≤ ε + ξ
    """
    
    def __init__(self, input_dim, epsilon=0.1, C=1.0):
        super(SVMRegressor, self).__init__()
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.C = C
        
        # Linear SVM parameters
        self.weight = nn.Parameter(torch.randn(input_dim, 1))
        self.bias = nn.Parameter(torch.randn(1))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        """Forward pass: compute w·x + b"""
        return torch.matmul(x, self.weight) + self.bias
    
    def svm_loss(self, y_pred, y_true):
        """
        Compute SVM regression loss with epsilon-insensitive loss function.
        
        Args:
            y_pred: Predicted values
            y_true: True values
            
        Returns:
            SVM loss value
        """
        # Compute residuals
        residuals = y_true - y_pred
        
        # Epsilon-insensitive loss
        epsilon_loss = torch.max(torch.zeros_like(residuals), 
                               torch.abs(residuals) - self.epsilon)
        
        # L2 regularization for weight
        l2_reg = 0.5 * torch.sum(self.weight ** 2)
        
        # Total SVM loss
        total_loss = l2_reg + self.C * torch.mean(epsilon_loss)
        
        return total_loss

# Initialize the model
input_dim = X_train.shape[1]
model = SVMRegressor(input_dim=input_dim, epsilon=0.1, C=1.0)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"SVM Model initialized:")
print(f"- Input dimensions: {input_dim}")
print(f"- Epsilon: {model.epsilon}")
print(f"- C (regularization parameter): {model.C}")
print(f"- Optimizer: Adam with learning rate 0.001")

# Print model summary
print(f"\nModel parameters:")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# ============================================================================
# 4. TRAINING
# ============================================================================
print("\n\n4. TRAINING")
print("-" * 30)

# Training parameters
num_epochs = 100
batch_size = 32

# Lists to store training history
train_losses = []
train_rmse = []

print(f"Starting training for {num_epochs} epochs...")
print(f"Batch size: {batch_size}")

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    
    # Mini-batch training
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        # Forward pass
        y_pred = model(batch_X)
        loss = model.svm_loss(y_pred, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    # Calculate average loss for this epoch
    avg_loss = np.mean(epoch_losses)
    train_losses.append(avg_loss)
    
    # Calculate RMSE for this epoch
    with torch.no_grad():
        model.eval()
        train_pred = model(X_train_tensor)
        train_rmse_val = torch.sqrt(torch.mean((train_pred - y_train_tensor) ** 2)).item()
        train_rmse.append(train_rmse_val)
    
    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, RMSE: {train_rmse_val:.6f}")

print("Training completed!")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('SVM Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_rmse)
plt.title('Training RMSE Over Time')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. TESTING
# ============================================================================
print("\n\n5. TESTING")
print("-" * 30)

# Set model to evaluation mode
model.eval()

# Make predictions on test set
with torch.no_grad():
    y_pred_test = model(X_test_tensor)
    y_pred_train = model(X_train_tensor)

# Convert predictions back to numpy for evaluation
y_pred_test_np = y_pred_test.numpy().flatten()
y_pred_train_np = y_pred_train.numpy().flatten()

# Calculate metrics
train_mse = mean_squared_error(y_train, y_pred_train_np)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_pred_train_np)
train_r2 = r2_score(y_train, y_pred_train_np)

test_mse = mean_squared_error(y_test, y_pred_test_np)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred_test_np)
test_r2 = r2_score(y_test, y_pred_test_np)

print("Model Performance Metrics:")
print("=" * 50)
print(f"{'Metric':<15} {'Training':<15} {'Testing':<15}")
print("-" * 50)
print(f"{'RMSE':<15} {train_rmse:<15.6f} {test_rmse:<15.6f}")
print(f"{'MSE':<15} {train_mse:<15.6f} {test_mse:<15.6f}")
print(f"{'MAE':<15} {train_mae:<15.6f} {test_mae:<15.6f}")
print(f"{'R² Score':<15} {train_r2:<15.6f} {test_r2:<15.6f}")

# Visualize predictions
plt.figure(figsize=(15, 5))

# Training set predictions
plt.subplot(1, 3, 1)
plt.scatter(y_train, y_pred_train_np, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual House Value')
plt.ylabel('Predicted House Value')
plt.title(f'Training Set (R² = {train_r2:.3f})')
plt.grid(True)

# Test set predictions
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_test_np, alpha=0.5, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual House Value')
plt.ylabel('Predicted House Value')
plt.title(f'Test Set (R² = {test_r2:.3f})')
plt.grid(True)

# Residuals plot
plt.subplot(1, 3, 3)
residuals = y_test - y_pred_test_np
plt.scatter(y_pred_test_np, residuals, alpha=0.5, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted House Value')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.grid(True)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nModel evaluation completed!")
print(f"Test R² Score: {test_r2:.3f}")
print(f"Test RMSE: {test_rmse:.6f}")

# ============================================================================
# 6. SAVING MODEL
# ============================================================================
print("\n\n6. SAVING MODEL")
print("-" * 30)

# Create model directory if it doesn't exist
model_dir = "saved_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")

# Save the model state dict
model_path = os.path.join(model_dir, "svm_housing_model.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_dim': input_dim,
        'epsilon': model.epsilon,
        'C': model.C
    },
    'scaler': scaler,
    'feature_names': feature_names,
    'training_metrics': {
        'train_rmse': train_rmse,
        'train_r2': train_r2
    },
    'test_metrics': {
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }
}, model_path)

print(f"Model saved successfully to: {model_path}")

# Save model information
info_path = os.path.join(model_dir, "model_info.txt")
with open(info_path, 'w') as f:
    f.write("SVM Housing Price Prediction Model\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Model Type: Support Vector Machine (PyTorch)\n")
    f.write(f"Input Features: {input_dim}\n")
    f.write(f"Feature Names: {', '.join(feature_names)}\n")
    f.write(f"Epsilon: {model.epsilon}\n")
    f.write(f"C (Regularization): {model.C}\n")
    f.write(f"Optimizer: Adam\n")
    f.write(f"Learning Rate: 0.001\n")
    f.write(f"Epochs: {num_epochs}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"Training RMSE: {train_rmse:.6f}\n")
    f.write(f"Training R²: {train_r2:.6f}\n")
    f.write(f"Test RMSE: {test_rmse:.6f}\n")
    f.write(f"Test R²: {test_r2:.6f}\n")

print(f"Model information saved to: {info_path}")

# Create a simple prediction function for future use
def load_and_predict(model_path, scaler, new_data):
    """
    Load the trained model and make predictions on new data.
    
    Args:
        model_path: Path to the saved model
        scaler: Fitted StandardScaler
        new_data: New data to predict (numpy array or list)
    
    Returns:
        Predictions as numpy array
    """
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = SVMRegressor(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Preprocess new data
    if isinstance(new_data, list):
        new_data = np.array(new_data)
    
    new_data_scaled = scaler.transform(new_data.reshape(1, -1))
    new_data_tensor = torch.FloatTensor(new_data_scaled)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(new_data_tensor)
    
    return prediction.numpy().flatten()[0]

print("\nExample usage function created: load_and_predict()")
print("=" * 60)
print("SVM HOUSING PRICE PREDICTION MODEL COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"Files created:")
print(f"- {model_path} (trained model)")
print(f"- {info_path} (model information)")
print(f"- training_history.png (training plots)")
print(f"- model_evaluation.png (evaluation plots)")
