"""
Enhanced Housing Price Prediction with Meaningful Features
========================================================

This script creates a new SVM model using 12 meaningful features instead of 
polynomial feature expansion. Users can input real-world housing features
that make intuitive sense.

Features:
- 8 original California housing features
- 4 additional meaningful features (distance to coast, crime rate, school rating, property tax)
- Linear SVM model (no polynomial expansion)
- Simple, interpretable predictions
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("ENHANCED HOUSING PRICE PREDICTION - MEANINGFUL FEATURES")
print("=" * 60)

# ============================================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================================
print("\n1. DATA LOADING & FEATURE ENGINEERING")
print("-" * 40)

# Load the California Housing dataset
print("Loading California Housing dataset...")
housing_data = fetch_california_housing()
X = housing_data.data
y = housing_data.target

# Create DataFrame
feature_names = list(housing_data.feature_names)
df = pd.DataFrame(X, columns=feature_names)
df['MedHouseVal'] = y

print(f"Original dataset shape: {df.shape}")
print(f"Features: {feature_names}")

# Create meaningful additional features based on location and demographics
print("\nCreating meaningful additional features...")

# 1. Distance to Coast (estimated from longitude)
# California coastline is roughly at longitude -122 to -124
coast_longitude = -122.5  # Approximate coastal longitude
df['Distance_to_Coast'] = np.abs(df['Longitude'] - coast_longitude) * 50  # Rough miles conversion

# 2. Crime Rate (estimated from population density and income)
# Higher population density and lower income areas tend to have higher crime
population_density = df['Population'] / 1000  # Rough density metric
income_factor = 10 / (df['MedInc'] + 1)  # Inverse income factor
df['Crime_Rate'] = (population_density * income_factor * 5).clip(0, 100)

# 3. School Rating (estimated from income and location)
# Higher income areas typically have better schools
# Coastal areas also tend to have better schools
coastal_factor = np.where(df['Longitude'] < -122, 1.5, 1.0)  # Coastal boost
df['School_Rating'] = ((df['MedInc'] * 2 + coastal_factor) * 0.8).clip(1, 10)

# 4. Property Tax Rate (estimated from location and house value)
# Higher value areas typically have higher tax rates
estimated_value = df['MedInc'] * 50  # Rough house value estimate
df['Property_Tax_Rate'] = (0.5 + estimated_value * 0.01).clip(0.5, 3.0)

# Update feature names
enhanced_feature_names = feature_names + [
    'Distance_to_Coast', 'Crime_Rate', 'School_Rating', 'Property_Tax_Rate'
]

print(f"Enhanced dataset shape: {df.shape}")
print(f"New features created:")
for i, feature in enumerate(enhanced_feature_names[8:], 1):
    print(f"  {i}. {feature}")

# Display feature statistics
print(f"\nFeature Statistics:")
print(df[enhanced_feature_names].describe())

# ============================================================================
# 2. PREPROCESSING
# ============================================================================
print("\n\n2. PREPROCESSING")
print("-" * 40)

# Prepare features (exclude target)
X_enhanced = df[enhanced_feature_names].values

# Feature scaling
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_enhanced)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# 3. MODEL TRAINING & HYPERPARAMETER TUNING
# ============================================================================
print("\n\n3. MODEL TRAINING & HYPERPARAMETER TUNING")
print("-" * 40)

# Define parameter grid for different kernels
param_grids = {
    'Linear': {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5]
    },
    'RBF': {
        'C': [1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.5]
    }
}

print("Performing hyperparameter tuning...")
best_models = {}

# Use a smaller sample for faster tuning
sample_size = min(2000, len(X_train))
X_train_sample = X_train[:sample_size]
y_train_sample = y_train[:sample_size]

for kernel_name, params in param_grids.items():
    print(f"\nTuning {kernel_name} kernel...")
    
    # Create SVR model
    svr = SVR(kernel=kernel_name.lower())
    
    # Perform grid search
    grid_search = GridSearchCV(
        svr, 
        params, 
        cv=3, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_sample, y_train_sample)
    
    best_models[kernel_name] = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

# ============================================================================
# 4. MODEL EVALUATION
# ============================================================================
print("\n\n4. MODEL EVALUATION")
print("-" * 40)

print("Training final models on full dataset...")
model_results = {}

for kernel_name, model in best_models.items():
    print(f"\nTraining {kernel_name} model...")
    
    # Train on full dataset
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    model_results[kernel_name] = {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test
    }
    
    print(f"Training RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

# Find best model
best_model_name = min(model_results.keys(), 
                     key=lambda x: model_results[x]['test_rmse'])

print(f"\n🏆 Best performing model: {best_model_name}")
print(f"Test RMSE: {model_results[best_model_name]['test_rmse']:.4f}")
print(f"Test R²: {model_results[best_model_name]['test_r2']:.4f}")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================
print("\n\n5. VISUALIZATION")
print("-" * 40)

# Create comparison plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Model comparison
model_names = list(model_results.keys())
test_rmses = [model_results[name]['test_rmse'] for name in model_names]
test_r2s = [model_results[name]['test_r2'] for name in model_names]

axes[0].bar(model_names, test_rmses, color=['skyblue', 'lightcoral'])
axes[0].set_title('Test RMSE Comparison')
axes[0].set_ylabel('RMSE')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(model_names, test_r2s, color=['skyblue', 'lightcoral'])
axes[1].set_title('Test R² Comparison')
axes[1].set_ylabel('R² Score')
axes[1].tick_params(axis='x', rotation=45)

# Best model predictions vs actual
best_results = model_results[best_model_name]
axes[2].scatter(y_test, best_results['y_pred_test'], alpha=0.5, color='orange')
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[2].set_xlabel('Actual House Value')
axes[2].set_ylabel('Predicted House Value')
axes[2].set_title(f'Best Model ({best_model_name}) - Test Set\nR² = {best_results["test_r2"]:.3f}')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('enhanced_features_model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. SAVING MODEL
# ============================================================================
print("\n\n6. SAVING MODEL")
print("-" * 40)

# Create model directory
model_dir = "enhanced_features_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")

# Save the best model
best_model_path = os.path.join(model_dir, f"best_enhanced_features_model_{best_model_name.lower()}.pkl")
joblib.dump({
    'model': best_results['model'],
    'scaler': scaler,
    'feature_names': enhanced_feature_names,
    'model_type': best_model_name,
    'performance': {
        'test_rmse': best_results['test_rmse'],
        'test_r2': best_results['test_r2'],
        'test_mae': best_results['test_mae']
    }
}, best_model_path)

print(f"Best model saved to: {best_model_path}")

# Save model information
info_path = os.path.join(model_dir, "model_info.txt")
with open(info_path, 'w') as f:
    f.write("Enhanced Features Housing Price Prediction Model\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model Type: Support Vector Machine ({best_model_name} kernel)\n")
    f.write(f"Input Features: {len(enhanced_feature_names)}\n")
    f.write(f"Feature Names: {', '.join(enhanced_feature_names)}\n")
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Best Parameters: {best_models[best_model_name].get_params()}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"Training RMSE: {best_results['train_rmse']:.6f}\n")
    f.write(f"Training R²: {best_results['train_r2']:.6f}\n")
    f.write(f"Test RMSE: {best_results['test_rmse']:.6f}\n")
    f.write(f"Test R²: {best_results['test_r2']:.6f}\n")

print(f"Model information saved to: {info_path}")

print("\n" + "=" * 60)
print("ENHANCED FEATURES MODEL TRAINING COMPLETED!")
print("=" * 60)
print(f"Best model: {best_model_name}")
print(f"Test RMSE: {model_results[best_model_name]['test_rmse']:.4f}")
print(f"Test R²: {model_results[best_model_name]['test_r2']:.4f}")
print(f"\nFeatures used:")
for i, feature in enumerate(enhanced_feature_names, 1):
    print(f"  {i}. {feature}")
print(f"\nFiles created:")
print(f"- {best_model_path} (trained model)")
print(f"- {info_path} (model information)")
print(f"- enhanced_features_model_evaluation.png (evaluation plots)")
