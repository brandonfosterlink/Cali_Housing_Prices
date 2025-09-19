"""
Enhanced Housing Price Prediction using Support Vector Machine (SVM)
==================================================================

This script implements an enhanced SVM model with:
- Polynomial feature engineering
- Multiple kernel options (Linear, RBF, Polynomial)
- Hyperparameter tuning with grid search
- Feature interactions and transformations
- Model comparison and evaluation

Sections:
1. Data Import & Feature Engineering
2. Preprocessing with Polynomial Features
3. Enhanced Model Setup (Multiple Kernels)
4. Hyperparameter Tuning
5. Training & Model Comparison
6. Testing & Evaluation
7. Saving Best Model
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("ENHANCED HOUSING PRICE PREDICTION WITH SVM")
print("=" * 70)

# ============================================================================
# 1. DATA IMPORT & FEATURE ENGINEERING
# ============================================================================
print("\n1. DATA IMPORT & FEATURE ENGINEERING")
print("-" * 40)

# Load the California Housing dataset
print("Loading California Housing dataset...")
housing_data = fetch_california_housing()
X = housing_data.data
y = housing_data.target

# Create DataFrame for better visualization
feature_names = housing_data.feature_names
df = pd.DataFrame(X, columns=feature_names)
df['MedHouseVal'] = y

print(f"Original dataset shape: {df.shape}")
print(f"Features: {feature_names}")

# Feature Engineering: Create new meaningful features
print("\nCreating engineered features...")

# 1. Feature interactions (domain knowledge)
df['Income_per_Room'] = df['MedInc'] / (df['AveRooms'] + 1e-8)  # Income per room
df['Income_per_Bedroom'] = df['MedInc'] / (df['AveBedrms'] + 1e-8)  # Income per bedroom
df['Rooms_per_Household'] = df['AveRooms'] * df['AveOccup']  # Total rooms per household
df['Bedrooms_per_Household'] = df['AveBedrms'] * df['AveOccup']  # Total bedrooms per household

# 2. Location-based features
df['Distance_to_Ocean'] = df['Longitude'] * df['Latitude']  # Distance proxy
df['Coastal_Area'] = (df['Longitude'] > -118.5).astype(int)  # Binary coastal indicator

# 3. Age-based features
df['House_Age_Group'] = pd.cut(df['HouseAge'], bins=[0, 10, 20, 30, 50], labels=[0, 1, 2, 3])
df['House_Age_Group'] = df['House_Age_Group'].astype(int)

# 4. Log transformations for skewed features
df['Log_MedInc'] = np.log1p(df['MedInc'])
df['Log_Population'] = np.log1p(df['Population'])

# Update feature names
new_feature_names = list(feature_names) + [
    'Income_per_Room', 'Income_per_Bedroom', 'Rooms_per_Household', 
    'Bedrooms_per_Household', 'Distance_to_Ocean', 'Coastal_Area',
    'House_Age_Group', 'Log_MedInc', 'Log_Population'
]

print(f"Enhanced dataset shape: {df.shape}")
print(f"New features created: {len(new_feature_names) - len(feature_names)}")

# Display feature correlations with target
correlations = df.corr()['MedHouseVal'].abs().sort_values(ascending=False)
print(f"\nTop 5 features correlated with house value:")
for i, (feature, corr) in enumerate(correlations.head(6).items()):
    if feature != 'MedHouseVal':
        print(f"{i+1}. {feature}: {corr:.3f}")

# ============================================================================
# 2. PREPROCESSING WITH POLYNOMIAL FEATURES
# ============================================================================
print("\n\n2. PREPROCESSING WITH POLYNOMIAL FEATURES")
print("-" * 40)

# Prepare features (exclude target)
X_enhanced = df.drop('MedHouseVal', axis=1).values

# Test different polynomial degrees
polynomial_degrees = [1, 2, 3]
X_poly_versions = {}

print("Creating polynomial features...")
for degree in polynomial_degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X_enhanced)
    X_poly_versions[degree] = X_poly
    print(f"Degree {degree}: {X_enhanced.shape[1]} → {X_poly.shape[1]} features")

# Choose degree 2 for main analysis (balance between complexity and performance)
chosen_degree = 2
X_final = X_poly_versions[chosen_degree]

print(f"\nUsing polynomial degree {chosen_degree} for main analysis")
print(f"Final feature matrix shape: {X_final.shape}")

# Feature scaling
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# 3. ENHANCED MODEL SETUP (MULTIPLE KERNELS)
# ============================================================================
print("\n\n3. ENHANCED MODEL SETUP (MULTIPLE KERNELS)")
print("-" * 40)

# Define different kernel configurations to test
kernel_configs = {
    'Linear': {'kernel': 'linear'},
    'RBF': {'kernel': 'rbf', 'gamma': 'scale'},
    'Polynomial': {'kernel': 'poly', 'degree': 2, 'gamma': 'scale'},
    'Polynomial_Degree3': {'kernel': 'poly', 'degree': 3, 'gamma': 'scale'}
}

print("Available kernel configurations:")
for name, config in kernel_configs.items():
    print(f"- {name}: {config}")

# ============================================================================
# 4. HYPERPARAMETER TUNING
# ============================================================================
print("\n\n4. HYPERPARAMETER TUNING")
print("-" * 40)

# Define parameter grids for different kernels
param_grids = {
    'Linear': {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5, 1.0]
    },
    'RBF': {
        'C': [1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.5]
    },
    'Polynomial': {
        'C': [1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.5]
    }
}

print("Performing hyperparameter tuning...")
best_models = {}
tuning_results = {}

# Use a smaller sample for faster tuning
sample_size = min(2000, len(X_train))
X_train_sample = X_train[:sample_size]
y_train_sample = y_train[:sample_size]

for kernel_name in ['Linear', 'RBF', 'Polynomial']:
    print(f"\nTuning {kernel_name} kernel...")
    
    # Create SVR model with base parameters
    base_params = kernel_configs[kernel_name]
    svr = SVR(**base_params)
    
    # Perform grid search
    grid_search = GridSearchCV(
        svr, 
        param_grids[kernel_name], 
        cv=3, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train_sample, y_train_sample)
    
    best_models[kernel_name] = grid_search.best_estimator_
    tuning_results[kernel_name] = {
        'best_params': grid_search.best_params_,
        'best_score': -grid_search.best_score_,
        'best_rmse': np.sqrt(-grid_search.best_score_)
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.4f}")

# ============================================================================
# 5. TRAINING & MODEL COMPARISON
# ============================================================================
print("\n\n5. TRAINING & MODEL COMPARISON")
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
# 6. TESTING & EVALUATION
# ============================================================================
print("\n\n6. TESTING & EVALUATION")
print("-" * 40)

# Create comprehensive comparison table
print("Model Performance Comparison:")
print("=" * 80)
print(f"{'Model':<15} {'Train RMSE':<12} {'Test RMSE':<12} {'Train R²':<10} {'Test R²':<10}")
print("-" * 80)

for name, results in model_results.items():
    print(f"{name:<15} {results['train_rmse']:<12.4f} {results['test_rmse']:<12.4f} "
          f"{results['train_r2']:<10.4f} {results['test_r2']:<10.4f}")

# Visualization of results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Model comparison
model_names = list(model_results.keys())
test_rmses = [model_results[name]['test_rmse'] for name in model_names]
test_r2s = [model_results[name]['test_r2'] for name in model_names]

axes[0, 0].bar(model_names, test_rmses, color=['skyblue', 'lightcoral', 'lightgreen'])
axes[0, 0].set_title('Test RMSE Comparison')
axes[0, 0].set_ylabel('RMSE')
axes[0, 0].tick_params(axis='x', rotation=45)

axes[0, 1].bar(model_names, test_r2s, color=['skyblue', 'lightcoral', 'lightgreen'])
axes[0, 1].set_title('Test R² Comparison')
axes[0, 1].set_ylabel('R² Score')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 2: Best model predictions vs actual
best_results = model_results[best_model_name]
axes[1, 0].scatter(y_test, best_results['y_pred_test'], alpha=0.5, color='orange')
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual House Value')
axes[1, 0].set_ylabel('Predicted House Value')
axes[1, 0].set_title(f'Best Model ({best_model_name}) - Test Set\nR² = {best_results["test_r2"]:.3f}')
axes[1, 0].grid(True)

# Plot 3: Residuals for best model
residuals = y_test - best_results['y_pred_test']
axes[1, 1].scatter(best_results['y_pred_test'], residuals, alpha=0.5, color='green')
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Predicted House Value')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title(f'Residuals Plot - {best_model_name}')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('enhanced_model_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance analysis (for linear kernel)
if best_model_name == 'Linear':
    print("\nFeature Importance Analysis (Linear Model):")
    print("-" * 50)
    
    # Get coefficients from linear model
    coefficients = best_results['model'].coef_[0]
    
    # Create feature importance DataFrame
    # Note: With polynomial features, we have many more features
    feature_importance = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(len(coefficients))],
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("Top 10 most important features:")
    print(feature_importance.head(10))

# ============================================================================
# 7. SAVING BEST MODEL
# ============================================================================
print("\n\n7. SAVING BEST MODEL")
print("-" * 40)

# Create model directory
model_dir = "enhanced_saved_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created directory: {model_dir}")

# Save the best model
best_model_path = os.path.join(model_dir, f"best_svm_model_{best_model_name.lower()}.pkl")
import joblib
joblib.dump({
    'model': best_results['model'],
    'scaler': scaler,
    'polynomial_degree': chosen_degree,
    'feature_names': new_feature_names,
    'model_type': best_model_name,
    'performance': {
        'test_rmse': best_results['test_rmse'],
        'test_r2': best_results['test_r2'],
        'test_mae': best_results['test_mae']
    }
}, best_model_path)

print(f"Best model saved to: {best_model_path}")

# Save comprehensive results
results_path = os.path.join(model_dir, "model_comparison_results.txt")
with open(results_path, 'w') as f:
    f.write("Enhanced SVM Housing Price Prediction - Model Comparison\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("Feature Engineering:\n")
    f.write(f"- Original features: {len(feature_names)}\n")
    f.write(f"- Engineered features: {len(new_feature_names)}\n")
    f.write(f"- Polynomial degree: {chosen_degree}\n")
    f.write(f"- Final features: {X_final.shape[1]}\n\n")
    
    f.write("Model Performance Comparison:\n")
    f.write("-" * 40 + "\n")
    for name, results in model_results.items():
        f.write(f"{name}:\n")
        f.write(f"  Test RMSE: {results['test_rmse']:.4f}\n")
        f.write(f"  Test R²: {results['test_r2']:.4f}\n")
        f.write(f"  Test MAE: {results['test_mae']:.4f}\n\n")
    
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Best Test RMSE: {best_results['test_rmse']:.4f}\n")
    f.write(f"Best Test R²: {best_results['test_r2']:.4f}\n")

print(f"Results saved to: {results_path}")

# Create prediction function
def predict_house_price(features_dict):
    """
    Predict house price using the best trained model.
    
    Args:
        features_dict: Dictionary with feature values
    
    Returns:
        Predicted house value
    """
    # Load model
    model_data = joblib.load(best_model_path)
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Convert features to array
    feature_values = []
    for feature in new_feature_names:
        if feature in features_dict:
            feature_values.append(features_dict[feature])
        else:
            feature_values.append(0)  # Default value
    
    feature_array = np.array(feature_values).reshape(1, -1)
    
    # Apply polynomial transformation
    poly = PolynomialFeatures(degree=chosen_degree, include_bias=False)
    feature_array_poly = poly.fit_transform(feature_array)
    
    # Scale features
    feature_array_scaled = scaler.transform(feature_array_poly)
    
    # Make prediction
    prediction = model.predict(feature_array_scaled)[0]
    
    return prediction

print("\nExample prediction function created: predict_house_price()")
print("=" * 70)
print("ENHANCED SVM HOUSING PRICE PREDICTION COMPLETED!")
print("=" * 70)
print(f"Best model: {best_model_name}")
print(f"Test RMSE improvement: {model_results[best_model_name]['test_rmse']:.4f}")
print(f"Test R² improvement: {model_results[best_model_name]['test_r2']:.4f}")
print(f"\nFiles created:")
print(f"- {best_model_path} (best trained model)")
print(f"- {results_path} (comprehensive results)")
print(f"- enhanced_model_evaluation.png (evaluation plots)")
