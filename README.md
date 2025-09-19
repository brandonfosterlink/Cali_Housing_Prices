# Housing Price Prediction with SVM

This project implements a Support Vector Machine (SVM) regression model to predict housing prices using PyTorch.

## Features

- Complete ML pipeline with 6 distinct sections as requested
- Uses the California Housing dataset
- PyTorch-based SVM implementation
- Comprehensive evaluation metrics
- Model saving and loading functionality
- Visualization of results

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python housing_svm_model.py
```

## Model Sections

1. **Data Import**: Loads the California Housing dataset using pandas
2. **Preprocessing**: Data cleaning, scaling, and train/test split
3. **Model Setup**: PyTorch SVM regression implementation
4. **Training**: Model training with Adam optimizer
5. **Testing**: Model evaluation with RMSE, MAE, and R² metrics
6. **Saving Model**: Saves trained model and metadata

## Output Files

- `saved_models/svm_housing_model.pth`: Trained model
- `saved_models/model_info.txt`: Model information and metrics
- `training_history.png`: Training loss and RMSE plots
- `model_evaluation.png`: Prediction vs actual plots and residuals

## Model Performance

The model provides comprehensive evaluation metrics including:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Visualization plots for training history and predictions

## Custom SVM Implementation

The SVM uses an epsilon-insensitive loss function:
- **Epsilon**: 0.1 (tolerance for errors)
- **C**: 1.0 (regularization parameter)
- **Optimizer**: Adam with learning rate 0.001
- **Training**: 100 epochs with mini-batch training
