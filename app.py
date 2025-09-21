"""
Flask Web Application for Housing Price Prediction
=================================================

This Flask app provides a web interface for predicting California housing prices
using the trained SVM models. Users can input housing features and get price predictions.

Features:
- Load both PyTorch and scikit-learn models
- User-friendly web interface
- Real-time price predictions
- Model comparison
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for models
enhanced_model = None
enhanced_scaler = None
pytorch_model = None
scaler = None
feature_names = None

class SVMRegressor(nn.Module):
    """
    Support Vector Machine for Regression implemented in PyTorch.
    This matches the model architecture from the training script.
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

def load_models():
    """Load the enhanced features model"""
    global enhanced_model, enhanced_scaler, feature_names
    
    try:
        # Load enhanced features model (primary model)
        enhanced_path = "enhanced_features_models/best_enhanced_features_model_rbf.pkl"
        print(f"🔍 Looking for enhanced features model at: {enhanced_path}")
        if os.path.exists(enhanced_path):
            print("📁 Enhanced features model file found, loading...")
            model_data = joblib.load(enhanced_path)
            enhanced_model = model_data['model']
            enhanced_scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            
            print("✅ Enhanced features model loaded successfully")
            print(f"📊 Model features: {len(feature_names)}")
            print(f"🎯 Model performance: R² = {model_data['performance']['test_r2']:.3f}")
        else:
            print("❌ Enhanced features model file not found")
            
        # Also try to load the original PyTorch model as backup
        pytorch_path = "saved_models/svm_housing_model.pth"
        print(f"🔍 Looking for PyTorch backup model at: {pytorch_path}")
        if os.path.exists(pytorch_path):
            print("📁 PyTorch model file found, loading as backup...")
            checkpoint = torch.load(pytorch_path, map_location='cpu', weights_only=False)
            
            # Recreate model architecture
            input_dim = checkpoint['model_config']['input_dim']
            pytorch_model = SVMRegressor(
                input_dim=input_dim,
                epsilon=checkpoint['model_config']['epsilon'],
                C=checkpoint['model_config']['C']
            )
            pytorch_model.load_state_dict(checkpoint['model_state_dict'])
            pytorch_model.eval()
            
            # Load scaler and feature names for backup
            scaler = checkpoint['scaler']
            pytorch_feature_names = checkpoint['feature_names']
            
            print("✅ PyTorch backup model loaded successfully")
        else:
            print("❌ PyTorch backup model file not found")
            
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")

def predict_pytorch(features):
    """Make prediction using PyTorch model"""
    global pytorch_model, scaler
    
    if pytorch_model is None or scaler is None:
        return None
    
    try:
        # Convert to numpy array and reshape
        feature_array = np.array(features).reshape(1, -1)
        
        # Scale features
        feature_array_scaled = scaler.transform(feature_array)
        
        # Convert to PyTorch tensor
        feature_tensor = torch.FloatTensor(feature_array_scaled)
        
        # Make prediction
        with torch.no_grad():
            prediction = pytorch_model(feature_tensor)
        
        return prediction.numpy().flatten()[0]
    except Exception as e:
        print(f"Error in PyTorch prediction: {str(e)}")
        return None

def predict_enhanced(features):
    """Make prediction using enhanced features model"""
    global enhanced_model, enhanced_scaler
    
    if enhanced_model is None or enhanced_scaler is None:
        return None
    
    try:
        # Convert to numpy array and reshape
        feature_array = np.array(features).reshape(1, -1)
        
        # Scale features
        feature_array_scaled = enhanced_scaler.transform(feature_array)
        
        # Make prediction
        prediction = enhanced_model.predict(feature_array_scaled)[0]
        
        return prediction
    except Exception as e:
        print(f"Error in enhanced model prediction: {str(e)}")
        return None

def predict_pytorch_backup(features):
    """Make prediction using PyTorch model (backup)"""
    global pytorch_model, scaler
    
    if pytorch_model is None or scaler is None:
        return None
    
    try:
        # Convert to numpy array and reshape
        feature_array = np.array(features).reshape(1, -1)
        
        # Scale features
        feature_array_scaled = scaler.transform(feature_array)
        
        # Convert to PyTorch tensor
        feature_tensor = torch.FloatTensor(feature_array_scaled)
        
        # Make prediction
        with torch.no_grad():
            prediction = pytorch_model(feature_tensor)
        
        return prediction.numpy().flatten()[0]
    except Exception as e:
        print(f"Error in PyTorch prediction: {str(e)}")
        return None

@app.route('/')
def index():
    """Render the main prediction form"""
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = request.get_json()
        
        # Extract features in the correct order (12 features)
        features = [
            float(data.get('MedInc', 0)),
            float(data.get('HouseAge', 0)),
            float(data.get('AveRooms', 0)),
            float(data.get('AveBedrms', 0)),
            float(data.get('Population', 0)),
            float(data.get('AveOccup', 0)),
            float(data.get('Latitude', 0)),
            float(data.get('Longitude', 0)),
            float(data.get('Distance_to_Coast', 0)),
            float(data.get('Crime_Rate', 0)),
            float(data.get('School_Rating', 0)),
            float(data.get('Property_Tax_Rate', 0))
        ]
        
        # Validate inputs
        if any(f < 0 for f in features[:6]):  # First 6 features should be positive
            return jsonify({'error': 'Invalid input: Negative values not allowed for income, age, rooms, bedrooms, population, and occupancy'})
        
        # Make predictions with enhanced model (primary) and PyTorch (backup)
        enhanced_pred = predict_enhanced(features)
        pytorch_pred = predict_pytorch_backup(features[:8])  # Only use first 8 features for PyTorch
        
        # Prepare response
        response = {
            'success': True,
            'input_features': dict(zip(feature_names, features)),
            'predictions': {}
        }
        
        if enhanced_pred is not None:
            response['predictions']['enhanced'] = {
                'value': round(enhanced_pred, 4),
                'formatted': f"${enhanced_pred * 100000:,.0f}",
                'model_name': 'Enhanced Features RBF SVM'
            }
        
        if pytorch_pred is not None:
            response['predictions']['pytorch'] = {
                'value': round(pytorch_pred, 4),
                'formatted': f"${pytorch_pred * 100000:,.0f}",
                'model_name': 'PyTorch Linear SVM'
            }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'})

@app.route('/model_info')
def model_info():
    """Return information about loaded models"""
    info = {
        'enhanced_loaded': enhanced_model is not None,
        'pytorch_loaded': pytorch_model is not None,
        'feature_names': feature_names,
        'num_features': len(feature_names) if feature_names else 0
    }
    
    return jsonify(info)

@app.route('/health')
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({
        'status': 'healthy',
        'timestamp': pd.Timestamp.now().isoformat(),
        'models_loaded': {
            'enhanced_model': enhanced_model is not None,
            'pytorch_model': pytorch_model is not None
        }
    }), 200

if __name__ == '__main__':
    print("🏠 Loading Housing Price Prediction Models...")
    load_models()
    
    if enhanced_model is None and pytorch_model is None:
        print("❌ No models loaded. Please ensure model files exist.")
        exit(1)
    
    # Get port from environment variable (Cloud Run sets this)
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 Starting Flask application...")
    print(f"📱 Service will be available on port: {port}")
    print("🎯 Enhanced Features Model: 12 meaningful features")
    print("📊 Expected R² Score: ~0.75")
    
    # Run in production mode (no debug)
    app.run(host='0.0.0.0', port=port, debug=False)
