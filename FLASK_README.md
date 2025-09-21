# 🏠 California Housing Price Predictor - Flask Web App

A beautiful web application for predicting California housing prices using trained SVM models.

## 🚀 Features

- **Dual Model Support**: Uses both PyTorch and scikit-learn SVM models
- **User-Friendly Interface**: Clean, responsive web design
- **Real-time Predictions**: Instant price predictions as you type
- **Input Validation**: Ensures valid housing feature inputs
- **Model Comparison**: Shows predictions from both models side-by-side

## 📋 Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application

1. **Activate your virtual environment:**
   ```bash
   source venv/Scripts/activate  # Windows Git Bash
   # or
   venv\Scripts\activate        # Windows Command Prompt
   ```

2. **Install Flask (if not already installed):**
   ```bash
   pip install flask
   ```

3. **Run the Flask application:**
   ```bash
   python app.py
   ```

4. **Open your browser and go to:**
   ```
   http://127.0.0.1:5000
   ```

## 🎯 How to Use

1. **Fill in the form** with housing features:
   - **Median Income**: Income in $10,000s (e.g., 3.5 = $35,000)
   - **House Age**: Age of house in years
   - **Average Rooms**: Average rooms per household
   - **Average Bedrooms**: Average bedrooms per household
   - **Population**: Population in the block group
   - **Average Occupancy**: Average household members
   - **Latitude**: Latitude coordinate (32-42°N)
   - **Longitude**: Longitude coordinate (-125 to -114°W)

2. **Click "Predict House Price"** to get predictions

3. **View results** from both models:
   - PyTorch SVM model
   - Scikit-learn RBF model

## 📊 Model Information

- **PyTorch Model**: Custom SVM implementation with epsilon-insensitive loss
- **Scikit-learn Model**: RBF kernel SVM with hyperparameter tuning
- **Input Features**: 8 California housing features
- **Output**: House price in $100,000s (multiply by 100,000 for actual price)

## 🛠️ Technical Details

- **Backend**: Flask web framework
- **Frontend**: HTML5, CSS3, JavaScript
- **Models**: PyTorch and scikit-learn SVM
- **Data Processing**: NumPy, Pandas
- **Styling**: Modern gradient design with responsive layout

## 📁 File Structure

```
├── app.py                 # Main Flask application
├── templates/
│   └── index.html         # Web interface template
├── saved_models/          # Trained model files
│   ├── svm_housing_model.pth
│   └── best_svm_model_rbf.pkl
├── requirements.txt       # Python dependencies
└── FLASK_README.md       # This file
```

## 🔧 Troubleshooting

**Model not loading?**
- Ensure model files exist in `saved_models/` directory
- Check that all dependencies are installed

**Prediction errors?**
- Verify input values are within valid ranges
- Check browser console for JavaScript errors

**Port already in use?**
- Change the port in `app.py`: `app.run(port=5001)`

## 🌟 Example Usage

Visit the web interface and try these sample values:
- **Median Income**: 3.5 ($35,000)
- **House Age**: 25 years
- **Average Rooms**: 6.5
- **Average Bedrooms**: 1.2
- **Population**: 2,500
- **Average Occupancy**: 3.2
- **Latitude**: 34.05
- **Longitude**: -118.25

Expected prediction: Around $200,000-$300,000
