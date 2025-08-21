# Interactive Regression Analysis Library

A comprehensive Python library that combines multiple scikit-learn regression algorithms with interactive 3D visualization using tkinter. This library allows you to quickly fit various regression models to your dataset and explore the relationships between features using an intuitive 3D plotting interface.

## Features

### Regression Algorithms Included
- **Linear Regression** - Basic linear regression
- **Ridge Regression** - L2 regularized linear regression
- **Lasso Regression** - L1 regularized linear regression  
- **Elastic Net** - Combined L1/L2 regularization
- **Polynomial Regression (degree=2)** - Quadratic feature expansion
- **Polynomial Regression (degree=3)** - Cubic feature expansion
- **Decision Tree** - Non-linear tree-based regression
- **Random Forest** - Ensemble of decision trees
- **Gradient Boosting** - Boosted ensemble method
- **XGBoost** - Extreme gradient boosting (high-performance)
- **Support Vector Regression (SVR)** - SVM for regression
- **K-Nearest Neighbors** - Instance-based regression

### Interactive Visualization
- **3D Scatter Plots** - Visualize data points in 3D space
- **Prediction Surfaces** - See model predictions as 3D surfaces
- **Feature Selection** - Choose X and Y axes from dropdown menus
- **Model Comparison** - Switch between different fitted models
- **Performance Metrics** - View R² scores and other metrics
- **Color-coded Points** - Data points colored by target values

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. For tkinter (usually included with Python, but if needed):
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS (with Homebrew)
brew install python-tk

# Windows - usually included with Python installation
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from regression_analyzer import RegressionAnalyzer

# Load your data
df = pd.read_csv('your_data.csv')

# Initialize the analyzer
analyzer = RegressionAnalyzer(df, target_column='your_target_column')

# Fit all regression algorithms
results = analyzer.fit_all_algorithms()

# Launch interactive 3D visualizer
analyzer.launch_interactive_visualizer()
```

### Using Sample Data

```python
from regression_analyzer import RegressionAnalyzer, create_sample_dataset

# Create sample dataset
sample_data = create_sample_dataset()

# Initialize analyzer
analyzer = RegressionAnalyzer(sample_data, 'target_value')

# Fit models and launch visualizer
analyzer.fit_all_algorithms()
analyzer.launch_interactive_visualizer()
```

## Demo Script

Run the included demo script to see the library in action:

```bash
python demo.py
```

The demo includes:
1. **Sample Dataset Demo** - Built-in synthetic dataset
2. **Housing Dataset Demo** - Custom housing price prediction dataset
3. **CSV Loading Demo** - Shows how to load data from files

## API Reference

### RegressionAnalyzer Class

#### Constructor
```python
RegressionAnalyzer(dataframe, target_column)
```
- `dataframe`: pandas DataFrame containing your data
- `target_column`: string name of the column to predict

#### Methods

##### `fit_all_algorithms(test_size=0.2, random_state=42)`
Fits all available regression algorithms and evaluates their performance.
- Returns: Dictionary containing performance metrics for each model

##### `get_best_model()`
Returns the best performing model based on R² score.
- Returns: Tuple of (model_name, model_object, is_scaled)

##### `predict_with_model(model_name, X_new)`
Make predictions using a specific fitted model.
- `model_name`: Name of the model to use
- `X_new`: New data to predict on
- Returns: Array of predictions

##### `launch_interactive_visualizer()`
Launches the interactive 3D visualization GUI.

### Interactive3DVisualizer Class

The visualizer provides:
- **Feature Selection Dropdowns** - Choose X and Y axis features
- **Model Selection** - Switch between fitted models
- **3D Plot** - Interactive 3D scatter plot with prediction surface
- **Model Comparison Window** - Compare all model performances

## Data Requirements

- Input must be a pandas DataFrame
- Target column must be numeric
- At least 2 numeric feature columns are required
- Non-numeric columns are automatically excluded from analysis

## Performance Metrics

The library evaluates models using:
- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Squared Error  
- **MAE** - Mean Absolute Error
- **R² Score** - Coefficient of determination
- **Cross-Validation R²** - 5-fold cross-validated R² score

## GUI Controls

### Main Interface
- **X-Axis Feature** - Select feature for X axis
- **Y-Axis Feature** - Select feature for Y axis  
- **Model** - Choose which fitted model to visualize
- **Refresh Plot** - Update the 3D visualization
- **Show Model Comparison** - Open performance comparison window

### 3D Plot Features
- **Rotate** - Click and drag to rotate the 3D plot
- **Zoom** - Use mouse wheel to zoom in/out
- **Pan** - Right-click and drag to pan
- **Data Points** - Colored by target value using viridis colormap
- **Prediction Surface** - Semi-transparent red surface showing model predictions

## Example Datasets

### Built-in Sample Dataset
The `create_sample_dataset()` function generates synthetic data with:
- Temperature, humidity, pressure, and random features
- Target variable with realistic relationships
- 300 data points with added noise

### Custom Dataset Example
```python
import pandas as pd
import numpy as np

# Create custom dataset
np.random.seed(42)
n = 200

data = pd.DataFrame({
    'feature1': np.random.normal(10, 2, n),
    'feature2': np.random.uniform(0, 100, n), 
    'feature3': np.random.exponential(5, n),
    'target': np.random.normal(50, 10, n)
})

# Add realistic relationships
data['target'] = (data['feature1'] * 2 + 
                  data['feature2'] * 0.5 + 
                  data['feature3'] * -1 + 
                  np.random.normal(0, 5, n))

analyzer = RegressionAnalyzer(data, 'target')
```

## Troubleshooting

### Common Issues

1. **"Target column not found"**
   - Check that the target column name exactly matches a column in your DataFrame

2. **"Need at least 2 numeric feature columns"**
   - Ensure your DataFrame has at least 2 numeric columns besides the target

3. **tkinter import errors**
   - Install tkinter using your system package manager
   - On some Linux distributions, tkinter needs to be installed separately

4. **Plot not updating**
   - Click "Refresh Plot" button
   - Check that different features are selected for X and Y axes

5. **Model fitting errors**
   - Check for missing values in your data
   - Ensure sufficient data points (recommend >50 samples)
   - Check for infinite or very large values

### Performance Tips

- For large datasets (>10,000 rows), consider sampling for visualization
- SVR can be slow on large datasets - consider excluding it for quick analysis
- Use cross-validation results for better model comparison on small datasets

## Contributing

Feel free to contribute by:
- Adding new regression algorithms
- Improving the visualization interface
- Adding new plot types
- Enhancing performance metrics
- Adding data preprocessing options

## License

This project is open source and available under the MIT License.
