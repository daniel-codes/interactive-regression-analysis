"""
Interactive Regression Analysis Library

A Python library that provides sklearn regression algorithms with interactive 3D visualization
using tkinter for feature selection and data exploration.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class RegressionAnalyzer:
    """
    A comprehensive regression analysis tool that fits multiple sklearn algorithms
    and provides interactive 3D visualization capabilities.
    """
    
    def __init__(self, dataframe, target_column):
        """
        Initialize the RegressionAnalyzer with a dataframe and target column.
        
        Args:
            dataframe (pd.DataFrame): Input dataset
            target_column (str): Name of the target column to predict
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        self.df = dataframe.copy()
        self.target_column = target_column
        self.feature_columns = [col for col in self.df.columns if col != target_column]
        
        # Remove non-numeric columns for regression
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = [col for col in self.feature_columns if col in numeric_cols]
        
        # Check for required spatial coordinates
        if 'row' not in self.feature_columns or 'col' not in self.feature_columns:
            raise ValueError("Dataset must contain 'row' and 'col' columns for spatial visualization")
        
        # Dynamically identify analysis features (all columns except row, col, and target)
        self.spatial_columns = ['row', 'col']
        excluded_columns = self.spatial_columns + [target_column]
        self.analysis_features = [col for col in self.feature_columns if col not in excluded_columns]
        
        if len(self.analysis_features) < 1:
            raise ValueError("Need at least 1 analysis feature column (excluding 'row', 'col', and target)")
        
        print(f"Identified {len(self.analysis_features)} analysis features: {self.analysis_features}")
        print(f"Spatial columns: {self.spatial_columns}")
        print(f"Target column: {target_column}")
        
        self.X = self.df[self.feature_columns]
        self.y = self.df[target_column]
        
        # Initialize regression algorithms
        self.algorithms = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'Polynomial Regression (degree=2)': Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('linear', LinearRegression())
            ]),
            'Polynomial Regression (degree=3)': Pipeline([
                ('poly', PolynomialFeatures(degree=3, include_bias=False)),
                ('linear', LinearRegression())
            ]),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42, eval_metric='rmse'),
            'Support Vector Regression': SVR(kernel='rbf'),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
        }
        
        self.fitted_models = {}
        self.model_scores = {}
        self.scaler = StandardScaler()
        
    def fit_all_algorithms(self, test_size=0.2, random_state=42):
        """
        Fit all regression algorithms and evaluate their performance.
        
        Args:
            test_size (float): Proportion of dataset for testing
            random_state (int): Random state for reproducibility
            
        Returns:
            dict: Dictionary containing model performance metrics
        """
        print("Fitting all regression algorithms...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, algorithm in self.algorithms.items():
            try:
                print(f"  Fitting {name}...")
                
                # Fit the model
                if name in ['Support Vector Regression']:
                    # SVR works better with scaled data
                    algorithm.fit(X_train_scaled, y_train)
                    y_pred = algorithm.predict(X_test_scaled)
                    self.fitted_models[name] = (algorithm, True)  # True indicates scaled
                elif 'XGBoost' in name:
                    # XGBoost can handle unscaled data and has specific parameters
                    algorithm.fit(X_train, y_train, verbose=False)
                    y_pred = algorithm.predict(X_test)
                    self.fitted_models[name] = (algorithm, False)  # False indicates not scaled
                else:
                    # All other algorithms including polynomial regression
                    algorithm.fit(X_train, y_train)
                    y_pred = algorithm.predict(X_test)
                    self.fitted_models[name] = (algorithm, False)  # False indicates not scaled
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                if name in ['Support Vector Regression']:
                    cv_scores = cross_val_score(algorithm, X_train_scaled, y_train, cv=5, scoring='r2')
                elif 'XGBoost' in name:
                    # XGBoost cross-validation with reduced verbosity
                    cv_scores = cross_val_score(algorithm, X_train, y_train, cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(algorithm, X_train, y_train, cv=5, scoring='r2')
                
                results[name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2_Score': r2,
                    'CV_R2_Mean': cv_scores.mean(),
                    'CV_R2_Std': cv_scores.std()
                }
                
                self.model_scores[name] = r2
                
            except Exception as e:
                print(f"  Error fitting {name}: {str(e)}")
                results[name] = {
                    'Error': str(e)
                }
        
        # Sort models by R2 score
        self.model_scores = dict(sorted(self.model_scores.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        print("Model fitting complete!")
        return results
    
    def get_best_model(self):
        """
        Get the best performing model based on R2 score.
        
        Returns:
            tuple: (model_name, model_object, is_scaled)
        """
        if not self.model_scores:
            raise ValueError("No models have been fitted. Call fit_all_algorithms() first.")
        
        best_model_name = list(self.model_scores.keys())[0]
        model, is_scaled = self.fitted_models[best_model_name]
        return best_model_name, model, is_scaled
    
    def predict_with_model(self, model_name, X_new):
        """
        Make predictions using a specific model.
        
        Args:
            model_name (str): Name of the model to use
            X_new (array-like): New data to predict on
            
        Returns:
            array: Predictions
        """
        if model_name not in self.fitted_models:
            raise ValueError(f"Model '{model_name}' has not been fitted")
        
        model, is_scaled = self.fitted_models[model_name]
        
        if is_scaled:
            X_new_scaled = self.scaler.transform(X_new)
            return model.predict(X_new_scaled)
        else:
            # Handle XGBoost and other models
            return model.predict(X_new)
    
    def launch_interactive_visualizer(self):
        """
        Launch the interactive 3D visualization GUI.
        """
        if not self.fitted_models:
            print("No models fitted. Fitting all algorithms first...")
            self.fit_all_algorithms()
        
        from visualizer import Interactive3DVisualizer
        visualizer = Interactive3DVisualizer(self)
        visualizer.run()



def create_sample_dataset():
    """
    Create a sample dataset for demonstration purposes with row/col spatial features.
    
    Returns:
        pd.DataFrame: Sample dataset with row/col coordinates, multiple features and a target variable
    """
    np.random.seed(42)
    n_samples = 300
    
    # Create spatial grid coordinates
    grid_size = int(np.sqrt(n_samples)) + 1  # Approximately square grid
    rows = np.random.randint(0, grid_size, n_samples)
    cols = np.random.randint(0, grid_size, n_samples)
    
    # Generate 10 additional features with realistic names and relationships
    temperature = np.random.normal(50, 15, n_samples)
    humidity = np.random.normal(30, 10, n_samples)
    pressure = np.random.normal(25, 5, n_samples)
    wind_speed = np.random.exponential(10, n_samples)
    elevation = np.random.uniform(0, 1000, n_samples)
    soil_ph = np.random.normal(6.5, 1.0, n_samples)
    rainfall = np.random.gamma(2, 5, n_samples)
    vegetation_index = np.random.beta(2, 2, n_samples)
    population_density = np.random.lognormal(2, 1, n_samples)
    industrial_index = np.random.uniform(0, 100, n_samples)
    
    # Add spatial correlation to some features based on row/col position
    # This makes the spatial visualization more interesting
    spatial_factor = 0.1
    temperature += spatial_factor * (rows + cols)
    humidity -= spatial_factor * np.abs(rows - cols)
    elevation += spatial_factor * (rows * cols) / grid_size
    
    # Create target with complex relationships including spatial effects
    noise = np.random.normal(0, 8, n_samples)
    target = (0.3 * temperature + 
              0.2 * humidity + 
              0.15 * pressure + 
              0.1 * wind_speed +
              0.05 * elevation / 100 +
              0.1 * soil_ph * 10 +
              0.05 * rainfall +
              0.1 * vegetation_index * 50 +
              0.02 * np.log(population_density + 1) * 10 +
              0.03 * industrial_index +
              # Spatial effects
              0.05 * (rows + cols) +
              0.02 * np.sin(rows / grid_size * 2 * np.pi) * 20 +
              0.02 * np.cos(cols / grid_size * 2 * np.pi) * 20 +
              noise + 50)
    
    # Create DataFrame with row/col as first features
    df = pd.DataFrame({
        'row': rows,
        'col': cols,
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'elevation': elevation,
        'soil_ph': soil_ph,
        'rainfall': rainfall,
        'vegetation_index': vegetation_index,
        'population_density': population_density,
        'industrial_index': industrial_index,
        'target_value': target
    })
    
    return df


if __name__ == "__main__":
    # Demo usage
    print("Creating sample dataset...")
    sample_data = create_sample_dataset()
    
    print("Initializing RegressionAnalyzer...")
    analyzer = RegressionAnalyzer(sample_data, 'target_value')
    
    print("Fitting regression models...")
    results = analyzer.fit_all_algorithms()
    
    print("\nModel Performance Summary:")
    print("-" * 50)
    for model, score in analyzer.model_scores.items():
        print(f"{model:25}: R² = {score:.4f}")
    
    print(f"\nBest Model: {analyzer.get_best_model()[0]}")
    
    print("\nLaunching interactive 3D visualizer...")
    analyzer.launch_interactive_visualizer()
