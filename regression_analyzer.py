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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, messagebox
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
        
        if len(self.feature_columns) < 2:
            raise ValueError("Need at least 2 numeric feature columns for analysis")
        
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
        
        visualizer = Interactive3DVisualizer(self)
        visualizer.run()


class Interactive3DVisualizer:
    """
    Interactive 3D visualization component using tkinter and matplotlib.
    """
    
    def __init__(self, analyzer):
        """
        Initialize the visualizer with a RegressionAnalyzer instance.
        
        Args:
            analyzer (RegressionAnalyzer): The analyzer instance to visualize
        """
        self.analyzer = analyzer
        self.root = tk.Tk()
        self.root.title("Interactive Regression Analysis - 3D Visualization")
        self.root.geometry("1200x800")
        
        # Variables for dropdowns
        self.x_var = tk.StringVar(value=self.analyzer.feature_columns[0])
        self.y_var = tk.StringVar(value=self.analyzer.feature_columns[1] if len(self.analyzer.feature_columns) > 1 else self.analyzer.feature_columns[0])
        self.model_var = tk.StringVar(value=list(self.analyzer.fitted_models.keys())[0])
        
        self.setup_gui()
        
    def setup_gui(self):
        """Set up the GUI components."""
        # Create main frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Control panel
        self.setup_controls(control_frame)
        
        # Plot area
        self.setup_plot_area(plot_frame)
        
        # Initial plot
        self.update_plot()
    
    def setup_controls(self, parent):
        """Set up the control panel."""
        # Feature selection
        ttk.Label(parent, text="X-Axis Feature:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        x_dropdown = ttk.Combobox(parent, textvariable=self.x_var, values=self.analyzer.feature_columns, state='readonly')
        x_dropdown.grid(row=0, column=1, padx=5, pady=5)
        x_dropdown.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        
        ttk.Label(parent, text="Y-Axis Feature:").grid(row=0, column=2, padx=5, pady=5, sticky='w')
        y_dropdown = ttk.Combobox(parent, textvariable=self.y_var, values=self.analyzer.feature_columns, state='readonly')
        y_dropdown.grid(row=0, column=3, padx=5, pady=5)
        y_dropdown.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        
        ttk.Label(parent, text="Model:").grid(row=0, column=4, padx=5, pady=5, sticky='w')
        model_dropdown = ttk.Combobox(parent, textvariable=self.model_var, values=list(self.analyzer.fitted_models.keys()), state='readonly')
        model_dropdown.grid(row=0, column=5, padx=5, pady=5)
        model_dropdown.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        
        # Buttons
        ttk.Button(parent, text="Refresh Plot", command=self.update_plot).grid(row=0, column=6, padx=5, pady=5)
        ttk.Button(parent, text="Show Model Comparison", command=self.show_model_comparison).grid(row=0, column=7, padx=5, pady=5)
    
    def setup_plot_area(self, parent):
        """Set up the matplotlib plot area."""
        self.figure = plt.Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_plot(self):
        """Update the 3D plot with current selections."""
        try:
            self.figure.clear()
            
            x_feature = self.x_var.get()
            y_feature = self.y_var.get()
            model_name = self.model_var.get()
            
            if x_feature == y_feature:
                messagebox.showwarning("Warning", "X and Y features should be different for better visualization")
            
            # Get data
            x_data = self.analyzer.df[x_feature].values
            y_data = self.analyzer.df[y_feature].values
            z_data = self.analyzer.df[self.analyzer.target_column].values
            
            # Create 3D subplot
            ax = self.figure.add_subplot(111, projection='3d')
            
            # Plot actual data points
            scatter = ax.scatter(x_data, y_data, z_data, c=z_data, cmap='viridis', alpha=0.6, s=50)
            
            # Create prediction surface if model is available
            if model_name in self.analyzer.fitted_models:
                self.add_prediction_surface(ax, x_feature, y_feature, model_name)
            
            # Labels and title
            ax.set_xlabel(f'{x_feature}', fontsize=12)
            ax.set_ylabel(f'{y_feature}', fontsize=12)
            ax.set_zlabel(f'{self.analyzer.target_column}', fontsize=12)
            ax.set_title(f'3D Regression Visualization\nModel: {model_name}', fontsize=14)
            
            # Add colorbar
            self.figure.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
            
            # Model performance info
            if model_name in self.analyzer.model_scores:
                r2_score = self.analyzer.model_scores[model_name]
                ax.text2D(0.02, 0.98, f'R² Score: {r2_score:.4f}', transform=ax.transAxes, 
                         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error updating plot: {str(e)}")
    
    def add_prediction_surface(self, ax, x_feature, y_feature, model_name):
        """Add prediction surface to the 3D plot."""
        try:
            # Create a grid for predictions
            x_min, x_max = self.analyzer.df[x_feature].min(), self.analyzer.df[x_feature].max()
            y_min, y_max = self.analyzer.df[y_feature].min(), self.analyzer.df[y_feature].max()
            
            # Extend the range slightly
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min -= 0.1 * x_range
            x_max += 0.1 * x_range
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range
            
            # Create grid
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 20),
                np.linspace(y_min, y_max, 20)
            )
            
            # Prepare prediction data
            grid_points = pd.DataFrame()
            for col in self.analyzer.feature_columns:
                if col == x_feature:
                    grid_points[col] = xx.ravel()
                elif col == y_feature:
                    grid_points[col] = yy.ravel()
                else:
                    # Use median value for other features
                    grid_points[col] = self.analyzer.df[col].median()
            
            # Make predictions
            predictions = self.analyzer.predict_with_model(model_name, grid_points)
            zz = predictions.reshape(xx.shape)
            
            # Plot surface
            ax.plot_surface(xx, yy, zz, alpha=0.3, color='red', linewidth=0.5)
            
        except Exception as e:
            print(f"Could not add prediction surface: {str(e)}")
    
    def show_model_comparison(self):
        """Show a comparison window with all model performances."""
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Model Performance Comparison")
        comparison_window.geometry("800x600")
        
        # Create treeview for model comparison
        columns = ('Model', 'R² Score', 'CV R² Mean', 'CV R² Std')
        tree = ttk.Treeview(comparison_window, columns=columns, show='headings')
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        # Populate with model scores
        for model_name in self.analyzer.model_scores:
            r2_score = self.analyzer.model_scores[model_name]
            tree.insert('', 'end', values=(model_name, f'{r2_score:.4f}', 'N/A', 'N/A'))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add close button
        ttk.Button(comparison_window, text="Close", command=comparison_window.destroy).pack(pady=10)
    
    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()


def create_sample_dataset():
    """
    Create a sample dataset for demonstration purposes.
    
    Returns:
        pd.DataFrame: Sample dataset with multiple features and a target variable
    """
    np.random.seed(42)
    n_samples = 300
    
    # Generate features
    feature1 = np.random.normal(50, 15, n_samples)  # Temperature
    feature2 = np.random.normal(30, 10, n_samples)  # Humidity
    feature3 = np.random.normal(25, 5, n_samples)   # Pressure
    feature4 = np.random.uniform(0, 100, n_samples) # Random feature
    
    # Create target with some relationship to features
    noise = np.random.normal(0, 5, n_samples)
    target = (0.5 * feature1 + 0.3 * feature2 + 0.2 * feature3 + 
              0.1 * feature4 + noise + 10)
    
    # Create DataFrame
    df = pd.DataFrame({
        'temperature': feature1,
        'humidity': feature2,
        'pressure': feature3,
        'random_feature': feature4,
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
