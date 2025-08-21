"""
Interactive 3D Visualization Module

This module contains the Interactive3DVisualizer class that provides 
3D visualization capabilities for regression analysis using tkinter and matplotlib.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox


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
