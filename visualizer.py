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
        
        # Fixed spatial coordinates for X/Y axes
        self.x_var = tk.StringVar(value="row")
        self.y_var = tk.StringVar(value="col")
        self.model_var = tk.StringVar(value=list(self.analyzer.fitted_models.keys())[0])
        
        # Feature selection for regression analysis
        self.feature_vars = {}
        self.selected_features = []
        for feature in self.analyzer.analysis_features:
            var = tk.BooleanVar(value=True)  # All features selected by default
            self.feature_vars[feature] = var
            self.selected_features.append(feature)
        
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
        # Create main control frame and feature selection frame
        main_controls = ttk.Frame(parent)
        main_controls.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        feature_frame = ttk.LabelFrame(parent, text="Feature Selection for Regression", padding=10)
        feature_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Fixed spatial coordinates (read-only)
        ttk.Label(main_controls, text="X-Axis (Fixed):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        x_label = ttk.Label(main_controls, text="row", relief="sunken", width=10)
        x_label.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(main_controls, text="Y-Axis (Fixed):").grid(row=0, column=2, padx=5, pady=5, sticky='w')
        y_label = ttk.Label(main_controls, text="col", relief="sunken", width=10)
        y_label.grid(row=0, column=3, padx=5, pady=5)
        
        # Model selection
        ttk.Label(main_controls, text="Model:").grid(row=0, column=4, padx=5, pady=5, sticky='w')
        model_dropdown = ttk.Combobox(main_controls, textvariable=self.model_var, values=list(self.analyzer.fitted_models.keys()), state='readonly')
        model_dropdown.grid(row=0, column=5, padx=5, pady=5)
        model_dropdown.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
        
        # Buttons
        ttk.Button(main_controls, text="Refresh Plot", command=self.update_plot).grid(row=0, column=6, padx=5, pady=5)
        ttk.Button(main_controls, text="Refit Models", command=self.refit_models).grid(row=0, column=7, padx=5, pady=5)
        ttk.Button(main_controls, text="Model Comparison", command=self.show_model_comparison).grid(row=0, column=8, padx=5, pady=5)
        
        # Feature selection checkboxes
        ttk.Label(feature_frame, text="Select features to include in regression analysis:").grid(row=0, column=0, columnspan=4, sticky='w', pady=(0, 10))
        
        # Create checkboxes for features in a grid
        row_num = 1
        col_num = 0
        for i, feature in enumerate(self.analyzer.analysis_features):
            checkbox = ttk.Checkbutton(
                feature_frame, 
                text=feature, 
                variable=self.feature_vars[feature],
                command=lambda: self.update_selected_features()
            )
            checkbox.grid(row=row_num, column=col_num, padx=10, pady=2, sticky='w')
            
            col_num += 1
            if col_num >= 4:  # 4 columns
                col_num = 0
                row_num += 1
    
    def setup_plot_area(self, parent):
        """Set up the matplotlib plot area."""
        self.figure = plt.Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_selected_features(self):
        """Update the list of selected features for regression analysis."""
        self.selected_features = []
        for feature, var in self.feature_vars.items():
            if var.get():
                self.selected_features.append(feature)
        
        # Always include spatial coordinates
        all_features = self.analyzer.spatial_columns + self.selected_features
        print(f"Selected features: {all_features}")
    
    def refit_models(self):
        """Refit all models with the currently selected features."""
        if len(self.selected_features) == 0:
            messagebox.showwarning("Warning", "Please select at least one feature for regression analysis")
            return
        
        # Update analyzer's feature selection
        all_features = self.analyzer.spatial_columns + self.selected_features
        self.analyzer.feature_columns = all_features
        self.analyzer.X = self.analyzer.df[all_features]
        
        # Refit all algorithms
        print("Refitting models with selected features...")
        try:
            self.analyzer.fit_all_algorithms()
            
            # Update model dropdown with new models
            model_dropdown = None
            for child in self.root.winfo_children():
                for grandchild in child.winfo_children():
                    if hasattr(grandchild, 'winfo_children'):
                        for control in grandchild.winfo_children():
                            if isinstance(control, ttk.Combobox) and 'model' in str(control).lower():
                                control['values'] = list(self.analyzer.fitted_models.keys())
                                self.model_var.set(list(self.analyzer.fitted_models.keys())[0])
                                break
            
            self.update_plot()
            messagebox.showinfo("Success", f"Models refitted with {len(self.selected_features)} features")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error refitting models: {str(e)}")
    
    def update_plot(self):
        """Update the 3D plot with current selections."""
        try:
            self.figure.clear()
            
            # Fixed spatial coordinates
            x_feature = "row"
            y_feature = "col"
            model_name = self.model_var.get()
            
            # Get data - spatial coordinates for X/Y, target for Z
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
            ax.set_xlabel('Row (Spatial X)', fontsize=12)
            ax.set_ylabel('Col (Spatial Y)', fontsize=12)
            ax.set_zlabel(f'{self.analyzer.target_column}', fontsize=12)
            
            # Create title with selected features info
            selected_count = len(self.selected_features) if hasattr(self, 'selected_features') else len(self.analyzer.analysis_features)
            ax.set_title(f'Spatial 3D Regression Visualization\nModel: {model_name} | Features: {selected_count}', fontsize=14)
            
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
