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
        self.root.geometry("1400x1000")
        
        # Fixed spatial coordinates for X/Y axes
        self.x_var = tk.StringVar(value="row")
        self.y_var = tk.StringVar(value="col")
        self.model_var = tk.StringVar(value=list(self.analyzer.fitted_models.keys())[0])
        
        # Feature selection for regression analysis (dynamic based on dataframe)
        self.feature_vars = {}
        self.selected_features = []
        
        # Initialize feature selection with all available analysis features
        print(f"Initializing feature selection for {len(self.analyzer.analysis_features)} features")
        for feature in self.analyzer.analysis_features:
            var = tk.BooleanVar(value=True)  # All features selected by default
            self.feature_vars[feature] = var
            self.selected_features.append(feature)
        
        print(f"Feature selection initialized: {self.selected_features}")
        
        # Feature filtering for plotting (dropdown filters for each feature)
        self.plot_filter_vars = {}
        self.plot_filter_values = {}
        self.initialize_plot_filters()
        
        self.setup_gui()
    
    def initialize_plot_filters(self):
        """Initialize plot filters with unique values from training data for each feature."""
        print("Initializing plot filters...")
        
        for feature in self.analyzer.analysis_features:
            # Get unique values from the feature, sorted
            unique_values = sorted(self.analyzer.df[feature].dropna().unique())
            
            # Create string variable for the dropdown selection
            var = tk.StringVar()
            
            # Set default to "All" (no filtering)
            var.set("All")
            
            # Store the variable and available values
            self.plot_filter_vars[feature] = var
            self.plot_filter_values[feature] = ["All"] + [str(val) for val in unique_values]
        
        print(f"Plot filters initialized for {len(self.plot_filter_vars)} features")
        
    def setup_gui(self):
        """Set up the GUI components."""
        # Create main frames with adjusted proportions
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=2)
        
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
        
        feature_frame = ttk.LabelFrame(parent, text="Feature Selection for Regression", padding=5)
        feature_frame.pack(side=tk.TOP, fill=tk.X, pady=3)
        
        # Feature filtering frame for plotting
        plot_filter_frame = ttk.LabelFrame(parent, text="Feature Selection for Plotting", padding=5)
        plot_filter_frame.pack(side=tk.TOP, fill=tk.X, pady=3)
        
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
        num_features = len(self.analyzer.analysis_features)
        ttk.Label(feature_frame, text=f"Select features to include in regression analysis ({num_features} available):").grid(row=0, column=0, columnspan=5, sticky='w', pady=(0, 5))
        
        # Add Select All / Deselect All buttons
        button_frame = ttk.Frame(feature_frame)
        button_frame.grid(row=1, column=0, columnspan=5, sticky='w', pady=(0, 5))
        
        ttk.Button(button_frame, text="Select All", command=self.select_all_features).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Deselect All", command=self.deselect_all_features).pack(side=tk.LEFT)
        
        # Create checkboxes for features in a flexible grid
        # Adjust columns based on number of features
        max_cols = min(5, max(3, num_features // 3 + 1))  # 3-5 columns based on feature count
        row_num = 2
        col_num = 0
        
        for i, feature in enumerate(self.analyzer.analysis_features):
            # Create a more readable feature name for display
            display_name = feature.replace('_', ' ').title()
            
            checkbox = ttk.Checkbutton(
                feature_frame, 
                text=f"{display_name}",
                variable=self.feature_vars[feature],
                command=lambda: self.update_selected_features()
            )
            checkbox.grid(row=row_num, column=col_num, padx=10, pady=2, sticky='w')
            
            col_num += 1
            if col_num >= max_cols:
                col_num = 0
                row_num += 1
        
        # Set up plot filtering controls
        self.setup_plot_filter_controls(plot_filter_frame)
    
    def setup_plot_filter_controls(self, parent):
        """Set up plot filtering dropdown controls."""
        # Title and description
        ttk.Label(parent, text="Filter data points and prediction surface by selecting specific feature values:").grid(
            row=0, column=0, columnspan=6, sticky='w', pady=(0, 5)
        )
        
        # Add Clear All Filters button
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=1, column=0, columnspan=6, sticky='w', pady=(0, 5))
        
        ttk.Button(button_frame, text="Clear All Filters", command=self.clear_all_plot_filters).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Apply Filters", command=self.update_plot).pack(side=tk.LEFT)
        
        # Create dropdown filters for each feature in a flexible grid
        num_features = len(self.analyzer.analysis_features)
        max_cols = min(3, max(2, num_features // 4 + 1))  # 2-3 columns based on feature count
        row_num = 2
        col_num = 0
        
        for i, feature in enumerate(self.analyzer.analysis_features):
            # Create a more readable feature name for display
            display_name = feature.replace('_', ' ').title()
            
            # Feature label
            label = ttk.Label(parent, text=f"{display_name}:")
            label.grid(row=row_num, column=col_num*2, padx=(10, 5), pady=5, sticky='w')
            
            # Feature dropdown
            dropdown = ttk.Combobox(
                parent,
                textvariable=self.plot_filter_vars[feature],
                values=self.plot_filter_values[feature],
                state='readonly',
                width=15
            )
            dropdown.grid(row=row_num, column=col_num*2+1, padx=(0, 15), pady=5, sticky='w')
            dropdown.bind('<<ComboboxSelected>>', lambda e: self.update_plot())
            
            col_num += 1
            if col_num >= max_cols:
                col_num = 0
                row_num += 1
    
    def setup_plot_area(self, parent):
        """Set up the matplotlib plot area."""
        self.figure = plt.Figure(figsize=(16, 12))
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
    
    def select_all_features(self):
        """Select all available features for regression analysis."""
        for var in self.feature_vars.values():
            var.set(True)
        self.update_selected_features()
        print("All features selected")
    
    def deselect_all_features(self):
        """Deselect all features for regression analysis."""
        for var in self.feature_vars.values():
            var.set(False)
        self.update_selected_features()
        print("All features deselected")
    
    def clear_all_plot_filters(self):
        """Clear all plot filters by setting them to 'All'."""
        for var in self.plot_filter_vars.values():
            var.set("All")
        print("All plot filters cleared")
        self.update_plot()
    
    def get_filtered_data(self):
        """Get data filtered based on current plot filter selections."""
        filtered_df = self.analyzer.df.copy()
        
        # Apply each filter
        for feature, var in self.plot_filter_vars.items():
            selected_value = var.get()
            if selected_value != "All":
                # Convert string back to appropriate type for comparison
                try:
                    # Try to convert to float first (for numeric values)
                    numeric_value = float(selected_value)
                    filtered_df = filtered_df[filtered_df[feature] == numeric_value]
                except ValueError:
                    # If conversion fails, treat as string
                    filtered_df = filtered_df[filtered_df[feature].astype(str) == selected_value]
        
        print(f"Filtered data: {len(filtered_df)} out of {len(self.analyzer.df)} rows")
        return filtered_df
    
    def calculate_filtered_metrics(self, filtered_data, model_name):
        """Calculate performance metrics based on filtered training data visible in the plot."""
        if len(filtered_data) == 0:
            # No filtered data - return None to indicate fallback should be used
            return None
        
        try:
            # Get features for prediction (same as used in training)
            X_filtered = filtered_data[self.analyzer.feature_columns]
            y_true = filtered_data[self.analyzer.target_column].values
            
            # Make predictions on the filtered data
            y_pred = self.analyzer.predict_with_model(model_name, X_filtered)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            import numpy as np
            
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            largest_diff = np.max(np.abs(y_true - y_pred))
            
            metrics = {
                'R2': r2,
                'MSE': mse,
                'MAE': mae,
                'Max_Diff': largest_diff,
                'N_Points': len(filtered_data)
            }
            
            print(f"Calculated filtered metrics for {len(filtered_data)} points: R²={r2:.4f}, MSE={mse:.2f}, MAE={mae:.2f}, Max Diff={largest_diff:.2f}")
            return metrics
            
        except Exception as e:
            print(f"Error calculating filtered metrics: {str(e)}")
            return None
    
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
            
            # Reinitialize plot filters with the new feature set
            self.initialize_plot_filters()
            
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
            
            # Get filtered data based on plot filter selections
            filtered_data = self.get_filtered_data()
            
            # Create 3D subplot
            ax = self.figure.add_subplot(111, projection='3d')
            
            # Plot actual data points only if filtered data exists
            scatter = None
            if len(filtered_data) > 0:
                # Get data from filtered dataset - spatial coordinates for X/Y, target for Z
                x_data = filtered_data[x_feature].values
                y_data = filtered_data[y_feature].values
                z_data = filtered_data[self.analyzer.target_column].values
                
                # Plot actual data points
                scatter = ax.scatter(x_data, y_data, z_data, c=z_data, cmap='viridis', alpha=0.6, s=50)
            else:
                # Add text to indicate no training points match the filters
                ax.text2D(0.02, 0.02, 'No training points match current filters', 
                         transform=ax.transAxes, fontsize=10, 
                         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # Always create prediction surface if model is available
            if model_name in self.analyzer.fitted_models:
                self.add_prediction_surface(ax, x_feature, y_feature, model_name, filtered_data)
            
            # Labels and title
            ax.set_xlabel('Row (Spatial X)', fontsize=12)
            ax.set_ylabel('Col (Spatial Y)', fontsize=12)
            ax.set_zlabel(f'{self.analyzer.target_column}', fontsize=12)
            
            # Create title with selected features and filter info
            selected_count = len(self.selected_features) if hasattr(self, 'selected_features') else len(self.analyzer.analysis_features)
            
            # Count active filters
            active_filters = sum(1 for var in self.plot_filter_vars.values() if var.get() != "All")
            filter_info = f" | Filters: {active_filters}" if active_filters > 0 else ""
            data_info = f" | Data: {len(filtered_data)}/{len(self.analyzer.df)} points"
            
            ax.set_title(f'Spatial 3D Regression Visualization\nModel: {model_name} | Features: {selected_count}{filter_info}{data_info}', fontsize=14)
            
            # Add colorbar only if we have scatter points
            if scatter is not None:
                self.figure.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
            
            # Model performance info - calculate metrics based on visible data
            metrics = self.calculate_filtered_metrics(filtered_data, model_name)
            
            if metrics is not None:
                # Use metrics calculated from visible training points
                metrics_text = (f"Plot Metrics (n={metrics['N_Points']}):\n"
                               f"R² = {metrics['R2']:.4f}\n"
                               f"MSE = {metrics['MSE']:.2f}\n"
                               f"MAE = {metrics['MAE']:.2f}\n"
                               f"Max Diff = {metrics['Max_Diff']:.2f}")
                bbox_color = 'lightgreen'
            elif model_name in self.analyzer.model_results:
                # Fallback to overall model metrics when no visible points
                results = self.analyzer.model_results[model_name]
                metrics_text = (f"Overall Model Metrics:\n"
                               f"R² = {results['R2_Score']:.4f}\n"
                               f"MSE = {results['MSE']:.2f}\n"
                               f"MAE = {results['MAE']:.2f}\n"
                               f"RMSE = {results['RMSE']:.2f}")
                bbox_color = 'wheat'
            else:
                # Final fallback to basic R2 score
                r2_score = self.analyzer.model_scores.get(model_name, 0.0)
                metrics_text = f"R² Score: {r2_score:.4f}"
                bbox_color = 'wheat'
            
            ax.text2D(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                     fontsize=10, verticalalignment='top', 
                     bbox=dict(boxstyle='round', facecolor=bbox_color, alpha=0.8))
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error updating plot: {str(e)}")
    
    def add_prediction_surface(self, ax, x_feature, y_feature, model_name, filtered_data):
        """Add prediction surface to the 3D plot based on filtered data."""
        try:
            # Create a grid for predictions based on data range
            if len(filtered_data) > 0:
                # Use filtered data range if data exists
                x_min, x_max = filtered_data[x_feature].min(), filtered_data[x_feature].max()
                y_min, y_max = filtered_data[y_feature].min(), filtered_data[y_feature].max()
                reference_data = filtered_data
            else:
                # Use full dataset range if no filtered data exists
                x_min, x_max = self.analyzer.df[x_feature].min(), self.analyzer.df[x_feature].max()
                y_min, y_max = self.analyzer.df[y_feature].min(), self.analyzer.df[y_feature].max()
                reference_data = self.analyzer.df
            
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
            
            # Prepare prediction data based on filtered dataset
            grid_size = xx.size  # Total number of grid points
            grid_points = pd.DataFrame()
            
            # Always include spatial coordinates varying across the grid
            grid_points[x_feature] = xx.ravel()
            grid_points[y_feature] = yy.ravel()
            
            # For all analysis features, set values based on filters or filtered data
            for feature in self.analyzer.analysis_features:
                if feature in self.plot_filter_vars:
                    filter_value = self.plot_filter_vars[feature].get()
                    if filter_value != "All":
                        # Use the specific filtered value for this feature
                        try:
                            numeric_value = float(filter_value)
                            grid_points[feature] = numeric_value
                        except ValueError:
                            # Handle string values - find the first matching numeric value from reference data
                            matching_values = reference_data[reference_data[feature].astype(str) == filter_value][feature]
                            if len(matching_values) > 0:
                                grid_points[feature] = matching_values.iloc[0]
                            else:
                                # Fallback to reference data median
                                grid_points[feature] = reference_data[feature].median()
                    else:
                        # No filter applied - use median from reference dataset
                        grid_points[feature] = reference_data[feature].median()
                else:
                    # Feature not in filters (shouldn't happen, but handle gracefully)
                    grid_points[feature] = reference_data[feature].median()
            
            # Ensure all required features are present for the model
            # The model expects all features that were used during training
            required_features = self.analyzer.feature_columns
            for col in required_features:
                if col not in grid_points.columns:
                    # Add missing column with median value from reference data
                    if col in reference_data.columns:
                        grid_points[col] = reference_data[col].median()
                    else:
                        # Fallback to original data median if column not in reference data
                        grid_points[col] = self.analyzer.df[col].median()
            
            # Reorder columns to match the training feature order
            grid_points = grid_points[required_features]
            
            if len(filtered_data) > 0:
                print(f"Generated prediction grid with {grid_size} points using filtered feature values (filtered data available)")
            else:
                print(f"Generated prediction grid with {grid_size} points using filtered feature values (no filtered data, using full spatial range)")
            print(f"Grid features: {list(grid_points.columns)}")
            
            # Make predictions
            predictions = self.analyzer.predict_with_model(model_name, grid_points)
            zz = predictions.reshape(xx.shape)
            
            # Plot surface with different appearance based on filtering
            active_filters = sum(1 for var in self.plot_filter_vars.values() if var.get() != "All")
            if active_filters > 0:
                # Use a different color/style when filters are active to make it more obvious
                ax.plot_surface(xx, yy, zz, alpha=0.4, color='orange', linewidth=0.5)
                print(f"Plotted filtered prediction surface with {active_filters} active filters")
            else:
                # Default appearance when no filters are active
                ax.plot_surface(xx, yy, zz, alpha=0.3, color='red', linewidth=0.5)
                print("Plotted unfiltered prediction surface")
            
        except Exception as e:
            print(f"Could not add prediction surface: {str(e)}")
    
    def show_model_comparison(self):
        """Show a comparison window with all model performances."""
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Model Performance Comparison")
        comparison_window.geometry("1000x600")
        
        # Create treeview for model comparison with more columns
        columns = ('Model', 'R² Score', 'CV R² Mean', 'CV R² Std', 'RMSE', 'MAE')
        tree = ttk.Treeview(comparison_window, columns=columns, show='headings')
        
        # Configure column headings and widths
        column_widths = {'Model': 200, 'R² Score': 120, 'CV R² Mean': 120, 'CV R² Std': 120, 'RMSE': 100, 'MAE': 100}
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=column_widths.get(col, 100))
        
        # Populate with detailed model results
        for model_name in self.analyzer.model_scores:
            # Get detailed results if available
            if hasattr(self.analyzer, 'model_results') and model_name in self.analyzer.model_results:
                results = self.analyzer.model_results[model_name]
                r2_score = results['R2_Score']
                cv_mean = results['CV_R2_Mean']
                cv_std = results['CV_R2_Std']
                rmse = results['RMSE']
                mae = results['MAE']
                
                tree.insert('', 'end', values=(
                    model_name, 
                    f'{r2_score:.4f}', 
                    f'{cv_mean:.4f}', 
                    f'{cv_std:.4f}',
                    f'{rmse:.2f}',
                    f'{mae:.2f}'
                ))
            else:
                # Fallback to basic score only
                r2_score = self.analyzer.model_scores[model_name]
                tree.insert('', 'end', values=(model_name, f'{r2_score:.4f}', 'N/A', 'N/A', 'N/A', 'N/A'))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(comparison_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack components
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
        
        # Add information label
        info_frame = ttk.Frame(comparison_window)
        info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ttk.Label(info_frame, text="R² Score: Coefficient of determination (higher is better)\nCV R² Mean/Std: 5-fold cross-validation R² statistics\nRMSE: Root Mean Square Error (lower is better)\nMAE: Mean Absolute Error (lower is better)", 
                 justify=tk.LEFT, font=('TkDefaultFont', 9)).pack(side=tk.LEFT)
        
        # Add close button
        ttk.Button(info_frame, text="Close", command=comparison_window.destroy).pack(side=tk.RIGHT)
    
    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()
