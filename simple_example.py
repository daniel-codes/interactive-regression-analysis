#!/usr/bin/env python3
"""
Simple example showing how to use the regression analysis library.
This is the minimal code needed to get started.
"""

from regression_analyzer import RegressionAnalyzer, create_sample_dataset

# Step 1: Get your data (using built-in sample for this example)
print("Loading sample dataset...")
data = create_sample_dataset()
print(f"Dataset shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Step 2: Initialize the analyzer with your target column
print("\nInitializing regression analyzer...")
analyzer = RegressionAnalyzer(data, target_column='target_value')

# Step 3: Fit all regression models
print("Fitting regression models...")
results = analyzer.fit_all_algorithms()

# Step 4: See which models performed best
print("\nModel Performance Rankings:")
print("-" * 40)
for i, (model_name, r2_score) in enumerate(analyzer.model_scores.items(), 1):
    print(f"{i:2d}. {model_name:25s}: R² = {r2_score:.4f}")

# Step 5: Get the best model
best_model_name, best_model, is_scaled = analyzer.get_best_model()
print(f"\n🏆 Best model: {best_model_name}")
print(f"   R² Score: {analyzer.model_scores[best_model_name]:.4f}")

# Step 6: Launch interactive 3D visualizer
print(f"\nLaunching interactive 3D visualizer...")
print("In the GUI you can:")
print("- Select different X and Y features from dropdowns")
print("- Switch between different regression models")
print("- Rotate and zoom the 3D plot")
print("- View model comparison window")
print("\nClose the window when done exploring!")

analyzer.launch_interactive_visualizer()

print("\nDemo completed! 🎉")
