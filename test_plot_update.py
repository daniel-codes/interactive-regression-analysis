#!/usr/bin/env python3
"""
Test script to verify that the plot updates properly after model refitting.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from regression_analyzer import RegressionAnalyzer, create_sample_dataset

def test_plot_update_after_refit():
    """Test that the plot updates correctly after refitting models."""
    
    print("Creating sample dataset...")
    sample_data = create_sample_dataset()
    
    print("Initializing RegressionAnalyzer...")
    analyzer = RegressionAnalyzer(sample_data, 'target_value')
    
    print("Fitting regression models...")
    results = analyzer.fit_all_algorithms()
    
    print("Launching interactive 3D visualizer...")
    from visualizer import Interactive3DVisualizer
    visualizer = Interactive3DVisualizer(analyzer)
    
    # Test the plot update functionality
    print("\n=== Testing Plot Update After Refit ===")
    
    # Check initial state
    print(f"1. Initial model: {visualizer.model_var.get()}")
    print(f"2. Initial selected features: {visualizer.selected_features}")
    
    # Simulate deselecting some features
    print("3. Deselecting some features...")
    for feature in ['temperature', 'humidity', 'pressure']:
        if feature in visualizer.feature_vars:
            visualizer.feature_vars[feature].set(False)
    
    visualizer.update_selected_features()
    print(f"4. Selected features after deselection: {visualizer.selected_features}")
    
    # Test refitting
    print("5. Refitting models...")
    visualizer.refit_models()
    
    # Check if plot was updated
    print(f"6. Model after refit: {visualizer.model_var.get()}")
    print(f"7. Available models: {list(visualizer.analyzer.fitted_models.keys())}")
    
    # Test if plot updates when model changes
    print("8. Testing plot update with different model...")
    if len(visualizer.analyzer.fitted_models) > 1:
        # Try to set a different model
        models = list(visualizer.analyzer.fitted_models.keys())
        if 'Random Forest' in models:
            visualizer.model_var.set('Random Forest')
            print("9. Set model to Random Forest")
            visualizer.update_plot()
            print("10. Plot updated with Random Forest")
    
    print("\n=== Test Complete ===")
    print("Check the GUI to verify that the plot shows the correct model predictions.")
    
    # Keep the GUI open for manual testing
    visualizer.run()

if __name__ == "__main__":
    test_plot_update_after_refit()
