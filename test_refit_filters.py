#!/usr/bin/env python3
"""
Test script to verify that plot filters work after refitting models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from regression_analyzer import RegressionAnalyzer, create_sample_dataset

def test_refit_and_filters():
    """Test that plot filters work correctly after refitting models."""
    
    print("Creating sample dataset...")
    sample_data = create_sample_dataset()
    
    print("Initializing RegressionAnalyzer...")
    analyzer = RegressionAnalyzer(sample_data, 'target_value')
    
    print("Fitting regression models...")
    results = analyzer.fit_all_algorithms()
    
    print("Launching interactive 3D visualizer...")
    from visualizer import Interactive3DVisualizer
    visualizer = Interactive3DVisualizer(analyzer)
    
    # Test the refit functionality
    print("\n=== Testing Refit Functionality ===")
    print("1. Initial state - all features selected")
    print("2. Deselecting some features...")
    
    # Simulate deselecting some features
    for feature in ['temperature', 'humidity', 'pressure']:
        if feature in visualizer.feature_vars:
            visualizer.feature_vars[feature].set(False)
    
    visualizer.update_selected_features()
    print(f"Selected features after deselection: {visualizer.selected_features}")
    
    print("3. Refitting models...")
    visualizer.refit_models()
    
    print("4. Testing plot filters after refit...")
    # Test setting a filter
    if 'wind_speed' in visualizer.plot_filter_vars:
        visualizer.plot_filter_vars['wind_speed'].set("10.0")
        print("Set wind_speed filter to 10.0")
        
        # Test getting filtered data
        filtered_data = visualizer.get_filtered_data()
        print(f"Filtered data points: {len(filtered_data)}")
        
        # Test updating plot
        visualizer.update_plot()
        print("Plot updated successfully")
    
    print("\n=== Test Complete ===")
    print("If you see this message, the basic functionality is working.")
    print("Check the GUI to verify that plot filters work after refitting.")
    
    # Keep the GUI open for manual testing
    visualizer.run()

if __name__ == "__main__":
    test_refit_and_filters()
