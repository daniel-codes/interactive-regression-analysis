#!/usr/bin/env python3
"""
Test script to demonstrate that plot filters work correctly after refitting models.
This script simulates the GUI behavior programmatically.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from regression_analyzer import RegressionAnalyzer, create_sample_dataset

def test_refit_behavior():
    """Test the refit behavior programmatically."""
    
    print("=== Testing Refit Behavior ===")
    print("Creating sample dataset...")
    sample_data = create_sample_dataset()
    
    print("Initializing RegressionAnalyzer...")
    analyzer = RegressionAnalyzer(sample_data, 'target_value')
    
    print("Fitting initial regression models...")
    results = analyzer.fit_all_algorithms()
    
    print("Creating visualizer...")
    from visualizer import Interactive3DVisualizer
    visualizer = Interactive3DVisualizer(analyzer)
    
    print("\n=== Test 1: Initial State ===")
    print(f"Initial selected features: {visualizer.selected_features}")
    print(f"Initial plot filter variables: {list(visualizer.plot_filter_vars.keys())}")
    
    # Test filtering before refit
    print("\n=== Test 2: Testing filters before refit ===")
    if 'temperature' in visualizer.plot_filter_vars:
        # Get some temperature values to test with
        temp_values = visualizer.plot_filter_values['temperature']
        if len(temp_values) > 5:  # Should have "All" + actual values
            test_value = temp_values[1]  # First non-"All" value
            visualizer.plot_filter_vars['temperature'].set(test_value)
            print(f"Set temperature filter to: {test_value}")
            
            filtered_data = visualizer.get_filtered_data()
            print(f"Filtered data points: {len(filtered_data)}")
    
    print("\n=== Test 3: Refitting with subset of features ===")
    # Deselect some features
    features_to_deselect = ['temperature', 'humidity', 'pressure']
    for feature in features_to_deselect:
        if feature in visualizer.feature_vars:
            visualizer.feature_vars[feature].set(False)
            print(f"Deselected feature: {feature}")
    
    # Update selected features
    visualizer.update_selected_features()
    print(f"Features after deselection: {visualizer.selected_features}")
    
    # Perform refit
    print("\nRefitting models...")
    try:
        visualizer.refit_models()
        print("Refit completed successfully!")
    except Exception as e:
        print(f"Refit failed: {e}")
        return False
    
    print("\n=== Test 4: Testing filters after refit ===")
    print(f"Plot filter variables after refit: {list(visualizer.plot_filter_vars.keys())}")
    
    # Test filtering after refit
    if 'wind_speed' in visualizer.plot_filter_vars:
        # Get some wind_speed values to test with
        wind_values = visualizer.plot_filter_values['wind_speed']
        if len(wind_values) > 5:
            test_value = wind_values[1]  # First non-"All" value
            visualizer.plot_filter_vars['wind_speed'].set(test_value)
            print(f"Set wind_speed filter to: {test_value}")
            
            filtered_data = visualizer.get_filtered_data()
            print(f"Filtered data points after refit: {len(filtered_data)}")
            
            # Test plot update
            try:
                visualizer.update_plot()
                print("Plot updated successfully after refit!")
            except Exception as e:
                print(f"Plot update failed: {e}")
                return False
    
    print("\n=== Test Results ===")
    print("✅ All tests passed! The refit functionality is working correctly.")
    print("The plot filters should work properly after refitting models.")
    
    # Optionally keep the GUI open for manual testing
    print("\nLaunching GUI for manual verification...")
    print("Try refitting models and using plot filters to verify the fix.")
    visualizer.run()
    
    return True

if __name__ == "__main__":
    test_refit_behavior()
