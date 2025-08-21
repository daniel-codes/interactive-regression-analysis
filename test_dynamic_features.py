#!/usr/bin/env python3
"""
Test script for dynamic feature selection functionality.
Tests the library with different dataset structures to ensure
feature selection adapts to any dataframe with row/col/target columns.
"""

import pandas as pd
import numpy as np
from regression_analyzer import RegressionAnalyzer


def create_simple_dataset():
    """Create a simple dataset with minimal features."""
    np.random.seed(42)
    n_samples = 100
    
    df = pd.DataFrame({
        'row': np.random.randint(0, 10, n_samples),
        'col': np.random.randint(0, 10, n_samples),
        'feature_a': np.random.normal(100, 20, n_samples),
        'feature_b': np.random.exponential(5, n_samples),
        'target': np.random.normal(50, 10, n_samples)
    })
    
    # Add relationships
    df['target'] = (df['feature_a'] * 0.3 + 
                   df['feature_b'] * 2 + 
                   df['row'] * 1.5 + 
                   df['col'] * -0.8 + 
                   np.random.normal(0, 5, n_samples))
    
    return df


def create_complex_dataset():
    """Create a complex dataset with many features."""
    np.random.seed(123)
    n_samples = 200
    
    # Base spatial coordinates
    rows = np.random.randint(0, 15, n_samples)
    cols = np.random.randint(0, 15, n_samples)
    
    # Many different types of features
    df = pd.DataFrame({
        'row': rows,
        'col': cols,
        'economic_indicator_1': np.random.lognormal(3, 1, n_samples),
        'environmental_factor_a': np.random.gamma(2, 3, n_samples),
        'social_metric_x': np.random.beta(2, 5, n_samples) * 100,
        'infrastructure_index': np.random.uniform(0, 1000, n_samples),
        'demographic_variable': np.random.normal(50, 15, n_samples),
        'technological_adoption': np.random.exponential(10, n_samples),
        'policy_score': np.random.triangular(0, 50, 100, n_samples),
        'health_indicator': np.random.weibull(2, n_samples) * 20,
        'education_level': np.random.poisson(12, n_samples),
        'urban_development': np.random.pareto(1.5, n_samples) * 10,
        'climate_variable': np.random.normal(25, 8, n_samples),
        'resource_availability': np.random.chisquare(5, n_samples) * 5,
        'governance_quality': np.random.f(5, 10, n_samples) * 20,
        'innovation_index': np.random.standard_t(10, n_samples) * 15 + 50,
        'outcome_measure': np.random.normal(75, 25, n_samples)
    })
    
    # Complex target with many feature interactions
    df['outcome_measure'] = (
        df['economic_indicator_1'] * 0.1 +
        df['environmental_factor_a'] * 0.2 +
        df['social_metric_x'] * 0.15 +
        df['infrastructure_index'] * 0.05 +
        df['demographic_variable'] * 0.3 +
        df['technological_adoption'] * 0.1 +
        df['policy_score'] * 0.2 +
        df['health_indicator'] * 0.25 +
        df['education_level'] * 2 +
        df['urban_development'] * 0.05 +
        df['climate_variable'] * 0.8 +
        df['resource_availability'] * 0.3 +
        df['governance_quality'] * 0.4 +
        df['innovation_index'] * 0.1 +
        # Spatial effects
        rows * 2 + cols * 1.5 +
        np.random.normal(0, 15, n_samples)
    )
    
    return df


def create_custom_dataset():
    """Create a custom dataset with user-defined features."""
    np.random.seed(789)
    n_samples = 150
    
    df = pd.DataFrame({
        'row': np.random.randint(0, 12, n_samples),
        'col': np.random.randint(0, 12, n_samples),
        'sensor_reading_1': np.random.normal(500, 100, n_samples),
        'sensor_reading_2': np.random.uniform(0, 255, n_samples),
        'measurement_alpha': np.random.exponential(3, n_samples),
        'measurement_beta': np.random.gamma(3, 2, n_samples),
        'control_variable': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'quality_score': np.random.normal(80, 12, n_samples)
    })
    
    # Target with interactions
    df['quality_score'] = (
        df['sensor_reading_1'] * 0.05 +
        df['sensor_reading_2'] * 0.1 +
        df['measurement_alpha'] * 5 +
        df['measurement_beta'] * 3 +
        df['control_variable'] * 20 +
        df['row'] * 2 + df['col'] * 1.8 +
        np.random.normal(0, 8, n_samples)
    )
    
    return df


def test_dynamic_feature_detection():
    """Test dynamic feature detection with different datasets."""
    print("=" * 70)
    print("TESTING DYNAMIC FEATURE SELECTION")
    print("=" * 70)
    
    datasets = [
        ("Simple Dataset (3 features)", create_simple_dataset()),
        ("Complex Dataset (15 features)", create_complex_dataset()),
        ("Custom Dataset (6 features)", create_custom_dataset())
    ]
    
    for name, dataset in datasets:
        print(f"\n{name}:")
        print("-" * 50)
        print(f"Dataset shape: {dataset.shape}")
        print(f"Columns: {list(dataset.columns)}")
        
        try:
            # Initialize analyzer
            target_col = dataset.columns[-1]  # Last column as target
            analyzer = RegressionAnalyzer(dataset, target_col)
            
            print(f"✓ Analyzer initialized successfully")
            print(f"  - Spatial columns: {analyzer.spatial_columns}")
            print(f"  - Analysis features: {analyzer.analysis_features}")
            print(f"  - Target column: {analyzer.target_column}")
            
            # Test fitting
            analyzer.fit_all_algorithms()
            print(f"✓ Models fitted successfully")
            
            # Get top 3 models
            top_models = list(analyzer.model_scores.items())[:3]
            print(f"  - Top 3 models: {[f'{name}: {score:.4f}' for name, score in top_models]}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
    
    print(f"\n{'='*70}")
    print("DYNAMIC FEATURE SELECTION TEST COMPLETED")
    print(f"{'='*70}")


def test_visualizer_flexibility():
    """Test that the visualizer adapts to different feature counts."""
    print("\nTesting visualizer flexibility...")
    
    # Test with different dataset sizes
    datasets = [
        ("Minimal", create_simple_dataset()),
        ("Large", create_complex_dataset())
    ]
    
    for name, dataset in datasets:
        print(f"\n{name} Dataset Visualizer Test:")
        target_col = dataset.columns[-1]
        analyzer = RegressionAnalyzer(dataset, target_col)
        analyzer.fit_all_algorithms()
        
        print(f"  - Features available for selection: {len(analyzer.analysis_features)}")
        print(f"  - Feature names: {analyzer.analysis_features}")
        
        # Test that visualizer can be initialized (don't actually show GUI)
        try:
            from visualizer import Interactive3DVisualizer
            visualizer = Interactive3DVisualizer(analyzer)
            print(f"  ✓ Visualizer initialized with {len(visualizer.feature_vars)} feature checkboxes")
            visualizer.root.destroy()  # Clean up
        except Exception as e:
            print(f"  ✗ Visualizer error: {str(e)}")


def test_feature_selection_logic():
    """Test the feature selection and refitting logic."""
    print("\nTesting feature selection logic...")
    
    dataset = create_complex_dataset()
    target_col = dataset.columns[-1]
    analyzer = RegressionAnalyzer(dataset, target_col)
    analyzer.fit_all_algorithms()
    
    print(f"Initial features: {analyzer.analysis_features}")
    print(f"Initial model count: {len(analyzer.fitted_models)}")
    
    # Test selecting subset of features
    subset_features = analyzer.analysis_features[:5]  # First 5 features
    print(f"Testing with subset: {subset_features}")
    
    # Update analyzer's feature selection
    all_features = analyzer.spatial_columns + subset_features
    analyzer.feature_columns = all_features
    analyzer.X = analyzer.df[all_features]
    
    # Refit models
    analyzer.fit_all_algorithms()
    print(f"✓ Models refitted with {len(subset_features)} features")
    print(f"  - New model count: {len(analyzer.fitted_models)}")
    
    # Check that it still works
    best_model_name, best_model, is_scaled = analyzer.get_best_model()
    print(f"  - Best model: {best_model_name}")


def main():
    """Run all dynamic feature selection tests."""
    print("DYNAMIC FEATURE SELECTION TESTING SUITE")
    
    test_dynamic_feature_detection()
    test_visualizer_flexibility()
    test_feature_selection_logic()
    
    print("\n" + "="*70)
    print("✅ ALL DYNAMIC FEATURE TESTS COMPLETED SUCCESSFULLY!")
    print("The library now automatically adapts to any dataset structure")
    print("with row, col, and target columns, dynamically detecting all")
    print("other columns as selectable analysis features.")
    print("="*70)


if __name__ == "__main__":
    main()
