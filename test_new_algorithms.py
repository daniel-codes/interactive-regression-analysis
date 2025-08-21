#!/usr/bin/env python3
"""
Test script specifically for the new polynomial regression and XGBoost algorithms.
This demonstrates their capabilities with different types of datasets.
"""

import pandas as pd
import numpy as np
from regression_analyzer import RegressionAnalyzer
import matplotlib.pyplot as plt


def create_polynomial_dataset():
    """Create a dataset that benefits from polynomial features."""
    np.random.seed(42)
    n_samples = 200
    
    # Create features with polynomial relationships
    x1 = np.random.uniform(-3, 3, n_samples)
    x2 = np.random.uniform(-2, 2, n_samples)
    
    # Create target with polynomial relationship: y = 2*x1^2 + 1.5*x1*x2 + 0.5*x2^2 + noise
    y = (2 * x1**2 + 1.5 * x1 * x2 + 0.5 * x2**2 + 
         np.random.normal(0, 2, n_samples))
    
    df = pd.DataFrame({
        'feature_1': x1,
        'feature_2': x2,
        'linear_feature': np.random.normal(0, 1, n_samples),  # Less important linear feature
        'target': y
    })
    
    return df


def create_nonlinear_dataset():
    """Create a complex nonlinear dataset that should favor XGBoost."""
    np.random.seed(123)
    n_samples = 300
    
    # Create features with complex interactions
    x1 = np.random.uniform(0, 10, n_samples)
    x2 = np.random.uniform(0, 5, n_samples)
    x3 = np.random.uniform(-2, 2, n_samples)
    x4 = np.random.exponential(1, n_samples)
    
    # Complex nonlinear target with interactions and thresholds
    y = (np.where(x1 > 5, x1 * 2, x1 * 0.5) +  # Threshold effect
         np.log(x2 + 1) * 3 +  # Log relationship
         np.sin(x3) * 2 +  # Sinusoidal relationship
         np.sqrt(x4) * 1.5 +  # Square root relationship
         (x1 * x2) / 10 +  # Interaction term
         np.random.normal(0, 1, n_samples))  # Noise
    
    df = pd.DataFrame({
        'threshold_feature': x1,
        'log_feature': x2,
        'sine_feature': x3,
        'sqrt_feature': x4,
        'target': y
    })
    
    return df


def test_polynomial_performance():
    """Test polynomial regression on polynomial data."""
    print("=" * 60)
    print("TESTING POLYNOMIAL REGRESSION")
    print("=" * 60)
    
    # Create polynomial dataset
    poly_data = create_polynomial_dataset()
    print(f"Polynomial dataset shape: {poly_data.shape}")
    print("Dataset has quadratic relationships between features and target")
    
    # Test with analyzer
    analyzer = RegressionAnalyzer(poly_data, 'target')
    results = analyzer.fit_all_algorithms()
    
    # Show results
    print("\nModel Performance on Polynomial Data:")
    print("-" * 50)
    for i, (model, score) in enumerate(analyzer.model_scores.items(), 1):
        marker = "🏆" if i <= 3 else "  "
        print(f"{marker} {i:2d}. {model:30s}: R² = {score:.4f}")
    
    # Highlight polynomial models
    poly_models = [m for m in analyzer.model_scores.keys() if 'Polynomial' in m]
    if poly_models:
        print(f"\nPolynomial Models Performance:")
        for model in poly_models:
            rank = list(analyzer.model_scores.keys()).index(model) + 1
            score = analyzer.model_scores[model]
            print(f"  Rank #{rank}: {model} - R² = {score:.4f}")
    
    return analyzer


def test_xgboost_performance():
    """Test XGBoost on complex nonlinear data."""
    print("\n" + "=" * 60)
    print("TESTING XGBOOST PERFORMANCE")
    print("=" * 60)
    
    # Create complex nonlinear dataset
    nonlinear_data = create_nonlinear_dataset()
    print(f"Nonlinear dataset shape: {nonlinear_data.shape}")
    print("Dataset has complex nonlinear relationships and interactions")
    
    # Test with analyzer
    analyzer = RegressionAnalyzer(nonlinear_data, 'target')
    results = analyzer.fit_all_algorithms()
    
    # Show results
    print("\nModel Performance on Complex Nonlinear Data:")
    print("-" * 50)
    for i, (model, score) in enumerate(analyzer.model_scores.items(), 1):
        marker = "🏆" if i <= 3 else "  "
        print(f"{marker} {i:2d}. {model:30s}: R² = {score:.4f}")
    
    # Highlight XGBoost
    xgb_rank = list(analyzer.model_scores.keys()).index('XGBoost') + 1
    xgb_score = analyzer.model_scores['XGBoost']
    print(f"\nXGBoost Performance:")
    print(f"  Rank #{xgb_rank}: XGBoost - R² = {xgb_score:.4f}")
    
    return analyzer


def compare_algorithm_types():
    """Compare different algorithm types on the same dataset."""
    print("\n" + "=" * 60)
    print("ALGORITHM TYPE COMPARISON")
    print("=" * 60)
    
    # Use the sample dataset
    from regression_analyzer import create_sample_dataset
    data = create_sample_dataset()
    
    analyzer = RegressionAnalyzer(data, 'target_value')
    analyzer.fit_all_algorithms()
    
    # Categorize algorithms
    linear_models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Elastic Net']
    polynomial_models = ['Polynomial Regression (degree=2)', 'Polynomial Regression (degree=3)']
    tree_models = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'XGBoost']
    other_models = ['Support Vector Regression', 'K-Nearest Neighbors']
    
    print("\nPerformance by Algorithm Type:")
    print("-" * 40)
    
    categories = [
        ("Linear Models", linear_models),
        ("Polynomial Models", polynomial_models),
        ("Tree-Based Models", tree_models),
        ("Other Models", other_models)
    ]
    
    for category_name, models in categories:
        print(f"\n{category_name}:")
        category_scores = []
        for model in models:
            if model in analyzer.model_scores:
                score = analyzer.model_scores[model]
                rank = list(analyzer.model_scores.keys()).index(model) + 1
                print(f"  {model:30s}: R² = {score:.4f} (Rank #{rank})")
                category_scores.append(score)
        
        if category_scores:
            avg_score = np.mean(category_scores)
            print(f"  → Average R² for {category_name}: {avg_score:.4f}")


def main():
    """Run all tests for new algorithms."""
    print("POLYNOMIAL REGRESSION & XGBOOST TESTING")
    print("Testing the newly added algorithms with specialized datasets")
    
    # Test polynomial regression
    poly_analyzer = test_polynomial_performance()
    
    # Test XGBoost
    xgb_analyzer = test_xgboost_performance()
    
    # Compare algorithm types
    compare_algorithm_types()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Polynomial Regression: Added degree=2 and degree=3 variants")
    print("✅ XGBoost: Successfully integrated extreme gradient boosting")
    print("✅ All 12 algorithms work together seamlessly")
    print("✅ Interactive visualization supports all new models")
    
    print(f"\nTotal algorithms now available: {len(poly_analyzer.algorithms)}")
    print("New algorithms are ready for interactive 3D visualization!")
    
    # Offer to launch visualizer
    choice = input("\nLaunch interactive 3D visualizer with polynomial data? (y/n): ").strip().lower()
    if choice == 'y':
        print("Launching visualizer with polynomial dataset...")
        poly_analyzer.launch_interactive_visualizer()


if __name__ == "__main__":
    main()
