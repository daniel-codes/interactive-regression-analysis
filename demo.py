#!/usr/bin/env python3
"""
Demo script for the Interactive Regression Analysis Library

This script demonstrates how to use the RegressionAnalyzer with different datasets.
"""

import pandas as pd
import numpy as np
from regression_analyzer import RegressionAnalyzer, create_sample_dataset


def demo_with_sample_data():
    """Demo using the built-in sample dataset."""
    print("=" * 60)
    print("DEMO 1: Using Built-in Sample Dataset")
    print("=" * 60)
    
    # Create sample data
    sample_data = create_sample_dataset()
    print(f"Sample dataset shape: {sample_data.shape}")
    print(f"Columns: {list(sample_data.columns)}")
    print("\nFirst few rows:")
    print(sample_data.head())
    
    # Initialize analyzer
    analyzer = RegressionAnalyzer(sample_data, 'target_value')
    
    # Fit all algorithms
    print("\nFitting all regression algorithms...")
    results = analyzer.fit_all_algorithms()
    
    # Display results
    print("\nModel Performance Summary:")
    print("-" * 50)
    for model, score in analyzer.model_scores.items():
        print(f"{model:25}: R² = {score:.4f}")
    
    best_model_name, best_model, is_scaled = analyzer.get_best_model()
    print(f"\nBest performing model: {best_model_name}")
    
    # Launch interactive visualizer
    print("\nLaunching interactive 3D visualizer...")
    analyzer.launch_interactive_visualizer()


def demo_with_custom_data():
    """Demo using a custom dataset (Boston Housing style)."""
    print("=" * 60)
    print("DEMO 2: Using Custom Housing Dataset")
    print("=" * 60)
    
    # Create a more complex synthetic dataset
    np.random.seed(123)
    n_samples = 500
    
    # Housing features
    rooms = np.random.normal(6, 2, n_samples)
    rooms = np.clip(rooms, 1, 15)  # Reasonable range
    
    area = np.random.normal(2000, 500, n_samples)
    area = np.clip(area, 500, 5000)  # Square feet
    
    age = np.random.uniform(0, 50, n_samples)  # Years old
    
    distance_to_city = np.random.exponential(5, n_samples)  # Miles
    
    crime_rate = np.random.exponential(2, n_samples)
    
    # Create price with realistic relationships
    price = (
        20000 +  # Base price
        rooms * 15000 +  # More rooms = higher price
        area * 50 +  # Bigger area = higher price
        -age * 500 +  # Older = cheaper
        -distance_to_city * 2000 +  # Farther from city = cheaper
        -crime_rate * 5000 +  # Higher crime = cheaper
        np.random.normal(0, 10000, n_samples)  # Noise
    )
    
    # Ensure positive prices
    price = np.clip(price, 50000, 1000000)
    
    # Create DataFrame
    housing_data = pd.DataFrame({
        'rooms': rooms,
        'area_sqft': area,
        'age_years': age,
        'distance_to_city': distance_to_city,
        'crime_rate': crime_rate,
        'price': price
    })
    
    print(f"Housing dataset shape: {housing_data.shape}")
    print(f"Columns: {list(housing_data.columns)}")
    print("\nDataset statistics:")
    print(housing_data.describe())
    
    # Initialize analyzer
    analyzer = RegressionAnalyzer(housing_data, 'price')
    
    # Fit all algorithms
    print("\nFitting all regression algorithms...")
    results = analyzer.fit_all_algorithms()
    
    # Display results
    print("\nModel Performance Summary:")
    print("-" * 50)
    for model, score in analyzer.model_scores.items():
        print(f"{model:25}: R² = {score:.4f}")
    
    best_model_name, best_model, is_scaled = analyzer.get_best_model()
    print(f"\nBest performing model: {best_model_name}")
    
    # Launch interactive visualizer
    print("\nLaunching interactive 3D visualizer...")
    analyzer.launch_interactive_visualizer()


def demo_with_csv_file():
    """Demo showing how to load data from a CSV file."""
    print("=" * 60)
    print("DEMO 3: Loading Data from CSV")
    print("=" * 60)
    
    # Create a sample CSV file for demonstration
    sample_data = create_sample_dataset()
    csv_filename = '/root/store/code/cursor1/sample_data.csv'
    sample_data.to_csv(csv_filename, index=False)
    print(f"Created sample CSV file: {csv_filename}")
    
    # Load data from CSV
    try:
        df = pd.read_csv(csv_filename)
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Initialize analyzer
        analyzer = RegressionAnalyzer(df, 'target_value')
        
        # Fit algorithms (quick run)
        results = analyzer.fit_all_algorithms()
        
        print("\nTop 3 models:")
        for i, (model, score) in enumerate(list(analyzer.model_scores.items())[:3]):
            print(f"{i+1}. {model}: R² = {score:.4f}")
        
        print("\nTo launch visualizer, uncomment the line below:")
        print("# analyzer.launch_interactive_visualizer()")
        
    except Exception as e:
        print(f"Error loading CSV: {e}")


def main():
    """Main function to run demos."""
    print("Interactive Regression Analysis Library - Demo")
    print("=" * 60)
    
    while True:
        print("\nChoose a demo:")
        print("1. Sample dataset with interactive visualizer")
        print("2. Custom housing dataset with interactive visualizer")
        print("3. CSV file loading demo (no visualizer)")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            demo_with_sample_data()
        elif choice == '2':
            demo_with_custom_data()
        elif choice == '3':
            demo_with_csv_file()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
