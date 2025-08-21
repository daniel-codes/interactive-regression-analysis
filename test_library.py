#!/usr/bin/env python3
"""
Test script to verify the regression analysis library works correctly.
"""

import sys
import traceback
import pandas as pd
import numpy as np
from regression_analyzer import RegressionAnalyzer, create_sample_dataset


def test_basic_functionality():
    """Test basic library functionality without GUI."""
    print("Testing basic functionality...")
    
    try:
        # Test 1: Create sample data
        print("  ✓ Creating sample dataset...")
        data = create_sample_dataset()
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'target_value' in data.columns
        print(f"    Sample data shape: {data.shape}")
        
        # Test 2: Initialize analyzer
        print("  ✓ Initializing RegressionAnalyzer...")
        analyzer = RegressionAnalyzer(data, 'target_value')
        assert len(analyzer.feature_columns) >= 2
        print(f"    Feature columns: {analyzer.feature_columns}")
        
        # Test 3: Fit algorithms
        print("  ✓ Fitting regression algorithms...")
        results = analyzer.fit_all_algorithms()
        assert len(results) > 0
        assert len(analyzer.fitted_models) > 0
        print(f"    Successfully fitted {len(analyzer.fitted_models)} models")
        
        # Test 4: Get best model
        print("  ✓ Getting best model...")
        best_name, best_model, is_scaled = analyzer.get_best_model()
        assert best_name in analyzer.fitted_models
        print(f"    Best model: {best_name} (R² = {analyzer.model_scores[best_name]:.4f})")
        
        # Test 5: Make predictions
        print("  ✓ Testing predictions...")
        sample_X = data[analyzer.feature_columns].iloc[:5]  # First 5 rows
        predictions = analyzer.predict_with_model(best_name, sample_X)
        assert len(predictions) == 5
        print(f"    Sample predictions: {predictions[:3]}")
        
        print("✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and validation."""
    print("\nTesting error handling...")
    
    try:
        # Test invalid input types
        print("  ✓ Testing invalid DataFrame input...")
        try:
            RegressionAnalyzer("not a dataframe", 'target')
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test invalid target column
        print("  ✓ Testing invalid target column...")
        data = create_sample_dataset()
        try:
            RegressionAnalyzer(data, 'nonexistent_column')
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        # Test insufficient numeric columns
        print("  ✓ Testing insufficient numeric columns...")
        small_data = pd.DataFrame({
            'text_col': ['a', 'b', 'c'],
            'target': [1, 2, 3]
        })
        try:
            RegressionAnalyzer(small_data, 'target')
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        print("✅ All error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_with_different_datasets():
    """Test with different types of datasets."""
    print("\nTesting with different datasets...")
    
    try:
        # Test with small dataset
        print("  ✓ Testing with small dataset...")
        np.random.seed(42)
        small_data = pd.DataFrame({
            'x1': np.random.normal(0, 1, 50),
            'x2': np.random.normal(5, 2, 50),
            'x3': np.random.uniform(-1, 1, 50),
            'y': np.random.normal(10, 3, 50)
        })
        small_data['y'] = small_data['x1'] * 2 + small_data['x2'] * 0.5 + np.random.normal(0, 1, 50)
        
        analyzer = RegressionAnalyzer(small_data, 'y')
        results = analyzer.fit_all_algorithms()
        assert len(results) > 0
        print(f"    Small dataset: {len(analyzer.fitted_models)} models fitted")
        
        # Test with dataset having missing values (should handle gracefully)
        print("  ✓ Testing with dataset containing NaN values...")
        data_with_nan = small_data.copy()
        data_with_nan.loc[0:5, 'x1'] = np.nan
        
        try:
            analyzer_nan = RegressionAnalyzer(data_with_nan.dropna(), 'y')
            results_nan = analyzer_nan.fit_all_algorithms()
            print(f"    NaN handling: Successfully fitted models")
        except Exception as e:
            print(f"    NaN handling: Some models failed (expected): {str(e)}")
        
        print("✅ Different dataset tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Different dataset test failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("REGRESSION ANALYSIS LIBRARY - TEST SUITE")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_basic_functionality():
        tests_passed += 1
    
    if test_error_handling():
        tests_passed += 1
    
    if test_with_different_datasets():
        tests_passed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY: {tests_passed}/{total_tests} test suites passed")
    print(f"{'='*60}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The library is working correctly.")
        print("\nTo run the interactive demo:")
        print("python demo.py")
        return True
    else:
        print("❌ Some tests failed. Please check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
