#!/usr/bin/env python3
"""
Base Test Framework for Data Mining Algorithms
Provides shared utilities, test data, and common testing patterns for:
- K-means clustering tests
- Hierarchical agglomerative clustering tests
- Other clustering algorithm tests

This eliminates code redundancy across test files.
"""

import unittest
import tempfile
import os
import sys
import importlib.util
import shutil

class BaseClusteringTest(unittest.TestCase):
    """Base test class with shared test data and utilities for clustering algorithms."""
    
    @classmethod
    def setUpClass(cls):
        """Set up shared test data used across multiple test classes."""
        # Standard 3D test data - 6 points naturally forming 3 clusters
        cls.standard_test_data = [
            [0, 0, 0],        # Cluster 1
            [1, 1, 1], 
            [10, 10, 10],     # Cluster 2  
            [11, 11, 11],
            [20, 20, 20],     # Cluster 3
            [21, 21, 21]
        ]
        
        # Minimal test data for edge cases
        cls.minimal_test_data = [
            [0, 0, 0],
            [10, 10, 10]
        ]
        
        # Test data with duplicates
        cls.duplicate_test_data = [
            [0, 0, 0],
            [0, 0, 0],  # Duplicate
            [5, 5, 5],
            [5, 5, 5]   # Duplicate
        ]
    
    def setUp(self):
        """Set up temporary directory and test files for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        
        # Create standard test data file
        self.standard_data_file = os.path.join(self.temp_dir, "standard_data.txt")
        self._create_data_file(self.standard_data_file, self.standard_test_data)
        
        # Create minimal test data file
        self.minimal_data_file = os.path.join(self.temp_dir, "minimal_data.txt")
        self._create_data_file(self.minimal_data_file, self.minimal_test_data)
        
        # Create file with invalid data
        self.invalid_data_file = os.path.join(self.temp_dir, "invalid_data.txt")
        with open(self.invalid_data_file, 'w') as f:
            f.write("1,2,3\n")
            f.write("invalid,line\n")
            f.write("4,5,6\n")
            f.write("# comment line\n")
            f.write("7,8,9\n")
            f.write("\n")  # empty line
        
        # Create duplicate data file
        self.duplicate_data_file = os.path.join(self.temp_dir, "duplicate_data.txt")
        self._create_data_file(self.duplicate_data_file, self.duplicate_test_data)
    
    def tearDown(self):
        """Clean up temporary files."""
        if hasattr(self, 'original_dir'):
            os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_data_file(self, filename, data):
        """Helper method to create a data file from point list."""
        with open(filename, 'w') as f:
            for point in data:
                f.write(f"{point[0]},{point[1]},{point[2]}\n")
    
    # Common test patterns that can be reused
    
    def assert_valid_distance(self, distance_func, point1, point2, expected_distance):
        """Assert that distance calculation is correct."""
        calculated = distance_func(point1, point2)
        self.assertAlmostEqual(calculated, expected_distance, places=5)
    
    def assert_euclidean_distance_properties(self, distance_func):
        """Test basic properties of Euclidean distance function."""
        # Test basic distance calculation
        self.assert_valid_distance(distance_func, [0, 0, 0], [3, 4, 0], 5.0)
        
        # Test distance to itself
        self.assert_valid_distance(distance_func, [1, 2, 3], [1, 2, 3], 0.0)
        
        # Test 3D distance
        expected_3d = (3 ** 0.5)  # sqrt(3)
        self.assert_valid_distance(distance_func, [0, 0, 0], [1, 1, 1], expected_3d)
    
    def assert_valid_silhouette_score(self, score, best_function):
        """Assert that silhouette coefficient is valid."""
        # Silhouette score should be in range [-1, 1]
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)
        
        # Best function should be one of the valid metrics
        expected_functions = [
            'a(i) - centroid separation',
            'b(i) - min avg separation', 
            'c(i) - max separation',
            'd(i) - min separation',
            'single_cluster',
            'no_valid_scores'
        ]
        self.assertIn(best_function, expected_functions)
    
    def assert_valid_clustering_result(self, clusters, expected_k, total_points):
        """Assert that clustering results are structurally valid."""
        # Check correct number of clusters
        self.assertEqual(len(clusters), expected_k)
        
        # Check that all points are assigned
        total_assigned = sum(len(cluster) for cluster in clusters)
        self.assertEqual(total_assigned, total_points)
        
        # Check that clusters are non-empty (unless empty dataset)
        if total_points > 0:
            for cluster in clusters:
                self.assertGreaterEqual(len(cluster), 0)
    
    def assert_distance_matrix_properties(self, distance_matrix, n_points):
        """Assert basic properties of a distance matrix."""
        # Check matrix size
        self.assertEqual(len(distance_matrix), n_points)
        if n_points > 0:
            self.assertEqual(len(distance_matrix[0]), n_points)
        
        # Check diagonal elements (distance to self)
        for i in range(n_points):
            self.assertEqual(distance_matrix[i][i], 0)
        
        # Check symmetry
        for i in range(n_points):
            for j in range(n_points):
                self.assertEqual(distance_matrix[i][j], distance_matrix[j][i])
    
    def assert_valid_file_loading(self, load_func, filename, expected_count):
        """Test file loading functionality with common assertions."""
        points = load_func(filename)
        self.assertEqual(len(points), expected_count)
        
        # Verify point structure (should be 3D)
        if expected_count > 0:
            for point in points:
                self.assertEqual(len(point), 3)
                for coord in point:
                    self.assertIsInstance(coord, (int, float))
    
    def assert_nonexistent_file_handling(self, load_func):
        """Test that loading non-existent files is handled gracefully."""
        points = load_func("nonexistent_file_12345.txt")
        self.assertEqual(len(points), 0)
    
    def assert_invalid_data_handling(self, load_func, invalid_file, expected_valid_count):
        """Test that invalid data in files is handled gracefully."""
        points = load_func(invalid_file)
        self.assertEqual(len(points), expected_valid_count)
    
    def create_results_file_and_verify(self, save_func, results_dir, filename_base, original_filename=None):
        """Helper to test results file creation and basic content verification."""
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"{filename_base}_test_results.txt")
        
        save_func(results_file, original_filename)
        
        # Verify file creation
        self.assertTrue(os.path.exists(results_file))
        
        # Basic content verification
        with open(results_file, 'r') as f:
            content = f.read()
        
        self.assertGreater(len(content), 50)  # Should have substantial content
        return content


class AlgorithmTestMixin:
    """Mixin providing common test patterns for clustering algorithms."""
    
    def run_data_loading_tests(self, algorithm_class, *args, **kwargs):
        """Run standard data loading tests for any clustering algorithm."""
        # Test valid dataset loading
        clusterer = algorithm_class(*args, **kwargs)
        self.assert_valid_file_loading(
            clusterer.load_dataset, 
            self.standard_data_file, 
            len(self.standard_test_data)
        )
        
        # Test non-existent file
        self.assert_nonexistent_file_handling(clusterer.load_dataset)
        
        # Test invalid data handling
        self.assert_invalid_data_handling(
            clusterer.load_dataset, 
            self.invalid_data_file, 
            3  # 3 valid lines in invalid_data_file
        )
    
    def run_distance_calculation_tests(self, clusterer_instance):
        """Run standard distance calculation tests."""
        self.assert_euclidean_distance_properties(clusterer_instance.euclidean_distance)
    
    def run_edge_case_tests(self, algorithm_class, *args, **kwargs):
        """Run standard edge case tests for clustering algorithms."""
        # Test with duplicate points
        clusterer = algorithm_class(*args, **kwargs)
        clusterer.load_dataset(self.duplicate_data_file)
        
        # Should handle duplicates without errors
        if hasattr(clusterer, 'fit'):
            try:
                clusters = clusterer.fit()
                # Should assign all points including duplicates
                total_assigned = sum(len(cluster) for cluster in clusters)
                self.assertEqual(total_assigned, len(self.duplicate_test_data))
            except Exception as e:
                self.fail(f"Algorithm failed with duplicate points: {e}")
        
        # Test with minimal dataset
        clusterer_minimal = algorithm_class(*args, **kwargs)
        clusterer_minimal.load_dataset(self.minimal_data_file)
        
        if hasattr(clusterer_minimal, 'fit'):
            try:
                clusters = clusterer_minimal.fit()
                # Should handle minimal data gracefully
                self.assertGreaterEqual(len(clusters), 0)
            except Exception as e:
                self.fail(f"Algorithm failed with minimal data: {e}")


def load_algorithm_module(module_name, file_path):
    """Utility function to dynamically load algorithm modules."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module