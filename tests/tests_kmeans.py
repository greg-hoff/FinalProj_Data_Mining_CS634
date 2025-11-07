#!/usr/bin/env python3
"""
Unit tests for K-means Clustering Algorithm
Tests input handling, output generation, and algorithm correctness.
Uses shared test base to eliminate redundancy.
"""

import unittest
import os
import sys

# Import shared test base
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_base import BaseClusteringTest, AlgorithmTestMixin, load_algorithm_module

# Load K-means module
kmeans_module = load_algorithm_module("kmeans_module", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "K-means_cluster_algo.py"))
KMeansClusterer = kmeans_module.KMeansClusterer


class BaseKMeansTest(BaseClusteringTest, AlgorithmTestMixin):
    """Base test class for K-means with shared utilities."""
    
    def create_configured_clusterer(self, k=2):
        """Create a clusterer with standard test setup."""
        clusterer = KMeansClusterer(k=k)
        clusterer.points = self.standard_test_data
        clusterer.centroids = [[0.5, 0.5, 0.5], [20.5, 20.5, 20.5]] if k == 2 else None
        clusterer.clusters = [
            [[0, 0, 0], [1, 1, 1]],
            [[20, 20, 20], [21, 21, 21]]
        ] if k == 2 else None
        return clusterer


class TestKMeansClusterer(BaseKMeansTest):
    """Test cases for K-means clustering algorithm."""
    
    def test_clusterer_initialization(self):
        """Test that K-means clusterer initializes correctly."""
        clusterer = KMeansClusterer(k=3, max_iterations=50, tolerance=0.001)
        
        self.assertEqual(clusterer.k, 3)
        self.assertEqual(clusterer.max_iterations, 50)
        self.assertEqual(clusterer.tolerance, 0.001)
        self.assertEqual(clusterer.random_seed, 42)
        self.assertEqual(len(clusterer.centroids), 0)
        self.assertEqual(len(clusterer.clusters), 0)
        self.assertEqual(len(clusterer.points), 0)
    
    def test_data_loading_operations(self):
        """Test all data loading scenarios using shared base methods."""
        self.run_data_loading_tests(KMeansClusterer, k=2)
    
    def test_initialize_centroids(self):
        """Test centroid initialization."""
        clusterer = KMeansClusterer(k=3)
        clusterer.points = self.standard_test_data
        clusterer.initialize_centroids()
        
        self.assertEqual(len(clusterer.centroids), 3)
        # Each centroid should be a 3D point
        for centroid in clusterer.centroids:
            self.assertEqual(len(centroid), 3)
    
    def test_initialize_centroids_too_few_points(self):
        """Test centroid initialization with insufficient points."""
        clusterer = KMeansClusterer(k=5)
        clusterer.points = [[1, 2, 3], [4, 5, 6]]  # Only 2 points for 5 clusters
        
        with self.assertRaises(ValueError):
            clusterer.initialize_centroids()
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation using shared base methods."""
        clusterer = KMeansClusterer(k=2)
        self.run_distance_calculation_tests(clusterer)
    
    def test_assign_points_to_clusters(self):
        """Test point assignment to clusters."""
        clusterer = KMeansClusterer(k=2)
        clusterer.points = self.standard_test_data
        clusterer.centroids = [[1, 1, 1], [20, 20, 20]]  # Two distinct centroids
        
        clusters = clusterer.assign_points_to_clusters()
        
        self.assertEqual(len(clusters), 2)
        # Points should be assigned to nearest centroids
        self.assertGreater(len(clusters[0]), 0)
        self.assertGreater(len(clusters[1]), 0)
    
    def test_update_centroids(self):
        """Test centroid update calculation."""
        clusterer = KMeansClusterer(k=2)
        
        # Set up initial centroids and clusters
        clusterer.centroids = [[0, 0, 0], [5, 5, 5]]  # Initial centroids
        clusterer.clusters = [
            [[1, 1, 1], [3, 3, 3]],  # Cluster 1: centroid should be [2, 2, 2]
            [[8, 8, 8], [12, 12, 12]]  # Cluster 2: centroid should be [10, 10, 10]
        ]
        
        max_change = clusterer.update_centroids()
        
        # Method returns max change and updates centroids in place
        self.assertIsInstance(max_change, float)
        self.assertGreater(max_change, 0)  # Centroids should have moved
        
        # Check that centroids were updated correctly
        self.assertEqual(len(clusterer.centroids), 2)
        self.assertAlmostEqual(clusterer.centroids[0][0], 2.0, places=2)
        self.assertAlmostEqual(clusterer.centroids[0][1], 2.0, places=2)
        self.assertAlmostEqual(clusterer.centroids[0][2], 2.0, places=2)
        self.assertAlmostEqual(clusterer.centroids[1][0], 10.0, places=2)
        self.assertAlmostEqual(clusterer.centroids[1][1], 10.0, places=2)
        self.assertAlmostEqual(clusterer.centroids[1][2], 10.0, places=2)
    
    def test_algorithm_convergence(self):
        """Test that the algorithm converges and produces valid results."""
        clusterer = KMeansClusterer(k=2, max_iterations=10)
        clusterer.points = self.standard_test_data
        
        centroids, clusters = clusterer.fit()
        
        # Use shared validation method
        self.assert_valid_clustering_result(clusters, 2, len(self.standard_test_data))
        
        # Verify centroid validity
        self.assertEqual(len(centroids), 2)
        for centroid in centroids:
            self.assertEqual(len(centroid), 3)
            for coord in centroid:
                self.assertIsInstance(coord, (int, float))
    
    def test_silhouette_coefficient_calculation(self):
        """Test silhouette coefficient calculation."""
        clusterer = self.create_configured_clusterer(k=2)
        
        silhouette, best_function = clusterer.calculate_silhouette_coefficient()
        
        # Use shared validation method
        self.assert_valid_silhouette_score(silhouette, best_function)
    
    def test_output_operations(self):
        """Test save_results and print_results operations."""
        clusterer = self.create_configured_clusterer(k=2)
        clusterer.iteration_history = [0.5, 0.3, 0.1]
        
        # Test save_results using shared helper
        content = self.create_results_file_and_verify(
            clusterer.save_results,
            'K-means_Cluster_Results',
            'kmeans',
            self.standard_data_file
        )
        
        # Check K-means specific content
        self.assertIn('K-means Clustering Results', content)
        self.assertIn('Centroids:', content)
        self.assertIn('Cluster Assignments:', content)
        self.assertIn('Silhouette Coefficient:', content)
        
        # Test print_results (should not raise exceptions)
        try:
            clusterer.print_results()
        except Exception as e:
            self.fail(f"print_results raised an exception: {e}")
    
    def test_edge_cases(self):
        """Test edge cases using shared base methods."""
        self.run_edge_case_tests(KMeansClusterer, k=2)


class TestKMeansIntegration(BaseKMeansTest):
    """Integration tests for the complete K-means workflow."""
    
    def setUp(self):
        """Set up test fixtures with Prime directory structure."""
        super().setUp()  # Call parent setup
        
        # Create Prime directory and test file for integration testing
        prime_dir = os.path.join(self.temp_dir, 'Prime')
        os.makedirs(prime_dir, exist_ok=True)
        self.prime_file = os.path.join(prime_dir, 'test-prime.txt')
        self._create_data_file(self.prime_file, self.standard_test_data)
    
    def test_complete_workflow_with_file_handling(self):
        """Test end-to-end workflow focusing on file I/O and folder structure."""
        clusterer = KMeansClusterer(k=2)
        
        # Test loading from Prime folder structure
        points = clusterer.load_dataset(self.prime_file)
        self.assertEqual(len(points), len(self.standard_test_data))
        
        # Run full workflow
        centroids, clusters = clusterer.fit()
        
        # Use shared validation
        self.assert_valid_clustering_result(clusters, 2, len(self.standard_test_data))
        
        # Test output with folder structure using shared helper
        content = self.create_results_file_and_verify(
            clusterer.save_results,
            os.path.join(self.temp_dir, 'K-means_Cluster_Results'),
            'integration',
            self.prime_file
        )
        
        # Should extract base filename correctly from folder path
        self.assertIn('test-prime.txt', content)
        self.assertNotIn('Prime/', content)  # Should not include folder path


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
