#!/usr/bin/env python3
"""
Unit tests for K-means Clustering Algorithm
Tests input handling, output generation, and algorithm correctness.
"""

import unittest
import os
import sys
import tempfile
import shutil
import importlib.util
from unittest.mock import patch, mock_open

# Add parent directory to path to import the K-means module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import with the correct module name
spec = importlib.util.spec_from_file_location("kmeans_module", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "K-means_cluster_algo.py"))
kmeans_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kmeans_module)
KMeansClusterer = kmeans_module.KMeansClusterer


class BaseKMeansTest(unittest.TestCase):
    """Base test class with common setup and utilities."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data used across multiple tests."""
        # Standard test data - 6 points in 3 natural clusters
        cls.test_data = [
            [1.0, 2.0, 3.0],   # Cluster 1
            [2.0, 3.0, 4.0],   # Cluster 1
            [10.0, 11.0, 12.0], # Cluster 2
            [11.0, 12.0, 13.0], # Cluster 2
            [20.0, 21.0, 22.0], # Cluster 3
            [21.0, 22.0, 23.0]  # Cluster 3
        ]
    
    def setUp(self):
        """Set up temporary directory for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        # Create standard test dataset file
        with open('test_dataset.txt', 'w') as f:
            for point in self.test_data:
                f.write(f"{point[0]},{point[1]},{point[2]}\n")
    
    def tearDown(self):
        """Clean up temporary directory."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir)
    
    def create_configured_clusterer(self, k=2):
        """Create a clusterer with standard test setup."""
        clusterer = KMeansClusterer(k=k)
        clusterer.points = self.test_data
        clusterer.centroids = [[1.5, 2.5, 3.5], [20.5, 21.5, 22.5]] if k == 2 else None
        clusterer.clusters = [
            [[1, 2, 3], [2, 3, 4]],
            [[20, 21, 22], [21, 22, 23]]
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
    
    def test_load_dataset_success(self):
        """Test loading a valid dataset file."""
        clusterer = KMeansClusterer(k=2)
        points = clusterer.load_dataset('test_dataset.txt')
        
        self.assertEqual(len(points), 6)
        self.assertEqual(len(clusterer.points), 6)
        self.assertEqual(points[0], [1.0, 2.0, 3.0])
        self.assertEqual(points[-1], [21.0, 22.0, 23.0])
    
    def test_load_dataset_file_not_found(self):
        """Test loading a non-existent dataset file."""
        clusterer = KMeansClusterer(k=2)
        points = clusterer.load_dataset('nonexistent_file.txt')
        
        self.assertEqual(len(points), 0)
        self.assertEqual(len(clusterer.points), 0)
    
    def test_load_dataset_invalid_format(self):
        """Test loading a dataset with invalid data format."""
        # Create file with invalid data
        with open('invalid_dataset.txt', 'w') as f:
            f.write("1,2,3\n")
            f.write("invalid,line,here\n")
            f.write("4,5,6\n")
        
        clusterer = KMeansClusterer(k=2)
        points = clusterer.load_dataset('invalid_dataset.txt')
        
        # Should load valid lines only
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0], [1.0, 2.0, 3.0])
        self.assertEqual(points[1], [4.0, 5.0, 6.0])
    
    def test_initialize_centroids(self):
        """Test centroid initialization."""
        clusterer = KMeansClusterer(k=3)
        clusterer.points = self.test_data
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
        """Test Euclidean distance calculation."""
        clusterer = KMeansClusterer(k=2)
        
        point1 = [0, 0, 0]
        point2 = [3, 4, 0]
        distance = clusterer.euclidean_distance(point1, point2)
        
        self.assertAlmostEqual(distance, 5.0, places=2)
    
    def test_assign_points_to_clusters(self):
        """Test point assignment to clusters."""
        clusterer = KMeansClusterer(k=2)
        clusterer.points = self.test_data
        clusterer.centroids = [[1, 2, 3], [20, 21, 22]]  # Two distinct centroids
        
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
        clusterer.points = self.test_data
        
        centroids, clusters = clusterer.fit()
        
        # Verify structure
        self.assertEqual(len(centroids), 2)
        self.assertEqual(len(clusters), 2)
        
        # Verify all points are assigned
        total_assigned = sum(len(cluster) for cluster in clusters)
        self.assertEqual(total_assigned, len(self.test_data))
        
        # Verify centroid validity
        for centroid in centroids:
            self.assertEqual(len(centroid), 3)
            for coord in centroid:
                self.assertIsInstance(coord, (int, float))
    
    def test_silhouette_coefficient_calculation(self):
        """Test silhouette coefficient calculation."""
        clusterer = self.create_configured_clusterer(k=2)
        
        silhouette, best_function = clusterer.calculate_silhouette_coefficient()
        
        self.assertIsInstance(silhouette, float)
        self.assertGreaterEqual(silhouette, -1.0)
        self.assertLessEqual(silhouette, 1.0)
        self.assertIsInstance(best_function, str)
    
    def test_output_operations(self):
        """Test save_results and print_results operations."""
        clusterer = self.create_configured_clusterer(k=2)
        clusterer.iteration_history = [0.5, 0.3, 0.1]
        
        # Test save_results
        os.makedirs('K-means_Cluster_Results', exist_ok=True)
        result_file = 'K-means_Cluster_Results/test_results.txt'
        clusterer.save_results(result_file, 'test_dataset.txt')
        
        self.assertTrue(os.path.exists(result_file))
        
        # Check file contents
        with open(result_file, 'r') as f:
            content = f.read()
            self.assertIn('K-means Clustering Results', content)
            self.assertIn('test_dataset.txt', content)
            self.assertIn('Centroids:', content)
            self.assertIn('Cluster Assignments:', content)
            self.assertIn('Silhouette Coefficient:', content)
        
        # Test print_results (should not raise exceptions)
        try:
            clusterer.print_results()
        except Exception as e:
            self.fail(f"print_results raised an exception: {e}")


class TestKMeansIntegration(BaseKMeansTest):
    """Integration tests for the complete K-means workflow."""
    
    def setUp(self):
        """Set up test fixtures with Prime directory structure."""
        super().setUp()  # Call parent setup
        
        # Create Prime directory and test file for integration testing
        os.makedirs('Prime', exist_ok=True)
        with open('Prime/test-prime.txt', 'w') as f:
            for point in self.test_data:
                f.write(f"{point[0]},{point[1]},{point[2]}\n")
    
    def test_complete_workflow_with_file_handling(self):
        """Test end-to-end workflow focusing on file I/O and folder structure."""
        clusterer = KMeansClusterer(k=2)
        
        # Test loading from Prime folder structure
        points = clusterer.load_dataset('Prime/test-prime.txt')
        self.assertEqual(len(points), 6)
        
        # Run full workflow
        centroids, clusters = clusterer.fit()
        
        # Test output with folder structure
        os.makedirs('K-means_Cluster_Results', exist_ok=True)
        result_file = 'K-means_Cluster_Results/test_output.txt'
        clusterer.save_results(result_file, 'Prime/test-prime.txt')
        
        # Verify file handling and content
        self.assertTrue(os.path.exists(result_file))
        with open(result_file, 'r') as f:
            content = f.read()
            self.assertGreater(len(content), 100)
            # Should extract base filename correctly from folder path
            self.assertIn('test-prime.txt', content)
            self.assertNotIn('Prime/', content)  # Should not include folder path


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
