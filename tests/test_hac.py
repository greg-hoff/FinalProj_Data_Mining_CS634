#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Hierarchical Agglomerative Clustering Algorithm
Uses shared test base to eliminate redundancy with K-means tests.
Tests HAC-specific functionality while reusing common clustering test patterns.
"""

import unittest
import os
import sys

# Import shared test base
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_base import BaseClusteringTest, AlgorithmTestMixin, load_algorithm_module

# Load HAC module
hac_module = load_algorithm_module("hac_algo", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hac_algo.py"))

class BaseHACTest(BaseClusteringTest, AlgorithmTestMixin):
    """Base test class for HAC with shared utilities."""
    
    def create_configured_clusterer(self, k=3, linkage='complete'):
        """Create a HAC clusterer with test data loaded."""
        clusterer = hac_module.HierarchicalClusterer(k=k, linkage=linkage)
        clusterer.load_dataset(self.standard_data_file)
        return clusterer


class TestHACInitialization(BaseHACTest):
    """Test HAC clusterer initialization and basic properties."""
    
    def test_basic_initialization(self):
        """Test basic clusterer initialization with default parameters."""
        clusterer = hac_module.HierarchicalClusterer(k=3)
        self.assertEqual(clusterer.k, 3)
        self.assertEqual(clusterer.linkage, 'complete')
        self.assertEqual(len(clusterer.points), 0)
        self.assertEqual(len(clusterer.clusters), 0)
    
    def test_initialization_with_linkage(self):
        """Test clusterer initialization with different linkage criteria."""
        for linkage in ['single', 'complete', 'average']:
            clusterer = hac_module.HierarchicalClusterer(k=2, linkage=linkage)
            self.assertEqual(clusterer.linkage, linkage)
    
    def test_invalid_linkage_criterion(self):
        """Test that invalid linkage criterion raises error during clustering."""
        clusterer = hac_module.HierarchicalClusterer(k=2, linkage='invalid')
        clusterer.points = self.standard_test_data
        clusterer.compute_distance_matrix()
        
        with self.assertRaises(ValueError):
            clusterer.cluster_distance([0, 1], [2, 3])


class TestDataLoading(BaseHACTest):
    """Test data loading functionality using shared base methods."""
    
    def test_data_loading_operations(self):
        """Test all data loading scenarios using shared base methods."""
        self.run_data_loading_tests(hac_module.HierarchicalClusterer, k=3)


class TestDistanceCalculations(BaseHACTest):
    """Test distance calculation methods."""
    
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation using shared base methods."""
        clusterer = self.create_configured_clusterer()
        self.run_distance_calculation_tests(clusterer)
    
    def test_compute_distance_matrix(self):
        """Test distance matrix computation."""
        clusterer = self.create_configured_clusterer()
        clusterer.compute_distance_matrix()
        
        # Use shared validation method
        n = len(clusterer.points)
        self.assert_distance_matrix_properties(clusterer.distance_matrix, n)
    
    def test_cluster_distance_single_linkage(self):
        """Test cluster distance calculation with single linkage."""
        clusterer = self.create_configured_clusterer(linkage='single')
        clusterer.compute_distance_matrix()
        
        # Test distance between two single-point clusters
        dist = clusterer.cluster_distance([0], [1])
        expected = clusterer.euclidean_distance(clusterer.points[0], clusterer.points[1])
        self.assertAlmostEqual(dist, expected, places=5)
        
        # Test distance between multi-point clusters
        dist = clusterer.cluster_distance([0, 1], [2, 3])
        # Should be minimum distance between any two points
        min_dist = float('inf')
        for i in [0, 1]:
            for j in [2, 3]:
                d = clusterer.euclidean_distance(clusterer.points[i], clusterer.points[j])
                min_dist = min(min_dist, d)
        self.assertAlmostEqual(dist, min_dist, places=5)
    
    def test_cluster_distance_complete_linkage(self):
        """Test cluster distance calculation with complete linkage."""
        clusterer = self.create_configured_clusterer(linkage='complete')
        clusterer.compute_distance_matrix()
        
        # Test distance between multi-point clusters
        dist = clusterer.cluster_distance([0, 1], [2, 3])
        # Should be maximum distance between any two points
        max_dist = 0
        for i in [0, 1]:
            for j in [2, 3]:
                d = clusterer.euclidean_distance(clusterer.points[i], clusterer.points[j])
                max_dist = max(max_dist, d)
        self.assertAlmostEqual(dist, max_dist, places=5)
    
    def test_cluster_distance_average_linkage(self):
        """Test cluster distance calculation with average linkage."""
        clusterer = self.create_configured_clusterer(linkage='average')
        clusterer.compute_distance_matrix()
        
        # Test distance between multi-point clusters
        dist = clusterer.cluster_distance([0, 1], [2, 3])
        # Should be average distance between all pairs
        total_dist = 0
        count = 0
        for i in [0, 1]:
            for j in [2, 3]:
                d = clusterer.euclidean_distance(clusterer.points[i], clusterer.points[j])
                total_dist += d
                count += 1
        expected_avg = total_dist / count
        self.assertAlmostEqual(dist, expected_avg, places=5)


class TestClusteringOperations(BaseHACTest):
    """Test clustering algorithm functionality."""
    
    def test_find_closest_clusters(self):
        """Test finding closest clusters for merging."""
        clusterer = self.create_configured_clusterer()
        clusterer.compute_distance_matrix()
        
        # Start with each point as its own cluster
        clusters = [[i] for i in range(len(clusterer.points))]
        
        closest_pair, min_distance = clusterer.find_closest_clusters(clusters)
        
        # Verify that closest_pair is a valid pair
        self.assertIsNotNone(closest_pair)
        self.assertIsInstance(closest_pair, tuple)
        self.assertEqual(len(closest_pair), 2)
        self.assertNotEqual(closest_pair[0], closest_pair[1])
        self.assertGreaterEqual(min_distance, 0)
    
    def test_basic_clustering(self):
        """Test basic clustering functionality."""
        clusterer = self.create_configured_clusterer(k=2)
        clusters = clusterer.fit()
        
        # Use shared validation method
        self.assert_valid_clustering_result(clusters, 2, len(self.standard_test_data))
    
    def test_single_cluster(self):
        """Test clustering with k=1."""
        clusterer = self.create_configured_clusterer(k=1)
        clusters = clusterer.fit()
        
        self.assert_valid_clustering_result(clusters, 1, len(self.standard_test_data))
    
    def test_maximum_clusters(self):
        """Test clustering with k equal to number of points."""
        k = len(self.standard_test_data)
        clusterer = self.create_configured_clusterer(k=k)
        clusters = clusterer.fit()
        
        self.assert_valid_clustering_result(clusters, k, len(self.standard_test_data))
        # Each cluster should have exactly one point
        for cluster in clusters:
            self.assertEqual(len(cluster), 1)
    
    def test_empty_dataset(self):
        """Test clustering with empty dataset."""
        clusterer = hac_module.HierarchicalClusterer(k=3)
        clusters = clusterer.fit()
        
        self.assertEqual(len(clusters), 0)


class TestSilhouetteCoefficient(BaseHACTest):
    """Test silhouette coefficient calculation."""
    
    def test_silhouette_calculation(self):
        """Test silhouette coefficient calculation."""
        clusterer = self.create_configured_clusterer(k=3)
        clusterer.fit()
        
        score, best_function = clusterer.calculate_silhouette_coefficient()
        
        # Use shared validation method
        self.assert_valid_silhouette_score(score, best_function)
    
    def test_silhouette_single_cluster(self):
        """Test silhouette coefficient with single cluster."""
        clusterer = self.create_configured_clusterer(k=1)
        clusterer.fit()
        
        score, best_function = clusterer.calculate_silhouette_coefficient()
        
        self.assertEqual(score, 0.0)
        self.assertEqual(best_function, "single_cluster")
    
    def test_silhouette_maximum_clusters(self):
        """Test silhouette coefficient when k equals number of points."""
        k = len(self.standard_test_data)
        clusterer = self.create_configured_clusterer(k=k)
        clusterer.fit()
        
        score, best_function = clusterer.calculate_silhouette_coefficient()
        
        # Use shared validation method
        self.assert_valid_silhouette_score(score, best_function)


class TestFileOperations(BaseHACTest):
    """Test file I/O operations."""
    
    def test_save_results(self):
        """Test saving clustering results to file."""
        clusterer = self.create_configured_clusterer(k=2)
        clusterer.fit()
        
        # Use shared helper method
        content = self.create_results_file_and_verify(
            clusterer.save_results,
            'HAC_Cluster_Results',
            'hac',
            self.standard_data_file
        )
        
        # Check HAC-specific content
        self.assertIn("Hierarchical Agglomerative Clustering Results", content)
        self.assertIn("k=2", content)
        self.assertIn("Cluster Centroids", content)
        self.assertIn("Silhouette Coefficient", content)


class TestEdgeCases(BaseHACTest):
    """Test edge cases and error handling."""
    
    def test_edge_cases(self):
        """Test edge cases using shared base methods."""
        self.run_edge_case_tests(hac_module.HierarchicalClusterer, k=2)
    
    def test_invalid_k_value(self):
        """Test clustering with k greater than number of points."""
        clusterer = self.create_configured_clusterer(k=10)  # More than points available
        clusters = clusterer.fit()
        
        # Should have as many clusters as points
        self.assertEqual(len(clusters), len(self.standard_test_data))


class TestIntegration(BaseHACTest):
    """Integration tests comparing different configurations."""
    
    def test_linkage_comparison(self):
        """Test that different linkage criteria produce valid results."""
        results = {}
        
        for linkage in ['single', 'complete', 'average']:
            clusterer = self.create_configured_clusterer(k=2, linkage=linkage)
            clusterer.fit()
            score, best_function = clusterer.calculate_silhouette_coefficient()
            results[linkage] = score
            
            # Use shared validation for each result
            self.assert_valid_silhouette_score(score, best_function)
        
        # Should have results for all linkage types
        self.assertEqual(len(results), 3)
    
    def test_k_value_comparison(self):
        """Test clustering with different k values."""
        scores = {}
        
        for k in range(1, min(5, len(self.standard_test_data) + 1)):
            clusterer = self.create_configured_clusterer(k=k)
            clusterer.fit()
            score, best_function = clusterer.calculate_silhouette_coefficient()
            scores[k] = score
            
            # Use shared validation
            self.assert_valid_silhouette_score(score, best_function)
        
        # Should have results for each k
        self.assertGreater(len(scores), 0)
        
        # k=1 should always have score 0
        if 1 in scores:
            self.assertEqual(scores[1], 0.0)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)