#!/usr/bin/env python3
"""
Hierarchical Agglomerative Clustering (HAC) Algorithm Implementation
This program implements hierarchical agglomerative clustering to cluster 3D points
into k user-specified clusters using bottom-up approach.
The user specifies the number of clusters (k) and the linkage criterion upon execution.
Then calculates silhouette coefficient using enhanced method with four metrics. The metric with the best score is reported.
"""

import math
import numpy as np
import os

class HierarchicalClusterer:
    def __init__(self, k, linkage='complete'):
        """
        Initialize Hierarchical Agglomerative Clusterer.
        
        Args:
            k: Number of final clusters desired
            linkage: Linkage criterion ('single', 'complete', 'average')
        """
        self.k = k
        self.linkage = linkage
        self.points = []
        self.clusters = []
        self.cluster_history = []
        self.distance_matrix = []
        
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two 3D points."""
        return math.sqrt(
            (point1[0] - point2[0])**2 + 
            (point1[1] - point2[1])**2 + 
            (point1[2] - point2[2])**2
        )
    
    def load_dataset(self, filename):
        """Load dataset from CSV file."""
        points = []
        try:
            with open(filename, 'r') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        try:
                            parts = line.split(',')
                            if len(parts) >= 3:
                                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                                points.append([x, y, z])
                        except ValueError:
                            print(f"Warning: Invalid data format at line {line_num}: {line}")
            self.points = points
            return points
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return []
    
    def compute_distance_matrix(self):
        """Compute initial distance matrix between all points."""
        n = len(self.points)
        self.distance_matrix = [[float('inf')] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.euclidean_distance(self.points[i], self.points[j])
                self.distance_matrix[i][j] = dist
                self.distance_matrix[j][i] = dist
            self.distance_matrix[i][i] = 0
    
    def cluster_distance(self, cluster1_indices, cluster2_indices):
        """
        Calculate distance between two clusters based on linkage criterion.
        
        Args:
            cluster1_indices: List of point indices in first cluster
            cluster2_indices: List of point indices in second cluster
        
        Returns:
            Distance between clusters
        """
        distances = []
        
        for i in cluster1_indices:
            for j in cluster2_indices:
                distances.append(self.distance_matrix[i][j])
        
        if self.linkage == 'single':
            return min(distances)  # Single linkage: minimum distance
        elif self.linkage == 'complete':
            return max(distances)  # Complete linkage: maximum distance
        elif self.linkage == 'average':
            return sum(distances) / len(distances)  # Average linkage
        else:
            raise ValueError(f"Unknown linkage criterion: {self.linkage}")
    
    def find_closest_clusters(self, clusters):
        """Find the two closest clusters to merge."""
        min_distance = float('inf')
        closest_pair = None
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = self.cluster_distance(clusters[i], clusters[j])
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (i, j)
        
        return closest_pair, min_distance
    
    def fit(self):
        """
        Perform hierarchical agglomerative clustering.
        
        Returns:
            Final clusters as list of point lists
        """
        if not self.points:
            print("No data points loaded.")
            return []
        
        print(f"Starting hierarchical agglomerative clustering with {len(self.points)} points")
        print(f"Target clusters: {self.k}, Linkage: {self.linkage}")
        
        # Initialize distance matrix
        self.compute_distance_matrix()
        
        # Initialize clusters: each point starts as its own cluster
        clusters = [[i] for i in range(len(self.points))]
        self.cluster_history = []
        
        step = 0
        while len(clusters) > self.k:
            step += 1
            
            # Find closest pair of clusters
            closest_pair, min_distance = self.find_closest_clusters(clusters)
            
            if closest_pair is None:
                break
            
            i, j = closest_pair
            print(f"Step {step}: Merging cluster {i} (size {len(clusters[i])}) with cluster {j} (size {len(clusters[j])}) at distance {min_distance:.4f}")
            
            # Merge clusters i and j
            merged_cluster = clusters[i] + clusters[j]
            
            # Remove the merged clusters and add the new one
            # Remove in reverse order to maintain indices
            if i > j:
                clusters.pop(i)
                clusters.pop(j)
            else:
                clusters.pop(j)
                clusters.pop(i)
            
            clusters.append(merged_cluster)
            
            # Store history
            self.cluster_history.append({
                'step': step,
                'merged': (i, j),
                'distance': min_distance,
                'num_clusters': len(clusters)
            })
        
        print(f"Clustering completed after {step} merges")
        
        # Convert cluster indices to actual points
        self.clusters = []
        for cluster_indices in clusters:
            cluster_points = [self.points[i] for i in cluster_indices]
            self.clusters.append(cluster_points)
        
        return self.clusters
    
    def calculate_silhouette_coefficient(self):
        """
        Calculate the average silhouette coefficient for all points.
        Uses enhanced silhouette calculation with four metrics like in K-means implementation.
        """
        if len(self.clusters) < 2:
            return 0.0, "single_cluster"  # Silhouette is undefined for k=1
        
        # Calculate cluster centroids for enhanced metrics
        centroids = []
        for cluster in self.clusters:
            if len(cluster) > 0:
                centroid = [
                    sum(point[0] for point in cluster) / len(cluster),
                    sum(point[1] for point in cluster) / len(cluster),
                    sum(point[2] for point in cluster) / len(cluster)
                ]
                centroids.append(centroid)
        
        # Create a mapping from point to cluster index
        point_to_cluster = {}
        for cluster_idx, cluster in enumerate(self.clusters):
            for point in cluster:
                point_to_cluster[tuple(point)] = cluster_idx
        
        silhouette_scores = []
        
        for cluster_idx, cluster in enumerate(self.clusters):
            if len(cluster) == 0:
                continue
                
            for point in cluster:
                # Calculate a(i): average distance between cluster centroids
                current_centroid = centroids[cluster_idx]
                distances_to_other_centroids = []
                for other_cluster_idx, other_centroid in enumerate(centroids):
                    if other_cluster_idx != cluster_idx:
                        distance = self.euclidean_distance(current_centroid, other_centroid)
                        distances_to_other_centroids.append(distance)
                
                if len(distances_to_other_centroids) == 0:
                    a_i = 0  # Only one cluster exists
                else:
                    a_i = sum(distances_to_other_centroids) / len(distances_to_other_centroids)
                
                # Calculate b(i): minimum average distance to points in other clusters
                b_i = float('inf')
                for other_cluster_idx, other_cluster in enumerate(self.clusters):
                    if other_cluster_idx != cluster_idx and len(other_cluster) > 0:
                        distances_between = []
                        for other_point in other_cluster:
                            distances_between.append(self.euclidean_distance(point, other_point))
                        avg_distance_to_cluster = sum(distances_between) / len(distances_between)
                        b_i = min(b_i, avg_distance_to_cluster)
                
                if b_i == float('inf'):
                    b_i = 0
                
                # Calculate c(i): minimum of maximum distances between clusters
                c_i = float('inf')
                for other_cluster_idx, other_cluster in enumerate(self.clusters):
                    if other_cluster_idx != cluster_idx and len(other_cluster) > 0:
                        max_distance_between = 0
                        for other_point in other_cluster:
                            distance = self.euclidean_distance(point, other_point)
                            max_distance_between = max(max_distance_between, distance)
                        c_i = min(c_i, max_distance_between)
                
                if c_i == float('inf'):
                    c_i = 0
                
                # Calculate d(i): minimum of minimum distances between clusters
                d_i = float('inf')
                for other_cluster_idx, other_cluster in enumerate(self.clusters):
                    if other_cluster_idx != cluster_idx and len(other_cluster) > 0:
                        min_distance_between = float('inf')
                        for other_point in other_cluster:
                            distance = self.euclidean_distance(point, other_point)
                            min_distance_between = min(min_distance_between, distance)
                        d_i = min(d_i, min_distance_between)
                
                if d_i == float('inf'):
                    d_i = 0
                
                # Calculate silhouette scores for each metric
                # a(i): cohesion within cluster, separation between cluster centroids
                cohesion_a = sum(self.euclidean_distance(point, cluster_point) 
                                for cluster_point in cluster if cluster_point != point)
                cohesion_a = cohesion_a / max(1, len(cluster) - 1)
                
                silhouette_a = (a_i - cohesion_a) / max(a_i, cohesion_a) if max(a_i, cohesion_a) > 0 else 0
                silhouette_b = (b_i - cohesion_a) / max(b_i, cohesion_a) if max(b_i, cohesion_a) > 0 else 0
                silhouette_c = (c_i - cohesion_a) / max(c_i, cohesion_a) if max(c_i, cohesion_a) > 0 else 0
                silhouette_d = (d_i - cohesion_a) / max(d_i, cohesion_a) if max(d_i, cohesion_a) > 0 else 0
                
                silhouette_scores.append({
                    'a': silhouette_a,
                    'b': silhouette_b,
                    'c': silhouette_c,
                    'd': silhouette_d
                })
        
        if not silhouette_scores:
            return 0.0, "no_valid_scores"
        
        # Calculate average for each metric
        avg_a = sum(score['a'] for score in silhouette_scores) / len(silhouette_scores)
        avg_b = sum(score['b'] for score in silhouette_scores) / len(silhouette_scores)
        avg_c = sum(score['c'] for score in silhouette_scores) / len(silhouette_scores)
        avg_d = sum(score['d'] for score in silhouette_scores) / len(silhouette_scores)
        
        # Find the best performing metric
        metrics = {
            'a(i) - centroid separation, distance between centroids': avg_a,
            'b(i) - min avg separation, average distance between points in other clusters': avg_b,
            'c(i) - max separation, distance between farthest points': avg_c,
            'd(i) - min separation, distance between nearest points': avg_d
        }
        
        best_metric = max(metrics, key=metrics.get)
        best_score = metrics[best_metric]
        
        return best_score, best_metric
    
    def print_results(self):
        """Print clustering results."""
        print(f"\n=== Hierarchical Agglomerative Clustering Results (k={self.k}) ===")
        print(f"Linkage criterion: {self.linkage}")
        print(f"Total points clustered: {sum(len(cluster) for cluster in self.clusters)}")
        
        for i, cluster in enumerate(self.clusters):
            print(f"\nCluster {i+1}:")
            if len(cluster) > 0:
                # Calculate cluster statistics
                x_coords = [point[0] for point in cluster]
                y_coords = [point[1] for point in cluster]
                z_coords = [point[2] for point in cluster]
                
                centroid = [
                    sum(x_coords) / len(x_coords),
                    sum(y_coords) / len(y_coords),
                    sum(z_coords) / len(z_coords)
                ]
                
                print(f"  Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
                print(f"  Size: {len(cluster)} points")
                print(f"  X range: [{min(x_coords)}, {max(x_coords)}]")
                print(f"  Y range: [{min(y_coords)}, {max(y_coords)}]")
                print(f"  Z range: [{min(z_coords)}, {max(z_coords)}]")
        
        # Calculate and display silhouette coefficient
        silhouette, best_function = self.calculate_silhouette_coefficient()
        print(f"\nFinal Silhouette Coefficient: {silhouette:.4f}")
        print(f"Best performing function: {best_function}")
        
        # Interpret clustering quality
        if silhouette > 0.7:
            quality = "Excellent clustering"
        elif silhouette > 0.5:
            quality = "Good clustering"
        elif silhouette > 0.25:
            quality = "Reasonable clustering"
        elif silhouette > 0:
            quality = "Poor clustering"
        else:
            quality = "Very poor clustering"
        
        print(f"Clustering Quality: {quality}")
    
    def save_results(self, filename, original_dataset_filename=None):
        """Save clustering results to file."""
        with open(filename, 'w') as f:
            f.write(f"Hierarchical Agglomerative Clustering Results (k={self.k})\n")
            f.write("=" * 50 + "\n")
            # Show only the base filename without folder path
            import os
            if original_dataset_filename:
                dataset_name = os.path.basename(original_dataset_filename)
            else:
                dataset_name = "unknown_dataset.txt"
            f.write(f"{dataset_name}\n")
            f.write(f"Linkage criterion: {self.linkage}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Cluster Centroids:\n")
            for i, cluster in enumerate(self.clusters):
                if len(cluster) > 0:
                    centroid = [
                        sum(point[0] for point in cluster) / len(cluster),
                        sum(point[1] for point in cluster) / len(cluster),
                        sum(point[2] for point in cluster) / len(cluster)
                    ]
                    f.write(f"Cluster {i+1}: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})\n")
            
            f.write("\nCluster Assignments:\n")
            for i, cluster in enumerate(self.clusters):
                f.write(f"\nCluster {i+1} ({len(cluster)} points):\n")
                for point in cluster:
                    f.write(f"({point[0]}, {point[1]}, {point[2]})\n")
            
            final_silhouette, best_function = self.calculate_silhouette_coefficient()
            f.write(f"\nFinal Silhouette Coefficient: {final_silhouette:.4f}\n")
            f.write(f"Best performing function: {best_function}\n")
            f.write(f"Merge steps: {len(self.cluster_history)}\n")


def main():
    """Main function to run hierarchical agglomerative clustering."""
    print("=== Hierarchical Agglomerative Clustering Algorithm ===")
    
    # Get user input for filename
    user_input = input("Enter the dataset filename (q, r, or s-prime): ").strip()
    if not user_input:
        print("Error: Filename cannot be empty.")
        return
    
    # Check if this is a prime file and auto-direct to Prime folder
    import os
    if 'prime' in user_input.lower():
        # If user didn't include .txt extension, add it
        if not user_input.endswith('.txt'):
            user_input = user_input + '.txt'
        # Direct to Prime folder
        filename = os.path.join('Prime', user_input)
    else:
        filename = user_input
    
    print(f"Dataset: {filename}")
    
    # Get user input for k
    try:
        k = int(input("\nEnter the number of clusters (k): "))
        if k <= 0:
            print("k must be a positive integer.")
            return
    except ValueError:
        print("Invalid input. Please enter a positive integer.")
        return
    
    # Get user input for linkage criterion
    linkage_options = ['single', 'complete', 'average']
    print(f"\nAvailable linkage criteria: {', '.join(linkage_options)}")
    linkage = input("Enter linkage criterion (default: complete): ").strip().lower()
    if linkage not in linkage_options:
        linkage = 'complete'
        print(f"Using default linkage: {linkage}")
    
    # Create and run HAC clusterer
    clusterer = HierarchicalClusterer(k=k, linkage=linkage)
    
    # Load dataset
    points = clusterer.load_dataset(filename)
    if not points:
        return
    
    # Run clustering
    clusters = clusterer.fit()
    
    # Print results
    clusterer.print_results()
    
    # Save results
    # Create HAC_Cluster_Results directory if it doesn't exist
    results_dir = "HAC_Cluster_Results"
    os.makedirs(results_dir, exist_ok=True)
    
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    result_filename = os.path.join(results_dir, f'hac_k{k}_{linkage}_{base_filename}.txt')
    clusterer.save_results(result_filename, filename)
    print(f"\nResults saved to '{result_filename}'")


if __name__ == "__main__":
    main()
