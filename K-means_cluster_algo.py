#!/usr/bin/env python3
"""
K-means Clustering Algorithm Implementation for q, r, or s_prime.txt
This program implements the K-means clustering algorithm to cluster 3D points
into k user-specified clusters. k value is input by the user upon execution.
Then calculates silhouette coefficient using enhanced method with four metrics. The metric with the best score is reported.
"""

import math
import random
import numpy as np

class KMeansClusterer:
    def __init__(self, k, max_iterations=100, tolerance=1e-4, random_seed=42):
        """
        Initialize K-means clusterer.
        
        Args:
            k: Number of clusters
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            random_seed: Seed for reproducible results
        """
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_seed = random_seed
        self.centroids = []
        self.clusters = []
        self.points = []
        self.iteration_history = []
        
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
    
    def initialize_centroids(self):
        """Initialize centroids using random points from the dataset."""
        if len(self.points) < self.k:
            raise ValueError(f"Cannot create {self.k} clusters with only {len(self.points)} points")
        
        random.seed(self.random_seed)
        
        # Use K-means++ initialization for better results
        centroids = []
        
        # Choose first centroid randomly
        first_centroid = random.choice(self.points).copy()
        centroids.append(first_centroid)
        
        # Choose remaining centroids using K-means++
        for _ in range(1, self.k):
            distances = []
            for point in self.points:
                min_dist = min(self.euclidean_distance(point, centroid) for centroid in centroids)
                distances.append(min_dist**2)
            
            # Choose next centroid with probability proportional to squared distance
            total_distance = sum(distances)
            if total_distance == 0:
                # If all distances are 0, choose randomly
                next_centroid = random.choice(self.points).copy()
            else:
                probabilities = [d/total_distance for d in distances]
                cumulative_prob = 0
                r = random.random()
                for i, prob in enumerate(probabilities):
                    cumulative_prob += prob
                    if r <= cumulative_prob:
                        next_centroid = self.points[i].copy()
                        break
            centroids.append(next_centroid)
        
        self.centroids = centroids
        print(f"Initialized {self.k} centroids using K-means++")
        for i, centroid in enumerate(self.centroids):
            print(f"  Centroid {i+1}: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
    
    def assign_points_to_clusters(self):
        """Assign each point to the nearest centroid."""
        clusters = [[] for _ in range(self.k)]
        
        for point in self.points:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(point)
        
        self.clusters = clusters
        return clusters
    
    def update_centroids(self):
        """Update centroids based on current cluster assignments."""
        new_centroids = []
        centroid_changes = []
        
        for i, cluster in enumerate(self.clusters):
            if len(cluster) == 0:
                # If cluster is empty, keep the old centroid
                new_centroids.append(self.centroids[i].copy())
                centroid_changes.append(0)
            else:
                # Calculate mean of points in cluster
                new_centroid = [
                    sum(point[0] for point in cluster) / len(cluster),
                    sum(point[1] for point in cluster) / len(cluster),
                    sum(point[2] for point in cluster) / len(cluster)
                ]
                
                # Calculate how much the centroid moved
                change = self.euclidean_distance(self.centroids[i], new_centroid)
                centroid_changes.append(change)
                new_centroids.append(new_centroid)
        
        self.centroids = new_centroids
        return max(centroid_changes)
    
    def calculate_silhouette_coefficient(self):
        """
        Calculate the average silhouette coefficient for all points.
        
        Uses enhanced silhouette calculation with four metrics:
        - a(i): average distance between the centers of two clusters (centroid separation, distance between centroids)
        - b(i): minimum average distance to points in other clusters (separation, average distance between points in other clusters)
        - c(i): minimum of maximum distances between farthest points in clusters (max separation, distance between farthest points)
        - d(i): minimum of minimum distances between nearest points in clusters (min separation, distance between nearest points)
        
        Final score returns the best normalized silhouette score (range [-1,1]) among all metrics across all points.
        Uses standard silhouette formula: (separation - cohesion) / max(separation, cohesion) for each metric. 
        """
        if len(self.clusters) < 2:
            return 0  # Silhouette is undefined for k=1
        
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
                # Calculate a(i): average distance between the centers of two clusters
                # For each point, calculate average distance from current cluster centroid to all other cluster centroids
                current_centroid = self.centroids[cluster_idx]
                distances_to_other_centroids = []
                for other_cluster_idx, other_centroid in enumerate(self.centroids):
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
                
                # Calculate c(i): minimum of maximum distances between farthest points in clusters
                c_i = float('inf')
                for other_cluster_idx, other_cluster in enumerate(self.clusters):
                    if other_cluster_idx != cluster_idx and len(other_cluster) > 0:
                        # Find the maximum distance between any point in current cluster 
                        # and any point in the other cluster (farthest two points)
                        max_distance_between_clusters = 0
                        for current_cluster_point in cluster:
                            for other_point in other_cluster:
                                distance = self.euclidean_distance(current_cluster_point, other_point)
                                max_distance_between_clusters = max(max_distance_between_clusters, distance)
                        c_i = min(c_i, max_distance_between_clusters)
                
                # Calculate d(i): minimum of minimum distances between nearest points in clusters
                d_i = float('inf')
                for other_cluster_idx, other_cluster in enumerate(self.clusters):
                    if other_cluster_idx != cluster_idx and len(other_cluster) > 0:
                        # Find the minimum distance between any point in current cluster 
                        # and any point in the other cluster (nearest two points)
                        min_distance_between_clusters = float('inf')
                        for current_cluster_point in cluster:
                            for other_point in other_cluster:
                                distance = self.euclidean_distance(current_cluster_point, other_point)
                                min_distance_between_clusters = min(min_distance_between_clusters, distance)
                        d_i = min(d_i, min_distance_between_clusters)
                
                # Calculate normalized silhouette scores for each metric with tracking
                score_data = []  # Store (score, function_name) tuples
                
                # Calculate normalized scores for b(i), c(i), d(i)
                if b_i != float('inf'):
                    if max(a_i, b_i) == 0:
                        b_score = 0
                    else:
                        b_score = (b_i - a_i) / max(a_i, b_i)
                    score_data.append((b_score, 'b(i) - avg separation'))
                
                if c_i != float('inf'):
                    if max(a_i, c_i) == 0:
                        c_score = 0
                    else:
                        c_score = (c_i - a_i) / max(a_i, c_i)
                    score_data.append((c_score, 'c(i) - max separation'))
                
                if d_i != float('inf'):
                    if max(a_i, d_i) == 0:
                        d_score = 0
                    else:
                        d_score = (d_i - a_i) / max(a_i, d_i)
                    score_data.append((d_score, 'd(i) - min separation'))
                
                # Store all score data for this point
                silhouette_scores.extend(score_data)
        
        # Find the best score and corresponding function
        if len(silhouette_scores) == 0:
            return 0, "none"
        
        best_score, best_function = max(silhouette_scores, key=lambda x: x[0])
        return best_score, best_function
    
    def fit(self):
        """Run the K-means algorithm."""
        if not self.points:
            print("No data points loaded. Cannot perform clustering.")
            return
        
        print(f"\nRunning K-means clustering with k={self.k}")
        print(f"Dataset size: {len(self.points)} points")
        
        # Initialize centroids
        self.initialize_centroids()
        
        # Iteratively assign points and update centroids
        for iteration in range(self.max_iterations):
            # Assign points to clusters
            self.assign_points_to_clusters()
            
            # Update centroids and check for convergence
            max_change = self.update_centroids()
            
            # Calculate silhouette coefficient for this iteration
            silhouette_score, best_function = self.calculate_silhouette_coefficient()
            
            # Store iteration history
            self.iteration_history.append({
                'iteration': iteration + 1,
                'silhouette_score': silhouette_score,
                'best_function': best_function,
                'max_centroid_change': max_change,
                'centroids': [c.copy() for c in self.centroids]
            })
            
            print(f"Iteration {iteration + 1}: Silhouette = {silhouette_score:.4f} ({best_function}), Max centroid change = {max_change:.4f}")
            
            # Check for convergence
            if max_change < self.tolerance:
                print(f"Converged after {iteration + 1} iterations!")
                break
        else:
            print(f"Reached maximum iterations ({self.max_iterations})")
        
        # Final cluster assignment
        self.assign_points_to_clusters()
        
        return self.centroids, self.clusters
    
    def print_results(self):
        """Print clustering results."""
        print(f"\n=== K-means Clustering Results (k={self.k}) ===")
        
        total_points = sum(len(cluster) for cluster in self.clusters)
        print(f"Total points clustered: {total_points}")
        
        for i, (centroid, cluster) in enumerate(zip(self.centroids, self.clusters)):
            print(f"\nCluster {i+1}:")
            print(f"  Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
            print(f"  Size: {len(cluster)} points")
            
            if len(cluster) > 0:
                # Calculate cluster statistics
                x_coords = [p[0] for p in cluster]
                y_coords = [p[1] for p in cluster]
                z_coords = [p[2] for p in cluster]
                
                print(f"  X range: [{min(x_coords):.1f}, {max(x_coords):.1f}]")
                print(f"  Y range: [{min(y_coords):.1f}, {max(y_coords):.1f}]")
                print(f"  Z range: [{min(z_coords):.1f}, {max(z_coords):.1f}]")
        
                # Calculate and print final silhouette coefficient
        final_silhouette, best_function = self.calculate_silhouette_coefficient()
        print(f"\nFinal Silhouette Coefficient: {final_silhouette:.4f}")
        print(f"Best performing function: {best_function}")
        
        # Interpret silhouette coefficient quality
        if final_silhouette >= 0.7:
            quality = "Excellent clustering"
        elif final_silhouette >= 0.5:
            quality = "Good clustering"
        elif final_silhouette >= 0.25:
            quality = "Reasonable clustering"
        else:
            quality = "Poor clustering"
        print(f"Clustering Quality: {quality}")
        
   
    def save_results(self, filename, original_dataset_filename=None):
        """Save clustering results to file."""
        with open(filename, 'w') as f:
            f.write(f"K-means Clustering Results (k={self.k})\n")
            f.write("=" * 50 + "\n")
            # Show only the base filename without folder path
            import os
            if original_dataset_filename:
                dataset_name = os.path.basename(original_dataset_filename)
            else:
                # Fallback to old method if original filename not provided
                dataset_name = filename.replace(f'kmeans_results_k{self.k}_', '').replace('.txt', '') + '.txt'
            f.write(f"{dataset_name}" + "\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Centroids:\n")
            for i, centroid in enumerate(self.centroids):
                f.write(f"Cluster {i+1}: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})\n")
            
            f.write("\nCluster Assignments:\n")
            for i, cluster in enumerate(self.clusters):
                f.write(f"\nCluster {i+1} ({len(cluster)} points):\n")
                for point in cluster:
                    f.write(f"({point[0]}, {point[1]}, {point[2]})\n")
            
            final_silhouette, best_function = self.calculate_silhouette_coefficient()
            f.write(f"\nFinal Silhouette Coefficient: {final_silhouette:.4f}\n")
            f.write(f"Best performing function: {best_function}\n")
            f.write(f"Iterations: {len(self.iteration_history)}\n")
    
def main():
    """Main function to run K-means clustering."""
    print("=== K-means Clustering Algorithm ===")
    
    # Get user input for filename
    user_input = input("Enter the dataset filename (q, r or, s-prime): ").strip()
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
    
    
    # Create and run K-means clusterer
    clusterer = KMeansClusterer(k)
    
    # Load dataset
    points = clusterer.load_dataset(filename)
    if not points:
        return
    
    # Run clustering
    centroids, clusters = clusterer.fit()
    
    # Print results
    clusterer.print_results()
    
    # Save results
    # Extract base filename without extension and path for result file naming
    import os
    
    # Create K-means_Cluster_Results directory if it doesn't exist
    results_dir = "K-means_Cluster_Results"
    os.makedirs(results_dir, exist_ok=True)
    
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    result_filename = os.path.join(results_dir, f'k{k}_{base_filename}.txt')
    clusterer.save_results(result_filename, filename)
    print(f"\nResults saved to '{result_filename}'")

if __name__ == "__main__":
    main()
