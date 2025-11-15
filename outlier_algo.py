#!/usr/bin/env python3
"""
Outlier Detection in Dataset_X.txt using Euclidean Distance
This script identifies outliers by calculating the average distance from each point
to all other points and flagging points that exceed a Z-score threshold. Currently, the threshold is set to 2.0.
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two 3D points."""
    return math.sqrt(
        (point1[0] - point2[0])**2 + 
        (point1[1] - point2[1])**2 + 
        (point1[2] - point2[2])**2
    )

def load_dataset(filename):
    """Load dataset from CSV file."""
    points = []
    try:
        with open(filename, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line:
                    try:
                        x, y, z = map(int, line.split(','))
                        points.append((x, y, z, line_num))
                    except ValueError:
                        print(f"Warning: Invalid data format at line {line_num}: {line}")
        return points
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []

def calculate_average_distances(points):
    """Calculate average distance from each point to all other points."""
    n = len(points)
    avg_distances = []
    
    print(f"Calculating distances for {n} points...")
    
    for i, point1 in enumerate(points):
        total_distance = 0
        count = 0
        
        for j, point2 in enumerate(points):
            if i != j:  # Don't calculate distance to itself
                distance = euclidean_distance(point1[:3], point2[:3])
                total_distance += distance
                count += 1
        
        avg_distance = total_distance / count if count > 0 else 0
        avg_distances.append((point1, avg_distance))
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{n} points...")
    
    return avg_distances

def find_outliers_zscore(avg_distances, threshold=2.0):## <--- CHANGE Z SCORE THRESHOLD ##
    """Find outliers using Z-score method."""
    distances = [dist for _, dist in avg_distances]
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    outliers = []
    for point_data, distance in avg_distances:
        if std_dist > 0:
            z_score = abs(distance - mean_dist) / std_dist
            if z_score > threshold:
                outliers.append((point_data, distance, z_score))
    
    return outliers, mean_dist, std_dist

def print_statistics(avg_distances):
    """Print statistical summary of distances."""
    distances = [dist for _, dist in avg_distances]
    
    print(f"\n=== Distance Statistics ===")
    print(f"Total points: {len(distances)}")
    print(f"Mean distance: {np.mean(distances):.2f}")
    print(f"Median distance: {np.median(distances):.2f}")
    print(f"Standard deviation: {np.std(distances):.2f}")
    print(f"Min distance: {min(distances):.2f}")
    print(f"Max distance: {max(distances):.2f}")

def visualize_results(points, outliers_zscore, filename):
    """Create 3D visualization of points and outliers."""
    fig = plt.figure(figsize=(10, 8))
    
    # Extract coordinates
    x_coords = [p[0] for p, _ in points]
    y_coords = [p[1] for p, _ in points]
    z_coords = [p[2] for p, _ in points]
    
    # Z-score outliers  
    outlier_zscore_coords = [(p[0], p[1], p[2]) for p, _, _ in outliers_zscore]
    
    # Single plot: Z-score Method
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords, c='blue', alpha=0.6, s=20, label='Normal points')
    if outlier_zscore_coords:
        outlier_x, outlier_y, outlier_z = zip(*outlier_zscore_coords)
        ax.scatter(outlier_x, outlier_y, outlier_z, c='red', s=100, label='Outliers (Z-score)')
    ax.set_title('Outlier Detection - Z-score Method')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.tight_layout()
    # Create Outlier_Plots directory if it doesn't exist
    plots_dir = "Outlier_Plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate dynamic filename for visualization
    base_name = filename.replace('.txt', '').replace('.csv', '').replace('Dataset_', '')
    viz_filename = os.path.join(plots_dir, f'outlier_analysis_{base_name}.png')
    plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
    plt.show()
    return viz_filename

def main():
    """Main function to run outlier detection."""
    print("=== Outlier Detection Algorithm ===")
    
    # Get user input for dataset filename
    filename = input("Enter the dataset filename (Dataset_Q, R, or S.txt): ").strip()
    if not filename:
        print("Error: Filename cannot be empty.")
        return
    
    print(f"Loading dataset from '{filename}'...")
    points = load_dataset(filename)
    
    if not points:
        print("No valid data found. Exiting.")
        return
    
    print(f"Loaded {len(points)} points successfully.")
    
    # Calculate average distances
    avg_distances = calculate_average_distances(points)
    
    # Print statistics
    print_statistics(avg_distances)
    
    # Z-score based outlier detection
    print(f"\n=== Z-score-based Outlier Detection ===")
    outliers_zscore, mean_dist, std_dist = find_outliers_zscore(avg_distances)
    print(f"Mean distance: {mean_dist:.2f}, Std dev: {std_dist:.2f}")
    print(f"Found {len(outliers_zscore)} outliers using Z-score method (threshold=2.0):")
    
    for i, (point_data, distance, z_score) in enumerate(outliers_zscore):
        x, y, z, line_num = point_data
        print(f"  {i+1}. Point ({x}, {y}, {z}) at line {line_num}, avg distance: {distance:.2f}, Z-score: {z_score:.2f}")
    
    # Create output directories if they don't exist
    outlier_readouts_dir = "Outlier_Readouts"
    prime_dir = "Prime"
    os.makedirs(outlier_readouts_dir, exist_ok=True)
    os.makedirs(prime_dir, exist_ok=True)
    
    # Generate dynamic output filenames based on input dataset
    base_filename = filename.replace('.txt', '').replace('.csv', '')
    results_filename = os.path.join(outlier_readouts_dir, f'outlier_results_{base_filename.replace("Dataset_", "")}.txt')
    prime_filename = os.path.join(prime_dir, f'{base_filename.replace("Dataset_", "").lower()}-prime.txt')
    
    # Save results to file
    with open(results_filename, 'w') as f:
        f.write(f"Outlier Detection Results for {filename}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Z-score Method Outliers:\n")
        for point_data, distance, z_score in outliers_zscore:
            x, y, z, line_num = point_data
            f.write(f"Point ({x}, {y}, {z}) at line {line_num}, avg distance: {distance:.2f}, Z-score: {z_score:.2f}\n")
        
        # Add clean dataset summary
        outlier_line_numbers = {point_data[3] for point_data, _, _ in outliers_zscore}
        clean_count = len([p for p in points if p[3] not in outlier_line_numbers])
        f.write(f"\nClean Dataset Summary:\n")
        f.write(f"Total original points: {len(points)}\n")
        f.write(f"Outliers detected: {len(outliers_zscore)}\n")
        f.write(f"Clean points remaining: {clean_count}\n")
    
    # Create set of outlier line numbers for quick lookup
    outlier_line_numbers = {point_data[3] for point_data, _, _ in outliers_zscore}
    
    # Save clean data (non-outliers) to prime file
    clean_points = []
    for point in points:
        x, y, z, line_num = point
        if line_num not in outlier_line_numbers:
            clean_points.append((x, y, z))
    
    with open(prime_filename, 'w') as f:
        for x, y, z in clean_points:
            f.write(f"{x},{y},{z}\n")
    
    print(f"\nResults saved to '{results_filename}'")
    print(f"Clean dataset ({len(clean_points)} points after removing {len(outliers_zscore)} outliers) saved to '{prime_filename}'")
    
    # Create visualization
    try:
        viz_filename = visualize_results(avg_distances, outliers_zscore, filename)
        print(f"Visualization saved as '{viz_filename}'")
    except Exception as e:
        print(f"Could not create visualization: {e}")

if __name__ == "__main__":
    main()
