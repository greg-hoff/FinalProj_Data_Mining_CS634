import random
import numpy as np

# Set random seed for reproducibility
random.seed(26)
np.random.seed(26)

# Generate 500 random 3D points with integer coordinates
points = []
for i in range(500):
    x = random.randint(-100, 100)
    y = random.randint(-100, 100)
    z = random.randint(-100, 100)
    points.append(f'{x},{y},{z}')

# Write to Dataset_Q.txt
with open('Dataset_Q.txt', 'w') as f:
    for point in points:
        f.write(point + '\n')

print(f'Generated 500 3D points and saved to Dataset_Q.txt')
print(f'Sample points: {points[:5]}')