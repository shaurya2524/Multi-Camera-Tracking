import numpy as np
import json

# Load the homography matrix from the .npy file
H = np.load('homography_matrix.npy')

# Save it as homography.json in the required format
with open('homography.json', 'w') as f:
    json.dump({'homography': H.tolist()}, f, indent=2)

print("homography.json created successfully.")