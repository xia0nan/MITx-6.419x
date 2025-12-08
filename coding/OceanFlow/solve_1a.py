import os
import numpy as np
import pandas as pd
import sys

def solve():
    print("Starting analysis for Problem 1.a...")
    
    data_dir = 'data'
    num_timesteps = 100
    grid_spacing = 3  # km
    
    # Check if data dir exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return

    # Load first file to get dimensions
    print("Checking dimensions...")
    try:
        sample_u = pd.read_csv(os.path.join(data_dir, '1u.csv'), header=None).values
        rows, cols = sample_u.shape
        print(f"Grid Size: {rows} rows (Y) x {cols} columns (X)")
    except Exception as e:
        print(f"Error reading sample file: {e}")
        return

    # Pre-allocate array for speeds
    # We want to store speed for every point at every timestep
    # Shape: (Time, Y, X)
    print(f"Allocating memory for {num_timesteps} timesteps...")
    speeds = np.zeros((num_timesteps, rows, cols), dtype=np.float64)

    print("Loading data files...")
    for t in range(1, num_timesteps + 1):
        u_file = os.path.join(data_dir, f'{t}u.csv')
        v_file = os.path.join(data_dir, f'{t}v.csv')
        
        if not os.path.exists(u_file) or not os.path.exists(v_file):
            print(f"Missing files for timestep {t}")
            continue
            
        # percentage progress
        if t % 10 == 0:
            print(f"Processing timestep {t}/{num_timesteps}...", end='\r')

        u_data = pd.read_csv(u_file, header=None).values
        v_data = pd.read_csv(v_file, header=None).values

        # Calculate speed magnitude
        # speed = sqrt(u^2 + v^2)
        speeds[t-1] = np.sqrt(u_data**2 + v_data**2)
    
    print(f"\nData loaded. Calculating variance across time...")
    
    # Calculate variance along axis 0 (time)
    # Result shape: (Y, X)
    variances = np.var(speeds, axis=0)
    
    print(f"Variance map shape: {variances.shape}")
    print(f"Max variance found: {np.max(variances)}")
    print(f"Min variance (including zeros): {np.min(variances)}")

    # Filter out zero variances (land/border)
    # Using a small epsilon just in case, or strict 0 if data is clean
    # Problem says: "Remove any locations with a variance of zero"
    # Let's count how many are zero
    zero_mask = variances == 0
    non_zero_variances = variances[~zero_mask]
    
    print(f"Number of points with zero variance: {np.sum(zero_mask)}")
    print(f"Number of points with non-zero variance: {len(non_zero_variances)}")
    
    if len(non_zero_variances) == 0:
        print("Error: All variances are zero!")
        return

    min_var = np.min(non_zero_variances)
    print(f"Smallest non-zero variance: {min_var}")

    # Find the indices of this minimum variance in the original (Y, X) grid
    # np.where returns (row_indices, col_indices)
    min_locs = np.where(variances == min_var)
    
    # There might be multiple points with the same minimum, take the first one
    y_idx = min_locs[0][0] # Row -> Y axis index
    x_idx = min_locs[1][0] # Col -> X axis index
    
    print(f"Found minimum at Grid Indices: Row (Y)={y_idx}, Col (X)={x_idx}")
    
    # Convert to Kilometers
    # Origin (0,0) is bottom-left
    # x_km = col_idx * 3
    # y_km = row_idx * 3
    
    x_km = x_idx * grid_spacing
    y_km = y_idx * grid_spacing
    
    print("-" * 30)
    print(f"RESULT:")
    print(f"X coordinate: {x_km} km")
    print(f"Y coordinate: {y_km} km")
    print("-" * 30)

if __name__ == "__main__":
    solve()
