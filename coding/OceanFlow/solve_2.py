import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import random

def solve():
    print("Starting analysis for Problem 2 (Long-range correlations)...")
    
    data_dir = 'data'
    num_timesteps = 100
    rows = 504
    cols = 555
    grid_spacing = 3 # km
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return

    # Helper to load all data into a big array
    # Shape: (Time, Rows, Cols)
    def load_component(component_suffix):
        print(f"Loading {component_suffix} component...")
        data = np.zeros((num_timesteps, rows, cols), dtype=np.float32)
        for t in range(1, num_timesteps + 1):
            fname = os.path.join(data_dir, f'{t}{component_suffix}.csv')
            if os.path.exists(fname):
                data[t-1] = pd.read_csv(fname, header=None).values
            if t % 20 == 0:
                print(f"  Loaded {t}/{num_timesteps}", end='\r')
        print(f"\n  Done loading {component_suffix}.")
        return data

    u_data = load_component('u')
    v_data = load_component('v')

    # Flatten spatial dimensions for easier processing
    # New shape: (Time, N_points)
    # N_points = Rows * Cols
    u_flat = u_data.reshape(num_timesteps, -1)
    v_flat = v_data.reshape(num_timesteps, -1)
    
    # We can choose to analyze U, V, or Speed. 
    # The problem says: "compute the correlation coefficient along the x-direction, or the y-direction, or both."
    # Let's pick U (x-direction) for simplicity as it often shows strong zonal flows, 
    # or iterate to find the strongest. Let's start with U.
    
    data_matrix = u_flat
    
    print("Preprocessing data...")
    # Calculate std to identify valid (non-land) points and for standardization
    # std along time axis
    stds = np.std(data_matrix, axis=0)
    means = np.mean(data_matrix, axis=0)
    
    # Filter out land (std ~ 0)
    valid_mask = stds > 1e-6
    valid_indices = np.where(valid_mask)[0]
    
    print(f"Grid points: {rows*cols}")
    print(f"Valid water points: {len(valid_indices)}")
    
    # Standardize the data: (X - mean) / std
    # This allows correlation to be just dot product / N
    # We only keep valid points to save memory/compute
    
    # Extract valid columns
    valid_data = data_matrix[:, valid_indices]
    valid_means = means[valid_indices]
    valid_stds = stds[valid_indices]
    
    # Standardize
    # (Time, N_valid)
    norm_data = (valid_data - valid_means) / valid_stds
    
    # Optimization: Transpose to (N_valid, Time) for easier dot products?
    # Actually dot(v1, v2) where v are columns is easier if we keep as is
    # Cor(A, B) = mean( (A-muA)/sigA * (B-muB)/sigB )
    # Since we already normalized, Cor(A, B) = dot(A_norm, B_norm) / num_timesteps
    
    # Random sampling strategy
    num_probes = 500  # Number of random points to test against everyone else
    min_dist_km = 150 # Minimum distance to consider "long-range" (50 pixels * 3km = 150km)
    min_dist_sq_pixels = (min_dist_km / grid_spacing) ** 2
    
    print(f"Sampling {num_probes} random probe points...")
    
    # Randomly select indices from the 'valid_indices' array
    # We select indices *into* the valid_indices array
    probe_indices_local = np.random.choice(len(valid_indices), num_probes, replace=False)
    probe_indices_global = valid_indices[probe_indices_local] # The actual 1D index in full grid
    
    best_corr = 0.0
    best_pair_global = (-1, -1)
    best_pair_coords = ((0,0), (0,0))
    
    # Pre-calculate coordinates for all valid points to check distances quickly
    # Unravel all valid indices
    all_rows, all_cols = np.unravel_index(valid_indices, (rows, cols))
    
    # Convert probes to norm data subset
    # Shape: (Time, Num_Probes)
    probes_norm = norm_data[:, probe_indices_local]
    
    # Compute correlation matrix: result of dot product
    # (Num_Probes, Time) @ (Time, N_valid) -> (Num_Probes, N_valid)
    print("Computing correlations...")
    # matrix multiplication
    corr_matrix = (probes_norm.T @ norm_data) / num_timesteps
    
    print("Searching for long-range high correlations...")
    
    for i in range(num_probes):
        # probe global index
        p_idx_global = probe_indices_global[i]
        p_row, p_col = np.unravel_index(p_idx_global, (rows, cols))
        
        # Correlations for this probe with everyone
        probe_corrs = corr_matrix[i]
        
        # Filter by distance
        # Calculate squared dists to this probe for ALL valid points
        dists_sq = (all_rows - p_row)**2 + (all_cols - p_col)**2
        
        # Mask out close points (including itself)
        dist_mask = dists_sq > min_dist_sq_pixels
        
        # Apply mask
        far_corrs = probe_corrs[dist_mask]
        far_indices_local = np.where(dist_mask)[0] # Indices into valid_indices array
        
        if len(far_corrs) == 0:
            continue
            
        # Find max abs correlation
        # We want strongest signal, positive or negative
        abs_corrs = np.abs(far_corrs)
        max_idx_local_far = np.argmax(abs_corrs)
        max_val = abs_corrs[max_idx_local_far]
        signed_val = far_corrs[max_idx_local_far]
        
        if max_val > abs(best_corr):
            best_corr = signed_val
            
            # Recover global index of the match
            # far_indices_local[max_idx_local_far] gives the index into 'valid_indices'
            # then valid_indices[...] gives global 1D index
            match_idx_in_valid = far_indices_local[max_idx_local_far]
            match_idx_global = valid_indices[match_idx_in_valid]
            
            best_pair_global = (p_idx_global, match_idx_global)
            
            # Store Coords
            m_row, m_col = np.unravel_index(match_idx_global, (rows, cols))
            best_pair_coords = ((p_row, p_col), (m_row, m_col))
    
    print("-" * 30)
    print(f"RESULTS:")
    print(f"High correlation found: {best_corr:.4f}")
    
    p1_yx = best_pair_coords[0]
    p2_yx = best_pair_coords[1]
    
    p1_y_km, p1_x_km = p1_yx[0]*grid_spacing, p1_yx[1]*grid_spacing
    p2_y_km, p2_x_km = p2_yx[0]*grid_spacing, p2_yx[1]*grid_spacing
    
    print(f"Point A: Grid({p1_yx[1]}, {p1_yx[0]}) -> ({p1_x_km} km, {p1_y_km} km)")
    print(f"Point B: Grid({p2_yx[1]}, {p2_yx[0]}) -> ({p2_x_km} km, {p2_y_km} km)")
    
    dist_km = np.sqrt( (p1_x_km - p2_x_km)**2 + (p1_y_km - p2_y_km)**2 )
    print(f"Distance: {dist_km:.2f} km")
    
    # Visualization
    print("Generating visualization...")
    plt.figure(figsize=(10, 8))
    
    # Plot speed background (average speed)
    avg_speed = np.mean(np.sqrt(u_data**2 + v_data**2), axis=0)
    plt.imshow(avg_speed, origin='lower', cmap='viridis', extent=[0, cols*grid_spacing, 0, rows*grid_spacing])
    plt.colorbar(label='Average Speed (km/h)')
    
    # Plot points
    plt.plot(p1_x_km, p1_y_km, 'ro', markersize=10, markeredgecolor='white', label='Point A')
    plt.plot(p2_x_km, p2_y_km, 'ro', markersize=10, markeredgecolor='white', label='Point B')
    
    # Draw line? Maybe not, just points
    plt.title(f"Correlated Points (Corr: {best_corr:.3f})")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.legend()
    
    out_img = 'correlation_map.png'
    plt.savefig(out_img)
    print(f"Saved map to {out_img}")
    print("-" * 30)

if __name__ == "__main__":
    solve()
