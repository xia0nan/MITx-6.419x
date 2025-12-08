import os
import numpy as np
import pandas as pd
import sys

def solve():
    print("Starting analysis for Problem 1.b...")
    
    data_dir = 'data'
    num_timesteps = 100
    grid_spacing = 3  # km
    time_spacing = 3  # hours
    
    # Check if data dir exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return

    global_max_u = -float('inf')
    best_t_idx = -1
    best_row = -1
    best_col = -1

    print("Scanning data files for maximum x-axis velocity...")
    for t in range(1, num_timesteps + 1):
        u_file = os.path.join(data_dir, f'{t}u.csv')
        
        if not os.path.exists(u_file):
            print(f"Missing file for timestep {t}")
            continue
            
        # percentage progress
        if t % 10 == 0:
            print(f"Processing timestep {t}/{num_timesteps}...", end='\r')

        # Read only U component
        u_data = pd.read_csv(u_file, header=None).values
        
        # Find max in this file
        current_max = np.max(u_data)
        
        if current_max > global_max_u:
            global_max_u = current_max
            # Get location of max
            # argmax returns linear index, unravel_index converts to (row, col)
            max_loc = np.unravel_index(np.argmax(u_data, axis=None), u_data.shape)
            best_row = max_loc[0]
            best_col = max_loc[1]
            best_t_idx = t - 1 # 0-indexed time index

    print(f"\nAnalysis complete.")
    
    if best_t_idx == -1:
        print("Error: No data found.")
        return

    # Convert to physical units
    # Time: index * 3 hours
    # X: col * 3 km
    # Y: row * 3 km
    
    final_time = best_t_idx * time_spacing
    final_x = best_col * grid_spacing
    final_y = best_row * grid_spacing
    
    print("-" * 30)
    print(f"RESULT:")
    print(f"Maximum X-axis velocity: {global_max_u}")
    print(f"Time: {final_time} hours")
    print(f"X coordinate: {final_x} km")
    print(f"Y coordinate: {final_y} km")
    print("-" * 30)

if __name__ == "__main__":
    solve()
