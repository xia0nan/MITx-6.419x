import os
import numpy as np
import pandas as pd
import sys

def solve():
    print("Starting analysis for Problem 1.c...")
    
    data_dir = 'data'
    num_timesteps = 100
    
    # Check if data dir exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return

    # Using accumulators to avoid loading everything into memory at once if not needed,
    # though for this dataset size it might fit. But streaming is safer.
    total_sum_u = 0.0
    total_sum_v = 0.0
    total_count = 0

    print("Accumulating data from files...")
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
        
        # Add to sums
        total_sum_u += np.sum(u_data)
        total_sum_v += np.sum(v_data)
        
        # We assume u and v have same shape
        total_count += u_data.size

    print(f"\nProcessing complete.")
    
    if total_count == 0:
        print("Error: No data processed.")
        return
        
    avg_u = total_sum_u / total_count
    avg_v = total_sum_v / total_count
    
    print("-" * 30)
    print(f"RESULT:")
    print(f"Total elements processed: {total_count}")
    print(f"Average X velocity (u): {avg_u:.5f} km/h")
    print(f"Average Y velocity (v): {avg_v:.5f} km/h")
    print("-" * 30)

if __name__ == "__main__":
    solve()
