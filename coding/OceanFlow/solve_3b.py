import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def solve():
    print("Starting Debris Search Simulation (Problem 3.b)...")
    
    data_dir = 'data'
    num_timesteps = 100
    rows = 504
    cols = 555
    grid_spacing = 3.0 # km
    dt = 1.0 # hour
    total_time = 120.0 # hours
    
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found.")
        return

    # Load velocity fields
    print("Loading velocity fields...")
    u_field = np.zeros((num_timesteps, rows, cols), dtype=np.float32)
    v_field = np.zeros((num_timesteps, rows, cols), dtype=np.float32)
    
    for t in range(1, num_timesteps + 1):
        if t % 20 == 0:
            print(f"Loading {t}/{num_timesteps}...", end='\r')
        try:
            u_path = os.path.join(data_dir, f'{t}u.csv')
            v_path = os.path.join(data_dir, f'{t}v.csv')
            if os.path.exists(u_path):
                u_field[t-1] = pd.read_csv(u_path, header=None).values
            if os.path.exists(v_path):
                v_field[t-1] = pd.read_csv(v_path, header=None).values
        except Exception:
            pass
            
    print("\nData loaded.")
    
    # Simulation Parameters
    center_x = 300.0 # km
    center_y = 1050.0 # km
    
    # We will test 3 different standard deviations (sigmas) for the Gaussian
    # Sigma in km
    sigmas = [9.0, 30.0, 90.0] # 3 grid units, 10 grid units, 30 grid units
    
    target_times = [48.0, 72.0, 120.0]
    
    results = {}
    
    for sigma in sigmas:
        print(f"\nRunning simulation for Sigma = {sigma} km...")
        
        num_particles = 1000
        
        # Initialize Gaussian distribution
        p_x = np.random.normal(center_x, sigma, num_particles)
        p_y = np.random.normal(center_y, sigma, num_particles)
        
        # Track initial mean
        init_mean_x = np.mean(p_x)
        init_mean_y = np.mean(p_y)
        
        current_time = 0.0
        
        # Store positions for plotting at target times
        snapshots = {}
        
        # Run simulation
        while current_time <= total_time:
            # Check for exact matches or crossing target times (simplified check)
            # Since dt=1, we will hit integers exactly if started at 0
            if current_time in target_times or (current_time - 1.0 in target_times): 
                 # This logic is a bit loose, better to check proximity
                 pass
            
            # Store snapshots at exactly the target times (dt is 1.0, so exact match works)
            if current_time in target_times:
                snapshots[current_time] = (p_x.copy(), p_y.copy())
            
            if current_time >= total_time:
                break
                
            # Physics Step (Same as 3.a)
            idx_c = np.round(p_x / grid_spacing).astype(int)
            idx_r = np.round(p_y / grid_spacing).astype(int)
            idx_c = np.clip(idx_c, 0, cols - 1)
            idx_r = np.clip(idx_r, 0, rows - 1)
            
            time_idx = int(current_time // 3)
            if time_idx >= num_timesteps: time_idx = num_timesteps - 1
            
            u_p = u_field[time_idx, idx_r, idx_c]
            v_p = v_field[time_idx, idx_r, idx_c]
            
            p_x += u_p * dt
            p_y += v_p * dt
            
            current_time += dt
            
        results[sigma] = snapshots
        
        # Calculate statistics for this sigma
        print(f"Results for Sigma={sigma} km:")
        for t in target_times:
            px, py = snapshots[t]
            mean_x = np.mean(px)
            mean_y = np.mean(py)
            std_x = np.std(px)
            std_y = np.std(py)
            print(f"  T={t}h: Mean=({mean_x:.2f}, {mean_y:.2f}), Stddev=({std_x:.2f}, {std_y:.2f})")

    # Plotting
    print("\nGenerating comparison plots...")
    
    # Create a plot for each time, showing all sigmas
    for t in target_times:
        plt.figure(figsize=(10, 8))
        plt.title(f"Debris Cloud Location at T = {t} hours")
        plt.xlim(0, cols*grid_spacing)
        plt.ylim(0, rows*grid_spacing)
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        
        # Plot background speed? maybe too busy. Just white.
        
        colors = ['blue', 'green', 'red']
        for i, sigma in enumerate(sigmas):
            px, py = results[sigma][t]
            plt.scatter(px, py, s=10, alpha=0.5, label=f'Sigma={sigma} km', color=colors[i])
            
            # Mark the mean
            mx, my = np.mean(px), np.mean(py)
            plt.plot(mx, my, 'k+', markersize=15, markeredgewidth=2)
            
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        fname = f"debris_t{int(t)}.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")

if __name__ == "__main__":
    solve()
