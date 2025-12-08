import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def solve():
    print("Starting Particle Tracking Simulation (Problem 3.a)...")
    
    data_dir = 'data'
    num_timesteps = 100
    rows = 504
    cols = 555
    grid_spacing = 3.0 # km
    dt = 1.0 # hour (simulation time step)
    total_time = 300.0 # hours
    
    # Check data
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found.")
        return

    # Load all data into memory for fast lookup
    # Shape: (Time, Rows, Cols)
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
        except Exception as e:
            print(f"Error loading time {t}: {e}")
            
    print("\nData loaded.")
    
    # Initialize Particles
    num_particles = 2000
    
    # Random positions in km
    # X range: [0, cols * 3)
    # Y range: [0, rows * 3)
    max_x = cols * grid_spacing
    max_y = rows * grid_spacing
    
    # State: (x, y)
    particles_x = np.random.uniform(0, max_x, num_particles)
    particles_y = np.random.uniform(0, max_y, num_particles)
    
    # Colors for distinction (random)
    colors = np.random.rand(num_particles, 3)
    
    # Store history for snapshots? No, just store snapshots when reached.
    snapshots = []
    snapshot_times = [0.0, 100.0, 200.0, 300.0]
    current_snapshot_idx = 0
    
    print(f"Simulating {num_particles} particles for {total_time} hours...")
    
    # Simulation Loop
    current_time = 0.0
    while current_time <= total_time:
        # Check if we need to take a snapshot
        if current_snapshot_idx < len(snapshot_times) and current_time >= snapshot_times[current_snapshot_idx]:
            print(f"Taking snapshot at t={current_time:.1f} hours")
            snapshots.append({
                'time': current_time,
                'x': particles_x.copy(),
                'y': particles_y.copy()
            })
            current_snapshot_idx += 1
            
        if current_time >= total_time:
            break
            
        # Determine Flow for each particle
        
        # 1. Map physical coord to nearest grid index
        # index = round(coord / 3)
        # We need to clamp indices to [0, max_idx] to avoid segfaults/errors
        # particles leaving area will just stick to boundary velocity or zero? 
        # Usually zero if out of bounds. Let's clamp to valid range for lookup, 
        # but if it really goes out, maybe handled by "nearest" being the edge.
        
        idx_c = np.round(particles_x / grid_spacing).astype(int)
        idx_r = np.round(particles_y / grid_spacing).astype(int)
        
        # Clamp to valid array indices
        # If particle is far outside, this effectively extends the edge flow outwards
        # which might not be physically accurate but fits "nearest neighbor"
        # However, problem says "Flow inland is zero".
        # Let's ensure strict bounds for array access
        idx_c = np.clip(idx_c, 0, cols - 1)
        idx_r = np.clip(idx_r, 0, rows - 1)
        
        # 2. Determine Time Index
        # Data interval is 3 hours.
        # file 1u.csv (index 0) corresponds to t=0
        # For nearest neighbor in time? Or sample-hold?
        # "At time t... flow is given by data point."
        # Usually zero-order hold: data[0] is valid for [0, 3).
        time_idx = int(current_time // 3)
        
        # Handle end of data
        if time_idx >= num_timesteps:
            time_idx = num_timesteps - 1
            
        # 3. Lookup Velocities
        # Vectorized lookup
        # u = u_field[time_idx, idx_r, idx_c]
        u_p = u_field[time_idx, idx_r, idx_c]
        v_p = v_field[time_idx, idx_r, idx_c]
        
        # 4. Integrate
        # x_new = x + u * dt
        particles_x += u_p * dt
        particles_y += v_p * dt
        
        # Increment time
        current_time += dt

    # Plotting
    print("Generating plots...")
    
    # Helper to plot
    def plot_state(snap_idx, snap_data):
        plt.figure(figsize=(10, 10))
        plt.title(f"Particle Trajectories - T = {snap_data['time']:.0f} hours")
        plt.xlim(0, max_x)
        plt.ylim(0, max_y)
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        
        # Plot particles
        plt.scatter(snap_data['x'], snap_data['y'], s=2, c=colors, alpha=0.6)
        
        filename = f"particles_t{int(snap_data['time'])}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

    for i, snap in enumerate(snapshots):
        plot_state(i, snap)

    print("Done.")

if __name__ == "__main__":
    solve()
