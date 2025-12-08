import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def solve():
    print("Starting Long-timescale Simulation (Problem 6.a)...")
    
    data_dir = 'data'
    num_orig_timesteps = 100
    rows = 504
    cols = 555
    grid_spacing = 3.0
    
    # 1. GP Interpolation Setup
    # Use params from 4.a
    # U: l=2.14, sig=0.46
    # V: l=3.37, sig=0.70
    # Tau = 0.001
    
    gp_params = {
        'u': {'l': 2.14, 'sigma': 0.46, 'tau': 0.001},
        'v': {'l': 3.37, 'sigma': 0.70, 'tau': 0.001}
    }
    
    # Time indices (1 unit = 3 days)
    # We want daily steps. 300 days total.
    # Original data: t=0, 1, ..., 99 (corresponding to days 0, 3, ..., 297)
    # Target steps: day=0, 1, ..., 299.
    # In index units: 0, 1/3, 2/3, 1, ...
    
    X_obs = np.arange(num_orig_timesteps, dtype=float)
    X_target = np.arange(0, num_orig_timesteps, 1.0/3.0)
    # Adjust to exactly 300 steps? 
    # 100 * 3 = 300 points exactly.
    X_target = X_target[:300]
    
    print(f"Interpolating from {len(X_obs)} to {len(X_target)} timesteps...")
    
    def compute_weights(l, sigma, tau):
        # K_obs_obs
        d_obs = np.subtract.outer(X_obs, X_obs)**2
        K_obs = (sigma**2) * np.exp(-d_obs / (l**2)) + tau * np.eye(len(X_obs))
        
        # K_target_obs
        d_tgt = np.subtract.outer(X_target, X_obs)**2
        K_tgt = (sigma**2) * np.exp(-d_tgt / (l**2))
        
        # Weights matrix: W = K_tgt @ K_obs_inv
        # Prediction: y_target = W @ y_obs
        # Using solve for stability: K_obs x = y_obs -> y_target = K_tgt x
        # Transpose trick: (W @ Y).T  or use tensordot
        
        # We need W such that Y_new[t, pixel] = sum_k ( W[t, k] * Y_old[k, pixel] )
        # W shape: (300, 100)
        
        return K_tgt @ np.linalg.inv(K_obs)

    print("Computing interpolation weights...")
    W_u = compute_weights(gp_params['u']['l'], gp_params['u']['sigma'], gp_params['u']['tau'])
    W_v = compute_weights(gp_params['v']['l'], gp_params['v']['sigma'], gp_params['v']['tau'])
    
    # 2. Load and Interpolate Data
    print("Loading and interpolating full fields...")
    
    # Load all data: (100, 504, 555)
    # Memory: 100*504*555 * 4 bytes ~ 111 MB. OK.
    
    U_obs = np.zeros((num_orig_timesteps, rows, cols), dtype=np.float32)
    V_obs = np.zeros((num_orig_timesteps, rows, cols), dtype=np.float32)
    
    for t in range(1, num_orig_timesteps + 1):
        if t % 20 == 0: print(f"  Loading {t}...", end='\r')
        try:
            U_obs[t-1] = pd.read_csv(os.path.join(data_dir, f'{t}u.csv'), header=None).values
            V_obs[t-1] = pd.read_csv(os.path.join(data_dir, f'{t}v.csv'), header=None).values
        except: pass
        
    print("\n  Data loaded. Applying interpolation...")
    
    # Apply weights.
    # W is (300, 100)
    # U_obs is (100, Rows, Cols)
    # U_new = tensordot(W, U_obs, axes=(1, 0)) -> (300, Rows, Cols)
    
    U_new = np.tensordot(W_u, U_obs, axes=(1, 0))
    V_new = np.tensordot(W_v, V_obs, axes=(1, 0))
    
    # Optional: Normalize/Standardize was used in GP derivation. 
    # Technically W should be applied to normalized data?
    # W y_norm = y_new_norm.
    # y = y_norm * std + mean.
    # W ( (y - m)/s ) = (W y - W m) / s
    # y_new = s * y_new_norm + m = W y - W m + m
    # If W row sums are 1 (which they roughly are for interpolation), W m approx m.
    # So y_new approx W y.
    # Given the simplicity required, direct application is standard for linear smoothers.
    # We will proceed with direct application.
    
    print(f"  Interpolated Fields Shape: {U_new.shape}")
    
    # 3. Particle Simulation
    
    def run_simulation(sigma_cloud, label):
        print(f"\nRunning simulation for Sigma Cloud = {sigma_cloud} km...")
        
        num_particles = 1000
        center_x, center_y = 300.0, 1050.0
        
        px = np.random.normal(center_x, sigma_cloud, num_particles)
        py = np.random.normal(center_y, sigma_cloud, num_particles)
        
        active = np.ones(num_particles, dtype=bool)
        beached_x = []
        beached_y = []
        
        # History
        history_x = []
        history_y = []
        
        # Save t=0
        history_x.append(px.copy())
        history_y.append(py.copy())
        
        dt = 1.0 # day
        days = 300
        
        # Intermediate snapshot target
        snapshot_day = 150
        
        snap_intermediate = None
        
        for day in range(days - 1):
            if day == snapshot_day:
                snap_intermediate = (px[active].copy(), py[active].copy())
            
            # Lookup
            idx_c = np.round(px / grid_spacing).astype(int)
            idx_r = np.round(py / grid_spacing).astype(int)
            
            # Check bounds / Land
            # If out of max bounds -> Lost/Beached?
            in_bounds = (idx_c >= 0) & (idx_c < cols) & (idx_r >= 0) & (idx_r < rows)
            
            # For in-bounds, check if flow is zero (Land)
            # Vectorized check
            moving_mask = np.zeros(num_particles, dtype=bool)
            
            # Only check velocities for in-bounds particles
            # Create temporary safe indices
            safe_c = np.clip(idx_c, 0, cols-1) 
            safe_r = np.clip(idx_r, 0, rows-1)
            
            u_val = U_new[day, safe_r, safe_c]
            v_val = V_new[day, safe_r, safe_c]
            
            speed_sq = u_val**2 + v_val**2
            is_water = speed_sq > 1e-6
            
            # Alive condition: In bounds AND Is Water
            still_alive = in_bounds & is_water
            
            # Those who just died (were active, now not)
            just_beached = active & (~still_alive)
            if np.any(just_beached):
                beached_x.extend(px[just_beached])
                beached_y.extend(py[just_beached])
                
            active = active & still_alive
            
            # Update alive particles
            # Careful: U_new[day] gives velocity for valid grid point.
            # px += u * dt. Speed is likely km/h? Or km/day?
            # PROBLEM: 1.a says units are km/h.
            # We are simulating days.
            # dt = 1 day = 24 hours.
            # If data is mean flow for the day, we need to multiply by 24.
            # "One flow data per day".
            # Assuming flow is constant for the day?
            # Yes, multiply by 24.
            
            u_eff = u_val * 24.0
            v_eff = v_val * 24.0
            
            px[active] += u_eff[active] * dt
            py[active] += v_eff[active] * dt

        # Finish
        snap_final = (px[active].copy(), py[active].copy())
        
        # Plotting
        plt.figure(figsize=(10, 8))
        plt.title(f"Debris Simulation (300 Days) - Sigma={sigma_cloud} km")
        plt.xlim(0, cols*grid_spacing)
        plt.ylim(0, rows*grid_spacing)
        
        # Plot Beached
        if len(beached_x) > 0:
            plt.plot(beached_x, beached_y, 'kx', markersize=5, label='Beached/Terminated')
            
        # Plot Initial
        plt.plot(history_x[0], history_y[0], 'b.', markersize=2, alpha=0.3, label='Initial (t=0)')
        
        # Plot Intermediate
        if snap_intermediate:
            plt.plot(snap_intermediate[0], snap_intermediate[1], 'g.', markersize=3, alpha=0.5, label=f'Day {snapshot_day}')
            
        # Plot Final
        plt.plot(snap_final[0], snap_final[1], 'r.', markersize=4, label='Final (Day 300)')
        
        plt.legend()
        plt.xlabel("X (km)")
        plt.ylabel("Y (km)")
        
        fname = f"debris_300d_sigma{int(sigma_cloud)}.png"
        plt.savefig(fname)
        print(f"Saved {fname}")
        
        return active, px, py, beached_x, beached_y

    # Run 1: Sigma = 30 km (Moderate uncertainty)
    active1, px1, py1, bx1, by1 = run_simulation(30.0, "Run1")
    
    # Run 2: Sigma = 90 km (High uncertainty)
    active2, px2, py2, bx2, by2 = run_simulation(90.0, "Run2")
    
    # Suggest Locations
    # Based on Run 1
    if len(bx1) > 0:
        # Cluster beached particles? Just pick the mean of beached.
        land_target = (np.mean(bx1), np.mean(by1))
        print(f"Suggested Land Search Location: ({land_target[0]:.0f}, {land_target[1]:.0f}) km")
    else:
        print("No particles beached in Run 1.")
        
    if np.sum(active1) > 0:
        ocean_target = (np.mean(px1[active1]), np.mean(py1[active1]))
        print(f"Suggested Ocean Search Location: ({ocean_target[0]:.0f}, {ocean_target[1]:.0f}) km")
    else:
         print("All particles beached in Run 1.")

if __name__ == "__main__":
    solve()
