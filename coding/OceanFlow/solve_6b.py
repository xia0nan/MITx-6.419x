import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys

def solve():
    print("Starting Monitoring Station Placement (Problem 6.b)...")
    
    data_dir = 'data'
    num_orig_timesteps = 100
    rows = 504
    cols = 555
    grid_spacing = 3.0
    
    # --- 1. GP Interpolation (Re-used from 6.a) ---
    gp_params = {
        'u': {'l': 2.14, 'sigma': 0.46, 'tau': 0.001},
        'v': {'l': 3.37, 'sigma': 0.70, 'tau': 0.001}
    }
    
    X_obs = np.arange(num_orig_timesteps, dtype=float)
    X_target = np.arange(0, num_orig_timesteps, 1.0/3.0)[:300]
    
    print("Computing interpolation weights...")
    def compute_weights(l, sigma, tau):
        d_obs = np.subtract.outer(X_obs, X_obs)**2
        K_obs = (sigma**2) * np.exp(-d_obs / (l**2)) + tau * np.eye(len(X_obs))
        d_tgt = np.subtract.outer(X_target, X_obs)**2
        K_tgt = (sigma**2) * np.exp(-d_tgt / (l**2))
        return K_tgt @ np.linalg.inv(K_obs)

    W_u = compute_weights(gp_params['u']['l'], gp_params['u']['sigma'], gp_params['u']['tau'])
    W_v = compute_weights(gp_params['v']['l'], gp_params['v']['sigma'], gp_params['v']['tau'])
    
    print("Loading and interpolating full fields...")
    U_obs = np.zeros((num_orig_timesteps, rows, cols), dtype=np.float32)
    V_obs = np.zeros((num_orig_timesteps, rows, cols), dtype=np.float32)
    
    for t in range(1, num_orig_timesteps + 1):
        if t % 50 == 0: print(f"  Loading {t}...", end='\r')
        try:
            U_obs[t-1] = pd.read_csv(os.path.join(data_dir, f'{t}u.csv'), header=None).values
            V_obs[t-1] = pd.read_csv(os.path.join(data_dir, f'{t}v.csv'), header=None).values
        except: pass
        
    print("\n  Applying interpolation...")
    U_new = np.tensordot(W_u, U_obs, axes=(1, 0))
    V_new = np.tensordot(W_v, V_obs, axes=(1, 0))
    del U_obs, V_obs # Free memory
    
    # --- 2. Simulation Setup ---
    print("Initializing uniform particles...")
    num_particles = 3000
    
    # Uniform random distribution
    px = np.random.uniform(0, cols * grid_spacing, num_particles)
    py = np.random.uniform(0, rows * grid_spacing, num_particles)
    
    # Filter Land Start
    # Check T=0 velocity
    start_c = np.clip(np.round(px / grid_spacing).astype(int), 0, cols-1)
    start_r = np.clip(np.round(py / grid_spacing).astype(int), 0, rows-1)
    
    u0 = U_new[0, start_r, start_c]
    v0 = V_new[0, start_r, start_c]
    speed0 = u0**2 + v0**2
    
    valid_start = speed0 > 1e-6
    px = px[valid_start]
    py = py[valid_start]
    num_particles = len(px)
    print(f"  {num_particles} particles started on water.")
    
    active = np.ones(num_particles, dtype=bool)
    beached_x = []
    beached_y = []
    
    history_x = [px.copy()]
    history_y = [py.copy()]
    
    dt = 1.0 # day
    days = 300
    snapshot_day = 150
    intermediate_snap = None
    
    print("Running 300-day simulation...")
    for day in range(days - 1):
        if day == snapshot_day:
            intermediate_snap = (px[active].copy(), py[active].copy())
            
        # Current indices
        idx_c = np.round(px / grid_spacing).astype(int)
        idx_r = np.round(py / grid_spacing).astype(int)
        
        # Check bounds
        in_bounds = (idx_c >= 0) & (idx_c < cols) & (idx_r >= 0) & (idx_r < rows)
        
        # Check Water
        safe_c = np.clip(idx_c, 0, cols-1) 
        safe_r = np.clip(idx_r, 0, rows-1)
        
        curr_u = U_new[day, safe_r, safe_c]
        curr_v = V_new[day, safe_r, safe_c]
        
        is_water = (curr_u**2 + curr_v**2) > 1e-6
        
        still_alive = in_bounds & is_water
        
        # Identify newly beached (In bounds AND became Land)
        # Note: If they go out of bounds, they are lost, not beached on coast technically.
        # But for this problem, "end up on the coast".
        # We assume 0-velocity zones inside bounds are islands/coast.
        just_beached = active & in_bounds & (~is_water)
        
        if np.any(just_beached):
            beached_x.extend(px[just_beached])
            beached_y.extend(py[just_beached])
            
        active = active & still_alive
        
        # Advect
        u_eff = curr_u * 24.0
        v_eff = curr_v * 24.0
        
        px[active] += u_eff[active] * dt
        py[active] += v_eff[active] * dt
        
    final_snap = (px[active].copy(), py[active].copy())
    
    # --- 3. Identify Stations ---
    print(f"Simulation done. Total beached particles: {len(beached_x)}")
    
    if len(beached_x) < 3:
        print("Not enough beached particles to cluster. Picking random points from history.")
        stations = [(500, 500), (600, 600), (700, 700)]
    else:
        # Use KMeans to find 3 centers of density
        beached_pts = np.column_stack((beached_x, beached_y))
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(beached_pts)
        stations = kmeans.cluster_centers_
        
    print("Proposed Stations:")
    for i, s in enumerate(stations):
        print(f"  Station {i+1}: ({s[0]:.0f}, {s[1]:.0f})")
        
    # --- 4. Plotting ---
    plt.figure(figsize=(10, 8))
    plt.title("Monitoring Station Proposal (Prob 6.b)")
    plt.xlim(0, cols*grid_spacing)
    plt.ylim(0, rows*grid_spacing)
    
    # 1. Beached particles (Density map ideally, but scatter is fine)
    if len(beached_x) > 0:
        plt.plot(beached_x, beached_y, 'k.', markersize=4, alpha=0.3, label='Beached Particles')
        
    # 2. Initial
    plt.plot(history_x[0], history_y[0], 'b.', markersize=1, alpha=0.1, label='Initial Distribution')
    
    # 3. Intermediate
    if intermediate_snap:
        plt.plot(intermediate_snap[0], intermediate_snap[1], 'g.', markersize=1, alpha=0.1, label='Day 150')
        
    # 4. Final
    # plt.plot(final_snap[0], final_snap[1], 'r.', markersize=2, alpha=0.2, label='Final Day 300')
    
    # 5. Stations
    for i, s in enumerate(stations):
        plt.plot(s[0], s[1], 'r*', markersize=15, markeredgecolor='black', label=f'Station {i+1}' if i==0 else "")
        plt.text(s[0]+20, s[1]+20, f"S{i+1}", fontsize=12, color='red', weight='bold')
        
    plt.legend(loc='upper right')
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    
    plt.savefig("stations_proposal.png")
    print("Saved stations_proposal.png")

if __name__ == "__main__":
    solve()
