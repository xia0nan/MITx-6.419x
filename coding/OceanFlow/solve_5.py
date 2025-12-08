import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def solve():
    print("Starting GP Interpolation (Problem 5)...")
    
    data_dir = 'data'
    num_timesteps = 100
    
    # Target Location from 4.a: Grid(Row=264, Col=377)
    target_row = 264
    target_col = 377
    
    # Parameters from 4.a (U-Component)
    # l approx 2.14, sigma approx 0.46
    l_opt = 2.14
    sigma_opt = 0.46
    tau_opt = 0.001
    
    print(f"Target Location: Grid(Row={target_row}, Col={target_col})")
    print(f"Model Params: l={l_opt}, sigma={sigma_opt}, tau={tau_opt}")
    
    # Load Data (U component)
    u_series = np.zeros(num_timesteps)
    for t in range(1, num_timesteps + 1):
        try:
            df_u = pd.read_csv(os.path.join(data_dir, f'{t}u.csv'), header=None, skiprows=target_row, nrows=1)
            u_series[t-1] = df_u.iloc[0, target_col]
        except Exception: pass
            
    # Normalize
    global_mean = np.mean(u_series)
    global_std = np.std(u_series)
    u_norm = (u_series - global_mean) / global_std
    
    # Observed Data (Training)
    X_obs = np.arange(num_timesteps)
    y_obs = u_norm
    
    # Query Data (Interpolation)
    # Goal: "flows every day". 
    # Current spacing (index) = 3 days.
    # 1 index = 3 days.
    # Daily spacing = 1/3 index.
    X_new = np.arange(0, num_timesteps - 1 + 0.1, 1.0/3.0)
    
    print(f"Predicting at {len(X_new)} points (daily resolution)...")
    
    # Kernel Function
    def kernel(x1, x2, sigma, l):
        # x1, x2 are 1D arrays
        sq_dist = np.subtract.outer(x1, x2)**2
        return (sigma**2) * np.exp(- sq_dist / (l**2))
        
    print("Computing Covariance Matrices...")
    
    # K(Obs, Obs)
    K_obs_obs = kernel(X_obs, X_obs, sigma_opt, l_opt)
    K_obs_obs += tau_opt * np.eye(len(X_obs)) # Add noise
    
    # K(New, Obs)
    K_new_obs = kernel(X_new, X_obs, sigma_opt, l_opt)
    
    # K(New, New) - we only need diagonal for variance actually, to save memory
    # But for full covariance (if needed) we compute full.
    # len(X_new) ~ 300. 300x300 is tiny.
    K_new_new = kernel(X_new, X_new, sigma_opt, l_opt)
    # Add noise to prediction? 
    # Usually we want the distribution of the latent function f*, not y* (measurements).
    # If we want f*, we don't add tau to K_new_new.
    # If we want to predict future *measurements* with noise, we add tau.
    # "Estimate the flow" likely implies the underlying function.
    
    print("Inverting K_obs_obs...")
    K_inv = np.linalg.inv(K_obs_obs)
    
    print("Computing Conditional Distribution...")
    # Mean: K_new_obs * K_inv * y_obs
    mu_new_norm = K_new_obs @ K_inv @ y_obs
    
    # Covariance: K_new_new - K_new_obs * K_inv * K_obs_new
    cov_new_norm = K_new_new - K_new_obs @ K_inv @ K_new_obs.T
    
    # Variance (diagonal)
    var_new_norm = np.diag(cov_new_norm)
    std_new_norm = np.sqrt(var_new_norm)
    
    # Destandardize
    mu_physical = (mu_new_norm * global_std) + global_mean
    std_physical = std_new_norm * global_std
    
    # Confidence Intervals (3 sigma)
    lower_bound = mu_physical - 3 * std_physical
    upper_bound = mu_physical + 3 * std_physical
    
    print("Generating Plot...")
    
    plt.figure(figsize=(15, 6))
    
    # Plot observations
    # Convert index to Days
    t_obs_days = X_obs * 3
    t_new_days = X_new * 3
    
    plt.plot(t_obs_days, u_series, 'ko', markersize=4, label='Observed (Every 3 Days)')
    
    # Plot Mean
    plt.plot(t_new_days, mu_physical, 'b-', linewidth=2, label='Predicted Mean (Daily)')
    
    # Plot 3-sigma band
    plt.fill_between(t_new_days, lower_bound, upper_bound, color='blue', alpha=0.2, label='3-Sigma Confidence')
    
    plt.title(f"GP Interpolation of Flow U-Component\nParams: l={l_opt}, sigma={sigma_opt}, tau={tau_opt}")
    plt.xlabel("Time (Days)")
    plt.ylabel("Velocity U (km/h)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom in on a section to show detail?
    # Maybe 0 to 100 days
    plt.xlim(0, 300)
    
    fname = "gp_interpolation.png"
    plt.savefig(fname)
    print(f"Saved {fname}")

if __name__ == "__main__":
    solve()
