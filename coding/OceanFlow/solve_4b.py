import os
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import KFold
import random

def solve():
    print("Starting Multi-point GP Analysis (Problem 4.b)...")
    
    data_dir = 'data'
    num_timesteps = 100
    rows = 504
    cols = 555
    
    # Check data
    if not os.path.exists(data_dir): return

    # 1. Select 3 random valid points (approximate check by loading one file)
    print("Selecting random valid points...")
    sample_u = pd.read_csv(os.path.join(data_dir, '1u.csv'), header=None).values
    # Valid if non-zero
    valid_indices = np.argwhere(np.abs(sample_u) > 1e-6)
    
    # Pick 4 points (target 3, +1 extra just in case)
    num_points = 4
    random_indices = valid_indices[np.random.choice(len(valid_indices), num_points, replace=False)]
    
    points_to_analyze = [tuple(idx) for idx in random_indices]
    print(f"Selected points: {points_to_analyze}")
    
    # Parameters for Grid Search
    l_vals = np.linspace(0.1, 5.0, 15) # Coarser grid for speed
    sigma_vals = np.linspace(0.1, 3.0, 15)
    time_indices = np.arange(num_timesteps)

    def kernel_prompt(x1, x2, sigma, l):
        sq_dist = np.subtract.outer(x1.flatten(), x2.flatten())**2
        return (sigma**2) * np.exp(- sq_dist / (l**2))
        
    def normalize(data):
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-9: std = 1.0
        return (data - mean) / std, mean, std

    def optimize_gp(data_norm):
        kf = KFold(n_splits=5, shuffle=True, random_state=42) # 5-fold for speed
        best_ll = -float('inf')
        best_l = -1
        best_sigma = -1
        
        for sigma in sigma_vals:
            for l in l_vals:
                fold_scores = []
                for train_idx, test_idx in kf.split(time_indices):
                    X_train, y_train = time_indices[train_idx], data_norm[train_idx]
                    X_test, y_test = time_indices[test_idx], data_norm[test_idx]
                    
                    K_22 = kernel_prompt(X_train, X_train, sigma, l) + 0.001 * np.eye(len(X_train))
                    K_12 = kernel_prompt(X_test, X_train, sigma, l)
                    K_11 = kernel_prompt(X_test, X_test, sigma, l) + 0.001 * np.eye(len(X_test))
                    
                    try:
                        K_22_inv = np.linalg.inv(K_22)
                        mu_cond = K_12 @ K_22_inv @ y_train
                        cov_cond = K_11 - K_12 @ K_22_inv @ K_12.T
                        
                        sign, logdet = np.linalg.slogdet(cov_cond)
                        if sign <= 0: raise ValueError
                        
                        diff = y_test - mu_cond
                        quad = diff.T @ np.linalg.inv(cov_cond) @ diff
                        ll = -0.5 * logdet - 0.5 * quad - (len(y_test)/2)*np.log(2*np.pi)
                        fold_scores.append(ll)
                    except:
                        fold_scores.append(-float('inf'))
                
                avg = np.mean(fold_scores)
                if avg > best_ll:
                    best_ll = avg
                    best_l = l
                    best_sigma = sigma
        return best_l, best_sigma, best_ll

    # Data Loading Helper
    # Load all data first? No, 200 files read for each point is better than loading 2GB into RAM multiple times?
    # actually loading 2GB into RAM once is better if we have RAM.
    # But environment might be constrained.
    # Let's iterate files and extract points.
    
    print("Extracting time series for all points...")
    # shape: (NumPoints, Time)
    u_series = np.zeros((num_points, num_timesteps))
    v_series = np.zeros((num_points, num_timesteps))
    
    for t in range(1, num_timesteps+1):
        if t % 20 == 0: print(f"Reading files {t}/{num_timesteps}...", end='\r')
        try:
            u_df = pd.read_csv(os.path.join(data_dir, f'{t}u.csv'), header=None)
            v_df = pd.read_csv(os.path.join(data_dir, f'{t}v.csv'), header=None)
            
            for i, (r, c) in enumerate(points_to_analyze):
                u_series[i, t-1] = u_df.iloc[r, c]
                v_series[i, t-1] = v_df.iloc[r, c]
        except: pass
        
    print("\nData extracted. Running optimization...")
    
    results = []
    
    for i, pt in enumerate(points_to_analyze):
        print(f"\n--- Point {pt} ---")
        
        # Optimize U
        u_norm, um, us = normalize(u_series[i])
        best_l_u, best_sig_u, ll_u = optimize_gp(u_norm)
        print(f"  U-Comp: l={best_l_u:.2f}, sigma={best_sig_u:.2f} (mean={um:.2f}, std={us:.2f})")
        
        # Optimize V
        v_norm, vm, vs = normalize(v_series[i])
        best_l_v, best_sig_v, ll_v = optimize_gp(v_norm)
        print(f"  V-Comp: l={best_l_v:.2f}, sigma={best_sig_v:.2f} (mean={vm:.2f}, std={vs:.2f})")
        
        results.append({
            'point': pt,
            'u': (best_l_u, best_sig_u),
            'v': (best_l_v, best_sig_v)
        })

    print("\nSummary of Results:")
    print("Point | U_L | U_Sig | V_L | V_Sig")
    for r in results:
        print(f"{r['point']} | {r['u'][0]:.2f} | {r['u'][1]:.2f} | {r['v'][0]:.2f} | {r['v'][1]:.2f}")

if __name__ == "__main__":
    solve()
