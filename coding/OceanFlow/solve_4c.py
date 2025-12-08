import os
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import KFold

def solve():
    print("Starting Tau Sensitivity Analysis (Problem 4.c)...")
    
    data_dir = 'data'
    num_timesteps = 100
    
    # Target Location from 4.a: Grid(Row=264, Col=377)
    target_row = 264
    target_col = 377
    
    print(f"Target Location: Grid(Row={target_row}, Col={target_col})")
    
    # Load Data
    u_series = np.zeros(num_timesteps)
    for t in range(1, num_timesteps + 1):
        try:
            df_u = pd.read_csv(os.path.join(data_dir, f'{t}u.csv'), header=None, skiprows=target_row, nrows=1)
            u_series[t-1] = df_u.iloc[0, target_col]
        except Exception: pass
            
    # Normalize
    mean = np.mean(u_series)
    std = np.std(u_series)
    u_norm = (u_series - mean) / std
    
    # Grid Search Params
    l_vals = np.linspace(0.1, 5.0, 15)
    sigma_vals = np.linspace(0.1, 3.0, 15)
    time_indices = np.arange(num_timesteps)
    
    def kernel_prompt(x1, x2, sigma, l):
        sq_dist = np.subtract.outer(x1.flatten(), x2.flatten())**2
        return (sigma**2) * np.exp(- sq_dist / (l**2))

    # Tau values to test (Baseline was 0.001)
    tau_values = [0.0001, 0.001, 0.01, 0.1]
    
    print(f"Testing Tau values: {tau_values}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nResults for partial U-component optimization:")
    print("Tau     | Best L | Best Sigma | Max LogLikelihood")
    print("-" * 50)
    
    results = {}
    
    for tau in tau_values:
        best_ll = -float('inf')
        best_l = -1
        best_sigma = -1
        
        for sigma in sigma_vals:
            for l in l_vals:
                fold_scores = []
                for train_idx, test_idx in kf.split(time_indices):
                    X_train, y_train = time_indices[train_idx], u_norm[train_idx]
                    X_test, y_test = time_indices[test_idx], u_norm[test_idx]
                    
                    # Add Tau to diagonal
                    K_22 = kernel_prompt(X_train, X_train, sigma, l) + tau * np.eye(len(X_train))
                    K_12 = kernel_prompt(X_test, X_train, sigma, l)
                    K_11 = kernel_prompt(X_test, X_test, sigma, l) + tau * np.eye(len(X_test))
                    
                    # Estimate conditional distribution
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
        
        print(f"{tau:<7} | {best_l:.2f}   | {best_sigma:.2f}       | {best_ll:.4f}")
        results[tau] = (best_l, best_sigma, best_ll)

if __name__ == "__main__":
    solve()
