import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import KFold

def solve():
    print("Starting GP Model Estimation (Problem 4.a)...")
    
    data_dir = 'data'
    num_timesteps = 100
    rows = 504
    cols = 555
    
    # 1. Load Data for a specific point
    # Let's pick a valid point. I'll pick a fixed one that is likely water.
    # From problem 2, (377, 264) was water. Let's use Grid Coordinates (Row=264, Col=377).
    target_row = 264
    target_col = 377
    
    print(f"Target Location: Grid(Row={target_row}, Col={target_col})")
    
    u_series = np.zeros(num_timesteps)
    v_series = np.zeros(num_timesteps)
    
    print("Loading time series...")
    for t in range(1, num_timesteps + 1):
        u_path = os.path.join(data_dir, f'{t}u.csv')
        v_path = os.path.join(data_dir, f'{t}v.csv')
        try:
            # We only need one pixel, so reading the whole file is inefficient but robust
            # Given file size (2MB), it's okay. 
            # Optimization: use specific line reading if needed, but pd.read_csv is fast enough for 100x.
            # Actually, reading 200 files completely just for one pixel is slow.
            # Can we read just the line?
            # rows are lines. row 264.
            # But CSV parsing is tricky.
            # Let's just read full frames, it took ~30s in previous steps. Acceptable.
            
            # To speed up: Use 'skiprows'
            # But header=None.
            # skiprows = target_row. nrows = 1.
            
            df_u = pd.read_csv(u_path, header=None, skiprows=target_row, nrows=1)
            df_v = pd.read_csv(v_path, header=None, skiprows=target_row, nrows=1)
            
            u_series[t-1] = df_u.iloc[0, target_col]
            v_series[t-1] = df_v.iloc[0, target_col]
            
            if t % 10 == 0: print(f"Loaded {t}/{num_timesteps}...", end='\r')
            
        except Exception as e:
            print(f"Error reading t={t}: {e}")
            
    print("\nData loaded.")
    
    # Normalize Data
    def normalize(data):
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-9: std = 1.0
        return (data - mean) / std, mean, std

    u_norm, u_mean, u_std = normalize(u_series)
    v_norm, v_mean, v_std = normalize(v_series)
    
    print(f"U stats: mean={u_mean:.4f}, std={u_std:.4f}")
    print(f"V stats: mean={v_mean:.4f}, std={v_std:.4f}")
    
    # GP Kernel
    # Squared Exponential: K(xi, xj) = sigma^2 * exp( - (xi - xj)^2 / l^2 )
    def kernel(x1, x2, sigma, l):
        # x1: (N1, 1), x2: (N2, 1)
        # diff: (N1, N2)
        sq_dist = np.subtract.outer(x1.flatten(), x2.flatten())**2
        return (sigma**2) * np.exp(-0.5 * sq_dist / (l**2)) 
        # Note: Problem says exp( - ||z_i - z_j||^2 / l^2 ).
        # Usually it is 2*l^2. I will follow the FORMULA given in the prompt exactly:
        # K = sigma^2 * exp( - ||diff||^2 / l^2 )
        # Removing the 0.5 factor to match prompt exactly.
        
    def kernel_prompt(x1, x2, sigma, l):
        sq_dist = np.subtract.outer(x1.flatten(), x2.flatten())**2
        return (sigma**2) * np.exp(- sq_dist / (l**2))

    # Grid Search
    # l_range: 0.1 to 5.0 indices
    # sigma_range: 0.1 to 3.0
    l_vals = np.linspace(0.1, 5.0, 25)
    sigma_vals = np.linspace(0.1, 3.0, 25)
    
    time_indices = np.arange(num_timesteps)
    
    def run_optimization(data_norm, label):
        print(f"\nOptimizing for {label} component...")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        best_score = -float('inf')
        best_l = -1
        best_sigma = -1
        
        # Store scores for plotting (Sigma x L)
        score_grid = np.zeros((len(sigma_vals), len(l_vals)))
        
        total_iters = len(sigma_vals) * len(l_vals)
        iter_count = 0
        
        for i, sigma in enumerate(sigma_vals):
            for j, l in enumerate(l_vals):
                fold_scores = []
                
                for train_idx, test_idx in kf.split(time_indices):
                    # Training data
                    X_train = time_indices[train_idx]
                    y_train = data_norm[train_idx]
                    
                    # Test data
                    X_test = time_indices[test_idx]
                    y_test = data_norm[test_idx]
                    
                    # Construct Covariance Matrices
                    # K_train_train (Sigma_22)
                    K_tt = kernel_prompt(X_train, X_train, sigma, l)
                    
                    # Add noise to diagonal for stability
                    K_tt += 0.001 * np.eye(len(X_train))
                    
                    # K_test_scan (Sigma_12) - Wait, prompt notation:
                    # Sigma_11: test-test
                    # Sigma_12: test-train
                    # Sigma_21: train-test
                    # Sigma_22: train-train
                    
                    K_22 = K_tt
                    K_12 = kernel_prompt(X_test, X_train, sigma, l)
                    # K_21 = K_12.T
                    K_11 = kernel_prompt(X_test, X_test, sigma, l)
                    # Add noise to K_11 as well for covariance validity
                    K_11 += 0.001 * np.eye(len(X_test))
                    
                    # Conditional Mean and Covariance
                    # mu_1|2 = mu1 + Sigma_12 * inv(Sigma_22) * (y_train - mu2)
                    # Data is normalized, so prior means mu1, mu2 are 0.
                    
                    # Invert K_22 carefully
                    # Using linalg.solve is better: K_22 * alpha = y_train  -> alpha = K_22^-1 y_train
                    # But we need K_22^-1 for covariance update too?
                    # Update rule: Sigma_1|2 = Sigma_11 - Sigma_12 * K_22^-1 * Sigma_21
                    
                    try:
                        # Solve for weights
                        # L = cholesky(K_22)
                        # alpha = solve(L.T, solve(L, y_train))
                        # For simplicity, using pinv or solve
                        
                        K_22_inv = np.linalg.inv(K_22) # Direct inverse for formula application
                        
                        mu_cond = K_12 @ K_22_inv @ y_train
                        cov_cond = K_11 - K_12 @ K_22_inv @ K_12.T
                        
                        # Compute Log Likelihood of y_test under N(mu_cond, cov_cond)
                        # LL = -0.5 * log|cov| - 0.5 * (y-mu)^T cov^-1 (y-mu) - k/2 log(2pi)
                        
                        # Add epsilon to diagonal?
                        # cov_cond should be positive definite
                        
                        sign, logdet = np.linalg.slogdet(cov_cond)
                        if sign <= 0:
                            # Numerical issue
                            raise ValueError("Invalid covariance")
                            
                        diff = y_test - mu_cond
                        quad_form = diff.T @ np.linalg.inv(cov_cond) @ diff
                        
                        ll = -0.5 * logdet - 0.5 * quad_form - (len(y_test)/2)*np.log(2*np.pi)
                        fold_scores.append(ll)
                        
                    except np.linalg.LinAlgError:
                        fold_scores.append(-float('inf'))
                    except ValueError:
                        fold_scores.append(-float('inf'))
                
                avg_score = np.mean(fold_scores)
                score_grid[i, j] = avg_score
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_l = l
                    best_sigma = sigma
                    
                iter_count += 1
                if iter_count % 50 == 0:
                    print(f"  Processed {iter_count}/{total_iters} grid points...", end='\r')
        
        print(f"\n  Best parameters for {label}: L={best_l:.2f}, Sigma={best_sigma:.2f}, LL={best_score:.4f}")
        return best_l, best_sigma, score_grid

    # Run for U
    best_l_u, best_sigma_u, scores_u = run_optimization(u_norm, "U")
    
    # Run for V
    best_l_v, best_sigma_v, scores_v = run_optimization(v_norm, "V")
    
    # Visualize
    def plot_heatmap(scores, title, best_l, best_sigma, fname):
        plt.figure(figsize=(10, 8))
        # Extent: L (x), Sigma (y)
        plt.imshow(scores, origin='lower', aspect='auto',
                   extent=[l_vals.min(), l_vals.max(), sigma_vals.min(), sigma_vals.max()])
        plt.colorbar(label='Log Likelihood')
        plt.xlabel('Length Scale (l)')
        plt.ylabel('Sigma')
        plt.title(title)
        plt.plot(best_l, best_sigma, 'rx', markersize=12, markeredgewidth=3)
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
        
    plot_heatmap(scores_u, f"Log-Likelihood Surface (U) - Best: l={best_l_u:.2f}, sig={best_sigma_u:.2f}", 
                 best_l_u, best_sigma_u, "gp_u_optimization.png")
    
    plot_heatmap(scores_v, f"Log-Likelihood Surface (V) - Best: l={best_l_v:.2f}, sig={best_sigma_v:.2f}",
                 best_l_v, best_sigma_v, "gp_v_optimization.png")
                 
    # Save params
    with open("gp_params.txt", "w") as f:
        f.write(f"U_params: l={best_l_u}, sigma={best_sigma_u}, mean={u_mean}, std={u_std}\n")
        f.write(f"V_params: l={best_l_v}, sigma={best_sigma_v}, mean={v_mean}, std={v_std}\n")

if __name__ == "__main__":
    solve()
