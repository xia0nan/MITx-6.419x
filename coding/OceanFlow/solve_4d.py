import os
import numpy as np
import pandas as pd
import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

def solve():
    print("Starting Library Comparison Analysis (Problem 4.d)...")
    
    data_dir = 'data'
    num_timesteps = 100
    
    # Target Location from 4.a: Grid(Row=264, Col=377)
    target_row = 264
    target_col = 377
    
    print(f"Target Location: Grid(Row={target_row}, Col={target_col})")
    
    # Load Data (U component)
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
    
    # Prepare X and y
    X = np.arange(num_timesteps).reshape(-1, 1)
    y = u_norm
    
    # Define Kernel
    # Sklearn RBF: exp(-d^2 / (2 * l^2))
    # Prompt Formula: exp(-d^2 / l^2)
    # The 'l' in sklearn is actually 'length_scale'.
    # Sklearn definition: k(x_i, x_j) = exp(-1 / (2 * l^2) * d(x_i, x_j)^2)
    # Prompt definition: exp(-1 / l_prompt^2 * d^2)
    # Equating exponents: 1/(2 * l_sklearn^2) = 1/l_prompt^2
    # l_prompt^2 = 2 * l_sklearn^2
    # l_prompt = sqrt(2) * l_sklearn
    
    # We want to find l_prompt.
    # We will let sklearn optimize its own l, then convert.
    
    # Kernel: Constant * RBF + WhiteKernel (noise)
    # We use WhiteKernel to learn the noise level (Tau) or fix it.
    # To match 4.a, we should arguably fix noise to 0.001?
    # Or let it optimize everything? Prompt says "Use one library... compare results".
    # Usually libraries optimize everything. Let's let it optimize noise too, 
    # or fix it to be comparable.
    # Let's try fixing noise to 0.001 (approx 1e-3) for fair comparison if possible,
    # but WhiteKernel optimizes variance.
    # Let's start with full optimization (most standard library usage).
    
    print("\nFitting GaussianProcessRegressor (optimizing Marginal Log-Likelihood)...")
    
    # Initial bounds
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.01, 10.0)) * \
             RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0)) + \
             WhiteKernel(noise_level=0.001, noise_level_bounds=(1e-5, 1.0))
             
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gp.fit(X, y)
    
    print("\nModel Fitted.")
    print(f"Log-Likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta):.4f}")
    
    # Extract Parameters
    params = gp.kernel_.get_params()
    
    # Access components directly from kernel structure
    # kernel is Sum(Product(Constant, RBF), White)
    k_prod = params['k1']
    k_white = params['k2']
    k_const = k_prod.k1
    k_rbf = k_prod.k2
    
    sigma_sq = k_const.constant_value
    l_sklearn = k_rbf.length_scale
    noise_var = k_white.noise_level
    
    sigma = np.sqrt(sigma_sq)
    
    # Convert Length Scale
    # Sklearn: exp( - d^2 / (2 * l_sklearn^2) )
    # Prompt:  exp( - d^2 / l_prompt^2 )
    # => l_prompt = sqrt(2) * l_sklearn
    l_prompt = l_sklearn * np.sqrt(2)
    
    print("-" * 40)
    print("Sklearn Optimized Parameters:")
    print(f"  Length Scale (sklearn): {l_sklearn:.4f}")
    print(f"  Length Scale (prompt eq): {l_prompt:.4f}")
    print(f"  Signal Theta/Sigma: {sigma:.4f}")
    print(f"  Noise Variance: {noise_var:.6f}")
    print("-" * 40)
    
    print("Comparison with Problem 4.a (U-comp):")
    print("  4.a Best L: ~2.14 / Best Sigma: ~0.46 / Fixed Noise: 0.001")
    
    # Check if they match
    print("\nDiscussion:")
    print("Differences likely due to:")
    print("1. Objective Function: Sklearn optimizes Marginal Log Likelihood on Train set.")
    print("   Problem 4.a used Cross-Validation on Test sets (Conditional Log Likelihood).")
    print("2. Noise: Sklearn optimized noise variance, 4.a fixed it (tau=0.001).")
    print("3. Optimization: Gradient Descent vs Grid Search.")
    
    # Try fixing noise to 0.001 to see if it gets closer
    print("\nRetrying with fixed Noise = 0.001...")
    kernel_fixed = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(noise_level=0.001, noise_level_bounds="fixed")
    gp_fixed = GaussianProcessRegressor(kernel=kernel_fixed, n_restarts_optimizer=10, random_state=42)
    gp_fixed.fit(X, y)
    
    k_prod_f = gp_fixed.kernel_.k1
    sigma_f = np.sqrt(k_prod_f.k1.constant_value)
    l_sklearn_f = k_prod_f.k2.length_scale
    l_prompt_f = l_sklearn_f * np.sqrt(2)
    
    print(f"  Length Scale (prompt eq) [Fixed Noise]: {l_prompt_f:.4f}")
    print(f"  Signal Sigma [Fixed Noise]: {sigma_f:.4f}")

if __name__ == "__main__":
    solve()
