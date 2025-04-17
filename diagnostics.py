import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

def create_sample_data(n_samples=1000):
    """Create synthetic time series data for demonstration"""
    np.random.seed(42)
    X1 = np.arange(n_samples)  # Time feature
    X2 = np.sin(X1 / 50) + np.random.normal(0, 0.1, n_samples)
    X3 = np.random.normal(0, 1, n_samples)
    Y = 0.5 * X1 / 100 + 2 * X2 - 0.5 * X3 + np.random.normal(0, 0.5, n_samples)
    
    return pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'Y': Y})

def run_diagnostics(train_df, test_df, result_df, window_size):
    """Run diagnostic visualizations"""
    # ① Analyze target distribution drift
    plt.figure(figsize=(10, 6))
    print('Train Y mean:', train_df['Y'].mean(), 
          'Test Y mean:', test_df['Y'].mean())
    train_df['Y'].hist(alpha=0.5, label='train')
    test_df['Y'].hist(alpha=0.5, label='test')
    plt.legend()
    plt.title('Y distribution drift')
    plt.show()
    
    # ② Analyze residuals over time
    plt.figure(figsize=(12, 6))
    result_df['timestamp'] = test_df['X1'].iloc[window_size:].values  # Align timestamps
    result_df['residual'] = result_df['Y_pred'] - result_df['Y_true']
    plt.plot(result_df['timestamp'], result_df['residual'])
    plt.title('Residual vs time')
    plt.axhline(0, color='k')
    plt.show()

def main():
    # Load or generate data
    df = create_sample_data()  # Replace with your actual data loading
    
    # Split data (time-based)
    train_size = 0.7
    split_idx = int(len(df) * train_size)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Train model
    model = LinearRegression()
    features = [col for col in df.columns if col.startswith('X')]
    model.fit(train_df[features], train_df['Y'])
    
    # Make predictions
    window_size = 10
    y_test = test_df['Y'].values
    y_pred = model.predict(test_df[features])
    
    # Create results dataframe
    result_df = pd.DataFrame({
        'Y_true': y_test[window_size:],
        'Y_pred': y_pred[window_size:]
    })
    
    # Run diagnostics
    run_diagnostics(train_df, test_df, result_df, window_size)

if __name__ == "__main__":
    main()