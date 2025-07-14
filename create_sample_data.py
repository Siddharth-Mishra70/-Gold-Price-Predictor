import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

def create_sample_gold_data():
    """Create sample gold price data for testing"""
    
    # Generate dates for the last 2 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic gold price data
    np.random.seed(42)  # For reproducible results
    
    # Base price around $2000
    base_price = 2000
    
    # Create price series with trend and volatility
    n_days = len(dates)
    
    # Add trend component (slight upward trend)
    trend = np.linspace(0, 200, n_days)
    
    # Add seasonal component
    seasonal = 50 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    
    # Add random walk component
    random_walk = np.cumsum(np.random.normal(0, 5, n_days))
    
    # Add noise
    noise = np.random.normal(0, 10, n_days)
    
    # Combine all components
    prices = base_price + trend + seasonal + random_walk + noise
    
    # Ensure prices are positive
    prices = np.maximum(prices, 1500)
    
    # Create the dataframe
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices.round(2)
    })
    
    # Add some missing values randomly (about 5%)
    missing_indices = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
    df.loc[missing_indices, 'Price'] = np.nan
    
    # Save to CSV
    df.to_csv('gold_data.csv', index=False)
    
    print(" Sample gold data created successfully!")
    print(f" Dataset shape: {df.shape}")
    print(f" Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f" Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
    print(f" Average price: ${df['Price'].mean():.2f}")
    print("\n First 5 rows:")
    print(df.head())
    print("\n Last 5 rows:")
    print(df.tail())
    
    return df

if __name__ == "__main__":
    create_sample_gold_data() 