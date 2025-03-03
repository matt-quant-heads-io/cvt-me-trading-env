import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

def generate_synthetic_data(start_date='2023-01-01', days=30, volatility=0.015):
    """
    Generate synthetic OHLCV data in 1-minute increments.
    
    Args:
        start_date: Starting date for the time series
        days: Number of days to generate data for
        volatility: Volatility parameter for price movements
    
    Returns:
        DataFrame with datetime, open, high, low, close, volume columns
    """
    # Calculate number of minutes in the time period (excluding non-trading hours)
    trading_minutes_per_day = 6 * 60 + 30  # 6.5 hours (simplified trading day)
    num_points = trading_minutes_per_day * days
    
    # Generate datetime index
    start = datetime.strptime(start_date, '%Y-%m-%d')
    dates = []
    current_date = start
    
    for _ in range(days):
        # Only include trading hours (9:30 AM to 4:00 PM)
        for minute in range(9*60+30, 16*60):
            hour = minute // 60
            min_val = minute % 60
            dates.append(datetime(current_date.year, current_date.month, current_date.day, 
                                 hour, min_val))
        current_date += timedelta(days=1)
        # Skip weekends
        if current_date.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
            current_date += timedelta(days=7 - current_date.weekday())
    
    # Generate price data using geometric Brownian motion
    price = 100.0  # Starting price
    returns = np.random.normal(0, volatility, len(dates))
    prices = [price]
    
    for ret in returns:
        price *= (1 + ret)
        prices.append(price)
    
    prices = prices[:-1]  # Remove the extra price
    
    # Generate OHLCV data
    data = []
    
    for i, date in enumerate(dates):
        price = prices[i]
        high_low_range = price * volatility * random.uniform(0.5, 1.5)
        high = price + high_low_range / 2
        low = price - high_low_range / 2
        
        # Generate realistic open and close prices between high and low
        if i > 0:
            open_price = prices[i-1]
        else:
            open_price = price - price * volatility * random.uniform(-0.5, 0.5)
        
        open_price = max(min(open_price, high), low)
        close = price
        
        # Generate volume with some randomness
        volume = int(np.random.gamma(shape=2.0, scale=50000) * (1 + abs(returns[i]) * 10))
        
        data.append([date, open_price, high, low, close, volume])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    
    # Generate SP500 data that somewhat correlates with our stock
    sp500_prices = []
    sp_price = 4000.0
    sp_volatility = 0.01
    for i, ret in enumerate(returns):
        correlation = 0.7  # Correlation parameter
        sp_return = correlation * ret + (1 - correlation) * np.random.normal(0, sp_volatility)
        sp_price *= (1 + sp_return)
        sp500_prices.append(sp_price)
    
    df['sp500'] = sp500_prices
    
    return df

# Generate the data and save to CSV
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    data = generate_synthetic_data(days=60)  # 60 days of data
    data.to_csv('synthetic_market_data.csv', index=False)
    print(f"Generated {len(data)} minutes of synthetic market data")
    print(f"Sample data:\n{data.head()}")