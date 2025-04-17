import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_data(start_date='2015-01-01', end_date='2022-12-31', dataset="FTSC"):
    """Download financial data and calculate log returns and realized variance"""
    if dataset == "FTSC":
        print("Loading FTSC 5-minute interval data...")
        df = pd.read_csv("UK_Index_2000_2025.csv")
        df['Date-Time'] = pd.to_datetime(df['Date-Time'])
        df = df.dropna(axis=1, how='all')
        
        # Adjust timestamps to GMT
        df['Date-Time'] = df.apply(
            lambda row: row['Date-Time'] + pd.Timedelta(hours=row['GMT Offset']),
            axis=1
        )
        
        # Drop unnecessary columns
        df = df.drop(columns=["#RIC", "Domain", "Type", "No. Trades", "GMT Offset"])
        
        # Sort by date-time
        df = df.sort_values(by='Date-Time')
        df = df.set_index('Date-Time')
        
        # Calculate log returns
        df['log_return'] = np.log(df['Last'] / df['Last'].shift(1))
        
        # Calculate realized variance - adjusted for 5-minute data
        # For 5-minute data, there are roughly 78 intervals per day (6.5 hours × 12 intervals/hour)
        # So annualization factor is 252 days × 78 intervals = 19656
        df['realized_var'] = df['log_return'].rolling(window=78).var() * 19656  # One trading day window (78 5-min intervals)

        # Drop NA values
        df = df.dropna()
        
        return df
    
    else:
        print("Downloading S&P 500 data...")
        sp500 = yf.download('^GSPC', start=start_date, end=end_date)
    
        # Calculate log returns
        sp500['log_return'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
        sp500['realized_var'] = sp500['log_return'].rolling(window=5).var() * 252  # Annualized
    
        # Drop NA values
        sp500 = sp500.dropna()

        return sp500

def prepare_data(df, sequence_length=20, device='cpu'):
    """Prepare data for PINN training"""
    # Extract log returns and realized variance
    log_returns = df['log_return'].values
    realized_var = df['realized_var'].values

    # Create sequences
    X_returns = []
    X_rv = []

    for i in range(len(log_returns) - sequence_length):
        X_returns.append(log_returns[i:i+sequence_length])
        X_rv.append(realized_var[i:i+sequence_length])

    # Convert to tensors
    X_returns = torch.tensor(np.array(X_returns), dtype=torch.float32).reshape(-1, sequence_length, 1)
    X_rv = torch.tensor(np.array(X_rv), dtype=torch.float32).reshape(-1, sequence_length, 1)

    # Move tensors to device
    X_returns = X_returns.to(device)
    X_rv = X_rv.to(device)

    return X_returns, X_rv

def create_dataloader(returns, realized_var, batch_size=64, shuffle=True):
    """Create a DataLoader for training"""
    dataset = TensorDataset(returns, realized_var)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
