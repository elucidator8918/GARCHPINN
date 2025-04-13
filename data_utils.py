import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import TensorDataset, DataLoader

def get_data(start_date='2015-01-01', end_date='2022-12-31', dataset="FTSC"):
    """Download S&P 500 data and calculate log returns and realized variance"""
    if dataset =="FTSC":
        df = pd.read_csv("UK_Index_2000_2025.csv")
        df['Date-Time'] = pd.to_datetime(df['Date-Time'])
        df = df.dropna(axis=1, how='all')
        
        df['Date-Time'] = df.apply(
            lambda row: row['Date-Time'] + pd.Timedelta(hours=row['GMT Offset']),
            axis=1
        )
        
        df = df.drop(columns=["#RIC", "Domain", "Type", "No. Trades", "GMT Offset"])

        # Calculate log returns
        df['log_return'] = np.log(df['Last'] / df['Last'].shift(1))
        df['realized_var'] = df['log_return'].rolling(window=10).var() * 19656

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
