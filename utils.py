"""Utility functions for the PINN-GARCH model"""
import torch
import numpy as np
import os
import json

def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Additional settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, path="models/garch_pinn_model.pt"):
    """Save model weights and parameters"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), path)
    
    # Save model parameters as JSON
    params = {
        "omega": model.omega.item(),
        "beta": model.beta.item(),
        "tau1": model.tau1.item(),
        "tau2": model.tau2.item(),
        "gamma": model.gamma.item(),
        "xi": model.xi.item(),
        "phi": model.phi.item(),
        "delta1": model.delta1.item(),
        "delta2": model.delta2.item(),
        "mu": model.mu.item()
    }
    
    with open(path.replace(".pt", "_params.json"), "w") as f:
        json.dump(params, f, indent=4)
    
    print(f"Model saved to {path}")

def load_model(model, path="models/garch_pinn_model.pt"):
    """Load model weights"""
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

def annualize_volatility(variance, days=252):
    """Convert daily variance to annualized volatility in percentage"""
    return np.sqrt(variance * days) * 100
