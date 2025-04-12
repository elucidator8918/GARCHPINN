import numpy as np
import torch
import wandb
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Import modules
from data_utils import get_data, prepare_data
from models import RealizedGARCHPINNv2
from training import train_model, forecast_volatility
from visualization import plot_results

def main():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Weights & Biases
    wandb.init(project="pinn-garch-volatility-forecasting", entity=None)

    # Get S&P 500 data
    sp500 = get_data()
    print(f"Data shape: {sp500.shape}")

    # Log data statistics to W&B
    wandb.log({
        "data_points": len(sp500),
        "mean_return": sp500['log_return'].mean(),
        "mean_volatility": np.sqrt(sp500['realized_var'].mean() * 252) * 100
    })

    # Prepare data
    sequence_length  np.sqrt(sp500['realized_var'].mean() * 252) * 100
    })

    # Prepare data
    sequence_length = 20
    returns, realized_var = prepare_data(sp500, sequence_length, device)
    print(f"Prepared data shapes: Returns {returns.shape}, RV {realized_var.shape}")

    # Create model - use version 2 which avoids in-place operations
    model = RealizedGARCHPINNv2(hidden_dim=64)
    # Move model to device (GPU if available)
    model = model.to(device)

    # Log model architecture to W&B
    wandb.watch(model, log='all', log_freq=100)

    # Print initial parameters
    print("\nInitial Parameters:")
    print(f"omega: {model.omega.item():.6f}")
    print(f"beta: {model.beta.item():.6f}")
    print(f"tau1: {model.tau1.item():.6f}")
    print(f"tau2: {model.tau2.item():.6f}")

    # Verify tensor shapes before training
    test_batch_returns = returns[:2]
    test_batch_rv = realized_var[:2]
    print(f"Test batch shapes: returns {test_batch_returns.shape}, rv {test_batch_rv.shape}")

    try:
        # Test forward pass
        log_h, log_x, z, u = model.forward(test_batch_returns, torch.log(test_batch_rv))
        print(f"Forward pass shapes: log_h {log_h.shape}, z {z.shape}, u {u.shape}")

        # Now train the model with fewer epochs for demonstration
        loss_history, detailed_losses = train_model(model, returns, realized_var, epochs=100, lr=0.001, batch_size=512)

        # Print learned parameters
        print("\nLearned Parameters:")
        print(f"omega: {model.omega.item():.6f}")
        print(f"beta: {model.beta.item():.6f}")
        print(f"tau1: {model.tau1.item():.6f}")
        print(f"tau2: {model.tau2.item():.6f}")
        print(f"gamma: {model.gamma.item():.6f}")
        print(f"xi: {model.xi.item():.6f}")
        print(f"phi: {model.phi.item():.6f}")
        print(f"delta1: {model.delta1.item():.6f}")
        print(f"delta2: {model.delta2.item():.6f}")
        print(f"mu: {model.mu.item():.6f}")

        # Log final parameters to W&B
        wandb.log({
            "final_omega": model.omega.item(),
            "final_beta": model.beta.item(),
            "final_tau1": model.tau1.item(),
            "final_tau2": model.tau2.item(),
            "final_gamma": model.gamma.item(),
            "final_xi": model.xi.item(),
            "final_phi": model.phi.item(),
            "final_delta1": model.delta1.item(),
            "final_delta2": model.delta2.item(),
            "final_mu": model.mu.item()
        })

        # Forecast volatility
        forecast_horizon = 10
        forecasted_volatility = forecast_volatility(model, returns, realized_var, forecast_horizon)

        # Log forecast to W&B
        wandb.log({
            "forecast": wandb.Table(data=pd.DataFrame({
                "days_ahead": range(len(forecasted_volatility)),
                "volatility": np.sqrt(forecasted_volatility * 252) * 100
            }))
        })

        # Plot results
        plot_results(sp500, loss_history, detailed_losses, forecasted_volatility)

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Finish W&B run
        wandb.finish()

if __name__ == "__main__":
    main()
