import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm

def train_model(model, returns, realized_var, epochs=1000, lr=0.001, batch_size=64):
    """Train the PINN model"""
    from data_utils import create_dataloader
    
    # Check for GPU availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to GPU
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    dataloader = create_dataloader(returns, realized_var, batch_size, shuffle=True)

    loss_history = []
    detailed_losses = {'return_loss': [], 'measurement_loss': [], 'garch_loss': [],
                       'data_loss': [], 'sigma_u': []}

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    # Main training loop with tqdm progress bar
    for epoch in tqdm(range(epochs), desc="Training epochs", ncols=100):
        epoch_loss = 0.0
        epoch_detailed_losses = {'return_loss': 0.0, 'measurement_loss': 0.0,
                                'garch_loss': 0.0, 'data_loss': 0.0, 'sigma_u': 0.0}

        # Batch progress bar
        batch_progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, ncols=100)
        for batch_returns, batch_rv in batch_progress:
            # Move batch data to GPU
            batch_returns = batch_returns.to(device)
            batch_rv = batch_rv.to(device)
            
            optimizer.zero_grad()

            try:
                loss, detailed_loss = model.physics_loss(batch_returns, batch_rv)
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Update progress bar with current loss
                batch_progress.set_postfix(loss=f"{loss.item():.6f}")
                
                epoch_loss += loss.item()
                for k, v in detailed_loss.items():
                    if k in epoch_detailed_losses:
                        epoch_detailed_losses[k] += v
            except Exception as e:
                print(f"Error in batch: {e}")
                print(f"Batch shapes: returns {batch_returns.shape}, rv {batch_rv.shape}")
                continue

        scheduler.step()

        # Average losses for the epoch
        num_batches = len(dataloader)
        if num_batches > 0:  # Avoid division by zero
            epoch_loss /= num_batches
            for k in epoch_detailed_losses:
                epoch_detailed_losses[k] /= num_batches
                detailed_losses[k].append(epoch_detailed_losses[k])
        else:
            continue

        loss_history.append(epoch_loss)

        # Log metrics to Weights & Biases
        wandb.log({
            "epoch": epoch,
            "total_loss": epoch_loss,
            "return_loss": epoch_detailed_losses['return_loss'],
            "measurement_loss": epoch_detailed_losses['measurement_loss'],
            "garch_loss": epoch_detailed_losses['garch_loss'],
            "data_loss": epoch_detailed_losses['data_loss'],
            "sigma_u": epoch_detailed_losses['sigma_u'],
            "learning_rate": scheduler.get_last_lr()[0]
        })

        # Update main progress bar with epoch stats
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}, "
                   f"Return Loss: {epoch_detailed_losses['return_loss']:.6f}, "
                   f"GARCH Loss: {epoch_detailed_losses['garch_loss']:.6f}")

    # Disable anomaly detection after training
    torch.autograd.set_detect_anomaly(False)

    return loss_history, detailed_losses

def forecast_volatility(model, returns, realized_var, forecast_horizon=10):
    """Forecast volatility using the trained PINN model"""
    # Ensure model is in evaluation mode
    model.eval()
    
    # Check for GPU availability and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Make sure model and input data are on the same device
    model = model.to(device)
    
    with torch.no_grad():
        # Get last sequence for initialization
        last_returns = returns[-1:, -forecast_horizon:, :].to(device)
        last_rv = realized_var[-1:, -forecast_horizon:, :].to(device)

        # Forward pass to get current state
        log_h, _, z, u = model.forward(last_returns, torch.log(last_rv))

        # Extract last values and move to CPU for numpy operations
        current_h = torch.exp(log_h[0, -1, 0]).cpu().item()
        current_z = z[0, -1, 0].cpu().item()
        current_u = u[0, -1, 0].cpu().item()

        # Get model parameters and move to CPU for numpy operations
        omega = model.omega.cpu().item()
        beta = model.beta.cpu().item()
        tau1 = model.tau1.cpu().item()
        tau2 = model.tau2.cpu().item()
        gamma = model.gamma.cpu().item()

        # Forecast with progress bar
        forecasted_h = np.zeros(forecast_horizon + 1)
        forecasted_h[0] = current_h

        for t in tqdm(range(1, forecast_horizon + 1), desc="Forecasting volatility", ncols=100):
            log_h_t = (omega +
                      beta * np.log(forecasted_h[t-1]) +
                      tau1 * current_z +
                      tau2 * (current_z**2 - 1) +
                      gamma * current_u)

            forecasted_h[t] = np.exp(log_h_t)

            # Reset innovations to zero for future periods (or can sample from distributions)
            current_z = 0
            current_u = 0

    return forecasted_h
