import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

def train_model(model, returns, realized_var, epochs=1000, lr=0.001, batch_size=64):
    """Train the PINN model"""
    from data_utils import create_dataloader
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    dataloader = create_dataloader(returns, realized_var, batch_size, shuffle=True)

    loss_history = []
    detailed_losses = {'return_loss': [], 'measurement_loss': [], 'garch_loss': [],
                       'data_loss': [], 'sigma_u': []}

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_detailed_losses = {'return_loss': 0.0, 'measurement_loss': 0.0,
                                'garch_loss': 0.0, 'data_loss': 0.0, 'sigma_u': 0.0}

        for batch_returns, batch_rv in dataloader:
            optimizer.zero_grad()

            try:
                loss, detailed_loss = model.physics_loss(batch_returns, batch_rv)
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}, "
              f"Return Loss: {epoch_detailed_losses['return_loss']:.6f}, "
              f"GARCH Loss: {epoch_detailed_losses['garch_loss']:.6f}")

    # Disable anomaly detection after training
    torch.autograd.set_detect_anomaly(False)

    return loss_history, detailed_losses

def forecast_volatility(model, returns, realized_var, forecast_horizon=10):
    """Forecast volatility using the trained PINN model"""
    model.eval()
    with torch.no_grad():
        # Get last sequence for initialization
        last_returns = returns[-1:, -forecast_horizon:, :]
        last_rv = realized_var[-1:, -forecast_horizon:, :]

        # Forward pass to get current state
        log_h, _, z, u = model.forward(last_returns, torch.log(last_rv))

        # Extract last values
        current_h = torch.exp(log_h[0, -1, 0]).item()
        current_z = z[0, -1, 0].item()
        current_u = u[0, -1, 0].item()

        # Forecast
        forecasted_h = np.zeros(forecast_horizon + 1)
        forecasted_h[0] = current_h

        for t in range(1, forecast_horizon + 1):
            log_h_t = (model.omega.item() +
                      model.beta.item() * np.log(forecasted_h[t-1]) +
                      model.tau1.item() * current_z +
                      model.tau2.item() * (current_z**2 - 1) +
                      model.gamma.item() * current_u)

            forecasted_h[t] = np.exp(log_h_t)

            # Reset innovations to zero for future periods (or can sample from distributions)
            current_z = 0
            current_u = 0

    return forecasted_h
