import torch
import torch.nn as nn

class RealizedGARCHPINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super(RealizedGARCHPINN, self).__init__()

        # Parameters to learn from the GARCH equation
        self.omega = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor([0.8], dtype=torch.float32))
        self.tau1 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.tau2 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

        # Parameters for the measurement equation
        self.xi = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.phi = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.delta1 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.delta2 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

        # Mean parameter
        self.mu = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

        # Neural network for enhancing the model
        self.nn_enhancement = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def tau_func(self, z):
        """Leverage function τ(z) for the GARCH equation"""
        return self.tau1 * z + self.tau2 * (z**2 - 1)

    def delta_func(self, z):
        """Leverage function δ(z) for the measurement equation"""
        return self.delta1 * z + self.delta2 * (z**2 - 1)

    def forward(self, returns, log_rv):
        batch_size, seq_len, _ = returns.shape

        # Initialize output tensors
        log_h = torch.zeros(batch_size, seq_len, 1, dtype=torch.float32, device=returns.device)
        log_x = torch.zeros(batch_size, seq_len, 1, dtype=torch.float32, device=returns.device)
        z = torch.zeros(batch_size, seq_len, 1, dtype=torch.float32, device=returns.device)
        u = torch.zeros(batch_size, seq_len, 1, dtype=torch.float32, device=returns.device)

        # Initial values - use first element's variance as initial h
        h_prev = torch.mean(torch.exp(log_rv[:, 0:1, :]), dim=1, keepdim=True)

        # Process each time step
        for t in range(seq_len):
            # GARCH equation
            if t == 0:
                log_h_t = torch.log(h_prev)
            else:
                tau_term = self.tau_func(z[:, t-1:t, :])
                log_h_t = self.omega + self.beta * log_h[:, t-1:t, :] + tau_term + self.gamma * u[:, t-1:t, :]

            # NO IN-PLACE OPERATIONS - create a new tensor for each update
            log_h = torch.cat([log_h[:, :t, :], log_h_t, log_h[:, t+1:, :]], dim=1)
            h_t = torch.exp(log_h_t)

            # Return equation
            z_t = (returns[:, t:t+1, :] - self.mu) / torch.sqrt(h_t)
            z = torch.cat([z[:, :t, :], z_t, z[:, t+1:, :]], dim=1)

            # Measurement equation
            delta_term = self.delta_func(z_t)
            u_t = log_rv[:, t:t+1, :] - (self.xi + self.phi * log_h_t + delta_term)
            u = torch.cat([u[:, :t, :], u_t, u[:, t+1:, :]], dim=1)

            log_x_t = self.xi + self.phi * log_h_t + delta_term + u_t
            log_x = torch.cat([log_x[:, :t, :], log_x_t, log_x[:, t+1:, :]], dim=1)

            # Apply NN enhancement
            nn_input = torch.cat([log_h_t, z_t, u_t], dim=2)
            nn_output = self.nn_enhancement(nn_input)

            # Update log_h with neural network enhancement (no in-place operation)
            enhanced_log_h_t = log_h_t + 0.01 * nn_output
            log_h = torch.cat([log_h[:, :t, :], enhanced_log_h_t, log_h[:, t+1:, :]], dim=1)

        return log_h, log_x, z, u

class RealizedGARCHPINNv2(nn.Module):
    def __init__(self, hidden_dim=64):
        super(RealizedGARCHPINNv2, self).__init__()

        # Parameters to learn from the GARCH equation
        self.omega = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor([0.8], dtype=torch.float32))
        self.tau1 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.tau2 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

        # Parameters for the measurement equation
        self.xi = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.phi = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.delta1 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.delta2 = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

        # Mean parameter
        self.mu = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

        # Neural network for enhancing the model
        self.nn_enhancement = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def tau_func(self, z):
        """Leverage function τ(z) for the GARCH equation"""
        return self.tau1 * z + self.tau2 * (z**2 - 1)

    def delta_func(self, z):
        """Leverage function δ(z) for the measurement equation"""
        return self.delta1 * z + self.delta2 * (z**2 - 1)

    def forward(self, returns, log_rv):
        batch_size, seq_len, _ = returns.shape
        device = returns.device

        # Pre-allocate output arrays
        all_log_h = []
        all_log_x = []
        all_z = []
        all_u = []

        # Initialize state
        if seq_len > 0:
            h_t = torch.mean(torch.exp(log_rv[:, 0:1, :]), dim=1, keepdim=True)
            log_h_t = torch.log(h_t)
        else:
            h_t = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=device) * 1e-4
            log_h_t = torch.log(h_t)

        # Process each time step
        for t in range(seq_len):
            # Apply GARCH equation to get current volatility
            if t > 0:
                tau_term = self.tau_func(all_z[t-1])
                log_h_t = self.omega + self.beta * all_log_h[t-1] + tau_term + self.gamma * all_log_x[t-1]

            # Calculate standardized returns
            h_t = torch.exp(log_h_t)
            z_t = (returns[:, t:t+1, :] - self.mu) / torch.sqrt(h_t)

            # Apply measurement equation
            delta_term = self.delta_func(z_t)
            log_x_t = self.xi + self.phi * log_h_t + delta_term
            u_t = log_rv[:, t:t+1, :] - log_x_t

            # Apply NN enhancement
            nn_input = torch.cat([log_h_t, z_t, u_t], dim=2)
            nn_output = self.nn_enhancement(nn_input)
            enhanced_log_h_t = log_h_t + 0.01 * nn_output

            # Store values
            all_log_h.append(enhanced_log_h_t)
            all_z.append(z_t)
            all_u.append(u_t)
            all_log_x.append(log_x_t)

        # Stack outputs along time dimension
        if seq_len > 0:
            log_h = torch.cat(all_log_h, dim=1)
            log_x = torch.cat(all_log_x, dim=1)
            z = torch.cat(all_z, dim=1)
            u = torch.cat(all_u, dim=1)
        else:
            # Handle empty sequence case
            log_h = torch.zeros((batch_size, 0, 1), dtype=torch.float32, device=device)
            log_x = torch.zeros((batch_size, 0, 1), dtype=torch.float32, device=device)
            z = torch.zeros((batch_size, 0, 1), dtype=torch.float32, device=device)
            u = torch.zeros((batch_size, 0, 1), dtype=torch.float32, device=device)

        return log_h, log_x, z, u

    def physics_loss(self, returns, realized_var):
        """Calculate the physics-informed loss based on the Realized GARCH model"""
        log_rv = torch.log(realized_var)

        # Forward pass
        log_h, log_x, z, u = self.forward(returns, log_rv)

        # Loss for return equation
        return_loss = torch.mean((returns - self.mu - torch.sqrt(torch.exp(log_h)) * z)**2)

        # Loss for measurement equation
        measurement_loss = torch.mean((log_rv - log_x)**2)

        # Loss for GARCH equation (for t > 0)
        if log_h.shape[1] > 1:  # Check we have more than one time step
            # Fixed here: Using log_x_{t-1} instead of u_{t-1}
            garch_eq = log_h[:, 1:, :] - (self.omega + self.beta * log_h[:, :-1, :] +
                                         self.tau_func(z[:, :-1, :]) + self.gamma * log_x[:, :-1, :])
            garch_loss = torch.mean(garch_eq**2)
        else:
            garch_loss = torch.tensor(0.0, device=returns.device)

        # Variance of the error term u
        sigma_u_squared = torch.mean(u**2) if u.shape[1] > 0 else torch.tensor(0.0, device=returns.device)

        # Data-driven loss for prediction
        data_loss = torch.mean((torch.exp(log_x) - realized_var)**2) if log_x.shape[1] > 0 else torch.tensor(0.0, device=returns.device)

        # Total loss
        total_loss = return_loss + measurement_loss + garch_loss + 0.1 * data_loss + 0.01 * sigma_u_squared

        return total_loss, {
            'return_loss': return_loss.item(),
            'measurement_loss': measurement_loss.item(),
            'garch_loss': garch_loss.item(),
            'data_loss': data_loss.item(),
            'sigma_u': torch.sqrt(sigma_u_squared).item() if sigma_u_squared > 0 else 0.0,
            'total_loss': total_loss.item()
        }
