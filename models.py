import torch
import torch.nn as nn
import torch.nn.functional as F

class RealizedGARCHPINNv2(nn.Module):
    def __init__(self, hidden_dim=128, time_features=True):
        super(RealizedGARCHPINNv2, self).__init__()

        # Parameters to learn from the GARCH equation - adjusted for 5-minute data
        # Initialize with smaller values to prevent explosions during training
        self.omega = nn.Parameter(torch.tensor([0.01], dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor([0.7], dtype=torch.float32))  # Persistence
        self.tau1 = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))  # Smaller leverage effect
        self.tau2 = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))  # Feedback

        # Parameters for the measurement equation
        self.xi = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.phi = nn.Parameter(torch.tensor([0.95], dtype=torch.float32))  # Closer to 1 for stability
        self.delta1 = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))
        self.delta2 = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))

        # Mean parameter
        self.mu = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

        # Flag for time features
        self.time_features = time_features
        
        # Input size depends on whether we include time features
        input_size = 3
        if time_features:
            input_size += 2  # Add time of day and day of week features
            
        # Neural network with proper initialization for stability
        self.nn_enhancement = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),  # Add batch normalization
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),  # Add batch normalization
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Limit output range to [-1, 1] for stability
        )
        
        # Initialize weights properly
        for m in self.nn_enhancement.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)

    def tau_func(self, z):
        """Leverage function τ(z) for the GARCH equation with clipping for stability"""
        # Clip z to prevent extreme values
        z_clipped = torch.clamp(z, -10.0, 10.0)
        return self.tau1 * z_clipped + self.tau2 * (torch.clamp(z_clipped**2, 0, 100) - 1)

    def delta_func(self, z):
        """Leverage function δ(z) for the measurement equation with clipping for stability"""
        # Clip z to prevent extreme values
        z_clipped = torch.clamp(z, -10.0, 10.0)
        return self.delta1 * z_clipped + self.delta2 * (torch.clamp(z_clipped**2, 0, 100) - 1)
    
    def generate_time_features(self, batch_size, seq_len, device):
        """
        Generate time-of-day and day-of-week features for 5-min data
        For 5-min data, there are 78 intervals per day (6.5 trading hours)
        """
        # Time of day feature (0 to 1 over the trading day)
        time_of_day = torch.linspace(0, 1, 78, device=device)
        # Repeat for each batch and sequence
        time_of_day = time_of_day.repeat(batch_size, seq_len // 78 + 1)
        time_of_day = time_of_day[:, :seq_len].unsqueeze(-1)
        
        # Day of week feature (0 to 1 over the week)
        days = torch.arange(seq_len // 78 + 1, device=device) % 5  # 5 trading days
        days = days.repeat_interleave(78)
        days = days[:seq_len].repeat(batch_size, 1).unsqueeze(-1) / 4.0  # Normalize to [0,1]
        
        return time_of_day, days
        
    def forward(self, returns, log_rv):
        batch_size, seq_len, _ = returns.shape
        device = returns.device

        # Pre-allocate output arrays
        all_log_h = []
        all_log_x = []
        all_z = []
        all_u = []

        # Generate time features if enabled
        if self.time_features:
            time_of_day, day_of_week = self.generate_time_features(batch_size, seq_len, device)

        # Initialize state with a reasonable, clipped value
        if seq_len > 0:
            # Safely initialize with clipped values
            init_rv = torch.exp(torch.clamp(log_rv[:, 0:1, :], -10, 2))
            h_t = torch.mean(init_rv, dim=1, keepdim=True)
            log_h_t = torch.log(torch.clamp(h_t, 1e-6, 1e2))  # Prevent extreme logs
        else:
            h_t = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=device) * 1e-4
            log_h_t = torch.log(h_t)

        # Process each time step
        for t in range(seq_len):
            # Apply GARCH equation to get current volatility
            if t > 0:
                tau_term = self.tau_func(all_z[t-1])
                # Clip values to ensure numerical stability
                log_h_component = self.omega + self.beta * torch.clamp(all_log_h[t-1], -10, 2) + tau_term + self.gamma * torch.clamp(all_log_x[t-1], -10, 2)
                log_h_t = torch.clamp(log_h_component, -10, 2)  # Prevent extreme values

            # Calculate standardized returns with stability safeguards
            h_t = torch.exp(log_h_t)
            # Add small epsilon to prevent division by zero
            z_t = (returns[:, t:t+1, :] - self.mu) / (torch.sqrt(torch.clamp(h_t, 1e-6, 1e2)) + 1e-8)
            z_t = torch.clamp(z_t, -10, 10)  # Clip standardized returns

            # Apply measurement equation with stability safeguards
            delta_term = self.delta_func(z_t)
            log_x_t = self.xi + self.phi * log_h_t + delta_term
            log_x_t = torch.clamp(log_x_t, -10, 2)  # Clip to reasonable values
            
            # Calculate measurement residual
            u_t = torch.clamp(log_rv[:, t:t+1, :], -10, 2) - log_x_t

            # Apply NN enhancement with safety measures
            if self.time_features and t < seq_len:
                nn_input = torch.cat([
                    log_h_t, 
                    z_t, 
                    u_t, 
                    time_of_day[:, t:t+1, :],
                    day_of_week[:, t:t+1, :]
                ], dim=2)
            else:
                nn_input = torch.cat([log_h_t, z_t, u_t], dim=2)
                
            # Reshape for batch norm
            orig_shape = nn_input.shape
            nn_input_reshaped = nn_input.view(-1, orig_shape[-1])
            nn_output = self.nn_enhancement(nn_input_reshaped).view(orig_shape[0], orig_shape[1], 1)
            
            # Add a small enhancement that won't destabilize
            enhanced_log_h_t = log_h_t + 0.005 * nn_output  # Reduced impact
            enhanced_log_h_t = torch.clamp(enhanced_log_h_t, -10, 2)  # Clip again

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
        # Ensure realized_var is positive
        realized_var = torch.clamp(realized_var, 1e-8, 1e4)
        log_rv = torch.log(realized_var)

        # Forward pass
        log_h, log_x, z, u = self.forward(returns, log_rv)

        # Loss for return equation - with safety checks
        return_diff = returns - self.mu - torch.sqrt(torch.clamp(torch.exp(log_h), 1e-8, 1e4)) * z
        return_loss = torch.mean(torch.clamp(return_diff**2, 0, 100))

        # Loss for measurement equation - with stability constraints
        measurement_diff = log_rv - log_x
        measurement_loss = torch.mean(torch.clamp(measurement_diff**2, 0, 100))

        # Loss for GARCH equation (for t > 0) - with numerical stability
        if log_h.shape[1] > 1:  # Check we have more than one time step
            garch_term1 = log_h[:, 1:, :]
            garch_term2 = self.omega + self.beta * log_h[:, :-1, :] + self.tau_func(z[:, :-1, :]) + self.gamma * log_x[:, :-1, :]
            garch_eq = garch_term1 - garch_term2
            garch_loss = torch.mean(torch.clamp(garch_eq**2, 0, 100))
        else:
            garch_loss = torch.tensor(0.0, device=returns.device)

        # Variance of the error term u - with clamping
        u_squared = torch.clamp(u**2, 0, 100)
        sigma_u_squared = torch.mean(u_squared) if u.shape[1] > 0 else torch.tensor(0.0, device=returns.device)

        # Data-driven loss for prediction - with clamping
        if log_x.shape[1] > 0:
            pred_diff = torch.exp(torch.clamp(log_x, -10, 2)) - realized_var
            data_loss = torch.mean(torch.clamp(pred_diff**2, 0, 100))
        else:
            data_loss = torch.tensor(0.0, device=returns.device)

        # Smoothness penalty - with safe gradient computation
        if log_h.shape[1] > 1:
            diff = log_h[:, 1:, :] - log_h[:, :-1, :]
            smoothness_penalty = torch.mean(torch.clamp(diff**2, 0, 10))
        else:
            smoothness_penalty = torch.tensor(0.0, device=returns.device)

        # Parameter regularization to prevent extreme values
        param_reg = self.beta**2 + self.tau1**2 + self.tau2**2 + self.gamma**2 + self.phi**2 + self.delta1**2 + self.delta2**2

        # Total loss with weights adjusted for high-frequency data
        total_loss = (
            return_loss + 
            measurement_loss + 
            garch_loss + 
            0.1 * data_loss +  
            0.01 * sigma_u_squared +
            0.02 * smoothness_penalty +
            0.001 * param_reg  # Add regularization
        )

        # Final safety check - if NaN appears, return a reasonable default
        if torch.isnan(total_loss):
            print("Warning: NaN detected in loss calculation. Using default loss value.")
            total_loss = torch.tensor(1000.0, device=returns.device)
            return total_loss, {
                'return_loss': 200.0,
                'measurement_loss': 200.0,
                'garch_loss': 200.0,
                'data_loss': 200.0,
                'sigma_u': 10.0,
                'smoothness_penalty': 10.0,
                'total_loss': 1000.0
            }

        return total_loss, {
            'return_loss': return_loss.item(),
            'measurement_loss': measurement_loss.item(),
            'garch_loss': garch_loss.item(),
            'data_loss': data_loss.item(),
            'sigma_u': torch.sqrt(torch.clamp(sigma_u_squared, 1e-8, 100)).item(),
            'smoothness_penalty': smoothness_penalty.item() if smoothness_penalty > 0 else 0.0,
            'param_reg': param_reg.item(),
            'total_loss': total_loss.item()
        }
