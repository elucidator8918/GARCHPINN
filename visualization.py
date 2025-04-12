import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_results(df, loss_history, detailed_losses, forecasted_volatility):
    """Plot training results and volatility forecasts"""
    # Create matplotlib plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot loss history
    ax = axes[0, 0]
    ax.plot(loss_history)
    ax.set_title('Total Loss over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')  # Log scale for better visualization

    # Plot detailed losses
    ax = axes[0, 1]
    for k, v in detailed_losses.items():
        if k != 'sigma_u' and k != 'total_loss':  # Skip some for clarity
            ax.plot(v, label=k)
    ax.set_title('Component Losses over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')  # Log scale for better visualization
    ax.legend()

    # Plot historical and forecasted volatility
    ax = axes[1, 0]
    # Historical volatility (last 100 days)
    historical_vol = np.sqrt(df['realized_var'].values[-100:] * 252) * 100  # Convert to annualized percentage
    ax.plot(range(len(historical_vol)), historical_vol, label='Historical Volatility')

    # Forecasted volatility
    forecast_dates = np.array(range(len(historical_vol) - 1, len(historical_vol) + len(forecasted_volatility) - 1))
    forecasted_vol = np.sqrt(forecasted_volatility * 252) * 100  # Convert to annualized percentage
    ax.plot(forecast_dates, forecasted_vol, 'r--', label='Forecasted Volatility')

    ax.axvline(x=len(historical_vol) - 1, color='black', linestyle=':')
    ax.set_title('S&P 500 Volatility Forecast')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Annualized Volatility (%)')
    ax.legend()

    # Plot S&P 500 returns
    ax = axes[1, 1]
    returns = df['log_return'].values[-100:] * 100  # Convert to percentage
    ax.plot(returns)
    ax.set_title('S&P 500 Daily Returns')
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Daily Return (%)')

    plt.tight_layout()
    plt.show()

    # Create interactive Plotly figures
    create_interactive_plots(df, loss_history, detailed_losses, forecasted_volatility)

def create_interactive_plots(df, loss_history, detailed_losses, forecasted_volatility):
    """Create interactive Plotly visualizations"""
    # Create subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'Total Loss over Epochs',
        'Component Losses over Epochs',
        'S&P 500 Volatility Forecast',
        'S&P 500 Daily Returns'
    ))

    # Add total loss plot
    fig.add_trace(
        go.Scatter(y=loss_history, mode='lines', name='Total Loss'),
        row=1, col=1
    )
    fig.update_yaxes(title_text="Loss", type="log", row=1, col=1)

    # Add component losses plot
    for k, v in detailed_losses.items():
        if k != 'sigma_u' and k != 'total_loss':
            fig.add_trace(
                go.Scatter(y=v, mode='lines', name=k),
                row=1, col=2
            )
    fig.update_yaxes(title_text="Loss", type="log", row=1, col=2)

    # Add volatility forecast plot
    historical_vol = np.sqrt(df['realized_var'].values[-100:] * 252) * 100
    forecast_dates = np.array(range(len(historical_vol) - 1, len(historical_vol) + len(forecasted_volatility) - 1))
    forecasted_vol = np.sqrt(forecasted_volatility * 252) * 100

    fig.add_trace(
        go.Scatter(y=historical_vol, mode='lines', name='Historical Volatility'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=forecast_dates, y=forecasted_vol, mode='lines', name='Forecasted Volatility', line=dict(dash='dash')),
        row=2, col=1
    )
    fig.add_shape(
        type="line", x0=len(historical_vol)-1, y0=0, x1=len(historical_vol)-1, y1=1,
        yref="paper", line=dict(color="Black", dash="dot"),
        row=2, col=1
    )
    fig.update_yaxes(title_text="Annualized Volatility (%)", row=2, col=1)

    # Add returns plot
    returns = df['log_return'].values[-100:] * 100
    fig.add_trace(
        go.Scatter(y=returns, mode='lines', name='Daily Returns'),
        row=2, col=2
    )
    fig.update_yaxes(title_text="Daily Return (%)", row=2, col=2)

    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        title_text="S&P 500 Volatility Analysis",
        showlegend=True
    )

    # Show plot
    fig.show()

    # Create 3D plot of volatility surface
    create_3d_volatility_plot(df, forecasted_volatility)

def create_3d_volatility_plot(df, forecasted_volatility):
    """Create a 3D plot of the volatility surface"""
    # Prepare data for 3D plot
    historical_vol = np.sqrt(df['realized_var'].values[-100:] * 252) * 100
    forecast_vol = np.sqrt(forecasted_volatility * 252) * 100

    # Create proper time axis
    # Historical data points (0 to 99)
    hist_time = np.arange(len(historical_vol))
    # Forecast points continue from last historical point (100 to 100 + forecast length)
    forecast_time = np.arange(len(historical_vol), len(historical_vol) + len(forecast_vol))

    # Combine time axes
    full_time = np.concatenate([hist_time, forecast_time])
    # Create y-axis values (just zeros for this 2D path in 3D space)
    y_values = np.zeros_like(full_time)

    # Combine volatility values
    full_volatility = np.concatenate([historical_vol, forecast_vol])

    # Create 3D plot
    fig = go.Figure(data=[
        go.Scatter3d(
            x=full_time,
            y=y_values,
            z=full_volatility,
            mode='lines+markers',
            line=dict(width=6, color=full_volatility, colorscale='Viridis'),
            marker=dict(size=4, color=full_volatility, colorscale='Viridis'),
            name='Volatility Path',
            hovertemplate='Day: %{x}<br>Volatility: %{z:.2f}%<extra></extra>'
        ),
        go.Scatter3d(
            x=[hist_time[-1]],
            y=[0],
            z=[historical_vol[-1]],
            mode='markers',
            marker=dict(size=8, color='red'),
            name='Forecast Start',
            hovertemplate='Forecast Start<br>Day: %{x}<br>Volatility: %{z:.2f}%<extra></extra>'
        )
    ])

    # Highlight forecast period
    fig.add_trace(
        go.Scatter3d(
            x=forecast_time,
            y=np.zeros_like(forecast_time),
            z=forecast_vol,
            mode='lines',
            line=dict(width=6, color='red'),
            name='Forecast',
            hovertemplate='Forecast Day: %{x}<br>Volatility: %{z:.2f}%<extra></extra>'
        )
    )

    # Update layout
    fig.update_layout(
        title='3D Volatility Path with Forecast',
        scene=dict(
            xaxis_title='Time (Days)',
            yaxis_title='',
            zaxis_title='Annualized Volatility (%)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            ),
            annotations=[
                dict(
                    x=forecast_time[0],
                    y=0,
                    z=forecast_vol[0],
                    text="Forecast Start",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )
            ]
        ),
        height=800,
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.show()
