"""Configuration parameters for the PINN-GARCH model"""

# Data parameters
DATA_START_DATE = '2015-01-01'
DATA_END_DATE = '2022-12-31'
SEQUENCE_LENGTH = 20

# Model parameters
HIDDEN_DIM = 64

# Training parameters
EPOCHS = 1000
LEARNING_RATE = 0.001
BATCH_SIZE = 64
LR_STEP_SIZE = 100
LR_GAMMA = 0.5
GRAD_CLIP_VALUE = 1.0

# Forecasting parameters
FORECAST_HORIZON = 10

# Visualization parameters
HIST_WINDOW = 100  # Number of historical days to show in plots
