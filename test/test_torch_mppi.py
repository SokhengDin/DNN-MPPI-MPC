import torch
from pytorch_mppi import MPPI

# Define the dynamics of your system
def dynamics(state, control):
    # Example: simple linear dynamics
    A = torch.tensor([[1.0, 0.1], [0.0, 1.0]], device=state.device)
    B = torch.tensor([[0.0], [0.1]], device=state.device)

    # Ensure control is properly shaped for batch processing
    control = control.view(control.shape[0], -1)  # Shape (num_samples, control_dim)
    control = control.unsqueeze(-1)  # Shape (num_samples, control_dim, 1)
    
    next_state = A @ state + B @ control
    return next_state

# Define the running cost
def running_cost(state, control):
    cost = (state ** 2).sum(dim=-2) + (control ** 2).sum(dim=-2)
    return cost

# Define the terminal cost (optional)
def terminal_cost(state):
    cost = (state ** 2).sum(dim=-2)
    return cost

# Create the MPPI controller
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

horizon = 20
num_samples = 1000
noise_sigma = torch.diag(torch.tensor([0.5, 0.5], dtype=dtype, device=device))

mppi = MPPI(dynamics, running_cost, 1,
            terminal_state_cost=terminal_cost,
            noise_sigma=noise_sigma,
            num_samples=num_samples,
            horizon=horizon, device=device,
            u_max=torch.tensor([2.0], dtype=dtype, device=device),
            lambda_=1.0)

# Initial state
state = torch.tensor([[1.0], [0.0]], dtype=dtype, device=device)  # Shape (2, 1)
state = state.unsqueeze(0).repeat(num_samples, 1, 1)  # Shape (num_samples, 2, 1)

# Run MPPI to get the control action
control = mppi.command(state)
print("Control action:", control)
