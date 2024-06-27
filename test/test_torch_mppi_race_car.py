import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch_mppi import MPPI

def dynamics(states, actions, t=None):
    dt = 0.05  # time step
    x, y, theta, v = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    
    if actions.dim() == 1:
        actions = actions.unsqueeze(0)
    
    acc, delta = actions[:, 0], actions[:, 1]

    # Race car dynamics model
    lr = 1.5  # Distance from CG to rear axle
    lf = 1.0  # Distance from CG to front axle
    m = 1000.0  # Mass of the car
    Iz = 1500.0  # Moment of inertia about the z-axis
    Cf = 1000.0  # Cornering stiffness of front tires
    Cr = 1000.0  # Cornering stiffness of rear tires

    # Slip angles
    alpha_f = torch.atan2(v * torch.sin(delta) + lf * theta, v * torch.cos(delta)) - delta
    alpha_r = torch.atan2(v * torch.sin(delta) - lr * theta, v * torch.cos(delta))

    # Lateral forces
    Fyf = Cf * alpha_f
    Fyr = Cr * alpha_r

    # Update states
    x_next = x + v * torch.cos(theta) * dt
    y_next = y + v * torch.sin(theta) * dt
    theta_next = theta + (v / lr) * torch.sin(delta) * dt
    v_next = v + (acc - (Fyf * torch.sin(delta) + Fyr) / m) * dt

    next_states = torch.stack((x_next, y_next, theta_next, v_next), dim=1)
    return next_states


def running_cost(states, actions):
    desired_trajectory = torch.tensor([[5.0, 5.0, 1.57, 1.0]], dtype=torch.double, device=states.device).expand(states.shape[0], -1)

    Q = torch.diag(torch.tensor([10.0, 5.0, 9.0, 1.0], dtype=torch.double, device=states.device))
    R = torch.diag(torch.tensor([1.0, 10], dtype=torch.double, device=states.device))

    state_cost = torch.einsum('bi,ij,bj->b', states - desired_trajectory, Q, states - desired_trajectory)
    control_cost = torch.einsum('bi,ij,bj->b', actions, R, actions)

    # Obstacle avoidance
    obstacle_positions = torch.tensor([[5.0, 5.0], [7.0, 3.0]], dtype=torch.double, device=states.device)
    obstacle_cost = torch.zeros(states.shape[0], dtype=torch.double, device=states.device)
    
    for obstacle in obstacle_positions:
        distance_to_obstacle = torch.norm(states[:, :2] - obstacle, dim=1)
        obstacle_cost += torch.exp(-distance_to_obstacle) 

    total_cost = state_cost + control_cost + obstacle_cost

    return total_cost

nx = 4
nu = 2
T = 50
num_samples = 2000
lambda_ = 1.0

u_min = torch.tensor([-1.0, -0.5], dtype=torch.double)
u_max = torch.tensor([1.0, 0.5], dtype=torch.double)
noise_sigma = 0.1 * torch.eye(nu, dtype=torch.double)


device = torch.device("cuda")
dtype = torch.double
    
ctrl = MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=num_samples, horizon=T,
            lambda_=lambda_, device=device, 
            u_min=u_min, u_max=u_max)

state = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=dtype, device=device)

states_history = []
actions_history = []

for t in range(500):

    action = ctrl.command(state)

    state = dynamics(state, action)

    states_history.append(state.flatten().cpu().numpy())
    actions_history.append(action.flatten().cpu().numpy())

    print(f"Time step {t}:")
    print(f"State: {state.flatten().cpu().numpy()}")
    print(f"Optimal control: {action.flatten().cpu().numpy()}")
    print()


states_history = np.array(states_history)
actions_history = np.array(actions_history)


# Plot the states
plt.subplot(2, 1, 1)
plt.plot(states_history[:, 0], states_history[:, 1], label='Trajectory')
plt.scatter([5.0, 7.0], [5.0, 3.0], color='red', label='Obstacles')  # Plot obstacles
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Race Car Trajectory')
plt.legend()

# Plot the control inputs
plt.subplot(2, 1, 2)
plt.plot(actions_history[:, 0], label='Acceleration (acc)')
plt.plot(actions_history[:, 1], label='Steering (delta)')
plt.xlabel('Time step')
plt.ylabel('Control value')
plt.title('Control Inputs')
plt.legend()

plt.tight_layout()
plt.show()