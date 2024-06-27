import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from pytorch_mppi import MPPI
from models.differentialSimV2 import DiffSimulation

def dynamics(states, actions, t=None):
    dt = 0.05  # time step
    x, y, theta = states[:, 0], states[:, 1], states[:, 2]
    
    if actions.dim() == 1:
        actions = actions.unsqueeze(0)
    
    v, omega = actions[:, 0], actions[:, 1]

    x_next = x + v * torch.cos(theta) * dt
    y_next = y + v * torch.sin(theta) * dt
    theta_next = theta + omega * dt

    next_states = torch.stack((x_next, y_next, theta_next), dim=1)
    return next_states



def running_cost(states, actions):
    desired_trajectory = torch.tensor([[6.0, 6.0, 1.57]], dtype=torch.float32, device=states.device).expand(states.shape[0], -1)

    Q = torch.diag(torch.tensor([20, 5, 9], dtype=torch.float32, device=states.device))
    R = torch.diag(torch.tensor([0.1, 0.1], dtype=torch.float32, device=states.device))

    state_cost = torch.einsum('bi,ij,bj->b', states - desired_trajectory, Q, states - desired_trajectory)
    control_cost = torch.einsum('bi,ij,bj->b', actions, R, actions)

    obstacle_positions = torch.tensor([[5.0, 4.0], [3.5, 3.5]], dtype=torch.float32, device=states.device)
    obstacle_cost = torch.zeros(states.shape[0], dtype=torch.float32, device=states.device)

    safety_distance = 0.8
    
    for obstacle in obstacle_positions:
        distance_to_obstacle = torch.norm(states[:, :2] - obstacle, dim=1)
        obstacle_cost += 1.0 / (distance_to_obstacle + 1e-6) * (distance_to_obstacle < safety_distance)

    total_cost = state_cost + control_cost + 10.0 * obstacle_cost

    return total_cost


class MPPIWrapper(MPPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.u_min = kwargs.get('u_min', None)
        self.u_max = kwargs.get('u_max', None)
    
    def get_trajectories(self, state):
        self.command(state)  

        # Calculate optimal trajectory
        optimal_traj = torch.zeros((self.T, self.nx), device=self.d)
        x = state.unsqueeze(0)  # Add batch dimension
        for t in range(self.T):
            x = self._dynamics(x, self.u_scale * self.U[t].unsqueeze(0), t)
            optimal_traj[t] = x.squeeze(0)

        # Get top 10% of sampled trajectories
        num_top_samples = max(10, self.K // 10)
        sorted_idx = torch.argsort(self.cost_total)[:num_top_samples]
        sampled_traj_list = torch.zeros((num_top_samples, self.T, self.nx), device=self.d)
        for i, k in enumerate(sorted_idx):
            x = state.unsqueeze(0)
            for t in range(self.T):
                x = self._dynamics(x, self.u_scale * self.perturbed_action[k, t].unsqueeze(0), t)
                sampled_traj_list[i, t] = x.squeeze(0)

        return optimal_traj.cpu().numpy(), sampled_traj_list.cpu().numpy()

    def savitzky_golay_coefficients(self, window_size: int, polynomial_order: int) -> torch.Tensor:
        half_window = (window_size - 1) // 2
        j = torch.arange(-half_window, half_window + 1).float()
        b = torch.stack([j ** i for i in range(polynomial_order + 1)], dim=1)
        m = torch.linalg.pinv(b)
        coeffs = m[0, :]  # Take only the first row of the matrix
        return torch.flip(coeffs, dims=[0])
            
    def savitzky_golay_filter(self, x: torch.Tensor, window_size: int, polynomial_order: int) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)

        assert x.dim() == 2, "Input tensor must be 1D or 2D"

        num_channels = x.shape[1]
        filtered_x = torch.zeros_like(x)

        for channel in range(num_channels):
            x_channel = x[:, channel]
            half_window = (window_size - 1) // 2
            coefficients = self.savitzky_golay_coefficients(window_size, polynomial_order).to(x.device)

            num_samples = x_channel.shape[0]
            filtered_channel = torch.zeros_like(x_channel)

            for i in range(num_samples):
                start = max(0, i - half_window)
                end = min(num_samples, i + half_window + 1)
                window = x_channel[start:end]
                
                window_size = window.shape[0]
                
                # Adjust coefficients to match the window size
                if window_size < coefficients.shape[0]:
                    center = coefficients.shape[0] // 2
                    start_coeff = center - window_size // 2
                    end_coeff = start_coeff + window_size
                    coeffs = coefficients[start_coeff:end_coeff]
                else:
                    coeffs = coefficients[:window_size]

                # Ensure both window and coeffs are 1D and non-empty
                if window.numel() == 0 or coeffs.numel() == 0:
                    filtered_channel[i] = x_channel[i]  # Use original value if filter can't be applied
                elif window.numel() == 1 and coeffs.numel() == 1:
                    filtered_channel[i] = window.item() * coeffs.item()
                else:
                    filtered_channel[i] = torch.dot(window, coeffs)

            filtered_x[:, channel] = filtered_channel

        if filtered_x.shape[1] == 1:
            filtered_x = filtered_x.squeeze(1)

        return filtered_x
    
    def moving_average_filter(self, xx: torch.Tensor, window_size: int) -> torch.Tensor:
        if xx.dim() == 1:
            xx = xx.unsqueeze(0)  # Add batch dimension if input is 1D
        
        input_length = xx.shape[0]
        dim = xx.shape[1]
        
        # Adjust window_size if it's larger than the input
        window_size = min(window_size, input_length)
        
        kernel = torch.ones(1, 1, window_size, dtype=xx.dtype, device=xx.device) / window_size
        xx_mean = torch.zeros_like(xx)

        for d in range(dim):
            x_channel = xx[:, d].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, input_length]
            padded = torch.nn.functional.pad(x_channel, (window_size//2, window_size//2), mode='replicate')
            xx_mean[:, d] = torch.nn.functional.conv1d(padded, kernel, padding=0).squeeze()

        return xx_mean.squeeze(0) if xx_mean.shape[0] == 1 else xx_mean


def run_simulation(mppi_ctrl, initial_state, tSim, delta_t):
    state = initial_state
    states_history = [state.cpu().numpy()]
    non_filtered_actions_history = []
    filtered_actions_history = []
    optimal_trajs = []
    sampled_trajs = []

    window_size = 5 
    polynomial_order = 2  

    for i in range(tSim):
        action = mppi_ctrl.command(state)
        optimal_traj, sampled_traj_list = mppi_ctrl.get_trajectories(state)
        
        # Store non-filtered action
        non_filtered_actions_history.append(action.cpu().numpy())
        
        # Apply Savitzky-Golay filter to the optimal control input
        filtered_action = mppi_ctrl.savitzky_golay_filter(action.unsqueeze(0), window_size, polynomial_order).squeeze(0)
        
        state = mppi_ctrl._dynamics(state.unsqueeze(0), filtered_action.unsqueeze(0), 0).squeeze(0)
        
        states_history.append(state.cpu().numpy())
        filtered_actions_history.append(filtered_action.cpu().numpy())
        optimal_trajs.append(optimal_traj)
        sampled_trajs.append(sampled_traj_list)

        print(f"Time: {i * delta_t:>2.2f}[s], x={state[0]:>+3.3f}[m], y={state[1]:>+3.3f}[m], yaw={state[2]:>+3.3f}[rad]")
        print(f"Non-filtered Input: {action.cpu().numpy()}")
        print(f"Filtered Input: {filtered_action.cpu().numpy()}")

    return np.array(states_history), np.array(non_filtered_actions_history), np.array(filtered_actions_history), np.array(optimal_trajs), np.array(sampled_trajs)


def create_animation(states_history, optimal_trajs, sampled_trajs, ref_path, delta_t, robot):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('MPPI Differential Drive with Optimal and Sampled Trajectories')
    ax.axis('equal')
    ax.grid(True)

    robot_artist, = ax.plot([], [], 'ro', markersize=10, zorder=5, label='Robot')
    optimal_traj_artist, = ax.plot([], [], color='#990099', linestyle="solid", linewidth=2, zorder=4, label='Optimal Trajectory')
    predicted_traj_artist, = ax.plot([], [], color='green', linestyle="solid", linewidth=2, zorder=4, label='Predicted Trajectory')
    rollout_collection = LineCollection([], colors='gray', linewidths=0.5, alpha=0.3, zorder=3)
    ax.add_collection(rollout_collection)
    ref_traj_artist, = ax.plot(ref_path[:, 0], ref_path[:, 1], color='blue', linestyle="dashed", linewidth=1.0, zorder=2, label='Reference Trajectory')
    obstacles = ax.scatter([5.0, 3.5], [4.0, 3.5], color='red', s=100, zorder=6, label='Obstacles')

    safety_distance = 0.8
    safety_circle = plt.Circle((0, 0), safety_distance, color='r', fill=False, linestyle='--', linewidth=1.0, zorder=4, label='Safety Distance')
    ax.add_patch(safety_circle)

    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)
    ax.legend()

    def animate(i):
        x, y, yaw = states_history[i][:3]

        robot_artist.set_data([x], [y])

        robot.generate_each_wheel_and_draw(ax, [x], [y], [yaw])
        
        optimal_traj_artist.set_data(states_history[:i+1, 0], states_history[:i+1, 1])
        predicted_traj_artist.set_data(optimal_trajs[i][:, 0], optimal_trajs[i][:, 1])

        rollout_lines = [traj[:, :2] for traj in sampled_trajs[i]]
        rollout_collection.set_segments(rollout_lines)

        safety_circle.center = (x, y)

        ax.set_xlim(x - 5, x + 5)
        ax.set_ylim(y - 5, y + 5)

        return robot_artist, optimal_traj_artist, predicted_traj_artist, rollout_collection

    ani = FuncAnimation(fig, animate, frames=len(states_history)-1, interval=50, blit=True, repeat=False)
    ani.save("mppi_differential_drive_with_trajectories.mp4", writer='ffmpeg', fps=30)
    plt.show()

def plot_control_inputs(actions_history, delta_t):
    time = np.arange(len(actions_history)) * delta_t
    v = actions_history[:, 0]
    omega = actions_history[:, 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(time, v, 'b-')
    ax1.set_ylabel('Linear Velocity (v)')
    ax1.set_title('Control Inputs over Time')
    ax1.grid(True)

    ax2.plot(time, omega, 'r-')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (ω)')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('mppi_control_inputs.png')
    plt.close()


def plot_control_inputs_comparison(non_filtered_actions, filtered_actions, delta_t):
    time = np.arange(len(filtered_actions)) * delta_t
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot linear velocity
    ax1.plot(time, non_filtered_actions[:, 0], 'b-', alpha=0.5, label='Non-filtered v')
    ax1.plot(time, filtered_actions[:, 0], 'r-', label='Filtered v')
    ax1.set_ylabel('Linear Velocity (v)')
    ax1.set_title('Control Inputs Comparison: Non-filtered vs Filtered')
    ax1.legend()
    ax1.grid(True)

    # Plot angular velocity
    ax2.plot(time, non_filtered_actions[:, 1], 'b-', alpha=0.5, label='Non-filtered ω')
    ax2.plot(time, filtered_actions[:, 1], 'r-', label='Filtered ω')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (ω)')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(time, non_filtered_actions[:, 0] - filtered_actions[:, 0], 'g-', label='Difference v')
    ax3.plot(time, non_filtered_actions[:, 1] - filtered_actions[:, 1], 'm-', label='Difference ω')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Difference')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('mppi_control_inputs_comparison.png')
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nx = 3  # state dimension
    nu = 2  # control dimension
    robot = DiffSimulation()
    num_samples = 1000
    horizon = 25
    noise_sigma = torch.tensor([[0.5, 0.0], [0.0, 0.3]], device=device)
    lambda_ = 1.0

    u_min = torch.tensor([-2.0, -2.0], device=device)
    u_max = torch.tensor([2.0, 2.0], device=device)

    mppi_ctrl = MPPIWrapper(
        dynamics,
        running_cost,
        nx,
        noise_sigma,
        num_samples=num_samples,
        horizon=horizon,
        lambda_=lambda_,
        u_min=u_min,
        u_max=u_max,
        device=device,
        # terminal_state_cost=terminal_state_cost
    )

    initial_state = torch.tensor([0.0, 0.0, 0.0], device=device)
    
    # Generate reference path
    x = np.linspace(0, 6, 100)
    y = np.linspace(0, 6, 100)
    yaw = np.arctan2(4, 5) * np.ones(100)
    ref_path = np.array([x, y, yaw]).T

    tSim = 200
    delta_t = 0.05

    print("Running simulation...")
    states_history, non_filtered_actions, filtered_actions, optimal_trajs, sampled_trajs = run_simulation(mppi_ctrl, initial_state, tSim, delta_t)

    print("Plotting control inputs...")
    plot_control_inputs(filtered_actions, delta_t)
    
    print("Plotting control inputs comparison...")
    plot_control_inputs_comparison(non_filtered_actions, filtered_actions, delta_t)
    
    print("Creating animation...")
    create_animation(states_history, optimal_trajs, sampled_trajs, ref_path, delta_t, robot)