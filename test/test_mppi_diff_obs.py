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
from collections import deque
from scipy.signal import savgol_filter

initial_obstacle_positions = torch.tensor([[5.0, 4.0], [3.5, 3.5]], dtype=torch.float32)
obstacle_velocities = 0.09*torch.tensor([[0.1, 0.1], [-0.1, 0.1]], dtype=torch.float32)

def get_obstacle_positions(t):
    if t is None:
        return initial_obstacle_positions
    return initial_obstacle_positions + obstacle_velocities * torch.tensor(t, dtype=torch.float32)

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy() if isinstance(tensor, torch.Tensor) else tensor

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

def running_cost(states, actions, t=None):
    desired_trajectory = torch.tensor([[6.0, 6.0, 1.57]], dtype=torch.float32, device=states.device).expand(states.shape[0], -1)

    Q = torch.diag(torch.tensor([30, 5, 9], dtype=torch.float32, device=states.device))
    R = torch.diag(torch.tensor([0.1, 0.1], dtype=torch.float32, device=states.device))

    state_cost = torch.einsum('bi,ij,bj->b', states - desired_trajectory, Q, states - desired_trajectory)
    control_cost = torch.einsum('bi,ij,bj->b', actions, R, actions)

    obstacle_positions = get_obstacle_positions(t).to(device=states.device)
    obstacle_cost = torch.zeros(states.shape[0], dtype=torch.float32, device=states.device)

    safety_distance = 2.0  # Increased from 1.5
    obstacle_weight = 100.0  # Increased from 50.0
    
    for obstacle in obstacle_positions:
        distance_to_obstacle = torch.norm(states[:, :2] - obstacle, dim=1)
        # Modified obstacle cost calculation
        obstacle_cost += torch.exp(safety_distance - distance_to_obstacle) * (distance_to_obstacle < safety_distance)

    total_cost = state_cost + control_cost + obstacle_weight * obstacle_cost

    return total_cost


class MPPIWrapper(MPPI):
    def __init__(self, *args, **kwargs):
            self.delta_t = kwargs.pop('delta_t', 0.05)  
            super().__init__(*args, **kwargs)
            self.u_min = kwargs.get('u_min', None)
            self.u_max = kwargs.get('u_max', None)
            self.action_history = deque(maxlen=10)
            self.alpha = 0.3
            self.beta  = 0.1
            self.k     = 2

    def command(self, state):
        action = super().command(state)
        
        # Apply Savitzky-Golay filter to smooth the control input
        self.filtered_action = self.smooth_control_input(action)
        
        # Apply bounds to filtered action
        if self.u_min is not None and self.u_max is not None:
            self.filtered_action = torch.clamp(self.filtered_action, self.u_min, self.u_max)
        
        return self.filtered_action
    
    def get_trajectories(self, state):
        self.command(state)  

        # Calculate optimal trajectory
        optimal_traj = torch.zeros((self.T, self.nx), device=self.d)
        x = state.unsqueeze(0)  # Add batch dimension
        for t in range(self.T):
            x = self._dynamics(x, self.u_scale * self.U[t].unsqueeze(0), t * self.delta_t)
            optimal_traj[t] = x.squeeze(0)

        # Get top 10% of sampled trajectories
        num_top_samples = max(10, self.K // 10)
        sorted_idx = torch.argsort(self.cost_total)[:num_top_samples]
        sampled_traj_list = torch.zeros((num_top_samples, self.T, self.nx), device=self.d)
        for i, k in enumerate(sorted_idx):
            x = state.unsqueeze(0)
            for t in range(self.T):
                x = self._dynamics(x, self.u_scale * self.perturbed_action[k, t].unsqueeze(0), t * self.delta_t)
                sampled_traj_list[i, t] = x.squeeze(0)

        return optimal_traj.cpu().numpy(), sampled_traj_list.cpu().numpy()

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total.repeat(self.M, 1)
        cost_var = torch.zeros_like(cost_total)

        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        state = state.repeat(self.M, 1, 1)

        states = []
        actions = []
        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t].repeat(self.M, 1, 1)
            state = self._dynamics(state, u, t * self.delta_t)
            c = self._running_cost(state, u, t * self.delta_t)
            cost_samples = cost_samples + c
            if self.M > 1:
                cost_var += c.var(dim=0) * (self.rollout_var_discount ** t)

            states.append(state)
            actions.append(u)

        actions = torch.stack(actions, dim=-2)
        states = torch.stack(states, dim=-2)

        self.sampled_actions = actions

        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples = cost_samples + c
        cost_total = cost_total + cost_samples.mean(dim=0)
        cost_total = cost_total + cost_var * self.rollout_var_cost
        return cost_total, states, actions

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

    # def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
    #     """Apply moving average filter to the data"""
    #     if xx.size == 0:
    #         return xx  # Return empty array if input is empty

    #     kernel_size = min(window_size, xx.shape[0])  # Ensure kernel size doesn't exceed array length
    #     kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    #     dim = xx.shape[1]
    #     xx_mean = np.zeros_like(xx)

    #     for d in range(dim):
    #         if xx.shape[0] == 1:
    #             xx_mean[:, d] = xx[:, d]  # If only one element, no filtering needed
    #         else:
    #             xx_padded = np.pad(xx[:, d], (kernel_size//2, kernel_size//2), mode='edge')
    #             xx_mean[:, d] = np.convolve(xx_padded, kernel, mode='valid')

    #     return xx_mean

    # def filter_action(self, action: torch.Tensor, window_size: int) -> torch.Tensor:
    #     # Convert to numpy, apply filter, and convert back to torch tensor
    #     action_np = action.cpu().numpy()
    #     filtered_action_np = self._moving_average_filter(action_np.reshape(1, -1), window_size)
    #     return torch.from_numpy(filtered_action_np.flatten()).to(action.device)

    def moving_average_filter(self, action: torch.Tensor, window_size: int) -> torch.Tensor:
        # Add current action to history
        self.action_history.append(action.cpu().numpy())
        
        # If we don't have enough history, return the current action
        if len(self.action_history) < window_size:
            return action
        
        # Calculate moving average
        recent_actions = list(self.action_history)[-window_size:]
        averaged_action = np.mean(recent_actions, axis=0)
        
        return torch.from_numpy(averaged_action).to(action.device)
    
    def robust_moving_average_filter(self, action: torch.Tensor) -> torch.Tensor:
        action_np = action.cpu().numpy()
        self.action_history.append(action_np)
        
        if len(self.action_history) < 2:
            return action

        # Initialize EWMA and EWMV if not already set
        if not hasattr(self, 'ewma'):
            self.ewma = np.mean(self.action_history, axis=0)
            self.ewmv = np.var(self.action_history, axis=0)

        # Update EWMA and EWMV
        error = action_np - self.ewma
        self.ewma = self.ewma + self.alpha * error
        self.ewmv = (1 - self.beta) * self.ewmv + self.beta * error**2

        # Detect and handle outliers
        std = np.sqrt(self.ewmv)
        lower_bound = self.ewma - self.k * std
        upper_bound = self.ewma + self.k * std

        filtered_action = np.clip(action_np, lower_bound, upper_bound)

        return torch.from_numpy(filtered_action).to(action.device)
    
    def smooth_control_input(self, action, window_size=51, polyorder=3):
        action_np = action.cpu().numpy()

        # Adjust window size if it's larger than the input data size
        window_size = min(window_size, len(action_np))

        # Make sure window size is odd
        if window_size % 2 == 0:
            window_size -= 1

        # Adjust polynomial order if it's greater than or equal to the window size
        polyorder = min(polyorder, window_size - 1)

        # Reshape action_np to 2D if it's 1D
        if action_np.ndim == 1:
            action_np = action_np.reshape(1, -1)

        # Apply Savitzky-Golay filter to each component of the action
        smoothed_action = np.apply_along_axis(savgol_filter, 0, action_np, window_size, polyorder)

        # Reshape smoothed_action back to 1D if the input was 1D
        if action.dim() == 1:
            smoothed_action = smoothed_action.squeeze()

        smoothed_action = torch.from_numpy(smoothed_action).to(action.device)
        return smoothed_action
    

def moving_average_filter(signal, window_size):
    """
    Apply a moving average filter to the input signal.
    
    :param signal: Input signal (torch.Tensor)
    :param window_size: Size of the moving average window
    :return: Filtered signal (torch.Tensor)
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size for centered average
    
    padding = (window_size - 1) // 2
    kernel = torch.ones(1, 1, window_size, device=signal.device) / window_size
    
    if signal.dim() == 1:
        signal = signal.unsqueeze(0).unsqueeze(0)
    elif signal.dim() == 2:
        signal = signal.unsqueeze(1)
    
    filtered = torch.nn.functional.conv1d(signal, kernel, padding=padding)
    return filtered.squeeze()

def run_simulation(mppi_ctrl, initial_state, goal_state, delta_t, max_steps=1000, goal_threshold=0.1):
    state = initial_state
    states_history = [state.cpu().numpy()]
    non_filtered_actions_history = []
    filtered_actions_history = []
    optimal_trajs = []
    sampled_trajs = []
    distances_to_goal = []

    for step in range(max_steps):
        current_time = step * delta_t
        
        # Get the control action from MPPI (this will also apply the filter)
        action = mppi_ctrl.command(state)
        
        # Get the optimal trajectory and sampled trajectories
        optimal_traj, sampled_traj_list = mppi_ctrl.get_trajectories(state)
        
        # Store non-filtered action
        non_filtered_actions_history.append(action.cpu().numpy())
        
        filtered_action = mppi_ctrl.moving_average_filter(action, window_size=51)

        state = mppi_ctrl._dynamics(state.unsqueeze(0), filtered_action.unsqueeze(0), current_time).squeeze(0)
        
        # Store histories
        states_history.append(state.cpu().numpy())
        optimal_trajs.append(optimal_traj)
        sampled_trajs.append(sampled_traj_list)

        # Calculate distance to goal
        distance_to_goal = torch.norm(state[:2] - goal_state[:2]).item()
        distances_to_goal.append(distance_to_goal)

        # Print current status
        print(f"Step: {step}, Time: {current_time:.2f}, Distance to goal: {distance_to_goal:.3f}")
        print(f"State: x={state[0]:>+3.3f}[m], y={state[1]:>+3.3f}[m], yaw={state[2]:>+3.3f}[rad]")
        print(f"Non-filtered Input: {action.cpu().numpy()}")
        print(f"Filtered Input: {mppi_ctrl.filtered_action.cpu().numpy()}")

        # Check if goal is reached
        if distance_to_goal < goal_threshold:
            print(f"Goal reached in {step + 1} steps!")
            break
    
    if step == max_steps - 1:
        print(f"Maximum number of steps ({max_steps}) reached without achieving the goal.")

    return (np.array(states_history), np.array(non_filtered_actions_history), 
            np.array(filtered_actions_history), np.array(optimal_trajs), 
            np.array(sampled_trajs), np.array(distances_to_goal))
def create_animation(states_history, optimal_trajs, sampled_trajs, ref_path, goal_state, robot, safe_distance=1.0):
    # Convert tensors to numpy arrays
    states_history = tensor_to_numpy(states_history)
    optimal_trajs = tensor_to_numpy(optimal_trajs)
    sampled_trajs = tensor_to_numpy(sampled_trajs)
    ref_path = tensor_to_numpy(ref_path)
    goal_state = tensor_to_numpy(goal_state)

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Main plot setup
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('MPPI Differential Drive with Dynamic Obstacles')
    ax.axis('equal')
    ax.grid(True)

    robot_artist, = ax.plot([], [], 'ro', markersize=10, zorder=5, label='Robot')
    goal_artist, = ax.plot(goal_state[0], goal_state[1], 'g*', markersize=15, zorder=6, label='Goal')
    optimal_traj_artist, = ax.plot([], [], color='#990099', linestyle="solid", linewidth=2, zorder=4, label='Optimal Trajectory')
    predicted_traj_artist, = ax.plot([], [], color='green', linestyle="solid", linewidth=2, zorder=4, label='Predicted Trajectory')
    rollout_collection = LineCollection([], colors='gray', linewidths=0.5, alpha=0.3, zorder=3)
    ax.add_collection(rollout_collection)
    ref_traj_artist, = ax.plot(ref_path[:, 0], ref_path[:, 1], color='blue', linestyle="dashed", linewidth=1.0, zorder=2, label='Reference Trajectory')

    # Initialize moving obstacles
    num_obstacles = len(initial_obstacle_positions)
    obstacle_artists = [ax.plot([], [], 'ro', markersize=8, zorder=6)[0] for _ in range(num_obstacles)]

    # Safety boundary circle
    safety_circle = plt.Circle((0, 0), safe_distance, edgecolor='k', linestyle='--', facecolor='none', zorder=10)
    ax.add_artist(safety_circle)

    # Popup text for robot state
    robot_popup = ax.text(0, 0, '', fontsize=10, ha='left', va='bottom',
                          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'),
                          zorder=11)

    # Popup texts for obstacles
    obstacle_popups = [ax.text(0, 0, '', fontsize=8, ha='left', va='bottom',
                               bbox=dict(facecolor='lightyellow', edgecolor='red', boxstyle='round'),
                               zorder=12) for _ in range(num_obstacles)]

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

        # Update safety circle position
        safety_circle.center = (x, y)

        # Update obstacle positions and popups
        current_obstacle_positions = tensor_to_numpy(get_obstacle_positions(i))
        for j, (obstacle, popup) in enumerate(zip(obstacle_artists, obstacle_popups)):
            obs_x, obs_y = current_obstacle_positions[j]
            obstacle.set_data([obs_x], [obs_y])
            
            # Update obstacle popup
            obstacle_text = f"Obstacle {j+1}: ({obs_x:.2f}, {obs_y:.2f})"
            popup.set_text(obstacle_text)
            popup.set_position((obs_x + 0.2, obs_y + 0.2))  # Offset slightly from obstacle position

        # Update robot state popup
        robot_popup_text = f"Robot: ({x:.2f}, {y:.2f})\nYaw: {yaw:.2f}"
        robot_popup.set_text(robot_popup_text)
        robot_popup.set_position((x + 0.5, y + 0.5))  # Offset slightly from robot position

        ax.set_xlim(x - 5, x + 5)
        ax.set_ylim(y - 5, y + 5)

        return (robot_artist, optimal_traj_artist, predicted_traj_artist, rollout_collection,
                safety_circle, robot_popup, *obstacle_artists, *obstacle_popups)

    ani = FuncAnimation(fig, animate, frames=len(states_history)-1, interval=50, blit=True, repeat=False)
    ani.save("mppi_differential_drive_with_dynamic_obstacles.mp4", writer='ffmpeg', fps=30)
    plt.show()

def plot_control_inputs(actions_history, delta_t):
    time = np.arange(len(actions_history)) * delta_t
    v = actions_history[:, 0]
    omega = actions_history[:, 1]

    # Apply Savitzky-Golay filter to smooth the control inputs
    window_size = 51  # Window size for the filter (odd number)
    polyorder = 3     # Polynomial order for the filter
    v_smooth = savgol_filter(v, window_size, polyorder)
    omega_smooth = savgol_filter(omega, window_size, polyorder)

    # Plot raw control inputs
    fig_raw, (ax_raw, ax2_raw) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot linear velocity (raw)
    ax_raw.plot(time, v, 'b-', label='Linear Velocity (v)')
    ax_raw.set_ylabel('Linear Velocity (v)')
    ax_raw.set_title('Raw Control Inputs over Time')
    ax_raw.legend()
    ax_raw.grid(True)

    # Plot angular velocity (raw)
    ax2_raw.plot(time, omega, 'r-', label='Angular Velocity (ω)')
    ax2_raw.set_xlabel('Time (s)')
    ax2_raw.set_ylabel('Angular Velocity (ω)')
    ax2_raw.legend()
    ax2_raw.grid(True)

    plt.tight_layout()
    plt.savefig('mppi_control_inputs_raw.png')
    plt.close()

    # Plot smoothed control inputs
    fig_smooth, (ax_smooth, ax2_smooth) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot linear velocity (smoothed)
    ax_smooth.plot(time, v_smooth, 'b-', label='Linear Velocity (v)')
    ax_smooth.set_ylabel('Linear Velocity (v)')
    ax_smooth.set_title('Control Inputs over Time')
    ax_smooth.legend()
    ax_smooth.grid(True)

    # Plot angular velocity (smoothed)
    ax2_smooth.plot(time, omega_smooth, 'r-', label='Angular Velocity (ω)')
    ax2_smooth.set_xlabel('Time (s)')
    ax2_smooth.set_ylabel('Angular Velocity (ω)')
    ax2_smooth.legend()
    ax2_smooth.grid(True)

    plt.tight_layout()
    plt.savefig('mppi_control_inputs_smooth.png')
    plt.close()


def plot_control_inputs_comparison(non_filtered_actions, filtered_actions, delta_t):
    if non_filtered_actions.shape[0] != filtered_actions.shape[0]:
        print("Error: non_filtered_actions and filtered_actions must have the same first dimension.")
        return

    time = np.arange(len(filtered_actions)) * delta_t

    time = np.arange(len(filtered_actions)) * delta_t
    
    fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot linear velocity
    ax.plot(time, non_filtered_actions[:, 0], 'b-', alpha=0.5, label='Non-filtered v')
    ax.plot(time, filtered_actions[:, 0], 'r-', label='Filtered v')
    ax.set_ylabel('Linear Velocity (v)')
    ax.set_title('Control Inputs Comparison: Non-filtered vs Filtered')
    ax.legend()
    ax.grid(True)

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

def plot_state_error(states_history, goal_state):
    # Extract the final state from the trajectory
    final_state = states_history[-1]

    # Calculate the state errors
    state_errors = np.abs(final_state - goal_state[:3])

    # Create labels for the states
    state_labels = ['X', 'Y', 'Yaw']

    # Set custom colors for the bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Create a figure and axis with larger size
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the state errors as a bar plot with custom colors
    bars = ax.bar(state_labels, state_errors, color=colors)

    # Add labels and title with larger font sizes
    ax.set_xlabel('State', fontsize=16)
    ax.set_ylabel('Error', fontsize=16)
    ax.set_title('State Errors', fontsize=20)

    # Increase the font size of the tick labels
    ax.tick_params(axis='both', labelsize=14)

    # Add value labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12)

    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust the layout and display the plot
    fig.tight_layout()

    # Save the plot with higher resolution
    plt.savefig('mppi_state_errors_mppi.png', dpi=600, bbox_inches='tight')
    plt.close()

    print("State error plot saved as 'mppi_state_errors_mppi.png'")

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

    goal_state = torch.tensor([6.0, 6.0, 1.57], device=device) 
    max_steps = 250  
    goal_threshold = 0.05

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
        step_dependent_dynamics=True,
        delta_t=0.05
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
    states_history, non_filtered_actions, filtered_actions, optimal_trajs, sampled_trajs, distances_to_goal = run_simulation(
        mppi_ctrl, initial_state, goal_state, delta_t, max_steps=max_steps, goal_threshold=goal_threshold
    )

    print("Plotting control inputs...")
    plot_control_inputs(non_filtered_actions, delta_t)
    
    # print("Plotting control inputs comparison...")
    # plot_control_inputs_comparison(non_filtered_actions, filtered_actions, delta_t)

    print("Plotting state error...")
    plot_state_error(states_history, goal_state.cpu().numpy())
    
    print("Creating animation...")
    create_animation(states_history, optimal_trajs, sampled_trajs, ref_path, goal_state, robot, safe_distance=0.8)