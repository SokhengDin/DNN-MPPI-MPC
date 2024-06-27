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
    dt = 0.02
    m = 2.0  
    I = 0.05  
    r = 0.1 
    L = 0.4 
    
    x, y, theta, v, omega = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4]
    
    if actions.dim() == 1:
        actions = actions.unsqueeze(0)
    
    F_fr, F_fl, F_rr, F_rl = actions[:, 0], actions[:, 1], actions[:, 2], actions[:, 3]

    dx = v * torch.cos(theta)
    dy = v * torch.sin(theta)
    dtheta = omega
    dv = (r / (4 * m)) * (F_fr + F_fl + F_rr + F_rl) - 0.1 * v  # Linear damping included
    domega = (r / (L * I)) * ((F_fr + F_rr) - (F_fl + F_rl)) / 2 - 0.1 * omega  # Angular damping included

    x_next = x + dx * dt
    y_next = y + dy * dt
    theta_next = theta + dtheta * dt
    v_next = v + dv * dt
    omega_next = omega + domega * dt

    next_states = torch.stack((x_next, y_next, theta_next, v_next, omega_next), dim=1)
    return next_states



def running_cost(states, actions):
    desired_trajectory = torch.tensor([[6.0, 6.0, 1.57, 2.0, 0.0]], dtype=torch.float32, device=states.device).expand(states.shape[0], -1)

    Q = torch.diag(torch.tensor([200, 200, 20, 10, 20], dtype=torch.float32, device=states.device))
    R = torch.diag(torch.tensor([0.001, 0.001, 0.001, 0.001], dtype=torch.float32, device=states.device))

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

            print(f"Coefficients shape: {coefficients.shape}")

            # Ensure coefficients is 1D
            if coefficients.dim() > 1:
                coefficients = coefficients.squeeze()

            num_samples = x_channel.shape[0]
            filtered_channel = torch.zeros_like(x_channel)

            for i in range(num_samples):
                start = max(0, i - half_window)
                end = min(num_samples, i + half_window + 1)
                window = x_channel[start:end]
                
                print(f"Window shape: {window.shape}")
                
                window_size = window.shape[0]
                
                # Adjust coefficients to match the window size
                if window_size < coefficients.shape[0]:
                    center = coefficients.shape[0] // 2
                    start_coeff = center - window_size // 2
                    end_coeff = start_coeff + window_size
                    coeffs = coefficients[start_coeff:end_coeff]
                else:
                    coeffs = coefficients[:window_size]

                print(f"Coeffs shape: {coeffs.shape}")

                # Ensure both window and coeffs are 1D and non-empty
                if window.numel() == 0 or coeffs.numel() == 0:
                    print(f"Warning: Empty tensor encountered. Window size: {window.numel()}, Coeffs size: {coeffs.numel()}")
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
        """Apply moving average filter to the data"""
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

def create_animation(states_history, optimal_trajs, sampled_trajs, ref_path, delta_t, robot):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('MPPI 4-Wheeled Robot with Optimal and Sampled Trajectories')
    ax.axis('equal')
    ax.grid(True)

    robot_artist, = ax.plot([], [], 'ro', markersize=10, zorder=5, label='Robot')
    optimal_traj_artist, = ax.plot([], [], color='#990099', linestyle="solid", linewidth=2, zorder=4, label='Optimal Trajectory')
    predicted_traj_artist, = ax.plot([], [], color='green', linestyle="solid", linewidth=2, zorder=4, label='Predicted Trajectory')
    rollout_collection = LineCollection([], colors='gray', linewidths=0.5, alpha=0.3, zorder=3)
    ax.add_collection(rollout_collection)
    ref_traj_artist, = ax.plot(ref_path[:, 0], ref_path[:, 1], color='blue', linestyle="dashed", linewidth=1.0, zorder=2, label='Reference Trajectory')
    obstacles = ax.scatter([3.0], [2.0], color='red', s=100, zorder=6, label='Obstacles')

    safety_distance = 1.0
    safety_circle = plt.Circle((0, 0), safety_distance, color='r', fill=False, linestyle='--', linewidth=1.0, zorder=4, label='Safety Distance')
    ax.add_patch(safety_circle)

    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)
    ax.legend()

    def animate(i):
        x, y, yaw = states_history[i][:3]

        robot_artist.set_data([x], [y])

        # robot.generate_each_wheel_and_draw(ax, [x], [y], [yaw])
        
        optimal_traj_artist.set_data(states_history[:i+1, 0], states_history[:i+1, 1])
        predicted_traj_artist.set_data(optimal_trajs[i][:, 0], optimal_trajs[i][:, 1])

        rollout_lines = [traj[:, :2] for traj in sampled_trajs[i]]
        rollout_collection.set_segments(rollout_lines)

        safety_circle.center = (x, y)

        ax.set_xlim(x - 5, x + 5)
        ax.set_ylim(y - 5, y + 5)

        return robot_artist, optimal_traj_artist, predicted_traj_artist, rollout_collection

    ani = FuncAnimation(fig, animate, frames=len(states_history)-1, interval=50, blit=True, repeat=False)
    ani.save("mppi_4wheel_robot_with_trajectories.mp4", writer='ffmpeg', fps=30)
    plt.show()

def run_simulation(mppi_ctrl, initial_state, tSim, delta_t):
    state = initial_state
    states_history = [state.cpu().numpy()]
    actions_history = []
    optimal_trajs = []
    sampled_trajs = []

    window_size = 5  # Adjust the window size as needed

    for i in range(tSim):
        action = mppi_ctrl.command(state)
        optimal_traj, sampled_traj_list = mppi_ctrl.get_trajectories(state)
        
        print(f"Action shape: {action.shape}")
        
        # Apply moving average filter to the optimal control input
        if action.dim() == 1:
            filtered_action = mppi_ctrl.moving_average_filter(action.unsqueeze(0), window_size).squeeze(0)
        else:
            filtered_action = mppi_ctrl.moving_average_filter(action, window_size)
        
        print(f"Filtered action shape: {filtered_action.shape}")
        
        state = mppi_ctrl._dynamics(state.unsqueeze(0), filtered_action.unsqueeze(0), 0).squeeze(0)
        
        states_history.append(state.cpu().numpy())
        actions_history.append(filtered_action.cpu().numpy())
        optimal_trajs.append(optimal_traj)
        sampled_trajs.append(sampled_traj_list)

        print(f"Time: {i * delta_t:>2.2f}[s], x={state[0]:>+3.3f}[m], y={state[1]:>+3.3f}[m], yaw={state[2]:>+3.3f}[rad], v={state[3]:>+3.3f}[m/s], omega={state[4]:>+3.3f}[rad/s]")
        print(f"Optimal Input (Filtered): {filtered_action.cpu().numpy()}")

    return np.array(states_history), np.array(actions_history), np.array(optimal_trajs), np.array(sampled_trajs)

def plot_control_inputs(actions_history, delta_t):
    time = np.arange(len(actions_history)) * delta_t
    force_fr = actions_history[:, 0]
    force_fl = actions_history[:, 1]
    force_rr = actions_history[:, 2]
    force_rl = actions_history[:, 3]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
    
    ax1.plot(time, force_fr, 'b-', label='Front Right')
    ax1.set_ylabel('Force_fr (N)')
    ax1.set_title('Control Inputs (Forces) over Time')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time, force_fl, 'r-', label='Front Left')
    ax2.set_ylabel('Force_fl (N)')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(time, force_rr, 'g-', label='Rear Right')
    ax3.set_ylabel('Force_rr (N)')
    ax3.legend()
    ax3.grid(True)

    ax4.plot(time, force_rl, 'm-', label='Rear Left')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Force_rl (N)')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('mppi_control_inputs_forces.png')
    plt.close()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nx = 5  
    nu = 4  
    robot = DiffSimulation()
    num_samples = 500
    horizon = 30

    noise_sigma = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0], device=device))

    lambda_ = 1.0

    u_min = torch.tensor([-100.0, -100.0, -100.0, -100.0], device=device)
    u_max = torch.tensor([100.0, 100.0, 100.0, 100.0], device=device)

    U_init = torch.rand((horizon, nu), device=device) * (u_max - u_min) + u_min

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
        u_init=torch.zeros(nu, device=device),  
        U_init=U_init, 
        step_dependent_dynamics=False,
        rollout_samples=1,
        rollout_var_cost=0.0,
        rollout_var_discount=0.95,
        sample_null_action=True,
        noise_abs_cost=False,
        device=device,
    )


    initial_state = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    
    # Generate reference path
    x = np.linspace(0, 6, 100)
    y = np.linspace(0, 6, 100)
    yaw = np.arctan2(4, 5) * np.ones(100)
    v = np.zeros(100)
    omega = np.zeros(100)
    ref_path = np.array([x, y, yaw, v, omega]).T

    tSim = 200
    delta_t = 0.02

    print("Running simulation...")
    states_history, actions_history, optimal_trajs, sampled_trajs = run_simulation(mppi_ctrl, initial_state, tSim, delta_t)
    
    print("Plotting control inputs...")
    plot_control_inputs(actions_history, delta_t)
    
    print("Creating animation...")
    create_animation(states_history, optimal_trajs, sampled_trajs, ref_path, delta_t, robot)