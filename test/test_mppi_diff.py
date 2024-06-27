import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from pytorch_mppi import MPPI

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

def running_cost(states, actions, t):
    desired_trajectory = torch.tensor([[5.0, 4.0, 1.57]], dtype=torch.float32, device=states.device)
    
    Q = torch.diag(torch.tensor([10.0, 5.0, 9.0], dtype=torch.float32, device=states.device))
    R = torch.diag(torch.tensor([1.0, 10], dtype=torch.float32, device=states.device))

    state_cost = torch.einsum('bi,ij,bj->b', states - desired_trajectory, Q, states - desired_trajectory)
    control_cost = torch.einsum('bi,ij,bj->b', actions, R, actions)

    # Calculate current obstacle positions based on time
    obstacle_positions = get_obstacle_positions(t).to(states.device)

    safety_distance = 1.0  # Set the desired safety distance
    obstacle_cost = torch.zeros(states.shape[0], dtype=torch.float32, device=states.device)
    
    for obstacle in obstacle_positions:
        distance_to_obstacle = torch.norm(states[:, :2] - obstacle, dim=1)
        obstacle_cost += torch.exp(-(distance_to_obstacle - safety_distance))

    total_cost = state_cost + control_cost + obstacle_cost

    return total_cost

def get_obstacle_positions(t):
    initial_positions = torch.tensor([[5.0, 5.0], [7.0, 3.0]], dtype=torch.float32)
    velocities = torch.tensor([[0.0, -0.05], [0.0, 0.05]], dtype=torch.float32)  # Add velocities to obstacles
    current_positions = initial_positions + velocities * t
    return current_positions

class MPPIWrapper(MPPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.desired_trajectory = torch.tensor([[5.0, 4.0, 1.57]], dtype=torch.float32, device=self.d)

    def _compute_cost_total(self, costs):
        desired_trajectory = self.desired_trajectory.expand(costs.shape[0], -1)
        costs = costs + torch.einsum('bi,ij,bj->b', self.states - desired_trajectory, self.Q, self.states - desired_trajectory)
        return costs
    
    def get_trajectories(self, state):
        self.command(state)  # This updates internal states

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

    # @handle_batch_input(n=2)
    def _running_cost(self, state, u, t):
        if self.step_dependency:
            return self.running_cost(state, u, t)
        else:
            return self.running_cost(state, u)

def run_simulation(mppi_ctrl, initial_state, tSim, delta_t):
    state = initial_state
    states_history = [state.cpu().numpy()]
    actions_history = []
    optimal_trajs = []
    sampled_trajs = []

    for i in range(tSim):
        action = mppi_ctrl.command(state)
        optimal_traj, sampled_traj_list = mppi_ctrl.get_trajectories(state)
        
        state = mppi_ctrl._dynamics(state.unsqueeze(0), action.unsqueeze(0), 0).squeeze(0)
        
        states_history.append(state.cpu().numpy())
        actions_history.append(action.cpu().numpy())
        optimal_trajs.append(optimal_traj)
        sampled_trajs.append(sampled_traj_list)

        print(f"Time: {i * delta_t:>2.2f}[s], x={state[0]:>+3.3f}[m], y={state[1]:>+3.3f}[m], yaw={state[2]:>+3.3f}[rad]")
        print(f"Optimal Input: {action.cpu().numpy()}")

    return np.array(states_history), np.array(actions_history), np.array(optimal_trajs), np.array(sampled_trajs)

def create_animation(states_history, optimal_trajs, sampled_trajs, ref_path, delta_t):
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
    obstacles = ax.scatter([5.0, 7.0], [5.0, 3.0], color='red', s=100, zorder=6, label='Obstacles')

    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)
    ax.legend()

    def animate(i):
        x, y, yaw = states_history[i][:3]

        robot_artist.set_data([x], [y])

        
        optimal_traj_artist.set_data(states_history[:i+1, 0], states_history[:i+1, 1])
        predicted_traj_artist.set_data(optimal_trajs[i][:, 0], optimal_trajs[i][:, 1])

        rollout_lines = [traj[:, :2] for traj in sampled_trajs[i]]
        rollout_collection.set_segments(rollout_lines)

        ax.set_xlim(x - 5, x + 5)
        ax.set_ylim(y - 5, y + 5)

        return robot_artist, optimal_traj_artist, predicted_traj_artist, rollout_collection

    ani = FuncAnimation(fig, animate, frames=len(states_history), interval=50, blit=True, repeat=False)
    ani.save("mppi_differential_drive_with_trajectories.mp4", writer='ffmpeg', fps=30)
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nx = 3  # state dimension
    nu = 2  # control dimension

    num_samples = 200
    horizon = 20
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
        step_dependent_dynamics=True
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
    states_history, actions_history, optimal_trajs, sampled_trajs = run_simulation(mppi_ctrl, initial_state, tSim, delta_t)
    
    print("Creating animation...")
    create_animation(states_history, optimal_trajs, sampled_trajs, ref_path, delta_t)