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
    x, y, theta, v = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
    
    if actions.dim() == 1:
        actions = actions.unsqueeze(0)
    
    delta, a = actions[:, 0], actions[:, 1]

    # Race car kinematic model
    L = 0.3  # wheelbase
    x_next = x + v * torch.cos(theta) * dt
    y_next = y + v * torch.sin(theta) * dt
    theta_next = theta + v / L * torch.tan(delta) * dt
    v_next = v + a * dt

    next_states = torch.stack((x_next, y_next, theta_next, v_next), dim=1)
    return next_states


def running_cost(states, actions):
    desired_trajectory = torch.tensor([[6.0, 5.0, 0.0, 0.0]], dtype=torch.float32, device=states.device).expand(states.shape[0], -1)

    Q = torch.diag(torch.tensor([50.0, 35.0, 90.0, 0.1], dtype=torch.float32, device=states.device))
    R = torch.diag(torch.tensor([0.1, 0.1], dtype=torch.float32, device=states.device))

    state_cost = torch.einsum('bi,ij,bj->b', states - desired_trajectory, Q, states - desired_trajectory)
    control_cost = torch.einsum('bi,ij,bj->b', actions, R, actions)

    obstacle_positions = torch.tensor([[6.0, 5.0], [7.0, 3.0]], dtype=torch.float32, device=states.device)
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

def create_animation(states_history, optimal_trajs, sampled_trajs, ref_path, delta_t, robot):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('MPPI Race Car with Optimal and Sampled Trajectories')
    ax.axis('equal')
    ax.grid(True)

    robot_artist, = ax.plot([], [], 'ro', markersize=10, zorder=5, label='Robot')
    optimal_traj_artist, = ax.plot([], [], color='#990099', linestyle="solid", linewidth=2, zorder=4, label='Optimal Trajectory')
    predicted_traj_artist, = ax.plot([], [], color='green', linestyle="solid", linewidth=2, zorder=4, label='Predicted Trajectory')
    rollout_collection = LineCollection([], colors='gray', linewidths=0.5, alpha=0.3, zorder=3)
    ax.add_collection(rollout_collection)
    ref_traj_artist, = ax.plot(ref_path[:, 0], ref_path[:, 1], color='blue', linestyle="dashed", linewidth=1.0, zorder=2, label='Reference Trajectory')
    obstacles = ax.scatter([6.0, 7.0], [5.0, 3.0], color='red', s=100, zorder=6, label='Obstacles')

    safety_distance = 0.8
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
    ani.save("mppi_race_car_with_trajectories.mp4", writer='ffmpeg', fps=30)
    plt.show()


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

        print(f"Time: {i * delta_t:>2.2f}[s], x={state[0]:>+3.3f}[m], y={state[1]:>+3.3f}[m], yaw={state[2]:>+3.3f}[rad], v={state[3]:>+3.3f}[m/s]")
        print(f"Optimal Input: {action.cpu().numpy()}")

    return np.array(states_history), np.array(actions_history), np.array(optimal_trajs), np.array(sampled_trajs)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nx = 4  # state dimension for race car (x, y, theta, v)
    nu = 2  # control dimension (steering angle, acceleration)
    robot = DiffSimulation()  # Assuming you have a DiffSimulation model for visualization
    num_samples = 200
    horizon = 50
    noise_sigma = torch.tensor([[0.05, 0.0], [0.0, 0.05]], device=device)  # Adjust noise sigma for race car
    lambda_ = 1.0

    u_min = torch.tensor([-0.5, -1.0], device=device)  # Steering angle and acceleration limits
    u_max = torch.tensor([0.5, 1.0], device=device)

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
        device=device
    )

    initial_state = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
    
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
    create_animation(states_history, optimal_trajs, sampled_trajs, ref_path, delta_t, robot)
