import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from typing import Tuple
from path_generator.cubic_spline_planner import calc_spline_course
from models.differentialSim import DifferentialSimulation
from torch.distributions import MultivariateNormal

class DifferentialDrive:

    def __init__(self, init_x):
        # Initial state
        self.x0     = init_x[0]
        self.y0     = init_x[1]
        self.yaw0   = init_x[2]

    def forward_kinematic(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward kinematics of the differential drive robot
        x = [x, y, yaw]
        u = [v, w]
        """
        v = u[0]
        w = u[1]

        x_dot = v * torch.cos(x[2])
        y_dot = v * torch.sin(x[2])
        yaw_dot = w

        return x_dot, y_dot, yaw_dot
    
    def get_state(self) -> torch.Tensor:
        return torch.tensor([self.x0, self.y0, self.yaw0])
    
    def update_state(self, sample_time: float, u: torch.Tensor) -> None:
        x_dot, y_dot, yaw_dot = self.forward_kinematic(self.get_state(), u)
        self.x0 += x_dot * sample_time
        self.y0 += y_dot * sample_time
        self.yaw0 += yaw_dot * sample_time

        return torch.Tensor([self.x0, self.y0, self.yaw0])
    

class MPPIAlgorithms:

    def __init__(self,
            delta_t: torch.FloatTensor,
            ref_path: torch.Tensor,
            max_speed: torch.FloatTensor,
            max_omega: torch.FloatTensor,
            num_samples_K: torch.IntTensor,
            num_horizons_T: torch.IntTensor,
            param_exploration: torch.FloatTensor,
            param_lambda: torch.FloatTensor,
            param_alpha: torch.FloatTensor,
            sigma: torch.Tensor,
            stage_cost_weight: torch.Tensor,
            terminal_cost_weight: torch.Tensor,
            visualize_optimal_traj: torch.BoolTensor = True,
            visualize_sampled_traj: torch.BoolTensor = True,
        ) -> None:
        """Initilize the MPPI algorithm"""
        ## mppi_parameters
        self.delta_t = delta_t
        self.ref_path = ref_path
        self.max_speed = max_speed
        self.max_omega = max_omega
        self.dim_x = 3
        self.dim_u = 2
        self.T = num_horizons_T
        self.K = num_samples_K
        self.param_exploration = param_exploration
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))
        self.Sigma = sigma
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualize_sampled_traj = visualize_sampled_traj

        ## mppi variables
        self.u_prev = torch.zeros((self.T, self.dim_u))

        # ref path info
        self.prev_way_point_idx = 0

    def _calc_input_control(self, observed_x: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Calculate control input using MPPI"""
        # load the previous control input sequence
        u = self.u_prev.to(observed_x.device)

        # set the initial value x from observation
        x0 = observed_x

        # get the waypoint closed to current vehicle position
        self._get_nearest_waypoint(x0[0], x0[1], update_prev_idx=True)
        if self.prev_way_point_idx >= self.ref_path.shape[0] - 1:
            print("[ERROR] Reached the end of the reference path.")
            self.prev_way_point_idx = self.ref_path.shape[0]-1

        # prepare buffer
        S = torch.zeros((self.K), device=observed_x.device)

        # sample noise
        epsilon = self._calc_epsilon(
            self.Sigma.to(observed_x.device), self.K, self.T, self.dim_u
        )

        # prepare buffer of sampled control input sequence
        v = torch.zeros((self.K, self.T, self.dim_u), device=observed_x.device)

        for k in range(self.K):
            # set initial
            x = x0
            for t in range(1, self.T+1):
                # get control input with noise
                if k < (1.0 - self.param_exploration) * self.K:
                    v[k, t-1] = u[t-1] + epsilon[k, t-1]
                else:
                    v[k, t-1] = epsilon[k, t-1]

                # update state
                x = self._state_transition(x, v[k, t-1])

                # calculate stage cost
                u_t_1 = u[t-1].unsqueeze(0).to(observed_x.device)
                v_t_1 = v[k, t-1].unsqueeze(-1).to(observed_x.device)
                sigma_inv = torch.linalg.inv(self.Sigma.to(observed_x.device))
                S[k] = self._compute_cost(x) + self.param_gamma * torch.matmul(torch.matmul(u_t_1, sigma_inv), v_t_1).squeeze()

            # Add terminal cost
            S[k] += self._terminal_cost(x)
        
        # Compute information theoretic weight for each sample
        w = self._compute_weight(S)

        # Calculate w_K * epsilon_k
        w_epsilon = torch.zeros((self.T, self.dim_u), device=observed_x.device)
        for t in range(self.T):
            for k in range(self.K):
                w_epsilon[t] += w[k] * epsilon[k, t]

        # Apply moving average filter for smoothing input sequencce
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=10)

        # update control input sequence
        u += w_epsilon

        # calculate optimal trajectory
        optimal_traj = torch.zeros((self.T, self.dim_x), device=observed_x.device)
        if self.visualize_sampled_traj:
            x = x0
            for t in range(self.T):
                x = self._state_transition(x, self._clamp(u[t-1]))
                optimal_traj[t] = x

        # calculate sampled trajectories
        sampled_traj_list = torch.zeros((self.K, self.T, self.dim_x), device=observed_x.device)
        sorted_idx = torch.argsort(S)
        if self.visualize_sampled_traj:
            for k in sorted_idx:
                x = x0
                for t in range(self.T):
                    x = self._state_transition(x, self._clamp(v[k, t-1]))
                    sampled_traj_list[k, t] = x

        # Update previous control input
        self.u_prev[:-1] = u[1:].clone()
        self.u_prev[-1] = u[-1].clone()

        return u[0], u, optimal_traj, sampled_traj_list


    def _compute_weight(self, S: torch.Tensor) -> torch.Tensor:
        """Compute weights for each sample"""
        w = torch.zeros((self.K), device=S.device)
        # Calculate rho
        rho = S.min()
        # Calculate eta nominal
        eta = torch.tensor(0.0, device=S.device)
        for k in range(self.K):
            eta += torch.exp(-self.param_lambda.to(S.device) * (S[k] - rho))
        # Calculate weights
        for k in range(self.K):
            w[k] = torch.exp(-self.param_lambda.to(S.device) * (S[k] - rho)) / eta

        return w


    def _state_transition(self, x_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        """Calculate State Transition using Euler Lagrange"""
        x = x_t[0]
        y = x_t[1]
        yaw = x_t[2]

        speed = v_t[0]
        omega = v_t[1]

        # update state
        dt = self.delta_t.to(x_t.device)

        x += speed * torch.cos(yaw) * dt
        y += speed * torch.sin(yaw) * dt
        yaw += omega * dt

        return torch.tensor([x, y, yaw], device=x_t.device)


    def _compute_cost(self, x_t: torch.Tensor) -> torch.Tensor:
        """Calculate stage cost"""
        x, y, yaw = x_t

        # Calculate stage cost
        _, ref_x, ref_y, ref_yaw = self._get_nearest_waypoint(x, y, update_prev_idx=True)

        stage_cost = self.stage_cost_weight[0].to(x_t.device) * (x - ref_x)**2 + \
                    self.stage_cost_weight[1].to(x_t.device) * (y - ref_y)**2 + \
                    self.stage_cost_weight[2].to(x_t.device) * (yaw - ref_yaw)**2

        return stage_cost


    def _terminal_cost(self, x_t: torch.Tensor) -> torch.Tensor:
        """Calculate terminal cost"""
        x, y, yaw = x_t
        yaw = ((yaw + 2.0 * torch.pi) % (2.0 * torch.pi))

        # Calculate terminal cost
        _, ref_x, ref_y, ref_yaw = self._get_nearest_waypoint(x, y, update_prev_idx=True)
        terminal_cost = self.terminal_cost_weight[0].to(x_t.device) * (x - ref_x)**2 + \
                        self.terminal_cost_weight[1].to(x_t.device) * (y - ref_y)**2 + \
                        self.terminal_cost_weight[2].to(x_t.device) * (yaw - ref_yaw)**2

        return terminal_cost


    def _cross_entropy(self) -> None:
        """Cross entropy method"""
        pass


    def _savitky_golay_filter(self, data: torch.Tensor, window_size: torch.IntTensor, order: torch.IntTensor) -> torch.Tensor:
        """Apply Savitky-Golay filter to the data"""
        pass


    def _moving_average_filter(self, xx: torch.Tensor, window_size: int) -> torch.Tensor:
        """Apply moving average filter to the data"""
        kernel_size = window_size
        kernel = torch.ones(kernel_size, device=xx.device) / kernel_size
        dim = xx.shape[1]
        xx_mean = torch.zeros(xx.shape, device=xx.device)

        for d in range(dim):
            xx_padded = torch.cat([xx[:kernel_size//2, d], xx[:, d], xx[-kernel_size//2:, d]], dim=0)
            xx_mean[:, d] = torch.conv1d(xx_padded.view(1, 1, -1), kernel.view(1, 1, -1), padding=kernel_size//2).view(-1)[:xx.shape[0]]

        return xx_mean


    def _calc_epsilon(self, sigma: torch.Tensor, size_sample: torch.IntTensor, size_time_step: torch.IntTensor, size_dim_u: torch.IntTensor) -> torch.Tensor:
        """Sample epsilon"""
        # check if sigma row size == sigma col size == size_dim_u and size_dim_u > 0
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != self.dim_u or self.dim_u <= 0:
            print("[ERROR] Sigma matrix is not valid.")
            raise ValueError

        # sample epsilon
        mu = torch.zeros((size_dim_u), device=sigma.device)
        epsilon = MultivariateNormal(mu, sigma).sample((size_sample, size_time_step))
        return epsilon


    def _clamp(self, v: torch.Tensor) -> torch.Tensor:
        """Clamping the input control"""
        v[0] = torch.clamp(v[0], -self.max_speed.to(v.device), self.max_speed.to(v.device))
        v[1] = torch.clamp(v[1], -self.max_omega.to(v.device), self.max_omega.to(v.device))

        return v


    def _get_nearest_waypoint(self, x: torch.Tensor, y: torch.Tensor, update_prev_idx: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """search the closest waypoint to the vehicle on the reference path"""

        SEARCH_IDX_LEN = 20
        prev_idx = self.prev_way_point_idx
        dx = [x - ref_x.to(x.device) for ref_x in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 0]]
        dy = [y - ref_y.to(x.device) for ref_y in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 1]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        min_d = min(d)
        nearest_idx = d.index(min_d) + prev_idx

        # Get reference values of the nearest waypoint
        ref_x = self.ref_path[nearest_idx, 0].to(x.device)
        ref_y = self.ref_path[nearest_idx, 1].to(x.device)
        ref_yaw = self.ref_path[nearest_idx, 2].to(x.device)

        if update_prev_idx:
            self.prev_way_point_idx = nearest_idx

        return nearest_idx, ref_x, ref_y, ref_yaw
    
def plot_trajectories(diff_frame, diff_drive, mppi, delta_t, ref_path, tSim):
    fig, ax = plt.subplots()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('MPPI Differential Drive')
    ax.axis('equal')
    ax.grid(True)

    # Initialize empty lists for artists
    robot_artists = []
    optimal_traj_artist, = ax.plot([], [], color='#990099', linestyle="solid", linewidth=1.5, zorder=5)
    sampled_traj_artists = []
    ref_traj_artist = None

    for i in range(tSim):
        current_state = diff_drive.get_state()
        optimal_input, optimal_input_sequence, optimal_traj, sampled_traj_list = mppi._calc_input_control(current_state)

        x, y, yaw = current_state

        # Draw the robot
        # robot_artists = diff_frame.generate_each_wheel_and_draw(x, y, yaw)
        # for artist in robot_artists:
        #     ax.add_artist(artist)

        # Update the optimal trajectory artist
        if optimal_traj.any():
            optimal_traj_artist.set_data(optimal_traj[:, 0], optimal_traj[:, 1])
        else:
            optimal_traj_artist.set_data([], [])

        # Draw the sampled trajectories from mppi
        if sampled_traj_list.any():
            min_alpha_value = 0.25
            max_alpha_value = 0.35
            for idx, sampled_traj in enumerate(sampled_traj_list):
                # Draw darker for better samples
                alpha_value = (1.0 - (idx + 1) / len(sampled_traj_list)) * (max_alpha_value - min_alpha_value) + min_alpha_value

                sampled_traj_artist, = ax.plot(sampled_traj[:, 0], sampled_traj[:, 1], color='gray', linestyle="solid", linewidth=0.2, zorder=4, alpha=alpha_value)
                sampled_traj_artists.append(sampled_traj_artist)

        # Draw the reference trajectory
        ref_traj_x_offset = ref_path[:, 0]
        ref_traj_y_offset = ref_path[:, 1]
        ref_traj_artist, = ax.plot(ref_traj_x_offset, ref_traj_y_offset, color='blue', linestyle="dashed", linewidth=1.0, zorder=3, label='Reference Trajectory')

        ax.set_xlim(x - 5, x + 5)
        ax.set_ylim(y - 5, y + 5)
        ax.legend()

        print(f"Optimal Input: {optimal_input}")
        print(f"Time: {i * delta_t:>2.2f}[s], x={x:>+3.3f}[m], y={y:>+3.3f}[m], yaw={yaw:>+3.3f}[rad]")

        diff_drive.update_state(delta_t, optimal_input)

    plt.savefig("mppi_differential_drive.png")
    plt.close(fig)

def generate_circle_points(center_x, center_y, radius, num_points):
    device = center_x.device  # Get the device of the input tensors
    angles = torch.linspace(0, 2 * torch.pi, num_points, device=device)
    x = center_x + radius * torch.cos(angles)
    y = center_y + radius * torch.sin(angles)
    yaw = angles + torch.pi / 2

    return x, y, yaw


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the differential drive robot
    init_x = torch.tensor([0.0, 0.0, 0.0]).to(device)
    tSim = 1000
    diff_drive = DifferentialDrive(init_x)

    # Initialize the MPPI algorithm
    delta_t = torch.tensor(0.1).to(device)
    max_speed = torch.tensor(3.0).to(device)
    max_omega = torch.tensor(3.14).to(device)
    num_samples_K = torch.tensor(100).to(device)
    num_horizons_T = torch.tensor(20).to(device)
    param_exploration = torch.tensor(0.01).to(device)
    param_lambda = torch.tensor(1.0).to(device)
    param_alpha = torch.tensor(0.2).to(device)
    sigma = torch.tensor([[0.1, 0.0], [0.0, 0.1]]).to(device)
    stage_cost_weight = torch.tensor([5.0, 5.0, 10.0]).to(device)
    terminal_cost_weight = torch.tensor([5.0, 5.0, 10.0]).to(device)

    # Generate the reference path
    center_x = torch.tensor(0.0).to(device)
    center_y = torch.tensor(0.0).to(device)
    radius = torch.tensor(5.0).to(device)
    num_points = torch.tensor(100).to(device)
    cx, cy, cyaw = generate_circle_points(center_x, center_y, radius, num_points)
    ref_path = torch.stack([cx, cy, cyaw], dim=1).to(device)

    # Prepare simulation
    diff_frame = DifferentialSimulation()

    print("[INFO] Reference Path Generated.")

    mppi = MPPIAlgorithms(
        delta_t=delta_t,
        ref_path=ref_path,
        max_speed=max_speed,
        max_omega=max_omega,
        num_samples_K=num_samples_K,
        num_horizons_T=num_horizons_T,
        param_exploration=param_exploration,
        param_lambda=param_lambda,
        param_alpha=param_alpha,
        sigma=sigma,
        stage_cost_weight=stage_cost_weight,
        terminal_cost_weight=terminal_cost_weight,
        visualize_optimal_traj=True,
        visualize_sampled_traj=True
    )

    # Run the MPPI algorithms
    plot_trajectories(diff_frame, diff_drive, mppi, delta_t.item(), ref_path.cpu(), tSim)