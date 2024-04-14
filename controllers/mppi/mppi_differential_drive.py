import numpy as np
import math
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from typing import Tuple
from path_generator.cubic_spline_planner import CubicSpline2D, calc_spline_course
from path_generator.bezierPath import calc_4points_bezier_path, calc_bezier_path
from models.differetialSim import DifferentialSimulation
from math import pi, cos, sin

class DifferentialDrive:
    def __init__(self, init_x):
        # Initialize Params
        self.x0 = init_x[0]
        self.y0 = init_x[1]
        self.yaw0 = init_x[2]

    def forward_kinematic(self, x: np.array, u: np.array):
        # Forward kinematic of the differential drive
        vx = u[0]*np.cos(x[2])
        vy = u[0]*np.sin(x[2])
        vyaw = u[1]

        return vx, vy, vyaw
    
    def get_state(self) -> np.ndarray:
        return np.array([self.x0, self.y0, self.yaw0])
    
    def update_state(self, sampling_rate: float, x: np.array, u: np.array) -> np.array:
        # Integrate to update the state
        vx, vy, vyaw = self.forward_kinematic(x, u)
        self.x0 += vx * sampling_rate
        self.y0 += vy * sampling_rate
        self.yaw0 += vyaw * sampling_rate

        return np.array([self.x0, self.y0, self.yaw0])
    
class MPPIAlgorithms:
    
    def __init__(
            self,
            delta_t: float,
            ref_path: np.ndarray,
            max_speed: float,
            max_omega: float,
            num_samples_K: int,
            num_horizons_T: int,
            param_exploration: float,
            param_lambda: float,
            param_alpha: float,
            sigma: np.ndarray,
            stage_cost_weight: np.ndarray,
            terminal_cost_weight: np.ndarray,
            visualize_optimal_traj = True,
            visualze_sampled_trajs = True
        ) -> None:
        """Initialize the MPPI algorithm"""
        ## mppi parameters
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
        self.visualze_sampled_trajs = visualze_sampled_trajs

        # mppi variables 
        self.u_prev = np.zeros((self.T, self.dim_u))

        # ref path info
        self.prev_way_point_idx = 0

    def _calc_input_control(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Calculate control input using MPPI"""
        # load the previous control input sequence
        u = self.u_prev

        # set the initial value x from obeservation
        x0 = observed_x

        # get the waypoint closed to current vehicle position
        self._get_nearest_waypoint(x0[0], x0[1], update_prev_idx=True)
        if self.prev_way_point_idx >= self.ref_path.shape[0]-1:
            print("[ERROR] Reached the end of the reference path.")
            self.prev_way_point_idx = self.ref_path.shape[0]-1
            # raise IndexError
        
        # prepare buffer
        S = np.zeros((self.K))

        # sample noise 
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)

        # prepare buffer of sampled control input sequence
        v = np.zeros((self.K, self.T, self.dim_u))

        for k in range(self.K):
            # set intial 
            x = x0
            for t in range(1, self.T+1):
                # get control input with noise
                if k < (1.0-self.param_exploration)*self.K:
                    v[k, t-1] = u[t-1] + epsilon[k, t-1]
                else:
                    v[k, t-1] = epsilon[k, t-1]
                # update state
                x = self._state_transition(x, self._g(v[k, t-1]))

                # add stage cost
                S[k] = self._compute_cost(x) + self.param_gamma * u[t-1].T @ np.linalg.inv(self.Sigma)@v[k, t-1]
            # Add terminal cost
            S[k] += self._terminal_cost(x)

        # Compute information theoretic weight for each sample
        w = self._compute_weight(S)

        # Calculate w_K * epilon_k
        w_epsilon = np.zeros((self.T, self.dim_u))
        for t in range(self.T):
            for k in range(self.K):
                w_epsilon[t] += w[k]*epsilon[k, t]

        # Apply moving average filter for smoothing input sequence
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=10)

        # update control input sequence
        u += w_epsilon

        # calculate optimal trajectory
        optimal_traj = np.zeros((self.T, self.dim_x))
        if self.visualze_sampled_trajs:
            x = x0
            for t in range(self.T):
                x = self._state_transition(x, self._g(u[t-1]))
                optimal_traj[t] = x

        # calculate sampled trajectories
        sampled_traj_list = np.zeros((self.K, self.T, self.dim_x))
        sorted_idx = np.argsort(S)
        if self.visualze_sampled_trajs:
            for k in sorted_idx:
                x = x0
                for t in range(self.T):
                    x = self._state_transition(x, self._g(v[k, t-1]))
                    sampled_traj_list[k, t] = x

        # Update previous control input
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        return u[0], u, optimal_traj, sampled_traj_list

    def _compute_weight(self, S: np.ndarray) -> np.ndarray:
        """Compute weights for each sample"""
        w = np.zeros((self.K))
        # calculate rho
        rho = S.min()
        # Calculate eta nominal
        eta = 0.0
        for k in range(self.K):
            eta += np.exp(-(1.0/self.param_exploration)*(S[k]-rho))
        # Calculate weights
        for k in range(self.K):
            w[k] = (1/eta)*np.exp(-(1.0/self.param_exploration)*(S[k]-rho))

        return w

    def _state_transition(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """Calculate State Transition using Euler Lagrange"""
        x = x_t[0]
        y = x_t[1]
        yaw = x_t[2]

        speed = v_t[0]
        omega = v_t[1]

        # Update the state
        dt = self.delta_t

        x = x + speed*np.cos(yaw) * dt
        y = y + speed*np.sin(yaw) * dt
        yaw = yaw + omega * dt

        return np.array([x, y, yaw])


    def _get_nearest_waypoint(self, x: float, y: float, update_prev_idx: bool = False) -> int:
        """search the closest waypoint to the vehicle on the reference path"""
        
        SEARCH_IDX_LEN = 20
        prev_idx = self.prev_way_point_idx
        dx = [x - ref_x for ref_x in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 0]]
        dy = [y - ref_y for ref_y in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 1]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        min_d = min(d)
        nearest_idx = d.index(min_d) + prev_idx

        # Get reference values of the nearest waypoint
        ref_x = self.ref_path[nearest_idx, 0]
        ref_y = self.ref_path[nearest_idx, 1]
        ref_yaw = self.ref_path[nearest_idx, 2]

        if update_prev_idx:
            self.prev_way_point_idx = nearest_idx

        return nearest_idx, ref_x, ref_y, ref_yaw
    
    def _compute_cost(self, x_t: np.ndarray) -> float:
        """Calculate stage cost"""
        x, y, yaw = x_t
        # yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi))
        
        # Calculate stage cost
        _, ref_x, ref_y, ref_yaw = self._get_nearest_waypoint(x, y, update_prev_idx=True)

        # print(f"Ref_x: {ref_x}, Ref_y: {ref_y}, Ref_yaw: {ref_yaw}")

        stage_cost = self.stage_cost_weight[0] * (x - ref_x)**2 + \
                        self.stage_cost_weight[1] * (y - ref_y)**2 + \
                        self.stage_cost_weight[2] * (yaw - ref_yaw)**2 
        
        return stage_cost


    def _terminal_cost(self, x_t: np.ndarray) -> float:
        """Calculate terminal cost"""
        x, y, yaw = x_t
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi))
        # Calculate terminal cost
        _, ref_x, ref_y, ref_yaw = self._get_nearest_waypoint(x, y, update_prev_idx=True)
        terminal_cost = self.terminal_cost_weight[0] * (x - ref_x)**2 + \
                        self.terminal_cost_weight[1] * (y - ref_y)**2 + \
                        self.terminal_cost_weight[2] * (yaw - ref_yaw)**2
        
        return terminal_cost

    def _cross_entropy(self):
        pass
    
    def _savitky_galoy(self):
        pass

    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving average filter to the control input"""
        b = np.ones(window_size)/window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:, d] = np.convolve(xx[:, d], b, mode='same')
            n_conv = math.ceil(window_size/2)
            xx_mean[0, d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i, d] *= window_size/(i+n_conv)
                xx_mean[-1, d] *= window_size/(i + n_conv - (window_size % 2))
        
        return xx_mean

    def _calc_epsilon(self, sigma: np.ndarray, size_sample: int, size_time_step: int, size_dim_u: int) -> np.ndarray:
        """Sample epsilon"""
        # check if sigma row size == sigma col size == size_dim_u and size_dim_u > 0
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != size_dim_u or size_dim_u < 1:
            print("[ERROR] sigma must be a square matrix with the size of size_dim_u.")
            raise ValueError
        
        # sample epsilon
        mu = np.zeros((size_dim_u))
        epsilon = np.random.multivariate_normal(mu, sigma, (size_sample, size_time_step))
        return epsilon
    
    def _g(self, v: np.ndarray) -> np.ndarray:
        """Clamping the input control"""
        v[0] = np.clip(v[0], -self.max_speed, self.max_speed)
        v[1] = np.clip(v[1], -self.max_omega, self.max_omega)
        return v
    
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

    def animate(i):

        nonlocal ref_traj_artist

        current_state = diff_drive.get_state()
        optimal_input, optimal_input_sequence, optimal_traj, sampled_traj_list = mppi._calc_input_control(current_state)

        x, y, yaw = current_state

        # Clear the previous robot artists
        for artist in robot_artists:
            artist.remove()
        robot_artists.clear()

        # Draw the robot
        robot_artists.extend(diff_frame.generate_each_wheel_and_draw(x, y, yaw))
        for artist in robot_artists:
            ax.add_artist(artist)

        # Update the optimal trajectory artist
        if optimal_traj.any():
            optimal_traj_artist.set_data(optimal_traj[:, 0], optimal_traj[:, 1])
        else:
            optimal_traj_artist.set_data([], [])

        # Clear the previous sampled trajectory artists
        for artist in sampled_traj_artists:
            artist.remove()
        sampled_traj_artists.clear()

        # Draw the sampled trajectories from mppi
        if sampled_traj_list.any():
            min_alpha_value = 0.25
            max_alpha_value = 0.35
            for idx, sampled_traj in enumerate(sampled_traj_list):
                # Draw darker for better samples
                alpha_value = (1.0 - (idx + 1) / len(sampled_traj_list)) * (max_alpha_value - min_alpha_value) + min_alpha_value

                # Extend the length of the sampled trajectory lines
                extended_sampled_traj = np.zeros((sampled_traj.shape[0] + 1, sampled_traj.shape[1]))
                extended_sampled_traj[:-1, :] = sampled_traj
                extended_sampled_traj[-1, :] = sampled_traj[-1, :] + (sampled_traj[-1, :] - sampled_traj[-2, :]) * 0.5

                sampled_traj_artist, = ax.plot(sampled_traj[:, 0], sampled_traj[:, 1], color='gray', linestyle="solid", linewidth=0.2, zorder=4, alpha=alpha_value)
                sampled_traj_artists.append(sampled_traj_artist)

        # Clear the previous reference trajectory artist
        if ref_traj_artist is not None:
            ref_traj_artist.remove()

        # Draw the reference trajectory
        ref_traj_x_offset = ref_path[:, 0]
        ref_traj_y_offset = ref_path[:, 1]
        ref_traj_artist, = ax.plot(ref_traj_x_offset, ref_traj_y_offset, color='blue', linestyle="dashed", linewidth=1.0, zorder=3, label='Reference Trajectory')

        ax.set_xlim(x - 5, x + 5)
        ax.set_ylim(y - 5, y + 5)
        ax.legend()

        print(f"Optimal Input: {optimal_input}")
        print(f"Time: {i * delta_t:>2.2f}[s], x={x:>+3.3f}[m], y={y:>+3.3f}[m], yaw={yaw:>+3.3f}[rad]")

        diff_drive.update_state(delta_t, current_state, optimal_input)

        return robot_artists + [optimal_traj_artist] + sampled_traj_artists + [ref_traj_artist]

    ani = FuncAnimation(fig, animate, frames=tSim, interval=50, blit=True, repeat=False)
    ani.save("mppi_differential_drive.mp4", writer='ffmpeg', fps=30)

def generate_circle_points(center_x, center_y, radius, num_points=20):
    angles = np.linspace(0, 2 * pi, num_points)
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    return x, y


if __name__ == "__main__":
    # Initialize the differential drive model
    init_x = np.array([0.0, 0.0, 0.0])
    tSim = 500
    diff_drive = DifferentialDrive(init_x)

    # Initialize the MPPI algorithm
    delta_t = 0.1
    max_speed = 3.0
    max_omega = 3.14
    num_samples_K = 500
    num_horizons_T = 15
    param_exploration = 0.01
    param_lambda = 1.0 
    param_alpha = 0.2
    sigma = np.array([[0.1, 0.0], [0.0, 0.1]])
    stage_cost_weight = np.array([5.0, 5.0, 10.0])
    terminal_cost_weight = np.array([5.0, 5.0, 10.0])

    # Generate reference path
    # x = [0.0, 0.5, 1.0, 3.0, 3.0, 1.0, -4.0]
    # y = [0.0, 1.0, 1.0, 2.0, 5.0, 1.0, -1.0]
    # ds = 0.1
    # cx, cy, cyaw, ck, s = calc_spline_course(x, y, ds=ds)
    center_x = 0.0
    center_y = 0.0
    radius = 5.0
    ds = 0.1

    # Generate control points for the circular curve
    x, y = generate_circle_points(center_x, center_y, radius)

    # # Calculate the cubic spline course
    cx, cy, cyaw, ck, s = calc_spline_course(x, y, ds=ds)

    ref_path = np.array([cx, cy, cyaw]).T

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
        terminal_cost_weight=terminal_cost_weight
    )

    print("[INFO] MPPI Algorithm Initialized.")
    # Run the MPPI algorithm
    plot_trajectories(diff_frame, diff_drive, mppi, delta_t, ref_path, tSim)

