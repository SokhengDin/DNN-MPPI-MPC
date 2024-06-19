import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

from models.vehicle import Vehicle

class MPPIRacecarController:
    def __init__(
            self,
            delta_t: float = 0.05,
            wheel_base: float = 2.5, 
            max_steer_abs: float = 0.523,
            max_accel_abs: float = 2.000,
            ref_path: np.ndarray = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]]),
            horizon_step_T: int = 10,
            number_of_samples_K: int = 100,
            param_exploration: float = 0.01,
            param_lambda: float = 50.0,
            param_alpha: float = 1.0,
            sigma: np.ndarray = np.array([[0.5, 0.0], [0.0, 0.1]]), 
            stage_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]), 
            terminal_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]),
            visualize_optimal_traj = True, 
            visualze_sampled_trajs = True,
    ):
        
        self.dim_x = 4
        self.dim_u = 2

        self.T = horizon_step_T
        self.K = number_of_samples_K
        self.param_exploration = param_exploration
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))
        self.Sigma = sigma.astype(np.float32)
        self.stage_cost_weight = stage_cost_weight.astype(np.float32)
        self.terminal_cost_weight = terminal_cost_weight.astype(np.float32)
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualze_sampled_trajs = visualze_sampled_trajs

        self.delta_t = delta_t
        self.wheel_base = wheel_base
        self.max_steer_abs = max_steer_abs
        self.max_accel_abs = max_accel_abs
        self.ref_path = ref_path.astype(np.float32)

        self.u_prev = np.zeros((self.T, self.dim_u), dtype=np.float32)

        self.prev_waypoints_idx = 0


    def _calc_control_input(self, observed_x: np.ndarray) -> tuple:

        u = self.u_prev

        x0 = observed_x.astype(np.float32)

        self.get_nearest_waypoint(x0[0], x0[1], update_prev_idx=True)

        if self.prev_waypoints_idx >= self.ref_path.shape[0] - 1:
            print("[ERROR] Reached the end of the reference path.")
            raise IndexError
        
        S = np.zeros((self.K), dtype=np.float32)

        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)

        v = np.zeros((self.K, self.T, self.dim_u), dtype=np.float32)

        for k in range(self.K):
            x = x0
            
            for t in range(1, self.T + 1):
                if k < (1.0 - self.param_exploration) * self.K:
                    v[k, t-1] = u[t-1] + epsilon[k, t-1]
                else:
                    v[k, t-1] = epsilon[k, t-1]

                x = self._F(x, self._g(v[k, t-1]))

                S[k] += self._c(x) + self.param_gamma * np.matmul(u[t-1].reshape(1, -1), np.matmul(np.linalg.inv(self.Sigma), v[k, t-1].reshape(-1, 1))).squeeze()

            S[k] += self._phi(x)

        
        w = self._compute_weight(S)

        w_epsilon = np.zeros((self.T, self.dim_u), dtype=np.float32)

        for t in range(self.T):
            for k in range(self.K):
                w_epsilon[t] += w[k] * epsilon[k, t]

        w_epsilon = self._moving_average_filter(w_epsilon, window_size=10)

        u += w_epsilon

        optimal_traj = np.zeros((self.T, self.dim_x), dtype=np.float32)
        if self.visualize_optimal_traj:
            x = x0
            for t in range(self.T):
                x = self._F(x, self._g(u[t - 1]))
                optimal_traj[t] = x

        sampled_traj_list = np.zeros((self.K, self.T, self.dim_x), dtype=np.float32)
        sorted_idx = np.argsort(S)
        if self.visualze_sampled_trajs:
            for k in sorted_idx:
                x = x0
                for t in range(self.T):
                    x = self._F(x, self._g(v[k, t - 1]))
                    sampled_traj_list[k, t] = x

                    
        self.u_prev[:-1] = u[1:].copy()
        self.u_prev[-1] = u[-1].copy()

        return u[0], u, optimal_traj, sampled_traj_list

    def _calc_epsilon(self,
                      sigma: np.ndarray,
                      size_sample: int,
                      size_time_step: int,
                      size_dim_u: int
                      ) -> np.ndarray:
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != size_dim_u:
            print("[ERROR] sigma must be a square matrix with the size of size_dim_u.")
            # raise ValueError
        
        mu = np.zeros((size_dim_u), dtype=np.float32)
        epsilon = np.random.multivariate_normal(mu, sigma, (size_sample, size_time_step)).astype(np.float32)
        return epsilon
    
    def _c(self, x_t: np.ndarray) -> np.float32:

        x, y, yaw, v = x_t

        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi))

        _, ref_x, ref_y, ref_yaw, ref_v = self.get_nearest_waypoint(x, y)
        stage_cost = self.stage_cost_weight[0]*(x-ref_x)**2 + self.stage_cost_weight[1]*(y-ref_y)**2 + \
                    self.stage_cost_weight[2]*(yaw-ref_yaw)**2 + self.stage_cost_weight[3]*(v-ref_v)**2
        return stage_cost
    
    def _phi(self, x_T: np.ndarray) -> np.float32:
        x, y, yaw, v= x_T
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi))

        _, ref_x, ref_y, ref_yaw, ref_v = self.get_nearest_waypoint(x, y)
        terminal_cost = self.terminal_cost_weight[0]*(x-ref_x)**2 + self.terminal_cost_weight[1]*(y-ref_y)**2 + \
                        self.terminal_cost_weight[2]*(yaw-ref_yaw)**2 + self.terminal_cost_weight[3]*(v-ref_v)**2
        return terminal_cost
    
    def get_nearest_waypoint(self, x: np.ndarray, y: np.ndarray, update_prev_idx: bool = False):
        SEARCH_INDEX_LEN = 200
        prev_idx = self.prev_waypoints_idx
        dx = x - self.ref_path[prev_idx:prev_idx+SEARCH_INDEX_LEN, 0]
        dy = y - self.ref_path[prev_idx:prev_idx+SEARCH_INDEX_LEN, 1]
        d  = dx ** 2 + dy ** 2
        min_d = np.min(d)
        nearest_idx = np.argmin(d) + prev_idx

        ref_x = self.ref_path[nearest_idx, 0]
        ref_y = self.ref_path[nearest_idx, 1]
        ref_yaw = self.ref_path[nearest_idx, 2]
        ref_v = self.ref_path[nearest_idx, 3]

        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx

        return nearest_idx, ref_x, ref_y, ref_yaw, ref_v
    
    def _g(self, v: np.ndarray) -> np.float32:

        v[0] = np.clip(v[0], -self.max_steer_abs, self.max_steer_abs)
        v[1] = np.clip(v[1], -self.max_accel_abs, self.max_accel_abs)

        return v
    
    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        x, y, yaw, v = x_t
        steer, accel = v_t

        l = self.wheel_base
        dt = self.delta_t

        new_x = x + v * np.cos(yaw) * dt
        new_y = y + v * np.sin(yaw) * dt
        new_yaw = yaw + v/l * np.tan(steer) * dt
        new_v = v + accel * dt

        x_t_plus_1 = np.stack([new_x, new_y, new_yaw, new_v])

        return x_t_plus_1
    
    def _compute_weight(self, S: np.ndarray) -> np.ndarray:
        
        w = np.zeros((self.K), dtype=np.float32)

        rho = S.min()

        eta = np.exp((-1.0 / self.param_lambda) * (S - rho)).sum()
        
        w   = (1.0 / eta ) * np.exp((-1.0 / self.param_lambda) * (S - rho))

        return w
    
    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving average filter to the data"""
        kernel_size = window_size
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        dim = xx.shape[1]
        xx_mean = np.zeros_like(xx)

        for d in range(dim):
            xx_padded = np.concatenate([xx[:kernel_size//2, d], xx[:, d], xx[-kernel_size//2:, d]], axis=0)
            xx_mean[:, d] = np.convolve(xx_padded, kernel, mode='same')[kernel_size//2:-kernel_size//2]

        return xx_mean
    
    def generate_simple_trajectory(self, num_points: int, radius: float) -> np.ndarray:

        angles = np.linspace(0, 2*np.pi, num_points, dtype=np.float32)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        yaw = angles + np.pi / 2
        v = np.ones_like(angles) * 5.0

        trajectory = np.stack([x, y, yaw, v], axis=1)

        return trajectory
    
    def plot_control_signals(self, control_signals: np.ndarray) -> None:
 
        time = np.arange(0, control_signals.shape[0] * self.delta_t, self.delta_t)
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(time, control_signals[:, 0], linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Steering [rad]')
        plt.title('Steering Control Signal')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(time, control_signals[:, 1], linewidth=2)
        plt.xlabel('Time [s]')
        plt.ylabel('Acceleration [m/s^2]')
        plt.title('Acceleration Control Signal')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    controller = MPPIRacecarController()

    num_points = 100
    radius = 10.0
    ref_path = controller.generate_simple_trajectory(num_points, radius)

    controller.ref_path = ref_path.astype(np.float32)

    vehicle = Vehicle(ref_path=ref_path[:, :2], visualize=True)

    for i in range(num_points):
        current_state = ref_path[i]

        optimal_input, _, optimal_traj, sampled_traj_list = controller._calc_control_input(current_state)

        print(f"Current state: {current_state}")

        vehicle.update(u=optimal_input, delta_t=controller.delta_t, optimal_traj=optimal_traj[:, 0:2], sampled_traj_list=sampled_traj_list[:, :, 0:2])

    # Save the animation
    vehicle.save_animation("mppi_vehicle.mp4", interval=int(controller.delta_t * 1000), movie_writer="ffmpeg")