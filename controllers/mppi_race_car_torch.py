import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import matplotlib.pyplot as plt
import math

class MPPIRacecarController:
    def __init__(
            self,
            delta_t: float = 0.05,
            wheel_base: float = 2.5, # [m]
            max_steer_abs: float = 0.523, # [rad]
            max_accel_abs: float = 2.000, # [m/s^2]
            ref_path: torch.Tensor = torch.tensor([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]]),
            horizon_step_T: int = 10,
            number_of_samples_K: int = 100,
            param_exploration: float = 0.01,
            param_lambda: float = 50.0,
            param_alpha: float = 1.0,
            sigma: torch.Tensor = torch.tensor([[0.5, 0.0], [0.0, 0.1]]), 
            stage_cost_weight: torch.Tensor = torch.tensor([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
            terminal_cost_weight: torch.Tensor = torch.tensor([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
            visualize_optimal_traj = True, 
            visualze_sampled_trajs = False,
    ):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dim_x = 4
        self.dim_u = 2

        self.T = horizon_step_T
        self.K = number_of_samples_K
        self.param_exploration = param_exploration
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))
        self.Sigma = sigma.float().to(self.device)
        self.stage_cost_weight = stage_cost_weight.float().to(self.device)
        self.terminal_cost_weight = terminal_cost_weight.float().to(self.device)
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualze_sampled_trajs = visualze_sampled_trajs

        self.delta_t = delta_t
        self.wheel_base = wheel_base
        self.max_steer_abs = max_steer_abs
        self.max_accel_abs = max_accel_abs
        self.ref_path = ref_path.float().to(self.device)

        self.u_prev = torch.zeros((self.T, self.dim_u)).to(self.device)

        self.prev_waypoints_idx = 0


    def _calc_control_input(self, observed_x: torch.Tensor) -> tuple:

        u = self.u_prev

        x0 = observed_x.to(self.device)

        self.get_nearest_waypoint(x0[0], x0[1], update_prev_idx=True)
        if self.prev_waypoints_idx >= self.ref_path.shape[0] - 1:
            print("[ERROR] Reached the end of the reference path.")
            raise IndexError
        
        S = torch.zeros((self.K)).to(self.device)

        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)

        v = torch.zeros((self.K, self.T, self.dim_u)).to(self.device)

        for k in range(self.K):
            x = x0
            
            for t in range(1, self.T + 1):
                if k < (1.0 - self.param_exploration) * self.K:
                    v[k, t-1] = u[t-1] + epsilon[k, t-1]
                else:
                    v[k, t-1] = epsilon[k, t-1]

                x = self._F(x, self._g(v[k, t-1]))

                # S[k] += self._c(x) + self.param_gamma*u[t-1].T@torch.linalg.inv(self.Sigma)@v[k, t-1]
                S[k] += self._c(x) + self.param_gamma * torch.matmul(u[t-1].reshape(1, -1), torch.matmul(torch.linalg.inv(self.Sigma), v[k, t-1].reshape(-1, 1))).squeeze()

            S[k] += self._phi(x)

        
        w = self._compute_weight(S)

        w_epsilon = torch.zeros((self.T, self.dim_u)).to(self.device)

        for t in range(self.T):
            for k in range(self.K):
                w_epsilon[t] += w[k] * epsilon[k, t]

        w_epsilon = self._moving_average_filter(w_epsilon, window_size=10)
        # w_epsilon = self.savitzky_golay_filter(w_epsilon, window_size=10, polynomial_order=3)

        u += w_epsilon

        optimal_traj = torch.zeros((self.T, self.dim_x)).to(self.device)
        if self.visualize_optimal_traj:
            x = x0
            for t in range(self.T):
                x = self._F(x, self._g(u[t - 1]))
                optimal_traj[t] = x

        sampled_traj_list = torch.zeros((self.K, self.T, self.dim_x)).to(self.device)
        sorted_idx = torch.argsort(S)
        if self.visualze_sampled_trajs:
            for k in sorted_idx:
                x = x0
                for t in range(self.T):
                    x = self._F(x, self._g(v[k, t - 1]))
                    sampled_traj_list[k, t] = x

                    
        self.u_prev[:-1] = u[1:].clone()
        self.u_prev[-1] = u[-1].clone()

        return u[0], u, optimal_traj, sampled_traj_list

    def _calc_epsilon(self,
                      sigma: torch.Tensor,
                      size_sample: int,
                      size_time_step: int,
                      size_dim_u: int
                      ) -> torch.Tensor:
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != size_dim_u:
            print("[ERROR] sigma must be a square matrix with the size of size_dim_u.")
            raise ValueError
        
        mu = torch.zeros((size_dim_u)).to(self.device)
        epsilon = torch.distributions.MultivariateNormal(mu, sigma).sample((size_sample, size_time_step)).to(self.device)
        return epsilon
    
    def _c(self, x_t: torch.Tensor) -> torch.Tensor:

        x, y, yaw, v = x_t

        yaw = ((yaw + 2.0*torch.pi) % (2.0*torch.pi))

        _, ref_x, ref_y, ref_yaw, ref_v = self.get_nearest_waypoint(x, y)
        stage_cost = self.stage_cost_weight[0]*(x-ref_x)**2 + self.stage_cost_weight[1]*(y-ref_y)**2 + \
                    self.stage_cost_weight[2]*(yaw-ref_yaw)**2 + self.stage_cost_weight[3]*(v-ref_v)**2
        return stage_cost
    
    def _phi(self, x_T: torch.Tensor) -> torch.Tensor:
        x, y, yaw, v= x_T
        yaw = ((yaw + 2.0*torch.pi) % (2.0*torch.pi))

        _, ref_x, ref_y, ref_yaw, ref_v = self.get_nearest_waypoint(x, y)
        terminal_cost = self.terminal_cost_weight[0]*(x-ref_x)**2 + self.terminal_cost_weight[1]*(y-ref_y)**2 + \
                        self.terminal_cost_weight[2]*(yaw-ref_yaw)**2 + self.terminal_cost_weight[3]*(v-ref_v)**2
        return terminal_cost
    
    def get_nearest_waypoint(self, x: torch.Tensor, y: torch.Tensor, update_prev_idx: bool = False):
        SEARCH_INDEX_LEN = 200
        prev_idx = self.prev_waypoints_idx
        dx = x - self.ref_path[prev_idx:prev_idx+SEARCH_INDEX_LEN, 0]
        dy = y - self.ref_path[prev_idx:prev_idx+SEARCH_INDEX_LEN, 1]
        d  = dx ** 2 + dy ** 2
        min_d, nearest_idx = torch.min(d, dim=0)
        nearest_idx = nearest_idx.item() + prev_idx

        ref_x = self.ref_path[nearest_idx, 0]
        ref_y = self.ref_path[nearest_idx, 1]
        ref_yaw = self.ref_path[nearest_idx, 2]
        ref_v = self.ref_path[nearest_idx, 3]

        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx

        return nearest_idx, ref_x, ref_y, ref_yaw, ref_v
    
    def _g(self, v: torch.Tensor) -> torch.float:
        v[0] = torch.clamp(v[0], -self.max_steer_abs, self.max_steer_abs)
        v[1] = torch.clamp(v[1], -self.max_accel_abs, self.max_accel_abs)
        return v
    
    def _F(self, x_t: torch.Tensor, v_t: torch.Tensor) -> torch.Tensor:
        x, y, yaw, v = x_t
        steer, accel = v_t

        l = self.wheel_base
        dt = self.delta_t

        new_x = x + v * torch.cos(yaw) * dt
        new_y = y + v * torch.sin(yaw) * dt
        new_yaw = yaw + v/l * torch.tan(steer) * dt
        new_v = v + accel * dt

        x_t_plus_1 = torch.stack([new_x, new_y, new_yaw, new_v]).to(self.device)

        return x_t_plus_1
    
    def _compute_weight(self, S: torch.Tensor) -> torch.Tensor:
        
        w = torch.zeros((self.K)).to(self.device)

        rho = S.min()

        eta = torch.exp((-1.0 / self.param_lambda) * (S - rho)).sum()
        
        w   = (1.0 / eta ) * torch.exp((-1.0 / self.param_lambda) * (S - rho))

        return w
    
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

    def savitzky_golay_coefficients(self, window_size: int, polynomial_order: int) -> torch.Tensor:
        half_window = (window_size - 1 ) // 2
        j = torch.arange(-half_window, half_window + 1).float()
        b = torch.stack([j ** i for i in range(polynomial_order + 1)], dim=1)
        m = torch.linalg.pinv(b).squeeze(1)[0]
        return torch.flip(m, dims=[0])
    
    def savitzky_golay_filter(self, x: torch.Tensor, window_size: int, polynomial_order: int) -> torch.Tensor:
        half_window = (window_size - 1) // 2
        num_channels = x.shape[1]

        coefficients =  self.savitzky_golay_coefficients(window_size, polynomial_order).to(x.device)

        padded_x = torch.nn.functional.pad(x, (half_window, half_window), mode='reflect')

        filtered_x = torch.zeros_like(x)
        for i in range(x.shape[0]):
            start = max(0, i - half_window)
            end = min(x.shape[0], i + half_window + 1)
            window = x[start:end]
            window_size = window.shape[0]
            if window_size < coefficients.shape[0]:
                coeffs = coefficients[:window_size]
            else:
                coeffs = coefficients
            filtered_x[i] = torch.matmul(window.transpose(0, 1), coeffs).squeeze()

        return filtered_x
    
    def generate_simple_trajectory(self, num_points: int, radius: float) -> torch.Tensor:

        angles = torch.linspace(0, 2*torch.pi, num_points)
        x = radius * torch.cos(angles)
        y = radius * torch.sin(angles)
        yaw = angles + torch.pi / 2
        v = torch.ones_like(angles) * 5.0

        trajectory = torch.stack([x, y, yaw, v], dim=1)

        return trajectory
    
    def plot_control_signals(self, control_signals: torch.Tensor) -> None:
 
        time = torch.arange(0, control_signals.shape[0] * self.delta_t, self.delta_t)
        
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
    ref_path = controller.generate_simple_trajectory(
        num_points,
        radius
    )

    controller.ref_path = ref_path.float().to(controller.device)

    control_signals = torch.zeros((num_points, 2))

    for i in range(num_points):
        current_state = ref_path[i]

        optimal_input, _, _, _ = controller._calc_control_input(current_state)

        print(f"Current state: {current_state}")

        control_signals[i] = optimal_input


    controller.plot_control_signals(control_signals)