import numpy as np
import math


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
        ) -> None:
        """Initialize the MPPI algorithm"""
        ## mppi parameters
        self.delta_t = delta_t
        self.max_speed = max_speed
        self.max_omega = max_omega
        self.dim_x = 3
        self.dim_u = 2
        self.T = num_horizons_T
        self.K = num_samples_K
        self.param_exploration = param_exploration
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        self.Sigma = sigma
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight

        # mppi variables 
        self.u_prev = np.zeros((self.T, self.K, self.dim_u))

        # ref path info
        self.prev_way_point_idx = 0

    def mppi(self):
        pass

    def _compute_weight(self, S: np.ndarray) -> np.ndarray:
        """Compute weights for each sample"""
        w = np.zeros((self.K))

        # calculate rho
        rho = S.min()

        # Calculate eta nominal
        eta = 0.0
        for k in range(self.K):
            eta += np.exp((-1.0/self.param_exploration) * (S[k] - rho) )

        # Calculate weights
        for k in range(self.K):
            w[k] = (1/eta)*np.exp(-(1.0/self.param_exploration)*S[k]-rho)

        return w

    def _state_transition(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """Calculate State Transition using Euler Lagrange"""
        x, y, yaw = x_t
        speed, omega = v_t

        # Update the state
        dt = self.delta_t

        x = x + speed*np.cos(yaw) * dt
        y = y + speed*np.sin(yaw) * dt
        yaw = yaw + omega * dt

        return np.array([x, y, yaw])


    def _get_nearest_waypoint(self, x: float, y: float, ref_path: np.ndarray, update_prev_idx: bool = False) -> int:
        """search the closest waypoint to the vehicle on the reference path"""
        
        SEARCH_IDX_LEN = 200
        prev_idx = self.prev_way_point_idx
        dx = [x - ref_x for ref_x in ref_path[prev_idx:prev_idx+SEARCH_IDX_LEN, 0]]
        dy = [y - ref_y for ref_y in ref_path[prev_idx:prev_idx+SEARCH_IDX_LEN, 1]]
        d  = [ idx **2 + idy ** 2 for idx, idy in zip(dx, dy)]
        min_d = min(d)
        nearest_idx = d.index(min_d) + prev_idx

        # Get reference values of the nearest waypoint
        ref_x = ref_path[nearest_idx, 0]
        ref_y = ref_path[nearest_idx, 1]
        ref_yaw = ref_path[nearest_idx, 2]

        if update_prev_idx:
            self.prev_way_point_idx = nearest_idx

        return nearest_idx, ref_x, ref_y, ref_yaw
    
    def _compute_cost(self, x_t: np.ndarray, ref_path: np.ndarray) -> float:
        """Calculate stage cost"""
        x, y, yaw = x_t
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi))
        
        # Calculate stage cost
        _, ref_x, ref_y, ref_yaw = self._get_nearest_waypoint(x, y, ref_path)
        stage_cost = self.stage_cost_weight[0] * (x - ref_x)**2 + \
                        self.stage_cost_weight[1] * (y - ref_y)**2 + \
                        self.stage_cost_weight[2] * (yaw - ref_yaw)**2 
        
        return stage_cost


    def _terminal_cost(self, x_t: np.ndarray, ref_path: np.ndarray) -> float:
        """Calculate terminal cost"""
        x, y, yaw = x_t
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi))

        # Calculate terminal cost
        _, ref_x, ref_y, ref_yaw = self._get_nearest_waypoint(x, y, ref_path)
        terminal_cost = self.terminal_cost_weight[0] * (x - ref_x)**2 + \
                        self.terminal_cost_weight[1] * (y - ref_y)**2 + \
                        self.terminal_cost_weight[2] * (yaw - ref_yaw)**2
        
        return terminal_cost

    def _cross_entropy(self):
        pass

    
    def _savitky_galoy(self):
        pass

    def _moving_average_filter(self):
        pass
