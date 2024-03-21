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
            num_samples_K: int,
            num_horizons: int,
            

    )
        

    def mppi(self):
        pass

    def _compute_weight(self):
        pass

    def _savitsky_galoy(self):
        pass

    def _state_transition(self):
        pass

    def _compute_cost(self):
        pass

    def _terminal_cost(self):
        pass

    def _cross_entropy(self):
        pass
