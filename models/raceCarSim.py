import numpy as np

class RacecarModel:
    """
    A class representing a racecar model.

    This class provides methods for initializing the racecar's initial state
    and computing the forward kinematics based on acceleration and steering angle.

    Attributes:
        x (float): The x-coordinate of the racecar.
        y (float): The y-coordinate of the racecar.
        yaw (float): The orientation (yaw) of the racecar in radians.
        v (float): The velocity of the racecar.
    """

    def __init__(self, initial_state: np.ndarray, L=2.5, dt=0.1):
        """
        Initialize the racecar with the given initial state.

        Args:
            initial_state (np.ndarray): A 1D numpy array representing the initial state
                of the racecar in the format [x, y, yaw, v], where:
                    x (float): The initial x-coordinate of the racecar.
                    y (float): The initial y-coordinate of the racecar.
                    yaw (float): The initial orientation (yaw) of the racecar in radians.
                    v (float): The initial velocity of the racecar.
            L (float): Length of the racecar (distance between front and rear axles).
            dt (float): Time step for the simulation.
        """
        self.x = initial_state[0]
        self.y = initial_state[1]
        self.yaw = initial_state[2]
        self.v = initial_state[3]
        self.L = L
        self.dt = dt

    def forward_kinematic(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Compute the forward kinematics of the racecar.

        This method computes the state update of the racecar based on the given control input u = [a, delta],
        where a is the acceleration and delta is the steering angle.

        Args:
            u (np.ndarray): A 1D numpy array representing the control input
                in the format [a, delta], where:
                    a (float): The acceleration of the racecar.
                    delta (float): The steering angle of the racecar in radians.

        Returns:
            np.ndarray: A 1D numpy array representing the updated state of the racecar
                in the format [x, y, yaw, v], where:
                    x (float): The updated x-coordinate of the racecar.
                    y (float): The updated y-coordinate of the racecar.
                    yaw (float): The updated orientation (yaw) of the racecar in radians.
                    v (float): The updated velocity of the racecar.
        """
        a, delta = u
        dx = x[3] * np.cos(x[2])
        dy = x[3] * np.sin(x[2])
        dyaw = x[3] / self.L * np.tan(delta)
        dv = a

        return np.array([dx, dy, dyaw, dv])
