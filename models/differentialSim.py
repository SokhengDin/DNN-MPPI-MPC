import numpy as np
import matplotlib.pyplot as plt


class DifferentialSimulation:

    def __init__(self, r_h=0.2, r_w=0.30, w_h=0.025, w_w=0.075, rot_angle=0.0, w_pos=0.15):
        ## robot shape
        self.robot_h = r_h
        self.robot_w = r_w
        ## wheel shaspe
        self.wheel_h = w_h
        self.wheel_w = w_w
        ## wheel position
        self.w_pos = w_pos
        ## Rotation angle
        self.rotate_angle = rot_angle

    def rotation_angle(self, theta):
        matrix_transform = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]]).T
        return matrix_transform
    
    def robot_shape(self):
        matrix_shape = np.array([
            [-self.robot_w, self.robot_w, self.robot_w, -self.robot_w, -self.robot_w],
            [self.robot_h, self.robot_h, -self.robot_h, -self.robot_h, self.robot_h]
        ])
        return matrix_shape
    
    def robot_wheel(self):
        matrix_wheel = np.array([
            [-self.wheel_w, self.wheel_w, self.wheel_w, -self.wheel_w, -self.wheel_w],
            [self.wheel_h, self.wheel_h, -self.wheel_h, -self.wheel_h, self.wheel_h]
        ])
        return matrix_wheel
    
    def generate_each_wheel_and_draw(self, x, y, yaw):
        # Copy Matrix as robot_shape
        pos_wheel = self.robot_wheel()
        pos_wheel1 = pos_wheel.copy()
        pos_wheel2 = pos_wheel.copy()
        pos_wheel3 = pos_wheel.copy()
        pos_wheel4 = pos_wheel.copy()

        # Push each wheel to where it belong
        pos_wheel1[0, :] += self.w_pos
        pos_wheel1[1, :] -= self.w_pos
        pos_wheel2[0, :] += self.w_pos
        pos_wheel2[1, :] += self.w_pos
        pos_wheel3[0, :] -= self.w_pos
        pos_wheel3[1, :] += self.w_pos
        pos_wheel4[0, :] -= self.w_pos
        pos_wheel4[1, :] -= self.w_pos

        # Matrix Transforms each wheel 1, 2, 3, 4
        pos_wheel1 = np.dot(pos_wheel1.T, self.rotation_angle(yaw)).T
        pos_wheel2 = np.dot(pos_wheel2.T, self.rotation_angle(yaw)).T
        pos_wheel3 = np.dot(pos_wheel3.T, self.rotation_angle(yaw)).T
        pos_wheel4 = np.dot(pos_wheel4.T, self.rotation_angle(yaw)).T

        # Matrix Transforms robot shape
        robot_shape = self.robot_shape()
        robot_shaped = np.dot(robot_shape.T, self.rotation_angle(yaw)).T

        artists = []

        artists.append(plt.Line2D(robot_shaped[0, :]+x, robot_shaped[1, :]+y, color="black"))
        artists.append(plt.Line2D([x], [y], marker="x", color="red"))
        artists.append(plt.Line2D(pos_wheel1[0, :]+x, pos_wheel1[1, :]+y, color="black"))
        artists.append(plt.Line2D(pos_wheel2[0, :]+x, pos_wheel2[1, :]+y, color="black"))
        artists.append(plt.Line2D(pos_wheel3[0, :]+x, pos_wheel3[1, :]+y, color="black"))
        artists.append(plt.Line2D(pos_wheel4[0, :]+x, pos_wheel4[1, :]+y, color="black"))

        return artists
    
class DifferentialDrive:
    """
    A class representing a differential drive robot.

    This class provides methods for initializing the robot's initial state
    and computing the forward kinematics of the differential drive robot.

    Attributes:
        x0 (float): The initial x-coordinate of the robot.
        y0 (float): The initial y-coordinate of the robot.
        theta0 (float): The initial orientation (angle) of the robot in radians.
    """

    def __init__(self, x0_initial: np.ndarray):
        """
        Initialize the differential drive robot with the given initial state.

        Args:
            x0_initial (np.ndarray): A 1D numpy array representing the initial state
                of the robot in the format [x, y, theta], where:
                    x (float): The initial x-coordinate of the robot.
                    y (float): The initial y-coordinate of the robot.
                    theta (float): The initial orientation (angle) of the robot in radians.
        """
        self.x0 = x0_initial[0]
        self.y0 = x0_initial[1]
        self.theta0 = x0_initial[2]

    def forward_kinematic(self, u: np.ndarray) -> np.ndarray:
        """
        Compute the forward kinematics of the differential drive robot.

        This method computes the linear and angular velocities of the robot
        based on the given control input u = [v, w], where v is the linear
        velocity and w is the angular velocity.

        The state of the robot is represented as x = [x, y, theta], where
        x and y are the position coordinates, and theta is the orientation
        (angle) in radians.

        The forward kinematics equations for a differential drive robot are:

            [ẋ]   [v * cos(θ)]
            [ẏ] = [v * sin(θ)]
            [θ̇]   [     w    ]

        Args:
            u (np.ndarray): A 1D numpy array representing the control input
                in the format [v, w], where:
                    v (float): The linear velocity of the robot.
                    w (float): The angular velocity of the robot.

        Returns:
            np.ndarray: A 1D numpy array representing the linear and angular
                velocities of the robot in the format [ẋ, ẏ, θ̇], where:
                    ẋ (float): The linear velocity along the x-axis.
                    ẏ (float): The linear velocity along the y-axis.
                    θ̇ (float): The angular velocity (rate of change of orientation).
        """
        v, w = u
        theta = self.theta0
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = w
        return np.array([x_dot, y_dot, theta_dot])
    