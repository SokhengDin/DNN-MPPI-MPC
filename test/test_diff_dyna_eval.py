import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pybullet as p
import pybullet_data
import time
import torch
import torch.nn as nn
import casadi as ca
from scipy.linalg import block_diag
from sklearn.preprocessing import StandardScaler
from test_diff_mpc_dyna import MPCController

class ResidualDynamicsNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ResidualDynamicsNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

def load_dnn_model(model_path, input_dim, output_dim):
    model = ResidualDynamicsNet(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


class HybridMPCController(MPCController):
    def __init__(self, dnn_model, scaler_X, scaler_Y, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dnn_model = dnn_model
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.load_dnn_weights()

    def export_casadi_model(self):
        nominal_model = super().export_casadi_model()
        states = nominal_model.x
        controls = nominal_model.u

        input_scaled = (ca.vertcat(states, controls) - self.scaler_X.mean_) / self.scaler_X.scale_
        dnn_output = self.casadi_dnn(input_scaled)
        dnn_residual = dnn_output * self.scaler_Y.scale_ + self.scaler_Y.mean_

        rhs = nominal_model.f_expl_expr + dnn_residual

        hybrid_model = ca.types.SimpleNamespace()
        hybrid_model.x = states
        hybrid_model.u = controls
        hybrid_model.f_expl_expr = rhs
        hybrid_model.name = 'hybrid_dynamics'

        return hybrid_model

    def casadi_dnn(self, x):
        h1 = ca.tanh(self.W1 @ x + self.b1)
        h2 = ca.tanh(self.W2 @ h1 + self.b2)
        h3 = ca.tanh(self.W3 @ h2 + self.b3)
        y = self.W4 @ h3 + self.b4
        return y

    def load_dnn_weights(self):
        self.W1 = self.dnn_model.network[0].weight.data.numpy().T
        self.b1 = self.dnn_model.network[0].bias.data.numpy()
        self.W2 = self.dnn_model.network[3].weight.data.numpy().T
        self.b2 = self.dnn_model.network[3].bias.data.numpy()
        self.W3 = self.dnn_model.network[6].weight.data.numpy().T
        self.b3 = self.dnn_model.network[6].bias.data.numpy()
        self.W4 = self.dnn_model.network[9].weight.data.numpy().T
        self.b4 = self.dnn_model.network[9].bias.data.numpy()

def generate_trajectory(trajectory_type, duration, dt):
    t = np.arange(0, duration, dt)
    if trajectory_type == "circle":
        radius = 5.0
        omega = 0.5
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        theta = omega * t + np.pi/2
    elif trajectory_type == "lemniscate":
        a = 5.0
        x = a * np.cos(t) / (1 + np.sin(t)**2)
        y = a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
        theta = np.arctan2(np.cos(t), -np.sin(t))
    else:
        raise ValueError("Unknown trajectory type")
    
    v = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt
    v = np.append(v, v[-1])
    omega = np.diff(theta) / dt
    omega = np.append(omega, omega[-1])
    
    return np.column_stack((x, y, theta, v, omega))

def evaluate_controller(controller, trajectory, robot_id, wheel_joints):
    errors = []
    simX = np.zeros((controller.N + 1, controller.model.x.size()[0]))
    simU = np.zeros((controller.N, controller.model.u.size()[0]))

    for i in range(len(trajectory) - 1):
        # Get current state
        pos, ori = p.getBasePositionAndOrientation(robot_id)
        lin_vel, ang_vel = p.getBaseVelocity(robot_id)
        euler = p.getEulerFromQuaternion(ori)
        current_state = np.array([pos[0], pos[1], euler[2], np.linalg.norm(lin_vel[:2]), ang_vel[2]])

        # Set reference
        yref = trajectory[i]
        yref_N = trajectory[i+1][:3]  # Only position and orientation for terminal reference

        # Solve MPC
        simX, simU = controller.solve_mpc(current_state, simX, simU, yref, yref_N, controller.obstacle_positions)

        # Extract control inputs
        control_input = simU[0]

        # Apply control to the robot
        p.setJointMotorControlArray(robot_id, wheel_joints, p.TORQUE_CONTROL, forces=control_input)

        # Step the simulation
        p.stepSimulation()

        # Calculate error
        error = np.linalg.norm(current_state[:2] - yref[:2])  # Position error
        errors.append(error)

        time.sleep(controller.dt)

    return errors

def run_evaluation():
    # PyBullet setup
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load plane and robot
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("/home/eroxii/ocp_ws/bullet3/data/husky/husky.urdf", [0, 0, 0.1])

    # Identify wheel joints
    wheel_joints = []
    joints_list = ['front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel']
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[1].decode('utf-8') in joints_list:
            wheel_joints.append(i)

    # Load DNN model and scalers
    input_dim = 9  # 5 for state, 4 for control
    output_dim = 5  # Assuming the residual is for the full state
    dnn_model = load_dnn_model('residual_dynamics_model.pth', input_dim, output_dim)
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    scaler_X.mean_ = np.load('scaler_X_mean.npy')
    scaler_X.scale_ = np.load('scaler_X_scale.npy')
    scaler_Y.mean_ = np.load('scaler_Y_mean.npy')
    scaler_Y.scale_ = np.load('scaler_Y_scale.npy')

    # Initialize Hybrid MPC Controller
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # Initial state [x, y, theta, v, omega]
    u0 = np.array([0.0, 0.0, 0.0, 0.0])  # Initial control [tau_fr, tau_fl, tau_rr, tau_rl]
    state_cost_matrix = np.diag([2, 2, 9, 0.1, 0.1])
    control_cost_matrix = np.diag([0.1, 0.1, 0.1, 0.1])
    terminal_cost_matrix = np.diag([2, 2, 9, 0.1, 0.1])
    state_lower_bound = np.array([-10.0, -10.0, -3.14, -5.0, -3.14])
    state_upper_bound = np.array([10.0, 10.0, 3.14, 5.0, 3.14])
    control_lower_bound = np.array([-10.0, -10.0, -10.0, -10.0])
    control_upper_bound = np.array([10.0, 10.0, 10.0, 10.0])
    obstacles_positions = np.array([[100, 100], [100, 100], [100, 100]])  # Dummy obstacles far away
    obstacle_radii = np.array([0.5, 0.5, 0.5])
    safe_distance = 0.8
    N = 30
    sampling_time = 0.05

    hybrid_mpc = HybridMPCController(
        dnn_model, scaler_X, scaler_Y,
        x0=x0, u0=u0,
        state_cost_matrix=state_cost_matrix,
        control_cost_matrix=control_cost_matrix,
        terminal_cost_matrix=terminal_cost_matrix,
        state_lower_bound=state_lower_bound,
        state_upper_bound=state_upper_bound,
        control_lower_bound=control_lower_bound,
        control_upper_bound=control_upper_bound,
        obstacle_positions=obstacles_positions,
        obstacle_radii=obstacle_radii,
        safe_distance=safe_distance,
        N=N, dt=sampling_time, Ts=N*sampling_time,
        slack_weight=1e-2, cost_type='NONLINEAR_LS'
    )

    # Evaluate on Circle trajectory
    circle_trajectory = generate_trajectory("circle", duration=60, dt=sampling_time)
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.1], [0, 0, 0, 1])
    circle_errors = evaluate_controller(hybrid_mpc, circle_trajectory, robot_id, wheel_joints)

    # Evaluate on Lemniscate trajectory
    lemniscate_trajectory = generate_trajectory("lemniscate", duration=60, dt=sampling_time)
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.1], [0, 0, 0, 1])
    lemniscate_errors = evaluate_controller(hybrid_mpc, lemniscate_trajectory, robot_id, wheel_joints)

    p.disconnect()

    # Calculate and print metrics
    print("Circle Trajectory:")
    print(f"Mean Tracking Error: {np.mean(circle_errors):.4f}")
    print(f"Max Tracking Error: {np.max(circle_errors):.4f}")

    print("\nLemniscate Trajectory:")
    print(f"Mean Tracking Error: {np.mean(lemniscate_errors):.4f}")
    print(f"Max Tracking Error: {np.max(lemniscate_errors):.4f}")

if __name__ == "__main__":
    run_evaluation()