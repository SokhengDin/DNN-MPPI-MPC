import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pybullet as p
import pybullet_data
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from pytorch_mppi import MPPI 

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from models.differentialSimV2 import DiffSimulation

def dynamics(states, actions, t=None):
    dt = 0.05  # time step
    x, y, theta = states[:, 0], states[:, 1], states[:, 2]

    if actions.dim() == 1:
        actions = actions.unsqueeze(0)

    v, omega = actions[:, 0], actions[:, 1]

    x_next = x + v * torch.cos(theta) * dt
    y_next = y + v * torch.sin(theta) * dt
    theta_next = theta + omega * dt

    next_states = torch.stack((x_next, y_next, theta_next), dim=1)
    return next_states

def running_cost(states, actions, desired_trajectory):
    Q = torch.diag(torch.tensor([20, 5, 9], dtype=torch.float32, device=states.device))
    R = torch.diag(torch.tensor([0.1, 0.1], dtype=torch.float32, device=states.device))

    # Reshape states and desired_trajectory to have compatible dimensions
    states = states.view(-1, states.shape[-1])
    desired_trajectory = desired_trajectory.view(-1, desired_trajectory.shape[-1])

    state_cost = torch.einsum('bi,ij,bj->b', states - desired_trajectory, Q, states - desired_trajectory)

    # Reshape actions to have compatible dimensions
    actions = actions.view(-1, actions.shape[-1])

    control_cost = torch.einsum('bi,ij,bj->b', actions, R, actions)

    obstacle_positions = torch.tensor([[3.0, 3.0], [6.5, 2.8], [6.0, 5.5], [-5.5, 5.5], [-5.5, -5.5], [5.5, -5.5]], dtype=torch.float32, device=states.device)
    obstacle_cost = torch.zeros(states.shape[0], dtype=torch.float32, device=states.device)

    safety_distance = cube_size + 0.1

    for obstacle in obstacle_positions:
        distance_to_obstacle = torch.norm(states[:, :2] - obstacle, dim=1)
        obstacle_cost += 1.0 / (distance_to_obstacle + 1e-6) * (distance_to_obstacle < safety_distance)

    total_cost = state_cost + control_cost + 10.0 * obstacle_cost

    # Reshape the total cost to match the original dimensions
    total_cost = total_cost.view(actions.shape[0], -1).mean(dim=-1)

    return total_cost

class MPPIWrapper(MPPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.desired_trajectory = None

    def command(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=self.dtype, device=self.d)
        self.state = state
        self.desired_trajectory = state.unsqueeze(0).expand(self.K, -1)  # Expand the state to match the number of samples
        action = super().command(state)
        return action

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total.repeat(self.M, 1)
        cost_var = torch.zeros_like(cost_total)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        # rollout action trajectory M times to estimate expected cost
        state = state.repeat(self.M, 1, 1)

        states = []
        actions = []
        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t].repeat(self.M, 1, 1)
            state = self._dynamics(state, u, t)  # Pass the time step to the _dynamics function
            c = self._running_cost(state, u, self.desired_trajectory.repeat(self.M, 1))  # Pass the desired trajectory to the running cost function
            cost_samples += c.view(self.M, -1).mean(dim=0)  # Compute the mean cost across rollouts
            if self.M > 1:
                cost_var += c.view(self.M, -1).var(dim=0) * (self.rollout_var_discount ** t)

            # Save total states/actions
            states.append(state)
            actions.append(u)

        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        states = torch.stack(states, dim=-2)

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples += c.view(self.M, -1).mean(dim=0)  # Compute the mean cost across rollouts
        cost_total = cost_samples.mean(dim=0)  # Compute the mean cost across samples
        cost_total += cost_var * self.rollout_var_cost
        return cost_total, states, actions

    def _running_cost(self, state, u, desired_trajectory):
        return self.running_cost(state, u, desired_trajectory)

    def get_trajectories(self, state):
        self.command(state)

        # Calculate optimal trajectory
        optimal_traj = torch.zeros((self.T, self.nx), device=self.d)
        x = state.unsqueeze(0)  # Add batch dimension
        for t in range(self.T):
            x = self._dynamics(x, self.u_scale * self.U[t].unsqueeze(0), t)  # Pass the time step to the _dynamics function
            optimal_traj[t] = x.squeeze(0)

        # Get top 10% of sampled trajectories
        num_top_samples = max(10, self.K // 10)
        sorted_idx = torch.argsort(self.cost_total)[:num_top_samples]
        sampled_traj_list = torch.zeros((num_top_samples, self.T, self.nx), device=self.d)
        for i, k in enumerate(sorted_idx):
            x = state.unsqueeze(0)
            for t in range(self.T):
                x = self._dynamics(x, self.u_scale * self.perturbed_action[k, t].unsqueeze(0), t)  # Pass the time step to the _dynamics function
                sampled_traj_list[i, t] = x.squeeze(0)

        return optimal_traj.cpu().numpy(), sampled_traj_list.cpu().numpy()

def inverse_kinematics(v, omega):
    # Distance between wheels (wheelbase)
    L = 0.5708  # meters

    # Calculate wheel velocities
    v_left = v - (omega * L / 2)
    v_right = v + (omega * L / 2)

    # Assign velocities to each wheel
    v_front_left = v_left
    v_rear_left = v_left
    v_front_right = v_right
    v_rear_right = v_right

    return v_front_left, v_front_right, v_rear_left, v_rear_right


def circle_trajectory(t, radius, center):
    x = radius * np.cos(t) + center[0]
    y = radius * np.sin(t) + center[1]
    yaw = t
    return np.array([x, y, yaw])

def lemniscate_trajectory(t, scale, center):
    x = scale * np.cos(t) / (1 + np.sin(t)**2) + center[0]
    y = scale * np.cos(t) * np.sin(t) / (1 + np.sin(t)**2) + center[1]
    yaw = np.arctan2(y - center[1], x - center[0])
    return np.array([x, y, yaw])

def collect_data(num_samples, mppi_ctrl, dt, robot_id, yref, yref_N, obstacle_positions):
    states = []
    controls = []
    errors = []

    for _ in range(num_samples):
        # Get the current state
        pos, ori = p.getBasePositionAndOrientation(robot_id)
        vel, ang_vel = p.getBaseVelocity(robot_id)
        euler = p.getEulerFromQuaternion(ori)
        yaw = euler[2]
        state_current = torch.tensor([pos[0], pos[1], yaw], device=mppi_ctrl.d)

        # Solve the MPPI problem
        action = mppi_ctrl.command(state_current)

        v = action[0].item()  # Linear velocity
        omega = action[1].item()  # Angular velocity

        # Calculate wheel velocities using inverse kinematics
        v_front_left, v_front_right, v_rear_left, v_rear_right = inverse_kinematics(v, omega)

        # Apply the control input to the Husky robot
        p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=v_front_left, maxVelocity=10.0)
        p.setJointMotorControl2(robot_id, 3, p.VELOCITY_CONTROL, targetVelocity=v_front_right, maxVelocity=10.0)
        p.setJointMotorControl2(robot_id, 4, p.VELOCITY_CONTROL, targetVelocity=v_rear_left, maxVelocity=10.0)
        p.setJointMotorControl2(robot_id, 5, p.VELOCITY_CONTROL, targetVelocity=v_rear_right, maxVelocity=10.0)

        # Step the simulation
        p.stepSimulation()

        # Collect data
        states.append(state_current.cpu().numpy())
        controls.append(action.cpu().numpy())

        # Re-simulate using the nominal controller
        nominal_pos, _ = p.getBasePositionAndOrientation(robot_id)
        nominal_state = torch.tensor([nominal_pos[0], nominal_pos[1], yaw], device=mppi_ctrl.d)
        error = state_current - nominal_state
        errors.append(error.cpu().numpy())

        yield state_current.cpu().numpy(), action.cpu().numpy()

    return np.array(states), np.array(controls), np.array(errors)


def collect_data_series(num_series, num_samples_per_series, mppi_ctrl, dt, robot_id, cube_ids, obstacle_positions, distance_threshold=0.1):
    all_states = []
    all_controls = []
    all_errors = []

    for i in range(num_series):
        trajectory_type = i % 3  # Alternate between different trajectory types

        if trajectory_type == 0:
            # Random reference state and control
            state_ref = torch.tensor(np.random.uniform(low=[-10, -10, -np.pi], high=[10, 10, np.pi]), device=mppi_ctrl.d)
            control_ref = torch.tensor(np.random.uniform(low=[-5, -np.pi/2], high=[5, np.pi/2]), device=mppi_ctrl.d)
            print("Random reference state and control:")
            print("State ref:", state_ref.clone().detach().cpu().numpy())
            print("Control ref:", control_ref.clone().detach().cpu().numpy())
        elif trajectory_type == 1:
            # Circle trajectory
            radius = np.random.uniform(low=5, high=10)
            center = np.random.uniform(low=[-5, -5], high=[5, 5])
            state_ref = torch.tensor(circle_trajectory(0, radius, center), device=mppi_ctrl.d)
            control_ref = torch.tensor([4.0, 1.57], device=mppi_ctrl.d)
            print("Circle trajectory:")
            print("State ref:", state_ref.clone().detach().cpu().numpy())
            print("Control ref:", control_ref.clone().detach().cpu().numpy())
        else:
            # Lemniscate trajectory
            scale = np.random.uniform(low=5, high=10)
            center = np.random.uniform(low=[-5, -5], high=[5, 5])
            state_ref = torch.tensor(lemniscate_trajectory(0, scale, center), device=mppi_ctrl.d)
            control_ref = torch.tensor([4.0, 1.57], device=mppi_ctrl.d)
            print("Lemniscate trajectory:")
            print("State ref:", state_ref.clone().detach().cpu().numpy())
            print("Control ref:", control_ref.clone().detach().cpu().numpy())

        yref = torch.cat([state_ref, control_ref])
        yref_N = state_ref  # Terminal state reference

        data_generator = collect_data(num_samples_per_series, mppi_ctrl, dt, robot_id, yref, yref_N, obstacle_positions)

        states = []
        controls = []
        errors = []

        for state_current, u in data_generator:
            states.append(state_current)
            controls.append(u)
            errors.append(state_current - state_ref.clone().detach().cpu().numpy())

            if np.linalg.norm(state_current[:2] - state_ref[:2].clone().detach().cpu().numpy()) < distance_threshold:
                break

        all_states.append(states)
        all_controls.append(controls)
        all_errors.append(errors)

    all_states = np.concatenate(all_states, axis=0)
    all_controls = np.concatenate(all_controls, axis=0)
    all_errors = np.concatenate(all_errors, axis=0)

    return all_states, all_controls, all_errors

# Connect to PyBullet with GPU acceleration
physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane and the robot URDF
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("/home/eroxii/ocp_ws/RL-MPPI-MPC/urdf/husky/husky_kine.urdf", [0, 0, 0.1])

# Load the cube object
cube_size = 0.5
cube_mass = 1.0
cube_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[cube_size]*3, rgbaColor=[1, 0, 0, 1])
cube_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[cube_size]*3)
cube_id1 = p.createMultiBody(baseMass=cube_mass, baseCollisionShapeIndex=cube_collision_shape_id, baseVisualShapeIndex=cube_visual_shape_id, basePosition=[3, 3, cube_size])
cube_id2 = p.createMultiBody(baseMass=cube_mass, baseCollisionShapeIndex=cube_collision_shape_id, baseVisualShapeIndex=cube_visual_shape_id, basePosition=[6.5, 2.8, cube_size])
cude_id3 = p.createMultiBody(baseMass=cube_mass, baseCollisionShapeIndex=cube_collision_shape_id, baseVisualShapeIndex=cube_visual_shape_id, basePosition=[5.5, 5.5, cube_size])
cube_id4 = p.createMultiBody(baseMass=cube_mass, baseCollisionShapeIndex=cube_collision_shape_id, baseVisualShapeIndex=cube_visual_shape_id, basePosition=[-5.5, 5.5, cube_size])
cube_id5 = p.createMultiBody(baseMass=cube_mass, baseCollisionShapeIndex=cube_collision_shape_id, baseVisualShapeIndex=cube_visual_shape_id, basePosition=[-5.5, -5.5, cube_size])
cube_id6 = p.createMultiBody(baseMass=cube_mass, baseCollisionShapeIndex=cube_collision_shape_id, baseVisualShapeIndex=cube_visual_shape_id, basePosition=[5.5, -5.5, cube_size])

# Enable GPU acceleration and set real-time simulation
p.setRealTimeSimulation(1)
p.setPhysicsEngineParameter(enableFileCaching=0, numSolverIterations=10, numSubSteps=2)
p.setPhysicsEngineParameter(enableConeFriction=1)
p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

# Start the simulation
p.setGravity(0, 0, -9.8)

# Simulation parameters
target_fps = 240
timestep = 1.0 / target_fps
num_iterations = 1000

# Set up the MPPI controller
device = torch.device("cuda")
nx = 3  # state dimension
nu = 2  # control dimension
robot = DiffSimulation()
num_samples = 50
horizon = 5
noise_sigma = torch.tensor([[0.5, 0.0], [0.0, 0.3]], device=device)
lambda_ = 1.0

u_min = torch.tensor([-10.0, -np.pi/2], device=device)
u_max = torch.tensor([10.0, np.pi/2], device=device)

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
    device=device,
    # terminal_state_cost=terminal_state_cost
)

# Get the joint indices for the wheel joints
num_joints = p.getNumJoints(robot_id)
wheel_joints = []
joints_list = ['front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel']

output_dir = "saved_data"

os.makedirs(output_dir, exist_ok=True)

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_name = joint_info[1].decode('utf-8')

    if joint_name in joints_list:
        wheel_joints.append(i)
        print(joint_name)

print("Wheel joints:", wheel_joints)

# Collect data for system identification
num_series = 50
num_samples_per_series = 100
cube_ids = [cube_id1, cube_id2, cude_id3, cube_id4, cube_id5, cube_id6]  # Add the cube IDs to a list
states, controls, errors = collect_data_series(num_series, num_samples_per_series, mppi_ctrl, timestep, robot_id, cube_ids, obstacle_positions=0.2)

# Save the collected data to files
np.save(os.path.join(output_dir, "states_diff_mppi.npy"), states)
np.save(os.path.join(output_dir, "controls_diff_mppi.npy"), controls)
np.save(os.path.join(output_dir, "errors_diff_mppi.npy"), errors)

# Disconnect from PyBullet
p.disconnect()