import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt
from controllers.mpc_differential_drive_obstacle_dynamic import MPCController
from typing import Tuple
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from models.differentialSimV2 import DiffSimulation

def plot_arrow(x, y, yaw, length=0.5, width=0.1, fc="b", ec="k"):
    p.addUserDebugLine(
        lineFromXYZ=[x, y, 0.1], 
        lineToXYZ=[x + length * np.cos(yaw), y + length * np.sin(yaw), 0.1],
        lineColorRGB=[1, 0, 0],
        lineWidth=3
    )

# Function to plot the cube
def plot_cube(ax, position, size):
    r = [-size/2, size/2]
    X, Y = np.meshgrid(r, r)
    ax.plot_surface(X + position[0], Y + position[1], size/2 + position[2], color='red')
    ax.plot_surface(X + position[0], Y + position[1], -size/2 + position[2], color='red')
    ax.plot_surface(X + position[0], size/2 + position[1], Y + position[2], color='red')
    ax.plot_surface(X + position[0], -size/2 + position[1], Y + position[2], color='red')
    ax.plot_surface(size/2 + position[0], X + position[1], Y + position[2], color='red')
    ax.plot_surface(-size/2 + position[0], X + position[1], Y + position[2], color='red')


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


def collect_data(num_samples, mpc, dt, robot_id, yref, yref_N, obstacle_positions):
    states = []
    controls = []
    errors = []

    # Initialize simX and simU
    simX = np.zeros((mpc.mpc.dims.N+1, mpc.mpc.dims.nx))
    simU = np.zeros((mpc.mpc.dims.N, mpc.mpc.dims.nu))

    for _ in range(num_samples):
        # Get the current state
        pos, ori = p.getBasePositionAndOrientation(robot_id)
        vel, ang_vel = p.getBaseVelocity(robot_id)
        euler = p.getEulerFromQuaternion(ori)
        yaw = euler[2]
        state_current = np.array([pos[0], pos[1], yaw])

        # Solve the MPC problem
        simX, simU = mpc.solve_mpc(state_current, simX, simU, yref, yref_N, obstacle_positions)

        # Get the optimal control input
        u = simU[0, :]
        v = u[0]  # Linear velocity
        omega = u[1]  # Angular velocity

        # Calculate wheel velocities using inverse kinematics
        v_front_left, v_front_right, v_rear_left, v_rear_right = inverse_kinematics(v, omega)

        # Debug information
        # print("State:", state_current)
        # print("Control:", u)

        # Apply the control input to the Husky robot
        p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=v_front_left, maxVelocity=1.0)
        p.setJointMotorControl2(robot_id, 3, p.VELOCITY_CONTROL, targetVelocity=v_front_right, maxVelocity=1.0)
        p.setJointMotorControl2(robot_id, 4, p.VELOCITY_CONTROL, targetVelocity=v_rear_left, maxVelocity=1.0)
        p.setJointMotorControl2(robot_id, 5, p.VELOCITY_CONTROL, targetVelocity=v_rear_right, maxVelocity=1.0)

        # Step the simulation
        p.stepSimulation()

        # Collect data
        states.append(state_current)
        controls.append(u)

        # Re-simulate using the nominal controller
        nominal_pos, _ = p.getBasePositionAndOrientation(robot_id)
        nominal_state = np.array([nominal_pos[0], nominal_pos[1], yaw])
        error = state_current - nominal_state
        errors.append(error)

        yield state_current, u, simX, simU

        # Sleep to maintain a consistent simulation rate
        time.sleep(dt)

    return np.array(states), np.array(controls), np.array(errors)


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

def collect_data_series(num_series, num_samples_per_series, mpc, dt, robot_id, cube_ids, obstacle_positions, distance_threshold=0.1):
    all_states = []
    all_controls = []
    all_errors = []

    # Set up figure
    fig, ax = plt.subplots(figsize=(7, 7))
    differential_robot = DiffSimulation()

    for i in range(num_series):
        trajectory_type = i % 3  # Alternate between different trajectory types

        if trajectory_type == 0:
            # Random reference state and control
            state_ref = np.random.uniform(low=[-10, -10, -np.pi], high=[10, 10, np.pi])
            control_ref = np.random.uniform(low=[-5, -np.pi/2], high=[5, np.pi/2])
            print("Random reference state and control:")
            print("State ref:", state_ref)
            print("Control ref:", control_ref)
        elif trajectory_type == 1:
            # Circle trajectory
            radius = np.random.uniform(low=5, high=10)
            center = np.random.uniform(low=[-5, -5], high=[5, 5])
            state_ref = circle_trajectory(0, radius, center)
            control_ref = np.array([4.0, 1.57])
            print("Circle trajectory:")
            print("State ref:", state_ref)
            print("Control ref:", control_ref)
        else:
            # Lemniscate trajectory
            scale = np.random.uniform(low=5, high=10)
            center = np.random.uniform(low=[-5, -5], high=[5, 5])
            state_ref = lemniscate_trajectory(0, scale, center)
            control_ref = np.array([4.0, 1.57])
            print("Lemniscate trajectory:")
            print("State ref:", state_ref)
            print("Control ref:", control_ref)

        yref = np.concatenate([state_ref, control_ref])
        yref_N = state_ref  # Terminal state reference

        data_generator = collect_data(num_samples_per_series, mpc, dt, robot_id, yref, yref_N, obstacle_positions)

        states = []
        controls = []
        errors = []

        for state_current, u, simX, simU in data_generator:
            states.append(state_current)
            controls.append(u)
            errors.append(state_current - state_ref)

            # Clear the previous plot
            ax.clear()

            # Plot the robot frame
            differential_robot.generate_each_wheel_and_draw(ax, state_current[0], state_current[1], state_current[2])

            # Plot the goal point
            ax.plot(state_ref[0], state_ref[1], 'ro', markersize=10)

            for cube_id in cube_ids:
                cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
                obstacle = plt.Circle(cube_pos[:2], cube_size, color='b', alpha=0.7)
                ax.add_patch(obstacle)

            # Plot the robot's trajectory
            xs_array = np.array(states)
            ax.plot(xs_array[:, 0], xs_array[:, 1], 'r-', linewidth=1.5)

            # Plot predicting states
            ax.plot(simX[:, 0], simX[:, 1], 'g--', linewidth=1.5)

            # Add a popup window on the robot
            popup_text = f"Position: ({state_current[0]:.2f}, {state_current[1]:.2f})\nYaw: {state_current[2]:.2f}"
            popup = ax.text(8, 10, popup_text, fontsize=10, ha='center', va='bottom',
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            
            # Popup window for the reference state
            popup_text_ref = f"Position Ref: ({state_ref[0]:.2f}, {state_ref[1]:.2f})\nYaw: {state_ref[2]:.2f}"
            popup_ref = ax.text(-7, 10, popup_text_ref, fontsize=10, ha='center', va='bottom',
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
            

            # Set plot limits and labels
            ax.set_xlim(-12, 12)
            ax.set_ylim(-12, 12)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Differential Drive Robot Frame Animation')

            # Pause to update the plot
            plt.pause(0.01)

            if np.linalg.norm(state_current - state_ref) < distance_threshold:
                break

        all_states.append(states)
        all_controls.append(controls)
        all_errors.append(errors)

    all_states = np.concatenate(all_states, axis=0)
    all_controls = np.concatenate(all_controls, axis=0)
    all_errors = np.concatenate(all_errors, axis=0)

    return all_states, all_controls, all_errors

# Connect to PyBullet with GPU acceleration
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane and the robot URDF
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("/home/eroxii/ocp_ws/bullet3/data/husky/husky.urdf", [0, 0, 0.1])

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

# Set up the NMPC controller
state_init = np.array([0.0, 0.0, 0.0])
control_init = np.array([0.0, 0.0])

state_cost_matrix = np.diag([25.0, 20.0, 45])
control_cost_matrix = np.diag([1, 1])
terminal_cost_matrix = np.diag([25.0, 20.0, 45])

state_lower_bound = np.array([-15.0, -15.0, -3.14])
state_upper_bound = np.array([15.0, 15.0, 3.14])
control_lower_bound = np.array([-10, -31.4])
control_upper_bound = np.array([10, 31.4])

obstacle_positions = np.array([[3.0, 3.0], [6.5, 2.8], [6.0, 5.5]]) 
obstacle_radii = np.array([cube_size, cube_size, cube_size])
safe_distance = cube_size + 0.1

N = 100
dt = 0.01
Ts = 3.0

mpc = MPCController(
    x0=state_init,
    u0=control_init,
    state_cost_matrix=state_cost_matrix,
    control_cost_matrix=control_cost_matrix,
    terminal_cost_matrix=terminal_cost_matrix,
    state_lower_bound=state_lower_bound,
    state_upper_bound=state_upper_bound,
    control_lower_bound=control_lower_bound,
    control_upper_bound=control_upper_bound,
    obstacle_positions=[],
    obstacle_radii=[],
    safe_distance=safe_distance,
    N=N,
    dt=timestep,
    Ts=Ts,
    cost_type='NONLINEAR_LS'
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

# Simulation loop
state_current = state_init
control_current = control_init


# Get the reference state and control
state_ref = np.array([5, 10, 1.57])
control_ref = np.array([4.0, 1.57])
yref = np.concatenate([state_ref, control_ref])
yref_N = state_ref  # Terminal state reference

# Collect data for system identification
num_series = 1000
num_samples_per_series = 200
cube_ids = [cube_id1, cube_id2, cude_id3, cube_id4, cube_id5, cube_id6]  # Add the cube IDs to a list
states, controls, errors = collect_data_series(num_series, num_samples_per_series, mpc, timestep, robot_id, cube_ids, 0.3)

# Save the collected data to files
np.save(os.path.join(output_dir, "states_diff.npy"), states)
np.save(os.path.join(output_dir, "controls_diff.npy"), controls)
np.save(os.path.join(output_dir, "errors_diff.npy"), errors)
# Disconnect from PyBullet
p.disconnect()