import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pybullet as p
import pybullet_data
import numpy as np
import time
from controllers.mpc_differential_drive_obstacle_dynamic import MPCController
from typing import Tuple

def plot_arrow(x, y, yaw, length=0.5, width=0.1, fc="b", ec="k"):
    p.addUserDebugLine(
        lineFromXYZ=[x, y, 0.1], 
        lineToXYZ=[x + length * np.cos(yaw), y + length * np.sin(yaw), 0.1],
        lineColorRGB=[1, 0, 0],
        lineWidth=3
    )

def inverse_kinematics(v, omega):
    # Wheel base distances
    wheel_base_x = 0.512
    wheel_base_y = 0.5708

    # Calculate wheel velocities
    v_front_left = v + omega * (wheel_base_x + wheel_base_y) / 2
    v_front_right = v - omega * (wheel_base_x - wheel_base_y) / 2
    v_rear_left = v + omega * (wheel_base_x + wheel_base_y) / 2
    v_rear_right = v - omega * (wheel_base_x - wheel_base_y) / 2

    return v_front_left, v_front_right, v_rear_left, v_rear_right


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
cube_id = p.createMultiBody(baseMass=cube_mass, baseCollisionShapeIndex=cube_collision_shape_id, baseVisualShapeIndex=cube_visual_shape_id, basePosition=[3, 0, cube_size])

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

state_cost_matrix = np.diag([750.0, 750.0, 900])
control_cost_matrix = np.diag([1, 0.1])
terminal_cost_matrix = np.diag([750.0, 750.0, 900])

state_lower_bound = np.array([-10.0, -10.0, -3.14])
state_upper_bound = np.array([10.0, 10.0, 3.14])
control_lower_bound = np.array([-10, -3.14])
control_upper_bound = np.array([10, 3.14])

obstacle_positions = np.array([[-10.0, 0.0]])  # Cube position
obstacle_radii = np.array([cube_size])
safe_distance = 0.5

N = 10
dt = 0.01
Ts = 1.0

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
    obstacle_positions=obstacle_positions,
    obstacle_radii=obstacle_radii,
    safe_distance=safe_distance,
    N=N,
    dt=timestep,
    Ts=Ts,
    cost_type='NONLINEAR_LS'
)

# Get the joint indices for the wheel joints
joint_indices = [
    p.getJointInfo(robot_id, i)[0] for i in range(p.getNumJoints(robot_id))
    if "wheel" in str(p.getJointInfo(robot_id, i)[1])
]

# Simulation loop
state_current = state_init
control_current = control_init

# Get the optimal state and control trajectories
simX = np.zeros((mpc.mpc.dims.N+1, mpc.mpc.dims.nx))
simU = np.zeros((mpc.mpc.dims.N, mpc.mpc.dims.nu))

while True:
    try:
        # Get the reference state
        state_ref = np.array([-3, 2.0, 0.0])
        control_ref = np.array([2.0, 0.0])
        yref = np.concatenate([state_ref, control_ref])
        yref_N = state_ref  # Terminal state reference

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

        print("Control input:", u, "State:", state_current)


        # Calculate wheel velocities using inverse kinematics
        v_front_left, v_front_right, v_rear_left, v_rear_right = inverse_kinematics(v, omega)

        # Apply the control input to the Husky robot
        p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=v_front_left, maxVelocity=10.0)
        p.setJointMotorControl2(robot_id, 3, p.VELOCITY_CONTROL, targetVelocity=v_front_right, maxVelocity=10.0)
        p.setJointMotorControl2(robot_id, 4, p.VELOCITY_CONTROL, targetVelocity=v_rear_left, maxVelocity=10.0)
        p.setJointMotorControl2(robot_id, 5, p.VELOCITY_CONTROL, targetVelocity=v_rear_right, maxVelocity=10.0)


        # Step the simulation
        p.stepSimulation()

        # Sleep to maintain a consistent simulation rate
        time.sleep(dt)

    except Exception as e:
        print(f"Error in simulation loop: {e}")
        break

# Disconnect from PyBullet
p.disconnect()