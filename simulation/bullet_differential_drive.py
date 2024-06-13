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

state_cost_matrix = np.diag([9.0, 6.0, 45])
control_cost_matrix = np.diag([1, 0.1])
terminal_cost_matrix = np.diag([9.0, 6.0, 45])

state_lower_bound = np.array([-100.0, -100.0, -3.14])
state_upper_bound = np.array([100.0, 100.0, 3.14])
control_lower_bound = np.array([-10, -3.14])
control_upper_bound = np.array([10, 3.14])

obstacle_positions = np.array([[3.0, 3.0], [6.5, 2.8], [6.0, 5.5]]) 
obstacle_radii = np.array([cube_size, cube_size, cube_size])
safe_distance = cube_size + 0.1

N = 100
dt = 0.01
Ts = 0.5    

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
num_joints = p.getNumJoints(robot_id)
wheel_joints = []
joints_list = ['front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel']

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

# Get the optimal state and control trajectories
simX = np.zeros((mpc.mpc.dims.N+1, mpc.mpc.dims.nx))
simU = np.zeros((mpc.mpc.dims.N, mpc.mpc.dims.nu))

while True:
    try:
        # Get the reference state
        state_ref = np.array([5, 10, 1.57])
        control_ref = np.array([4.0, 1.57])
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
        p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=v_front_left, maxVelocity=1.0)
        p.setJointMotorControl2(robot_id, 3, p.VELOCITY_CONTROL, targetVelocity=v_front_right, maxVelocity=1.0)
        p.setJointMotorControl2(robot_id, 4, p.VELOCITY_CONTROL, targetVelocity=v_rear_left, maxVelocity=1.0)
        p.setJointMotorControl2(robot_id, 5, p.VELOCITY_CONTROL, targetVelocity=v_rear_right, maxVelocity=1.0)


        # Step the simulation
        p.stepSimulation()

        # Update the cube's position for plotting
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id1)

        # Clear the plot and draw the cube at the new position

        # Sleep to maintain a consistent simulation rate
        time.sleep(dt)

    except Exception as e:
        print(f"Error in simulation loop: {e}")
        break

# Disconnect from PyBullet
p.disconnect()