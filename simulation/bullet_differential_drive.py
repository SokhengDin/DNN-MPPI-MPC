import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import casadi as ca
import numpy as np
import torch
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import l4casadi as l4c
import matplotlib.animation as animation

from scipy.linalg import block_diag
from torchvision import models
from acados_template import AcadosModel
from controllers.mpc_differential_drive_obstacle_static import MPCController
from models.differentialSimV2 import DiffSimulation

def inverse_kinematics(v, omega):

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
    
def plot_control_input(v, omega, output_path):

    # Create an iteration array
    iterations = range(len(v))

    # Create a new figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot linear velocity
    ax1.plot(iterations, v, linewidth=2)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Linear Velocity (m/s)', fontsize=12)
    ax1.set_title('Control Input: Linear Velocity', fontsize=14)
    ax1.grid(True)

    # Plot angular velocity
    ax2.plot(iterations, omega, linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax2.set_title('Control Input: Angular Velocity', fontsize=14)
    ax2.grid(True)

    # Adjust spacing between subplots
    fig.tight_layout(pad=3.0)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def deaccelerate_control(v, omega, v_max, omega_max, t):
    v = v_max * (1-np.exp(-100*t))
    omega = omega_max * (1-np.exp(-100*t))

    return v, omega

def plot_state_errors(xs, yref_N):
    # Extract the final state from the trajectory
    final_state = xs[-1]

    # Calculate the state errors
    state_errors = np.abs(final_state - yref_N[:3])

    # Create labels for the states
    state_labels = ['X', 'Y', 'Yaw']

    # Set custom colors for the bars
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Create a figure and axis with larger size
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the state errors as a bar plot with custom colors
    bars = ax.bar(state_labels, state_errors, color=colors)

    # Add labels and title with larger font sizes
    ax.set_xlabel('State', fontsize=16)
    ax.set_ylabel('Error', fontsize=16)
    ax.set_title('State Errors', fontsize=20)

    # Increase the font size of the tick labels
    ax.tick_params(axis='both', labelsize=14)

    # Add value labels to the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12)

    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust the layout and display the plot
    fig.tight_layout()
    plt.show()

    # Save the plot with higher resolution
    fig.savefig('state_errors.png', dpi=600, bbox_inches='tight')

def animate(i, xs, us, simX_history, cube_ids, ax, differential_robot, cube_size, yref_N, safe_distance):
    # Clear the previous plot
    ax.clear()

    # Plot robot frame
    differential_robot.generate_each_wheel_and_draw(ax, xs[i][0], xs[i][1], xs[i][2])

    # Plot the goal point
    ax.plot(yref_N[0], yref_N[1], color='red', marker='o', ms=10)

    # Plot cube
    for cube_id in cube_ids:
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        obstacle = plt.Rectangle((cube_pos[0] - cube_size, cube_pos[1] - cube_size), 2 * cube_size, 2 * cube_size, color='blue')
        ax.add_patch(obstacle)
    
    # the robot trajectory
    xs_array = np.array(xs[:i+1])
    ax.plot(xs_array[:, 0], xs_array[:, 1], 'r-', linewidth=1.5)

    # Plot the prediction states
    if i < len(simX_history):
        simX = simX_history[i]
        ax.plot(simX[:, 0], simX[:, 1], 'g--', linewidth=1.5)

    # Plot the safety boundary circle
    ax.add_patch(plt.Circle((xs[i][0], xs[i][1]), safe_distance, edgecolor='k', linestyle='--', facecolor='none', zorder=10))

    # Add a popup window on the robot
    popup_text = f"Position: ({xs[i][0]:.2f}, {xs[i][1]:.2f})\nYaw: {xs[i][2]:.2f}"
    ax.text(8, 10, popup_text, fontsize=10, ha='center', va='bottom',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    
    # Popup window for the reference state
    popup_text_ref = f"Position Ref: ({yref_N[0]:.2f}, {yref_N[1]:.2f})\nYaw: {yref_N[2]:.2f}"
    ax.text(-7, 10, popup_text_ref, fontsize=10, ha='center', va='bottom',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    
    # Set plot limits and labels
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Differential Drive Robot Simulation')

def run():
    # Init states
    state_init = np.array([0.0, 0.0, 0.0])
    control_init = np.array([0.0, 0.0])

    # Current states
    state_current = state_init
    control_current = control_init

    # Cost matrices
    state_cost_matrix = 1*np.diag([2, 2, 9])
    control_cost_matrix = np.diag([0.1, 0.01])
    terminal_cost_matrix = 2*np.diag([2, 2, 9])

    ## Constraints
    state_lower_bound = np.array([-10.0, -10.0, -3.14])
    state_upper_bound = np.array([10.0, 10.0, 3.14])
    control_lower_bound = np.array([-10.0, -3.14])
    control_upper_bound = np.array([10.0, 3.14])

    # Obstacle
    obstacles_positions = np.array([
        [5.0, -5.0],
        [-3.0, -4.5],
        [5.0, 3.0]
    ])

    obstacle_velocities = 0.4*np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 5.0]
    ])

    obstacle_radii = np.array([0.5, 0.5, 0.5])
    safe_distance = 0.5

    # Simulation 
    simulation = DiffSimulation()

    # MPC params
    N = 25
    sampling_time = 0.05
    Ts = N * sampling_time
    Tsim = int(N / sampling_time)
    Tsim = 10000

    # Track history 
    xs = [state_init.copy()]
    us = []
    v_history = []
    omega_history = []
    simX_history = []


    solver = MPCController(
        x0=state_init,
        u0=control_init,
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
        N=N,
        dt=sampling_time,
        Ts=Ts,
        external_shared_lib_dir=None,
        external_shared_lib_name=None,
        model=None,
        is_dnn=False,
        cost_type='NONLINEAR_LS'
    )

    # # Connect to the PyBullet server
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    # Load urddf
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("/home/eroxii/ocp_ws/RL-MPPI-MPC/urdf/husky/husky_kine.urdf", [0, 0, 0.1])

    camera_distance = 10
    camera_yaw = 45
    camera_pitch = -60
    camera_target_position = [0, 0, 0]
    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

    # # Set gravity
    p.setGravity(0, 0, -9.81)

    # Load the cube object
    cube_size = 0.5
    cube_mass = 1.0
    cube_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[cube_size]*3, rgbaColor=[1, 0, 0, 1])
    cube_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[cube_size]*3)
    cube_id1 = p.createMultiBody(baseMass=cube_mass, baseCollisionShapeIndex=cube_collision_shape_id,
                                baseVisualShapeIndex=cube_visual_shape_id,
                                basePosition=[obstacles_positions[0][0],obstacles_positions[0][1], cube_size])
    cube_id2 = p.createMultiBody(baseMass=cube_mass, baseCollisionShapeIndex=cube_collision_shape_id,
                                baseVisualShapeIndex=cube_visual_shape_id, 
                                basePosition=[obstacles_positions[1][0],obstacles_positions[1][1], cube_size])
    cude_id3 = p.createMultiBody(baseMass=cube_mass, baseCollisionShapeIndex=cube_collision_shape_id,
                                baseVisualShapeIndex=cube_visual_shape_id, 
                                basePosition=[obstacles_positions[2][0],obstacles_positions[2][1], cube_size])

    cube_ids = [cube_id1, cube_id2, cude_id3]

    # Set the fps
    target_fps = 240
    timestep = 1 / target_fps
    num_iterations = 1000

    # Enable GPU
    p.setRealTimeSimulation(1)
    p.setPhysicsEngineParameter(enableFileCaching=0, numSolverIterations=10, numSubSteps=2)
    p.setPhysicsEngineParameter(enableConeFriction=1)
    p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)


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

    simX = np.zeros((solver.mpc.dims.N+1, solver.mpc.dims.nx))
    simU = np.zeros((solver.mpc.dims.N, solver.mpc.dims.nu))

    # Target position
    yref_N = np.array([3.0, 6.0, 1.57, 2.0, 0.0])

    obstacle_velocities = []
    for position in obstacles_positions:
        direction = yref_N[:2] - position
        direction_norm = np.linalg.norm(direction)
        if direction_norm != 0:
            velocity = 0.1 * direction / direction_norm
        else:
            velocity = np.array([0.0, 0.0])
        obstacle_velocities.append(velocity)

    obstacle_velocities = np.array(obstacle_velocities)

    # Prepare simulation 
    fig, ax = plt.subplots(figsize=(10, 10))
    differential_robot = DiffSimulation()

    p.setTimeStep(timestep)

    # log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "diff_mpc_bullet.mp4")

    for i in range(Tsim):
        # Get robot current state
        pos, ori = p.getBasePositionAndOrientation(robot_id)
        vel, ang_vel = p.getBaseVelocity(robot_id)
        euler = p.getEulerFromQuaternion(ori)
        yaw = euler[2]

        state_current = np.array([pos[0], pos[1], yaw])

        print(f"Current state: {state_current}")

        obstacles_positions[:] += obstacle_velocities * sampling_time

        for j, cube_id in enumerate(cube_ids):
            p.resetBasePositionAndOrientation(cube_id, [obstacles_positions[j][0], obstacles_positions[j][1], cube_size], [0, 0, 0, 1])

        # solve mpc
        simX, simU = solver.solve_mpc(state_current, simX, simU, yref_N, yref_N[:3], obstacles_positions)

        # Get the control inputs
        u = simU[0, :]
        v = u[0]
        omega = u[1]

        # v, omega = deaccelerate_control(v, omega, control_upper_bound[0], control_upper_bound[1], i)

        # Apply the control inputs
        v_history.append(v)
        omega_history.append(omega)

        v_front_left, v_front_right, v_rear_left, v_rear_right = inverse_kinematics(v, omega)

        # Apply to each joint
        p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=v_front_left, maxVelocity=100.0)
        p.setJointMotorControl2(robot_id, 3, p.VELOCITY_CONTROL, targetVelocity=v_front_right, maxVelocity=100.0)
        p.setJointMotorControl2(robot_id, 4, p.VELOCITY_CONTROL, targetVelocity=v_rear_left, maxVelocity=100.0)
        p.setJointMotorControl2(robot_id, 5, p.VELOCITY_CONTROL, targetVelocity=v_rear_right, maxVelocity=100.0)

        # Update the history
        xs.append(state_current)
        us.append(u)
        simX_history.append(simX.copy())

        # Print Tsim
        print(f"Tsim: {i}")
    
        # Step the simulation
        p.stepSimulation()
    

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, frames=len(xs), fargs=(xs, us, simX_history, cube_ids, ax, differential_robot, cube_size, yref_N, safe_distance), interval=1000, blit=False)
    # p.stopStateLogging(log_id)

    # Save the animation as a video
    ani.save('diff_mpc.mp4', writer='ffmpeg', fps=30)

    plot_control_input(v_history, omega_history, 'diff_mpc.png')
    plot_state_errors(xs, yref_N)

    # Close the figure
    plt.close(fig)

    p.disconnect()

if __name__ == "__main__":
    run()