import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pybullet as p
import pybullet_data
import numpy as np
import time
from controllers.mpc_racecar_casadi import CasadiMPCController


if __name__ == "__main__":
    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load the plane and the race car
    plane_id = p.loadURDF("plane.urdf")
    start_pos = [0, 0, 0.1]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    car_id = p.loadURDF("/home/eroxii/ocp_ws/bullet3/examples/pybullet/gym/pybullet_data/racecar/racecar.urdf", start_pos, start_orientation)

    # Get the joint information
    num_joints = p.getNumJoints(car_id)
    steering_joints = []
    drive_joints = []

    for i in range(num_joints):
        joint_info = p.getJointInfo(car_id, i)
        joint_name = joint_info[1].decode('utf-8')

        if 'steering' in joint_name:
            steering_joints.append(i)
        elif joint_name == 'left_rear_wheel_joint' or joint_name == 'right_rear_wheel_joint' or \
            joint_name == 'left_front_wheel_joint' or joint_name == 'right_front_wheel_joint':
            drive_joints.append(i)

    # Set up the MPC controller
    dt = 0.1
    N = 20
    Q = np.diag([1, 1, 1, 1])
    R = np.diag([0.1, 0.1])
    state_bounds = np.array([[-10, 10], [-10, 10], [-np.pi, np.pi], [-5, 5]])
    control_bounds = np.array([[-1, 1], [-0.5, 0.5]])

    mpc_controller = CasadiMPCController(dt, N, Q, R, state_bounds, control_bounds)

    # Enable real-time simulation
    p.setRealTimeSimulation(1)

    # Simulation loop
    current_state = np.array([0, 0, 0, 0])
    target_state = np.array([5, 5, 0, 0])
    control_ref = np.array([0, 0])

    while True:
        # Get the current state of the race car
        pos, ori = p.getBasePositionAndOrientation(car_id)
        vel, ang_vel = p.getBaseVelocity(car_id)
        euler = p.getEulerFromQuaternion(ori)
        yaw = euler[2]
        current_state = np.array([pos[0], pos[1], yaw, vel[0]])

        # Solve the MPC problem
        optimal_control = mpc_controller.solve(current_state, target_state, control_ref)

        # Apply the optimal control to the race car
        target_acceleration = optimal_control[0]
        target_steering = optimal_control[1]

        # Apply the wheel velocities to the rear wheels
        for j in drive_joints:
            p.setJointMotorControl2(car_id, j, p.VELOCITY_CONTROL, targetVelocity=target_acceleration, force=10)

        # Apply the steering angles to the front wheels
        for j in steering_joints:
            p.setJointMotorControl2(car_id, j, p.POSITION_CONTROL, targetPosition=target_steering)

        print("Current state:", current_state)
        print("Optimal control:", optimal_control)

        # Step the simulation
        p.stepSimulation()
        time.sleep(dt)

    # Disconnect from PyBullet
    p.disconnect()