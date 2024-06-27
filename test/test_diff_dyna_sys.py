import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pybullet as p
import pybullet_data
import time
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import block_diag
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from test_diff_mpc_dyna import MPCController

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

# Data Collection Functions
def generate_trajectory(trajectory_type, duration, dt):
    t = np.arange(0, duration, dt)
    if trajectory_type == "circle":
        radius = 3.0
        omega = 0.5
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        theta = omega * t + np.pi/2
    elif trajectory_type == "lemniscate":
        a = 3.0
        x = a * np.cos(t) / (1 + np.sin(t)**2)
        y = a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
        theta = np.arctan2(np.cos(t), -np.sin(t))
    elif trajectory_type == "random":
        x = np.cumsum(np.random.randn(len(t))) * 0.1
        y = np.cumsum(np.random.randn(len(t))) * 0.1
        theta = np.cumsum(np.random.randn(len(t))) * 0.05
    else:
        raise ValueError("Unknown trajectory type")
    
    v = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / dt
    v = np.append(v, v[-1])
    omega = np.diff(theta) / dt
    omega = np.append(omega, omega[-1])
    
    return np.column_stack((x, y, theta, v, omega))

def collect_data(num_trajectories, duration, dt):
    # PyBullet setup
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Enable debug visualization
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    p.setTimeStep(dt)
    p.setPhysicsEngineParameter(fixedTimeStep=dt, numSolverIterations=50)
    p.setRealTimeSimulation(0) 
    
    # Load plane and robot
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("/home/eroxii/ocp_ws/bullet3/data/husky/husky.urdf", [0, 0, 0.1])

    # Set ground friction
    p.changeDynamics(plane_id, -1, lateralFriction=0.8, spinningFriction=0.1, rollingFriction=0.1)

    # Identify wheel joints
    wheel_joints = []
    joint_names = ['front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel']
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        if joint_info[1].decode('utf-8') in joint_names:
            wheel_joints.append(i)

    print("Identified wheel joints:", wheel_joints)

    # Set up wheel joints for torque control
    # for joint in wheel_joints:
    #     p.setJointMotorControl2(robot_id, joint, p.VELOCITY_CONTROL, force=0)
    #     p.enableJointForceTorqueSensor(robot_id, joint, enableSensor=1)

    # MPC setup
    x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    u0 = np.array([0.0, 0.0, 0.0, 0.0])  
    state_cost_matrix = np.diag([100, 100, 10, 1, 1]) 
    control_cost_matrix = np.diag([0.01, 0.01, 0.01, 0.01])  
    terminal_cost_matrix = 2*state_cost_matrix  

    state_lower_bound = np.array([-10.0, -10.0, -3.14, -5.0, -3.14])
    state_upper_bound = np.array([10.0, 10.0, 3.14, 5.0, 3.14])
    control_lower_bound = np.array([-100.0, -100.0, -100.0, -100.0]) 
    control_upper_bound = np.array([100.0, 100.0, 100.0, 100.0])
    obstacles_positions = np.array([[100, 100], [100, 100], [100, 100]])  
    obstacle_radii = np.array([0.5, 0.5, 0.5])
    safe_distance = 0.8
    
    N = 20
    sampling_time = 0.1

    mpc = MPCController(
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
        slack_weight=1000.0, cost_type='NONLINEAR_LS'
    )

    # Data collection
    dataset = []
    trajectory_types = ["circle", "lemniscate", "random"]
    
    for trajectory_num in range(num_trajectories):
        trajectory_type = np.random.choice(trajectory_types)
        trajectory = generate_trajectory(trajectory_type, duration, dt)

        print(f"Trajectory {trajectory_num + 1}/{num_trajectories} - Type: {trajectory_type}")
        
        # Reset robot position
        p.resetBasePositionAndOrientation(robot_id, [0, 0, 0.1], [0, 0, 0, 1])
        
        simX = np.zeros((mpc.N + 1, mpc.model.x.size()[0]))
        simU = np.zeros((mpc.N, mpc.model.u.size()[0]))

        for i in range(len(trajectory) - 1):
            # Get current state
            pos, ori = p.getBasePositionAndOrientation(robot_id)
            lin_vel, ang_vel = p.getBaseVelocity(robot_id)
            euler = p.getEulerFromQuaternion(ori)
            current_state = np.array([pos[0], pos[1], euler[2], np.linalg.norm(lin_vel[:2]), ang_vel[2]])

            # Set reference
            state_target = trajectory[i]
            control_target = np.zeros(4)
            yref = np.concatenate([state_target, control_target])
            yref_N = trajectory[i] 

            # Solve MPC
            simX, simU = mpc.solve_mpc(current_state, simX, simU, yref, yref_N, obstacles_positions)

            # Extract control inputs
            control_input = simU[0]
            tau_fl, tau_fr, tau_rl, tau_rr = control_input

            # Apply control to the robot
            joint_torques = [tau_fl, tau_fr, tau_rl, tau_rr]
            for joint, torque in zip(wheel_joints, joint_torques):
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint,
                    controlMode=p.TORQUE_CONTROL,
                    force=torque
                )
                
                # Print applied torque for debugging
                print(f"Applied torque {torque} to joint {p.getJointInfo(robot_id, joint)[1].decode('utf-8')}")

            # Predict next state using nominal model
            predicted_next_state = mpc.update_state(current_state, control_input)

            p.stepSimulation()

            # Get actual next state
            next_pos, next_ori = p.getBasePositionAndOrientation(robot_id)
            next_lin_vel, next_ang_vel = p.getBaseVelocity(robot_id)
            next_euler = p.getEulerFromQuaternion(next_ori)
            actual_next_state = np.array([next_pos[0], next_pos[1], next_euler[2], np.linalg.norm(next_lin_vel[:2]), next_ang_vel[2]])

            # Calculate error
            error = actual_next_state - predicted_next_state

            # Print debug information
            print(f"Step {i + 1}/{len(trajectory) - 1}")
            print(f"Target state: {state_target}")
            print(f"Current state: {current_state}")
            print(f"Control input (torques): {control_input}")
            print(f"Predicted next state: {predicted_next_state}")
            print(f"Actual next state: {actual_next_state}")
            print(f"Error: {error}")
            print("---")

            # Store data
            dataset.append({
                'current_state': current_state,
                'control_input': control_input,
                'error': error
            })

            # Optional: add a small delay to visualize in real-time
            time.sleep(dt)

    p.disconnect()
    return dataset

# DNN Model and Training Functions
class ResidualDynamicsNet(nn.Module):
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

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

    return model

# Main execution
if __name__ == "__main__":
    # Data Collection
    num_trajectories = 100
    duration = 60  # seconds
    dt = 0.05  # 20 Hz

    print("Collecting data...")
    dataset = collect_data(num_trajectories, duration, dt)
    print(f"Collected {len(dataset)} data points")

    # Prepare data for training
    X = np.array([np.concatenate([d['current_state'], d['control_input']]) for d in dataset])
    Y = np.array([d['error'] for d in dataset])

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    # Scale data
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    X_val_scaled = scaler_X.transform(X_val)
    Y_val_scaled = scaler_Y.transform(Y_val)

    # Prepare PyTorch datasets and dataloaders
    train_data = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(Y_train_scaled))
    val_data = torch.utils.data.TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(Y_val_scaled))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64)

    # Initialize and train model
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    model = ResidualDynamicsNet(input_dim, output_dim)

    print("Training DNN model...")
    trained_model = train_model(model, train_loader, val_loader)

    # Save model and scalers
    torch.save(trained_model.state_dict(), 'residual_dynamics_model.pth')
    np.save('scaler_X_mean.npy', scaler_X.mean_)
    np.save('scaler_X_scale.npy', scaler_X.scale_)
    np.save('scaler_Y_mean.npy', scaler_Y.mean_)
    np.save('scaler_Y_scale.npy', scaler_Y.scale_)

    print("System identification complete. Model and scalers saved.")