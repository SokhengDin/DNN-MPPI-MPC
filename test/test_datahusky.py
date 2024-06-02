import pybullet as p
import pybullet_data
import numpy as np
import torch
import casadi as cs
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
from pyDOE import lhs
import time

# Define the input and output dimensions
state_dim = 13  # position (3) + orientation (4) + linear_velocity (3) + angular_velocity (3)
control_dim = 2  # left and right wheel velocities
input_dim = state_dim + control_dim  # Total input dimension is state_dim + control_dim
output_dim = state_dim  # Output dimension is the state dimension

# Initialize the PyBullet simulation
def initialize_simulation():
    p.connect(p.DIRECT)  # Use DIRECT for faster collection without GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    husky = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])
    return husky

# Collect state-control data from the simulation
def collect_data(husky, num_episodes=100, episode_length=1000):
    data = []
    for episode in range(num_episodes):
        # Reset the robot to a random initial state
        initial_position = np.random.uniform(-1, 1, size=3)
        initial_orientation = np.random.uniform(-1, 1, size=4)
        initial_orientation /= np.linalg.norm(initial_orientation)  # Normalize orientation quaternion
        p.resetBasePositionAndOrientation(husky, initial_position, initial_orientation)
        
        for step in range(episode_length):
            # Use Latin Hypercube Sampling to generate control inputs
            control_input = lhs(2, samples=1)[0] * 2 - 1  # Scale to [-1, 1]
            
            # Apply noise to the control inputs
            control_input += np.random.normal(0, 0.1, size=2)
            
            # Apply control inputs
            p.setJointMotorControl2(husky, 0, p.VELOCITY_CONTROL, targetVelocity=control_input[0])
            p.setJointMotorControl2(husky, 1, p.VELOCITY_CONTROL, targetVelocity=control_input[0])
            p.setJointMotorControl2(husky, 2, p.VELOCITY_CONTROL, targetVelocity=control_input[1])
            p.setJointMotorControl2(husky, 3, p.VELOCITY_CONTROL, targetVelocity=control_input[1])
            p.stepSimulation()
            
            # Get the current state
            position, orientation = p.getBasePositionAndOrientation(husky)
            linear_velocity, angular_velocity = p.getBaseVelocity(husky)
            state = np.concatenate([position, orientation, linear_velocity, angular_velocity])
            
            data.append((state, control_input))
        
        print(f"Episode {episode + 1}/{num_episodes} completed")
    
    np.save('husky_data.npy', data, allow_pickle=True)
    return data

# Preprocess the collected data
def preprocess_data():
    data = np.load('husky_data.npy', allow_pickle=True)
    states, controls = zip(*data)
    states = np.array(states)
    controls = np.array(controls)

    # Create input-output pairs
    X = np.concatenate([states[:-1], controls[:-1]], axis=1)
    y = states[1:] - states[:-1]

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# Define the neural network model
class ResNetDynamicsModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResNetDynamicsModel, self).__init__()
        self.input_dim = input_dim
        self.resnet = models.resnet18(weights=None)  # Use weights=None
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_dim)

    def forward(self, x):
        x = x.view(-1, 1, 1, self.input_dim)  # Reshape to [batch_size, 1, 1, input_dim]
        return self.resnet(x)

# Train the neural network model
def train_model(X_train, X_val, y_train, y_val, input_dim, output_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResNetDynamicsModel(input_dim=input_dim, output_dim=output_dim).to(device)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}')

    torch.save(model.state_dict(), 'resnet_dynamics_model.pth')
    return model

# Define the original dynamics using PyBullet
def original_dynamics(husky, x, u):
    # Reset the robot to the given state
    p.resetBasePositionAndOrientation(husky, x[:3], x[3:7])
    p.resetBaseVelocity(husky, x[7:10], x[10:13])
    
    # Apply control inputs
    p.setJointMotorControl2(husky, 0, p.VELOCITY_CONTROL, targetVelocity=u[0])
    p.setJointMotorControl2(husky, 1, p.VELOCITY_CONTROL, targetVelocity=u[0])
    p.setJointMotorControl2(husky, 2, p.VELOCITY_CONTROL, targetVelocity=u[1])
    p.setJointMotorControl2(husky, 3, p.VELOCITY_CONTROL, targetVelocity=u[1])
    
    # Step simulation
    p.stepSimulation()
    
    # Get the new state
    position, orientation = p.getBasePositionAndOrientation(husky)
    linear_velocity, angular_velocity = p.getBaseVelocity(husky)
    new_state = np.concatenate([position, orientation, linear_velocity, angular_velocity])
    
    return new_state

# Define the learned dynamics using the neural network model
def learned_dynamics(x, u, trained_model):
    x_u = np.concatenate((x, u), axis=0)
    x_u_tensor = torch.Tensor(x_u).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        x_dot = trained_model(x_u_tensor).numpy().flatten()
    return x_dot

# Define the combined dynamics model
def combined_dynamics(husky, x, u, trained_model):
    orig_dyn = original_dynamics(husky, x, u)
    learn_dyn = learned_dynamics(x, u, trained_model)
    combined_dyn = orig_dyn + learn_dyn
    return combined_dyn

# Create the OCP solver
def create_ocp_solver(husky, trained_model, N, T):
    # Define the CasADi dynamics function
    x = cs.MX.sym('x', state_dim)
    u = cs.MX.sym('u', control_dim)
    x_dot = combined_dynamics(husky, x, u, trained_model)
    dynamics = cs.Function('f', [x, u], [x_dot])

    ocp = AcadosOcp()
    ocp.model = AcadosModel()
    ocp.model.f_impl_expr = dynamics
    ocp.model.x = cs.MX.sym('x', state_dim)
    ocp.model.u = cs.MX.sym('u', control_dim)
    ocp.model.name = 'mpc_model'
    
    ocp.dims.N = N
    ocp.dims.nx = state_dim
    ocp.dims.nu = control_dim
    
    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W = np.eye(state_dim)
    ocp.cost.W_e = np.eye(state_dim)
    ocp.cost.Vx = np.eye(state_dim)
    ocp.cost.Vu = np.eye(control_dim)
    ocp.cost.Vx_e = np.eye(state_dim)
    ocp.cost.yref = np.zeros(state_dim)
    ocp.cost.yref_e = np.zeros(state_dim)
    
    ocp.constraints.lbx = np.array([-np.inf] * state_dim)  # Lower bounds for state variables
    ocp.constraints.ubx = np.array([np.inf] * state_dim)  # Upper bounds for state variables
    ocp.constraints.idxbx = np.array(range(state_dim))
    ocp.constraints.lbu = np.array([-1, -1])  # Lower bounds for control inputs
    ocp.constraints.ubu = np.array([1, 1])  # Upper bounds for control inputs
    ocp.constraints.idxbu = np.array([0, 1])
    ocp.constraints.x0 = np.zeros(state_dim)  # Initial state

    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.tf = T

    return AcadosOcpSolver(ocp, json_file='acados_ocp.json')

# Run the MPC controller
def run_mpc(solver, initial_state, reference_trajectory, num_steps):
    x = initial_state
    x_history = [x]
    u_history = []

    for step in range(num_steps):
        solver.set(0, 'lbx', x)
        solver.set(0, 'ubx', x)
        
        for i in range(N + 1):
            solver.set(i, 'yref', reference_trajectory[step + i])
        
        status = solver.solve()
        if status != 0:
            raise Exception(f'MPC solver failed at step {step}')
        
        u = solver.get(0, 'u')
        x = solver.get(1, 'x')
        
        x_history.append(x)
        u_history.append(u)
    
    return np.array(x_history), np.array(u_history)

# Main function to run the complete process
def main():
    husky = initialize_simulation()
    try:
        data = collect_data(husky)
        print("Data collection complete")
    except KeyboardInterrupt:
        print("Data collection interrupted.")
    finally:
        p.disconnect()

    X_train, X_val, y_train, y_val = preprocess_data()
    print("Data preprocessing complete")

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    try:
        model = train_model(X_train, X_val, y_train, y_val, input_dim, output_dim)
        print("Model training complete")
    except KeyboardInterrupt:
        print("Training interrupted.")
    
    # Reinitialize simulation for MPC
    husky = initialize_simulation()

    N = 20  # Horizon length
    T = 2.0  # Time horizon

    solver = create_ocp_solver(husky, model, N, T)

    initial_state = np.zeros(state_dim)
    num_steps = 50
    reference_trajectory = np.zeros((num_steps + N, state_dim))
    for i in range(num_steps + N):
        reference_trajectory[i, 0] = np.sin(0.1 * i)  # Adjust this to your specific needs

    x_history, u_history = run_mpc(solver, initial_state, reference_trajectory, num_steps)

    print("State history:\n", x_history)
    print("Control history:\n", u_history)

if __name__ == "__main__":
    main()