import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dnn.resnet18 import ResNet18

def train_resnet18(states, controls, errors, num_epochs, batch_size, learning_rate):
    # Convert data to PyTorch tensors
    states_tensor = torch.from_numpy(states).float()
    controls_tensor = torch.from_numpy(controls).float()
    errors_tensor = torch.from_numpy(errors).float()

    # Create a dataset and data loader
    dataset = TensorDataset(states_tensor, controls_tensor, errors_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the ResNet18 model
    input_dim = states.shape[1] + controls.shape[1]
    output_dim = errors.shape[1]
    model = ResNet18(input_dim, output_dim)  # Assuming the error has 3 components (x, y, yaw)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs_states, inputs_controls, labels = data
            inputs = torch.cat((inputs_states, inputs_controls), dim=1)  # Concatenate states and controls

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / (i+1):.4f}")

    print("Training finished.")
    return model

# Source dir
input_dir = "saved_data"

# Load the saved data
states = np.load(os.path.join(input_dir, "states.npy"))
controls = np.load(os.path.join(input_dir, "controls.npy"))
errors = np.load(os.path.join(input_dir, "errors.npy"))
# Train the ResNet18 model
num_epochs = 50
batch_size = 64
learning_rate = 0.001
trained_model = train_resnet18(states, controls, errors, num_epochs, batch_size, learning_rate)