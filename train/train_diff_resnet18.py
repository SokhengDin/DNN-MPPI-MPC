import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from dnn.resnet18 import ResNet18
import matplotlib.pyplot as plt


def plot_training_loss(losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Training loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss over Iteration")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def train_resnet18(states, controls, errors, num_epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    states_tensor = torch.from_numpy(states).float().to(device)
    controls_tensor = torch.from_numpy(controls).float().to(device)
    errors_tensor = torch.from_numpy(errors).float().to(device)


    dataset = TensorDataset(states_tensor, controls_tensor, errors_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    input_dim = states.shape[1] + controls.shape[1]
    output_dim = errors.shape[1]
    model = ResNet18(input_dim, output_dim).to(device)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs_states, inputs_controls, labels = data
            inputs = torch.cat((inputs_states, inputs_controls), dim=1)  
            inputs = inputs.unsqueeze(2)  

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses.append(loss.item())

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / (i+1):.4f}")

    print("Training finished.")

    # PLot training loss
    plot_path = "training_loss.png"
    plot_training_loss(losses, plot_path)

    # Save the model
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "resnet18_diff.pth")
    torch.save(model.state_dict(), model_save_path)

    print(f"Model saved at {model_save_path}")
    return model

# input dir
input_dir = "saved_data"

# Load the data
states = np.load(os.path.join(input_dir, "states_diff.npy"))
controls = np.load(os.path.join(input_dir, "controls_diff.npy"))
errors = np.load(os.path.join(input_dir, "errors_diff.npy"))

# Train the ResNet18 model
num_epochs = 100
batch_size = 64
learning_rate = 0.001
trained_model = train_resnet18(states, controls, errors, num_epochs, batch_size, learning_rate)