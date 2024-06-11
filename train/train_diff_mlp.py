import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_layer = torch.nn.Linear(input_dim, 512)
        hidden_layers = []
        for i in range(2):
            hidden_layers.append(torch.nn.Linear(512, 512))
        self.hidden_layer = torch.nn.ModuleList(hidden_layers)
        self.out_layer = torch.nn.Linear(512, 1)
        # Model is not trained -- setting output to zero
        with torch.no_grad():
            self.out_layer.bias.fill_(0.)
            self.out_layer.weight.fill_(0.)
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x

def plot_training_loss(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def train_mlp(states, controls, errors, num_epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split the dataset into training and validation sets
    train_states, val_states, train_controls, val_controls, train_errors, val_errors = train_test_split(
        states, controls, errors, test_size=0.2, random_state=42
    )

    train_states_tensor = torch.from_numpy(train_states).float().to(device)
    train_controls_tensor = torch.from_numpy(train_controls).float().to(device)
    train_errors_tensor = torch.from_numpy(train_errors).float().to(device)

    val_states_tensor = torch.from_numpy(val_states).float().to(device)
    val_controls_tensor = torch.from_numpy(val_controls).float().to(device)
    val_errors_tensor = torch.from_numpy(val_errors).float().to(device)

    train_dataset = TensorDataset(train_states_tensor, train_controls_tensor, train_errors_tensor)
    val_dataset = TensorDataset(val_states_tensor, val_controls_tensor, val_errors_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = states.shape[1] + controls.shape[1]
    model = MultiLayerPerceptron(input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs_states, inputs_controls, labels = data
            inputs = torch.cat((inputs_states, inputs_controls), dim=1)

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        train_losses.append(running_train_loss / len(train_dataloader))

        # Validation loss
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for data in val_dataloader:
                inputs_states, inputs_controls, labels = data
                inputs = torch.cat((inputs_states, inputs_controls), dim=1)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        val_losses.append(running_val_loss / len(val_dataloader))

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}")

    print("Training finished.")

    # Plot training loss
    plot_path = "training_loss_mlp.png"
    plot_training_loss(train_losses, val_losses, plot_path)

    # Save the model
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "mlp_diff.pth")
    torch.save(model.state_dict(), model_save_path)

    print(f"Model saved at {model_save_path}")
    return model

# input dir
input_dir = "saved_data"

# Load the data
states = np.load(os.path.join(input_dir, "states_diff.npy"))
controls = np.load(os.path.join(input_dir, "controls_diff.npy"))
errors = np.load(os.path.join(input_dir, "errors_diff.npy"))

# Train the MLP model
num_epochs = 50
batch_size = 64
learning_rate = 0.0001
trained_model = train_mlp(states, controls, errors, num_epochs, batch_size, learning_rate)