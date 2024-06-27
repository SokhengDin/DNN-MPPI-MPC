import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import models
from sklearn.model_selection import train_test_split
# from dnn.resnet50 import ResNet50
import matplotlib.pyplot as plt

class ResNet50(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 512)
        self.hidden_layers = nn.ModuleList()
        for i in range(2):
            self.hidden_layers.append(nn.Linear(512, 512))
        self.out_layer = nn.Linear(512, output_dim)
        
        # Model is not trained -- setting output to zero
        with torch.no_grad():
            self.out_layer.bias.fill_(0.)
            self.out_layer.weight.fill_(0.)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.repeat(1, 3, 1, 1)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x

def mean_absolute_error(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def plot_training_metrics(train_losses, val_losses, train_maes, val_maes, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(train_losses, label="Training loss")
    ax1.plot(val_losses, label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss over Epochs")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_maes, label="Training MAE")
    ax2.plot(val_maes, label="Validation MAE")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("MAE")
    ax2.set_title("Training and Validation MAE over Epochs")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def train_resnet50(states, controls, errors, num_epochs, batch_size, learning_rate):
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
    output_dim = errors.shape[1]
    model = ResNet50(input_dim, output_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        running_train_mae = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs_states, inputs_controls, labels = data
            inputs = torch.cat((inputs_states, inputs_controls), dim=1)  
            inputs = inputs.unsqueeze(2)  

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            running_train_mae += mean_absolute_error(outputs, labels).item()

        avg_train_loss = running_train_loss / len(train_dataloader)
        avg_train_mae = running_train_mae / len(train_dataloader)
        train_losses.append(avg_train_loss)
        train_maes.append(avg_train_mae)

        # Validation loss and MAE
        model.eval()
        running_val_loss = 0.0
        running_val_mae = 0.0
        with torch.no_grad():
            for data in val_dataloader:
                inputs_states, inputs_controls, labels = data
                inputs = torch.cat((inputs_states, inputs_controls), dim=1)
                inputs = inputs.unsqueeze(2)
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                running_val_mae += mean_absolute_error(outputs, labels).item()

        avg_val_loss = running_val_loss / len(val_dataloader)
        avg_val_mae = running_val_mae / len(val_dataloader)
        val_losses.append(avg_val_loss)
        val_maes.append(avg_val_mae)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training MAE: {avg_train_mae:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation MAE: {avg_val_mae:.4f}")

    print("Training finished.")

    # Calculate average loss and MAE
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_train_mae = sum(train_maes) / len(train_maes)
    avg_val_mae = sum(val_maes) / len(val_maes)

    print(f"Average Training Loss: {avg_train_loss:.4f}")
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Average Training MAE: {avg_train_mae:.4f}")
    print(f"Average Validation MAE: {avg_val_mae:.4f}")

    # Plot training metrics
    plot_path = "training_metrics_resnet50_300x100_mppi.png"
    plot_training_metrics(train_losses, val_losses, train_maes, val_maes, plot_path)

    # Save the model
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "resnet50_diff_300x100.pth")
    torch.save(model.state_dict(), model_save_path)

    print(f"Model saved at {model_save_path}")
    return model, avg_train_loss, avg_val_loss, avg_train_mae, avg_val_mae

# Main execution
input_dir = "saved_data"

# Load the data
states = np.load(os.path.join(input_dir, "states_diff.npy"))
controls = np.load(os.path.join(input_dir, "controls_diff.npy"))
errors = np.load(os.path.join(input_dir, "errors_diff.npy"))

# Train the ResNet50 model
num_epochs = 100
batch_size = 64
learning_rate = 0.0001
trained_model, avg_train_loss, avg_val_loss, avg_train_mae, avg_val_mae = train_resnet50(states, controls, errors, num_epochs, batch_size, learning_rate)

print(f"Final Average Training Loss: {avg_train_loss:.4f}")
print(f"Final Average Validation Loss: {avg_val_loss:.4f}")
print(f"Final Average Training MAE: {avg_train_mae:.4f}")
print(f"Final Average Validation MAE: {avg_val_mae:.4f}")