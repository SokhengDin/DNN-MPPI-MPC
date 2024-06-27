import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.input_layer = torch.nn.Linear(input_dim, 512)

        hidden_layers = []
        for i in range(3):
            hidden_layers.append(torch.nn.Linear(512,  512))

        self.hidden_layer = torch.nn.ModuleList(hidden_layers)
        self.out_layer = torch.nn.Linear(512, 3)

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

def mean_absolute_error(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

def train_mlp(states, controls, errors, num_epochs, batch_size, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Preprocessing
    train_states, val_states, train_controls, val_controls, train_errors, val_errors = train_test_split(
        states, controls, errors, test_size=0.2, random_state=42
    )

    state_scaler = StandardScaler()
    control_scaler = StandardScaler()
    error_scaler = StandardScaler()

    state_scaler.fit(train_states)
    control_scaler.fit(train_controls)
    error_scaler.fit(train_errors)

    train_states_scaled = state_scaler.transform(train_states)
    train_controls_scaled = control_scaler.transform(train_controls)
    train_errors_scaled = error_scaler.transform(train_errors)

    val_states_scaled = state_scaler.transform(val_states)
    val_controls_scaled = control_scaler.transform(val_controls)
    val_errors_scaled = error_scaler.transform(val_errors)

    train_states_tensor = torch.from_numpy(train_states_scaled).float().to(device)
    train_controls_tensor = torch.from_numpy(train_controls_scaled).float().to(device)
    train_errors_tensor = torch.from_numpy(train_errors_scaled).float().to(device)

    val_states_tensor = torch.from_numpy(val_states_scaled).float().to(device)
    val_controls_tensor = torch.from_numpy(val_controls_scaled).float().to(device)
    val_errors_tensor = torch.from_numpy(val_errors_scaled).float().to(device)

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
    train_maes = []
    val_maes = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        running_train_mae = 0.0
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

    # Plot training loss and MAE
    plot_path = "training_metrics_mlp_300x100_3l_mppi.png"
    plot_training_metrics(train_losses, val_losses, train_maes, val_maes, plot_path)

    # Save the model and scalers
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "mlp_diff_300x100_3l_mppi.pth")
    torch.save(model.state_dict(), model_save_path)

    scaler_save_path = os.path.join(save_dir, "scalers_mlp_diff_300x100_3l_mppi.pth")
    torch.save({
        "state_scaler": state_scaler,
        "control_scaler": control_scaler,
        "error_scaler": error_scaler
    }, scaler_save_path)

    print(f"Model and scalers saved at {model_save_path} and {scaler_save_path}")
    return model, avg_train_loss, avg_val_loss, avg_train_mae, avg_val_mae, state_scaler, control_scaler, error_scaler

if __name__ == "__main__":
    # input dir
    input_dir = "saved_data"

    # Load the data
    states = np.load(os.path.join(input_dir, "states_diff.npy"))
    controls = np.load(os.path.join(input_dir, "controls_diff.npy"))
    errors = np.load(os.path.join(input_dir, "errors_diff.npy"))

    # Train the MLP model
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.0001
    trained_model, avg_train_loss, avg_val_loss, avg_train_mae, avg_val_mae, state_scaler, control_scaler, error_scaler = train_mlp(states, controls, errors, num_epochs, batch_size, learning_rate)

    print(f"Final Average Training Loss: {avg_train_loss:.4f}")
    print(f"Final Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Final Average Training MAE: {avg_train_mae:.4f}")
    print(f"Final Average Validation MAE: {avg_val_mae:.4f}")