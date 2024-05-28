import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define custom dataset class
class RobotDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.scaler = StandardScaler()
        self.states = self.scaler.fit_transform(self.data[['x', 'y', 'theta']])
        self.actions = self.data[['target_velocity_1', 'target_velocity_2']].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.float32)
        return state, action

# Define ResNet-50 based model
class ResNet50Regression(nn.Module):
    def __init__(self):
        super(ResNet50Regression, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*(list(resnet50.children())[:-1]))
        self.fc = nn.Linear(resnet50.fc.in_features, 2)

    def forward(self, x):
        x = x.view(x.size(0), 3, 1, 1)  # Reshape to fit ResNet-50 input requirements
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Load dataset
dataset = RobotDataset('robot_random_actions_dataset.csv')
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50Regression().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for states, actions in train_loader:
        states, actions = states.to(device), actions.to(device)
        optimizer.zero_grad()
        outputs = model(states)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * states.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for states, actions in val_loader:
            states, actions = states.to(device), actions.to(device)
            outputs = model(states)
            loss = criterion(outputs, actions)
            val_loss += loss.item() * states.size(0)

    val_loss /= len(val_loader.dataset)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'resnet50_robot_model.pth')
