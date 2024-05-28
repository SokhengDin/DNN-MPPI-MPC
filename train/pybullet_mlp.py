import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from torchvision import models


p.connect(p.connect(p.GUI))
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])

# Set simulation parameters
p.setGravity(0, 0, -9.8)
time_step = 1. / 240.
p.setTimeStep(time_step)
p.setRealTimeSimulation(0)

class ResNet50Modified(nn.Module):
    def __init__(self):
        super(ResNet50Modified, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*(list(resnet50.children())[:-1]))  
        self.fc = nn.Linear(resnet50.fc.in_features, 2)  

    def forward(self, x):
     
        x = x.view(x.size(0), 3, 1, 1)
        x = self.features(x)
        x = torch.flatten(x, 1) 
        x = self.fc(x)
        return x

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50Modified().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.eval()


os.makedirs('saved_models', exist_ok=True)

#
target_velocity = [1, 0.1]
max_force = [10.0, 10.0]


states = []
actions = []
losses = []

num_episodes = 20
steps_per_episode = 1000
save_interval = 1000  

for episode in range(num_episodes):
    for step in range(steps_per_episode):
        # Get robot's state
        position, orientation = p.getBasePositionAndOrientation(robot_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)
        state = np.array(position[:2] + (p.getEulerFromQuaternion(orientation)[2],))

        # Apply control
        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=[2, 3],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=target_velocity,
            forces=max_force
        )

        # Step the simulation
        p.stepSimulation()
        time.sleep(time_step)

        # Get current feedback action
        joint_states = p.getJointStates(robot_id, [2, 3])
        feedback_action = [joint_states[0][1], joint_states[1][1]]


        print(f"State: {state}, Target Velocity: {target_velocity}, Feedback Action: {feedback_action}")


        states.append(state)
        actions.append(target_velocity)


        state_tensor = torch.tensor([state], dtype=torch.float32).to(device)
        action_tensor = torch.tensor([target_velocity], dtype=torch.float32).to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(state_tensor)
        loss = criterion(output, action_tensor)

        loss.backward()
        optimizer.step()

        # Store the loss
        losses.append(loss.item())

        # Print and save the model periodically
        if (episode * steps_per_episode + step + 1) % save_interval == 0:
            print(f'Episode [{episode+1}/{num_episodes}], Step [{step+1}/{steps_per_episode}], Loss: {loss.item():.4f}')
            torch.save(model.state_dict(), f'saved_models/mlp_learned_dyn_step_{episode * steps_per_episode + step + 1}.pth')

# Disconnect PyBullet
p.disconnect()

# Save the final model
torch.save(model.state_dict(), 'saved_models/mlp_learned_dyn_final.pth')

# Print final loss
print(f'Final Loss: {loss.item():.4f}')
