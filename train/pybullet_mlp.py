import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os

# Initialize PyBullet in DIRECT mode
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])

# Set simulation parameters
p.setGravity(0, 0, -9.8)
time_step = 1. / 240.
p.setTimeStep(time_step)
p.setRealTimeSimulation(0)

class ResNet50Policy(nn.Module):
    def __init__(self):
        super(ResNet50Policy, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*(list(resnet50.children())[:-1]))  
        self.fc_mean = nn.Linear(resnet50.fc.in_features, 2)  
        self.fc_log_std = nn.Linear(resnet50.fc.in_features, 2)  

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        std = torch.exp(log_std)
        return mean, std

# Initialize the model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = ResNet50Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

# Policy gradient update function
def update_policy(optimizer, log_probs, rewards, gamma=0.99):
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
    
    policy_loss = []
    for log_prob, reward in zip(log_probs, discounted_rewards):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

# Create directory for saving models if it does not exist
os.makedirs('saved_models', exist_ok=True)

# Define control parameters
max_force = [10.0, 10.0]

num_episodes = 20
steps_per_episode = 1000
save_interval = 1000  # Save the model every 1000 steps

for episode in range(num_episodes):
    log_probs = []
    rewards = []
    total_reward = 0
    
    states_batch = []

    for step in range(steps_per_episode):
        # Get robot's state
        position, orientation = p.getBasePositionAndOrientation(robot_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)
        state = np.array(position[:2] + (p.getEulerFromQuaternion(orientation)[2],))

        states_batch.append(state)
        
        if len(states_batch) >= 2:  # Ensure batch size > 1
            state_tensor = torch.tensor(states_batch, dtype=torch.float32).to(device)
            state_tensor = state_tensor.view(len(states_batch), 3, 1, 1)  # Reshape to [batch_size, channels, height, width]

            # Get action from policy
            policy.eval()  # Set to evaluation mode
            mean, std = policy(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            # Apply action to control the robot
            target_velocity = action[-1].cpu().numpy().flatten()  # Use the latest action
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

            # Get current feedback action (joint velocities)
            joint_states = p.getJointStates(robot_id, [2, 3])
            feedback_action = [joint_states[0][1], joint_states[1][1]]

            # Calculate reward
            reward = -np.sum((target_velocity - feedback_action) ** 2)
            total_reward += reward

            # Store log prob and reward
            log_probs.append(log_prob[-1])
            rewards.append(reward)

            # Print state and action values
            print(f"State: {state}, Target Velocity: {target_velocity}, Feedback Action: {feedback_action}, Reward: {reward}")

            states_batch = []

        # Print and save the model periodically
        if (episode * steps_per_episode + step + 1) % save_interval == 0:
            print(f'Episode [{episode+1}/{num_episodes}], Step [{step+1}/{steps_per_episode}], Total Reward: {total_reward}')
            torch.save(policy.state_dict(), f'saved_models/resnet50_policy_step_{episode * steps_per_episode + step + 1}.pth')

    # Update the policy
    update_policy(optimizer, log_probs, rewards)

    # Print total reward per episode
    print(f'Episode [{episode+1}/{num_episodes}], Total Reward: {total_reward}')

# Disconnect PyBullet
p.disconnect()

# Save the final model
torch.save(policy.state_dict(), 'saved_models/resnet50_policy_final.pth')
