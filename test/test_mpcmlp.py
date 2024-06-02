import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import cv2

# Initialize PyBullet in GUI mode
p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])

# Set simulation parameters
p.setGravity(0, 0, -9.8)
time_step = 1. / 240.
p.setTimeStep(time_step)
p.setRealTimeSimulation(0)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*(list(resnet50.children())[:-1]))
        self.fc_mean = nn.Linear(resnet50.fc.in_features, 2)
        self.fc_log_std = nn.Linear(resnet50.fc.in_features, 2)
        self.value_head = nn.Linear(resnet50.fc.in_features, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        std = torch.exp(log_std).clamp(min=1e-3)  # Clamping to avoid zero std
        value = self.value_head(x)
        return mean, std, value

# Initialize the model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = ActorCritic().to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

# PPO clip parameters
eps_clip = 0.2
gamma = 0.99

def update_policy(optimizer, log_probs, rewards, values, gamma=0.99):
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
    
    values = torch.cat(values)
    advantages = discounted_rewards - values

    policy_loss = []
    value_loss = nn.functional.mse_loss(values, discounted_rewards)
    for log_prob, advantage in zip(log_probs, advantages):
        ratio = torch.exp(log_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
        policy_loss.append(-torch.min(surr1, surr2))
    policy_loss = torch.cat(policy_loss).mean()

    optimizer.zero_grad()
    total_loss = policy_loss + value_loss
    total_loss.backward()
    optimizer.step()

# Create directory for saving models if it does not exist
os.makedirs('saved_models', exist_ok=True)

# Define control parameters
max_force = [10.0] * 4  # Ensure we have four forces, one for each wheel

num_episodes = 100
steps_per_episode = 1000
save_interval = 1000  # Save the model every 1000 steps

initial_position = [0, 0, 0.1]
initial_orientation = p.getQuaternionFromEuler([0, 0, 0])

# Define target position
target_position = [1.0, 1.0]  # Target x, y position
kp_linear = 0.5  # Proportional gain for linear velocity
kp_angular = 0.5  # Proportional gain for angular velocity

def compute_velocity(position, orientation, target_position):
    dx = target_position[0] - position[0]
    dy = target_position[1] - position[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    # Compute desired heading
    target_heading = np.arctan2(dy, dx)
    current_heading = p.getEulerFromQuaternion(orientation)[2]
    heading_error = target_heading - current_heading
    
    # Normalize heading error to [-pi, pi]
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

    # Compute linear and angular velocities
    linear_velocity = kp_linear * distance
    angular_velocity = kp_angular * heading_error
    
    return linear_velocity, angular_velocity

def normalize_rewards(rewards):
    rewards = np.array(rewards)
    mean = np.mean(rewards)
    std = np.std(rewards)
    return (rewards - mean) / (std + 1e-9)

# Add this to your simulation loop
for episode in range(num_episodes):
    # Reset the robot to the initial position and orientation
    p.resetBasePositionAndOrientation(robot_id, initial_position, initial_orientation)
    p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

    log_probs = []
    rewards = []
    values = []
    total_reward = 0  # Reset total reward for the episode
    
    for step in range(steps_per_episode):
        # Get robot's state
        position, orientation = p.getBasePositionAndOrientation(robot_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)
        state = np.array(position[:2] + (p.getEulerFromQuaternion(orientation)[2],))
        
        # Add batch dimension and reshape state to match ResNet input
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # Get action and value from policy
        policy.eval()  # Set to evaluation mode
        mean, std, value = policy(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Compute the velocity commands
        linear_velocity_command, angular_velocity_command = compute_velocity(position, orientation, target_position)

        # Calculate wheel velocities from linear and angular velocities
        wheel_radius = 0.1651  # Husky wheel radius in meters
        half_axle_length = 0.281  # Half the distance between wheels
        v_l = (2 * linear_velocity_command - angular_velocity_command * half_axle_length) / (2 * wheel_radius)
        v_r = (2 * linear_velocity_command + angular_velocity_command * half_axle_length) / (2 * wheel_radius)

        p.setJointMotorControlArray(
            bodyIndex=robot_id,
            jointIndices=[2, 3, 4, 5],  # Indices for Husky's wheels
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[v_l, v_r, v_l, v_r],
            forces=max_force
        )

        # Step the simulation
        p.stepSimulation()
        time.sleep(time_step)

        # Get current feedback action (joint velocities)
        joint_states = p.getJointStates(robot_id, [2, 3, 4, 5])
        feedback_action = [joint_states[0][1], joint_states[1][1], joint_states[2][1], joint_states[3][1]]

        # Calculate reward
        reward = -np.sum((np.array([v_l, v_r, v_l, v_r]) - feedback_action) ** 2)
        total_reward += reward

        # Store log prob, reward, and value
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)

        # Print state and action values
        print(f"Step: {step}, State: {state}, Target Velocities: {[round(v_l, 2), round(v_r, 2), round(v_l, 2), round(v_r, 2)]}, Feedback Action: {[round(a, 2) for a in feedback_action]}, Reward: {round(reward, 2)}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Save the model periodically
        if (episode * steps_per_episode + step + 1) % save_interval == 0:
            print(f'Episode [{episode+1}/{num_episodes}], Step [{step+1}/{steps_per_episode}], Total Reward: {total_reward}')
            torch.save(policy.state_dict(), f'saved_models/resnet50_policy_step_{episode * steps_per_episode + step + 1}.pth')

    # Normalize rewards before updating the policy and value function
    normalized_rewards = normalize_rewards(rewards)

    # Update the policy and value function using normalized rewards
    update_policy(optimizer, log_probs, normalized_rewards, values)

    # Print total reward per episode
    print(f'Episode [{episode+1}/{num_episodes}], Total Reward: {total_reward}')

# Disconnect PyBullet
p.disconnect()

# Save the final model
torch.save(policy.state_dict(), 'saved_models/resnet50_policy_final.pth')
