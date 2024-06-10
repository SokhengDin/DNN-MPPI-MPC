import pybullet as p
import time
import math
import pybullet_data
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the Actor-Critic networks
class Actor(nn.Module):
    def __init__(self, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, num_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.tanh(self.out(x))

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.out(x)

# Initialize the environment
clid = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
plane = p.loadURDF("plane.urdf")
p.changeDynamics(plane, -1, lateralFriction=1.0, restitution=0.0)
husky = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])
for i in range(p.getNumJoints(husky)):
    p.changeDynamics(husky, i, lateralFriction=1.0, restitution=0.0)

# Define the action space and observation space
num_actions = 3  # Forward/backward, rotation, and no action
observation_dim = 3  # x, y, orientation

# Create the Actor and Critic networks
actor = Actor(num_actions).cuda()
critic = Critic().cuda()

# Define the training parameters
num_episodes = 10000
max_steps_per_episode = 1000
learning_rate = 0.001
discount_factor = 0.99

# Define the optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

# Training loop
for episode in range(num_episodes):
    # Reset the environment
    p.resetBasePositionAndOrientation(husky, [0, 0, 0.1], [0, 0, 0, 1])
    p.resetBaseVelocity(husky, [0, 0, 0], [0, 0, 0])
    state = torch.from_numpy(np.array(p.getBasePositionAndOrientation(husky)[0][:2] + p.getEulerFromQuaternion(p.getBasePositionAndOrientation(husky)[1])[2:])).float().cuda()
    episode_reward = 0

    for step in range(max_steps_per_episode):
        # Get the action from the actor network
        action = actor(state)
        action = action.detach().cpu().numpy()

        # Execute the action in the environment
        forward_velocity = action[0]
        rotation_velocity = action[1]
        wheel_velocities = [forward_velocity + rotation_velocity, forward_velocity - rotation_velocity,
                            forward_velocity + rotation_velocity, forward_velocity - rotation_velocity]
        for i in range(len(wheel_velocities)):
            p.setJointMotorControl2(husky, i+2, p.VELOCITY_CONTROL, targetVelocity=wheel_velocities[i], force=1000)
        p.stepSimulation()

        # Get the next state and reward
        next_state = torch.from_numpy(np.array(p.getBasePositionAndOrientation(husky)[0][:2] + p.getEulerFromQuaternion(p.getBasePositionAndOrientation(husky)[1])[2:])).float().cuda()
        reward = np.linalg.norm(next_state[:2].cpu().numpy())  # Reward based on distance traveled
        episode_reward += reward

        # Compute the value estimates and advantage
        value = critic(state)
        next_value = critic(next_state)
        target = reward + discount_factor * next_value.detach()
        advantage = target - value

        # Compute the actor and critic losses
        actor_loss = -torch.mean(advantage * actor(state))
        critic_loss = torch.mean(advantage.pow(2))

        # Zero the gradients
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        # Compute the gradients and update the networks
        actor_loss.backward(retain_graph=True)  # Specify retain_graph=True
        critic_loss.backward()
        actor_optimizer.step()
        critic_optimizer.step()

        state = next_state

    print(f"Episode {episode+1}: Reward = {episode_reward}")

# Save the trained models
torch.save(actor.state_dict(), "actor_model.pth")
torch.save(critic.state_dict(), "critic_model.pth")

p.disconnect()