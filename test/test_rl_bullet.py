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

# Adjust the time.sleep() duration to match the time_step
time.sleep(time_step)


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*(list(resnet50.children())[:-1]))
        self.fc_mean = nn.Linear(resnet50.fc.in_features, 2)
        self.fc_log_std = nn.Linear(resnet50.fc.in_features, 2)
        self.value_head = nn.Linear(resnet50.fc.in_features, 1)
        self.tanh = nn.Tanh()

        self.apply(self.init_weights)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        mean = self.tanh(self.fc_mean(x)).clamp(-1.0, 1.0)
        log_std = self.fc_log_std(x)
        std = torch.exp(log_std).clamp(min=1e-3)  
        value = self.value_head(x).squeeze()
        return mean, std, value
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

# Initialize the model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = ActorCritic().to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

# PPO clip parameters
eps_clip = 0.2
gamma = 0.99

def update_policy(optimizer, log_probs, rewards, values, gamma=0.99, eps_clip=0.2, value_coef=0.5, entropy_coef=0.01):
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
    
    values = torch.stack(values).squeeze()
    advantages = discounted_rewards - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss = 0
    value_loss = 0.5 * nn.functional.mse_loss(values, discounted_rewards)
    entropy_loss = 0

    for log_prob, advantage in zip(log_probs, advantages):
        ratio = torch.exp(log_prob - log_prob.detach())
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
        policy_loss += -torch.min(surr1, surr2).mean()
        entropy_loss += -(log_prob * torch.exp(log_prob)).mean()

    optimizer.zero_grad()
    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    total_loss.backward()
    optimizer.step()

    
# Create directory for saving models if it does not exist
os.makedirs('saved_models', exist_ok=True)

# Define control parameters
max_force = [10.0] * 4  # Ensure we have four forces, one for each wheel

num_episodes = 1000
steps_per_episode = 1000
save_interval = 1000  # Save the model every 1000 steps

initial_position = [0, 0, 0.1]
initial_orientation = p.getQuaternionFromEuler([0, 0, 0])

# Define target position
target_position = [1.0, 1.0]  # Target x, y position
kp_linear = 0.5  # Proportional gain for linear velocity
kp_angular = 0.5  # Proportional gain for angular velocity
max_grad_norm = 1.0

def compute_velocity(position, orientation, target_position, max_linear_velocity=1.0, max_angular_velocity=1.0):
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
    linear_velocity = max_linear_velocity * min(distance, 1.0)
    angular_velocity = max_angular_velocity * heading_error / np.pi
    
    return linear_velocity, angular_velocity

def normalize_rewards(rewards):
    rewards = np.array(rewards)
    mean = np.mean(rewards)
    std = np.std(rewards)
    return (rewards - mean) / (std + 1e-9)

# Define camera parameters
camera_positions = [
    [0, -1, 1],  # Camera 1 position
    [1, 0, 1],   # Camera 2 position
    [-1, 0, 1]   # Camera 3 position
]

camera_target_positions = [
    [0, 0, 0.5],   # Camera 1 target (slightly above ground)
    [0, 0, 0.5],   # Camera 2 target (slightly above ground)
    [0, 0, 0.5]    # Camera 3 target (slightly above ground)
]

width, height = 320, 240

def get_camera_images():
    images = []
    for idx, (position, target) in enumerate(zip(camera_positions, camera_target_positions)):
        view_matrix = p.computeViewMatrix(position, target, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(width) / height, nearVal=0.01, farVal=100.0)
        img_arr = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        rgb_img = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]
        depth_img = np.reshape(img_arr[3], (height, width))
        seg_img = np.reshape(img_arr[4], (height, width)).astype(np.uint8)  # Convert to uint8
        
        # Normalize the depth image to the range [0, 255] and convert to uint8
        depth_img = ((depth_img - depth_img.min()) / (depth_img.max() - depth_img.min()) * 255).astype(np.uint8)
        
        images.append((rgb_img, depth_img, seg_img))
    return images

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
        yaw = p.getEulerFromQuaternion(orientation)[2]
        state = np.array([position[0], position[1], yaw])

        # Normalize the state components individually
        state_normalized = (state - np.array([0, 0, -np.pi])) / np.array([1, 1, 2*np.pi])

        # Print the position, orientation, and normalized state for verification
        print(f"Step: {step}")
        print(f"  Position:       {position}")
        print(f"  Yaw:            {yaw}")
        print(f"  State:          {state}")
        print(f"  Normalized State: {state_normalized}")

        # Reshape state to match ResNet input (1, 3, H, W)
        state_tensor = torch.tensor(state_normalized, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # Get action and value from policy
        policy.eval()  # Set to evaluation mode
        with torch.no_grad():
            mean, std, value = policy(state_tensor)

        if torch.isfinite(mean).all() and torch.isfinite(std).all():
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        else:
            action = torch.zeros_like(mean)
            log_prob = torch.zeros(1, device=device)

        # Compute the velocity commands
        linear_velocity_command, angular_velocity_command = compute_velocity(position, orientation, target_position, max_linear_velocity=1.0, max_angular_velocity=1.0)

        # Scale the velocity commands based on the normalized output of the actor-critic network
        linear_velocity_command *= action[0, 0].item()
        angular_velocity_command *= action[0, 1].item()

        # Calculate wheel velocities from linear and angular velocities
        wheel_radius = 0.1651  # Husky wheel radius in meters
        half_axle_length = 0.281  # Half the distance between wheels

        if np.isfinite(linear_velocity_command) and np.isfinite(angular_velocity_command):
            v_l = (2 * linear_velocity_command - angular_velocity_command * half_axle_length) / (2 * wheel_radius)
            v_r = (2 * linear_velocity_command + angular_velocity_command * half_axle_length) / (2 * wheel_radius)
        else:
            v_l = 0.0
            v_r = 0.0

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
        print(f"  Target Velocities: {[round(v_l, 2), round(v_r, 2), round(v_l, 2), round(v_r, 2)]}")
        print(f"  Feedback Action:   {[round(a, 2) for a in feedback_action]}")
        print(f"  Reward:            {round(reward, 2)}")
        print()

        # Capture and display camera images
        # images = get_camera_images()
        # for i, (rgb_img, depth_img, seg_img) in enumerate(images):
        #     cv2.imshow(f'Camera {i+1} - RGB', rgb_img)
        #     cv2.imshow(f'Camera {i+1} - Depth', depth_img)
        #     cv2.imshow(f'Camera {i+1} - Segmentation', seg_img)
        # cv2.waitKey(1)

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
    print()

# Disconnect PyBullet
p.disconnect()

# Save the final model
torch.save(policy.state_dict(), 'saved_models/resnet50_policy_final.pth')
