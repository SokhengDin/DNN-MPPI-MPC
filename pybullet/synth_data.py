import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import pandas as pd

# Initialize PyBullet in GUI mode
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])

# Set simulation parameters
p.setGravity(0, 0, -9.8)
time_step = 1. / 240.
p.setTimeStep(time_step)
p.setRealTimeSimulation(0)

# Generate dataset
num_steps = 5000
max_force = [10.0, 10.0]
data = []

for step in range(num_steps):
    # Get robot's state
    position, orientation = p.getBasePositionAndOrientation(robot_id)
    linear_velocity, angular_velocity = p.getBaseVelocity(robot_id)
    state = np.array(position[:2] + (p.getEulerFromQuaternion(orientation)[2],))

    # Generate random action
    target_velocity = np.random.uniform(-1, 1, size=2)
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

    # Record the state and action
    data.append(np.concatenate((state, target_velocity)))

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data, columns=['x', 'y', 'theta', 'target_velocity_1', 'target_velocity_2'])
df.to_csv('robot_random_actions_dataset.csv', index=False)

# Disconnect PyBullet
p.disconnect()
