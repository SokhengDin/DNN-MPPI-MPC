import pybullet as p
import pybullet_data
import time
import numpy as np
import torch
import l4casadi as l4c

from acados_template import AcadosOcp, AcadosOcpSolver
from torchvision import models



p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
robot_id = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])


p.setGravity(0, 0, -9.8)
time_step = 1. / 240.
p.setTimeStep(time_step)
p.setRealTimeSimulation(0)

target_velocity = [0.1, 0.1]
max_force = [10.0, 10.0]


states = []
actions = []

for i in range(1000):
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

    # Store state and action
    states.append(state)
    actions.append(target_velocity)

# Disconnect PyBullet
p.disconnect()

# Convert collected data to tensors
states = torch.tensor(states, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.float32)