import pybullet as p
import pybullet_data
import time
import numpy as np
import pandas as pd

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
planeId = p.loadURDF("plane.urdf")

# Load Husky URDF
husky_urdf_path = "husky/husky.urdf"  # Change this to the correct path
huskyId = p.loadURDF(husky_urdf_path, [0, 0, 0.1])

# Simulation parameters
time_step = 1.0 / 240.0  # 240 Hz sampling rate
p.setTimeStep(time_step)
p.setRealTimeSimulation(0)

# Data collection
data = []

# Define input methods
def step_input(step_interval, low, high, step):
    if step % step_interval < step_interval / 2:
        return low, low
    else:
        return high, high

def sinusoidal_input(A, B, omega, phi, time):
    return A * np.sin(omega * time), B * np.sin(omega * time + phi)

def ramp_input(max_velocity, duration, step):
    velocity = (max_velocity / duration) * step
    return min(velocity, max_velocity), min(velocity, max_velocity)

def random_input(min_val, max_val):
    return np.random.uniform(min_val, max_val), np.random.uniform(min_val, max_val)

# Simulation loop
num_steps = 5000  # Total number of steps for data collection
for i in range(num_steps):
    time_elapsed = i * time_step
    
    # Example using step input
    # left_wheel_velocity, right_wheel_velocity = step_input(500, 0.5, 1.0, i)
    
    # Uncomment to use sinusoidal input
    # left_wheel_velocity, right_wheel_velocity = sinusoidal_input(3.0, 3.0, 0.1, 0, time_elapsed)
    
    # Uncomment to use ramp input
    left_wheel_velocity, right_wheel_velocity = ramp_input(5.0, 10.0, time_elapsed)
    
    # Uncomment to use random input
    # left_wheel_velocity, right_wheel_velocity = random_input(0.0, 1.0)
    
    p.setJointMotorControl2(huskyId, 2, p.VELOCITY_CONTROL, targetVelocity=left_wheel_velocity)
    p.setJointMotorControl2(huskyId, 3, p.VELOCITY_CONTROL, targetVelocity=left_wheel_velocity)
    p.setJointMotorControl2(huskyId, 4, p.VELOCITY_CONTROL, targetVelocity=right_wheel_velocity)
    p.setJointMotorControl2(huskyId, 5, p.VELOCITY_CONTROL, targetVelocity=right_wheel_velocity)
    
    p.stepSimulation()
    
    pos, orn = p.getBasePositionAndOrientation(huskyId)
    linear_vel, angular_vel = p.getBaseVelocity(huskyId)
    
    data.append({
        'time': time_elapsed,
        'pos_x': pos[0],
        'pos_y': pos[1],
        'pos_z': pos[2],
        'orn_x': orn[0],
        'orn_y': orn[1],
        'orn_z': orn[2],
        'orn_w': orn[3],
        'linear_vel_x': linear_vel[0],
        'linear_vel_y': linear_vel[1],
        'linear_vel_z': linear_vel[2],
        'angular_vel_x': angular_vel[0],
        'angular_vel_y': angular_vel[1],
        'angular_vel_z': angular_vel[2],
        'left_wheel_velocity': left_wheel_velocity,
        'right_wheel_velocity': right_wheel_velocity
    })
    
    time.sleep(time_step)

# Save data to CSV
df = pd.DataFrame(data)
df.to_csv('husky_simulation_data_ramp.csv', index=False)

# Disconnect PyBullet
p.disconnect()
