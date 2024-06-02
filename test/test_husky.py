import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt

# Connect to PyBullet with GPU acceleration
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane and the robot URDF
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("/home/eroxii/ocp_ws/bullet3/data/husky/husky.urdf", [0, 0, 0.1])

# Load the cube object
cube_size = 0.5
cube_mass = 1.0
cube_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[cube_size]*3, rgbaColor=[1, 0, 0, 1])
cube_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[cube_size]*3)
cube_id = p.createMultiBody(baseMass=cube_mass, baseCollisionShapeIndex=cube_collision_shape_id, baseVisualShapeIndex=cube_visual_shape_id, basePosition=[3, 0, cube_size])


# Enable GPU acceleration and set real-time simulation
p.setRealTimeSimulation(1)
p.setPhysicsEngineParameter(enableFileCaching=0, numSolverIterations=10, numSubSteps=2)
p.setPhysicsEngineParameter(enableConeFriction=1)
p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)

# Start the simulation
p.setGravity(0, 0, -9.8)

# Simulation parameters
target_fps = 240
timestep = 1.0 / target_fps
num_iterations = 1000

# Keyboard control parameters
keyboard_control = True
max_force = 100
max_velocity = 10
speed = 0.0
steering = 0.0


def simulate_lidar(robot_id, num_rays=360, max_distance=30):
    base_position, base_orientation = p.getBasePositionAndOrientation(robot_id)
    base_orientation_matrix = p.getMatrixFromQuaternion(base_orientation)
    base_orientation_matrix = np.array(base_orientation_matrix).reshape(3, 3)
    base_forward = base_orientation_matrix.dot(np.array([1, 0, 0]))
    base_position = np.array(base_position)
    
    rays_from = np.tile(base_position, (num_rays, 1))
    angles = np.linspace(-np.pi, np.pi, num_rays)
    ray_directions = np.stack([np.cos(angles), np.sin(angles), np.zeros(num_rays)], axis=1)
    ray_directions = ray_directions.dot(base_orientation_matrix.T)
    
    rays_to = rays_from + ray_directions * max_distance
    ray_results = p.rayTestBatch(rays_from.tolist(), rays_to.tolist())
    
    distances = []
    for result in ray_results:
        hit_fraction = result[2]
        distances.append(hit_fraction * max_distance)
    
    return distances
# Create a separate plot window
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

def plot_lidar(lidar_readings):
    ax.clear()
    angles = np.linspace(-np.pi, np.pi, len(lidar_readings))
    x = lidar_readings * np.cos(angles)
    y = lidar_readings * np.sin(angles)
    ax.scatter(x, y, s=1)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('LiDAR Scan')
    fig.canvas.draw()
    fig.canvas.flush_events()

# Simulation loop
while True:
    # Keyboard control
    keys = p.getKeyboardEvents()
    if keyboard_control:
        if p.B3G_UP_ARROW in keys:
            speed = max_velocity
        elif p.B3G_DOWN_ARROW in keys:
            speed = -max_velocity
        else:
            speed = 0.0
        
        if p.B3G_LEFT_ARROW in keys:
            steering = 0.5
        elif p.B3G_RIGHT_ARROW in keys:
            steering = -0.5
        else:
            steering = 0.0

    # Apply control actions to the Husky robot
    p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=speed + steering * max_velocity, force=max_force)
    p.setJointMotorControl2(robot_id, 3, p.VELOCITY_CONTROL, targetVelocity=speed - steering * max_velocity, force=max_force)
    p.setJointMotorControl2(robot_id, 4, p.VELOCITY_CONTROL, targetVelocity=speed + steering * max_velocity, force=max_force)
    p.setJointMotorControl2(robot_id, 5, p.VELOCITY_CONTROL, targetVelocity=speed - steering * max_velocity, force=max_force)

    # Simulate LiDAR and plot the readings
    lidar_readings = simulate_lidar(robot_id)
    plot_lidar(lidar_readings)

    # Step the simulation
    p.stepSimulation()

    # Sleep to maintain a consistent simulation rate
    time.sleep(timestep)

# Disconnect from PyBullet
p.disconnect()