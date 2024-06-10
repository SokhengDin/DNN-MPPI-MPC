import pybullet as p
import pybullet_data
import numpy as np

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

# Define camera parameters
camera_target_position = [0, 0, 0]
camera_distance = 3
camera_yaw = 50
camera_pitch = -35
camera_roll = 0
up_axis_index = 2
near_plane = 0.01
far_plane = 100
fov = 60

width, height = 320, 240
aspect = width / height

# Capture camera image
def capture_camera_image():
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=camera_target_position,
        distance=camera_distance,
        yaw=camera_yaw,
        pitch=camera_pitch,
        roll=camera_roll,
        upAxisIndex=up_axis_index
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=near_plane,
        farVal=far_plane
    )

    _, _, rgba_image, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix
    )
    
    return np.array(rgba_image).reshape(height, width, 4)

# Overlay image in PyBullet window
def overlay_image_in_pybullet(image):
    for y in range(height):
        for x in range(width):
            pixel = image[y, x]
            p.addUserDebugText('.', [x * 0.01, y * 0.01, 1], textColorRGB=[pixel[0] / 255.0, pixel[1] / 255.0, pixel[2] / 255.0], textSize=0.3)

# Simulation loop
for step in range(1000):
    p.stepSimulation()
    if step % 100 == 0:  # Capture and overlay image every 100 steps
        rgba_image = capture_camera_image()
        overlay_image_in_pybullet(rgba_image)
    p.setRealTimeSimulation(1)

# Disconnect PyBullet
p.disconnect()
