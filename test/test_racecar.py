import pybullet as p
import pybullet_data
import time

# Connect to PyBullet physics server with GPU acceleration
physicsClient = p.connect(p.GUI, options="--opengl2")

# Enable GPU acceleration
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(1)
p.setPhysicsEngineParameter(enableFileCaching=0, numSolverIterations=10, numSubSteps=2)

# Load the ground plane
p.loadURDF("plane.urdf")

# Load the racecar URDF
racecar = p.loadURDF("urdf/racecar/racecar.urdf", [0, 0, 0.2])

# Simulation settings
p.setGravity(0, 0, -9.81)

# Get joint information
num_joints = p.getNumJoints(racecar)
steering_joints = []
drive_joints = []

for i in range(num_joints):
    joint_info = p.getJointInfo(racecar, i)
    joint_name = joint_info[1].decode('utf-8')
    
    if 'steering' in joint_name:
        steering_joints.append(i)
    elif joint_name == 'left_rear_wheel_joint' or joint_name == 'right_rear_wheel_joint' or \
         joint_name == 'left_front_wheel_joint' or joint_name == 'right_front_wheel_joint':
        drive_joints.append(i)

print(f"Steering Joints: {steering_joints}")
print(f"Drive Joints: {drive_joints}")

while True:
    # Get keyboard events
    keys = p.getKeyboardEvents()

    # Control the joints based on keyboard input
    steering_angle = 0
    drive_velocity = 0
    max_force = 10

    # Steer left
    if p.B3G_LEFT_ARROW in keys:
        steering_angle = 0.5

    # Steer right
    elif p.B3G_RIGHT_ARROW in keys:
        steering_angle = -0.5

    # Drive forward
    if p.B3G_UP_ARROW in keys:
        drive_velocity = 10  # Negative velocity for forward motion

    # Drive backward
    elif p.B3G_DOWN_ARROW in keys:
        drive_velocity = -10  # Positive velocity for backward motion

    # Apply steering angle to steering joints
    p.setJointMotorControlArray(
        racecar, steering_joints,
        p.POSITION_CONTROL,
        targetPositions=[steering_angle] * len(steering_joints),
        forces=[max_force] * len(steering_joints)
    )

    # Apply drive velocity to drive joints
    p.setJointMotorControlArray(
        racecar, drive_joints,
        p.VELOCITY_CONTROL,
        targetVelocities=[drive_velocity] * len(drive_joints),
        forces=[max_force] * len(drive_joints)
    )

    time.sleep(1./240.)