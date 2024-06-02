import pybullet as p
import pybullet_data
import time
import math
import numpy as np

# Connect to PyBullet with GPU acceleration
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane and the race car URDF
plane_id = p.loadURDF("plane.urdf")
car_id = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.2])

# Load the cube object
cube_size = 0.5
cube_mass = 1.0
cube_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[cube_size]*3, rgbaColor=[1, 0, 0, 1])
cube_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[cube_size]*3)
cube_id = p.createMultiBody(baseMass=cube_mass, baseCollisionShapeIndex=cube_collision_shape_id, baseVisualShapeIndex=cube_visual_shape_id, basePosition=[3, 0, cube_size])

# Set simulation parameters
p.setGravity(0, 0, -9.8)
p.setTimeStep(1./120.)

# Disable real-time simulation
p.setRealTimeSimulation(0)

# Lidar parameters
numRays = 360
rayLen = 8
rayStartLen = 0.25
rayFrom = []
rayTo = []
rayIds = []
rayHitColor = [1, 0, 0]
rayMissColor = [0, 1, 0]
replaceLines = True

# Get the joint indices for the wheels and steering
num_joints = p.getNumJoints(car_id)
wheel_joints = []
steering_joints = []

for i in range(num_joints):
    joint_info = p.getJointInfo(car_id, i)
    joint_name = joint_info[1].decode('utf-8')
    if 'wheel' in joint_name:
        wheel_joints.append(i)
    elif 'steer' in joint_name:
        steering_joints.append(i)

# Find the Hokuyo joint index
hokuyo_joint_index = -1
for i in range(num_joints):
    joint_info = p.getJointInfo(car_id, i)
    joint_name = joint_info[1].decode('utf-8')
    if 'hokuyo' in joint_name.lower():
        hokuyo_joint_index = i
        break

if hokuyo_joint_index == -1:
    raise ValueError("Hokuyo joint not found in the URDF.")

# Prepare LiDAR rays
for i in range(numRays):
    rayFrom.append([rayStartLen*math.sin(2.*math.pi*float(i)/numRays), rayStartLen*math.cos(2.*math.pi*float(i)/numRays),0])
    rayTo.append([rayLen*math.sin(2.*math.pi*float(i)/numRays), rayLen*math.cos(2.*math.pi*float(i)/numRays),0])
    if replaceLines:
        rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor,parentObjectUniqueId=car_id, parentLinkIndex=hokuyo_joint_index))
    else:
        rayIds.append(-1)

# Simulation loop
max_force = 100
speed = 0
max_speed = 50
steering = 0.0
max_steering_angle = 0.5

lastControlTime = time.time()
lastLidarTime = time.time()

while True:
    nowControlTime = time.time()
    nowLidarTime = time.time()

    # Lidar at 20Hz
    if (nowLidarTime - lastLidarTime > .3):
        numThreads = 0
        results = p.rayTestBatch(rayFrom, rayTo, numThreads, parentObjectUniqueId=car_id, parentLinkIndex=hokuyo_joint_index)
        
        # Clear the previous obstacle points
        p.removeAllUserDebugItems()
        
        # Visualize the obstacle points
        for i in range(numRays):
            hitObjectUid = results[i][0]
            hitFraction = results[i][2]
            hitPosition = results[i][3]
            if (hitFraction < 1.):
                localHitTo = [rayFrom[i][0]+hitFraction*(rayTo[i][0]-rayFrom[i][0]),
                              rayFrom[i][1]+hitFraction*(rayTo[i][1]-rayFrom[i][1]),
                              rayFrom[i][2]+hitFraction*(rayTo[i][2]-rayFrom[i][2])]
                p.addUserDebugLine(localHitTo, localHitTo, lineColorRGB=[1, 0, 0], lineWidth=5, lifeTime=0.1)
        
        lastLidarTime = nowLidarTime

    # Control at 100Hz
    if (nowControlTime - lastControlTime > .01):
        keys = p.getKeyboardEvents()
        if p.B3G_UP_ARROW in keys:
            speed += 1
        elif p.B3G_DOWN_ARROW in keys:
            speed -= 1
        else:
            speed = 0

        if p.B3G_LEFT_ARROW in keys:
            steering += 0.01
        elif p.B3G_RIGHT_ARROW in keys:
            steering -= 0.01
        else:
            steering = 0.0

        # Limit speed and steering angle
        speed = max(min(speed, max_speed), -max_speed)
        steering = max(min(steering, max_steering_angle), -max_steering_angle)

        # Apply control actions to the race car
        for joint in wheel_joints:
            p.setJointMotorControl2(car_id, joint, p.VELOCITY_CONTROL, targetVelocity=speed, force=max_force)
        for joint in steering_joints:
            p.setJointMotorControl2(car_id, joint, p.POSITION_CONTROL, targetPosition=steering)

        lastControlTime = nowControlTime

    # Step the simulation
    p.stepSimulation()

    # Sleep to maintain a consistent simulation rate
    time.sleep(1./240.)

# Disconnect from PyBullet
p.disconnect()