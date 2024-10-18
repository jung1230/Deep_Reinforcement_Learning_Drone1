import airsim
import torch
import numpy as np
import time
from PIL import Image
import math
import random

MOVEMENT_INTERVAL = 1

class DroneEnv(object):
    def __init__(self, useDepth=False):
        self.client = airsim.MultirotorClient()
        
        # Ensure CHECKPOINT is initialized
        self.CHECKPOINT = 0  # Set initial checkpoint to 0
        # Set offsets based on the AirSim to Unreal coordinate differences
        
        # Set the initial distance to goal
        self.last_dist = self.get_distance_to_goal(
            self.client.getMultirotorState().kinematics_estimated.position
        )

        # Rotation and movement offsets
        self.speed_offset = (0, 0, 0, 0, 0)
        self.angle = 0.0
        self.useDepth = useDepth

    def step(self, drone_action):
        # adjust velocity
        self.speed_offset = self.interpret_action(drone_action)
        drone_velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.client.moveByVelocityAsync(
            drone_velocity.x_val + self.speed_offset[0],
            drone_velocity.y_val + self.speed_offset[1],
            drone_velocity.z_val + self.speed_offset[2],
            MOVEMENT_INTERVAL
        ).join()

        if self.speed_offset[3] == 1:
            self.client.moveByAngleRatesZAsync(0, 0, 1, 0, duration=MOVEMENT_INTERVAL).join()
        elif self.speed_offset[4] == 1:
            self.client.moveByAngleRatesZAsync(0, 0, -1, 0, duration=MOVEMENT_INTERVAL).join()

        collision = self.client.simGetCollisionInfo().has_collided
        time.sleep(0.5)

        drone_position = self.client.getMultirotorState().kinematics_estimated.position
        reward, done = self.compute_reward(drone_position, collision)
        observation, image = self.get_vision()

        return observation, reward, done, image

    def interpret_action(self, drone_action):
        scaling_factor = 3
        if drone_action == 0:
            self.speed_offset = (scaling_factor, 0, 0, 0, 0)
        elif drone_action == 1:
            self.speed_offset = (-scaling_factor, 0, 0, 0, 0)
        elif drone_action == 2:
            self.speed_offset = (0, scaling_factor, 0, 0, 0)
        elif drone_action == 3:
            self.speed_offset = (0, -scaling_factor, 0, 0, 0)
        elif drone_action == 4:
            self.speed_offset = (0, 0, scaling_factor, 0, 0)
        elif drone_action == 5:
            self.speed_offset = (0, 0, -scaling_factor, 0, 0)
        elif drone_action == 6:
            self.speed_offset = (0, 0, 0, 1, 0)
        elif drone_action == 7:
            self.speed_offset = (0, 0, 0, 0, 1)

        return self.speed_offset

    def compute_reward(self, drone_position, collision):


        # Convert drone position from AirSim to Unreal coordinates
        drone_x = drone_position.x_val
        drone_y = drone_position.y_val
        drone_z = drone_position.z_val

        print(f"Current Drone Pos ({drone_x}, {drone_y}, {drone_z})")

        reward = -1  # Base reward to encourage movement
        done = 0  # Continue episode unless collision occurs or checkpoint is reached

        angle = self.angle
        x = math.cos(angle)
        y = math.sin(angle)
        heading = np.array([x, y])

        # Checkpoint handling and conversion of checkpoint coordinates to AirSim system
        if self.CHECKPOINT == 0:
            goal_loc = np.array([22.469, 17.281, 0.5])
        elif self.CHECKPOINT == 1:
            goal_loc = np.array([32.05, 27.822, 0.5])
        elif self.CHECKPOINT == 2:
            goal_loc = np.array([58.6, 29.82, 0.5])
        elif self.CHECKPOINT == 3:
            goal_loc = np.array([58.6, 61.02, 0.5])
        elif self.CHECKPOINT == 4:
            goal_loc = np.array([37.604, 61.205, 0.5])
        elif self.CHECKPOINT == 5:
            goal_loc = np.array([37.24, 79.469, 0.5])

        # Calculate direction and distance to the goal
        direction = goal_loc - np.array([drone_x, drone_y, drone_z])
        direction_norm = np.linalg.norm(direction)

        # Calculate heading vector and dot product for movement direction
        heading_vec = direction / direction_norm if direction_norm > 1 else direction
        heading_dp = np.dot(heading_vec[:2], heading)

        # Get the distance to the goal and compute progress
        dist = self.get_distance_to_goal(drone_position)
        progress = self.last_dist - dist
        self.last_dist = dist

        # Collision handling
        if collision:
            reward += -10  # Collision penalty
            done = 1  # End the episode
            print(f"Collision penalty applied: {reward}")
        else:
            # Progressive reward based on distance to the goal
            if dist > 0.75:
                reward += 5
            elif 0.5 < dist <= 0.75:
                reward += 10
            elif 0.25 < dist <= 0.5:
                reward += 15
            else:
                reward += 20  # Reward for being close to the goal

            # Penalize backward movement
            if heading_dp < 0:
                backward_penalty = -20 * abs(heading_dp)
                reward += backward_penalty
                print(f"Penalizing backward movement: {backward_penalty}")
            else:
                # Reward for forward progress
                forward_reward = 10 * progress
                reward += forward_reward
                print(f"Rewarding forward movement: {forward_reward}")

            # Check if a checkpoint is reached
            checkpoint_reached = False
            if dist <= 0.1 and self.CHECKPOINT == 0:
                reward += 50
                self.CHECKPOINT = 1
                checkpoint_reached = True
                print("\nCheckpoint 1 reached\n")
            elif dist <= 0.1 and self.CHECKPOINT == 1:
                reward += 100
                self.CHECKPOINT = 2
                checkpoint_reached = True
                print("\nCheckpoint 2 reached\n")
            elif dist <= 0.1 and self.CHECKPOINT == 2:
                reward += 150
                self.CHECKPOINT = 3
                checkpoint_reached = True
                print("\nCheckpoint 3 reached\n")
            elif dist <= 0.1 and self.CHECKPOINT == 3:
                reward += 200
                self.CHECKPOINT = 4
                checkpoint_reached = True
                print("\nCheckpoint 4 reached\n")
            elif dist <= 0.1 and self.CHECKPOINT == 4:
                reward += 250
                self.CHECKPOINT = 5
                checkpoint_reached = True
                print("\nCheckpoint 5 reached\n")

            # Reset distance calculation if a checkpoint is reached
            if checkpoint_reached:
                self.last_dist = self.get_distance_to_goal(drone_position)

        print(f"Final reward: {reward}, Location: ({drone_position.x_val}, {drone_position.y_val}, {drone_position.z_val})")
        return reward, done


    def get_distance_to_goal(self, drone_position, goal_loc=None):
        # Ensure the goal location is correctly set based on the current checkpoint

        if goal_loc is None:
            if self.CHECKPOINT == 0:
                goal_loc = np.array([22.469, 17.281, 0.5])
            elif self.CHECKPOINT == 1:
                goal_loc = np.array([32.05, 27.822, 0.5])
            elif self.CHECKPOINT == 2:
                goal_loc = np.array([58.6, 29.82, 0.5])
            elif self.CHECKPOINT == 3:
                goal_loc = np.array([58.6, 61.02, 0.5])
            elif self.CHECKPOINT == 4:
                goal_loc = np.array([37.604, 61.205, 0.5])
            elif self.CHECKPOINT == 5:
                goal_loc = np.array([37.24, 79.469, 0.5])

        # Convert drone position to numpy array
        drone_loc = np.array([drone_position.x_val, drone_position.y_val, drone_position.z_val])

        # Ensure that goal_loc is a valid numpy array and not None
        if goal_loc is None or not isinstance(goal_loc, np.ndarray):
            raise ValueError(f"Invalid goal_loc: {goal_loc}")

        # Compute the Euclidean distance between the current position and the goal position
        dist = np.linalg.norm(drone_loc - goal_loc)
        return dist


    def get_vision(self):
        if self.useDepth:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)
            ])
        else:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

        response = responses[0]

        if self.useDepth:
            img1d = np.array(response.image_data_float, dtype=np.float)
            img1d = img1d * 3.5 + 30
            img1d[img1d > 255] = 255
            image = np.reshape(img1d, (response.height, response.width))
        else:
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            image = img1d.reshape(response.height, response.width, 3)

        image_array = Image.fromarray(image).resize((84, 84)).convert("L")
        observation = np.array(image_array)

        return observation, image

    def reset(self):
        self.client.reset()
        self.last_dist = self.get_distance_to_goal(self.client.getMultirotorState().kinematics_estimated.position)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        observation, image = self.get_vision()
        return observation, image
