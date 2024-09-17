import airsim
import torch
import numpy as np
import time
from PIL import Image
import csv
import math
import pprint
import random
# reset done
# reward done
# play(action) done
# gameIteration done
# stuck it will not be stucked

MOVEMENT_INTERVAL = 1

class DroneEnv(object):
    def __init__(self, useDepth=False): # in python, underscore before function name is the convention of private(not actually private) 
        self.client = airsim.MultirotorClient() 
        self.last_dist = self.get_distance_to_goal(self.client.getMultirotorState().kinematics_estimated.position)

        # rotation added!!!!!!!!!!
        self.speed_offset = (0, 0, 0, 0, 0) 

        self.useDepth = useDepth

    def step(self, drone_action):

        # adjust velocity
        self.speed_offset = self.interpret_action(drone_action)
        drone_velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        self.client.moveByVelocityAsync(
            drone_velocity.x_val + self.speed_offset[0],
            drone_velocity.y_val + self.speed_offset[1],
            drone_velocity.z_val + self.speed_offset[2],
            MOVEMENT_INTERVAL  # control how long the drone will continue moving with the specified velocity, resume vel after the interval
        ).join()

        if self.speed_offset[3] == 1:
            # added rotation to the movement decision!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.client.moveByAngleRatesZAsync(0, 0, 1, 0, duration=MOVEMENT_INTERVAL).join()
        elif self.speed_offset[4] == 1:
            # added rotation to the movement decision!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.client.moveByAngleRatesZAsync(0, 0, -1, 0, duration=MOVEMENT_INTERVAL).join()

        
        collision = self.client.simGetCollisionInfo().has_collided

        time.sleep(0.5) # or 1 here? by Setting the speed directly may cause the flight not smooth and stable

        drone_position = self.client.getMultirotorState().kinematics_estimated.position
        drone_velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity

        reward, done = self.compute_reward(drone_position, collision)
        observation, image = self.get_vision()

        return observation, reward, done, image


    def interpret_action(self, drone_action):
        # added more movement action here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # (x,y,z,rot right 45 deg, rot left 45 deg)
        scaling_factor = 3
        if drone_action == 0:
            self.speed_offset = (scaling_factor, 0, 0, 0, 0)
        elif drone_action == 1:
            self.speed_offset = (-scaling_factor, 0, 0, 0, 0)
        elif drone_action == 2:
            self.speed_offset = (0, scaling_factor, 0, 0, 0)
        elif drone_action == 3:
            self.speed_offset = (0, -scaling_factor, 0, 0 , 0)
        elif drone_action == 4:
            self.speed_offset = (0, 0, scaling_factor, 0 , 0)
        elif drone_action == 5:
            self.speed_offset = (0, 0, -scaling_factor, 0 , 0)
        elif drone_action == 6:
            self.speed_offset = (0, 0, 0, 1, 0)
        elif drone_action == 7:
            self.speed_offset = (0, 0, 0, 0, 1)
            
        return self.speed_offset
        
    # def compute_reward(self, drone_position, collision):
    #     reward = -1
    #     if collision:
    #         reward = -100
    #     else:
    #         dist = self.get_distance_to_goal(drone_position)
    #         progress = self.last_dist - dist
    #         self.last_dist = dist

    #         if dist < 10:
    #             reward = 500
    #         else:
    #             reward +=  progress

    #     done = 0
    #     if reward <= -10:
    #         done = 1
    #         time.sleep(1)
    #     elif reward >= 500:
    #         done = 1
    #         time.sleep(1)

    #     return reward, done
    def compute_reward(self, drone_position, collision):
        reward = -1  # Start with a small negative reward to encourage movement
        if collision:
            reward = -100  # Big penalty for collision
        else:
            dist = self.get_distance_to_goal(drone_position)  # Current distance to the goal
            progress = self.last_dist - dist  # Calculate progress (negative if moving away)
    
            # Update the last distance
            self.last_dist = dist
    
            # If the drone is within a small distance to the goal, provide a large reward
            if dist < 10:
                reward = 500
            else:
                if progress > 0:
                    reward += progress  # Reward for moving closer to the goal
                else:
                    reward += progress * 10  # Penalize heavily for moving away from the goal
    
        # Check if the episode is done
        done = 0
        if reward <= -10:  # If the drone accumulates too much negative reward, end the episode
            done = 1
            time.sleep(1)
        elif reward >= 500:  # If the drone reaches the goal
            done = 1
            time.sleep(1)

        return reward, done
  

    def get_distance_to_goal(self, drone_position):
        goal_loc = np.array([3677.37, 7950.53, -71.76])
        drone_loc = np.array(list((drone_position.x_val, drone_position.y_val, drone_position.z_val)))
        
        # Compute the Euclidean distance between the current position and the goal position
        dist = np.linalg.norm(drone_loc - goal_loc)
        return dist

    def get_vision(self):
        if self.useDepth:
            # get depth image
            responses = self.client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.DepthPlannar, False, False)])
            response = responses[0]
            img1d = np.array(response.image_data_float, dtype=np.float)
            img1d = img1d * 3.5 + 30
            img1d[img1d > 255] = 255
            image = np.reshape(img1d, (responses[0].height, responses[0].width))
            image_array = Image.fromarray(image).resize((84, 84)).convert("L")
        else:
            # Get rgb image
            responses = self.client.simGetImages(
                [airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)]
            )
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
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
        # quad_state = self.client.getMultirotorState().kinematics_estimated.position
        # print(quad_state.x_val,quad_state.y_val)
        # # # Define a random offset
        # # random_offset_x = random.uniform(-100, 100)  # Random value between -100 and 100
        # # random_offset_y = random.uniform(-100, 100)  # Random value between -100 and 100
        
        # # Set the new position with the random offset
        # target_x = quad_state.x_val + random_offset_x
        # target_y = quad_state.y_val + random_offset_y


        # # Define the desired position
        # target_position = airsim.Vector3r(target_x, -target_y, 91.766731)  # Adjust the coordinates as needed
        
        # # Define the desired orientation (optional)
        # # target_orientation = airsim.to_quaternion(0, 0, random.uniform(-math.pi, math.pi))  # Adjust the orientation as needed
        # target_orientation = airsim.to_quaternion(0, 0, 0)  # Adjust the orientation as needed

        # self.client.simSetVehiclePose(airsim.Pose(target_position, target_orientation), True)

        # # Check for collisions at the target position
        # collision_info = self.client.simGetCollisionInfo()
        # while collision_info.has_collided:
        #     print("reset")
        #     self.client.reset()
        #     # Define a random offset
        #     random_offset_x = random.uniform(-100, 100)  # Random value between -100 and 100
        #     random_offset_y = random.uniform(-100, 100)  # Random value between -100 and 100
            
        #     # Set the new position with the random offset
        #     target_x = quad_state.x_val + random_offset_x
        #     target_y = quad_state.y_val + random_offset_y

        #     # Define the desired position
        #     target_position = airsim.Vector3r(target_x, -target_y, -10)  # Adjust the coordinates as needed
            
        #     # Define the desired orientation (optional)
        #     target_orientation = airsim.to_quaternion(0, 0, random.uniform(-math.pi, math.pi))  # Adjust the orientation as needed

        #     self.client.simSetVehiclePose(airsim.Pose(target_position, target_orientation), True)
        #     collision_info = self.client.simGetCollisionInfo()

        observation, image = self.get_vision()
        return observation, image

