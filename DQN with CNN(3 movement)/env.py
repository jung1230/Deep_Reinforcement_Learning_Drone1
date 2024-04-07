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
checkpoint = 0

class DroneEnv(object):

    def __init__(self, useDepth=False): # in python, underscore before function name is the convention of private(not actually private) 

        self.client = airsim.MultirotorClient() 

        self.last_dist = self.get_distance_to_goal(self.client.getMultirotorState().kinematics_estimated.position)


        self.angle = 0.0

        self.useDepth = useDepth


    def step(self, drone_action):



        # adjust velocity
        self.angle = self.interpret_action(drone_action)

        # Set angle for yaw change (in degrees)
        yaw_angle = self.angle  # Adjust this value as needed

        self.client.moveByAngleRatesThrottleAsync(0, 0.3, yaw_angle, 0.61, 1).join() # Wait for movement to complete
        quad_state = self.client.getMultirotorState().kinematics_estimated.position
        print(f"{quad_state.x_val:.1f} {quad_state.y_val:.1f} {quad_state.z_val:.1f}", end=' || ')

        self.client.moveToZAsync(-1.5, 1, 1).join()


        collision = self.client.simGetCollisionInfo().has_collided
        
        drone_position = self.client.getMultirotorState().kinematics_estimated.position

        drone_velocity = self.client.getMultirotorState().kinematics_estimated.linear_velocity


        reward, done = self.compute_reward(drone_position, collision)

        observation, image = self.get_vision()


        return observation, reward, done, image



    def interpret_action(self, drone_action):

        # (x,y,z,rot right 45 deg, rot left 45 deg)

        # scaling_factor = 3

        # if drone_action == 0:

        #     self.speed_offset = (scaling_factor, 0, 0, 0, 0)

        # elif drone_action == 1:

        #     self.speed_offset = (-scaling_factor, 0, 0, 0, 0)

        # elif drone_action == 2:

        #     self.speed_offset = (0, scaling_factor, 0, 0, 0)

        if (drone_action == 1):

            self.angle = -1

        elif (drone_action == 0):
            self.angle = 0

        elif (drone_action == 2):

            self.angle = 1

        return self.angle
        

    def compute_reward(self, drone_position, collision):

        reward = -1
        done = 0

        if collision:
            reward = -50
            done = 1
        else:
            dist = self.get_distance_to_goal(drone_position)

            progress = self.last_dist - dist
            self.last_dist = dist

            if dist < 10 and checkpoint == 0:
                reward += 10
                checkpoint = 1
                print("\nCK1\n")
            elif dist < 10 and checkpoint == 1:
                reward = 500
                print("\nCK2\n")

            else:
                reward +=  progress

        
        print("reward: {:.1f}".format(reward), "Loc:",end='')
        # if reward <= -1:
        #     reward = -50
        #     done = 1

        #     time.sleep(1)
        
        
        if reward >= 400:
            done = 1
            time.sleep(1)


        return reward, done

  


    def get_distance_to_goal(self, drone_position):
        # quad_state = self.client.getMultirotorState().kinematics_estimated.position
        # print(quad_state.x_val, quad_state.y_val, quad_state.z_val)
        if checkpoint == 0:
            goal_loc = np.array([ 61.45759963989258, 36.440147399902344, -2]) #first goal
        else:
            goal_loc = np.array([38.70063400268555,  78.32078552246094, -2]) #final goal

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

        checkpoint = 0
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



        # myenv\Scripts\activate.bat before run the program 


