import airsim
import time

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Function to print current drone position
def print_current_location():
    state = client.getMultirotorState()
    position = state.kinematics_estimated.position
    print(f"Current Drone Location: x={position.x_val}, y={position.y_val}, z={position.z_val}")

# Example: Move the drone manually with specific commands
def manual_control():
    try:
        while True:
            command = input("Enter command (w/a/s/d/q/e to control, 'land' to land, 'exit' to quit): ")
            
            if command == "w":
                client.moveByVelocityAsync(3, 0, 0, 1).join()  # Move forward
            elif command == "s":
                client.moveByVelocityAsync(-3, 0, 0, 1).join()  # Move backward
            elif command == "a":
                client.moveByVelocityAsync(0, -3, 0, 1).join()  # Move left
            elif command == "d":
                client.moveByVelocityAsync(0, 3, 0, 1).join()  # Move right
            elif command == "q":
                client.moveByVelocityAsync(0, 0, -3, 1).join()  # Move up
            elif command == "e":
                client.moveByVelocityAsync(0, 0, 3, 1).join()   # Move down
            elif command == "land":
                client.landAsync().join()  # Land the drone
            elif command == "exit":
                break
            else:
                print("Invalid command. Use w/a/s/d/q/e or 'land', 'exit'.")

            # Print current location after each move
            print_current_location()

    finally:
        # Disable API control and disarm the drone
        client.armDisarm(False)
        client.enableApiControl(False)

# Run manual control function
manual_control()

