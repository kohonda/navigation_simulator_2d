import matplotlib.animation as animation
from matplotlib import pyplot as plt
import numpy as np
import os
import yaml
from navigation_simulator_2d.utils import visualizer, ParameterHandler
from navigation_simulator_2d.common import RobotCommand, RobotObservation, AgentState
from navigation_simulator_2d.simulator import Simulator

# Parameters
config_path = 'config/default.yaml'
params = ParameterHandler()
params.init(config_path)

# Simulator
simulator = Simulator()
simulator.reset(params)

# constant velocity motion
linear_vel = np.array([2.0, 0.0])
robot_command = RobotCommand(linear_vel=linear_vel, angular_vel=0.5)

save_animation = False
save_folder = '/home/honda/Videos/'

frames = []

for _ in range(130):
    robot_obs = simulator.get_observation()
    static_map = simulator.get_static_map()
    
    obstacle_map, robot_traj = simulator.step(robot_command)

    image_arr = visualizer.render(
        static_map=static_map, 
        obstacle_map=obstacle_map, 
        robot_observation=robot_obs,
        robot_traj=robot_traj,
        goal_state=params.goal_state, 
        robot_radius=params.robot_radius,
        global_path=None,
        local_path_list=None)
    
    if save_animation:
        frames.append([plt.imshow(image_arr)])      
    else: 
        # For faster rendering
        plt.imshow(image_arr)
        plt.pause(0.01)
        plt.clf()
        
# save animation from image list
if save_animation:
    anim = animation.ArtistAnimation(plt.gcf(), frames, interval=100)
    anim.save(save_folder+'animation.mp4', fps=10, writer='ffmpeg')
    
plt.close()