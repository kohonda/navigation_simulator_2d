from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import io

from navigation_simulator_2d.common import RobotObservation, AgentState
from navigation_simulator_2d.utils import MapHandler


def cmap_with_transparency(cmap: cm)->cm:
    cmap_data = cmap(np.arange(cmap.N))
    cmap_data[0, 3] = 0.0
    customized_cmap = colors.ListedColormap(cmap_data)
    return customized_cmap

def render(
    static_map: MapHandler,
    obstacle_map: MapHandler,
    robot_observation: RobotObservation,
    robot_traj: np.ndarray,
    goal_state: AgentState,
    robot_radius: float,
    inflation_layer: np.ndarray = None,
    global_path: np.ndarray = None,
    local_path_list: list = None,
    local_path_best_index: int = None,
    sub_goal_index: int = None,
    visualize_local_path: bool = False,
    )->np.ndarray:
    
    """Render the robot navigation
    """
    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
        
    # image of 2d static map
    axes.imshow(1-static_map.get_map_as_np('occupancy').T, cmap='gray', vmin=0, vmax=1, alpha=1.0)
    
    # image of 2d obstacle map
    obstacle_cmap = cmap_with_transparency(cm.get_cmap('Blues'))
    axes.imshow(obstacle_map.get_map_as_np('occupancy').T, cmap=obstacle_cmap, vmin=0, vmax=1, alpha=1.0)
    
    # inflation
    if inflation_layer is not None:
        inflation_cmap = cmap_with_transparency(cm.get_cmap('gray'))
        axes.imshow(inflation_layer.T, cmap=inflation_cmap, vmin=0, vmax=2, alpha=0.5)
    
    # image of scan points from lidar
    scan_cmap = cmap_with_transparency(cm.bwr)
    scan_points = robot_observation.scan_points.get_map_as_np('occupancy')
    axes.imshow(scan_points.T, cmap=scan_cmap, vmin=0, vmax=1, alpha=1.0)
    
    # robot color switched when collision
    robot_color : str
    if robot_observation.is_collision==False:
        robot_color = 'g'
    else:
        robot_color = 'r'
    
    # image of robot position with circle
    # viz_radius = int(robot_radius / static_map.get_resolution())
    viz_radius = static_map.meter2pixel_float(robot_radius)
    robot_in_ij = static_map.pose2index_float(robot_observation.state.pos)
    axes.add_patch(plt.Circle(robot_in_ij, viz_radius, color=robot_color, fill=False, linewidth=6))
    
    # image of robot heading
    robot_heading_ij = static_map.pose2index_float(robot_observation.state.pos + np.array([np.cos(robot_observation.state.yaw), np.sin(robot_observation.state.yaw)]).T * robot_radius)
    axes.plot([robot_in_ij[0], robot_heading_ij[0]], [robot_in_ij[1], robot_heading_ij[1]], color=robot_color)
    
    # image of robot trajectory
    traj_in_ij = static_map.pose_array2index_array_float(robot_traj)
    axes.plot(traj_in_ij[:,0], traj_in_ij[:,1], color='b', linewidth=viz_radius)
    
    # image of goal position with arrow
    goal_in_ij = static_map.pose2index_float(goal_state.pos)
    goal_heading_ij = static_map.pose2index_float(goal_state.pos + np.array([np.cos(goal_state.yaw), np.sin(goal_state.yaw)]).T * robot_radius)
    axes.add_patch(plt.Arrow(goal_in_ij[0], goal_in_ij[1], goal_heading_ij[0] - goal_in_ij[0], goal_heading_ij[1] - goal_in_ij[1], color='blue', width=viz_radius))
    
    # image of global path
    if global_path is not None:
        axes.plot(global_path[:,0], global_path[:,1], color='r', linewidth=2.0)
        
    # image of local planner predicted path
    if local_path_list is not None and local_path_best_index is not None:
        ### visualize all local paths
        if visualize_local_path:
            for local_path in local_path_list:
                local_paths_ij = static_map.pose_array2index_array_float(local_path)
                axes.plot(local_paths_ij[:,0], local_paths_ij[:,1], color='b', linewidth=1.0)
        ### visualize best local path
        local_path_ij = static_map.pose_array2index_array_float(local_path_list[local_path_best_index])
        axes.plot(local_path_ij[:,0], local_path_ij[:,1], color='orange', linewidth=2.0)
        
    # image of sub goal
    if sub_goal_index is not None:
        sub_goal_radius = static_map.meter2pixel_float(0.2)
        axes.add_patch(plt.Circle(global_path[sub_goal_index], sub_goal_radius, color='orange', fill=True, linewidth=3))
                    
    # settings
    image_size = static_map.get_image_size()
    axes.set_xlim([0, image_size[0]])
    axes.set_ylim([image_size[1], 0])
    fig.tight_layout()
    axes,plt.axis('off')
    
    # convert to numpy array
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format="raw")
    plt.close(fig)
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    io_buf.close()
    
    return img_arr[:, :, :3]