"""
Parameter load and handle class
"""
from __future__ import annotations

from typing import Union
import yaml
import os
import numpy as np

from navigation_simulator_2d.common import AgentState, MovingObstacle, MotionModel, Shape


class ParameterHandler():
    def __init__(self):
        self._config : dict = None
        current_path : str = os.path.dirname(os.path.abspath(__file__))
        self._project_path : str = os.path.dirname(os.path.dirname(current_path))
        self._rng = None
        
    def init(self, config : Union[str, dict], seed: int = 0) -> None:
        
        if isinstance(config, str):
            config_path = os.path.join(self._project_path, config)
            self._config: dict = yaml.safe_load(open(config_path, 'r'))    
        elif isinstance(config, dict):
            self._config = config
        else:
            raise Exception('Unknown config type.')
        
        if self._config is None:
            raise Exception('Config file not found.')
        
        # set random generator seed
        self._rng = np.random.default_rng(seed)
        
        # set params
        self._control_interval_time = self._config['common']['control_interval_time']
        self._robot_radius = self._config['common']['robot_radius']
        self._known_static_map = self._load_known_static_map()
        self._unknown_static_obs = self._load_unknown_static_obs()
        self._robot_initial_state = self._load_robot_initial_state()
        self._goal_state = self._load_goal_state()
        self._moving_obstacles = self._load_moving_obstacles()
        
        # navigation envs params
        self._max_episode_steps = self._config['common']['max_episode_steps']
        self._max_global_planner_hz = self._config['common']['max_global_planner_hz']
    
        
        # DWA params (if there is)
        if 'dwa_config' in self._config:
            self.dwa_config = self._config['dwa_config']
        else:
            self.dwa_config = None
            
        # Dijkstra params (if there is)
        if 'dijkstra_config' in self._config:
            self.dijkstra_config = self._config['dijkstra_config']
        else:
            self.dijkstra_config = None

    
    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps
    
    @property
    def max_global_planner_hz(self) -> int:
        return self._max_global_planner_hz
    
    @property
    def min_path_length_ratio(self) -> float:
        return self._min_path_length_ratio
    
    @property
    def control_interval_time(self) -> float:
        return self._control_interval_time

    @property
    def robot_radius(self) -> float:
        return self._robot_radius
    
    @property
    def robot_initial_state(self) -> AgentState:
        return self._robot_initial_state
    
    @property
    def goal_state(self) -> AgentState:
        return self._goal_state

    @property
    def known_static_map(self) -> str:
        return self._known_static_map
    
    @property
    def unknown_static_obs(self) -> Union[None, str]:
        return self._unknown_static_obs
        
    @property
    def moving_obstacles(self) -> list:
        return self._moving_obstacles
    
    def _load_known_static_map(self) -> str:
        if 'random_map_mode' in self._config['common']:
            # random load in known static map dir
            random_map_dir = self._get_path(self._config['common']['known_static_map'])
            if not os.path.isdir(random_map_dir):
                raise Exception('Known static map should be a directory in random map mode.')
            dir_list = os.listdir(random_map_dir)
            if len(dir_list) == 0:
                raise Exception('No known static map found in random map mode.')
            # randomly select a map
            map_dir = self._rng.choice(dir_list)
            map_path = os.path.join(random_map_dir, map_dir, 'map.yaml')
            return map_path
        else:
            # load given known static map
            static_map = self._get_path(self._config['common']['known_static_map'])
            if static_map is None:
                raise Exception('Known static map not found.')
            return static_map
        
    def reset_static_map(self) -> None:
        self._known_static_map = self._load_known_static_map()

    def _load_unknown_static_obs(self) -> Union[None, str]:
        if 'unknown_static_obs' in self._config['common']:
            return self._get_path(self._config['common']['unknown_static_obs'])
        else:
            return None

    def reset_robot_initial_state(self) -> None:
        self._robot_initial_state = self._load_robot_initial_state()
        
    def _load_robot_initial_state(self) -> AgentState:
        if self._config['common']['start_pose_type'] == 'specified':
            # determine start pose from specified position
            start_pose: dict = self._config['common']['start_pose']
        elif self._config['common']['start_pose_type'] == 'candidates':
            # randomly determine start pose from candidates
            start_pose: dict = self._get_random_from_candidates(self._config['start_pose_candidates'])
        elif self._config['common']['start_pose_type'] == 'random':
            # randomly determine start pose from constraints
            start_pose: dict = self._get_random_initial_pose(self._config['start_pose_constraints'])
        else:
            raise Exception('Unknown start pose type.')
        
        return AgentState(np.array([start_pose['x'], start_pose['y']]), start_pose['yaw'])
    
    def reset_goal_state(self) -> None:
        self._goal_state = self._load_goal_state()
        
    def _load_goal_state(self) -> AgentState:
        if self._config['common']['goal_pose_type'] == 'specified':
            # determine goal pose from specified position
            goal_pose: dict = self._config['common']['goal_pose']
        elif self._config['common']['goal_pose_type'] == 'candidates':
            # randomly determine goal pose from candidates
            goal_pose: dict = self._get_random_from_candidates(self._config['goal_pose_candidates'])
        elif self._config['common']['goal_pose_type'] == 'random':
            # randomly determine goal pose from constraints
            goal_pose: dict = self._get_random_initial_pose(self._config['goal_pose_constraints'])
        else:
            raise Exception('Unknown goal pose type.')
        
        return AgentState(np.array([goal_pose['x'], goal_pose['y']]), goal_pose['yaw'])
    
    def reset_moving_obstacles(self) -> None:
        self._moving_obstacles = self._load_moving_obstacles()

    def _load_moving_obstacles(self) -> Union[list, None]:
        if 'moving_obstacles' not in self._config:
            return None
        
        obs_info : dict = self._config['moving_obstacles']
        obs_ids = []
        obs_list = []        
        if obs_info['type'] == 'all':
            # Spawn all obstacles.
            obs_candidates = obs_info['candidates']
            for obs in obs_candidates:
                obs_list.append(self._obs_to_instance(obs))                
        elif obs_info['type'] == 'candidates':
            # Spawn obstacles randomly from candidates with overlap.
            obs_candidates = obs_info['candidates']
            obs_num = obs_info['num']
            for _ in range(obs_num):
                obs = self._get_random_from_candidates(obs_candidates)
                obs_list.append(self._obs_to_instance(obs))
        elif obs_info['type'] == 'candidates_without_overlap':
            # Spawn obstacles randomly from candidates without overlap.
            obs_candidates = obs_info['candidates']
            obs_num = obs_info['num']
            while len(obs_list) < obs_num:
                obs = self._get_random_from_candidates(obs_candidates)
                # check overlap in the obs_list
                if obs['id'] in obs_ids:
                    continue
                else:
                    obs_ids.append(obs['id'])
                    obs_list.append(self._obs_to_instance(obs))
                
        elif obs_info['type'] == 'candidates_pickup':
            # pick up 'must' obstacle and randomly spawn other obstacles from candidates.
            ## separate must and maybe candidates
            must_candidates = []
            maybe_candidates = []
            for obs in obs_info['candidates']:
                if obs['type'] == 'must':
                    must_candidates.append(obs)
                elif obs['type'] == 'maybe':
                    maybe_candidates.append(obs)
                else:
                    raise Exception('Unknown obstacle type.')
            
            if len(must_candidates) > obs_info['num']:
                raise Exception('Too many must obstacles.')
            
            ## add must obstacles
            obs_list = [self._obs_to_instance(obs) for obs in must_candidates]
            
            ## add maybe obstacles
            maybe_num = obs_info['num'] - len(must_candidates)
            for _ in range(maybe_num):
                obs = self._get_random_from_candidates(maybe_candidates)
                obs_list.append(self._obs_to_instance(obs))
        
        elif obs_info['type'] == 'random':
            # Spawn obstacles randomly with some constraints.
            obs_constraints = obs_info['constraints']
            obs_candidates = obs_info['candidates']
            obs_num = obs_info['num']
            default_obs = self._get_random_from_candidates(obs_candidates)
            for _  in range(obs_num):
                obs = self._get_random_obs_with_constraints(obs_constraints, default_obs)
                obs_list.append(self._obs_to_instance(obs))
        else:
            raise Exception('Unknown obstacle selection type.')
        
        return obs_list
    
    def _get_path(self, path: str):
        return os.path.join(self._project_path, path)
    
    def _get_random_from_candidates(self, candidates: list) -> Union[dict, None]:
        if len(candidates) == 0:
            return None
        elif len(candidates) == 1:
            return candidates[0]
        else:
            return candidates[self._rng.integers(0, len(candidates))]
    
    def _obs_to_instance(self, obs: dict) -> MovingObstacle:
        x = obs['x']
        y = obs['y']
        yaw = obs['yaw']
        vx = obs['vx']
        vy = obs['vy']
        vyaw = obs['vyaw']
        shape = obs['shape']
        size = obs['size']
        motion_model = obs['motion_model']
        
        if motion_model == 'random':
            motion_model = self._rng.choice(['social_force_model', 'reactive_stop_model'])
        
        ins = MovingObstacle(
            pos = np.array([x, y]),
            yaw = yaw,
            linear_vel=np.array([vx, vy]),
            target_vel=np.array([vx, vy]),
            angular_vel=vyaw,
            size = size,
            shape = Shape.from_str(shape),
            motion_model= MotionModel.from_str(motion_model))
        return ins
    
    def _get_random_obs_with_constraints(self, obs_constraints: dict, default_obs: dict) -> dict:
        if default_obs is None:
            raise Exception('Default obstacle is not specified.')
        
        # Get constraints
        size_min = obs_constraints['size']['min']
        size_max = obs_constraints['size']['max']
        x_min = obs_constraints['position']['x']['min']
        x_max = obs_constraints['position']['x']['max']
        y_min = obs_constraints['position']['y']['min']
        y_max = obs_constraints['position']['y']['max']
        yaw_min = obs_constraints['position']['yaw']['min']
        yaw_max = obs_constraints['position']['yaw']['max']
        vx_min = obs_constraints['velocity']['vx']['min']
        vx_max = obs_constraints['velocity']['vx']['max']
        vy_min = obs_constraints['velocity']['vy']['min']
        vy_max = obs_constraints['velocity']['vy']['max']
        vyaw_min = obs_constraints['velocity']['vyaw']['min']
        vyaw_max = obs_constraints['velocity']['vyaw']['max']
        
        # Generate random values
        size = self._rng.uniform(size_min, size_max)
        x = self._rng.uniform(x_min, x_max)
        y = self._rng.uniform(y_min, y_max)
        yaw = self._rng.uniform(yaw_min, yaw_max)
        vx = self._rng.uniform(vx_min, vx_max)
        vy = self._rng.uniform(vy_min, vy_max)
        vyaw = self._rng.uniform(vyaw_min, vyaw_max)

        random_obs = default_obs
        random_obs['size'] = size
        random_obs['x'] = x
        random_obs['y'] = y
        random_obs['yaw'] = yaw
        random_obs['vx'] = vx
        random_obs['vy'] = vy
        random_obs['vyaw'] = vyaw
        
        return random_obs
    
    def _get_random_initial_pose(self, constraint: dict) -> dict:
        if constraint is None:
            raise Exception('Constraint is not specified.')
        
        x_min = constraint['x']['min']
        x_max = constraint['x']['max']
        y_min = constraint['y']['min']
        y_max = constraint['y']['max']
        yaw_min = constraint['yaw']['min']
        yaw_max = constraint['yaw']['max']
        
        x = self._rng.uniform(x_min, x_max)
        y = self._rng.uniform(y_min, y_max)
        yaw = self._rng.uniform(yaw_min, yaw_max)
        
        start_pose = {'x': x, 'y': y, 'yaw': yaw}
        
        return start_pose
        