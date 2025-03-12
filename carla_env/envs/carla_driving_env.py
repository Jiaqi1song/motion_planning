import os
import gym
import numpy as np
import carla
import random
import math
import time
from gym import spaces
import cv2
import pygame
from pygame.locals import *
from model.agent_state import *
from carla_env.wrappers import *
from carla_env.tools.hud import HUD
import itertools
from carla_env.navigation.planner import RoadOption, compute_route_waypoints
from collections import deque
import torch

def smooth_action(current, target, smoothing_factor):
    return current * smoothing_factor + target * (1 - smoothing_factor)

def transform_to_ego_frame(ego_transform, target_location):
    dx = target_location.x - ego_transform.location.x
    dy = target_location.y - ego_transform.location.y
    yaw = math.radians(ego_transform.rotation.yaw)
    rel_x = dx * math.cos(-yaw) - dy * math.sin(-yaw)
    rel_y = dx * math.sin(-yaw) + dy * math.cos(-yaw)
    return rel_x, rel_y

# Dictionary for discrete actions mapping indices to [steer, throttle]
discrete_actions = {
    0: [-1.0, 1.0],
    1: [0.0, 1.0],
    2: [1.0, 1.0],
    3: [0.0, 0.0],  # braking / stopping
}
intersection_routes = itertools.cycle([(57, 81)])
eval_routes = itertools.cycle([(48, 21), (0, 72), (28, 83), (61, 39)])
class CarlaDrivingEnv(gym.Env):
    """
    Gym environment for autonomous driving in CARLA designed for DQ-GAT.
    
    Observations:
      A dict with two keys:
        - "bev_image": A (3, 224, 224) float32 image (CHW) representing the bird's-eye view,
                       obtained from a semantic segmentation camera.
        - "agent_feats": A (20, 10) float32 array of agent features in the ego-vehicle's coordinate frame.
    
    Actions:
      - Continuous: A 2-element array [steer, throttle] with steer in [-1, 1] and throttle in [0, 1].
      - Discrete: An integer index mapping to [steer, throttle] using the discrete_actions dict.
    
    Reward:
      - -50 if a collision occurs.
      - Otherwise, reward = current speed (km/h)/40, capped at 1.
    
    Additional features:
      - Customizable reward, state encoding, and decoding hooks.
      - Supports both training and evaluation modes and various rendering modes.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 host="127.0.0.1",
                 port=2000,
                 fps=15,
                 map='Town07',
                 action_space_type="discrete",  # "continuous" or "discrete"
                 reward_fn=None,
                 encode_state_fn=None,
                 decode_vae_fn=None,
                 action_smoothing=0.2,
                 eval=False,
                 activate_render=True):
        super(CarlaDrivingEnv, self).__init__()

        # Connect to CARLA and set synchronous mode.
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = World(self.client, map)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / fps
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.client.reload_world(False)

        self.fps = fps
        self.map = map
        self.action_space_type = action_space_type
        self.action_smoothing = action_smoothing
        self.episode_idx = -2
        self.max_distance = 3000
        self.eval = eval
        self.activate_render = activate_render

        # Custom hooks.
        self.reward_fn = reward_fn if callable(reward_fn) else self.default_reward_fn
        self.encode_state_fn = encode_state_fn if callable(encode_state_fn) else (lambda obs: obs)
        self.decode_vae_fn = decode_vae_fn if callable(decode_vae_fn) else None

        # Define action space.
        if self.action_space_type == "continuous":
            self.action_space = spaces.Box(low=np.array([-1.0, 0.0]),
                                           high=np.array([1.0, 1.0]),
                                           dtype=np.float32)
        elif self.action_space_type == "discrete":
            self.action_space = spaces.Discrete(len(discrete_actions))
        else:
            raise ValueError("Unsupported action_space_type. Use 'continuous' or 'discrete'.")

        # Define observation space.
        # BEV image: now directly (3, 224, 224) from the segmentation sensor.
        self.observation_space = spaces.Dict({
            "bev_image": spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.float32),
            "agent_feats": spaces.Box(low=-np.inf, high=np.inf, shape=(20, 12), dtype=np.float32)
        })
    
        # Initialize actors.
        self.vehicle = None
        self.collision_sensor = None
        self.bev_camera = None
        self.bev_image = None
        self.collision_history = []
        self.step_count = 0
        self.max_episode_steps = 1000
        self.episode_ended = False

        # Pygame display for rendering.
        if self.activate_render:
            pygame.init()
            pygame.font.init()
            width, height = 900, 900
            self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.clock = pygame.time.Clock()
            self.hud = HUD(width, height)
            self.hud.set_vehicle(self.vehicle)
            self.world.on_tick(self.hud.on_world_tick)

        # Spawn ego vehicle.

        self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[0],
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))

        # Attach semantic segmentation BEV camera.
        # We directly set the image size to 224x224.
        self.bev_camera = Camera(
            self.world, 224, 224,
            transform=sensor_transforms["bev"],
            attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image(e),
            camera_type="sensor.camera.semantic_segmentation",
            custom_palette= True
        )
        self.camera = Camera(self.world, width, height,
            transform=sensor_transforms["spectator"],
            attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e),
            camera_type="sensor.camera.rgb",
            custom_palette= False
        )
        self.reset()

    def default_reward_fn(self, env):
        # Default: -50 if collision; else speed (km/h)/40, capped at 1.
        speed = self.vehicle.get_speed() * 3.6
        if len(self.collision_history) > 0:
            return -50.0
        return min(speed / 40.0, 1.0)

    def _on_collision(self, event):
        self.collision_history.append(event)
        self.episode_ended = True

    def _on_bev_image(self, image):
        # For semantic segmentation sensor, the image is already 224x224.
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.bev_image = array[:, :, :3]  # drop alpha

    def _process_bev_image(self, image):
        """
        Since the sensor directly outputs a 224x224 image, we only need to convert
        from HWC to CHW and cast to float32.
        """
        chw = np.transpose(image, (2, 0, 1))
        chw = np.expand_dims(chw, axis=0)
        return chw.astype(np.float32)

    def get_agent_features(self):
        """
        Returns an array (shape: (1, 20, 12)) of agent features.
        Each row is a NpSequenceArray with:
        [x, y, d, psi, v_x, v_y, a_x, a_y, w, l, class_type, token_type]
        Coordinates and kinematics are expressed in the ego-vehicle's frame.
        All agents are sorted in ascending order of their vehicle ID with the ego vehicle always first.
        If a non-ego vehicle is farther than the threshold (50 m), its features are zeroed and token_type set to PAD_TOKEN.
        Pads with zeros if fewer than 20 agents are detected.
        """
        max_distance_threshold = 50.0  # meters

        ego_transform = self.vehicle.get_transform()
        ego_velocity = self.vehicle.get_velocity()
        vehicles = self.world.get_actors().filter("vehicle.*")
        
        ego_feature = None
        other_features = []
        for veh in vehicles:
            # Determine if this is the ego vehicle.
            if veh.id == self.vehicle.id:
                veh_transform = ego_transform
                veh_velocity = ego_velocity
                veh_accel = self.vehicle.get_acceleration()
                token = TokenType.EGO_TOKEN.value
            else:
                veh_transform = veh.get_transform()
                veh_velocity = veh.get_velocity()
                veh_accel = veh.get_acceleration()
                token = TokenType.AGENT_TOKEN.value

            # Compute relative position.
            rel_x, rel_y = transform_to_ego_frame(ego_transform, veh_transform.location)
            d = math.sqrt(rel_x**2 + rel_y**2)
            
            # For non-ego vehicles, if the distance is too large, zero out the features.
            # print(d)
            if veh.id != self.vehicle.id and d > max_distance_threshold:
                arr = np.zeros((NpSequenceArray.dim,), dtype=np.float32)
                arr[TOKEN_TYPE_DIM] = TokenType.PAD_TOKEN.value
            else:
                # Compute relative yaw.
                psi = veh_transform.rotation.yaw - ego_transform.rotation.yaw
                yaw_rad = math.radians(ego_transform.rotation.yaw)
                # Transform velocities to ego frame.
                vx_world = veh_velocity.x
                vy_world = veh_velocity.y
                rel_vx = vx_world * math.cos(-yaw_rad) - vy_world * math.sin(-yaw_rad)
                rel_vy = vx_world * math.sin(-yaw_rad) + vy_world * math.cos(-yaw_rad)
                # Transform acceleration similarly.
                ax_world = veh_accel.x
                ay_world = veh_accel.y
                rel_ax = ax_world * math.cos(-yaw_rad) - ay_world * math.sin(-yaw_rad)
                rel_ay = ax_world * math.sin(-yaw_rad) + ay_world * math.cos(-yaw_rad)
                # Get vehicle dimensions (CARLA gives half extents).
                bb = veh.bounding_box
                w = bb.extent.y * 2
                l = bb.extent.x * 2
                
                arr = np.zeros((NpSequenceArray.dim,), dtype=np.float32)
                arr[X_DIM] = rel_x
                arr[Y_DIM] = rel_y
                arr[DIST_DIM] = d
                arr[YAW_DIM] = psi
                arr[VX_DIM] = rel_vx
                arr[VY_DIM] = rel_vy
                arr[AX_DIM] = rel_ax
                arr[AY_DIM] = rel_ay
                arr[WIDTH_DIM] = w
                arr[LENGTH_DIM] = l
                arr[CLASS_TYPE_DIM] = ClassType.VEHICLE.value
                arr[TOKEN_TYPE_DIM] = token

            # Print for debugging.
            # print("Vehicle ID:", veh.id)
            # print("Relative x, y:", rel_x, rel_y)
            # print("Distance:", d)
            # if veh.id == self.vehicle.id or d <= max_distance_threshold:
            #     print("Relative yaw:", psi)
            #     print("Relative velocities (vx, vy):", rel_vx, rel_vy)
            #     print("Relative accelerations (ax, ay):", rel_ax, rel_ay)
            #     print("Dimensions (w, l):", w, l)
            #     print("Token type:", token)
            # else:
            #     print("Vehicle too far; features padded.")
            # print("Full feature vector:", arr)
            # print("-" * 50)
            
            feature = NpSequenceArray(arr)
            # print(feature)
            if veh.id == self.vehicle.id:
                ego_feature = feature
            else:
                other_features.append((veh.id, feature))
        
        # Sort non-ego vehicles by their ID in ascending order.
        other_features.sort(key=lambda x: x[0])
        other_features = [f for _, f in other_features]
        
        # Combine with ego feature (which is always first).
        feature_list = []
        if ego_feature is not None:
            feature_list.append(ego_feature)
        feature_list.extend(other_features)
        
        # Convert list to numpy array.
        if feature_list:
            features = np.array(feature_list, dtype=np.float32)
        else:
            features = np.zeros((0, NpSequenceArray.dim), dtype=np.float32)
        
        max_agents = 20
        num_agents = features.shape[0]
        if num_agents < max_agents:
            pad = np.zeros((max_agents - num_agents, NpSequenceArray.dim), dtype=np.float32)
            features = np.vstack([features, pad])
        elif num_agents > max_agents:
            distances = features[:, DIST_DIM]
            idx = np.argsort(distances)[:max_agents]
            features = features[idx]
        
        # Add a new axis so the final shape is (1, 20, 12).
        features = np.expand_dims(features, axis=0)
        # print("Final features shape:", features.shape)
        # print(features)
        return features


    def step(self, action):
        """
        Executes one simulation step.
        For continuous actions, action is [steer, throttle].
        For discrete actions, action is an index mapping to [steer, throttle] in discrete_actions.
        Action smoothing is applied.
        """

        if action is not None:
            # Create new route on route completion
            if self.current_waypoint_index >= len(self.route_waypoints) - 1:
                if not self.eval:
                    self.new_route()
                else:
                    self.success_state = True

            if self.action_space_type == "continuous":
                steer, throttle = [float(a) for a in action]
            elif self.action_space_type == "discrete":
                steer, throttle = discrete_actions[action]

            self.vehicle.control.steer = smooth_action(self.vehicle.control.steer, steer, self.action_smoothing)
            self.vehicle.control.throttle = smooth_action(self.vehicle.control.throttle, throttle,
                                                          self.action_smoothing)
        # Tick game
        self.world.tick()
        self.step_count += 1
        self.observation = self._get_observation()
        self.viewer_image = self._get_viewer_image()
        # timeout = time.time() + 1.0
        # while self.bev_image is None and time.time() < timeout:
        #     time.sleep(0.01)
        # raw_bev = self.bev_image if self.bev_image is not None else np.zeros((224, 224, 3), dtype=np.uint8)
        # bev_obs = self._process_bev_image(raw_bev)
        agent_feats = self.get_agent_features()
        observation = {
            "bev_image": torch.from_numpy(self.observation), 
            "agent_feats": torch.from_numpy(agent_feats)
        }


        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # Keep track of closest waypoint on the route
        self.prev_waypoint_index = self.current_waypoint_index
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0:  # Did we pass the waypoint?
                waypoint_index += 1  # Go to next waypoint
            else:
                break
        self.current_waypoint_index = waypoint_index

        # Check for route completion
        if self.current_waypoint_index < len(self.route_waypoints) - 1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[
                (self.current_waypoint_index + 1) % len(self.route_waypoints)]

        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[
            self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(
            self.route_waypoints)

        # Calculate deviation from center of the lane
        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        # Calculate distance traveled
        if action is not None:
            self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()
        # Terminal on max distance
        if self.distance_traveled >= self.max_distance and not self.eval:
            self.success_state = True

        self.distance_from_center_history.append(self.distance_from_center)

        # Call external reward fn
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward

        if self.activate_render:
            pygame.event.pump()
            if pygame.key.get_pressed()[K_ESCAPE]:
                self.close()
                self.terminal_state = True
            self.render()


        info = {
            "closed": self.closed,
            'total_reward': self.total_reward,
            'routes_completed': self.routes_completed,
            'total_distance': self.distance_traveled,
            'avg_center_dev': (self.center_lane_deviation / self.step_count),
            'avg_speed': (self.speed_accum / self.step_count),
            'mean_reward': (self.total_reward / self.step_count)
        }

        if self.activate_render:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    self.close()
                    done = True
            self.render(mode="human")
            self.clock.tick(self.fps)

        # encoded_obs = self.encode_state_fn(observation).detach().cpu().numpy()
        return observation, self.last_reward, self.terminal_state or self.success_state, info

    def reset(self):
        # Create new route
        self._clear_vehicles()
        self._spawn_vehicles() 
        self.num_routes_completed = -1
        self.episode_idx += 1
        self.new_route()

        # Two different variables to differ between success episode and fail episode
        self.terminal_state = False  # Set to True when we want to end episode
        self.success_state = False  # Set to True when we want to end episode.

        self.closed = False  # Set to True when ESC is pressed
        self.extra_info = []  # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None  # Last received observation
        self.viewer_image = self.viewer_image_buffer = None  # Last received image to show in the viewer
        self.lidar_data = self.lidar_data_buffer = None
        self.episode_ended = False 
        self.step_count = 0

        # Init metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0
        time.sleep(0.2)
        self.world.tick()
        time.sleep(0.2)
        self.observation = self._get_observation()
        agent_feats = self.get_agent_features()
        observation = {
            "bev_image": torch.from_numpy(self.observation), 
            "agent_feats": torch.from_numpy(agent_feats)
        }
        return observation

    def render(self, mode="human"):
        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation

        # Tick render clock
        self.clock.tick()
        self.hud.tick(self.world, self.clock)

        # Get maneuver name
        if self.current_road_maneuver == RoadOption.LANEFOLLOW:
            maneuver = "Follow Lane"
        elif self.current_road_maneuver == RoadOption.LEFT:
            maneuver = "Left"
        elif self.current_road_maneuver == RoadOption.RIGHT:
            maneuver = "Right"
        elif self.current_road_maneuver == RoadOption.STRAIGHT:
            maneuver = "Straight"
        elif self.current_road_maneuver == RoadOption.CHANGELANELEFT:
            maneuver = "Change Left Lane"
        elif self.current_road_maneuver == RoadOption.CHANGELANERIGHT:
            maneuver = "Change Right Lane"
        else:
            maneuver = "INVALID"

        # Add metrics to HUD
        self.extra_info.extend([
            "Episode {}".format(self.episode_idx),
            "Reward: % 19.2f" % self.last_reward,
            "",
            "Maneuver:        % 11s" % maneuver,
            "Routes completed:    % 7.2f" % self.routes_completed,
            "Distance traveled: % 7d m" % self.distance_traveled,
            "Center deviance:   % 7.2f m" % self.distance_from_center,
            "Avg center dev:    % 7.2f m" % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h" % (self.speed_accum / self.step_count),
            "Total reward:        % 7.2f" % self.total_reward,
        ])

        # Remove the batch dimension (from (1, 3, 224, 224) to (3, 224, 224))
        self.viewer_image = self._draw_path(self.camera, self.viewer_image)
        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))
        img = np.squeeze(self.observation, axis=0)
        # Transpose to (224, 224, 3)
        img = np.transpose(img, (1, 2, 0))

        # Define pos_observation: place image at top-right with a 10-pixel margin.
        display_width, display_height = self.display.get_size()
        img_height, img_width = img.shape[:2]
        pos_observation = (display_width - img_width - 10, 10)

        # Now create a surface and blit the image at pos_observation.
        self.display.blit(pygame.surfarray.make_surface(img), pos_observation)
        # pos_vae_decoded = (self.display.get_size()[0] - 2 * obs_w - 10, 10)
        # if self.decode_vae_fn:
        #     self.display.blit(pygame.surfarray.make_surface(self.observation_decoded.swapaxes(0, 1)), pos_vae_decoded)

        # if self.activate_lidar:
        #     lidar_h, lidar_w = self.lidar_data.shape[:2]
        #     pos_lidar = (self.display.get_size()[0] - obs_w - 10, 100)
        #     self.display.blit(pygame.surfarray.make_surface(self.lidar_data.swapaxes(0, 1)), pos_lidar)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = []  # Reset extra info list

        # Render to screen
        pygame.display.flip()

    def close(self):
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.bev_camera is not None:
            self.bev_camera.destroy()
        if self.camera is not None:
            self.camera.destroy()
        pygame.quit()
        self.world.apply_settings(carla.WorldSettings())
    
    def _get_observation(self):
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs
    
    def _set_observation_image(self, image):
        self.observation_buffer = self._process_bev_image(image)

    def _clear_vehicles(self):
        actors = self.world.get_actors().filter("vehicle.*")
        vehicle_ids = [actor.id for actor in actors if actor.id != self.vehicle.actor.id]
        self.client.apply_batch([carla.command.DestroyActor(vehicle_id) for vehicle_id in vehicle_ids])

    def _spawn_vehicles(self, num_vehicles=20, random_flag=True):
        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        blueprints = [bp for bp in blueprints if int(bp.get_attribute('number_of_wheels')) == 4]

        spawn_points = self.world.get_map().get_spawn_points()
        num_vehicles = min(num_vehicles, len(spawn_points))
        batch = []
        
        # Shuffle spawn points to ensure uniqueness.
        available_spawn_points = list(spawn_points)
        random.shuffle(available_spawn_points)
        
        if random_flag:
            for i in range(num_vehicles):
                blueprint = np.random.choice(blueprints)
                blueprint.set_attribute('role_name', 'autopilot')
                # Use unique spawn points by popping from the shuffled list.
                spawn_point = available_spawn_points.pop(0)
                batch.append(carla.command.SpawnActor(blueprint, spawn_point)
                            .then(carla.command.SetAutopilot(carla.command.FutureActor, True)))
        else:
            for i in range(num_vehicles):
                blueprint = blueprints[4]
                blueprint.set_attribute('role_name', 'autopilot')
                batch.append(carla.command.SpawnActor(blueprint, available_spawn_points[i])
                            .then(carla.command.SetAutopilot(carla.command.FutureActor, True)))
        
        self.client.apply_batch_sync(batch, True)
    
    def new_route(self):
        # Do a soft reset (teleport vehicle)
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        self.vehicle.set_simulate_physics(False)  # Reset the car's physics

        # Generate waypoints along the lap
        if not self.eval:
            # if self.episode_idx % 2 == 0 and self.num_routes_completed == -1:
            spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(intersection_routes)]
            # else:
            #     spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)
        else:
            spawn_points_list = [self.world.map.get_spawn_points()[index] for index in next(eval_routes)]
        route_length = 1
        while route_length <= 1:
            self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in
                                          spawn_points_list]
            self.route_waypoints = compute_route_waypoints(self.world.map, self.start_wp, self.end_wp, resolution=1.0)
            route_length = len(self.route_waypoints)
            if route_length <= 1:
                spawn_points_list = np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)

        self.distance_from_center_history = deque(maxlen=30)

        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self.vehicle.set_transform(self.start_wp.transform)
        time.sleep(0.2)
        self.vehicle.set_simulate_physics(True)
    
    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        if self.activate_render:
            self.hud.notification("Crossed line %s" % " and ".join(text))

    def _draw_path(self, camera, image):
        """
            Draw a connected path from start of route to end using homography.
        """
        vehicle_vector = vector(self.vehicle.get_transform().location)
        # Get the world to camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

        # Get the attributes from the camera
        image_w = int(camera.actor.attributes['image_size_x'])
        image_h = int(camera.actor.attributes['image_size_y'])
        fov = float(camera.actor.attributes['fov'])
        for i in range(self.current_waypoint_index, len(self.route_waypoints)):
            waypoint_location = self.route_waypoints[i][0].transform.location + carla.Location(z=1.25)
            waypoint_vector = vector(waypoint_location)
            if not (2 < abs(np.linalg.norm(vehicle_vector - waypoint_vector)) < 50):
                continue
            # Calculate the camera projection matrix to project from 3D -> 2D
            K = build_projection_matrix(image_w, image_h, fov)
            x, y = get_image_point(waypoint_location, K, world_2_camera)
            if i == len(self.route_waypoints) - 1:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            image = cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)
        return image
    
    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

